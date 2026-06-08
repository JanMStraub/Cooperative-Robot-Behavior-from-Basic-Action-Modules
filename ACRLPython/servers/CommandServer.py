#!/usr/bin/env python3

"""Bidirectional command and results server (port 5007). Sends commands to Unity, receives completion callbacks."""

import itertools
import socket
import struct
import json
import threading
import time
from typing import Dict, Any, Optional, List
from queue import Queue, Empty

try:
    from config.Servers import (
        DEFAULT_HOST,
        COMMAND_SERVER_PORT,
        MAX_RESULT_QUEUE_SIZE,
        MAX_STRING_LENGTH,
    )
    from core.LoggingSetup import get_logger
except ImportError:
    from ..config.Servers import (
        DEFAULT_HOST,
        COMMAND_SERVER_PORT,
        MAX_RESULT_QUEUE_SIZE,
        MAX_STRING_LENGTH,
    )
    from ..core.LoggingSetup import get_logger

try:
    from core.TCPServerBase import TCPServerBase, ServerConfig
    from core.UnityProtocol import UnityProtocol
except ImportError:
    from ..core.TCPServerBase import TCPServerBase, ServerConfig
    from ..core.UnityProtocol import UnityProtocol

logger = get_logger(__name__)


class CommandBroadcaster:
    """Singleton for sending commands to Unity and routing completion callbacks."""

    _instance = None
    _lock = threading.RLock()
    # Atomic counter for request IDs. Starts at 1 — 0 is the protocol sentinel
    # meaning "no ID". itertools.count is thread-safe for next() calls in CPython
    # because GIL protects the integer increment, but we wrap it in the existing
    # _queue_lock anyway for correctness on all runtimes.
    _id_counter = itertools.count(1)

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self):
        self._server: Optional["CommandServer"] = None
        self._result_queue: List[Dict] = []
        self._completion_queues: Dict[int, Queue] = {}
        self._queue_lock = threading.Lock()
        self._max_queue_size = MAX_RESULT_QUEUE_SIZE

        # Tracks request IDs that Python originated; kept for _LATE_ARRIVAL_TTL seconds
        # after queue removal so put_completion can distinguish late arrivals from
        # Unity-initiated messages that never had a queue.
        self._recently_removed: Dict[int, float] = {}
        self._LATE_ARRIVAL_TTL = 30.0  # seconds

        # Thread-safe command tracking
        self._active_commands: Dict[int, Dict[str, Any]] = {}
        self._active_commands_lock = threading.RLock()

        # Robot-specific client tracking for targeted commands
        self._robot_clients: Dict[str, Any] = {}  # robot_id -> client socket

    def set_server(self, server: "CommandServer"):
        self._server = server

    def send_command(self, command: Dict[str, Any], request_id: int = 0) -> bool:
        """Send command to all connected Unity clients. Returns True if sent."""
        if self._server is None:
            logger.warning("CommandBroadcaster not initialized")
            return False

        try:
            command["request_id"] = request_id
            message = UnityProtocol.encode_result_message(command, request_id)
            sent_count = self._server.broadcast_to_all_clients(message)

            if sent_count == 0:
                # Queue for later delivery
                with self._queue_lock:
                    if len(self._result_queue) < self._max_queue_size:
                        self._result_queue.append(command)
                        return True
                    logger.warning("Command queue full")
                    return False

            logger.debug(f"Command sent to {sent_count} client(s)")
            return True

        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return False

    def send_result(self, result: Dict[str, Any]) -> bool:
        request_id = result.get("request_id", 0)
        return self.send_command(result, request_id)

    def create_completion_queue(self, request_id: int):
        with self._queue_lock:
            self._completion_queues[request_id] = Queue()
            self._recently_removed.pop(request_id, None)

    def remove_completion_queue(self, request_id: int):
        """Remove a completion queue, recording it as a known Python-originated request."""
        with self._queue_lock:
            if self._completion_queues.pop(request_id, None) is not None:
                self._recently_removed[request_id] = time.time()
            # Prune stale entries to bound memory usage.
            cutoff = time.time() - self._LATE_ARRIVAL_TTL
            stale = [rid for rid, ts in self._recently_removed.items() if ts < cutoff]
            for rid in stale:
                del self._recently_removed[rid]

    def put_completion(self, request_id: int, completion: Dict[str, Any]):
        """Put a completion result into the appropriate queue."""
        with self._queue_lock:
            if request_id in self._completion_queues:
                self._completion_queues[request_id].put(completion)
                logger.debug(f"Completion queued for request {request_id}")
            elif request_id in self._recently_removed:
                logger.warning(
                    f"Late completion for request {request_id} (queue already removed — operation timed out or returned early)"
                )
            else:
                logger.debug(
                    f"Ignoring Unity-initiated message for request {request_id} (no queue expected)"
                )

    def get_completion(
        self, request_id: int, timeout: float = 5.0
    ) -> Optional[Dict[str, Any]]:
        with self._queue_lock:
            queue = self._completion_queues.get(request_id)

        if queue is None:
            return None

        try:
            return queue.get(timeout=timeout)
        except Empty:
            return None

    def abort_all_pending(self):
        """Unblock all threads waiting in send_command_and_wait by injecting an abort sentinel."""
        with self._queue_lock:
            for queue in self._completion_queues.values():
                queue.put({"success": False, "aborted": True})
        logger.debug(
            f"Aborted {len(self._completion_queues)} pending completion queue(s)"
        )

    def get_queued_results(self) -> List[Dict]:
        """Get and clear all queued results."""
        with self._queue_lock:
            results = self._result_queue.copy()
            self._result_queue.clear()
            return results

    def track_command(
        self, request_id: int, command: Dict[str, Any], robot_id: Optional[str] = None
    ):
        with self._active_commands_lock:
            self._active_commands[request_id] = {
                "command": command,
                "robot_id": robot_id,
                "timestamp": time.time(),
                "status": "active",
            }
            logger.debug(f"Tracking command {request_id} for robot {robot_id}")

    def complete_command(
        self, request_id: int, success: bool, result: Optional[Dict[str, Any]] = None
    ):
        with self._active_commands_lock:
            if request_id in self._active_commands:
                self._active_commands[request_id]["status"] = (
                    "completed" if success else "failed"
                )
                self._active_commands[request_id]["result"] = result
                self._active_commands[request_id]["completion_time"] = time.time()
                logger.debug(f"Completed command {request_id}: {success}")

    def get_active_commands(self) -> Dict[int, Dict[str, Any]]:
        with self._active_commands_lock:
            return self._active_commands.copy()

    def register_robot_client(self, robot_id: str, client):
        with self._active_commands_lock:
            self._robot_clients[robot_id] = client
            logger.info(f"Registered client for robot {robot_id}")

    def unregister_robot_client(self, robot_id: str):
        with self._active_commands_lock:
            self._robot_clients.pop(robot_id, None)
            logger.info(f"Unregistered client for robot {robot_id}")

    def send_command_and_wait(
        self, command: Dict[str, Any], timeout: float = 5.0
    ) -> Optional[Dict[str, Any]]:
        # Generate request ID — use caller-supplied ID if non-zero, otherwise
        # allocate the next value from the atomic counter.
        request_id = command.get("request_id", 0)
        if request_id == 0:
            request_id = next(self.__class__._id_counter)

        # Create completion queue
        self.create_completion_queue(request_id)

        try:
            if not self.send_command(command, request_id):
                return None

            return self.get_completion(request_id, timeout)

        finally:
            self.remove_completion_queue(request_id)

    def send_command_to_robot(
        self, robot_id: str, command: Dict[str, Any], request_id: int = 0
    ) -> bool:
        with self._active_commands_lock:
            client = self._robot_clients.get(robot_id)

        if client is None:
            logger.warning(f"No client registered for robot {robot_id}, broadcasting")
            return self.send_command(command, request_id)

        if self._server is None:
            logger.warning("CommandBroadcaster not initialized")
            return False

        try:
            command["request_id"] = request_id
            message = UnityProtocol.encode_result_message(command, request_id)
            sent = self._server.send_to_client(client, message)

            if sent:
                logger.debug(f"Command sent to robot {robot_id}")
            else:
                logger.warning(f"Failed to send command to robot {robot_id}")

            return sent

        except Exception as e:
            logger.error(f"Error sending command to robot {robot_id}: {e}")
            return False


class CommandServer(TCPServerBase):
    """Bidirectional TCP server — sends commands to Unity, receives completion callbacks."""

    def __init__(self, config: Optional[ServerConfig] = None):
        if config is None:
            config = ServerConfig(host=DEFAULT_HOST, port=COMMAND_SERVER_PORT)
        super().__init__(config)

        self._broadcaster = CommandBroadcaster()
        self._broadcaster.set_server(self)

    _client_timeout: float = 1.0

    def _pre_connection_setup(self, client: socket.socket, _address: tuple) -> None:
        self._send_queued_results(client)

    def _handle_message(self, client: socket.socket, _address: tuple) -> None:
        completion = self._receive_completion(client)
        if completion:
            request_id = completion.get("request_id", 0)
            self._broadcaster.put_completion(request_id, completion)

    def _receive_completion(self, client: socket.socket) -> Optional[Dict[str, Any]]:
        """Read one completion from Unity. Accepts RESULT and STATUS_RESPONSE types."""
        from core.UnityProtocol import MessageType

        header = self._recv_exact(client, 5)
        if not header:
            return None

        msg_type = header[0]
        request_id = struct.unpack("<I", header[1:5])[0]  # little-endian, matches Unity

        valid_types = [MessageType.RESULT, MessageType.STATUS_RESPONSE]
        if msg_type not in valid_types:
            logger.warning(
                f"Unexpected message type: {msg_type} (expected RESULT or STATUS_RESPONSE)"
            )
            return None

        len_bytes = self._recv_exact(client, 4)
        if not len_bytes:
            return None
        json_len = struct.unpack("<I", len_bytes)[0]

        if json_len > MAX_STRING_LENGTH * 10:
            logger.error(f"Completion too large: {json_len}")
            return None

        json_bytes = self._recv_exact(client, json_len)
        if not json_bytes:
            return None

        try:
            completion = json.loads(json_bytes.decode("utf-8"))
            completion["request_id"] = request_id

            if completion.get("type") == "world_state_update":
                self._handle_world_state_update(completion)
                # Don't return world state updates as completions
                logger.debug(f"Processed world state update")
                return None

            logger.debug(
                f"Received completion for request {request_id}: {completion.get('type', 'unknown')}"
            )
            return completion
        except json.JSONDecodeError as e:
            logger.error(f"Invalid completion JSON: {e}")
            return None

    def _handle_world_state_update(self, update: Dict[str, Any]):
        try:
            from operations.WorldState import get_world_state

            world_state = get_world_state()

            # Update robot states
            robots = update.get("robots", [])
            for robot_data in robots:
                robot_id = robot_data.get("robot_id")
                if not robot_id:
                    continue

                # Convert Unity format to WorldState format
                position = robot_data.get("position")
                if position:
                    position = (position.get("x"), position.get("y"), position.get("z"))

                rotation = robot_data.get("rotation")
                if rotation:
                    # WorldState expects (roll, pitch, yaw) in degrees
                    # Unity sends quaternion (x, y, z, w)
                    # For now, we'll store the quaternion as-is
                    rotation = (
                        rotation.get("x"),
                        rotation.get("y"),
                        rotation.get("z"),
                        rotation.get("w"),
                    )

                target_position = robot_data.get("target_position")
                if target_position:
                    target_position = (
                        target_position.get("x"),
                        target_position.get("y"),
                        target_position.get("z"),
                    )

                state_data = {
                    "position": position,
                    "rotation": rotation,
                    "target_position": target_position,
                    "gripper_state": robot_data.get("gripper_state", "unknown"),
                    "is_moving": robot_data.get("is_moving", False),
                    "is_initialized": robot_data.get("is_initialized", False),
                    "joint_angles": robot_data.get("joint_angles", []),
                }

                world_state.update_robot_state(robot_id, state_data)

            # Update object positions
            objects = update.get("objects", [])
            for obj_data in objects:
                object_id = obj_data.get("object_id")
                if not object_id:
                    continue

                position = obj_data.get("position")
                if position:
                    position = (position.get("x"), position.get("y"), position.get("z"))

                    world_state.update_object_position(
                        object_id=object_id,
                        position=position,
                        color=obj_data.get("color", "unknown"),
                        object_type=obj_data.get("object_type", "unknown"),
                        confidence=obj_data.get("confidence", 1.0),
                    )

            logger.debug(
                f"Updated world state: {len(robots)} robots, {len(objects)} objects"
            )

        except Exception as e:
            logger.error(f"Error handling world state update: {e}", exc_info=True)

    def _send_queued_results(self, client: socket.socket):
        """Send any queued results to a newly connected client."""
        queued = self._broadcaster.get_queued_results()

        if queued:
            logger.info(f"Sending {len(queued)} queued results")
            for result in queued:
                try:
                    request_id = result.get("request_id", 0)
                    message = UnityProtocol.encode_result_message(result, request_id)
                    client.sendall(message)
                except Exception as e:
                    logger.warning(f"Failed to send queued result: {e}")
                    break


def get_command_broadcaster() -> CommandBroadcaster:
    """Get the global CommandBroadcaster singleton."""
    return CommandBroadcaster()


def run_command_server_background(
    port: int = COMMAND_SERVER_PORT, host: str = DEFAULT_HOST
) -> CommandServer:
    config = ServerConfig(host=host, port=port)
    server = CommandServer(config)
    server.start()
    return server


if __name__ == "__main__":
    import argparse
    import signal

    parser = argparse.ArgumentParser(description="Command Server")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=COMMAND_SERVER_PORT)
    args = parser.parse_args()

    config = ServerConfig(host=args.host, port=args.port)
    server = CommandServer(config)

    def signal_handler(_sig, _frame):
        logger.info("Shutting down...")
        server.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info(f"Starting CommandServer on {args.host}:{args.port}")
    server.start()

    try:
        while server.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
