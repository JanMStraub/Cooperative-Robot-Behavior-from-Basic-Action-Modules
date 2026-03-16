#!/usr/bin/env python3
"""
CommandServer.py - Bidirectional command and results server

Consolidates ResultsServer and StatusServer into a single bidirectional server.
- Sends commands to Unity (move, gripper, status queries)
- Receives completion callbacks from Unity
- Broadcasts results to all connected clients

Port: 5010
"""

import itertools
import socket
import struct
import json
import threading
import time
from typing import Dict, Any, Optional, List
from queue import Queue, Empty

# Import config
try:
    from config.Servers import (
        DEFAULT_HOST,
        LLM_RESULTS_PORT,
        MAX_RESULT_QUEUE_SIZE,
        MAX_STRING_LENGTH,
    )
    from core.LoggingSetup import get_logger
except ImportError:
    from ..config.Servers import (
        DEFAULT_HOST,
        LLM_RESULTS_PORT,
        MAX_RESULT_QUEUE_SIZE,
        MAX_STRING_LENGTH,
    )
    from ..core.LoggingSetup import get_logger

# Import base classes
try:
    from core.TCPServerBase import TCPServerBase, ServerConfig
    from core.UnityProtocol import UnityProtocol
except ImportError:
    from ..core.TCPServerBase import TCPServerBase, ServerConfig
    from ..core.UnityProtocol import UnityProtocol

logger = get_logger(__name__)


class CommandBroadcaster:
    """
    Singleton for sending commands and receiving completions.

    Replaces both ResultsBroadcaster and StatusResponseHandler with a unified interface.
    """

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
        """Initialize the broadcaster."""
        self._server: Optional["CommandServer"] = None
        self._result_queue: List[Dict] = []
        self._completion_queues: Dict[int, Queue] = {}
        self._queue_lock = threading.Lock()
        self._max_queue_size = MAX_RESULT_QUEUE_SIZE

        # Thread-safe command tracking
        self._active_commands: Dict[int, Dict[str, Any]] = {}
        self._active_commands_lock = threading.RLock()

        # Robot-specific client tracking for targeted commands
        self._robot_clients: Dict[str, Any] = {}  # robot_id -> client socket

    def set_server(self, server: "CommandServer"):
        """Set the server instance for broadcasting."""
        self._server = server

    def send_command(self, command: Dict[str, Any], request_id: int = 0) -> bool:
        """
        Send a command to all connected Unity clients.

        Args:
            command: Command dictionary
            request_id: Request ID for correlation

        Returns:
            True if sent successfully
        """
        if self._server is None:
            logger.warning("CommandBroadcaster not initialized")
            return False

        try:
            # Add request_id to command
            command["request_id"] = request_id

            # Encode message
            message = UnityProtocol.encode_result_message(command, request_id)

            # Broadcast to all clients
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
        """
        Send a result to Unity.

        Args:
            result: Result dictionary

        Returns:
            True if sent successfully
        """
        request_id = result.get("request_id", 0)
        return self.send_command(result, request_id)

    def create_completion_queue(self, request_id: int):
        """Create a queue to receive completion for a request."""
        with self._queue_lock:
            self._completion_queues[request_id] = Queue()

    def remove_completion_queue(self, request_id: int):
        """Remove a completion queue."""
        with self._queue_lock:
            self._completion_queues.pop(request_id, None)

    def put_completion(self, request_id: int, completion: Dict[str, Any]):
        """Put a completion result into the appropriate queue."""
        with self._queue_lock:
            if request_id in self._completion_queues:
                self._completion_queues[request_id].put(completion)
                logger.debug(f"Completion queued for request {request_id}")
            else:
                logger.warning(
                    f"No queue for request {request_id} (late arrival or Unity-initiated)"
                )

    def get_completion(
        self, request_id: int, timeout: float = 5.0
    ) -> Optional[Dict[str, Any]]:
        """
        Wait for and get a completion result.

        Args:
            request_id: Request ID to wait for
            timeout: Timeout in seconds

        Returns:
            Completion dictionary or None if timed out
        """
        with self._queue_lock:
            queue = self._completion_queues.get(request_id)

        if queue is None:
            return None

        try:
            return queue.get(timeout=timeout)
        except Empty:
            return None

    def get_queued_results(self) -> List[Dict]:
        """Get and clear all queued results."""
        with self._queue_lock:
            results = self._result_queue.copy()
            self._result_queue.clear()
            return results

    def track_command(
        self, request_id: int, command: Dict[str, Any], robot_id: Optional[str] = None
    ):
        """
        Track an active command with thread safety.

        Args:
            request_id: Request ID for this command
            command: Command dictionary
            robot_id: Optional robot ID this command targets
        """
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
        """
        Mark a command as completed with thread safety.

        Args:
            request_id: Request ID of the completed command
            success: Whether the command succeeded
            result: Optional result data
        """
        with self._active_commands_lock:
            if request_id in self._active_commands:
                self._active_commands[request_id]["status"] = (
                    "completed" if success else "failed"
                )
                self._active_commands[request_id]["result"] = result
                self._active_commands[request_id]["completion_time"] = time.time()
                logger.debug(f"Completed command {request_id}: {success}")

    def get_active_commands(self) -> Dict[int, Dict[str, Any]]:
        """
        Get all active commands (thread-safe).

        Returns:
            Dictionary of active commands by request ID
        """
        with self._active_commands_lock:
            return self._active_commands.copy()

    def register_robot_client(self, robot_id: str, client):
        """
        Register a robot's client socket for targeted commands.

        Args:
            robot_id: Robot identifier
            client: Client socket
        """
        with self._active_commands_lock:
            self._robot_clients[robot_id] = client
            logger.info(f"Registered client for robot {robot_id}")

    def unregister_robot_client(self, robot_id: str):
        """
        Unregister a robot's client socket.

        Args:
            robot_id: Robot identifier
        """
        with self._active_commands_lock:
            self._robot_clients.pop(robot_id, None)
            logger.info(f"Unregistered client for robot {robot_id}")

    def send_command_and_wait(
        self, command: Dict[str, Any], timeout: float = 5.0
    ) -> Optional[Dict[str, Any]]:
        """
        Send a command and wait for its completion.

        Args:
            command: Command dictionary
            timeout: Timeout in seconds

        Returns:
            Completion result or None if timed out
        """
        # Generate request ID — use caller-supplied ID if non-zero, otherwise
        # allocate the next value from the atomic counter.
        request_id = command.get("request_id", 0)
        if request_id == 0:
            request_id = next(self.__class__._id_counter)

        # Create completion queue
        self.create_completion_queue(request_id)

        try:
            # Send command
            if not self.send_command(command, request_id):
                return None

            # Wait for completion
            return self.get_completion(request_id, timeout)

        finally:
            # Clean up queue
            self.remove_completion_queue(request_id)

    def send_command_to_robot(
        self, robot_id: str, command: Dict[str, Any], request_id: int = 0
    ) -> bool:
        """
        Send a command to a specific robot (targeted delivery).

        Args:
            robot_id: Target robot identifier
            command: Command dictionary
            request_id: Request ID for correlation

        Returns:
            True if sent successfully
        """
        with self._active_commands_lock:
            client = self._robot_clients.get(robot_id)

        if client is None:
            logger.warning(f"No client registered for robot {robot_id}, broadcasting")
            return self.send_command(command, request_id)

        if self._server is None:
            logger.warning("CommandBroadcaster not initialized")
            return False

        try:
            # Add request_id to command
            command["request_id"] = request_id

            # Encode message
            message = UnityProtocol.encode_result_message(command, request_id)

            # Send to specific client
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
    """
    Bidirectional TCP server for commands and completions.

    Sends commands to Unity and receives completion callbacks.
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        if config is None:
            config = ServerConfig(host=DEFAULT_HOST, port=LLM_RESULTS_PORT)
        super().__init__(config)

        # Initialize broadcaster
        self._broadcaster = CommandBroadcaster()
        self._broadcaster.set_server(self)

    def handle_client_connection(self, client: socket.socket, address: tuple):
        """
        Handle a Unity client connection.

        Receives completion messages and keeps connection alive for sending commands.
        """
        logger.info(f"Command client connected from {address}")

        # Send any queued results
        self._send_queued_results(client)

        try:
            # Set timeout for receiving completions
            client.settimeout(1.0)

            while self.is_running():
                try:
                    # Try to receive completion message
                    completion = self._receive_completion(client)

                    if completion:
                        request_id = completion.get("request_id", 0)
                        self._broadcaster.put_completion(request_id, completion)

                except socket.timeout:
                    # Expected - allows checking is_running()
                    continue
                except Exception as e:
                    is_fatal, desc = self._is_connection_error_fatal(e)
                    if is_fatal:
                        logger.debug(f"Client {address} disconnected: {desc}")
                        break

        except Exception as e:
            logger.error(f"Error handling command client: {e}")

        logger.info(f"Command client disconnected from {address}")

    def _receive_completion(self, client: socket.socket) -> Optional[Dict[str, Any]]:
        """
        Receive a completion message from Unity.

        Protocol: [type:1][request_id:4][json_len:4][json_data:N]

        Accepts both RESULT (0x02) and STATUS_RESPONSE (0x06) message types.

        Returns:
            Completion dictionary or None
        """
        from core.UnityProtocol import MessageType

        # Read header
        header = self._recv_exact(client, 5)
        if not header:
            return None

        msg_type = header[0]
        request_id = struct.unpack("<I", header[1:5])[0]  # Little-endian to match Unity

        # Validate message type - accept both RESULT and STATUS_RESPONSE
        valid_types = [MessageType.RESULT, MessageType.STATUS_RESPONSE]
        if msg_type not in valid_types:
            logger.warning(
                f"Unexpected message type: {msg_type} (expected RESULT or STATUS_RESPONSE)"
            )
            return None

        # Read JSON length
        len_bytes = self._recv_exact(client, 4)
        if not len_bytes:
            return None
        # Little-endian to match Unity
        json_len = struct.unpack("<I", len_bytes)[0]

        if json_len > MAX_STRING_LENGTH * 10:
            logger.error(f"Completion too large: {json_len}")
            return None

        # Read JSON data
        json_bytes = self._recv_exact(client, json_len)
        if not json_bytes:
            return None

        try:
            completion = json.loads(json_bytes.decode("utf-8"))
            completion["request_id"] = request_id

            # Handle world state updates
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
        """
        Process a world state update from Unity and update the WorldState singleton.

        Args:
            update: World state update dictionary with 'robots' and 'objects' lists
        """
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

    def _recv_exact(self, client: socket.socket, num_bytes: int) -> Optional[bytes]:
        """
        Receive exactly num_bytes, preserving partial reads across socket timeouts.

        The client socket has a 1-second timeout so the outer loop can check
        is_running(). Without this guard, a timeout mid-read discards accumulated bytes and causes TCP stream desynchronization (next bytes are mis-parsed as a new message header).
        """
        data = b""
        while len(data) < num_bytes:
            try:
                chunk = client.recv(num_bytes - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                # Timeout during partial read — check for shutdown, then retry.
                # Do NOT discard data: partial bytes must be preserved.
                if not self.is_running():
                    return None
                continue
        return data

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
    port: int = LLM_RESULTS_PORT, host: str = DEFAULT_HOST
) -> CommandServer:
    """
    Start the CommandServer in the background.

    Args:
        port: Server port
        host: Server host

    Returns:
        CommandServer instance
    """
    config = ServerConfig(host=host, port=port)
    server = CommandServer(config)
    server.start()
    return server


if __name__ == "__main__":
    import argparse
    import signal

    parser = argparse.ArgumentParser(description="Command Server")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=LLM_RESULTS_PORT)
    args = parser.parse_args()

    config = ServerConfig(host=args.host, port=args.port)
    server = CommandServer(config)

    def signal_handler(sig, frame):
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
