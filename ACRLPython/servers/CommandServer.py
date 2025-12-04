#!/usr/bin/env python3
"""
CommandServer.py - Bidirectional command and results server

Consolidates ResultsServer and StatusServer into a single bidirectional server.
- Sends commands to Unity (move, gripper, status queries)
- Receives completion callbacks from Unity
- Broadcasts results to all connected clients

Port: 5010
"""

import socket
import struct
import json
import threading
import time
import logging
from typing import Dict, Any, Optional, List
from queue import Queue, Empty

# Import config
try:
    import LLMConfig as cfg
except ImportError:
    from .. import LLMConfig as cfg

# Import base classes
try:
    from core.TCPServerBase import TCPServerBase, ServerConfig, ConnectionState
    from core.UnityProtocol import UnityProtocol
except ImportError:
    from ..core.TCPServerBase import TCPServerBase, ServerConfig, ConnectionState
    from ..core.UnityProtocol import UnityProtocol

logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=cfg.LOG_FORMAT)
logger = logging.getLogger(__name__)


class CommandBroadcaster:
    """
    Singleton for sending commands and receiving completions.

    Replaces both ResultsBroadcaster and StatusResponseHandler with a unified interface.
    """

    _instance = None
    _lock = threading.RLock()

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
        self._max_queue_size = cfg.MAX_RESULT_QUEUE_SIZE

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
        Send a result to Unity (backward compatibility with ResultsBroadcaster).

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
                logger.warning(f"No queue for request {request_id}")

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


# Backward compatibility aliases
ResultsBroadcaster = CommandBroadcaster
StatusResponseHandler = CommandBroadcaster


class CommandServer(TCPServerBase):
    """
    Bidirectional TCP server for commands and completions.

    Sends commands to Unity and receives completion callbacks.
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        if config is None:
            config = ServerConfig(host=cfg.DEFAULT_HOST, port=cfg.LLM_RESULTS_PORT)
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
            logger.warning(f"Unexpected message type: {msg_type} (expected RESULT or STATUS_RESPONSE)")
            return None

        # Read JSON length
        len_bytes = self._recv_exact(client, 4)
        if not len_bytes:
            return None
        json_len = struct.unpack("<I", len_bytes)[0]  # Little-endian to match Unity

        if json_len > cfg.MAX_STRING_LENGTH * 10:
            logger.error(f"Completion too large: {json_len}")
            return None

        # Read JSON data
        json_bytes = self._recv_exact(client, json_len)
        if not json_bytes:
            return None

        try:
            completion = json.loads(json_bytes.decode("utf-8"))
            completion["request_id"] = request_id
            logger.debug(f"Received completion for request {request_id}: {completion.get('type', 'unknown')}")
            return completion
        except json.JSONDecodeError as e:
            logger.error(f"Invalid completion JSON: {e}")
            return None

    def _recv_exact(self, client: socket.socket, num_bytes: int) -> Optional[bytes]:
        """Receive exactly num_bytes."""
        data = b""
        while len(data) < num_bytes:
            chunk = client.recv(num_bytes - len(data))
            if not chunk:
                return None
            data += chunk
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
    port: int = cfg.LLM_RESULTS_PORT, host: str = cfg.DEFAULT_HOST
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
    parser.add_argument("--host", default=cfg.DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=cfg.LLM_RESULTS_PORT)
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
