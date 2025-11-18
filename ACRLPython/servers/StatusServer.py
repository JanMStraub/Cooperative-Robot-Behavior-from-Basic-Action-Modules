#!/usr/bin/env python3
"""
StatusServer.py - Bidirectional robot status query server

Handles status queries from Unity and returns robot state information.
Follows the RAGServer pattern for bidirectional communication.
"""

import socket
import logging
import threading
import queue
import time
import json
import struct
from typing import Dict, Optional

# Import dependencies
# Import config - try both import styles
try:
    import LLMConfig as cfg
except ImportError:
    from .. import LLMConfig as cfg

# Import base classes - try both import styles
try:
    from core.TCPServerBase import TCPServerBase, ServerConfig, ConnectionState
    from core.UnityProtocol import UnityProtocol
except ImportError:
    from ..core.TCPServerBase import TCPServerBase, ServerConfig, ConnectionState
    from ..core.UnityProtocol import UnityProtocol

# Configure logging
logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=cfg.LOG_FORMAT)


class StatusResponseHandler:
    """
    Singleton for handling status responses from Unity.

    Manages a queue of pending status requests and matches incoming
    responses from Unity to the waiting clients using request IDs.

    Protocol V2: Uses request_id instead of robot_id for queue keys to avoid race conditions.

    Thread Safety Improvements (Phase 2.2):
    - Uses RLock (reentrant lock) to allow nested locking within the same thread
    - Provides safe queue access methods that check existence before operations
    - Protects all dictionary operations with lock
    - Validates queue state before put/get operations
    """

    _instance: Optional["StatusResponseHandler"] = None
    _response_queues: Dict[int, queue.Queue] = {}  # Changed from str to int (request_id)
    _lock = threading.RLock()  # Use RLock for reentrant locking

    @classmethod
    def get_instance(cls) -> "StatusResponseHandler":
        """Get singleton instance with thread-safe initialization"""
        if cls._instance is None:
            with cls._lock:
                # Double-check pattern for thread-safe singleton
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def create_request_queue(cls, request_id: int) -> queue.Queue:
        """
        Create a queue for a status request (Protocol V2).

        Thread-safe: Creates queue while holding lock, prevents duplicate creation.

        Args:
            request_id: Unique request identifier

        Returns:
            Queue that will receive the status response

        Raises:
            ValueError: If request_id already has a pending queue
        """
        with cls._lock:
            # Check for duplicate request_id to catch programming errors
            if request_id in cls._response_queues:
                logging.warning(f"Request queue already exists for request_id {request_id}, removing old queue")
                cls._response_queues.pop(request_id, None)

            request_queue = queue.Queue(maxsize=1)
            cls._response_queues[request_id] = request_queue
            logging.debug(f"Created response queue for request_id {request_id}")
            return request_queue

    @classmethod
    def put_response(cls, request_id: int, status_data: dict):
        """
        Put a status response from Unity into the appropriate queue (Protocol V2).

        Thread-safe: Validates queue existence before putting data.

        Args:
            request_id: Request identifier from Protocol V2 header
            status_data: Status data from Unity
        """
        with cls._lock:
            if request_id not in cls._response_queues:
                logging.warning(f"No pending request for request_id {request_id} (may have timed out)")
                return

            request_queue = cls._response_queues[request_id]

        # Put outside the lock to avoid blocking other operations
        # queue.Queue is already thread-safe
        try:
            request_queue.put_nowait(status_data)
            logging.debug(f"Put response for request_id {request_id}")
        except queue.Full:
            logging.warning(f"Response queue full for request {request_id}")

    @classmethod
    def get_response(cls, request_id: int, timeout: float = 5.0) -> Optional[dict]:
        """
        Get a response from the queue with timeout (Protocol V2).

        Thread-safe: Validates queue existence before getting data.

        Args:
            request_id: Request identifier
            timeout: Timeout in seconds

        Returns:
            Status data or None if timeout or queue not found
        """
        # Get queue reference while holding lock
        with cls._lock:
            if request_id not in cls._response_queues:
                logging.warning(f"No pending request for request_id {request_id}")
                return None
            request_queue = cls._response_queues[request_id]

        # Get from queue outside the lock (queue.Queue is thread-safe)
        try:
            status_data = request_queue.get(timeout=timeout)
            logging.debug(f"Got response for request_id {request_id}")
            return status_data
        except queue.Empty:
            logging.debug(f"Timeout waiting for response for request_id {request_id}")
            return None

    @classmethod
    def remove_request_queue(cls, request_id: int):
        """
        Remove a request queue after completion (Protocol V2).

        Thread-safe: Safely removes queue even if already removed.

        Args:
            request_id: Request identifier
        """
        with cls._lock:
            if request_id in cls._response_queues:
                cls._response_queues.pop(request_id, None)
                logging.debug(f"Removed response queue for request_id {request_id}")

    @classmethod
    def get_pending_request_count(cls) -> int:
        """
        Get the number of pending requests.

        Thread-safe: Returns count of active request queues.

        Returns:
            Number of pending requests
        """
        with cls._lock:
            return len(cls._response_queues)

    @classmethod
    def clear_all_queues(cls):
        """
        Clear all pending request queues (for cleanup/testing).

        Thread-safe: Removes all queues while holding lock.
        """
        with cls._lock:
            count = len(cls._response_queues)
            cls._response_queues.clear()
            if count > 0:
                logging.info(f"Cleared {count} pending request queues")


class StatusServer(TCPServerBase):
    """
    TCP server that handles status queries from Unity.

    Inherits connection management from TCPServerBase.
    Handles status query decoding and response encoding.

    Sends status requests to ResultsServer (port 5010) via TCP client connection,
    bypassing the ResultsBroadcaster singleton to work across processes.
    """

    def __init__(self, server_config: ServerConfig, results_host: Optional[str] = None, results_port: Optional[int] = None):
        if server_config is None:
            server_config = cfg.get_status_config()

        super().__init__(server_config)
        self._response_handler = StatusResponseHandler.get_instance()

        # Store ResultsServer connection info for TCP client
        self._results_host = results_host or cfg.DEFAULT_HOST
        self._results_port = results_port or cfg.RESULTS_SERVER_PORT

        logging.info(f"StatusServer initialized (will connect to ResultsServer at {self._results_host}:{self._results_port})")

    def handle_client_connection(self, client: socket.socket, address: tuple):
        """
        Handle a Unity client connection (Protocol V2).

        Receives TWO types of messages:
        1. STATUS_QUERY from StatusClient - Unity asks for robot status
        2. STATUS_RESPONSE from StatusResponseSender - Unity sends robot status

        This method is called by TCPServerBase in a separate thread per client.

        Phase 2.3: Persistent Connection Support
        - Sets socket timeout for health checks
        - Enables TCP keepalive for detecting dead connections
        - Allows multiple queries on the same connection
        - Handles StatusResponseSender's one-shot connections separately

        Args:
            client: Client socket
            address: Client address tuple
        """
        logging.info(f"Unity status client connected from {address}")

        # Configure socket for persistent connections (Phase 2.3)
        try:
            # Set socket timeout for periodic health checks
            # This allows the server to periodically check is_running() without blocking indefinitely
            client.settimeout(5.0)

            # Enable TCP keepalive to detect dead connections
            # This helps detect when Unity disconnects without proper shutdown
            client.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

            # Platform-specific keepalive configuration
            if hasattr(socket, 'TCP_KEEPIDLE'):
                # Linux/Mac: Start keepalive after 60 seconds of idle time
                # Use getattr to safely access platform-specific constant
                TCP_KEEPIDLE = getattr(socket, 'TCP_KEEPIDLE')
                client.setsockopt(socket.IPPROTO_TCP, TCP_KEEPIDLE, 60)
            if hasattr(socket, 'TCP_KEEPINTVL'):
                # Linux/Mac: Send keepalive probes every 10 seconds
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
            if hasattr(socket, 'TCP_KEEPCNT'):
                # Linux/Mac: Close connection after 3 failed probes
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)

            logging.debug(f"Configured persistent connection for {address} (timeout=5s, keepalive=enabled)")
        except Exception as e:
            logging.warning(f"Failed to configure socket options for {address}: {e}")

        try:
            while self.is_running():
                try:
                    # Update state to IDLE before receiving
                    self._update_client_state(client, ConnectionState.IDLE)

                    # Receive message using Protocol V2
                    request_id, message_type, message_data = self._receive_message(client)

                    if message_data is None:
                        # Check if it was a fatal error or just idle timeout
                        client_info = self.get_client_info(client)
                        if client_info and client_info.state == ConnectionState.ERROR:
                            # Fatal error - exit gracefully
                            break
                        # Otherwise timeout/idle - continue to check is_running()
                        continue

                    # Import MessageType for comparison
                    from core.UnityProtocol import MessageType

                    if message_type == MessageType.STATUS_RESPONSE:
                        # This is a status response from Unity StatusResponseSender
                        robot_id = message_data.get("robot_id", "unknown")
                        logging.debug(f"[req={request_id}] Received status response from Unity for {robot_id}")

                        # Put response into queue for waiting query handler
                        self._response_handler.put_response(request_id, message_data)

                        # Close this connection (StatusResponseSender sends one message and disconnects)
                        break

                    elif message_type == MessageType.STATUS_QUERY:
                        # This is a status query from Unity StatusClient
                        robot_id = message_data.get("robot_id", "")
                        detailed = message_data.get("detailed", False)

                        logging.info(
                            f"[req={request_id}] Status query from {address}: robot='{robot_id}' detailed={detailed}"
                        )

                        # Execute status query
                        status_result = self._query_robot_status(robot_id, detailed, request_id)

                        # Send results back to Unity (Protocol V2)
                        response_message = UnityProtocol.encode_status_response(
                            status_result, request_id
                        )
                        client.sendall(response_message)

                        logging.debug(f"[req={request_id}] Sent status for {robot_id} to {address}")

                        # Keep connection open for more queries (persistent connection)
                        # Client can send multiple queries on the same connection

                    else:
                        logging.warning(f"Unknown message type: {message_type}")
                        break

                except socket.timeout:
                    # Timeout is expected - allows checking is_running() periodically
                    # Keep connection alive for persistent connections
                    continue
                except (ConnectionResetError, BrokenPipeError, OSError) as e:
                    # Connection lost - client disconnected
                    logging.debug(f"Connection lost from {address}: {e}")
                    break

        except Exception as e:
            logging.debug(f"Client connection error: {e}")
        finally:
            logging.info(f"Unity status client disconnected from {address}")

    def _receive_message(self, client: socket.socket) -> tuple:
        """
        Receive a message from Unity using Protocol V2.

        Reads the message type header and decodes the appropriate message format.

        Args:
            client: Client socket

        Returns:
            Tuple of (request_id, message_type, message_data) or (0, None, None) if failed
        """
        try:
            from core.UnityProtocol import MessageType

            # Read header (type + request_id) - Protocol V2
            header_bytes = self._recv_exactly(client, UnityProtocol.HEADER_SIZE)
            if not header_bytes:
                return (0, None, None)

            # Decode header
            msg_type = header_bytes[0]
            request_id = struct.unpack(UnityProtocol.INT_FORMAT, header_bytes[1:5])[0]

            # Decode based on message type
            if msg_type == MessageType.STATUS_QUERY:
                # Decode status query: [robot_id_len:4][robot_id:N][detailed:1]
                robot_id_len_bytes = self._recv_exactly(client, UnityProtocol.INT_SIZE)
                if not robot_id_len_bytes:
                    return (0, None, None)

                robot_id_len = struct.unpack(UnityProtocol.INT_FORMAT, robot_id_len_bytes)[0]

                if robot_id_len > UnityProtocol.MAX_STRING_LENGTH:
                    logging.error(f"Robot ID length {robot_id_len} exceeds maximum")
                    return (0, None, None)

                robot_id_bytes = self._recv_exactly(client, robot_id_len)
                if not robot_id_bytes:
                    return (0, None, None)
                robot_id = robot_id_bytes.decode("utf-8")

                detailed_bytes = self._recv_exactly(client, 1)
                if not detailed_bytes:
                    return (0, None, None)
                detailed = struct.unpack("B", detailed_bytes)[0]

                query_data = {
                    "robot_id": robot_id,
                    "detailed": bool(detailed)
                }

                return (request_id, MessageType.STATUS_QUERY, query_data)

            elif msg_type == MessageType.STATUS_RESPONSE:
                # Decode status response: [json_len:4][json_data:N]
                json_len_bytes = self._recv_exactly(client, UnityProtocol.INT_SIZE)
                if not json_len_bytes:
                    return (0, None, None)

                json_len = struct.unpack(UnityProtocol.INT_FORMAT, json_len_bytes)[0]

                if json_len > UnityProtocol.MAX_IMAGE_SIZE:
                    logging.error(f"JSON length {json_len} exceeds maximum")
                    return (0, None, None)

                json_bytes = self._recv_exactly(client, json_len)
                if not json_bytes:
                    return (0, None, None)

                json_str = json_bytes.decode("utf-8")
                status_data = json.loads(json_str)

                return (request_id, MessageType.STATUS_RESPONSE, status_data)

            else:
                logging.error(f"Unknown message type: {msg_type}")
                return (0, None, None)

        except Exception as e:
            logging.debug(f"Error receiving message: {e}")
            return (0, None, None)

    def _recv_exactly(self, sock: socket.socket, num_bytes: int) -> Optional[bytes]:
        """
        Receive exactly num_bytes from socket.

        Args:
            sock: Socket to receive from
            num_bytes: Exact number of bytes to receive

        Returns:
            Bytes received or None if connection closed or error
        """
        # Update state to receiving
        self._update_client_state(sock, ConnectionState.RECEIVING)

        data = b""
        while len(data) < num_bytes:
            try:
                chunk = sock.recv(num_bytes - len(data))
                if not chunk:
                    # Connection closed cleanly
                    logging.debug(f"Client connection closed cleanly")
                    self._update_client_state(sock, ConnectionState.DISCONNECTED)
                    return None
                data += chunk
                self._record_bytes_received(sock, len(chunk))

            except Exception as e:
                # Determine if error is fatal
                is_fatal, error_desc = self._is_connection_error_fatal(e)

                if is_fatal:
                    # Fatal error - connection lost
                    logging.debug(f"Client disconnected: {error_desc}")
                    self._record_client_error(sock)
                else:
                    # Non-fatal error - just log at debug level
                    logging.debug(f"Socket idle: {error_desc}")

                return None

        return data

    def _send_to_results_server(self, message_dict: Dict, request_id: int) -> bool:
        """
        Send a message to ResultsServer via TCP client connection (Protocol V2).

        This bypasses the ResultsBroadcaster singleton, allowing StatusServer
        to work across process boundaries.

        Args:
            message_dict: Dictionary containing the message to send
            request_id: Request ID for Protocol V2 correlation

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Create TCP socket
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(5.0)

            # Connect to ResultsServer
            client_socket.connect((self._results_host, self._results_port))

            # Encode message using UnityProtocol (Protocol V2)
            message_bytes = UnityProtocol.encode_result_message(message_dict, request_id)

            # Send message
            client_socket.sendall(message_bytes)

            # Close connection
            client_socket.close()

            logging.debug(f"[req={request_id}] Sent message to ResultsServer: {message_dict.get('command_type', 'unknown')}")
            return True

        except socket.timeout:
            logging.error(f"Timeout connecting to ResultsServer at {self._results_host}:{self._results_port}")
            return False
        except ConnectionRefusedError:
            logging.error(f"ResultsServer not available at {self._results_host}:{self._results_port}")
            return False
        except Exception as e:
            logging.error(f"Error sending to ResultsServer: {e}")
            return False

    def _query_robot_status(self, robot_id: str, detailed: bool, request_id: int) -> Dict:
        """
        Query robot status by sending request to Unity and waiting for response (Protocol V2).

        This creates a bidirectional flow:
        1. Unity client sends status query to this server
        2. Server sends status request to Unity via TCP connection to ResultsServer (port 5010)
        3. Unity PythonCommandHandler processes request and sends status back
        4. Server receives status and returns it to original Unity client

        Protocol V2: Uses request_id for queue management to avoid race conditions.
        Thread-safe: Uses StatusResponseHandler's safe queue access methods.

        Args:
            robot_id: Robot identifier
            detailed: If True, return detailed joint information
            request_id: Unique request identifier for correlation

        Returns:
            Robot status dictionary
        """
        # Create response queue using request_id (not robot_id) - Protocol V2
        # This is thread-safe and will warn if duplicate request_id is used
        self._response_handler.create_request_queue(request_id)

        try:
            # Send status request to Unity via TCP client to ResultsServer
            status_request = {
                "command_type": "check_robot_status",
                "robot_id": robot_id,
                "parameters": {"detailed": detailed},
                "timestamp": time.time(),
            }

            success = self._send_to_results_server(status_request, request_id)

            if not success:
                return {
                    "success": False,
                    "error": {
                        "code": "RESULTS_SERVER_UNAVAILABLE",
                        "message": f"Cannot connect to ResultsServer at {self._results_host}:{self._results_port}",
                    },
                }

            # Wait for response from Unity (timeout 5 seconds) using thread-safe method
            status_data = self._response_handler.get_response(request_id, timeout=5.0)

            if status_data is None:
                # Timeout or queue not found
                return {
                    "success": False,
                    "error": {
                        "code": "TIMEOUT",
                        "message": f"Unity did not respond with status for {robot_id} within 5 seconds",
                    },
                }

            return status_data

        finally:
            # Clean up response queue using request_id - Protocol V2
            # Thread-safe removal
            self._response_handler.remove_request_queue(request_id)


def run_status_server(server_config: ServerConfig, setup_signals: bool = True):
    """
    Start the StatusServer (blocking).

    Args:
        server_config: Server configuration
        setup_signals: If True, setup signal handlers (only valid in main thread)
    """
    import signal

    # Create server
    server = StatusServer(server_config)

    # Setup signal handlers (only if in main thread)
    if setup_signals:

        def signal_handler(_sig, _frame):
            logging.info("Shutdown signal received")
            server.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    try:
        server.start()
        logging.info("StatusServer ready to handle queries from Unity")

        # Keep server running
        while server.is_running():
            time.sleep(1.0)

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        server.stop()


def run_status_server_background(server_config: ServerConfig):
    """
    Start the StatusServer in a background thread.

    Args:
        server_config: Server configuration

    Returns:
        Thread object running the server
    """
    import threading

    server_config = server_config or cfg.get_status_config()

    # Start server in background thread
    thread = threading.Thread(
        target=run_status_server,
        args=(server_config, False),  # setup_signals=False
        daemon=True,
    )
    thread.start()
    logging.info("StatusServer started in background thread")
    return thread


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Status server for Unity robot queries"
    )
    parser.add_argument("--host", default=cfg.DEFAULT_HOST, help="Host to bind to")
    parser.add_argument(
        "--port", type=int, default=cfg.STATUS_SERVER_PORT, help="Port to bind to"
    )

    args = parser.parse_args()

    server_config = ServerConfig(host=args.host, port=args.port)

    # Start server
    run_status_server(server_config)
