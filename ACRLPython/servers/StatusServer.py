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
    from core.TCPServerBase import TCPServerBase, ServerConfig
    from core.UnityProtocol import UnityProtocol
except ImportError:
    from ..core.TCPServerBase import TCPServerBase, ServerConfig
    from ..core.UnityProtocol import UnityProtocol

# Configure logging
logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=cfg.LOG_FORMAT)


class StatusResponseHandler:
    """
    Singleton for handling status responses from Unity.

    Manages a queue of pending status requests and matches incoming
    responses from Unity to the waiting clients.
    """

    _instance: Optional["StatusResponseHandler"] = None
    _response_queues: Dict[str, queue.Queue] = {}
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "StatusResponseHandler":
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def create_request_queue(cls, robot_id: str) -> queue.Queue:
        """
        Create a queue for a status request.

        Args:
            robot_id: Robot identifier

        Returns:
            Queue that will receive the status response
        """
        with cls._lock:
            request_queue = queue.Queue(maxsize=1)
            cls._response_queues[robot_id] = request_queue
            return request_queue

    @classmethod
    def put_response(cls, robot_id: str, status_data: dict):
        """
        Put a status response from Unity into the appropriate queue.

        Args:
            robot_id: Robot identifier
            status_data: Status data from Unity
        """
        with cls._lock:
            if robot_id in cls._response_queues:
                try:
                    cls._response_queues[robot_id].put_nowait(status_data)
                except queue.Full:
                    logging.warning(f"Response queue full for {robot_id}")
            else:
                logging.warning(f"No pending request for robot {robot_id}")

    @classmethod
    def remove_request_queue(cls, robot_id: str):
        """Remove a request queue after completion"""
        with cls._lock:
            cls._response_queues.pop(robot_id, None)


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
        Handle a Unity client connection.

        Receives TWO types of messages:
        1. Status queries from StatusClient - Unity asks for robot status
        2. Status responses from StatusResponseSender - Unity sends robot status

        This method is called by TCPServerBase in a separate thread per client.

        Args:
            client: Client socket
            address: Client address tuple
        """
        logging.info(f"Unity status client connected from {address}")

        try:
            while self.is_running():
                # Set timeout for query receive to allow periodic server state checks
                # Longer timeout since Unity may keep connection idle between queries
                # client.settimeout(60.0)

                try:
                    # Peek at the first message to determine type
                    # Both queries and responses start with length field
                    first_message = self._receive_message(client)

                    if first_message is None:
                        # Client disconnected gracefully or timeout occurred
                        # On timeout, continue to keep connection alive
                        continue

                    # Try to parse as status response first (from StatusResponseSender)
                    if self._is_status_response(first_message):
                        # This is a status response from Unity StatusResponseSender
                        robot_id = first_message.get("robot_id", "unknown")
                        logging.debug(f"Received status response from Unity for {robot_id}")

                        # Put response into queue for waiting query handler
                        self._response_handler.put_response(robot_id, first_message)

                        # Close this connection (StatusResponseSender sends one message and disconnects)
                        break
                    else:
                        # This is a status query from Unity StatusClient
                        robot_id = first_message.get("robot_id", "")
                        detailed = first_message.get("detailed", False)

                        logging.info(
                            f"Status query from {address}: robot='{robot_id}' detailed={detailed}"
                        )

                        # Execute status query
                        status_result = self._query_robot_status(robot_id, detailed)

                        # Send results back to Unity
                        response_message = UnityProtocol.encode_status_response(
                            status_result
                        )
                        client.sendall(response_message)

                        logging.debug(f"Sent status for {robot_id} to {address}")

                except socket.timeout:
                    # Timeout is expected - allows checking is_running() periodically
                    # Keep connection alive
                    continue
                except (ConnectionResetError, BrokenPipeError, OSError) as e:
                    # Connection lost - client disconnected
                    logging.debug(f"Connection lost from {address}: {e}")
                    break

        except Exception as e:
            logging.debug(f"Client connection error: {e}")
        finally:
            logging.info(f"Unity status client disconnected from {address}")

    def _receive_message(self, client: socket.socket) -> Optional[Dict]:
        """
        Receive a generic message from Unity (either query or response).

        This method reads the raw bytes and determines the message type.

        Message formats:
        - Query: [robot_id_len][robot_id][detailed_byte]
        - Response: [json_len][json_data]

        Args:
            client: Client socket

        Returns:
            Message data dictionary or None if client disconnected
        """
        try:
            # Read the first length field (4 bytes)
            length_bytes = client.recv(UnityProtocol.INT_SIZE)
            if not length_bytes or len(length_bytes) < UnityProtocol.INT_SIZE:
                return None  # Connection closed

            length = struct.unpack(UnityProtocol.INT_FORMAT, length_bytes)[0]

            # Read the data of that length
            data = bytearray()
            remaining = length
            while remaining > 0:
                chunk = client.recv(min(remaining, 4096))
                if not chunk:
                    return None  # Connection closed
                data.extend(chunk)
                remaining -= len(chunk)

            data_str = data.decode("utf-8")

            # Try to parse as JSON first (status response)
            try:
                parsed = json.loads(data_str)
                if isinstance(parsed, dict) and ("success" in parsed or "status" in parsed or "error" in parsed):
                    # This is a status response from Unity
                    return parsed
            except json.JSONDecodeError:
                pass

            # Not JSON, so it must be a status query
            # Query format: [robot_id_len][robot_id][detailed_byte]
            # We already read the robot_id, now read the detailed byte
            detailed_byte = client.recv(1)
            if not detailed_byte:
                return None

            detailed = struct.unpack("B", detailed_byte)[0]

            return {
                "robot_id": data_str,
                "detailed": bool(detailed)
            }

        except Exception as e:
            logging.debug(f"Error receiving message: {e}")
            return None

    def _is_status_response(self, message: Dict) -> bool:
        """
        Check if a message is a status response (from Unity).

        Status responses have 'success', 'status', or 'error' fields.
        Status queries have 'robot_id' and 'detailed' fields.

        Args:
            message: Parsed message dictionary

        Returns:
            True if this is a status response, False if it's a query
        """
        # Status responses have 'success', 'status', 'error' fields
        if "success" in message or "status" in message or "error" in message:
            return True
        # Status queries have 'robot_id' and 'detailed' fields only
        return False

    def _receive_query_message(self, client: socket.socket) -> Optional[Dict]:
        """
        Receive a status query message from Unity.

        Message format: [robot_id_len][robot_id][detailed]

        Args:
            client: Client socket

        Returns:
            Query data dictionary or None if client disconnected
        """
        try:
            # Read query message using UnityProtocol
            query_data = UnityProtocol.decode_status_query(client)
            return query_data

        except Exception as e:
            logging.debug(f"Error receiving query message: {e}")
            return None

    def _send_to_results_server(self, message_dict: Dict) -> bool:
        """
        Send a message to ResultsServer via TCP client connection.

        This bypasses the ResultsBroadcaster singleton, allowing StatusServer
        to work across process boundaries.

        Args:
            message_dict: Dictionary containing the message to send

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Create TCP socket
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(5.0)

            # Connect to ResultsServer
            client_socket.connect((self._results_host, self._results_port))

            # Encode message using UnityProtocol
            message_bytes = UnityProtocol.encode_result_message(message_dict)

            # Send message
            client_socket.sendall(message_bytes)

            # Close connection
            client_socket.close()

            logging.debug(f"Sent message to ResultsServer: {message_dict.get('command_type', 'unknown')}")
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

    def _query_robot_status(self, robot_id: str, detailed: bool) -> Dict:
        """
        Query robot status by sending request to Unity and waiting for response.

        This creates a bidirectional flow:
        1. Unity client sends status query to this server
        2. Server sends status request to Unity via TCP connection to ResultsServer (port 5010)
        3. Unity PythonCommandHandler processes request and sends status back
        4. Server receives status and returns it to original Unity client

        Args:
            robot_id: Robot identifier
            detailed: If True, return detailed joint information

        Returns:
            Robot status dictionary
        """
        # Create response queue
        response_queue = self._response_handler.create_request_queue(robot_id)

        try:
            # Send status request to Unity via TCP client to ResultsServer
            status_request = {
                "command_type": "get_robot_status",
                "robot_id": robot_id,
                "parameters": {"detailed": detailed},
                "timestamp": time.time(),
            }

            success = self._send_to_results_server(status_request)

            if not success:
                return {
                    "success": False,
                    "error": {
                        "code": "RESULTS_SERVER_UNAVAILABLE",
                        "message": f"Cannot connect to ResultsServer at {self._results_host}:{self._results_port}",
                    },
                }

            # Wait for response from Unity (timeout 5 seconds)
            try:
                status_data = response_queue.get(timeout=5.0)
                return status_data

            except queue.Empty:
                return {
                    "success": False,
                    "error": {
                        "code": "TIMEOUT",
                        "message": f"Unity did not respond with status for {robot_id} within 5 seconds",
                    },
                }

        finally:
            # Clean up response queue
            self._response_handler.remove_request_queue(robot_id)


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
