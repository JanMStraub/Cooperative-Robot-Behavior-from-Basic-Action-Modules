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

# Import ResultsBroadcaster for sending status requests to Unity
try:
    from servers.ResultsServer import ResultsBroadcaster
except ImportError:
    from ..servers.ResultsServer import ResultsBroadcaster

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
    """

    def __init__(self, server_config: ServerConfig):
        if server_config is None:
            server_config = cfg.get_status_config()

        super().__init__(server_config)
        self._response_handler = StatusResponseHandler.get_instance()
        logging.info("StatusServer initialized")

    def handle_client_connection(self, client: socket.socket, address: tuple):
        """
        Handle a Unity client connection.

        Receives status query messages and sends back robot status.
        This method is called by TCPServerBase in a separate thread per client.

        Args:
            client: Client socket
            address: Client address tuple
        """
        logging.info(f"Unity status client connected from {address}")

        try:
            while self.is_running():
                # Set timeout for query receive
                client.settimeout(30.0)

                try:
                    # Receive query message from Unity
                    query_data = self._receive_query_message(client)

                    if query_data is None:
                        # Client disconnected gracefully
                        break

                    # Extract query parameters
                    robot_id = query_data["robot_id"]
                    detailed = query_data.get("detailed", False)

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
                    # Expected - allows checking is_running()
                    continue

        except Exception as e:
            logging.debug(f"Client connection error: {e}")
        finally:
            logging.info(f"Unity status client disconnected from {address}")

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

    def _query_robot_status(self, robot_id: str, detailed: bool) -> Dict:
        """
        Query robot status by sending request to Unity and waiting for response.

        This creates a bidirectional flow:
        1. Unity client sends status query to this server
        2. Server sends status request to Unity via ResultsBroadcaster (port 5010)
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
            # Send status request to Unity via ResultsBroadcaster
            status_request = {
                "command_type": "get_robot_status",
                "robot_id": robot_id,
                "parameters": {"detailed": detailed},
                "timestamp": time.time(),
            }

            success = ResultsBroadcaster.send_result(status_request)

            if not success:
                return {
                    "success": False,
                    "error": {
                        "code": "UNITY_NOT_CONNECTED",
                        "message": "Unity not connected to ResultsServer (port 5010)",
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
