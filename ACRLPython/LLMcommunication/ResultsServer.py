#!/usr/bin/env python3
"""
ResultsServer.py - Sends LLM analysis results back to Unity
"""

import socket
import threading
import logging
import time
from typing import Dict, Optional, List
import signal

# Import config and base classes
import config as cfg
from core.TCPServerBase import TCPServerBase, ServerConfig
from core.UnityProtocol import UnityProtocol

logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=cfg.LOG_FORMAT)


class ResultsBroadcaster:
    """
    Singleton for broadcasting LLM results to Unity clients.

    Renamed from 'ResultsNotifier' for consistency with ImageStorage.
    Maintains reference to ResultsServer for sending data.
    """

    _instance = None
    _server: Optional["ResultsServer"] = None
    _result_queue: List[Dict] = []
    _queue_lock = threading.Lock()
    _max_queue_size: int = cfg.MAX_RESULT_QUEUE_SIZE

    @classmethod
    def initialize(cls, server: "ResultsServer"):
        """Initialize the broadcaster with a server instance"""
        if cls._instance is None:
            cls._instance = cls()
        cls._server = server
        logging.info("ResultsBroadcaster initialized")

    @classmethod
    def send_result(cls, result: Dict) -> bool:
        """
        Send a result to all connected Unity clients.

        Args:
            result: Dictionary containing analysis result

        Returns:
            True if sent or queued successfully
        """
        if cls._server is None:
            logging.warning("ResultsBroadcaster not initialized - result not sent")
            return False

        try:
            # Add camera_id to result if not present
            if "camera_id" not in result and "metadata" in result:
                camera_ids = result["metadata"].get("camera_ids", [])
                if camera_ids:
                    result["camera_id"] = camera_ids[0]

            # Add timestamp if not present
            if "timestamp" not in result and "metadata" in result:
                result["timestamp"] = result["metadata"].get("timestamp", "")

            # Encode the message once before checking client count
            # This prevents race condition where clients disconnect after check
            message = UnityProtocol.encode_result_message(result)

            # Check client count and send in single atomic operation
            client_count = cls._server.get_client_count()

            if client_count == 0:
                # No clients - queue the result
                with cls._queue_lock:
                    if len(cls._result_queue) >= cls._max_queue_size:
                        dropped = cls._result_queue.pop(0)
                        logging.warning(
                            f"Queue full - dropped result for {dropped.get('camera_id', 'unknown')}"
                        )
                    cls._result_queue.append(result)
                logging.debug("Result queued (no clients connected)")
                return True

            # Send to all clients (they may have disconnected, but that's OK)
            sent_count = cls._server.broadcast_to_all_clients(message)

            # If send failed but we thought there were clients, queue it
            if sent_count == 0:
                with cls._queue_lock:
                    if len(cls._result_queue) >= cls._max_queue_size:
                        cls._result_queue.pop(0)
                    cls._result_queue.append(result)
                logging.debug("Result queued (send failed)")

            logging.debug(f"Result sent to {sent_count} client(s)")
            return sent_count > 0 or True  # Queued results also count as success

        except Exception as e:
            logging.error(f"Error sending result: {e}")
            return False

    @classmethod
    def get_queued_results(cls) -> List[Dict]:
        """Get and clear all queued results"""
        with cls._queue_lock:
            results = cls._result_queue.copy()
            cls._result_queue.clear()
            return results


class ResultsServer(TCPServerBase):
    """
    TCP server that sends LLM analysis results to Unity.

    Inherits connection management from TCPServerBase (~120 lines saved).
    Handles result-specific JSON encoding and broadcasting.
    """

    def __init__(self, server_config: ServerConfig = None):
        if server_config is None:
            server_config = cfg.get_results_config()

        super().__init__(server_config)
        ResultsBroadcaster.initialize(self)
        logging.info("ResultsServer initialized")

    def handle_client_connection(self, client: socket.socket, address: tuple):
        """
        Handle a Unity client connection - keeps connection alive for sending results.

        This method is called by TCPServerBase in a separate thread per client.
        Results server only sends data, doesn't receive (except keep-alive).
        """
        logging.info(f"Unity results client connected from {address}")

        # Send any queued results to this new client
        self._send_queued_results(client)

        try:
            # Keep connection alive - receive keep-alive packets from Unity
            while self.is_running():
                # Simple keep-alive check
                client.settimeout(cfg.RESULTS_SERVER_KEEPALIVE)
                try:
                    data = client.recv(1024)
                    if not data:
                        break  # Client disconnected
                except socket.timeout:
                    continue  # Expected - allows checking is_running()

        except Exception as e:
            logging.debug(f"Client connection closed: {e}")

    def _send_queued_results(self, client: socket.socket):
        """Send any queued results to a newly connected client"""
        queued = ResultsBroadcaster.get_queued_results()

        if queued:
            logging.info(f"Sending {len(queued)} queued results to new client")
            for result in queued:
                try:
                    message = UnityProtocol.encode_result_message(result)
                    client.sendall(message)
                except Exception as e:
                    logging.warning(f"Failed to send queued result: {e}")
                    break


def run_results_server(server_config: ServerConfig = None, setup_signals: bool = True):
    """
    Start the ResultsServer (blocking)

    Args:
        server_config: Server configuration
        setup_signals: If True, setup signal handlers (only valid in main thread)
    """
    server = ResultsServer(server_config)

    # Setup signal handlers (only if in main thread)
    if setup_signals:

        def signal_handler(_sig, _frame):
            logging.info("Shutdown signal received")
            server.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    try:
        server.start()
        logging.info("ResultsServer ready to send results to Unity")

        # Keep server running
        while server.is_running():
            time.sleep(cfg.RESULTS_SERVER_KEEPALIVE)

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        server.stop()


def run_results_server_background(server_config: ServerConfig = None):
    """Start the ResultsServer in a background thread"""
    server_config = server_config or cfg.get_results_config()

    thread = threading.Thread(
        target=run_results_server,
        args=(server_config, False),  # setup_signals=False in background thread
        daemon=True,
    )
    thread.start()
    logging.info("ResultsServer started in background thread")
    return thread


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM results server for Unity")
    parser.add_argument("--host", default=cfg.DEFAULT_HOST, help="Host to bind to")
    parser.add_argument(
        "--port", type=int, default=cfg.RESULTS_SERVER_PORT, help="Port to bind to"
    )
    parser.add_argument(
        "--test", action="store_true", help="Send test results every 5 seconds"
    )

    args = parser.parse_args()

    server_config = ServerConfig(host=args.host, port=args.port)

    if args.test:
        # Test mode - run server and send test results
        run_results_server_background(server_config)

        print("Sending test results every 5 seconds...")
        print("Press Ctrl+C to stop")

        try:
            counter = 0
            while True:
                time.sleep(5)
                counter += 1

                test_result = {
                    "camera_id": f"TestCamera{counter % 2}",
                    "response": f"Test response #{counter}: Object detected",
                    "timestamp": str(time.time()),
                    "metadata": {
                        "model": "test-model",
                        "duration_seconds": 0.5,
                        "test": True,
                    },
                }

                ResultsBroadcaster.send_result(test_result)
                print(f"Sent test result #{counter}")

        except KeyboardInterrupt:
            print("\nStopping test server...")
    else:
        # Normal mode
        run_results_server(server_config)
