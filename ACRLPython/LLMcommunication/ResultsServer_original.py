#!/usr/bin/env python3
"""
ResultsServer.py - Sends LLM analysis results back to Unity

This server provides a singleton that Unity can connect to and receive
LLM analysis results in real-time. Works alongside StreamingServer.

Protocol:
    1. Unity connects to this server (default port 5006)
    2. Server sends results as JSON when available
    3. Format: [length:4bytes][json_data]

Usage:
    from ResultsServer import ResultsNotifier

    # Send result to Unity
    result = {"camera_id": "AR4Left", "response": "Object detected"}
    ResultsNotifier.send_result(result)
"""

import socket
import struct
import json
import threading
import logging
import time
from typing import Dict, Optional, List
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


@dataclass
class ResultsServerConfig:
    """Configuration for the results server"""

    host: str = "127.0.0.1"
    port: int = 5006
    max_queue_size: int = 100


class ResultsNotifier:
    """
    Singleton for sending LLM results to Unity.

    This class maintains connections to Unity clients and broadcasts
    analysis results when they become available.
    """

    _instance = None
    _clients: List[socket.socket] = []
    _clients_lock = threading.Lock()
    _result_queue: List[Dict] = []
    _queue_lock = threading.Lock()
    _config: Optional[ResultsServerConfig] = None
    _server_running = False

    @classmethod
    def initialize(cls, config: ResultsServerConfig = None):
        """
        Initialize the notifier with configuration

        Args:
            config: Server configuration (uses defaults if None)
        """
        if cls._instance is None:
            cls._instance = cls()
            cls._config = config or ResultsServerConfig()
            logging.info(
                f"ResultsNotifier initialized (port {cls._config.port})"
            )

    @classmethod
    def send_result(cls, result: Dict) -> bool:
        """
        Send a result to all connected Unity clients

        Args:
            result: Dictionary containing analysis result

        Returns:
            True if result was queued/sent successfully
        """
        if cls._config is None:
            logging.warning("ResultsNotifier not initialized - result not sent")
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

            with cls._clients_lock:
                if not cls._clients:
                    # No clients connected - queue the result
                    with cls._queue_lock:
                        if len(cls._result_queue) >= cls._config.max_queue_size:
                            cls._result_queue.pop(0)  # Remove oldest
                        cls._result_queue.append(result)
                    logging.debug("Result queued (no clients connected)")
                    return True

                # Send to all connected clients
                cls._broadcast_result(result)
                return True

        except Exception as e:
            logging.error(f"Error sending result: {e}")
            return False

    @classmethod
    def _broadcast_result(cls, result: Dict):
        """
        Broadcast result to all connected clients (internal use)

        Args:
            result: Result dictionary to send
        """
        # Encode result as JSON
        json_data = json.dumps(result, ensure_ascii=False).encode("utf-8")
        data_length = len(json_data)

        # Build message: [length:4bytes][json_data]
        message = struct.pack("I", data_length) + json_data

        disconnected_clients = []

        # Send to all clients
        for client in cls._clients:
            try:
                client.sendall(message)
                logging.debug(f"Sent result to client: {client.getpeername()}")
            except Exception as e:
                logging.warning(f"Failed to send to client: {e}")
                disconnected_clients.append(client)

        # Remove disconnected clients
        for client in disconnected_clients:
            try:
                client.close()
            except:
                pass
            cls._clients.remove(client)
            logging.info(f"Removed disconnected client")

    @classmethod
    def add_client(cls, client: socket.socket):
        """
        Add a new Unity client connection

        Args:
            client: Client socket
        """
        with cls._clients_lock:
            cls._clients.append(client)
            logging.info(
                f"Unity client connected: {client.getpeername()} (total: {len(cls._clients)})"
            )

            # Send any queued results to the new client
            with cls._queue_lock:
                if cls._result_queue:
                    logging.info(
                        f"Sending {len(cls._result_queue)} queued results to new client"
                    )
                    for result in cls._result_queue:
                        try:
                            json_data = json.dumps(result, ensure_ascii=False).encode(
                                "utf-8"
                            )
                            message = struct.pack("I", len(json_data)) + json_data
                            client.sendall(message)
                        except Exception as e:
                            logging.warning(f"Failed to send queued result: {e}")
                            break
                    # Clear queue after sending
                    cls._result_queue.clear()

    @classmethod
    def remove_client(cls, client: socket.socket):
        """
        Remove a Unity client connection

        Args:
            client: Client socket to remove
        """
        with cls._clients_lock:
            if client in cls._clients:
                cls._clients.remove(client)
                logging.info(
                    f"Unity client disconnected (remaining: {len(cls._clients)})"
                )

    @classmethod
    def get_client_count(cls) -> int:
        """Get number of connected Unity clients"""
        with cls._clients_lock:
            return len(cls._clients)


def handle_client(client: socket.socket):
    """
    Handle a Unity client connection

    Args:
        client: Client socket
    """
    ResultsNotifier.add_client(client)

    try:
        # Keep connection alive (Unity will disconnect when done)
        while True:
            # Simple keep-alive - receive any data Unity might send
            data = client.recv(1024)
            if not data:
                break  # Client disconnected
    except Exception as e:
        logging.debug(f"Client connection closed: {e}")
    finally:
        ResultsNotifier.remove_client(client)
        try:
            client.close()
        except:
            pass


def accept_clients(server_socket: socket.socket, config: ResultsServerConfig):
    """
    Accept incoming Unity client connections

    Args:
        server_socket: Server socket
        config: Server configuration
    """
    server_socket.settimeout(1.0)  # Allow periodic checks

    while ResultsNotifier._server_running:
        try:
            client, addr = server_socket.accept()
            logging.info(f"Unity results client connected from {addr}")
            threading.Thread(target=handle_client, args=(client,), daemon=True).start()
        except socket.timeout:
            continue
        except Exception as e:
            if ResultsNotifier._server_running:
                logging.error(f"Error accepting client: {e}")


def run_results_server(config: ResultsServerConfig = None):
    """
    Start the results server (blocking)

    Args:
        config: Server configuration
    """
    config = config or ResultsServerConfig()
    ResultsNotifier.initialize(config)
    ResultsNotifier._server_running = True

    server_socket = None

    try:
        # Create server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((config.host, config.port))
        server_socket.listen(5)
        logging.info(
            f"ResultsServer listening on {config.host}:{config.port}..."
        )
        logging.info("Waiting for Unity to connect and receive LLM results")

        # Accept clients
        accept_clients(server_socket, config)

    except Exception as e:
        logging.error(f"ResultsServer error: {e}")
    finally:
        ResultsNotifier._server_running = False

        if server_socket:
            try:
                server_socket.close()
            except:
                pass

        logging.info("ResultsServer shutdown complete")


def run_results_server_background(config: ResultsServerConfig = None):
    """
    Start the results server in a background thread

    Args:
        config: Server configuration

    Returns:
        Thread object
    """
    config = config or ResultsServerConfig()
    ResultsNotifier.initialize(config)

    thread = threading.Thread(
        target=run_results_server, args=(config,), daemon=True
    )
    thread.start()
    logging.info("ResultsServer started in background thread")
    return thread


if __name__ == "__main__":
    # Test mode - run server and send test results
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM results server for Unity")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=5006, help="Port to bind to (default: 5006)"
    )
    parser.add_argument(
        "--test", action="store_true", help="Send test results every 5 seconds"
    )

    args = parser.parse_args()

    config = ResultsServerConfig(host=args.host, port=args.port)

    if args.test:
        # Run server in background and send test results
        run_results_server_background(config)

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
                    "timestamp": time.time(),
                    "metadata": {
                        "model": "test-model",
                        "duration_seconds": 0.5,
                        "test": True,
                    },
                }

                ResultsNotifier.send_result(test_result)
                print(f"Sent test result #{counter}")

        except KeyboardInterrupt:
            print("\nStopping test server...")
    else:
        # Run server in foreground
        run_results_server(config)
