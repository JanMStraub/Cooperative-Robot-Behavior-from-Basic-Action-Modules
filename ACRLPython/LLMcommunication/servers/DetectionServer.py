#!/usr/bin/env python3
"""
DetectionServer.py - TCP server for broadcasting object detection results to Unity

This server listens for Unity client connections on port 5007 and broadcasts
object detection results (cube positions in pixel coordinates) to all connected clients.

Architecture:
- Inherits from TCPServerBase for connection management
- Uses DetectionBroadcaster singleton to send results from any thread
- Clients connect from Unity (DetectionResultsReceiver.cs)

Usage:
    # Start server in background thread
    from DetectionServer import run_detection_server_background
    from core.TCPServerBase import ServerConfig

    config = ServerConfig(host="127.0.0.1", port=5007)
    run_detection_server_background(config)

    # Send detection results from anywhere
    from DetectionServer import DetectionBroadcaster
    DetectionBroadcaster.send_result(detection_dict)
"""

import logging
import socket
import threading
import queue
from typing import Dict, List
from datetime import datetime
import sys
from pathlib import Path

# Import base class and protocol
from core.TCPServerBase import TCPServerBase, ServerConfig
from core.UnityProtocol import UnityProtocol

# Add LLMCommunication package directory to path
_package_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_package_dir))

# Import config
import config as cfg

class DetectionBroadcaster:
    """
    Singleton broadcaster for sending detection results to all connected Unity clients.
    Thread-safe and can be called from any thread.
    """

    _instance = None
    _lock = threading.Lock()
    _clients: List[socket.socket] = []
    _clients_lock = threading.Lock()
    _result_queue = queue.Queue(maxsize=cfg.MAX_RESULT_QUEUE_SIZE)

    @classmethod
    def get_instance(cls):
        """
        Get the singleton instance

        Returns:
            DetectionBroadcaster instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def register_client(cls, client_socket: socket.socket):
        """
        Register a new Unity client connection

        Args:
            client_socket: Socket of the connected client
        """
        newly_registered = False
        with cls._clients_lock:
            if client_socket not in cls._clients:
                cls._clients.append(client_socket)
                newly_registered = True
                logging.info(
                    f"DetectionBroadcaster: Registered client (total: {len(cls._clients)})"
                )

        # Send any queued results to the newly connected client
        if newly_registered and not cls._result_queue.empty():
            cls._send_queued_results()

    @classmethod
    def unregister_client(cls, client_socket: socket.socket):
        """
        Unregister a disconnected Unity client

        Args:
            client_socket: Socket of the disconnected client
        """
        with cls._clients_lock:
            if client_socket in cls._clients:
                cls._clients.remove(client_socket)
                logging.info(
                    f"DetectionBroadcaster: Unregistered client (total: {len(cls._clients)})"
                )

    @classmethod
    def send_result(cls, result: Dict, _from_queue: bool = False):
        """
        Send detection result to all connected Unity clients

        Args:
            result: Detection result dictionary (from DetectionResult.to_dict())
            _from_queue: Internal flag to prevent infinite recursion
        """
        # Add metadata
        if "metadata" not in result:
            result["metadata"] = {}
        result["metadata"]["server_timestamp"] = datetime.now().isoformat()

        # Encode using Unity protocol (pass dict directly)
        try:
            message = UnityProtocol.encode_result_message(result)
        except Exception as e:
            logging.error(f"Failed to encode detection result: {e}")
            return

        # Broadcast to all clients
        with cls._clients_lock:
            if not cls._clients:
                logging.debug("No clients connected, queueing result")
                try:
                    cls._result_queue.put_nowait(result)
                except queue.Full:
                    logging.warning("Result queue full, dropping oldest result")
                    try:
                        cls._result_queue.get_nowait()
                        cls._result_queue.put_nowait(result)
                    except:
                        pass
                return

            # Send to all clients
            disconnected_clients = []
            for client in cls._clients:
                try:
                    client.sendall(message)
                    logging.debug(
                        f"Sent detection result to client (camera: {result.get('camera_id', 'unknown')})"
                    )
                except Exception as e:
                    logging.error(f"Failed to send result to client: {e}")
                    disconnected_clients.append(client)

            # Remove disconnected clients (do this while still holding the lock)
            for client in disconnected_clients:
                if client in cls._clients:
                    cls._clients.remove(client)
                    logging.info(
                        f"DetectionBroadcaster: Unregistered client (total: {len(cls._clients)})"
                    )

    @classmethod
    def _send_queued_results(cls):
        """
        Send queued results when a new client connects
        """
        while not cls._result_queue.empty():
            try:
                queued_result = cls._result_queue.get_nowait()
                cls.send_result(queued_result, _from_queue=True)
            except queue.Empty:
                break


class DetectionServer(TCPServerBase):
    """
    TCP server for broadcasting detection results to Unity clients
    """

    def __init__(self, config: ServerConfig):
        """
        Initialize the detection server

        Args:
            config: Server configuration (host, port, etc.)
        """
        super().__init__(config)

    def handle_client_connection(self, client: socket.socket, address: tuple):
        """
        Handle a single Unity client connection.
        The client just needs to stay connected to receive broadcasts.

        Args:
            client: Client socket
            address: Client address tuple
        """
        try:
            # Register client for broadcasts
            DetectionBroadcaster.register_client(client)

            # Keep connection alive and wait for disconnection
            # Unity clients don't send data, they just receive results
            client.settimeout(5.0)  # Check for disconnection every 5s

            while self._running:
                try:
                    # Try to receive data (will timeout if no data)
                    # This is just to detect disconnection
                    data = client.recv(1)
                    if not data:
                        # Client disconnected
                        logging.info(f"Client {address} disconnected")
                        break
                except socket.timeout:
                    # No data received, just continue (client still connected)
                    continue
                except Exception as e:
                    logging.debug(f"Client {address} connection error: {e}")
                    break

        except Exception as e:
            logging.error(f"Error handling client {address}: {e}")
        finally:
            # Unregister client
            DetectionBroadcaster.unregister_client(client)

            # Close connection
            try:
                client.close()
            except:
                pass


def run_detection_server_background(config: ServerConfig) -> DetectionServer:
    """
    Start the detection server in a background thread

    Args:
        config: Server configuration

    Returns:
        DetectionServer instance (running in background)
    """
    server = DetectionServer(config)

    def server_thread():
        try:
            server.start()
        except Exception as e:
            logging.error(f"DetectionServer thread error: {e}")

    thread = threading.Thread(target=server_thread, daemon=True, name="DetectionServer")
    thread.start()

    logging.info(f"DetectionServer started in background on {config.host}:{config.port}")
    return server


def main():
    """
    Run the detection server as a standalone process
    """
    import time

    logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL), format=cfg.LOG_FORMAT)

    # Create server config
    config = ServerConfig(host=cfg.DEFAULT_HOST, port=cfg.DETECTION_SERVER_PORT)

    logging.info("Starting DetectionServer...")
    logging.info(f"Listening on {config.host}:{config.port}")
    logging.info("Waiting for Unity client connections...")

    # Start server
    server = DetectionServer(config)

    try:
        server.start()
    except KeyboardInterrupt:
        logging.info("\nShutting down (interrupted by user)")
    except Exception as e:
        logging.error(f"Server error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
