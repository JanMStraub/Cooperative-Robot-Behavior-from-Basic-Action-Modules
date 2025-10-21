#!/usr/bin/env python3
"""
TCPServerBase.py - Base class for TCP servers

Provides common functionality for all TCP servers:
- Client connection management
- Thread-safe client tracking
- Graceful shutdown
- Error handling

All Unity-facing servers should inherit from this class.
"""

import socket
import threading
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


@dataclass
class ServerConfig:
    """Configuration for TCP servers"""
    host: str = "127.0.0.1"
    port: int = 5000
    max_connections: int = 5
    socket_timeout: float = 1.0  # Timeout for accept() to allow periodic shutdown checks


class TCPServerBase(ABC):
    """
    Abstract base class for TCP servers.

    Subclasses must implement:
    - handle_client_connection(): Process individual client connections
    """

    def __init__(self, config: ServerConfig):
        """
        Initialize the TCP server.

        Args:
            config: Server configuration
        """
        self._config = config
        self._running = False
        self._clients: List[socket.socket] = []
        self._clients_lock = threading.Lock()
        self._server_socket: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None

    @abstractmethod
    def handle_client_connection(self, client: socket.socket, address: tuple):
        """
        Handle a client connection (must be implemented by subclass).

        This method is called in a separate thread for each client.

        Args:
            client: Client socket
            address: Client address tuple (host, port)
        """
        pass

    def start(self):
        """Start the TCP server"""
        if self._running:
            logging.warning(f"Server already running on {self._config.host}:{self._config.port}")
            return

        try:
            # Create server socket
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.bind((self._config.host, self._config.port))
            self._server_socket.listen(self._config.max_connections)
            self._server_socket.settimeout(self._config.socket_timeout)

            self._running = True
            logging.info(f"Server started on {self._config.host}:{self._config.port}")

            # Start accept thread
            self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
            self._accept_thread.start()

        except Exception as e:
            logging.error(f"Failed to start server: {e}")
            self._cleanup()
            raise

    def stop(self):
        """Stop the TCP server gracefully"""
        if not self._running:
            return

        logging.info("Stopping server...")
        self._running = False

        # Wait for accept thread to finish
        if self._accept_thread and self._accept_thread.is_alive():
            self._accept_thread.join(timeout=2.0)

        # Close all client connections
        with self._clients_lock:
            for client in self._clients:
                try:
                    client.close()
                except Exception as e:
                    logging.debug(f"Error closing client: {e}")
            self._clients.clear()

        # Close server socket
        self._cleanup()
        logging.info("Server stopped")

    def is_running(self) -> bool:
        """Check if server is running"""
        return self._running

    def get_client_count(self) -> int:
        """Get number of connected clients"""
        with self._clients_lock:
            return len(self._clients)

    def broadcast_to_all_clients(self, data: bytes) -> int:
        """
        Send data to all connected clients.

        Args:
            data: Bytes to send

        Returns:
            Number of clients successfully sent to
        """
        disconnected = []
        success_count = 0

        with self._clients_lock:
            for client in self._clients:
                try:
                    client.sendall(data)
                    success_count += 1
                except Exception as e:
                    logging.warning(f"Failed to send to client {client.getpeername()}: {e}")
                    disconnected.append(client)

            # Remove disconnected clients
            for client in disconnected:
                self._remove_client(client)

        return success_count

    def _accept_loop(self):
        """Accept incoming client connections (runs in separate thread)"""
        while self._running:
            try:
                client, address = self._server_socket.accept()
                logging.info(f"Client connected from {address}")

                # Add to client list
                with self._clients_lock:
                    self._clients.append(client)

                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self._handle_client_wrapper,
                    args=(client, address),
                    daemon=True
                )
                client_thread.start()

            except socket.timeout:
                # Expected - allows us to check _running periodically
                continue
            except Exception as e:
                if self._running:
                    logging.error(f"Error accepting client: {e}")

    def _handle_client_wrapper(self, client: socket.socket, address: tuple):
        """
        Wrapper around subclass's handle_client_connection method.
        Ensures cleanup happens even if handler raises exception.
        """
        try:
            self.handle_client_connection(client, address)
        except Exception as e:
            logging.error(f"Error handling client {address}: {e}")
        finally:
            # Remove from client list
            with self._clients_lock:
                self._remove_client(client)

            # Close client socket
            try:
                client.close()
            except:
                pass

            logging.info(f"Client disconnected from {address}")

    def _remove_client(self, client: socket.socket):
        """Remove a client from the tracked list (must hold _clients_lock)"""
        try:
            self._clients.remove(client)
        except ValueError:
            pass  # Already removed

    def _cleanup(self):
        """Clean up server socket"""
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception as e:
                logging.debug(f"Error closing server socket: {e}")
            self._server_socket = None


if __name__ == "__main__":
    # Example usage and testing
    class EchoServer(TCPServerBase):
        """Simple echo server for testing"""

        def handle_client_connection(self, client: socket.socket, address: tuple):
            """Echo back any data received"""
            try:
                while True:
                    data = client.recv(1024)
                    if not data:
                        break
                    client.sendall(b"ECHO: " + data)
            except Exception as e:
                logging.debug(f"Echo client error: {e}")

    # Test the echo server
    config = ServerConfig(host="127.0.0.1", port=9999)
    server = EchoServer(config)

    print("Starting echo server on port 9999...")
    print("Test with: echo 'hello' | nc localhost 9999")
    print("Press Ctrl+C to stop")

    server.start()

    try:
        import time
        while True:
            time.sleep(1)
            print(f"Server running, clients: {server.get_client_count()}")
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop()
