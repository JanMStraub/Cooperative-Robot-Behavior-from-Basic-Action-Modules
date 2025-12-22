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
import errno
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from enum import Enum
from datetime import datetime

# Import config - try both import styles
try:
    import LLMConfig as cfg
    from core.LoggingSetup import setup_logging
except ImportError:
    from .. import LLMConfig as cfg
    from ..core.LoggingSetup import setup_logging


class ConnectionState(Enum):
    """Connection state enumeration"""

    CONNECTED = "connected"
    IDLE = "idle"
    RECEIVING = "receiving"
    SENDING = "sending"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class ClientInfo:
    """Information about a connected client"""

    socket: socket.socket
    address: tuple
    state: ConnectionState
    connected_at: datetime
    last_activity: datetime
    bytes_received: int = 0
    bytes_sent: int = 0
    error_count: int = 0


@dataclass
class ServerConfig:
    """Configuration for TCP servers"""

    host: str = cfg.DEFAULT_HOST
    port: int = 5000
    max_connections: int = cfg.MAX_CONNECTIONS_BACKLOG  # Max backlog for listen()
    max_client_threads: int = (
        cfg.MAX_CLIENT_THREADS
    )  # Max concurrent client handler threads
    socket_timeout: float = (
        cfg.SOCKET_ACCEPT_TIMEOUT
    )  # Timeout for accept() to allow periodic shutdown checks


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
        self._shutdown_flag = False
        self._clients: List[socket.socket] = []
        self._clients_lock = threading.Lock()
        self._client_info: Dict[socket.socket, ClientInfo] = {}  # Track client state
        self._server_socket: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._client_threads: List[threading.Thread] = (
            []
        )  # Track client handler threads
        self._client_threads_lock = threading.Lock()  # Protect thread list

        # Setup centralized logging
        self._logger = setup_logging(self.__class__.__module__)

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
            self._logger.warning(
                f"Server already running on {self._config.host}:{self._config.port}"
            )
            return

        try:
            # Create server socket
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.bind((self._config.host, self._config.port))
            self._server_socket.listen(self._config.max_connections)
            self._server_socket.settimeout(self._config.socket_timeout)

            self._running = True
            self._logger.info(f"Server started on {self._config.host}:{self._config.port}")

            # Start accept thread
            self._accept_thread = threading.Thread(
                target=self._accept_loop, daemon=True
            )
            self._accept_thread.start()

        except Exception as e:
            self._logger.error(f"Failed to start server: {e}")
            self._cleanup()
            raise

    def stop(self):
        """Stop the TCP server gracefully"""
        if not self._running:
            return

        self._logger.info("Stopping server...")
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
                    self._logger.debug(f"Error closing client: {e}")
            self._clients.clear()

        # Wait for client handler threads to finish
        self._logger.info("Waiting for client threads to finish...")
        with self._client_threads_lock:
            threads_to_join = list(self._client_threads)

        for thread in threads_to_join:
            if thread.is_alive():
                thread.join(timeout=2.0)

        self._logger.info("All client threads stopped")

        # Close server socket
        self._cleanup()
        self._logger.info("Server stopped")

    def is_running(self) -> bool:
        """Check if server is running"""
        return self._running

    def should_shutdown(self) -> bool:
        """Return True if the server should stop."""
        return self._shutdown_flag

    def shutdown(self):
        """Trigger server shutdown."""
        self._shutdown_flag = True

    def get_client_count(self) -> int:
        """Get number of connected clients"""
        with self._clients_lock:
            return len(self._clients)

    def _update_client_state(self, client: socket.socket, state: ConnectionState):
        """Update client connection state"""
        with self._clients_lock:
            if client in self._client_info:
                self._client_info[client].state = state
                self._client_info[client].last_activity = datetime.now()

    def _record_bytes_received(self, client: socket.socket, num_bytes: int):
        """Record bytes received from client"""
        with self._clients_lock:
            if client in self._client_info:
                self._client_info[client].bytes_received += num_bytes
                self._client_info[client].last_activity = datetime.now()

    def _record_bytes_sent(self, client: socket.socket, num_bytes: int):
        """Record bytes sent to client"""
        with self._clients_lock:
            if client in self._client_info:
                self._client_info[client].bytes_sent += num_bytes
                self._client_info[client].last_activity = datetime.now()

    def _record_client_error(self, client: socket.socket):
        """Record client error"""
        with self._clients_lock:
            if client in self._client_info:
                self._client_info[client].error_count += 1
                self._client_info[client].state = ConnectionState.ERROR

    def get_client_info(self, client: socket.socket) -> Optional[ClientInfo]:
        """Get information about a client"""
        with self._clients_lock:
            return self._client_info.get(client)

    def _is_connection_error_fatal(self, error: Exception) -> Tuple[bool, str]:
        """
        Determine if a connection error is fatal (client disconnected).

        Args:
            error: Exception that occurred

        Returns:
            Tuple of (is_fatal, error_description)
        """
        # Check for specific error types
        if isinstance(error, socket.timeout):
            return False, "Connection idle (timeout)"

        if isinstance(error, ConnectionResetError):
            return True, "Connection reset by peer"

        if isinstance(error, BrokenPipeError):
            return True, "Broken pipe (client disconnected)"

        if isinstance(error, OSError):
            # Check errno for specific connection errors
            if hasattr(error, "errno"):
                if error.errno == errno.ECONNRESET:
                    return True, "Connection reset by peer"
                elif error.errno == errno.EPIPE:
                    return True, "Broken pipe"
                elif error.errno == errno.ECONNABORTED:
                    return True, "Connection aborted"
                elif error.errno == errno.ETIMEDOUT:
                    return False, "Connection idle (timeout)"

        # Unknown error - treat as potentially fatal
        return True, f"Unknown error: {type(error).__name__}"

    def broadcast_to_all_clients(self, data: bytes) -> int:
        """
        Send data to all connected clients.

        Args:
            data: Bytes to send

        Returns:
            Number of clients successfully sent to
        """
        # Get snapshot of clients without holding lock during send
        # This prevents deadlock if sendall() blocks
        with self._clients_lock:
            clients_snapshot = list(self._clients)

        disconnected = []
        success_count = 0

        # Send to all clients outside of lock
        for client in clients_snapshot:
            try:
                client.sendall(data)
                success_count += 1
            except Exception as e:
                self._logger.warning(f"Failed to send to client: {e}")
                disconnected.append(client)

        # Remove disconnected clients
        if disconnected:
            with self._clients_lock:
                for client in disconnected:
                    self._remove_client(client)

        return success_count

    def _accept_loop(self):
        """Accept incoming client connections (runs in separate thread)"""
        while self._running:
            try:
                if not self._server_socket:
                    break
                client, address = self._server_socket.accept()

                # Clean up completed threads before checking limit
                self._cleanup_completed_threads()

                # Check if we've reached max client threads
                with self._client_threads_lock:
                    active_threads = len(
                        [t for t in self._client_threads if t.is_alive()]
                    )

                if active_threads >= self._config.max_client_threads:
                    self._logger.warning(
                        f"Max client threads ({self._config.max_client_threads}) reached. "
                        f"Rejecting connection from {address}"
                    )
                    try:
                        client.close()
                    except:
                        pass
                    continue

                self._logger.debug(f"Client connected from {address}")

                # Add to client list and initialize client info
                now = datetime.now()
                with self._clients_lock:
                    self._clients.append(client)
                    self._client_info[client] = ClientInfo(
                        socket=client,
                        address=address,
                        state=ConnectionState.CONNECTED,
                        connected_at=now,
                        last_activity=now,
                    )

                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self._handle_client_wrapper,
                    args=(client, address),
                    daemon=True,
                )

                # Track the client thread
                with self._client_threads_lock:
                    self._client_threads.append(client_thread)

                client_thread.start()

            except socket.timeout:
                # Expected - allows us to check _running periodically
                continue
            except Exception as e:
                if self._running:
                    self._logger.error(f"Error accepting client: {e}")

    def _handle_client_wrapper(self, client: socket.socket, address: tuple):
        """
        Wrapper around subclass's handle_client_connection method.
        Ensures cleanup happens even if handler raises exception.
        """
        try:
            self.handle_client_connection(client, address)
        except Exception as e:
            self._logger.error(f"Error handling client {address}: {e}")
        finally:
            # Remove from client list
            with self._clients_lock:
                self._remove_client(client)

            # Close client socket
            try:
                client.close()
            except:
                pass

            self._logger.debug(f"Client disconnected from {address}")

    def _cleanup_completed_threads(self):
        """Remove completed threads from the thread list"""
        with self._client_threads_lock:
            self._client_threads = [t for t in self._client_threads if t.is_alive()]

    def _remove_client(self, client: socket.socket):
        """Remove a client from the tracked list (must hold _clients_lock)"""
        try:
            self._clients.remove(client)
        except ValueError:
            pass  # Already removed

        # Remove client info
        if client in self._client_info:
            del self._client_info[client]

    def _cleanup(self):
        """Clean up server socket"""
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception as e:
                self._logger.debug(f"Error closing server socket: {e}")
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
                self._logger.debug(f"Echo client error: {e}")

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
