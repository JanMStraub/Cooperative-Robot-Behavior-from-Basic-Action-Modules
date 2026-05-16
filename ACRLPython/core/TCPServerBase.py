#!/usr/bin/env python3
"""Abstract base class for all Unity-facing TCP servers."""

import socket
import threading
import errno
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from enum import Enum
from datetime import datetime

try:
    from config.Servers import (
        DEFAULT_HOST,
        MAX_CONNECTIONS_BACKLOG,
        MAX_CLIENT_THREADS,
        SOCKET_ACCEPT_TIMEOUT,
        SERVER_HEARTBEAT_INTERVAL,
    )
    from core.LoggingSetup import setup_logging
except ImportError:
    from ..config.Servers import (
        DEFAULT_HOST,
        MAX_CONNECTIONS_BACKLOG,
        MAX_CLIENT_THREADS,
        SOCKET_ACCEPT_TIMEOUT,
        SERVER_HEARTBEAT_INTERVAL,
    )
    from ..core.LoggingSetup import setup_logging


class ConnectionState(Enum):
    """Connection state."""

    CONNECTED = "connected"
    IDLE = "idle"
    RECEIVING = "receiving"
    SENDING = "sending"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class ClientInfo:
    """Tracks per-client connection state and metrics."""

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
    """TCP server configuration."""

    host: str = DEFAULT_HOST
    port: int = 5000
    max_connections: int = MAX_CONNECTIONS_BACKLOG  # Max backlog for listen()
    max_client_threads: int = (
        MAX_CLIENT_THREADS  # Max concurrent client handler threads
    )
    socket_timeout: float = (
        SOCKET_ACCEPT_TIMEOUT  # Timeout for accept() to allow periodic shutdown checks
    )


class TCPServerBase(ABC):
    """Abstract base class for TCP servers. Subclasses implement handle_client_connection()."""

    # Heartbeat interval in seconds — how often the server logs its health status.
    HEARTBEAT_INTERVAL: float = SERVER_HEARTBEAT_INTERVAL

    def __init__(self, config: ServerConfig):
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
        self._heartbeat_thread: Optional[threading.Thread] = None

        # Connection monitoring counters (thread-safe: GIL protects int +=)
        self._total_connections: int = 0
        self._total_disconnections: int = 0
        self._start_time: Optional[datetime] = None

        # Last-logged connection snapshot — heartbeat only logs when these change
        self._last_heartbeat_snapshot: Optional[tuple] = None

        # Use class name so each server type gets its own logger.
        self._logger = setup_logging(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    @abstractmethod
    def handle_client_connection(self, client: socket.socket, address: tuple):
        """Process a client connection; called in a dedicated thread per client."""
        pass

    def start(self):
        if self._running:
            self._logger.warning(
                f"Server already running on {self._config.host}:{self._config.port}"
            )
            return

        try:
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.bind((self._config.host, self._config.port))
            self._server_socket.listen(self._config.max_connections)
            self._server_socket.settimeout(self._config.socket_timeout)

            self._running = True
            self._start_time = datetime.now()

            self._accept_thread = threading.Thread(
                target=self._accept_loop, daemon=True
            )
            self._accept_thread.start()

            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True,
                name=f"{self.__class__.__name__}_heartbeat",
            )
            self._heartbeat_thread.start()

        except Exception as e:
            self._logger.error(f"Failed to start server: {e}")
            self._cleanup()
            raise

    def stop(self):
        if not self._running:
            return

        self._logger.info(
            f"Stopping {self.__class__.__name__} (port {self._config.port})..."
        )
        self._running = False

        # Close the server socket first so accept() unblocks immediately
        # instead of waiting for the socket_timeout (up to 1s per server).
        self._cleanup()

        # Wait for threads to finish
        if self._accept_thread and self._accept_thread.is_alive():
            self._accept_thread.join(timeout=2.0)
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2.0)

        with self._clients_lock:
            for client in self._clients:
                try:
                    client.close()
                except Exception as e:
                    self._logger.debug(f"Error closing client: {e}")
            self._clients.clear()

        with self._client_threads_lock:
            threads_to_join = list(self._client_threads)

        for thread in threads_to_join:
            if thread.is_alive():
                thread.join(timeout=2.0)

        self._cleanup()

    def is_running(self) -> bool:
        return self._running

    def should_shutdown(self) -> bool:
        return self._shutdown_flag

    def shutdown(self):
        self._shutdown_flag = True

    def get_client_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    def _update_client_state(self, client: socket.socket, state: ConnectionState):
        with self._clients_lock:
            if client in self._client_info:
                self._client_info[client].state = state
                self._client_info[client].last_activity = datetime.now()

    def _record_bytes_received(self, client: socket.socket, num_bytes: int):
        with self._clients_lock:
            if client in self._client_info:
                self._client_info[client].bytes_received += num_bytes
                self._client_info[client].last_activity = datetime.now()

    def _record_bytes_sent(self, client: socket.socket, num_bytes: int):
        with self._clients_lock:
            if client in self._client_info:
                self._client_info[client].bytes_sent += num_bytes
                self._client_info[client].last_activity = datetime.now()

    def _record_client_error(self, client: socket.socket):
        with self._clients_lock:
            if client in self._client_info:
                self._client_info[client].error_count += 1
                self._client_info[client].state = ConnectionState.ERROR

    def get_client_info(self, client: socket.socket) -> Optional[ClientInfo]:
        with self._clients_lock:
            return self._client_info.get(client)

    def _recv_exactly(self, sock: socket.socket, num_bytes: int) -> Optional[bytes]:
        """Receive exactly num_bytes; returns None on close or error."""
        self._update_client_state(sock, ConnectionState.RECEIVING)
        chunks = []
        received = 0
        while received < num_bytes:
            try:
                chunk = sock.recv(num_bytes - received)
                if not chunk:
                    return None
                chunks.append(chunk)
                received += len(chunk)
                self._record_bytes_received(sock, len(chunk))
            except Exception:
                return None
        return b"".join(chunks)

    def _read_int(self, sock: socket.socket) -> Optional[int]:
        """Read a 4-byte little-endian unsigned integer; returns None on failure."""
        import struct

        data = self._recv_exactly(sock, 4)
        if data:
            return struct.unpack("<I", data)[0]  # Little-endian unsigned int
        return None

    def _receive_length_prefixed_string(self, sock: socket.socket) -> Optional[bytes]:
        """Receive [length:4][data:N]; returns undecoded bytes or None on failure."""
        import struct

        len_bytes = self._recv_exactly(sock, 4)
        if not len_bytes:
            return None

        str_len = struct.unpack("<I", len_bytes)[0]

        if str_len == 0:
            return b""

        str_bytes = self._recv_exactly(sock, str_len)
        return str_bytes

    def _receive_complete_rag_query(
        self, sock: socket.socket, header_bytes: bytes
    ) -> Optional[bytes]:
        """Receive a complete RAG query message; returns full bytes or None."""
        import struct

        data = bytearray(header_bytes)

        query_bytes = self._receive_length_prefixed_string(sock)
        if query_bytes is None:
            return None

        data.extend(struct.pack("<I", len(query_bytes)))
        data.extend(query_bytes)

        top_k_bytes = self._recv_exactly(sock, 4)
        if not top_k_bytes:
            return None
        data.extend(top_k_bytes)

        filters_bytes = self._receive_length_prefixed_string(sock)
        if filters_bytes is None:
            return None

        data.extend(struct.pack("<I", len(filters_bytes)))
        data.extend(filters_bytes)

        return bytes(data)

    def _receive_complete_status_query(
        self, sock: socket.socket, header_bytes: bytes
    ) -> Optional[bytes]:
        """Receive a complete status query message; returns full bytes or None."""
        import struct

        data = bytearray(header_bytes)

        robot_id_bytes = self._receive_length_prefixed_string(sock)
        if robot_id_bytes is None:
            return None

        data.extend(struct.pack("<I", len(robot_id_bytes)))
        data.extend(robot_id_bytes)

        detailed_byte = self._recv_exactly(sock, 1)
        if not detailed_byte:
            return None
        data.extend(detailed_byte)

        return bytes(data)

    def _receive_complete_autort_command(
        self, sock: socket.socket, header_bytes: bytes
    ) -> Optional[bytes]:
        """Receive a complete AutoRT command message; returns full bytes or None."""
        import struct

        data = bytearray(header_bytes)

        cmd_type_bytes = self._receive_length_prefixed_string(sock)
        if cmd_type_bytes is None:
            return None

        data.extend(struct.pack("<I", len(cmd_type_bytes)))
        data.extend(cmd_type_bytes)

        params_bytes = self._receive_length_prefixed_string(sock)
        if params_bytes is None:
            return None

        data.extend(struct.pack("<I", len(params_bytes)))
        data.extend(params_bytes)

        return bytes(data)

    def _is_connection_error_fatal(self, error: Exception) -> Tuple[bool, str]:
        """Returns (is_fatal, description) for a connection error."""
        if isinstance(error, socket.timeout):
            return False, "Connection idle (timeout)"

        if isinstance(error, ConnectionResetError):
            return True, "Connection reset by peer"

        if isinstance(error, BrokenPipeError):
            return True, "Broken pipe (client disconnected)"

        if isinstance(error, OSError):
            if hasattr(error, "errno"):
                if error.errno == errno.ECONNRESET:
                    return True, "Connection reset by peer"
                elif error.errno == errno.EPIPE:
                    return True, "Broken pipe"
                elif error.errno == errno.ECONNABORTED:
                    return True, "Connection aborted"
                elif error.errno == errno.ETIMEDOUT:
                    return False, "Connection idle (timeout)"

        return True, f"Unknown error: {type(error).__name__}"

    def send_to_client(self, client: socket.socket, data: bytes) -> bool:
        """Send data to a client; returns True on success."""
        try:
            client.sendall(data)
            self._record_bytes_sent(client, len(data))
            return True
        except Exception as e:
            self._logger.warning(f"Failed to send to client: {e}")
            self._record_client_error(client)
            with self._clients_lock:
                self._remove_client(client)
            return False

    def broadcast_to_all_clients(self, data: bytes) -> int:
        """Send data to all connected clients; returns success count."""
        # Snapshot without lock to prevent deadlock if sendall() blocks.
        with self._clients_lock:
            clients_snapshot = list(self._clients)

        disconnected = []
        success_count = 0

        for client in clients_snapshot:
            try:
                client.sendall(data)
                self._record_bytes_sent(client, len(data))
                success_count += 1
            except Exception as e:
                self._logger.warning(f"Failed to send to client: {e}")
                disconnected.append(client)

        if disconnected:
            with self._clients_lock:
                for client in disconnected:
                    self._remove_client(client)

        return success_count

    def _accept_loop(self):
        while self._running:
            try:
                if not self._server_socket:
                    break
                client, address = self._server_socket.accept()

                self._cleanup_completed_threads()

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
                    except (OSError, ConnectionError) as e:
                        self._logger.debug(f"Error closing rejected client socket: {e}")
                    continue

                self._logger.debug(f"Client connected from {address}")

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

                client_thread = threading.Thread(
                    target=self._handle_client_wrapper,
                    args=(client, address),
                    daemon=True,
                )

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
        """Wraps handle_client_connection to guarantee cleanup and update counters."""
        self._total_connections += 1
        self._logger.info(
            f"Client connected from {address} "
            f"(total connections: {self._total_connections})"
        )
        try:
            self.handle_client_connection(client, address)
        except Exception as e:
            self._logger.error(f"Error handling client {address}: {e}")
        finally:
            self._total_disconnections += 1
            with self._clients_lock:
                self._remove_client(client)

            try:
                client.close()
            except (OSError, ConnectionError) as e:
                self._logger.debug(f"Error closing client socket: {e}")

            self._logger.info(
                f"Client disconnected from {address} "
                f"(active: {self.get_client_count()}, "
                f"total disconnections: {self._total_disconnections})"
            )

    def _heartbeat_loop(self):
        """
        Periodic health logging thread.

        Logs server uptime, active client count, and cumulative connection
        metrics every HEARTBEAT_INTERVAL seconds. Exits when the server stops.
        """
        import time

        while self._running:
            # Sleep in small increments so we respond to stop() promptly.
            for _ in range(int(self.HEARTBEAT_INTERVAL / 1.0)):
                if not self._running:
                    break
                time.sleep(1.0)

            if not self._running:
                break

            stats = self.get_stats()
            snapshot = (
                stats["active_clients"],
                stats["total_connections"],
                stats["total_disconnections"],
            )
            if snapshot != self._last_heartbeat_snapshot:
                self._last_heartbeat_snapshot = snapshot
                self._logger.debug(
                    f"[HEARTBEAT] {self.__class__.__name__} on :{self._config.port} | "
                    f"uptime={stats['uptime_seconds']:.0f}s | "
                    f"active_clients={stats['active_clients']} | "
                    f"total_connections={stats['total_connections']} | "
                    f"total_disconnections={stats['total_disconnections']}"
                )

    def get_stats(self) -> dict:
        """
        Return a snapshot of server health and connection metrics.

        Returns:
            Dict with keys: uptime_seconds, active_clients, total_connections,
            total_disconnections, port.
        """
        uptime = 0.0
        if self._start_time is not None:
            uptime = (datetime.now() - self._start_time).total_seconds()

        return {
            "port": self._config.port,
            "uptime_seconds": uptime,
            "active_clients": self.get_client_count(),
            "total_connections": self._total_connections,
            "total_disconnections": self._total_disconnections,
        }

    def _cleanup_completed_threads(self):
        with self._client_threads_lock:
            self._client_threads = [t for t in self._client_threads if t.is_alive()]

    def _remove_client(self, client: socket.socket):
        """Remove client from tracked list. Caller must hold _clients_lock."""
        try:
            self._clients.remove(client)
        except ValueError:
            pass  # Already removed
        if client in self._client_info:
            del self._client_info[client]

    def _cleanup(self):
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception as e:
                self._logger.debug(f"Error closing server socket: {e}")
            self._server_socket = None


if __name__ == "__main__":
    # Example usage and testing
    class EchoServer(TCPServerBase):
        def handle_client_connection(self, client: socket.socket, address: tuple):
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
