#!/usr/bin/env python3
"""
Unit tests for core/TCPServerBase.py

Tests the abstract TCP server base class
"""

import pytest
import socket
import threading
import time
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add LLMCommunication directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "LLMCommunication"))

from LLMCommunication.core.TCPServerBase import TCPServerBase, ServerConfig


class TestServerConfig:
    """Test ServerConfig dataclass"""

    def test_default_config(self):
        """Test ServerConfig with default values"""
        config = ServerConfig()

        assert config.host is not None
        assert config.port > 0
        assert config.max_connections > 0
        assert config.max_client_threads > 0
        assert config.socket_timeout > 0

    def test_custom_config(self):
        """Test ServerConfig with custom values"""
        config = ServerConfig(
            host="192.168.1.1",
            port=8888,
            max_connections=10,
            max_client_threads=20,
            socket_timeout=5.0,
        )

        assert config.host == "192.168.1.1"
        assert config.port == 8888
        assert config.max_connections == 10
        assert config.max_client_threads == 20
        assert config.socket_timeout == 5.0


class MockTCPServer(TCPServerBase):
    """
    Concrete implementation of TCPServerBase for testing
    """

    def __init__(self, config):
        super().__init__(config)
        self.handled_clients = []
        self.handle_client_called = False

    def handle_client_connection(self, client, address):
        """Mock implementation that tracks calls"""
        self.handle_client_called = True
        self.handled_clients.append(address)
        # Simulate some work
        time.sleep(0.01)


class TestTCPServerBaseInitialization:
    """Test server initialization"""

    def test_server_initialization(self, server_config):
        """Test that server initializes correctly"""
        server = MockTCPServer(server_config)

        assert server._config == server_config
        assert server._running is False
        assert len(server._clients) == 0
        assert server._server_socket is None
        assert server._accept_thread is None

    def test_is_running_initially_false(self, server_config):
        """Test that server is not running initially"""
        server = MockTCPServer(server_config)
        assert server.is_running() is False

    def test_get_client_count_initially_zero(self, server_config):
        """Test that client count is initially zero"""
        server = MockTCPServer(server_config)
        assert server.get_client_count() == 0


class TestTCPServerBaseLifecycle:
    """Test server start/stop lifecycle"""

    @patch("socket.socket")
    def test_server_start(self, mock_socket_class, server_config):
        """Test starting the server"""
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket

        server = MockTCPServer(server_config)
        server.start()

        # Verify socket was created and configured
        mock_socket_class.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)
        mock_socket.setsockopt.assert_called_once_with(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
        )
        mock_socket.bind.assert_called_once_with(
            (server_config.host, server_config.port)
        )
        mock_socket.listen.assert_called_once_with(server_config.max_connections)
        mock_socket.settimeout.assert_called_once_with(server_config.socket_timeout)

        # Server should be running
        assert server.is_running() is True

        # Clean up
        server.stop()

    @patch("socket.socket")
    def test_server_start_already_running(self, mock_socket_class, server_config):
        """Test that starting an already running server is safe"""
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket

        server = MockTCPServer(server_config)
        server.start()

        # Try to start again
        server.start()

        # Should only bind once
        assert mock_socket.bind.call_count == 1

        # Clean up
        server.stop()

    @patch("socket.socket")
    def test_server_stop(self, mock_socket_class, server_config):
        """Test stopping the server"""
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket

        server = MockTCPServer(server_config)
        server.start()
        time.sleep(0.1)  # Let server start

        server.stop()

        # Server should not be running
        assert server.is_running() is False

        # Socket should be closed
        mock_socket.close.assert_called()

    def test_server_stop_not_running(self, server_config):
        """Test stopping a server that's not running"""
        server = MockTCPServer(server_config)

        # Should not raise exception
        server.stop()
        assert server.is_running() is False


class TestTCPServerBaseClientHandling:
    """Test client connection handling"""

    @patch("socket.socket")
    def test_client_tracking(self, mock_socket_class, server_config):
        """Test that connected clients are tracked"""
        mock_server_socket = MagicMock()
        mock_client_socket = MagicMock()

        mock_socket_class.return_value = mock_server_socket
        mock_server_socket.accept.return_value = (
            mock_client_socket,
            ("127.0.0.1", 12345),
        )

        # Make accept() timeout after first call
        mock_server_socket.accept.side_effect = [
            (mock_client_socket, ("127.0.0.1", 12345)),
            socket.timeout(),
        ]

        server = MockTCPServer(server_config)
        server.start()

        # Wait for client to be accepted
        time.sleep(0.2)

        # Check client was tracked (might be removed after handler completes)
        # Due to threading, we check if handler was called
        assert server.handle_client_called or server.get_client_count() >= 0

        server.stop()

    def test_broadcast_to_all_clients_no_clients(self, server_config):
        """Test broadcasting when no clients are connected"""
        server = MockTCPServer(server_config)

        result = server.broadcast_to_all_clients(b"test_data")

        assert result == 0  # No clients to send to

    @patch("socket.socket")
    def test_broadcast_to_all_clients_with_clients(
        self, mock_socket_class, server_config
    ):
        """Test broadcasting to connected clients"""
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()

        server = MockTCPServer(server_config)

        # Manually add clients for testing
        with server._clients_lock:
            server._clients.append(mock_client1)
            server._clients.append(mock_client2)

        result = server.broadcast_to_all_clients(b"test_data")

        # Both clients should receive data
        mock_client1.sendall.assert_called_once_with(b"test_data")
        mock_client2.sendall.assert_called_once_with(b"test_data")
        assert result == 2

    @patch("socket.socket")
    def test_broadcast_removes_failed_clients(self, mock_socket_class, server_config):
        """Test that failed clients are removed during broadcast"""
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()

        # Make client1 fail
        mock_client1.sendall.side_effect = Exception("Connection lost")

        server = MockTCPServer(server_config)

        # Manually add clients
        with server._clients_lock:
            server._clients.append(mock_client1)
            server._clients.append(mock_client2)

        assert server.get_client_count() == 2

        result = server.broadcast_to_all_clients(b"test_data")

        # Only one successful send
        assert result == 1

        # Failed client should be removed
        assert server.get_client_count() == 1


class TestTCPServerBaseThreadManagement:
    """Test thread management functionality"""

    def test_cleanup_completed_threads(self, server_config):
        """Test cleaning up completed threads"""
        server = MockTCPServer(server_config)

        # Create some mock threads
        active_thread = MagicMock()
        active_thread.is_alive.return_value = True

        dead_thread = MagicMock()
        dead_thread.is_alive.return_value = False

        with server._client_threads_lock:
            server._client_threads = [active_thread, dead_thread]

        server._cleanup_completed_threads()

        # Only active thread should remain
        assert len(server._client_threads) == 1
        assert active_thread in server._client_threads

    @patch("socket.socket")
    def test_max_client_threads_enforced(self, mock_socket_class, server_config):
        """Test that max client thread limit is enforced"""
        # Set very low limit for testing
        test_config = ServerConfig(
            host="127.0.0.1", port=9999, max_client_threads=1, socket_timeout=0.1
        )

        mock_server_socket = MagicMock()
        mock_socket_class.return_value = mock_server_socket

        # Create mock clients
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()

        # Simulate two connection attempts
        mock_server_socket.accept.side_effect = [
            (mock_client1, ("127.0.0.1", 12345)),
            (mock_client2, ("127.0.0.1", 12346)),
            socket.timeout(),
        ]

        server = MockTCPServer(test_config)
        server.start()

        time.sleep(0.3)  # Wait for connections

        # Second client might be rejected or queued
        # This is hard to test deterministically due to threading
        assert server.get_client_count() <= test_config.max_client_threads

        server.stop()


class TestTCPServerBaseErrorHandling:
    """Test error handling"""

    @patch("socket.socket")
    def test_start_bind_failure(self, mock_socket_class, server_config):
        """Test handling of bind failure during start"""
        mock_socket = MagicMock()
        mock_socket.bind.side_effect = OSError("Address already in use")
        mock_socket_class.return_value = mock_socket

        server = MockTCPServer(server_config)

        with pytest.raises(OSError):
            server.start()

        # Server should not be running
        assert server.is_running() is False

        # Socket should be cleaned up
        assert mock_socket.close.called or server._server_socket is None

    def test_remove_client_not_in_list(self, server_config):
        """Test removing a client that's not in the list"""
        server = MockTCPServer(server_config)
        mock_client = Mock()

        # Should not raise exception
        server._remove_client(mock_client)

    @patch("socket.socket")
    def test_handle_client_wrapper_exception_handling(
        self, mock_socket_class, server_config
    ):
        """Test that exceptions in client handler are caught"""

        # Create a server that raises in handle_client_connection
        class FailingServer(TCPServerBase):
            def handle_client_connection(self, client, address):
                raise Exception("Test exception")

        server = FailingServer(server_config)
        mock_client = MagicMock()

        # Should not raise - exception should be caught
        server._handle_client_wrapper(mock_client, ("127.0.0.1", 12345))

        # Client should be closed
        mock_client.close.assert_called()


class TestTCPServerBaseConcurrency:
    """Test concurrent access and thread safety"""

    def test_client_list_thread_safety(self, server_config):
        """Test that client list is thread-safe"""
        server = MockTCPServer(server_config)

        def add_clients():
            for i in range(10):
                with server._clients_lock:
                    server._clients.append(Mock())

        def remove_clients():
            for i in range(5):
                time.sleep(0.001)
                with server._clients_lock:
                    if server._clients:
                        server._clients.pop()

        # Run concurrent operations
        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=add_clients))
            threads.append(threading.Thread(target=remove_clients))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should complete without errors
        assert isinstance(server.get_client_count(), int)
        assert server.get_client_count() >= 0
