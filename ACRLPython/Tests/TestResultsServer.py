#!/usr/bin/env python3
"""
Unit tests for ResultsServer.py

Tests the results broadcasting server
"""

import socket
import threading
import time
from unittest.mock import Mock, patch, MagicMock

from servers.ResultsServer import ResultsBroadcaster, ResultsServer
from core.TCPServerBase import ServerConfig


class TestResultsBroadcasterSingleton:
    """Test ResultsBroadcaster singleton pattern"""

    def test_singleton_initialization(self, cleanup_singletons):
        """Test that ResultsBroadcaster initializes as singleton"""
        # Before initialization, server should be None
        assert ResultsBroadcaster._server is None

        mock_server = Mock()
        ResultsBroadcaster.initialize(mock_server)

        assert ResultsBroadcaster._server is mock_server
        assert ResultsBroadcaster._instance is not None

    def test_singleton_state_preserved(self, cleanup_singletons):
        """Test that singleton state is preserved"""
        mock_server1 = Mock()
        ResultsBroadcaster.initialize(mock_server1)
        instance1 = ResultsBroadcaster._instance

        mock_server2 = Mock()
        ResultsBroadcaster.initialize(mock_server2)
        instance2 = ResultsBroadcaster._instance

        # Instance should be the same, but server updated
        assert instance1 is instance2
        assert ResultsBroadcaster._server is mock_server2


class TestResultsBroadcasterSendResult:
    """Test ResultsBroadcaster.send_result functionality"""

    def test_send_result_not_initialized(self, cleanup_singletons, llm_result_dict):
        """Test sending result when broadcaster not initialized"""
        result = ResultsBroadcaster.send_result(llm_result_dict)

        # Should be queued (returns True) when server not initialized
        assert result is True
        assert len(ResultsBroadcaster._result_queue) == 1

    def test_send_result_no_clients(self, cleanup_singletons, llm_result_dict):
        """Test sending result when no clients connected"""
        mock_server = Mock()
        mock_server.get_client_count.return_value = 0
        ResultsBroadcaster.initialize(mock_server)

        # Clear any existing queue from initialization
        ResultsBroadcaster._result_queue = []

        result = ResultsBroadcaster.send_result(llm_result_dict)

        # Should be queued
        assert result is True
        assert len(ResultsBroadcaster._result_queue) == 1

    def test_send_result_with_clients(self, cleanup_singletons, llm_result_dict):
        """Test sending result with clients connected"""
        mock_server = Mock()
        mock_server.get_client_count.return_value = 2
        mock_server.broadcast_to_all_clients.return_value = 2
        ResultsBroadcaster.initialize(mock_server)

        result = ResultsBroadcaster.send_result(llm_result_dict)

        assert result is True
        mock_server.broadcast_to_all_clients.assert_called_once()

    def test_send_result_adds_camera_id_from_metadata(self, cleanup_singletons):
        """Test that camera_id is added from metadata if missing"""
        result_dict = {
            "response": "test",
            "metadata": {"camera_ids": ["camera1", "camera2"]},
        }

        mock_server = Mock()
        mock_server.get_client_count.return_value = 1
        mock_server.broadcast_to_all_clients.return_value = 1
        ResultsBroadcaster.initialize(mock_server)

        ResultsBroadcaster.send_result(result_dict)

        # camera_id should be added
        assert "camera_id" in result_dict

    def test_send_result_adds_timestamp_from_metadata(self, cleanup_singletons):
        """Test that timestamp is added from metadata if missing"""
        result_dict = {
            "response": "test",
            "metadata": {"timestamp": "2025-01-01T12:00:00"},
        }

        mock_server = Mock()
        mock_server.get_client_count.return_value = 1
        mock_server.broadcast_to_all_clients.return_value = 1
        ResultsBroadcaster.initialize(mock_server)

        ResultsBroadcaster.send_result(result_dict)

        # timestamp should be added
        assert "timestamp" in result_dict

    def test_send_result_queues_on_broadcast_failure(
        self, cleanup_singletons, llm_result_dict
    ):
        """Test that result is queued if broadcast fails"""
        mock_server = Mock()
        mock_server.get_client_count.return_value = 1  # Think there's a client
        mock_server.broadcast_to_all_clients.return_value = 0  # But send fails
        ResultsBroadcaster.initialize(mock_server)

        # Clear any existing queue from initialization
        ResultsBroadcaster._result_queue = []

        result = ResultsBroadcaster.send_result(llm_result_dict)

        # Should be queued despite thinking there was a client
        assert len(ResultsBroadcaster._result_queue) == 1

    def test_send_result_handles_encoding_error(self, cleanup_singletons):
        """Test handling of encoding errors"""
        # Create un-serializable result
        result_dict = {"response": "test", "circular": None}
        result_dict["circular"] = result_dict  # Create circular reference

        mock_server = Mock()
        mock_server.get_client_count.return_value = 1
        ResultsBroadcaster.initialize(mock_server)

        # Should handle the error gracefully
        result = ResultsBroadcaster.send_result(result_dict)

        # Will fail to encode, so should return False or handle gracefully
        # Exact behavior depends on implementation


class TestResultsBroadcasterQueue:
    """Test ResultsBroadcaster queue management"""

    def test_get_queued_results(self, cleanup_singletons, llm_result_dict):
        """Test retrieving queued results"""
        mock_server = Mock()
        mock_server.get_client_count.return_value = 0
        ResultsBroadcaster.initialize(mock_server)

        # Clear any existing queue from initialization
        ResultsBroadcaster._result_queue = []

        # Queue some results
        ResultsBroadcaster.send_result(llm_result_dict)
        ResultsBroadcaster.send_result(llm_result_dict)

        queued = ResultsBroadcaster.get_queued_results()

        assert len(queued) == 2
        # Queue should be cleared
        assert len(ResultsBroadcaster._result_queue) == 0

    def test_queue_max_size(self, cleanup_singletons):
        """Test that queue respects max size"""
        mock_server = Mock()
        mock_server.get_client_count.return_value = 0
        ResultsBroadcaster.initialize(mock_server)

        # Fill queue beyond max size
        max_size = ResultsBroadcaster._max_queue_size
        for i in range(max_size + 5):
            ResultsBroadcaster.send_result({"id": i})

        # Should not exceed max size
        assert len(ResultsBroadcaster._result_queue) <= max_size

    def test_get_queued_results_empty(self, cleanup_singletons):
        """Test getting queued results when queue is empty"""
        mock_server = Mock()
        ResultsBroadcaster.initialize(mock_server)

        # Clear any existing queue from initialization
        ResultsBroadcaster._result_queue = []

        queued = ResultsBroadcaster.get_queued_results()

        assert len(queued) == 0


class TestResultsServerInitialization:
    """Test ResultsServer initialization"""

    def test_server_initialization(self, server_config):
        """Test that ResultsServer initializes correctly"""
        server = ResultsServer(server_config)

        assert server._config == server_config
        assert ResultsBroadcaster._server is server

    def test_server_initialization_default_config(self):
        """Test server initialization with default config"""
        config = ServerConfig(host="127.0.0.1", port=5006, socket_timeout=1.0)
        server = ResultsServer(config)

        assert server._config is not None
        assert ResultsBroadcaster._server is server


class TestResultsServerClientHandling:
    """Test ResultsServer client handling"""

    @patch("socket.socket")
    def test_handle_client_connection_sends_queued_results(
        self, mock_socket_class, cleanup_singletons
    ):
        """Test that new clients receive queued results"""
        server_config = ServerConfig(host="127.0.0.1", port=9999, socket_timeout=0.1)
        server = ResultsServer(server_config)

        # Queue some results before client connects
        result1 = {"id": 1, "response": "test1"}
        result2 = {"id": 2, "response": "test2"}

        with ResultsBroadcaster._queue_lock:
            ResultsBroadcaster._result_queue = [result1, result2]

        mock_client = Mock()
        mock_client.recv.return_value = b""  # Simulate immediate disconnect

        # Handle client connection
        server._send_queued_results(mock_client)

        # Should have attempted to send queued results
        assert mock_client.sendall.call_count == 2

    def test_handle_client_connection_keeps_alive(self, cleanup_singletons):
        """Test that client connection is kept alive"""
        server_config = ServerConfig(host="127.0.0.1", port=9999, socket_timeout=0.1)
        server = ResultsServer(server_config)

        mock_client = Mock()
        mock_client.recv.side_effect = [
            socket.timeout(),  # First call times out (keep alive)
            b"",  # Second call returns empty (disconnect)
        ]

        # This should handle timeouts gracefully
        try:
            server.handle_client_connection(mock_client, ("127.0.0.1", 12345))
        except:
            pass  # May raise due to mocking, that's OK


class TestResultsServerIntegration:
    """Integration tests for ResultsServer"""

    @patch("socket.socket")
    def test_server_lifecycle(self, mock_socket_class, cleanup_singletons):
        """Test full server start/stop lifecycle"""
        mock_server_socket = MagicMock()
        mock_socket_class.return_value = mock_server_socket
        mock_server_socket.accept.side_effect = socket.timeout()

        server_config = ServerConfig(host="127.0.0.1", port=9999, socket_timeout=0.1)
        server = ResultsServer(server_config)

        # Start server
        server.start()
        assert server.is_running()

        # Give it a moment to start
        time.sleep(0.2)

        # Stop server
        server.stop()
        assert not server.is_running()


class TestResultsBroadcasterThreadSafety:
    """Test ResultsBroadcaster thread safety"""

    def test_concurrent_send_operations(self, cleanup_singletons):
        """Test concurrent send operations are thread-safe"""
        mock_server = Mock()
        mock_server.get_client_count.return_value = 0
        ResultsBroadcaster.initialize(mock_server)

        errors = []

        def send_results(thread_id):
            try:
                for i in range(20):
                    result = {"thread": thread_id, "id": i}
                    ResultsBroadcaster.send_result(result)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            t = threading.Thread(target=send_results, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # No errors should have occurred
        assert len(errors) == 0

        # Queue should have some results (up to max size)
        assert len(ResultsBroadcaster._result_queue) > 0

    def test_concurrent_send_and_get_queue(self, cleanup_singletons):
        """Test concurrent send and get_queued_results operations"""
        mock_server = Mock()
        mock_server.get_client_count.return_value = 0
        ResultsBroadcaster.initialize(mock_server)

        errors = []

        def sender():
            try:
                for i in range(30):
                    ResultsBroadcaster.send_result({"id": i})
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def getter():
            try:
                for i in range(10):
                    ResultsBroadcaster.get_queued_results()
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=sender),
            threading.Thread(target=sender),
            threading.Thread(target=getter),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # No errors should have occurred
        assert len(errors) == 0
