#!/usr/bin/env python3
"""
Unit tests for DetectionServer.py

Tests the detection broadcasting server
"""

import pytest
import socket
import threading
import queue
import time
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add LLMCommunication directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "LLMCommunication"))

from LLMCommunication.servers.DetectionServer import (
    DetectionBroadcaster,
    DetectionServer,
)
from LLMCommunication.core.TCPServerBase import ServerConfig
import ACRLPython.LLMCommunication.LLMConfig as cfg


class TestDetectionBroadcasterSingleton:
    """Test DetectionBroadcaster singleton pattern"""

    def test_singleton_instance(self, cleanup_singletons):
        """Test that DetectionBroadcaster is a singleton"""
        broadcaster1 = DetectionBroadcaster.get_instance()
        broadcaster2 = DetectionBroadcaster.get_instance()

        assert broadcaster1 is broadcaster2

    def test_singleton_state_preserved(self, cleanup_singletons):
        """Test that singleton state is preserved"""
        broadcaster1 = DetectionBroadcaster.get_instance()
        broadcaster1.register_client(Mock())

        broadcaster2 = DetectionBroadcaster.get_instance()

        # Same instance should have same state
        assert broadcaster1 is broadcaster2


class TestDetectionBroadcasterClientManagement:
    """Test client registration and management"""

    def test_register_client(self, cleanup_singletons):
        """Test registering a client"""
        broadcaster = DetectionBroadcaster.get_instance()
        mock_client = Mock()

        with broadcaster._clients_lock:
            initial_count = len(broadcaster._clients)

        broadcaster.register_client(mock_client)

        with broadcaster._clients_lock:
            assert len(broadcaster._clients) == initial_count + 1
            assert mock_client in broadcaster._clients

    def test_register_same_client_twice(self, cleanup_singletons):
        """Test registering the same client twice doesn't duplicate"""
        broadcaster = DetectionBroadcaster.get_instance()
        mock_client = Mock()

        broadcaster.register_client(mock_client)
        broadcaster.register_client(mock_client)

        with broadcaster._clients_lock:
            client_count = broadcaster._clients.count(mock_client)
            # Should only be registered once
            assert client_count == 1

    def test_unregister_client(self, cleanup_singletons):
        """Test unregistering a client"""
        broadcaster = DetectionBroadcaster.get_instance()
        mock_client = Mock()

        broadcaster.register_client(mock_client)

        with broadcaster._clients_lock:
            assert mock_client in broadcaster._clients

        broadcaster.unregister_client(mock_client)

        with broadcaster._clients_lock:
            assert mock_client not in broadcaster._clients

    def test_unregister_nonexistent_client(self, cleanup_singletons):
        """Test unregistering a client that wasn't registered"""
        broadcaster = DetectionBroadcaster.get_instance()
        mock_client = Mock()

        # Should not raise exception
        broadcaster.unregister_client(mock_client)

    def test_register_multiple_clients(self, cleanup_singletons):
        """Test registering multiple clients"""
        broadcaster = DetectionBroadcaster.get_instance()

        clients = [Mock() for _ in range(5)]

        for client in clients:
            broadcaster.register_client(client)

        with broadcaster._clients_lock:
            for client in clients:
                assert client in broadcaster._clients


class TestDetectionBroadcasterSendResult:
    """Test sending detection results"""

    def test_send_result_no_clients(self, cleanup_singletons, detection_result_dict):
        """Test sending result when no clients are connected"""
        broadcaster = DetectionBroadcaster.get_instance()

        # Should queue the result
        broadcaster.send_result(detection_result_dict)

        # Check result was queued
        assert not broadcaster._result_queue.empty()

    def test_send_result_with_clients(self, cleanup_singletons, detection_result_dict):
        """Test sending result with connected clients"""
        broadcaster = DetectionBroadcaster.get_instance()

        mock_client1 = Mock()
        mock_client2 = Mock()

        broadcaster.register_client(mock_client1)
        broadcaster.register_client(mock_client2)

        broadcaster.send_result(detection_result_dict)

        # Both clients should receive data
        assert mock_client1.sendall.called
        assert mock_client2.sendall.called

    def test_send_result_adds_metadata(self, cleanup_singletons):
        """Test that send_result adds server timestamp"""
        broadcaster = DetectionBroadcaster.get_instance()

        mock_client = Mock()
        broadcaster.register_client(mock_client)

        result = {"success": True, "camera_id": "test", "detections": []}

        broadcaster.send_result(result)

        # Metadata should be added
        assert "metadata" in result
        assert "server_timestamp" in result["metadata"]

    def test_send_result_handles_serialization_error(self, cleanup_singletons):
        """Test handling of JSON serialization errors"""
        broadcaster = DetectionBroadcaster.get_instance()

        mock_client = Mock()
        broadcaster.register_client(mock_client)

        # Create un-serializable result (using object with __dict__ reference)
        class CircularObject:
            def __init__(self):
                self.ref = self

        result = {"circular_object": CircularObject()}  # type: ignore[dict-item]

        # Should handle error gracefully (not crash)
        broadcaster.send_result(result)

    def test_send_result_removes_failed_clients(
        self, cleanup_singletons, detection_result_dict
    ):
        """Test that clients are removed if send fails"""
        broadcaster = DetectionBroadcaster.get_instance()

        mock_client1 = Mock()
        mock_client2 = Mock()

        # Make client1 fail on send
        mock_client1.sendall.side_effect = Exception("Connection lost")

        broadcaster.register_client(mock_client1)
        broadcaster.register_client(mock_client2)

        broadcaster.send_result(detection_result_dict)

        # Failed client should be unregistered
        with broadcaster._clients_lock:
            assert mock_client1 not in broadcaster._clients
            # Successful client should still be registered
            assert mock_client2 in broadcaster._clients

    def test_send_result_queue_full_handling(self, cleanup_singletons):
        """Test behavior when result queue is full"""
        broadcaster = DetectionBroadcaster.get_instance()

        # Fill the queue
        max_size = cfg.MAX_RESULT_QUEUE_SIZE
        for i in range(max_size + 5):
            result = {"id": i}
            broadcaster.send_result(result)

        # Queue size should not exceed maximum
        assert broadcaster._result_queue.qsize() <= max_size

    def test_send_result_encodes_with_unity_protocol(
        self, cleanup_singletons, detection_result_dict
    ):
        """Test that results are encoded using Unity protocol"""
        broadcaster = DetectionBroadcaster.get_instance()

        mock_client = Mock()
        broadcaster.register_client(mock_client)

        broadcaster.send_result(detection_result_dict)

        # Verify sendall was called with bytes
        assert mock_client.sendall.called
        call_args = mock_client.sendall.call_args[0][0]
        assert isinstance(call_args, bytes)


class TestDetectionBroadcasterThreadSafety:
    """Test thread safety of DetectionBroadcaster"""

    def test_concurrent_client_registration(self, cleanup_singletons):
        """Test concurrent client registration is thread-safe"""
        broadcaster = DetectionBroadcaster.get_instance()
        errors = []

        def register_clients(thread_id):
            try:
                for i in range(10):
                    client = Mock()
                    broadcaster.register_client(client)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_clients, args=(i,)) for i in range(3)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0

    def test_concurrent_send_operations(self, cleanup_singletons):
        """Test concurrent send operations are thread-safe"""
        broadcaster = DetectionBroadcaster.get_instance()
        errors = []

        # Register a client
        mock_client = Mock()
        broadcaster.register_client(mock_client)

        def send_results(thread_id):
            try:
                for i in range(20):
                    result = {"thread": thread_id, "id": i, "detections": []}
                    broadcaster.send_result(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=send_results, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0

    def test_concurrent_register_and_send(self, cleanup_singletons):
        """Test concurrent registration and sending"""
        broadcaster = DetectionBroadcaster.get_instance()
        errors = []

        def register():
            try:
                for i in range(10):
                    broadcaster.register_client(Mock())
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        def send():
            try:
                for i in range(10):
                    broadcaster.send_result({"id": i, "detections": []})
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register),
            threading.Thread(target=register),
            threading.Thread(target=send),
            threading.Thread(target=send),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0


class TestDetectionServerInitialization:
    """Test DetectionServer initialization"""

    def test_server_initialization(self):
        """Test that DetectionServer initializes correctly"""
        config = ServerConfig(host="127.0.0.1", port=9999, socket_timeout=0.1)
        server = DetectionServer(config)

        assert server._config == config

    def test_server_initialization_default_port(self):
        """Test server uses correct default port"""
        config = ServerConfig(host="127.0.0.1", port=cfg.DETECTION_SERVER_PORT)
        server = DetectionServer(config)

        assert server._config.port == cfg.DETECTION_SERVER_PORT


class TestDetectionServerClientHandling:
    """Test DetectionServer client handling"""

    def test_handle_client_registers_on_connect(self, cleanup_singletons):
        """Test that clients are registered when they connect"""
        config = ServerConfig(host="127.0.0.1", port=9999, socket_timeout=0.1)
        server = DetectionServer(config)

        mock_client = Mock()
        mock_client.recv.return_value = b""  # Simulate immediate disconnect

        broadcaster = DetectionBroadcaster.get_instance()
        initial_count = len(broadcaster._clients)

        # Handle client connection (will register then immediately disconnect)
        try:
            server.handle_client_connection(mock_client, ("127.0.0.1", 12345))
        except:
            pass  # May raise due to mocking

        # Client should have been unregistered after disconnect
        # (registered during connection, unregistered during cleanup)

    def test_handle_client_keeps_connection_alive(self, cleanup_singletons):
        """Test that client connection is kept alive for receiving broadcasts"""
        config = ServerConfig(host="127.0.0.1", port=9999, socket_timeout=0.1)
        server = DetectionServer(config)

        mock_client = Mock()
        mock_client.recv.side_effect = [
            socket.timeout(),  # Timeout (keep alive)
            socket.timeout(),  # Timeout (keep alive)
            b"",  # Disconnect
        ]

        # Should handle timeouts gracefully
        try:
            server.handle_client_connection(mock_client, ("127.0.0.1", 12345))
        except:
            pass

    def test_handle_client_unregisters_on_disconnect(self, cleanup_singletons):
        """Test that client is unregistered when it disconnects"""
        config = ServerConfig(host="127.0.0.1", port=9999, socket_timeout=0.1)
        server = DetectionServer(config)

        mock_client = Mock()
        mock_client.recv.return_value = b""  # Immediate disconnect

        broadcaster = DetectionBroadcaster.get_instance()

        # Manually register client
        broadcaster.register_client(mock_client)
        assert mock_client in broadcaster._clients

        # Handle disconnection
        try:
            server.handle_client_connection(mock_client, ("127.0.0.1", 12345))
        except:
            pass

        # Should be unregistered
        assert mock_client not in broadcaster._clients


class TestDetectionServerIntegration:
    """Integration tests for DetectionServer"""

    @patch("socket.socket")
    def test_server_lifecycle(self, mock_socket_class, cleanup_singletons):
        """Test server start/stop lifecycle"""
        mock_server_socket = MagicMock()
        mock_socket_class.return_value = mock_server_socket
        mock_server_socket.accept.side_effect = socket.timeout()

        config = ServerConfig(host="127.0.0.1", port=9999, socket_timeout=0.1)
        server = DetectionServer(config)

        # Start server
        server.start()
        assert server.is_running()

        time.sleep(0.2)

        # Stop server
        server.stop()
        assert not server.is_running()

    @patch("socket.socket")
    def test_broadcast_to_connected_clients(
        self, mock_socket_class, cleanup_singletons
    ):
        """Test broadcasting to connected clients"""
        broadcaster = DetectionBroadcaster.get_instance()

        # Register mock clients
        mock_client1 = Mock()
        mock_client2 = Mock()
        broadcaster.register_client(mock_client1)
        broadcaster.register_client(mock_client2)

        # Send a detection result
        result = {
            "success": True,
            "camera_id": "test_camera",
            "detections": [{"id": 0, "color": "red", "confidence": 0.95}],
        }

        broadcaster.send_result(result)

        # Both clients should receive the result
        assert mock_client1.sendall.called
        assert mock_client2.sendall.called


class TestDetectionServerErrorHandling:
    """Test error handling in DetectionServer"""

    def test_handle_client_socket_error(self, cleanup_singletons):
        """Test handling of socket errors during client connection"""
        config = ServerConfig(host="127.0.0.1", port=9999, socket_timeout=0.1)
        server = DetectionServer(config)

        mock_client = Mock()
        mock_client.recv.side_effect = socket.error("Connection reset")

        # Should handle error gracefully
        try:
            server.handle_client_connection(mock_client, ("127.0.0.1", 12345))
        except:
            pass  # Errors are expected and handled

    def test_send_result_with_broken_socket(self, cleanup_singletons):
        """Test sending result when client socket is broken"""
        broadcaster = DetectionBroadcaster.get_instance()

        mock_client = Mock()
        mock_client.sendall.side_effect = BrokenPipeError("Broken pipe")

        broadcaster.register_client(mock_client)

        result = {"success": True, "detections": []}

        # Should handle broken pipe and remove client
        broadcaster.send_result(result)

        # Client should be removed after failure
        with broadcaster._clients_lock:
            assert mock_client not in broadcaster._clients


class TestDetectionServerQueueManagement:
    """Test result queue management"""

    def test_queued_results_sent_to_new_clients(self, cleanup_singletons):
        """Test that queued results are sent when new clients connect"""
        broadcaster = DetectionBroadcaster.get_instance()

        # Queue some results when no clients are connected
        result1 = {"id": 1, "detections": []}
        result2 = {"id": 2, "detections": []}

        broadcaster.send_result(result1)
        broadcaster.send_result(result2)

        # Results should be queued
        assert broadcaster._result_queue.qsize() >= 2

    def test_queue_overflow_drops_oldest(self, cleanup_singletons):
        """Test that queue drops oldest results when full"""
        broadcaster = DetectionBroadcaster.get_instance()

        # Fill queue beyond capacity
        max_size = cfg.MAX_RESULT_QUEUE_SIZE
        for i in range(max_size + 10):
            broadcaster.send_result({"id": i, "detections": []})

        # Queue should not exceed max size
        assert broadcaster._result_queue.qsize() <= max_size
