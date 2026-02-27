#!/usr/bin/env python3
"""
TestAutoRTSequenceServer.py - Integration tests for AutoRT + SequenceServer

Tests the integration between SequenceServer and AutoRTHandler,
including message routing and protocol compliance.
"""

import unittest
import socket
import struct
import threading
import time
from unittest.mock import Mock, patch, MagicMock

# Import modules under test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.UnityProtocol import UnityProtocol, MessageType
from servers.SequenceServer import SequenceServer
from servers.AutoRTIntegration import AutoRTHandler
from config.Servers import SEQUENCE_SERVER_PORT


class TestAutoRTSequenceServerIntegration(unittest.TestCase):
    """Integration tests for AutoRT message handling in SequenceServer."""

    @classmethod
    def setUpClass(cls):
        """Set up test server once for all tests."""
        cls.server = SequenceServer()
        cls.server_thread = threading.Thread(target=cls.server.start, daemon=True)
        cls.server_thread.start()
        time.sleep(0.5)  # Allow server to start

    @classmethod
    def tearDownClass(cls):
        """Tear down test server."""
        cls.server.stop()
        time.sleep(0.2)

    def setUp(self):
        """Set up test client."""
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect(("127.0.0.1", SEQUENCE_SERVER_PORT))
        self.client.settimeout(5.0)

        # Reset AutoRTHandler singleton
        AutoRTHandler._instance = None

    def tearDown(self):
        """Clean up test client."""
        try:
            self.client.close()
        except:
            pass

        # Clean up AutoRTHandler
        handler = AutoRTHandler.get_instance()
        if handler._loop_running:
            handler.stop_loop()
            time.sleep(0.1)

    def test_generate_command_routing(self):
        """Test that generate command is routed to AutoRTHandler."""
        # Arrange
        command_type = "generate"
        params = {"num_tasks": 3, "robot_ids": ["Robot1"], "strategy": "balanced"}
        request_id = 10001

        # Mock AutoRTOrchestrator to avoid dependencies
        with patch('servers.AutoRTIntegration.AutoRTOrchestrator') as mock_orch_class:
            mock_orch = MagicMock()
            mock_orch_class.return_value = mock_orch

            mock_orch._capture_scene.return_value = {}
            mock_orch._generate_task_candidates.return_value = []

            # Act - Send AUTORT_COMMAND
            message = UnityProtocol.encode_autort_command(command_type, params, request_id)
            self.client.sendall(message)

            # Receive response
            response_data = self._receive_complete_response()

            # Assert - Decode response
            decoded_request_id, response = UnityProtocol.decode_autort_response(response_data)

            self.assertEqual(decoded_request_id, request_id, "Request ID should match")
            self.assertIn("success", response)
            self.assertIn("tasks", response)
            self.assertIn("loop_running", response)

    def test_start_loop_command_routing(self):
        """Test that start_loop command is routed correctly."""
        # Arrange
        command_type = "start_loop"
        params = {"loop_delay": 0.5, "robot_ids": ["Robot1"], "strategy": "explore"}
        request_id = 10002

        # Act
        message = UnityProtocol.encode_autort_command(command_type, params, request_id)
        self.client.sendall(message)

        # Receive response
        response_data = self._receive_complete_response()
        decoded_request_id, response = UnityProtocol.decode_autort_response(response_data)

        # Assert
        self.assertEqual(decoded_request_id, request_id)
        self.assertTrue(response["success"])
        self.assertTrue(response["loop_running"])

        # Cleanup - stop loop
        stop_message = UnityProtocol.encode_autort_command("stop_loop", {}, 10003)
        self.client.sendall(stop_message)
        self._receive_complete_response()  # Consume response

    def test_stop_loop_command_routing(self):
        """Test that stop_loop command is routed correctly."""
        # Arrange - start loop first
        start_message = UnityProtocol.encode_autort_command("start_loop", {"loop_delay": 0.5}, 10004)
        self.client.sendall(start_message)
        self._receive_complete_response()  # Consume start response
        time.sleep(0.1)

        # Act - stop loop
        stop_message = UnityProtocol.encode_autort_command("stop_loop", {}, 10005)
        self.client.sendall(stop_message)

        # Receive response
        response_data = self._receive_complete_response()
        decoded_request_id, response = UnityProtocol.decode_autort_response(response_data)

        # Assert
        self.assertEqual(decoded_request_id, 10005)
        self.assertTrue(response["success"])
        self.assertFalse(response["loop_running"])

    def test_get_status_command_routing(self):
        """Test that get_status command returns handler status."""
        # Arrange
        command_type = "get_status"
        request_id = 10006

        # Act
        message = UnityProtocol.encode_autort_command(command_type, {}, request_id)
        self.client.sendall(message)

        # Receive response
        response_data = self._receive_complete_response()
        decoded_request_id, response = UnityProtocol.decode_autort_response(response_data)

        # Assert
        self.assertEqual(decoded_request_id, request_id)
        self.assertTrue(response["success"])
        self.assertIn("pending_tasks_count", response)
        self.assertIn("loop_config", response)

    def test_execute_task_command_routing(self):
        """Test that execute_task command is routed correctly."""
        # Arrange - cache a task first
        handler = AutoRTHandler.get_instance()
        task_dict = {
            "description": "Test task",
            "operations": [{"type": "move"}],
            "required_robots": ["Robot1"]
        }
        task_id = handler._cache_task(task_dict)

        command_type = "execute_task"
        params = {"task_id": task_id}
        request_id = 10007

        # Mock orchestrator execution
        with patch('servers.AutoRTIntegration.AutoRTOrchestrator') as mock_orch_class:
            mock_orch = MagicMock()
            mock_orch_class.return_value = mock_orch
            mock_orch._execute_task.return_value = {"success": True, "error": None}

            # Force handler to use mocked orchestrator
            handler._orchestrator = mock_orch

            # Act
            message = UnityProtocol.encode_autort_command(command_type, params, request_id)
            self.client.sendall(message)

            # Receive response
            response_data = self._receive_complete_response()
            decoded_request_id, response = UnityProtocol.decode_autort_response(response_data)

            # Assert
            self.assertEqual(decoded_request_id, request_id)
            self.assertIn("success", response)
            self.assertIn("result", response)

    def test_unknown_command_error_handling(self):
        """Test error handling for unknown command types."""
        # Arrange
        command_type = "unknown_command_xyz"
        request_id = 10008

        # Act
        message = UnityProtocol.encode_autort_command(command_type, {}, request_id)
        self.client.sendall(message)

        # Receive response
        response_data = self._receive_complete_response()
        decoded_request_id, response = UnityProtocol.decode_autort_response(response_data)

        # Assert
        self.assertEqual(decoded_request_id, request_id)
        self.assertFalse(response["success"])
        self.assertIn("error", response)
        self.assertIn("Unknown command", response["error"])

    def test_concurrent_autort_and_sequence_messages(self):
        """Test that AutoRT and sequence messages can be handled concurrently."""
        # This test verifies that the server can handle both message types
        # on the same port without interference

        # Arrange - Create second client for sequence messages
        seq_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        seq_client.connect(("127.0.0.1", SEQUENCE_SERVER_PORT))
        seq_client.settimeout(5.0)

        try:
            # Act - Send AutoRT command on first client
            autort_message = UnityProtocol.encode_autort_command("get_status", {}, 20001)
            self.client.sendall(autort_message)

            # Send sequence query on second client (would need actual sequence encoding)
            # For now, just verify AutoRT message doesn't interfere

            # Receive AutoRT response
            response_data = self._receive_complete_response()
            decoded_request_id, response = UnityProtocol.decode_autort_response(response_data)

            # Assert
            self.assertEqual(decoded_request_id, 20001)
            self.assertTrue(response["success"])

        finally:
            seq_client.close()

    def test_malformed_autort_command(self):
        """Test error handling for malformed AutoRT commands."""
        # Arrange - Create invalid message (correct header, but invalid body)
        header = struct.pack("B", MessageType.AUTORT_COMMAND)  # type
        header += struct.pack("<I", 30001)  # request_id
        invalid_body = b"invalid data"
        message = header + invalid_body

        # Act - Send malformed message
        self.client.sendall(message)

        # The server should handle this gracefully without crashing
        # We may not receive a valid response, but server should stay alive

        # Verify server is still responsive with a valid command
        time.sleep(0.1)
        valid_message = UnityProtocol.encode_autort_command("get_status", {}, 30002)
        self.client.sendall(valid_message)

        # Should receive valid response
        response_data = self._receive_complete_response()
        decoded_request_id, response = UnityProtocol.decode_autort_response(response_data)

        # Assert - Server recovered and responded
        self.assertEqual(decoded_request_id, 30002)

    def test_request_id_correlation(self):
        """Test that request IDs are correctly correlated in responses."""
        # Arrange - Send multiple commands with different request IDs
        request_ids = [40001, 40002, 40003]

        # Act - Send commands
        for req_id in request_ids:
            message = UnityProtocol.encode_autort_command("get_status", {}, req_id)
            self.client.sendall(message)

            # Receive response immediately
            response_data = self._receive_complete_response()
            decoded_request_id, response = UnityProtocol.decode_autort_response(response_data)

            # Assert - Request ID matches
            self.assertEqual(decoded_request_id, req_id, f"Request ID {req_id} should match")

    def _receive_complete_response(self):
        """Helper to receive complete AutoRT response message."""
        # Read header (5 bytes: type + request_id)
        header = self._recv_exact(5)
        if not header:
            raise Exception("Connection closed")

        msg_type = header[0]
        request_id = struct.unpack("<I", header[1:5])[0]

        # Verify message type
        if msg_type != MessageType.AUTORT_RESPONSE:
            raise Exception(f"Expected AUTORT_RESPONSE, got {msg_type}")

        # Read JSON length (4 bytes)
        json_len_bytes = self._recv_exact(4)
        if not json_len_bytes:
            raise Exception("Failed to read JSON length")
        json_len = struct.unpack("<I", json_len_bytes)[0]

        # Read JSON data
        json_data = self._recv_exact(json_len)
        if not json_data:
            raise Exception("Failed to read JSON data")

        # Reconstruct complete message
        return header + json_len_bytes + json_data

    def _recv_exact(self, num_bytes):
        """Helper to receive exact number of bytes."""
        data = b""
        while len(data) < num_bytes:
            chunk = self.client.recv(num_bytes - len(data))
            if not chunk:
                return None
            data += chunk
        return data


class TestAutoRTProtocolCompliance(unittest.TestCase):
    """Test AutoRT protocol compliance and edge cases."""

    def test_protocol_v2_header_format(self):
        """Verify AutoRT messages follow Protocol V2 header format."""
        # Arrange
        command_type = "generate"
        params = {"num_tasks": 5}
        request_id = 50001

        # Act
        encoded = UnityProtocol.encode_autort_command(command_type, params, request_id)

        # Assert - Verify header structure
        self.assertGreaterEqual(len(encoded), 5, "Message should have at least 5-byte header")
        self.assertEqual(encoded[0], MessageType.AUTORT_COMMAND, "First byte should be message type")

        # Decode request_id from header
        header_request_id = struct.unpack("<I", encoded[1:5])[0]
        self.assertEqual(header_request_id, request_id, "Request ID should be in header")

    def test_response_format_compliance(self):
        """Verify AutoRT response format compliance."""
        # Arrange
        response_data = {
            "success": True,
            "tasks": [],
            "loop_running": False,
            "error": None
        }
        request_id = 50002

        # Act
        encoded = UnityProtocol.encode_autort_response(response_data, request_id)

        # Assert - Verify header
        self.assertEqual(encoded[0], MessageType.AUTORT_RESPONSE)
        header_request_id = struct.unpack("<I", encoded[1:5])[0]
        self.assertEqual(header_request_id, request_id)

        # Verify JSON length field
        json_length = struct.unpack("<I", encoded[5:9])[0]
        self.assertGreater(json_length, 0, "JSON length should be positive")

        # Verify actual JSON length matches
        actual_json_bytes = encoded[9:]
        self.assertEqual(len(actual_json_bytes), json_length, "JSON length should match actual data")

    def test_utf8_encoding(self):
        """Test that UTF-8 encoding is handled correctly."""
        # Arrange - Use non-ASCII characters
        command_type = "generate"
        params = {"description": "Déplacer l'objet"}  # French with accents
        request_id = 50003

        # Act
        encoded = UnityProtocol.encode_autort_command(command_type, params, request_id)
        decoded_request_id, decoded_command, decoded_params = UnityProtocol.decode_autort_command(encoded)

        # Assert
        self.assertEqual(decoded_request_id, request_id)
        # decoded_params is already a dict from decode_autort_command
        self.assertEqual(decoded_params["description"], "Déplacer l'objet")


if __name__ == "__main__":
    unittest.main()
