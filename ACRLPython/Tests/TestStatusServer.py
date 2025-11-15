#!/usr/bin/env python3
"""
test_status_server.py - Tests for StatusServer and status query protocol

Tests the bidirectional status query system for robot state information.
"""

import unittest
import socket
import threading
import time
import json
from unittest.mock import Mock, patch

# Import modules to test
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.UnityProtocol import UnityProtocol
from servers.StatusServer import StatusServer, StatusResponseHandler, run_status_server_background
from core.TCPServerBase import ServerConfig
import LLMConfig as cfg


class TestUnityProtocolStatus(unittest.TestCase):
    """Test UnityProtocol status encoding/decoding"""

    def test_encode_status_query_basic(self):
        """Test encoding a basic status query"""
        robot_id = "Robot1"
        detailed = False

        encoded = UnityProtocol.encode_status_query(robot_id, detailed)

        # Should have: [robot_id_len:4][robot_id:N][detailed:1]
        self.assertIsInstance(encoded, bytes)
        self.assertGreaterEqual(len(encoded), 4 + len(robot_id) + 1)

    def test_encode_status_query_detailed(self):
        """Test encoding a detailed status query"""
        robot_id = "AR4_Robot"
        detailed = True

        encoded = UnityProtocol.encode_status_query(robot_id, detailed)

        # Verify detailed flag is set
        self.assertIsInstance(encoded, bytes)
        # Last byte should be 1 for detailed=True
        self.assertEqual(encoded[-1], 1)

    def test_decode_status_query(self):
        """Test decoding a status query"""
        robot_id = "Robot1"
        detailed = True

        # Encode then decode
        encoded = UnityProtocol.encode_status_query(robot_id, detailed)
        decoded = UnityProtocol.decode_status_query(encoded)

        self.assertEqual(decoded["robot_id"], robot_id)
        self.assertEqual(decoded["detailed"], detailed)

    def test_encode_status_response(self):
        """Test encoding a status response"""
        status_data = {
            "success": True,
            "robot_id": "Robot1",
            "status": {
                "position": {"x": 0.5, "y": 0.2, "z": 0.3},
                "distance_to_target": 0.1,
                "is_moving": True
            }
        }

        encoded = UnityProtocol.encode_status_response(status_data)

        # Should be: [json_len:4][json_data:N]
        self.assertIsInstance(encoded, bytes)
        self.assertGreater(len(encoded), 4)

    def test_decode_status_response(self):
        """Test decoding a status response"""
        status_data = {
            "success": True,
            "robot_id": "Robot1",
            "detailed": False,
            "status": {
                "position": {"x": 0.5, "y": 0.2, "z": 0.3},
                "distance_to_target": 0.1,
                "is_moving": True
            }
        }

        # Encode then decode
        encoded = UnityProtocol.encode_status_response(status_data)
        decoded = UnityProtocol.decode_status_response(encoded)

        self.assertEqual(decoded["success"], True)
        self.assertEqual(decoded["robot_id"], "Robot1")
        self.assertIn("status", decoded)

    def test_empty_robot_id_rejected(self):
        """Test that empty robot ID is rejected"""
        with self.assertRaises(ValueError):
            UnityProtocol.encode_status_query("", False)

    def test_long_robot_id_rejected(self):
        """Test that overly long robot ID is rejected"""
        long_id = "A" * (UnityProtocol.MAX_STRING_LENGTH + 1)
        with self.assertRaises(ValueError):
            UnityProtocol.encode_status_query(long_id, False)


class TestStatusResponseHandler(unittest.TestCase):
    """Test StatusResponseHandler singleton"""

    def setUp(self):
        """Reset singleton before each test"""
        StatusResponseHandler._instance = None
        StatusResponseHandler._response_queues = {}

    def test_create_request_queue(self):
        """Test creating a request queue"""
        robot_id = "Robot1"
        queue = StatusResponseHandler.create_request_queue(robot_id)

        self.assertIsNotNone(queue)
        self.assertEqual(queue.maxsize, 1)

    def test_put_response(self):
        """Test putting a response into queue"""
        robot_id = "Robot1"
        queue = StatusResponseHandler.create_request_queue(robot_id)

        status_data = {"success": True, "robot_id": robot_id}
        StatusResponseHandler.put_response(robot_id, status_data)

        # Queue should have the response
        self.assertFalse(queue.empty())
        received = queue.get_nowait()
        self.assertEqual(received["robot_id"], robot_id)

    def test_remove_request_queue(self):
        """Test removing a request queue"""
        robot_id = "Robot1"
        StatusResponseHandler.create_request_queue(robot_id)

        StatusResponseHandler.remove_request_queue(robot_id)

        # Queue should be removed
        self.assertNotIn(robot_id, StatusResponseHandler._response_queues)

    def test_put_response_no_queue(self):
        """Test putting response when no queue exists (should log warning)"""
        robot_id = "Robot1"
        status_data = {"success": True}

        # Should not crash, just log warning
        StatusResponseHandler.put_response(robot_id, status_data)


class TestStatusServer(unittest.TestCase):
    """Test StatusServer functionality"""

    def setUp(self):
        """Set up test server"""
        self.server_config = ServerConfig(host="127.0.0.1", port=15012)  # Use different port for testing
        self.server = None

    def tearDown(self):
        """Stop server after test"""
        if self.server is not None:
            self.server.stop()
            time.sleep(0.5)  # Give server time to clean up

    def test_server_initialization(self):
        """Test that server initializes correctly"""
        self.server = StatusServer(self.server_config)

        self.assertEqual(self.server._config.port, 15012)
        self.assertIsNotNone(self.server._response_handler)

    def test_server_start_stop(self):
        """Test starting and stopping the server"""
        self.server = StatusServer(self.server_config)

        # Start in background thread
        server_thread = threading.Thread(target=self.server.start, daemon=True)
        server_thread.start()
        time.sleep(0.5)  # Give server time to start

        self.assertTrue(self.server.is_running())

        # Stop server
        self.server.stop()
        time.sleep(0.5)

        self.assertFalse(self.server.is_running())


class TestStatusServerIntegration(unittest.TestCase):
    """Integration tests for status query flow"""

    def setUp(self):
        """Set up test server and client"""
        self.server_config = ServerConfig(host="127.0.0.1", port=15013)
        self.server = StatusServer(self.server_config)
        self.server_thread = None

    def tearDown(self):
        """Clean up server"""
        if self.server is not None:
            self.server.stop()
            if self.server_thread is not None:
                self.server_thread.join(timeout=2.0)

    def test_client_connection(self):
        """Test that clients can connect to server"""
        # Start server
        self.server_thread = threading.Thread(target=self.server.start, daemon=True)
        self.server_thread.start()
        time.sleep(0.5)

        # Try to connect
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(2.0)
            client.connect(("127.0.0.1", 15013))

            # If we get here, connection succeeded
            self.assertTrue(True)

            client.close()
        except Exception as e:
            self.fail(f"Failed to connect to server: {e}")

    def test_send_status_query(self):
        """Test sending a status query to server"""
        # Start server
        self.server_thread = threading.Thread(target=self.server.start, daemon=True)
        self.server_thread.start()
        time.sleep(0.5)

        # Connect client
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(5.0)

        try:
            client.connect(("127.0.0.1", 15013))

            # Send status query
            query = UnityProtocol.encode_status_query("Robot1", detailed=False)
            client.sendall(query)

            # Server should receive and process
            # In real scenario, would receive response here
            # For now, just verify no crash

            time.sleep(1.0)

        except Exception as e:
            self.fail(f"Error during status query: {e}")
        finally:
            client.close()


class TestStatusOperationWorkflow(unittest.TestCase):
    """Test complete status operation workflow"""

    def test_encode_decode_roundtrip(self):
        """Test full encode/decode roundtrip"""
        # Create query
        robot_id = "AR4_Robot"
        detailed = True

        # Encode query
        query_encoded = UnityProtocol.encode_status_query(robot_id, detailed)

        # Decode query
        query_decoded = UnityProtocol.decode_status_query(query_encoded)

        self.assertEqual(query_decoded["robot_id"], robot_id)
        self.assertEqual(query_decoded["detailed"], detailed)

        # Create response
        response_data = {
            "success": True,
            "robot_id": robot_id,
            "detailed": detailed,
            "status": {
                "position": {"x": 0.3, "y": 0.15, "z": 0.1},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "joint_angles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "target_position": {"x": 0.4, "y": 0.2, "z": 0.15},
                "distance_to_target": 0.15,
                "is_moving": True,
                "current_action": "moving_to_target"
            },
            "error": None
        }

        # Encode response
        response_encoded = UnityProtocol.encode_status_response(response_data)

        # Decode response
        response_decoded = UnityProtocol.decode_status_response(response_encoded)

        self.assertEqual(response_decoded["success"], True)
        self.assertEqual(response_decoded["robot_id"], robot_id)
        self.assertIn("status", response_decoded)
        self.assertEqual(response_decoded["status"]["is_moving"], True)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
