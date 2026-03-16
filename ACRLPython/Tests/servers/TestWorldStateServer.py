#!/usr/bin/env python3
"""
Test suite for WorldStateServer

Tests the dedicated world state streaming server that receives robot/object
state updates from Unity's WorldStatePublisher.
"""

import unittest
import socket
import time
import threading
from servers.WorldStateServer import WorldStateServer
from core.UnityProtocol import UnityProtocol


class TestWorldStateServer(unittest.TestCase):
    """Test WorldStateServer functionality."""

    @classmethod
    def setUpClass(cls):
        """Start server once for all tests."""
        from core.TCPServerBase import ServerConfig

        config = ServerConfig(
            host="127.0.0.1", port=5914
        )  # Use different port for testing
        cls.server = WorldStateServer(config=config)
        cls.server.start()
        time.sleep(0.5)  # Wait for server to start

    @classmethod
    def tearDownClass(cls):
        """Stop server after all tests."""
        cls.server.stop()
        time.sleep(0.5)

    def setUp(self):
        """Set up before each test."""
        self.client = None

    def tearDown(self):
        """Clean up after each test."""
        if self.client:
            try:
                self.client.close()
            except:
                pass

    def _create_client(self) -> socket.socket:
        """Create and connect a test client."""
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", 5914))
        return client

    def _send_world_state(self, client: socket.socket, state_data: dict) -> None:
        """
        Send a world state update to the server.

        Args:
            client: Connected socket
            state_data: World state dictionary
        """
        # encode_status_response expects a dict, not a JSON string
        message = UnityProtocol.encode_status_response(state_data, request_id=0)
        client.sendall(message)

    def test_server_starts_and_stops(self):
        """Test that server can start and stop cleanly."""
        # Server is already running from setUpClass
        self.assertTrue(self.server.is_running())

    def test_receive_world_state_update(self):
        """Test receiving a world state update."""
        # Create test world state
        world_state = {
            "type": "world_state_update",
            "robots": [
                {
                    "robot_id": "Robot1",
                    "position": {"x": 1.0, "y": 0.5, "z": 2.0},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    "target_position": {"x": 1.5, "y": 0.5, "z": 2.5},
                    "gripper_state": "open",
                    "is_moving": True,
                    "is_initialized": True,
                    "joint_angles": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                }
            ],
            "objects": [
                {
                    "object_id": "RedCube",
                    "position": {"x": 2.0, "y": 0.1, "z": 3.0},
                    "color": "red",
                    "object_type": "cube",
                    "confidence": 0.95,
                }
            ],
            "timestamp": 123.45,
        }

        # Send world state
        self.client = self._create_client()
        self._send_world_state(self.client, world_state)

        # Wait for server to process
        time.sleep(0.5)

        # Verify server received and stored the state
        latest_state = self.server.get_latest_state()
        self.assertIsNotNone(latest_state)
        assert latest_state is not None
        self.assertEqual(latest_state["type"], "world_state_update")
        self.assertEqual(len(latest_state["robots"]), 1)
        self.assertEqual(len(latest_state["objects"]), 1)
        self.assertEqual(latest_state["timestamp"], 123.45)

    def test_get_robot_state(self):
        """Test retrieving specific robot state."""
        world_state = {
            "type": "world_state_update",
            "robots": [
                {
                    "robot_id": "Robot1",
                    "position": {"x": 1.0, "y": 0.5, "z": 2.0},
                    "gripper_state": "closed",
                    "is_moving": False,
                    "is_initialized": True,
                    "joint_angles": [],
                },
                {
                    "robot_id": "Robot2",
                    "position": {"x": -1.0, "y": 0.5, "z": -2.0},
                    "gripper_state": "open",
                    "is_moving": True,
                    "is_initialized": True,
                    "joint_angles": [],
                },
            ],
            "objects": [],
            "timestamp": 200.0,
        }

        self.client = self._create_client()
        self._send_world_state(self.client, world_state)
        time.sleep(0.5)

        # Get specific robot state
        robot1 = self.server.get_robot_state("Robot1")
        self.assertIsNotNone(robot1)
        assert robot1 is not None
        self.assertEqual(robot1["robot_id"], "Robot1")
        self.assertEqual(robot1["gripper_state"], "closed")
        self.assertFalse(robot1["is_moving"])

        robot2 = self.server.get_robot_state("Robot2")
        self.assertIsNotNone(robot2)
        assert robot2 is not None
        self.assertEqual(robot2["robot_id"], "Robot2")
        self.assertTrue(robot2["is_moving"])

        # Non-existent robot
        robot3 = self.server.get_robot_state("Robot3")
        self.assertIsNone(robot3)

    def test_get_object_state(self):
        """Test retrieving specific object state."""
        world_state = {
            "type": "world_state_update",
            "robots": [],
            "objects": [
                {
                    "object_id": "RedCube",
                    "position": {"x": 1.0, "y": 0.1, "z": 2.0},
                    "color": "red",
                    "object_type": "cube",
                    "confidence": 0.9,
                },
                {
                    "object_id": "BlueSphere",
                    "position": {"x": -1.0, "y": 0.2, "z": -2.0},
                    "color": "blue",
                    "object_type": "sphere",
                    "confidence": 0.85,
                },
            ],
            "timestamp": 300.0,
        }

        self.client = self._create_client()
        self._send_world_state(self.client, world_state)
        time.sleep(0.5)

        # Get specific object state
        red_cube = self.server.get_object_state("RedCube")
        self.assertIsNotNone(red_cube)
        assert red_cube is not None
        self.assertEqual(red_cube["object_id"], "RedCube")
        self.assertEqual(red_cube["color"], "red")

        blue_sphere = self.server.get_object_state("BlueSphere")
        self.assertIsNotNone(blue_sphere)
        assert blue_sphere is not None
        self.assertEqual(blue_sphere["color"], "blue")

        # Non-existent object
        green_cube = self.server.get_object_state("GreenCube")
        self.assertIsNone(green_cube)

    def test_get_all_ids(self):
        """Test retrieving all robot and object IDs."""
        world_state = {
            "type": "world_state_update",
            "robots": [
                {"robot_id": "Robot1", "is_moving": False},
                {"robot_id": "Robot2", "is_moving": True},
            ],
            "objects": [
                {"object_id": "Cube1", "color": "red"},
                {"object_id": "Cube2", "color": "blue"},
                {"object_id": "Sphere1", "color": "green"},
            ],
            "timestamp": 400.0,
        }

        self.client = self._create_client()
        self._send_world_state(self.client, world_state)
        time.sleep(0.5)

        # Get all robot IDs
        robot_ids = self.server.get_all_robot_ids()
        self.assertEqual(len(robot_ids), 2)
        self.assertIn("Robot1", robot_ids)
        self.assertIn("Robot2", robot_ids)

        # Get all object IDs
        object_ids = self.server.get_all_object_ids()
        self.assertEqual(len(object_ids), 3)
        self.assertIn("Cube1", object_ids)
        self.assertIn("Cube2", object_ids)
        self.assertIn("Sphere1", object_ids)

    def test_statistics(self):
        """Test server statistics tracking."""
        # Get initial stats
        stats = self.server.get_statistics()
        initial_count = stats["updates_received"]

        # Send multiple updates
        for i in range(3):
            world_state = {
                "type": "world_state_update",
                "robots": [],
                "objects": [],
                "timestamp": float(i),
            }
            self.client = self._create_client()
            self._send_world_state(self.client, world_state)
            time.sleep(0.3)
            self.client.close()

        # Check stats updated
        stats = self.server.get_statistics()
        self.assertEqual(stats["updates_received"], initial_count + 3)
        self.assertTrue(stats["has_state"])
        self.assertIsNotNone(stats["last_update_time"])

    def test_thread_safety(self):
        """Test concurrent access to world state."""
        # Send initial state
        world_state = {
            "type": "world_state_update",
            "robots": [{"robot_id": "Robot1", "is_moving": False}],
            "objects": [],
            "timestamp": 500.0,
        }
        self.client = self._create_client()
        self._send_world_state(self.client, world_state)
        time.sleep(0.5)

        # Concurrent reads
        results = []

        def read_state():
            for _ in range(10):
                state = self.server.get_latest_state()
                results.append(state is not None)
                time.sleep(0.01)

        threads = [threading.Thread(target=read_state) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads should succeed
        self.assertTrue(all(results))


if __name__ == "__main__":
    unittest.main()
