#!/usr/bin/env python3
"""
TestAutoRTIntegration.py - Unit tests for AutoRT Unity integration

Tests AutoRTHandler singleton, task generation, loop control, and caching.
"""

import unittest
import threading
import time
from unittest.mock import Mock, patch, MagicMock

# Import modules under test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.AutoRTIntegration import AutoRTHandler
from config.AutoRT import (
    TASK_CACHE_SIZE,
    TASK_EXPIRATION_SECONDS,
)


class TestAutoRTHandler(unittest.TestCase):
    """Test suite for AutoRTHandler class."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset singleton instance for each test
        AutoRTHandler._instance = None
        self.handler = AutoRTHandler.get_instance()

    def tearDown(self):
        """Clean up after each test."""
        # Stop any running loops
        if self.handler._loop_running:
            self.handler.stop_loop()
            time.sleep(0.1)  # Allow thread to finish

        # Reset singleton
        AutoRTHandler._instance = None

    def test_singleton_pattern(self):
        """Test that AutoRTHandler follows singleton pattern."""
        handler1 = AutoRTHandler.get_instance()
        handler2 = AutoRTHandler.get_instance()

        self.assertIs(handler1, handler2, "Should return same instance")
        self.assertIs(handler1, self.handler, "Should match initial instance")

    def test_initial_state(self):
        """Test initial state of handler."""
        self.assertIsNone(self.handler._orchestrator, "Orchestrator should be lazy-initialized")
        self.assertIsNone(self.handler._loop_thread, "No loop thread initially")
        self.assertFalse(self.handler._loop_running, "Loop should not be running")
        self.assertEqual(len(self.handler._pending_tasks), 0, "No pending tasks initially")
        self.assertIsNone(self.handler._task_callback, "No callback initially")

    def test_set_task_callback(self):
        """Test setting task callback."""
        mock_callback = Mock()
        self.handler.set_task_callback(mock_callback)

        self.assertEqual(self.handler._task_callback, mock_callback)

    @patch('servers.AutoRTIntegration.AutoRTOrchestrator')
    def test_generate_tasks_success(self, mock_orchestrator_class):
        """Test successful task generation."""
        # Mock AutoRTOrchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator

        # Mock scene capture
        mock_scene_state = {"robots": ["Robot1", "Robot2"]}
        mock_orchestrator._capture_scene.return_value = mock_scene_state

        # Mock task candidates
        mock_candidates = [
            {
                "description": "Task 1",
                "operations": [{"type": "move", "robot_id": "Robot1"}],
                "required_robots": ["Robot1"],
                "reasoning": "Test task 1"
            },
            {
                "description": "Task 2",
                "operations": [{"type": "grasp", "robot_id": "Robot2"}],
                "required_robots": ["Robot2"],
                "reasoning": "Test task 2"
            }
        ]
        mock_orchestrator._generate_task_candidates.return_value = mock_candidates

        # Mock validation (all valid)
        mock_orchestrator._validate_task_safety.return_value = (True, "Valid")

        # Mock selection (return all)
        mock_orchestrator._select_tasks.return_value = mock_candidates

        # Generate tasks
        result = self.handler.generate_tasks(num_tasks=2, robot_ids=["Robot1", "Robot2"])

        # Assertions
        self.assertTrue(result["success"], "Should succeed")
        self.assertEqual(len(result["tasks"]), 2, "Should return 2 tasks")
        self.assertIsNone(result["error"], "Should have no error")
        self.assertFalse(result["loop_running"], "Loop should not be running")

        # Check tasks have required fields
        for task in result["tasks"]:
            self.assertIn("task_id", task)
            self.assertIn("description", task)
            self.assertIn("operations", task)
            self.assertIn("required_robots", task)
            self.assertIn("estimated_complexity", task)
            self.assertIn("reasoning", task)

        # Verify tasks are cached
        self.assertEqual(len(self.handler._pending_tasks), 2, "Tasks should be cached")

    @patch('servers.AutoRTIntegration.AutoRTOrchestrator')
    def test_generate_tasks_validation_filters_invalid(self, mock_orchestrator_class):
        """Test that invalid tasks are filtered out during generation."""
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator

        mock_orchestrator._capture_scene.return_value = {}

        # 3 candidates, 1 invalid
        mock_candidates = [
            {"description": "Valid 1", "operations": [], "required_robots": []},
            {"description": "Invalid", "operations": [], "required_robots": []},
            {"description": "Valid 2", "operations": [], "required_robots": []}
        ]
        mock_orchestrator._generate_task_candidates.return_value = mock_candidates

        # Second task is invalid
        def mock_validate(task):
            if task["description"] == "Invalid":
                return (False, "Safety violation")
            return (True, "Valid")

        mock_orchestrator._validate_task_safety.side_effect = mock_validate

        # Selection returns only valid tasks
        mock_orchestrator._select_tasks.return_value = [
            mock_candidates[0],
            mock_candidates[2]
        ]

        result = self.handler.generate_tasks(num_tasks=3)

        self.assertTrue(result["success"])
        self.assertEqual(len(result["tasks"]), 2, "Should only return valid tasks")

    @patch('servers.AutoRTIntegration.AutoRTOrchestrator')
    def test_generate_tasks_no_candidates(self, mock_orchestrator_class):
        """Test generation when no candidates are produced."""
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator

        mock_orchestrator._capture_scene.return_value = {}
        mock_orchestrator._generate_task_candidates.return_value = []

        result = self.handler.generate_tasks()

        self.assertTrue(result["success"], "Should succeed even with no tasks")
        self.assertEqual(len(result["tasks"]), 0, "Should return empty list")
        self.assertIsNone(result["error"])

    @patch('servers.AutoRTIntegration.AutoRTOrchestrator')
    def test_generate_tasks_error_handling(self, mock_orchestrator_class):
        """Test error handling during task generation."""
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator

        # Simulate error
        mock_orchestrator._capture_scene.side_effect = Exception("Scene capture failed")

        result = self.handler.generate_tasks()

        self.assertFalse(result["success"], "Should fail")
        self.assertEqual(len(result["tasks"]), 0)
        self.assertIsNotNone(result["error"])
        self.assertIn("Scene capture failed", result["error"])

    def test_start_loop_success(self):
        """Test starting continuous loop."""
        result = self.handler.start_loop(loop_delay=0.1)

        self.assertTrue(result["success"])
        self.assertTrue(result["loop_running"])
        self.assertIsNone(result["error"])

        # Verify loop is actually running
        self.assertTrue(self.handler._loop_running)
        self.assertIsNotNone(self.handler._loop_thread)
        if self.handler._loop_thread is not None:
            self.assertTrue(self.handler._loop_thread.is_alive())

    def test_start_loop_already_running(self):
        """Test starting loop when already running."""
        self.handler.start_loop(loop_delay=0.1)

        # Try to start again
        result = self.handler.start_loop()

        self.assertTrue(result["success"])
        self.assertTrue(result["loop_running"])
        self.assertIn("already running", result["error"].lower())

    def test_stop_loop_success(self):
        """Test stopping continuous loop."""
        # Start loop first
        self.handler.start_loop(loop_delay=0.1)
        time.sleep(0.05)  # Let it start

        # Stop loop
        result = self.handler.stop_loop()

        self.assertTrue(result["success"])
        self.assertFalse(result["loop_running"])
        self.assertIsNone(result["error"])

        # Verify loop is stopped
        self.assertFalse(self.handler._loop_running)

        # Wait for thread to finish
        if self.handler._loop_thread:
            self.handler._loop_thread.join(timeout=1.0)
            self.assertFalse(self.handler._loop_thread.is_alive())

    def test_stop_loop_not_running(self):
        """Test stopping loop when not running."""
        result = self.handler.stop_loop()

        self.assertTrue(result["success"])
        self.assertFalse(result["loop_running"])

    @patch('servers.AutoRTIntegration.AutoRTOrchestrator')
    def test_execute_task_success(self, mock_orchestrator_class):
        """Test executing a cached task."""
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator

        # Cache a task first
        task_dict = {
            "description": "Test task",
            "operations": [{"type": "move"}],
            "required_robots": ["Robot1"]
        }
        task_id = self.handler._cache_task(task_dict)

        # Mock execution
        mock_orchestrator._execute_task.return_value = {
            "success": True,
            "error": None
        }

        # Execute
        result = self.handler.execute_task(task_id)

        self.assertTrue(result["success"])
        self.assertIsNone(result["error"])
        self.assertIsNotNone(result["result"])

        # Verify task removed from cache
        with self.handler._task_lock:
            self.assertNotIn(task_id, self.handler._pending_tasks)

    def test_execute_task_not_found(self):
        """Test executing non-existent task."""
        result = self.handler.execute_task("nonexistent_task_id")

        self.assertFalse(result["success"])
        self.assertIsNone(result["result"])
        self.assertIn("not found", result["error"].lower())

    @patch('servers.AutoRTIntegration.AutoRTOrchestrator')
    def test_execute_task_error(self, mock_orchestrator_class):
        """Test error handling during task execution."""
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator

        # Cache a task
        task_dict = {"description": "Test", "operations": []}
        task_id = self.handler._cache_task(task_dict)

        # Mock execution error
        mock_orchestrator._execute_task.side_effect = Exception("Execution failed")

        result = self.handler.execute_task(task_id)

        self.assertFalse(result["success"])
        self.assertIsNone(result["result"])
        self.assertIn("Execution failed", result["error"])

    def test_get_status(self):
        """Test getting handler status."""
        # Cache some tasks
        self.handler._cache_task({"description": "Task 1", "operations": []})
        self.handler._cache_task({"description": "Task 2", "operations": []})

        status = self.handler.get_status()

        self.assertTrue(status["success"])
        self.assertFalse(status["loop_running"])
        self.assertEqual(status["pending_tasks_count"], 2)
        self.assertIn("loop_config", status)
        self.assertIsNone(status["error"])

    def test_task_caching(self):
        """Test task caching mechanism."""
        task_dict = {
            "description": "Test task",
            "operations": [{"type": "move"}],
            "required_robots": ["Robot1"],
            "reasoning": "Test reasoning"
        }

        task_id = self.handler._cache_task(task_dict)

        # Verify cached
        with self.handler._task_lock:
            self.assertIn(task_id, self.handler._pending_tasks)
            cached_task, _ = self.handler._pending_tasks[task_id]
            self.assertEqual(cached_task, task_dict)

    def test_cache_size_limit(self):
        """Test that cache respects size limit."""
        # Fill cache to limit
        for i in range(TASK_CACHE_SIZE + 5):
            task = {"description": f"Task {i}", "operations": []}
            self.handler._cache_task(task)

        # Verify cache doesn't exceed limit
        with self.handler._task_lock:
            self.assertLessEqual(len(self.handler._pending_tasks), TASK_CACHE_SIZE)

    def test_cleanup_expired_tasks(self):
        """Test that expired tasks are cleaned up."""
        # Cache a task
        task_dict = {"description": "Test", "operations": []}
        task_id = self.handler._cache_task(task_dict)

        # Manually set timestamp to expired
        from datetime import datetime, timedelta
        with self.handler._task_lock:
            expired_time = datetime.now() - timedelta(seconds=TASK_EXPIRATION_SECONDS + 1)
            self.handler._pending_tasks[task_id] = (task_dict, expired_time)

        # Run cleanup
        self.handler._cleanup_expired_tasks()

        # Verify task removed
        with self.handler._task_lock:
            self.assertNotIn(task_id, self.handler._pending_tasks)

    def test_serialize_task(self):
        """Test task serialization for Unity."""
        from autort.DataModels import ProposedTask, Operation

        # Create a ProposedTask object (not a dict) to match method signature
        task = ProposedTask(
            task_id="test_task_123",
            description="Move to position",
            operations=[
                Operation(type="move", robot_id="Robot1", parameters={}),
                Operation(type="grasp", robot_id="Robot1", parameters={})
            ],
            required_robots=["Robot1"],
            estimated_complexity=3,
            reasoning="Need to pick up object"
        )

        serialized = self.handler._serialize_task(task)

        # Verify all required fields
        self.assertEqual(serialized["task_id"], "test_task_123")
        self.assertEqual(serialized["description"], "Move to position")
        self.assertEqual(len(serialized["operations"]), 2)
        self.assertEqual(serialized["required_robots"], ["Robot1"])
        self.assertEqual(serialized["estimated_complexity"], 3)
        self.assertEqual(serialized["reasoning"], "Need to pick up object")

    @patch('servers.AutoRTIntegration.AutoRTOrchestrator')
    def test_loop_worker_generates_tasks(self, mock_orchestrator_class):
        """Test that loop worker generates tasks periodically."""
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator

        mock_orchestrator._capture_scene.return_value = {}
        mock_candidates = [{"description": "Loop task", "operations": []}]
        mock_orchestrator._generate_task_candidates.return_value = mock_candidates
        mock_orchestrator._validate_task_safety.return_value = (True, "Valid")
        mock_orchestrator._select_tasks.return_value = mock_candidates

        # Set up callback to track calls
        callback_called = threading.Event()
        received_tasks = []

        def mock_callback(response, request_id=0):  # noqa: ARG001
            received_tasks.append(response)
            callback_called.set()

        self.handler.set_task_callback(mock_callback)

        # Start loop with short delay
        self.handler.start_loop(loop_delay=0.2, robot_ids=["Robot1"])

        # Wait for at least one callback
        callback_called.wait(timeout=1.0)

        # Stop loop
        self.handler.stop_loop()

        # Verify callback was called with tasks
        self.assertGreater(len(received_tasks), 0, "Callback should be called")
        self.assertIn("tasks", received_tasks[0])

    def test_thread_safety(self):
        """Test thread-safe operations on pending tasks."""
        results = []

        def cache_tasks():
            for i in range(10):
                task = {"description": f"Task {i}", "operations": []}
                task_id = self.handler._cache_task(task)
                results.append(task_id)

        # Run multiple threads caching tasks
        threads = [threading.Thread(target=cache_tasks) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all tasks cached (within cache size limit)
        with self.handler._task_lock:
            self.assertLessEqual(len(self.handler._pending_tasks), TASK_CACHE_SIZE)
            self.assertGreater(len(self.handler._pending_tasks), 0)


class TestAutoRTProtocol(unittest.TestCase):
    """Test suite for AutoRT protocol encoding/decoding."""

    def test_command_encoding_decoding(self):
        """Test AutoRT command message encoding and decoding."""
        from core.UnityProtocol import UnityProtocol

        command_type = "generate"
        params = {"num_tasks": 5, "robot_ids": ["Robot1", "Robot2"]}
        request_id = 12345

        # Encode
        encoded = UnityProtocol.encode_autort_command(command_type, params, request_id)

        # Decode
        decoded_request_id, decoded_command, decoded_params = UnityProtocol.decode_autort_command(encoded)

        # Verify
        self.assertEqual(decoded_request_id, request_id)
        self.assertEqual(decoded_command, command_type)
        self.assertEqual(decoded_params["num_tasks"], 5)
        self.assertEqual(decoded_params["robot_ids"], ["Robot1", "Robot2"])

    def test_response_encoding_decoding(self):
        """Test AutoRT response message encoding and decoding."""
        from core.UnityProtocol import UnityProtocol

        response_data = {
            "success": True,
            "tasks": [
                {
                    "task_id": "task_123",
                    "description": "Test task",
                    "operations": [],
                    "required_robots": ["Robot1"],
                    "estimated_complexity": 2,
                    "reasoning": "Test"
                }
            ],
            "loop_running": False,
            "error": None
        }
        request_id = 67890

        # Encode
        encoded = UnityProtocol.encode_autort_response(response_data, request_id)

        # Decode
        decoded_request_id, decoded_response = UnityProtocol.decode_autort_response(encoded)

        # Verify
        self.assertEqual(decoded_request_id, request_id)
        self.assertEqual(decoded_response["success"], True)
        self.assertEqual(len(decoded_response["tasks"]), 1)
        self.assertEqual(decoded_response["tasks"][0]["task_id"], "task_123")
        self.assertEqual(decoded_response["loop_running"], False)


if __name__ == "__main__":
    unittest.main()
