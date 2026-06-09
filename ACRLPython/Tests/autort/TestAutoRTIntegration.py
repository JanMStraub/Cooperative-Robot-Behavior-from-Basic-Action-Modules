import threading
import time
from unittest.mock import Mock, patch, MagicMock

from servers.AutoRTIntegration import AutoRTHandler
from autort.DataModels import ProposedTask, Operation
from config.AutoRT import (
    TASK_CACHE_SIZE,
    TASK_EXPIRATION_SECONDS,
)


def _make_task(
    description="Test task", task_id=None, operations=None, required_robots=None
):
    """Helper to create a ProposedTask object for caching tests."""
    import uuid

    return ProposedTask(
        task_id=task_id or str(uuid.uuid4()),
        description=description,
        operations=operations
        or [Operation(type="wait", robot_id="Robot1", parameters={"seconds": 1})],
        required_robots=required_robots or ["Robot1"],
        estimated_complexity=1,
        reasoning="test",
    )


class TestAutoRTHandler:

    def setup_method(self):
        AutoRTHandler._instance = None
        self.handler = AutoRTHandler.get_instance()

    def teardown_method(self):
        if self.handler._loop_running:
            self.handler.stop_loop()
            time.sleep(0.1)

        AutoRTHandler._instance = None

    def test_singleton_pattern(self):
        handler1 = AutoRTHandler.get_instance()
        handler2 = AutoRTHandler.get_instance()

        assert handler1 is handler2
        assert handler1 is self.handler

    def test_initial_state(self):
        assert self.handler._orchestrator is None
        assert self.handler._loop_thread is None
        assert not self.handler._loop_running
        assert len(self.handler._pending_tasks) == 0
        assert self.handler._task_callback is None

    def test_set_task_callback(self):
        mock_callback = Mock()
        self.handler.set_task_callback(mock_callback)

        assert self.handler._task_callback == mock_callback

    def test_generate_tasks_success(self):
        mock_orchestrator = MagicMock()
        self.handler._orchestrator = mock_orchestrator

        from autort.DataModels import SceneDescription

        mock_scene = SceneDescription(
            timestamp=0.0, objects=[], scene_summary="test", robot_states={}
        )
        mock_orchestrator._capture_scene.return_value = mock_scene

        task1 = _make_task("Task 1", task_id="t1")
        task2 = _make_task("Task 2", task_id="t2")
        mock_candidates = [task1, task2]
        mock_orchestrator.task_generator.generate_tasks.return_value = mock_candidates

        mock_verdict = MagicMock()
        mock_verdict.approved = True
        mock_verdict.warnings = []
        mock_orchestrator.constitution.evaluate_task.return_value = mock_verdict

        mock_orchestrator.task_selector.select_task.side_effect = mock_candidates + [
            None
        ]

        with patch("servers.AutoRTIntegration.ENABLE_SAFETY_VALIDATION", False):
            result = self.handler.generate_tasks(
                num_tasks=2, robot_ids=["Robot1", "Robot2"]
            )

        assert result["success"]
        assert len(result["tasks"]) == 2
        assert result["error"] is None
        assert not result["loop_running"]

        for task in result["tasks"]:
            assert "task_id" in task
            assert "description" in task
            assert "operations" in task
            assert "required_robots" in task
            assert "estimated_complexity" in task
            assert "reasoning" in task

        assert len(self.handler._pending_tasks) == 2

    def test_generate_tasks_validation_filters_invalid(self):
        mock_orchestrator = MagicMock()
        self.handler._orchestrator = mock_orchestrator

        from autort.DataModels import SceneDescription

        mock_scene = SceneDescription(
            timestamp=0.0, objects=[], scene_summary="test", robot_states={}
        )
        mock_orchestrator._capture_scene.return_value = mock_scene

        task_valid1 = _make_task("Valid 1", task_id="v1")
        task_invalid = _make_task("Invalid", task_id="inv")
        task_valid2 = _make_task("Valid 2", task_id="v2")
        mock_candidates = [task_valid1, task_invalid, task_valid2]
        mock_orchestrator.task_generator.generate_tasks.return_value = mock_candidates

        def mock_evaluate(task, _scene):
            verdict = MagicMock()
            verdict.warnings = []
            verdict.approved = task.description != "Invalid"
            verdict.rejection_reason = (
                "Safety violation" if task.description == "Invalid" else ""
            )
            return verdict

        mock_orchestrator.constitution.evaluate_task.side_effect = mock_evaluate
        mock_orchestrator.task_selector.select_task.side_effect = [
            task_valid1,
            task_valid2,
            None,
        ]

        result = self.handler.generate_tasks(num_tasks=3)

        assert result["success"]
        assert len(result["tasks"]) == 2

    def test_generate_tasks_no_candidates(self):
        mock_orchestrator = MagicMock()
        self.handler._orchestrator = mock_orchestrator

        from autort.DataModels import SceneDescription

        mock_scene = SceneDescription(
            timestamp=0.0, objects=[], scene_summary="test", robot_states={}
        )
        mock_orchestrator._capture_scene.return_value = mock_scene
        mock_orchestrator.task_generator.generate_tasks.return_value = []

        result = self.handler.generate_tasks()

        assert result["success"]
        assert len(result["tasks"]) == 0
        assert result["error"] is None

    def test_generate_tasks_error_handling(self):
        mock_orchestrator = MagicMock()
        self.handler._orchestrator = mock_orchestrator

        mock_orchestrator._capture_scene.side_effect = Exception("Scene capture failed")

        result = self.handler.generate_tasks()

        assert not result["success"]
        assert len(result["tasks"]) == 0
        assert result["error"] is not None
        assert "Scene capture failed" in result["error"]

    def test_start_loop_success(self):
        result = self.handler.start_loop(loop_delay=0.1)

        assert result["success"]
        assert result["loop_running"]
        assert result["error"] is None

        assert self.handler._loop_running
        assert self.handler._loop_thread is not None
        if self.handler._loop_thread is not None:
            assert self.handler._loop_thread.is_alive()

    def test_start_loop_already_running(self):
        self.handler.start_loop(loop_delay=0.1)

        result = self.handler.start_loop()

        assert result["success"]
        assert result["loop_running"]
        assert "already running" in result["error"].lower()

    def test_stop_loop_success(self):
        mock_orchestrator = MagicMock()
        self.handler._orchestrator = mock_orchestrator
        from autort.DataModels import SceneDescription

        mock_scene = SceneDescription(
            timestamp=0.0, objects=[], scene_summary="", robot_states={}
        )
        mock_orchestrator._capture_scene.return_value = mock_scene
        mock_orchestrator.task_generator.generate_tasks.return_value = []

        with patch("servers.AutoRTIntegration.ENABLE_SAFETY_VALIDATION", False):
            self.handler.start_loop(loop_delay=60.0)
        time.sleep(0.1)

        result = self.handler.stop_loop()

        assert result["success"]
        assert not result["loop_running"]
        assert result["error"] is None

        assert not self.handler._loop_running

        if self.handler._loop_thread:
            self.handler._loop_thread.join(timeout=2.0)
            assert not self.handler._loop_thread.is_alive()

    def test_stop_loop_not_running(self):
        result = self.handler.stop_loop()

        assert result["success"]
        assert not result["loop_running"]

    def test_execute_task_success(self):
        mock_orchestrator = MagicMock()
        self.handler._orchestrator = mock_orchestrator

        task = _make_task("Test task")
        task_id = self.handler._cache_task(task)

        mock_orchestrator._execute_task.return_value = {"success": True, "error": None}

        result = self.handler.execute_task(task_id)

        assert result["success"]
        assert result["error"] is None
        assert result["result"] is not None

        with self.handler._task_lock:
            assert task_id not in self.handler._pending_tasks

    def test_execute_task_not_found(self):
        result = self.handler.execute_task("nonexistent_task_id")

        assert not result["success"]
        assert result["result"] is None
        assert "not found" in result["error"].lower()

    def test_execute_task_error(self):
        result = self.handler.execute_task("nonexistent_id_for_error_test")

        assert not result["success"]
        assert result["result"] is None
        assert result["error"] is not None

    def test_get_status(self):
        self.handler._cache_task(_make_task("Task 1"))
        self.handler._cache_task(_make_task("Task 2"))

        status = self.handler.get_status()

        assert status["success"]
        assert not status["loop_running"]
        assert status["pending_tasks_count"] == 2
        assert "loop_config" in status
        assert status["error"] is None

    def test_task_caching(self):
        task = _make_task("Test task")

        task_id = self.handler._cache_task(task)

        with self.handler._task_lock:
            assert task_id in self.handler._pending_tasks
            cached_task, _ = self.handler._pending_tasks[task_id]
            assert cached_task is task

    def test_cache_size_limit(self):
        for i in range(TASK_CACHE_SIZE + 5):
            task = _make_task(f"Task {i}")
            self.handler._cache_task(task)

        with self.handler._task_lock:
            assert len(self.handler._pending_tasks) <= TASK_CACHE_SIZE

    def test_cleanup_expired_tasks(self):
        task = _make_task("Test")
        task_id = self.handler._cache_task(task)

        from datetime import datetime, timedelta

        with self.handler._task_lock:
            expired_time = datetime.now() - timedelta(
                seconds=TASK_EXPIRATION_SECONDS + 1
            )
            self.handler._pending_tasks[task_id] = (task, expired_time)

        self.handler._cleanup_expired_tasks()

        with self.handler._task_lock:
            assert task_id not in self.handler._pending_tasks

    def test_serialize_task(self):
        from autort.DataModels import ProposedTask, Operation

        task = ProposedTask(
            task_id="test_task_123",
            description="Move to position",
            operations=[
                Operation(type="move", robot_id="Robot1", parameters={}),
                Operation(type="grasp", robot_id="Robot1", parameters={}),
            ],
            required_robots=["Robot1"],
            estimated_complexity=3,
            reasoning="Need to pick up object",
        )

        serialized = self.handler._serialize_task(task)

        assert serialized["task_id"] == "test_task_123"
        assert serialized["description"] == "Move to position"
        assert len(serialized["operations"]) == 2
        assert serialized["required_robots"] == ["Robot1"]
        assert serialized["estimated_complexity"] == 3
        assert serialized["reasoning"] == "Need to pick up object"

    def test_loop_worker_generates_tasks(self):
        mock_orchestrator = MagicMock()
        self.handler._orchestrator = mock_orchestrator

        from autort.DataModels import SceneDescription

        mock_scene = SceneDescription(
            timestamp=0.0, objects=[], scene_summary="test", robot_states={}
        )
        mock_orchestrator._capture_scene.return_value = mock_scene
        loop_task = _make_task("Loop task")
        mock_orchestrator.task_generator.generate_tasks.return_value = [loop_task]
        mock_orchestrator.task_selector.select_task.return_value = loop_task

        callback_called = threading.Event()
        received_tasks = []

        def mock_callback(response, request_id=0):
            received_tasks.append(response)
            callback_called.set()

        self.handler.set_task_callback(mock_callback)

        with patch("servers.AutoRTIntegration.ENABLE_SAFETY_VALIDATION", False):
            self.handler.start_loop(loop_delay=0.2, robot_ids=["Robot1"])

        callback_called.wait(timeout=1.0)

        self.handler.stop_loop()

        assert len(received_tasks) > 0
        assert "tasks" in received_tasks[0]

    def test_thread_safety(self):
        results = []

        def cache_tasks():
            for i in range(10):
                task = _make_task(f"Task {i}")
                task_id = self.handler._cache_task(task)
                results.append(task_id)

        threads = [threading.Thread(target=cache_tasks) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with self.handler._task_lock:
            assert len(self.handler._pending_tasks) <= TASK_CACHE_SIZE
            assert len(self.handler._pending_tasks) > 0


class TestAutoRTProtocol:

    def test_command_encoding_decoding(self):
        from core.UnityProtocol import UnityProtocol

        command_type = "generate"
        params = {"num_tasks": 5, "robot_ids": ["Robot1", "Robot2"]}
        request_id = 12345

        encoded = UnityProtocol.encode_autort_command(command_type, params, request_id)

        decoded_request_id, decoded_command, decoded_params = (
            UnityProtocol.decode_autort_command(encoded)
        )

        assert decoded_request_id == request_id
        assert decoded_command == command_type
        assert decoded_params["num_tasks"] == 5
        assert decoded_params["robot_ids"] == ["Robot1", "Robot2"]

    def test_response_encoding_decoding(self):
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
                    "reasoning": "Test",
                }
            ],
            "loop_running": False,
            "error": None,
        }
        request_id = 67890

        encoded = UnityProtocol.encode_autort_response(response_data, request_id)

        decoded_request_id, decoded_response = UnityProtocol.decode_autort_response(
            encoded
        )

        assert decoded_request_id == request_id
        assert decoded_response["success"] is True
        assert len(decoded_response["tasks"]) == 1
        assert decoded_response["tasks"][0]["task_id"] == "task_123"
        assert decoded_response["loop_running"] is False
