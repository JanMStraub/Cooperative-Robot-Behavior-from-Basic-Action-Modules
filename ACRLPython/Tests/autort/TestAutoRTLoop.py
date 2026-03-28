#!/usr/bin/env python3
"""
Test AutoRT Orchestration Loop

Integration tests for the main AutoRT loop.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from autort.AutoRTLoop import AutoRTOrchestrator
from autort.DataModels import (
    SceneDescription,
    GroundedObject,
    ProposedTask,
    Operation,
    TaskVerdict,
)


@pytest.fixture
def mock_config():
    """Mock AutoRT config"""
    config = Mock()
    config.DEFAULT_ROBOTS = ["Robot1", "Robot2"]
    config.HUMAN_IN_LOOP_DEFAULT = False  # Disable for automated testing
    config.LOOP_DELAY_SECONDS = 0.1
    config.MAX_TASK_CANDIDATES = 3
    config.ENABLE_COLLABORATIVE_TASKS = True
    config.USE_VLM_REASONING = False
    return config


@pytest.fixture
def mock_registry():
    """Mock operation registry"""
    registry = Mock()
    registry.execute_operation_by_name = Mock(return_value=Mock(success=False))
    return registry


@pytest.fixture
def mock_world_state():
    """Mock WorldState"""
    world_state = Mock()
    world_state.get_all_objects = Mock(return_value=[])
    world_state.get_robot_state = Mock(return_value=None)
    world_state.get_robot_position = Mock(return_value=None)
    return world_state


@pytest.fixture
def sample_scene():
    """Sample scene with objects"""
    return SceneDescription(
        timestamp=time.time(),
        objects=[
            GroundedObject(
                object_id="cube_01",
                color="red",
                position=(0.3, 0.2, 0.1),
                confidence=0.95,
                graspable=True,
            )
        ],
        scene_summary="One red cube on table",
        robot_states={},
    )


@pytest.fixture
def sample_task():
    """Sample proposed task"""
    return ProposedTask(
        task_id="task_001",
        description="Pick red cube",
        operations=[
            Operation(
                type="move_to_coordinate",
                robot_id="Robot1",
                parameters={"target_position": [0.3, 0.2, 0.1]},
            ),
            Operation(
                type="control_gripper",
                robot_id="Robot1",
                parameters={"action": "close"},
            ),
        ],
        required_robots=["Robot1"],
        estimated_complexity=3,
        reasoning="Simple pick task",
    )


# ============================================================================
# Initialization Tests
# ============================================================================


def test_orchestrator_init_defaults():
    """AutoRTOrchestrator initializes with defaults"""
    with patch("autort.AutoRTLoop.config") as mock_cfg:
        mock_cfg.DEFAULT_ROBOTS = ["Robot1"]
        mock_cfg.HUMAN_IN_LOOP_DEFAULT = True
        mock_cfg.LOOP_DELAY_SECONDS = 5.0

        with patch("autort.AutoRTLoop.get_global_registry"):
            with patch("autort.AutoRTLoop.get_world_state"):
                with patch("autort.AutoRTLoop.TaskGenerator"):
                    with patch("autort.AutoRTLoop.RobotConstitution"):
                        with patch("orchestrators.SequenceExecutor.SequenceExecutor"):
                            orchestrator = AutoRTOrchestrator()

                            assert orchestrator.robot_ids == ["Robot1"]
                            assert orchestrator.human_in_loop is True
                            assert orchestrator.loop_delay == 5.0
                            assert orchestrator.strategy == "balanced"


def test_orchestrator_init_custom():
    """AutoRTOrchestrator accepts custom parameters"""
    with patch("autort.AutoRTLoop.get_global_registry"):
        with patch("autort.AutoRTLoop.get_world_state"):
            orchestrator = AutoRTOrchestrator(
                robot_ids=["Robot2"],
                human_in_loop=False,
                loop_delay_seconds=1.0,
                strategy="explore",
            )

            assert orchestrator.robot_ids == ["Robot2"]
            assert orchestrator.human_in_loop is False
            assert orchestrator.loop_delay == 1.0
            assert orchestrator.strategy == "explore"


def test_orchestrator_autonomous_overrides_human_in_loop():
    """Autonomous flag overrides human_in_loop"""
    with patch("autort.AutoRTLoop.get_global_registry"):
        with patch("autort.AutoRTLoop.get_world_state"):
            orchestrator = AutoRTOrchestrator(
                human_in_loop=True, autonomous=True  # Should override
            )

            assert orchestrator.human_in_loop is False


# ============================================================================
# Scene Capture Tests
# ============================================================================


def test_capture_scene_stereo_detection(mock_config, mock_registry, mock_world_state):
    """Scene capture uses detect_object_stereo"""
    # Mock detection result
    detection_result = Mock(
        success=True,
        result={
            "detections": [
                {
                    "object_id": "cube_01",
                    "color": "red",
                    "x": 0.3,
                    "y": 0.2,
                    "z": 0.1,
                    "confidence": 0.95,
                    "is_graspable": True,
                }
            ]
        },
    )
    mock_registry.execute_operation_by_name = Mock(return_value=detection_result)

    with patch("autort.AutoRTLoop.config", mock_config):
        with patch("autort.AutoRTLoop.get_global_registry", return_value=mock_registry):
            with patch(
                "autort.AutoRTLoop.get_world_state", return_value=mock_world_state
            ):
                with patch("autort.AutoRTLoop.TaskGenerator"):
                    with patch("autort.AutoRTLoop.RobotConstitution"):
                        with patch("orchestrators.SequenceExecutor.SequenceExecutor"):
                            orchestrator = AutoRTOrchestrator()
                            scene = orchestrator._capture_scene()

                            # Verify stereo detection was called
                            mock_registry.execute_operation_by_name.assert_called_with(
                                "detect_object_stereo",
                                selection="all",
                                camera_id="TableStereoCamera",
                            )

                            # Verify scene has object
                            assert len(scene.objects) == 1
                            assert scene.objects[0].color == "red"


def test_capture_scene_supplements_with_world_state(mock_config, mock_registry):
    """Scene capture supplements with WorldState objects"""
    # Mock empty detection
    mock_registry.execute_operation_by_name = Mock(return_value=Mock(success=False))

    # Mock WorldState with object
    world_state_obj = Mock()
    world_state_obj.object_id = "cube_02"
    world_state_obj.color = "blue"
    world_state_obj.position = (0.4, 0.3, 0.1)
    world_state_obj.confidence = 0.9
    world_state_obj.is_graspable = True

    mock_world_state = Mock()
    mock_world_state.get_all_objects = Mock(return_value=[world_state_obj])
    mock_world_state.get_robot_state = Mock(return_value=None)

    with patch("autort.AutoRTLoop.config", mock_config):
        with patch("autort.AutoRTLoop.get_global_registry", return_value=mock_registry):
            with patch(
                "autort.AutoRTLoop.get_world_state", return_value=mock_world_state
            ):
                with patch("autort.AutoRTLoop.TaskGenerator"):
                    with patch("autort.AutoRTLoop.RobotConstitution"):
                        with patch("orchestrators.SequenceExecutor.SequenceExecutor"):
                            orchestrator = AutoRTOrchestrator()
                            scene = orchestrator._capture_scene()

                            # Verify WorldState object was added
                            assert len(scene.objects) == 1
                            assert scene.objects[0].color == "blue"


def test_capture_scene_deduplicates_objects(mock_config, mock_registry):
    """Scene capture avoids duplicating objects from stereo and WorldState"""
    # Mock detection with one object
    detection_result = Mock(
        success=True,
        result={
            "detections": [
                {
                    "object_id": "cube_01",
                    "color": "red",
                    "x": 0.3,
                    "y": 0.2,
                    "z": 0.1,
                    "confidence": 0.95,
                    "is_graspable": True,
                }
            ]
        },
    )
    mock_registry.execute_operation_by_name = Mock(return_value=detection_result)

    # Mock WorldState with same object (same position)
    world_state_obj = Mock()
    world_state_obj.object_id = "cube_01"
    world_state_obj.color = "red"
    world_state_obj.position = (0.3, 0.2, 0.1)  # Same position
    world_state_obj.confidence = 0.95
    world_state_obj.is_graspable = True

    mock_world_state = Mock()
    mock_world_state.get_all_objects = Mock(return_value=[world_state_obj])
    mock_world_state.get_robot_state = Mock(return_value=None)

    with patch("autort.AutoRTLoop.config", mock_config):
        with patch("autort.AutoRTLoop.get_global_registry", return_value=mock_registry):
            with patch(
                "autort.AutoRTLoop.get_world_state", return_value=mock_world_state
            ):
                with patch("autort.AutoRTLoop.TaskGenerator"):
                    with patch("autort.AutoRTLoop.RobotConstitution"):
                        with patch("orchestrators.SequenceExecutor.SequenceExecutor"):
                            orchestrator = AutoRTOrchestrator()
                            scene = orchestrator._capture_scene()

                            # Should only have one object (deduplicated)
                            assert len(scene.objects) == 1


# ============================================================================
# Task Execution Tests
# ============================================================================


def test_execute_task_converts_to_sequence_format(mock_config, sample_task):
    """Task execution converts ProposedTask to SequenceExecutor format"""
    mock_executor = Mock()
    mock_executor.execute_sequence = Mock(return_value={"success": True})

    with patch("autort.AutoRTLoop.get_global_registry"):
        with patch("autort.AutoRTLoop.get_world_state"):
            with patch("autort.AutoRTLoop.TaskGenerator"):
                with patch("autort.AutoRTLoop.RobotConstitution"):
                    with patch(
                        "orchestrators.SequenceExecutor.SequenceExecutor",
                        return_value=mock_executor,
                    ):
                        orchestrator = AutoRTOrchestrator()
                        result = orchestrator._execute_task(sample_task)

                        # Verify executor was called with correct format
                        # Format: [{"operation": "...", "params": {"robot_id": "...", ...}}, ...]
                        call_args = mock_executor.execute_sequence.call_args[0][0]
                        assert len(call_args) == 2
                        assert call_args[0]["operation"] == "move_to_coordinate"
                        assert "robot_id" in call_args[0]["params"]
                        assert "target_position" in call_args[0]["params"]

                        assert result["success"] is True


def test_execute_task_handles_errors(mock_config, sample_task):
    """Task execution handles exceptions gracefully"""
    mock_executor = Mock()
    mock_executor.execute_sequence = Mock(side_effect=Exception("Execution failed"))

    with patch("autort.AutoRTLoop.get_global_registry"):
        with patch("autort.AutoRTLoop.get_world_state"):
            with patch("autort.AutoRTLoop.TaskGenerator"):
                with patch("autort.AutoRTLoop.RobotConstitution"):
                    with patch(
                        "orchestrators.SequenceExecutor.SequenceExecutor",
                        return_value=mock_executor,
                    ):
                        orchestrator = AutoRTOrchestrator()
                        result = orchestrator._execute_task(sample_task)

                        assert result["success"] is False
                        assert "error" in result


# ============================================================================
# Full Iteration Tests
# ============================================================================


def test_run_one_iteration_no_objects_skips(mock_config):
    """Iteration skips when no objects detected"""
    mock_orchestrator = MagicMock()
    mock_orchestrator._capture_scene = Mock(
        return_value=SceneDescription(
            timestamp=time.time(),
            objects=[],  # No objects
        )
    )

    # Call the real method with mocked self
    AutoRTOrchestrator._run_one_iteration(mock_orchestrator)

    # Should not proceed to task generation
    mock_orchestrator.task_generator.generate_tasks.assert_not_called()


def test_run_one_iteration_no_tasks_generated_skips(mock_config, sample_scene):
    """Iteration skips when no tasks generated"""
    mock_orchestrator = MagicMock()
    mock_orchestrator._capture_scene = Mock(return_value=sample_scene)
    mock_orchestrator.task_generator.generate_tasks = Mock(return_value=[])  # No tasks
    mock_orchestrator.robot_ids = ["Robot1"]

    # Mock config access
    with patch("autort.AutoRTLoop.config", mock_config):
        AutoRTOrchestrator._run_one_iteration(mock_orchestrator)

    # Should not proceed to constitution
    mock_orchestrator.constitution.evaluate_task.assert_not_called()


def test_run_one_iteration_all_tasks_rejected_skips(
    mock_config, sample_scene, sample_task
):
    """Iteration skips when all tasks rejected by constitution"""
    mock_orchestrator = MagicMock()
    mock_orchestrator._capture_scene = Mock(return_value=sample_scene)
    mock_orchestrator.task_generator.generate_tasks = Mock(return_value=[sample_task])
    mock_orchestrator.constitution.evaluate_task = Mock(
        return_value=TaskVerdict(approved=False, rejection_reason="Unsafe")
    )
    mock_orchestrator.robot_ids = ["Robot1"]

    with patch("autort.AutoRTLoop.config", mock_config):
        AutoRTOrchestrator._run_one_iteration(mock_orchestrator)

    # Should not proceed to selection
    mock_orchestrator.task_selector.select_task.assert_not_called()


def test_run_one_iteration_full_success(mock_config, sample_scene, sample_task):
    """Full iteration executes task successfully"""
    mock_orchestrator = MagicMock()
    mock_orchestrator._capture_scene = Mock(return_value=sample_scene)
    mock_orchestrator.task_generator.generate_tasks = Mock(return_value=[sample_task])
    mock_orchestrator.constitution.evaluate_task = Mock(
        return_value=TaskVerdict(approved=True)
    )
    mock_orchestrator.task_selector.select_task = Mock(return_value=sample_task)
    mock_orchestrator._execute_task = Mock(return_value={"success": True})
    mock_orchestrator.task_selector.update_history = Mock()
    mock_orchestrator.robot_ids = ["Robot1"]
    mock_orchestrator.human_in_loop = False

    with patch("autort.AutoRTLoop.config", mock_config):
        AutoRTOrchestrator._run_one_iteration(mock_orchestrator)

    # Verify full pipeline executed
    mock_orchestrator._capture_scene.assert_called_once()
    mock_orchestrator.task_generator.generate_tasks.assert_called_once()
    mock_orchestrator.constitution.evaluate_task.assert_called()
    mock_orchestrator.task_selector.select_task.assert_called_once()
    mock_orchestrator._execute_task.assert_called_once_with(sample_task)
    mock_orchestrator.task_selector.update_history.assert_called_once()


# ============================================================================
# Human Approval Tests
# ============================================================================


def test_request_approval_accepts_y(mock_config, sample_task):
    """Human approval accepts 'y' input"""
    with patch("autort.AutoRTLoop.get_global_registry"):
        with patch("autort.AutoRTLoop.get_world_state"):
            with patch("builtins.input", return_value="y"):
                orchestrator = AutoRTOrchestrator()
                approved_task = orchestrator._request_approval(sample_task)

                assert approved_task == sample_task


def test_request_approval_rejects_n(mock_config, sample_task):
    """Human approval rejects 'n' input"""
    with patch("autort.AutoRTLoop.get_global_registry"):
        with patch("autort.AutoRTLoop.get_world_state"):
            with patch("builtins.input", return_value="n"):
                orchestrator = AutoRTOrchestrator()
                approved_task = orchestrator._request_approval(sample_task)

                assert approved_task is None


def test_request_approval_handles_eof(mock_config, sample_task):
    """Human approval handles EOF gracefully"""
    with patch("autort.AutoRTLoop.get_global_registry"):
        with patch("autort.AutoRTLoop.get_world_state"):
            with patch("builtins.input", side_effect=EOFError):
                orchestrator = AutoRTOrchestrator()
                approved_task = orchestrator._request_approval(sample_task)

                assert approved_task is None


# ============================================================================
# Loop Control Tests
# ============================================================================


def test_stop_stops_loop(mock_config):
    """Stop method stops the loop"""
    with patch("autort.AutoRTLoop.get_global_registry"):
        with patch("autort.AutoRTLoop.get_world_state"):
            orchestrator = AutoRTOrchestrator()
            orchestrator._running = True

            orchestrator.stop()

            assert orchestrator._running is False
