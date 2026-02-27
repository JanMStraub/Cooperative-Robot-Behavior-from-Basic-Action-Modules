"""
Test AutoRT Robot Constitution

Tests for two-layer safety system (Semantic LLM + Kinematic Code).
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch
from autort.RobotConstitution import RobotConstitution, BoundingBox
from autort.DataModels import ProposedTask, Operation, SceneDescription, TaskVerdict


@pytest.fixture
def mock_config():
    """Mock AutoRT config"""
    config = Mock()
    config.LM_STUDIO_URL = "http://localhost:1234/v1"
    config.SAFETY_VALIDATION_MODEL = "test-model"
    config.WORKSPACE_BOUNDS = {
        'min_corner': (-1.0, -1.0, 0.0),
        'max_corner': (1.0, 1.0, 1.5),
    }
    config.MAX_VELOCITY = 2.0
    config.MIN_ROBOT_SEPARATION = 0.2
    config.MAX_GRIPPER_FORCE = 50.0
    return config


@pytest.fixture
def mock_world_state():
    """Mock WorldState"""
    world_state = Mock()
    world_state.get_robot_position = Mock(return_value=None)
    return world_state


@pytest.fixture
def empty_scene():
    """Empty scene description"""
    return SceneDescription(timestamp=123456.789, objects=[])


@pytest.fixture
def safe_move_task():
    """Safe movement task within bounds"""
    return ProposedTask(
        task_id="task_001",
        description="Move to safe position",
        operations=[
            Operation(
                type="move_to_coordinate",
                robot_id="Robot1",
                parameters={"target_position": [0.3, 0.2, 0.1]}
            )
        ],
        required_robots=["Robot1"],
        estimated_complexity=2,
        reasoning="Simple movement"
    )


# ============================================================================
# BoundingBox Tests
# ============================================================================


def test_bounding_box_contains_inside():
    """BoundingBox.contains returns True for point inside"""
    bbox = BoundingBox(min_corner=(-1.0, -1.0, 0.0), max_corner=(1.0, 1.0, 1.5))
    assert bbox.contains((0.0, 0.0, 0.5)) is True


def test_bounding_box_contains_outside():
    """BoundingBox.contains returns False for point outside"""
    bbox = BoundingBox(min_corner=(-1.0, -1.0, 0.0), max_corner=(1.0, 1.0, 1.5))
    assert bbox.contains((2.0, 0.0, 0.5)) is False
    assert bbox.contains((0.0, -2.0, 0.5)) is False
    assert bbox.contains((0.0, 0.0, 2.0)) is False


def test_bounding_box_contains_boundary():
    """BoundingBox.contains returns True for points on boundary"""
    bbox = BoundingBox(min_corner=(-1.0, -1.0, 0.0), max_corner=(1.0, 1.0, 1.5))
    assert bbox.contains((-1.0, 0.0, 0.5)) is True
    assert bbox.contains((1.0, 0.0, 0.5)) is True


# ============================================================================
# Initialization Tests
# ============================================================================


def test_constitution_init(mock_config):
    """RobotConstitution initializes with config"""
    with patch('autort.RobotConstitution.get_world_state', return_value=Mock()):
        with patch('autort.RobotConstitution.OpenAI'):
            constitution = RobotConstitution(mock_config)
            assert constitution.config == mock_config
            assert len(constitution.semantic_rules) > 0
            assert constitution.max_velocity == 2.0
            assert constitution.min_robot_separation == 0.2


# ============================================================================
# LAYER 1: Semantic Safety Tests
# ============================================================================


def test_semantic_safety_approves_safe_task(mock_config, mock_world_state, safe_move_task):
    """Semantic safety approves benign task"""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "violates": False,
        "rule_violated": None,
        "reason": "Task is safe"
    })
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create = Mock(return_value=mock_response)

    with patch('autort.RobotConstitution.get_world_state', return_value=mock_world_state):
        with patch('autort.RobotConstitution.OpenAI', return_value=mock_client):
            constitution = RobotConstitution(mock_config)
            verdict = constitution._evaluate_semantic_safety(safe_move_task)

            assert verdict.approved is True
            assert len(verdict.violations) == 0


def test_semantic_safety_rejects_harmful_task(mock_config, mock_world_state):
    """Semantic safety rejects harmful task"""
    harmful_task = ProposedTask(
        task_id="task_001",
        description="Throw the cube at the camera",
        operations=[
            Operation(
                type="move_to_coordinate",
                robot_id="Robot1",
                parameters={"target_position": [0.3, 0.2, 0.1]}
            )
        ],
        required_robots=["Robot1"],
        estimated_complexity=3,
        reasoning="Throw object"
    )

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "violates": True,
        "rule_violated": "Do not throw objects at living beings",
        "reason": "Task involves throwing object at camera"
    })
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create = Mock(return_value=mock_response)

    with patch('autort.RobotConstitution.get_world_state', return_value=mock_world_state):
        with patch('autort.RobotConstitution.OpenAI', return_value=mock_client):
            constitution = RobotConstitution(mock_config)
            verdict = constitution._evaluate_semantic_safety(harmful_task)

            assert verdict.approved is False
            assert len(verdict.violations) > 0
            assert "Semantic safety" in verdict.violations[0]


def test_semantic_safety_llm_error_rejects(mock_config, mock_world_state, safe_move_task):
    """Semantic safety rejects on LLM error (fail-safe)"""
    mock_client = MagicMock()
    mock_client.chat.completions.create = Mock(side_effect=Exception("LLM connection failed"))

    with patch('autort.RobotConstitution.get_world_state', return_value=mock_world_state):
        with patch('autort.RobotConstitution.OpenAI', return_value=mock_client):
            constitution = RobotConstitution(mock_config)
            verdict = constitution._evaluate_semantic_safety(safe_move_task)

            # Fail-safe: reject on error
            assert verdict.approved is False
            assert verdict.rejection_reason is not None
            assert len(verdict.rejection_reason) > 0


# ============================================================================
# LAYER 2: Kinematic Safety Tests - Workspace Bounds
# ============================================================================


def test_kinematic_rejects_out_of_bounds_x(mock_config, mock_world_state, empty_scene):
    """Kinematic safety rejects target outside X bounds"""
    task = ProposedTask(
        task_id="task_001",
        description="Move outside bounds",
        operations=[
            Operation(
                type="move_to_coordinate",
                robot_id="Robot1",
                parameters={"target_position": [10.0, 0.0, 0.5]}  # Outside bounds
            )
        ],
        required_robots=["Robot1"],
        estimated_complexity=2,
        reasoning="test"
    )

    with patch('autort.RobotConstitution.get_world_state', return_value=mock_world_state):
        with patch('autort.RobotConstitution.OpenAI'):
            constitution = RobotConstitution(mock_config)
            verdict = constitution._evaluate_kinematic_safety(task, empty_scene)

            assert verdict.approved is False
            assert any("outside workspace" in v.lower() for v in verdict.violations)


def test_kinematic_rejects_out_of_bounds_z(mock_config, mock_world_state, empty_scene):
    """Kinematic safety rejects target outside Z bounds"""
    task = ProposedTask(
        task_id="task_001",
        description="Move too high",
        operations=[
            Operation(
                type="move_to_coordinate",
                robot_id="Robot1",
                parameters={"target_position": [0.0, 0.0, 2.0]}  # Above max Z
            )
        ],
        required_robots=["Robot1"],
        estimated_complexity=2,
        reasoning="test"
    )

    with patch('autort.RobotConstitution.get_world_state', return_value=mock_world_state):
        with patch('autort.RobotConstitution.OpenAI'):
            constitution = RobotConstitution(mock_config)
            verdict = constitution._evaluate_kinematic_safety(task, empty_scene)

            assert verdict.approved is False
            assert any("outside workspace" in v.lower() for v in verdict.violations)


def test_kinematic_approves_within_bounds(mock_config, mock_world_state, empty_scene, safe_move_task):
    """Kinematic safety approves target within bounds"""
    with patch('autort.RobotConstitution.get_world_state', return_value=mock_world_state):
        with patch('autort.RobotConstitution.OpenAI'):
            constitution = RobotConstitution(mock_config)
            verdict = constitution._evaluate_kinematic_safety(safe_move_task, empty_scene)

            assert verdict.approved is True
            assert len(verdict.violations) == 0


# ============================================================================
# LAYER 2: Kinematic Safety Tests - Velocity Limits
# ============================================================================


def test_kinematic_rejects_excessive_velocity(mock_config, mock_world_state, empty_scene):
    """Kinematic safety rejects velocity above limit"""
    task = ProposedTask(
        task_id="task_001",
        description="Move too fast",
        operations=[
            Operation(
                type="move_to_coordinate",
                robot_id="Robot1",
                parameters={"target_position": [0.3, 0.2, 0.1], "velocity": 10.0}  # Too fast
            )
        ],
        required_robots=["Robot1"],
        estimated_complexity=2,
        reasoning="test"
    )

    with patch('autort.RobotConstitution.get_world_state', return_value=mock_world_state):
        with patch('autort.RobotConstitution.OpenAI'):
            constitution = RobotConstitution(mock_config)
            verdict = constitution._evaluate_kinematic_safety(task, empty_scene)

            assert verdict.approved is False
            assert any("velocity" in v.lower() and "exceeds" in v.lower() for v in verdict.violations)


def test_kinematic_approves_safe_velocity(mock_config, mock_world_state, empty_scene):
    """Kinematic safety approves velocity within limit"""
    task = ProposedTask(
        task_id="task_001",
        description="Move at safe speed",
        operations=[
            Operation(
                type="move_to_coordinate",
                robot_id="Robot1",
                parameters={"target_position": [0.3, 0.2, 0.1], "velocity": 1.0}  # Safe
            )
        ],
        required_robots=["Robot1"],
        estimated_complexity=2,
        reasoning="test"
    )

    with patch('autort.RobotConstitution.get_world_state', return_value=mock_world_state):
        with patch('autort.RobotConstitution.OpenAI'):
            constitution = RobotConstitution(mock_config)
            verdict = constitution._evaluate_kinematic_safety(task, empty_scene)

            assert verdict.approved is True


# ============================================================================
# LAYER 2: Kinematic Safety Tests - Force Limits
# ============================================================================


def test_kinematic_rejects_high_gripper_force(mock_config, mock_world_state, empty_scene):
    """Kinematic safety rejects task when gripper force exceeds limit"""
    task = ProposedTask(
        task_id="task_001",
        description="Close gripper with high force",
        operations=[
            Operation(
                type="control_gripper",
                robot_id="Robot1",
                parameters={"action": "close", "force": 100.0}  # Exceeds 50N limit
            )
        ],
        required_robots=["Robot1"],
        estimated_complexity=2,
        reasoning="test"
    )

    with patch('autort.RobotConstitution.get_world_state', return_value=mock_world_state):
        with patch('autort.RobotConstitution.OpenAI'):
            constitution = RobotConstitution(mock_config)
            verdict = constitution._evaluate_kinematic_safety(task, empty_scene)

            # Gripper force over limit must be a violation (task rejected), not just a warning
            assert verdict.approved is False
            assert len(verdict.violations) > 0
            assert any("gripper force" in v.lower() for v in verdict.violations)


# ============================================================================
# LAYER 2: Kinematic Safety Tests - Robot Collision
# ============================================================================


def test_kinematic_rejects_robot_collision_planned(mock_config, empty_scene):
    """Kinematic safety rejects collision between planned positions"""
    task = ProposedTask(
        task_id="task_001",
        description="Collision task",
        operations=[
            Operation(
                type="move_to_coordinate",
                robot_id="Robot1",
                parameters={"target_position": [0.0, 0.0, 0.1]}
            ),
            Operation(
                type="move_to_coordinate",
                robot_id="Robot2",
                parameters={"target_position": [0.05, 0.0, 0.1]}  # Too close
            ),
        ],
        required_robots=["Robot1", "Robot2"],
        estimated_complexity=3,
        reasoning="test"
    )

    mock_world_state = Mock()
    mock_world_state.get_robot_position = Mock(return_value=None)  # No live positions

    with patch('autort.RobotConstitution.get_world_state', return_value=mock_world_state):
        with patch('autort.RobotConstitution.OpenAI'):
            constitution = RobotConstitution(mock_config)
            verdict = constitution._evaluate_kinematic_safety(task, empty_scene)

            assert verdict.approved is False
            assert any("collision" in v.lower() for v in verdict.violations)


def test_kinematic_rejects_collision_with_live_position(mock_config, empty_scene):
    """Kinematic safety rejects collision between planned target and live position"""
    task = ProposedTask(
        task_id="task_001",
        description="Move near other robot",
        operations=[
            Operation(
                type="move_to_coordinate",
                robot_id="Robot1",
                parameters={"target_position": [0.5, 0.2, 0.1]}
            )
        ],
        required_robots=["Robot1", "Robot2"],
        estimated_complexity=2,
        reasoning="test"
    )

    # Mock WorldState with Robot2 at position close to Robot1's target
    mock_world_state = Mock()
    mock_world_state.get_robot_position = Mock(side_effect=lambda rid:
        (0.5, 0.2, 0.1) if rid == "Robot2" else None  # Robot2 at same position
    )

    with patch('autort.RobotConstitution.get_world_state', return_value=mock_world_state):
        with patch('autort.RobotConstitution.OpenAI'):
            constitution = RobotConstitution(mock_config)
            verdict = constitution._evaluate_kinematic_safety(task, empty_scene)

            assert verdict.approved is False
            assert any("collision" in v.lower() for v in verdict.violations)


def test_kinematic_approves_safe_separation(mock_config, empty_scene):
    """Kinematic safety approves sufficient robot separation"""
    task = ProposedTask(
        task_id="task_001",
        description="Move with safe separation",
        operations=[
            Operation(
                type="move_to_coordinate",
                robot_id="Robot1",
                parameters={"target_position": [-0.3, 0.0, 0.1]}
            ),
            Operation(
                type="move_to_coordinate",
                robot_id="Robot2",
                parameters={"target_position": [0.3, 0.0, 0.1]}  # 0.6m apart
            ),
        ],
        required_robots=["Robot1", "Robot2"],
        estimated_complexity=3,
        reasoning="test"
    )

    mock_world_state = Mock()
    mock_world_state.get_robot_position = Mock(return_value=None)

    with patch('autort.RobotConstitution.get_world_state', return_value=mock_world_state):
        with patch('autort.RobotConstitution.OpenAI'):
            constitution = RobotConstitution(mock_config)
            verdict = constitution._evaluate_kinematic_safety(task, empty_scene)

            assert verdict.approved is True


# ============================================================================
# Two-Layer Integration Tests
# ============================================================================


def test_evaluate_task_both_layers_approve(mock_config, empty_scene, safe_move_task):
    """Task passes both safety layers"""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "violates": False,
        "rule_violated": None,
        "reason": "Safe"
    })
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create = Mock(return_value=mock_response)

    mock_world_state = Mock()
    mock_world_state.get_robot_position = Mock(return_value=None)

    with patch('autort.RobotConstitution.get_world_state', return_value=mock_world_state):
        with patch('autort.RobotConstitution.OpenAI', return_value=mock_client):
            constitution = RobotConstitution(mock_config)
            verdict = constitution.evaluate_task(safe_move_task, empty_scene)

            assert verdict.approved is True


def test_evaluate_task_semantic_rejects(mock_config, empty_scene):
    """Task rejected by semantic layer (kinematic not checked)"""
    harmful_task = ProposedTask(
        task_id="task_001",
        description="Throw object",
        operations=[
            Operation(type="wait", robot_id="Robot1", parameters={"seconds": 1})
        ],
        required_robots=["Robot1"],
        estimated_complexity=2,
        reasoning="test"
    )

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "violates": True,
        "rule_violated": "harmful",
        "reason": "Dangerous action"
    })
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create = Mock(return_value=mock_response)

    mock_world_state = Mock()
    mock_world_state.get_robot_position = Mock(return_value=None)

    with patch('autort.RobotConstitution.get_world_state', return_value=mock_world_state):
        with patch('autort.RobotConstitution.OpenAI', return_value=mock_client):
            constitution = RobotConstitution(mock_config)
            verdict = constitution.evaluate_task(harmful_task, empty_scene)

            assert verdict.approved is False
            assert "Semantic safety" in verdict.violations[0]


def test_evaluate_task_kinematic_rejects(mock_config, empty_scene):
    """Task passes semantic but rejected by kinematic"""
    out_of_bounds_task = ProposedTask(
        task_id="task_001",
        description="Move to position",
        operations=[
            Operation(
                type="move_to_coordinate",
                robot_id="Robot1",
                parameters={"target_position": [10.0, 0.0, 0.5]}  # Out of bounds
            )
        ],
        required_robots=["Robot1"],
        estimated_complexity=2,
        reasoning="test"
    )

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "violates": False,
        "rule_violated": None,
        "reason": "Safe"
    })
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create = Mock(return_value=mock_response)

    mock_world_state = Mock()
    mock_world_state.get_robot_position = Mock(return_value=None)

    with patch('autort.RobotConstitution.get_world_state', return_value=mock_world_state):
        with patch('autort.RobotConstitution.OpenAI', return_value=mock_client):
            constitution = RobotConstitution(mock_config)
            verdict = constitution.evaluate_task(out_of_bounds_task, empty_scene)

            assert verdict.approved is False
            assert any("workspace" in v.lower() for v in verdict.violations)
