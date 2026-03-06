"""
Test AutoRT Data Models

Tests for Pydantic models in autort/DataModels.py
"""

import pytest
from pydantic import ValidationError
from autort.DataModels import (
    GroundedObject,
    SceneDescription,
    Operation,
    ProposedTask,
    TaskVerdict,
)


# ============================================================================
# GroundedObject Tests
# ============================================================================


def test_grounded_object_valid():
    """GroundedObject accepts valid data"""
    obj = GroundedObject(
        object_id="cube_01",
        color="red",
        position=(0.3, 0.2, 0.1),
        confidence=0.95,
        graspable=True,
    )
    assert obj.object_id == "cube_01"
    assert obj.color == "red"
    assert obj.position == (0.3, 0.2, 0.1)
    assert obj.confidence == 0.95
    assert obj.graspable is True


def test_grounded_object_confidence_bounds():
    """GroundedObject rejects confidence outside [0, 1]"""
    with pytest.raises(ValidationError):
        GroundedObject(
            object_id="cube_01",
            color="red",
            position=(0.3, 0.2, 0.1),
            confidence=1.5,  # Invalid
        )

    with pytest.raises(ValidationError):
        GroundedObject(
            object_id="cube_01",
            color="red",
            position=(0.3, 0.2, 0.1),
            confidence=-0.1,  # Invalid
        )


def test_grounded_object_forbids_extra_fields():
    """GroundedObject rejects extra fields"""
    with pytest.raises(ValidationError):
        GroundedObject(
            object_id="cube_01",
            color="red",
            position=(0.3, 0.2, 0.1),
            confidence=0.95,
            extra_field="should_fail",  # type: ignore[call-arg]  # Extra field
        )


# ============================================================================
# SceneDescription Tests
# ============================================================================


def test_scene_description_valid():
    """SceneDescription accepts valid data"""
    obj = GroundedObject(
        object_id="cube_01",
        color="red",
        position=(0.3, 0.2, 0.1),
        confidence=0.95,
    )
    scene = SceneDescription(
        timestamp=123456.789,
        objects=[obj],
        scene_summary="One red cube on table",
        robot_states={"Robot1": {"position": (0.0, 0.0, 0.0)}},
    )
    assert scene.timestamp == 123456.789
    assert len(scene.objects) == 1
    assert scene.scene_summary == "One red cube on table"
    assert "Robot1" in scene.robot_states


def test_scene_description_empty_defaults():
    """SceneDescription allows empty objects and defaults"""
    scene = SceneDescription(timestamp=123456.789, objects=[])
    assert len(scene.objects) == 0
    assert scene.scene_summary == ""
    assert len(scene.robot_states) == 0


# ============================================================================
# Operation Tests
# ============================================================================


def test_operation_valid():
    """Operation accepts valid data"""
    op = Operation(
        type="move_to_coordinate",
        robot_id="Robot1",
        parameters={"x": 0.3, "y": 0.2, "z": 0.1},
    )
    assert op.type == "move_to_coordinate"
    assert op.robot_id == "Robot1"
    assert op.parameters["x"] == 0.3


def test_operation_empty_parameters():
    """Operation allows empty parameters dict"""
    op = Operation(type="wait", robot_id="Robot1", parameters={})
    assert op.parameters == {}


# ============================================================================
# ProposedTask Tests
# ============================================================================


def test_proposed_task_valid():
    """ProposedTask accepts valid data"""
    task = ProposedTask(
        task_id="task_001",
        description="Pick up red cube",
        operations=[
            Operation(
                type="detect_object_stereo",
                robot_id="Robot1",
                parameters={"color": "red"},
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
    assert task.task_id == "task_001"
    assert len(task.operations) == 2
    assert task.required_robots == ["Robot1"]
    assert task.estimated_complexity == 3


def test_proposed_task_rejects_empty_operations():
    """ProposedTask rejects empty operations list"""
    with pytest.raises(ValidationError):
        ProposedTask(
            task_id="task_001",
            description="test",
            operations=[],  # Empty list not allowed
            required_robots=["Robot1"],
            estimated_complexity=3,
            reasoning="test",
        )


def test_proposed_task_rejects_empty_required_robots():
    """ProposedTask rejects empty required_robots list"""
    with pytest.raises(ValidationError):
        ProposedTask(
            task_id="task_001",
            description="test",
            operations=[
                Operation(type="wait", robot_id="Robot1", parameters={"seconds": 1})
            ],
            required_robots=[],  # Empty list not allowed
            estimated_complexity=3,
            reasoning="test",
        )


def test_proposed_task_validates_robot_ids_consistent():
    """ProposedTask rejects operations with robot_ids not in required_robots"""
    with pytest.raises(ValidationError, match="not in required_robots"):
        ProposedTask(
            task_id="task_001",
            description="test",
            operations=[
                Operation(
                    type="move_to_coordinate",
                    robot_id="Robot2",  # Not in required_robots
                    parameters={"x": 0.3, "y": 0.2, "z": 0.1},
                )
            ],
            required_robots=["Robot1"],  # Only Robot1 listed
            estimated_complexity=3,
            reasoning="test",
        )


def test_proposed_task_complexity_bounds():
    """ProposedTask validates complexity is between 1 and 10"""
    with pytest.raises(ValidationError):
        ProposedTask(
            task_id="task_001",
            description="test",
            operations=[
                Operation(type="wait", robot_id="Robot1", parameters={"seconds": 1})
            ],
            required_robots=["Robot1"],
            estimated_complexity=0,  # Too low
            reasoning="test",
        )

    with pytest.raises(ValidationError):
        ProposedTask(
            task_id="task_001",
            description="test",
            operations=[
                Operation(type="wait", robot_id="Robot1", parameters={"seconds": 1})
            ],
            required_robots=["Robot1"],
            estimated_complexity=11,  # Too high
            reasoning="test",
        )


def test_proposed_task_multi_robot():
    """ProposedTask accepts multi-robot tasks with consistent robot_ids"""
    task = ProposedTask(
        task_id="task_001",
        description="Collaborative handoff",
        operations=[
            Operation(
                type="move_to_coordinate",
                robot_id="Robot1",
                parameters={"x": 0.0, "y": 0.0, "z": 0.1},
            ),
            Operation(type="signal", robot_id="Robot1", parameters={"event": "ready"}),
            Operation(
                type="wait_for_signal",
                robot_id="Robot2",
                parameters={"event": "ready"},
            ),
            Operation(
                type="move_to_coordinate",
                robot_id="Robot2",
                parameters={"x": 0.0, "y": 0.0, "z": 0.1},
            ),
        ],
        required_robots=["Robot1", "Robot2"],
        estimated_complexity=5,
        reasoning="Handoff coordination",
    )
    assert len(task.operations) == 4
    assert set(task.required_robots) == {"Robot1", "Robot2"}


def test_proposed_task_forbids_extra_fields():
    """ProposedTask rejects extra fields"""
    with pytest.raises(ValidationError):
        ProposedTask(
            task_id="task_001",
            description="test",
            operations=[
                Operation(type="wait", robot_id="Robot1", parameters={"seconds": 1})
            ],
            required_robots=["Robot1"],
            estimated_complexity=3,
            reasoning="test",
            extra_field="should_fail",  # type: ignore[call-arg]  # Extra field
        )


# ============================================================================
# TaskVerdict Tests
# ============================================================================


def test_task_verdict_approved():
    """TaskVerdict for approved task"""
    verdict = TaskVerdict(approved=True, violations=[], warnings=[])
    assert verdict.approved is True
    assert len(verdict.violations) == 0
    assert len(verdict.warnings) == 0
    assert verdict.rejection_reason is None


def test_task_verdict_rejected():
    """TaskVerdict for rejected task"""
    verdict = TaskVerdict(
        approved=False,
        violations=["Out of bounds"],
        rejection_reason="Target outside workspace",
    )
    assert verdict.approved is False
    assert len(verdict.violations) == 1
    assert verdict.rejection_reason == "Target outside workspace"


def test_task_verdict_with_warnings():
    """TaskVerdict can have warnings even when approved"""
    verdict = TaskVerdict(
        approved=True,
        violations=[],
        warnings=["High gripper force"],
    )
    assert verdict.approved is True
    assert len(verdict.warnings) == 1
    assert verdict.warnings[0] == "High gripper force"


def test_task_verdict_with_modified_task():
    """TaskVerdict can include modified task"""
    original_task = ProposedTask(
        task_id="task_001",
        description="test",
        operations=[
            Operation(type="wait", robot_id="Robot1", parameters={"seconds": 1})
        ],
        required_robots=["Robot1"],
        estimated_complexity=3,
        reasoning="test",
    )

    verdict = TaskVerdict(
        approved=True,
        violations=[],
        warnings=["Task was modified"],
        modified_task=original_task,
    )
    assert verdict.approved is True
    assert verdict.modified_task is not None
    assert verdict.modified_task.task_id == "task_001"
