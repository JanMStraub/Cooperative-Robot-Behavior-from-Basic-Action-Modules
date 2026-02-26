#!/usr/bin/env python3
"""
Unit tests for IntermediateOperations.py

Tests the intermediate-level manipulation operations including:
- grip_object: Simplified grasping wrapper
- align_object: Object/gripper alignment
- follow_path: Multi-waypoint trajectory following
- draw_with_pen: Tool manipulation for drawing
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch

from operations.IntermediateOperations import (
    grip_object,
    align_object,
    follow_path,
    draw_with_pen,
    GRIP_OBJECT_OPERATION,
    ALIGN_OBJECT_OPERATION,
    FOLLOW_PATH_OPERATION,
    DRAW_WITH_PEN_OPERATION,
)
from operations.Base import OperationResult


# ============================================================================
# Test Class: grip_object - Simplified Grasping
# ============================================================================


class TestGripObject:
    """Test simplified grasping operation."""

    def test_grip_object_success(self, patch_command_broadcaster):
        """Test successful grip operation."""
        object_pos = {"x": 0.3, "y": 0.15, "z": 0.1}
        result = grip_object("Robot1", object_pos)

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "Robot1"
        assert result.result["object_position"] == object_pos
        assert result.result["approach_direction"] == "top"  # Default
        assert result.result["status"] == "command_sent"
        patch_command_broadcaster.send_command.assert_called_once()

    def test_grip_object_with_approach_direction(self, patch_command_broadcaster):
        """Test grip with custom approach direction."""
        object_pos = {"x": 0.3, "y": 0.15, "z": 0.1}

        for direction in ["top", "front", "side"]:
            result = grip_object("Robot1", object_pos, approach_direction=direction)
            assert result.success is True
            assert result.result is not None
            assert result.result["approach_direction"] == direction

    def test_grip_object_invalid_robot_id(self):
        """Test grip with invalid robot ID."""
        object_pos = {"x": 0.3, "y": 0.15, "z": 0.1}
        result = grip_object("", object_pos)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_grip_object_invalid_position_not_dict(self):
        """Test grip with non-dictionary position."""
        result = grip_object("Robot1", "invalid_position")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_OBJECT_POSITION"

    def test_grip_object_invalid_position_missing_keys(self):
        """Test grip with position missing required keys."""
        # Missing 'z'
        result = grip_object("Robot1", {"x": 0.3, "y": 0.15})

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_OBJECT_POSITION"

    def test_grip_object_invalid_approach_direction(self):
        """Test grip with invalid approach direction."""
        object_pos = {"x": 0.3, "y": 0.15, "z": 0.1}
        result = grip_object("Robot1", object_pos, approach_direction="invalid")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_APPROACH_DIRECTION"

    def test_grip_object_command_structure(self, patch_command_broadcaster):
        """Test that grip command has correct structure."""
        object_pos = {"x": 0.3, "y": 0.15, "z": 0.1}
        result = grip_object("Robot1", object_pos, approach_direction="side", request_id=111)

        patch_command_broadcaster.send_command.assert_called_once()
        call_args = patch_command_broadcaster.send_command.call_args

        command = call_args[0][0]
        assert command["command_type"] == "grip_object"
        assert command["robot_id"] == "Robot1"
        assert command["parameters"]["object_position"] == object_pos
        assert command["parameters"]["approach_direction"] == "side"
        assert "timestamp" in command

        request_id = call_args[0][1]
        assert request_id == 111

    def test_grip_object_communication_failed(self, patch_command_broadcaster):
        """Test grip when communication fails."""
        patch_command_broadcaster.send_command = Mock(return_value=False)
        object_pos = {"x": 0.3, "y": 0.15, "z": 0.1}

        result = grip_object("Robot1", object_pos)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "COMMUNICATION_FAILED"


# ============================================================================
# Test Class: align_object - Object/Gripper Alignment
# ============================================================================


class TestAlignObject:
    """Test object/gripper alignment operation."""

    def test_align_object_success(self, patch_command_broadcaster):
        """Test successful alignment operation."""
        orientation = {"roll": 0, "pitch": -90, "yaw": 0}
        result = align_object("Robot1", orientation)

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "Robot1"
        assert result.result["target_orientation"] == orientation
        assert result.result["alignment_type"] == "gripper"  # Default
        assert result.result["status"] == "command_sent"
        patch_command_broadcaster.send_command.assert_called_once()

    def test_align_object_with_alignment_type(self, patch_command_broadcaster):
        """Test alignment with different alignment types."""
        orientation = {"roll": 0, "pitch": 0, "yaw": 90}

        for align_type in ["gripper", "object"]:
            result = align_object("Robot1", orientation, alignment_type=align_type)
            assert result.success is True
            assert result.result is not None
            assert result.result["alignment_type"] == align_type

    def test_align_object_invalid_robot_id(self):
        """Test alignment with invalid robot ID."""
        orientation = {"roll": 0, "pitch": -90, "yaw": 0}
        result = align_object("", orientation)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_align_object_invalid_orientation_not_dict(self):
        """Test alignment with non-dictionary orientation."""
        result = align_object("Robot1", "invalid_orientation")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ORIENTATION"

    def test_align_object_invalid_orientation_missing_keys(self):
        """Test alignment with orientation missing required keys."""
        # Missing 'yaw'
        result = align_object("Robot1", {"roll": 0, "pitch": -90})

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ORIENTATION"

    def test_align_object_invalid_alignment_type(self):
        """Test alignment with invalid alignment type."""
        orientation = {"roll": 0, "pitch": -90, "yaw": 0}
        result = align_object("Robot1", orientation, alignment_type="invalid")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ALIGNMENT_TYPE"

    def test_align_object_command_structure(self, patch_command_broadcaster):
        """Test that align command has correct structure."""
        orientation = {"roll": 45, "pitch": -45, "yaw": 90}
        result = align_object("Robot1", orientation, alignment_type="object", request_id=222)

        patch_command_broadcaster.send_command.assert_called_once()
        call_args = patch_command_broadcaster.send_command.call_args

        command = call_args[0][0]
        assert command["command_type"] == "align_object"
        assert command["robot_id"] == "Robot1"
        assert command["parameters"]["target_orientation"] == orientation
        assert command["parameters"]["alignment_type"] == "object"
        assert "timestamp" in command

        request_id = call_args[0][1]
        assert request_id == 222

    def test_align_object_communication_failed(self, patch_command_broadcaster):
        """Test alignment when communication fails."""
        patch_command_broadcaster.send_command = Mock(return_value=False)
        orientation = {"roll": 0, "pitch": -90, "yaw": 0}

        result = align_object("Robot1", orientation)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "COMMUNICATION_FAILED"


# ============================================================================
# Test Class: follow_path - Multi-Waypoint Trajectory
# ============================================================================


class TestFollowPath:
    """Test multi-waypoint path following operation."""

    def test_follow_path_success(self, patch_command_broadcaster):
        """Test successful path following."""
        waypoints = [
            {"x": 0.0, "y": 0.0, "z": 0.3},
            {"x": 0.2, "y": 0.1, "z": 0.2},
            {"x": 0.3, "y": 0.15, "z": 0.1},
        ]
        result = follow_path("Robot1", waypoints)

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "Robot1"
        assert result.result["waypoints"] == waypoints
        assert result.result["waypoint_count"] == 3
        assert result.result["speed"] == 1.0  # Default
        assert result.result["status"] == "command_sent"
        patch_command_broadcaster.send_command.assert_called_once()

    def test_follow_path_with_custom_speed(self, patch_command_broadcaster):
        """Test path following with custom speed."""
        waypoints = [
            {"x": 0.0, "y": 0.0, "z": 0.3},
            {"x": 0.3, "y": 0.15, "z": 0.1},
        ]
        result = follow_path("Robot1", waypoints, speed=0.5)

        assert result.success is True
        assert result.result is not None
        assert result.result["speed"] == 0.5

    def test_follow_path_two_waypoints(self, patch_command_broadcaster):
        """Test path with minimum waypoints (2)."""
        waypoints = [
            {"x": 0.0, "y": 0.0, "z": 0.3},
            {"x": 0.3, "y": 0.15, "z": 0.1},
        ]
        result = follow_path("Robot1", waypoints)

        assert result.success is True
        assert result.result is not None
        assert result.result["waypoint_count"] == 2

    def test_follow_path_many_waypoints(self, patch_command_broadcaster):
        """Test path with many waypoints."""
        waypoints = [{"x": i * 0.05, "y": 0.0, "z": 0.1 + i * 0.02} for i in range(10)]
        result = follow_path("Robot1", waypoints)

        assert result.success is True
        assert result.result is not None
        assert result.result["waypoint_count"] == 10

    def test_follow_path_invalid_robot_id(self):
        """Test path following with invalid robot ID."""
        waypoints = [
            {"x": 0.0, "y": 0.0, "z": 0.3},
            {"x": 0.3, "y": 0.15, "z": 0.1},
        ]
        result = follow_path("", waypoints)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_follow_path_not_a_list(self):
        """Test path following with non-list waypoints."""
        result = follow_path("Robot1", "invalid_waypoints")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_WAYPOINTS"

    def test_follow_path_too_few_waypoints(self):
        """Test path following with too few waypoints."""
        waypoints = [{"x": 0.0, "y": 0.0, "z": 0.3}]  # Only 1 waypoint
        result = follow_path("Robot1", waypoints)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_WAYPOINTS"

    def test_follow_path_invalid_waypoint_structure(self):
        """Test path following with invalid waypoint structure."""
        waypoints = [
            {"x": 0.0, "y": 0.0, "z": 0.3},
            {"x": 0.2, "y": 0.1},  # Missing 'z'
        ]
        result = follow_path("Robot1", waypoints)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_WAYPOINT"

    def test_follow_path_invalid_speed_too_low(self):
        """Test path following with speed below minimum."""
        waypoints = [
            {"x": 0.0, "y": 0.0, "z": 0.3},
            {"x": 0.3, "y": 0.15, "z": 0.1},
        ]
        result = follow_path("Robot1", waypoints, speed=0.05)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_SPEED"

    def test_follow_path_invalid_speed_too_high(self):
        """Test path following with speed above maximum."""
        waypoints = [
            {"x": 0.0, "y": 0.0, "z": 0.3},
            {"x": 0.3, "y": 0.15, "z": 0.1},
        ]
        result = follow_path("Robot1", waypoints, speed=3.0)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_SPEED"

    def test_follow_path_command_structure(self, patch_command_broadcaster):
        """Test that follow_path command has correct structure."""
        waypoints = [
            {"x": 0.0, "y": 0.0, "z": 0.3},
            {"x": 0.2, "y": 0.1, "z": 0.2},
            {"x": 0.3, "y": 0.15, "z": 0.1},
        ]
        result = follow_path("Robot1", waypoints, speed=1.5, request_id=333)

        patch_command_broadcaster.send_command.assert_called_once()
        call_args = patch_command_broadcaster.send_command.call_args

        command = call_args[0][0]
        assert command["command_type"] == "follow_path"
        assert command["robot_id"] == "Robot1"
        assert command["parameters"]["waypoints"] == waypoints
        assert command["parameters"]["speed_multiplier"] == 1.5
        assert "timestamp" in command

        request_id = call_args[0][1]
        assert request_id == 333

    def test_follow_path_communication_failed(self, patch_command_broadcaster):
        """Test path following when communication fails."""
        patch_command_broadcaster.send_command = Mock(return_value=False)
        waypoints = [
            {"x": 0.0, "y": 0.0, "z": 0.3},
            {"x": 0.3, "y": 0.15, "z": 0.1},
        ]

        result = follow_path("Robot1", waypoints)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "COMMUNICATION_FAILED"


# ============================================================================
# Test Class: draw_with_pen - Tool Manipulation
# ============================================================================


class TestDrawWithPen:
    """Test pen drawing operation."""

    def test_draw_with_pen_success(self, patch_command_broadcaster):
        """Test successful drawing operation."""
        pen_pos = {"x": 0.2, "y": 0.0, "z": 0.3}
        paper_pos = {"x": 0.3, "y": 0.0, "z": 0.0}

        result = draw_with_pen("Robot1", pen_pos, paper_pos)

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "Robot1"
        assert result.result["pen_position"] == pen_pos
        assert result.result["paper_position"] == paper_pos
        assert result.result["shape"] == "line"  # Default
        assert result.result["status"] == "command_sent"
        patch_command_broadcaster.send_command.assert_called_once()

    def test_draw_with_pen_all_shapes(self, patch_command_broadcaster):
        """Test drawing with all supported shapes."""
        pen_pos = {"x": 0.2, "y": 0.0, "z": 0.3}
        paper_pos = {"x": 0.3, "y": 0.0, "z": 0.0}

        for shape in ["line", "circle", "square", "custom"]:
            result = draw_with_pen("Robot1", pen_pos, paper_pos, shape=shape)
            assert result.success is True
            assert result.result is not None
            assert result.result["shape"] == shape

    def test_draw_with_pen_shape_params(self, patch_command_broadcaster):
        """Test drawing with shape-specific parameters."""
        pen_pos = {"x": 0.2, "y": 0.0, "z": 0.3}
        paper_pos = {"x": 0.3, "y": 0.0, "z": 0.0}

        # Line parameters
        line_params = {"length": 0.1, "angle": 45}
        result = draw_with_pen("Robot1", pen_pos, paper_pos, shape="line", shape_params=line_params)
        assert result.success is True
        assert result.result is not None
        assert result.result["shape_params"] == line_params

        # Circle parameters
        circle_params = {"radius": 0.05}
        result = draw_with_pen(
            "Robot1", pen_pos, paper_pos, shape="circle", shape_params=circle_params
        )
        assert result.success is True
        assert result.result is not None
        assert result.result["shape_params"] == circle_params

    def test_draw_with_pen_invalid_robot_id(self):
        """Test drawing with invalid robot ID."""
        pen_pos = {"x": 0.2, "y": 0.0, "z": 0.3}
        paper_pos = {"x": 0.3, "y": 0.0, "z": 0.0}

        result = draw_with_pen("", pen_pos, paper_pos)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_draw_with_pen_invalid_pen_position(self):
        """Test drawing with invalid pen position."""
        paper_pos = {"x": 0.3, "y": 0.0, "z": 0.0}

        # Missing 'z'
        result = draw_with_pen("Robot1", {"x": 0.2, "y": 0.0}, paper_pos)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_PEN_POSITION"

    def test_draw_with_pen_invalid_paper_position(self):
        """Test drawing with invalid paper position."""
        pen_pos = {"x": 0.2, "y": 0.0, "z": 0.3}

        # Missing 'z'
        result = draw_with_pen("Robot1", pen_pos, {"x": 0.3, "y": 0.0})

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_PAPER_POSITION"

    def test_draw_with_pen_invalid_shape(self):
        """Test drawing with invalid shape."""
        pen_pos = {"x": 0.2, "y": 0.0, "z": 0.3}
        paper_pos = {"x": 0.3, "y": 0.0, "z": 0.0}

        result = draw_with_pen("Robot1", pen_pos, paper_pos, shape="invalid_shape")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_SHAPE"

    def test_draw_with_pen_command_structure(self, patch_command_broadcaster):
        """Test that draw_with_pen command has correct structure."""
        pen_pos = {"x": 0.2, "y": 0.0, "z": 0.3}
        paper_pos = {"x": 0.3, "y": 0.0, "z": 0.0}
        shape_params = {"radius": 0.05}

        result = draw_with_pen(
            "Robot1", pen_pos, paper_pos, shape="circle", shape_params=shape_params, request_id=444
        )

        patch_command_broadcaster.send_command.assert_called_once()
        call_args = patch_command_broadcaster.send_command.call_args

        command = call_args[0][0]
        assert command["command_type"] == "draw_with_pen"
        assert command["robot_id"] == "Robot1"
        assert command["parameters"]["pen_position"] == pen_pos
        assert command["parameters"]["paper_position"] == paper_pos
        assert command["parameters"]["shape"] == "circle"
        assert command["parameters"]["shape_params"] == shape_params
        assert "timestamp" in command

        request_id = call_args[0][1]
        assert request_id == 444

    def test_draw_with_pen_communication_failed(self, patch_command_broadcaster):
        """Test drawing when communication fails."""
        patch_command_broadcaster.send_command = Mock(return_value=False)
        pen_pos = {"x": 0.2, "y": 0.0, "z": 0.3}
        paper_pos = {"x": 0.3, "y": 0.0, "z": 0.0}

        result = draw_with_pen("Robot1", pen_pos, paper_pos)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "COMMUNICATION_FAILED"


# ============================================================================
# Test Class: Operation Definitions
# ============================================================================


class TestIntermediateOperationDefinitions:
    """Test BasicOperation definitions for intermediate operations."""

    def test_grip_object_operation_definition(self):
        """Test GRIP_OBJECT_OPERATION is properly defined."""
        assert GRIP_OBJECT_OPERATION is not None
        assert GRIP_OBJECT_OPERATION.name == "grip_object"
        assert GRIP_OBJECT_OPERATION.operation_id == "manipulation_grip_object_003"

    def test_align_object_operation_definition(self):
        """Test ALIGN_OBJECT_OPERATION is properly defined."""
        assert ALIGN_OBJECT_OPERATION is not None
        assert ALIGN_OBJECT_OPERATION.name == "align_object"
        assert ALIGN_OBJECT_OPERATION.operation_id == "manipulation_align_object_004"

    def test_follow_path_operation_definition(self):
        """Test FOLLOW_PATH_OPERATION is properly defined."""
        assert FOLLOW_PATH_OPERATION is not None
        assert FOLLOW_PATH_OPERATION.name == "follow_path"
        assert FOLLOW_PATH_OPERATION.operation_id == "navigation_follow_path_004"

    def test_draw_with_pen_operation_definition(self):
        """Test DRAW_WITH_PEN_OPERATION is properly defined."""
        assert DRAW_WITH_PEN_OPERATION is not None
        assert DRAW_WITH_PEN_OPERATION.name == "draw_with_pen"
        assert DRAW_WITH_PEN_OPERATION.operation_id == "manipulation_draw_with_pen_005"

    def test_all_operations_have_metadata(self):
        """Test all operations have required metadata."""
        operations = [
            GRIP_OBJECT_OPERATION,
            ALIGN_OBJECT_OPERATION,
            FOLLOW_PATH_OPERATION,
            DRAW_WITH_PEN_OPERATION,
        ]

        for op in operations:
            assert op.description is not None
            assert op.parameters is not None
            assert len(op.parameters) >= 1
            assert op.implementation is not None

    def test_operation_execution_through_definition(self, patch_command_broadcaster):
        """Test executing operations through BasicOperation.execute()."""
        # grip_object
        result = GRIP_OBJECT_OPERATION.execute(
            robot_id="Robot1", object_position={"x": 0.3, "y": 0.15, "z": 0.1}
        )
        assert result.success is True

        # align_object
        result = ALIGN_OBJECT_OPERATION.execute(
            robot_id="Robot1", target_orientation={"roll": 0, "pitch": -90, "yaw": 0}
        )
        assert result.success is True

        # follow_path
        result = FOLLOW_PATH_OPERATION.execute(
            robot_id="Robot1",
            waypoints=[
                {"x": 0.0, "y": 0.0, "z": 0.3},
                {"x": 0.3, "y": 0.15, "z": 0.1},
            ],
        )
        assert result.success is True

        # draw_with_pen
        result = DRAW_WITH_PEN_OPERATION.execute(
            robot_id="Robot1",
            pen_position={"x": 0.2, "y": 0.0, "z": 0.3},
            paper_position={"x": 0.3, "y": 0.0, "z": 0.0},
        )
        assert result.success is True


# ============================================================================
# Test Class: Concurrent Execution
# ============================================================================


class TestIntermediateOperationsConcurrency:
    """Test thread safety for intermediate operations."""

    def test_concurrent_grip_operations(self, patch_command_broadcaster):
        """Test concurrent grip operations."""
        import threading

        results = []

        def grip_worker(robot_id):
            result = grip_object(robot_id, {"x": 0.3, "y": 0.15, "z": 0.1})
            results.append(result)

        threads = [threading.Thread(target=grip_worker, args=(f"Robot{i}",)) for i in range(1, 4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_concurrent_path_following(self, patch_command_broadcaster):
        """Test concurrent path following operations."""
        import threading

        results = []
        waypoints = [
            {"x": 0.0, "y": 0.0, "z": 0.3},
            {"x": 0.3, "y": 0.15, "z": 0.1},
        ]

        def path_worker(robot_id):
            result = follow_path(robot_id, waypoints)
            results.append(result)

        threads = [threading.Thread(target=path_worker, args=(f"Robot{i}",)) for i in range(1, 4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 3
        assert all(r.success for r in results)
