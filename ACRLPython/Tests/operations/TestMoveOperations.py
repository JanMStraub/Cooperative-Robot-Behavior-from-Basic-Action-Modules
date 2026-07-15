import pytest
from unittest.mock import Mock, MagicMock, patch

from operations.MoveOperations import (
    move_to_coordinate,
    adjust_end_effector_orientation,
    MOVE_TO_COORDINATE_OPERATION,
)

# Test Class: Basic Movement Operations


class TestMoveOperations:

    def test_move_to_coordinate_success(self, patch_command_broadcaster):

        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=0.1)

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "Robot1"
        assert result.result["target_position"]["x"] == 0.3
        assert result.result["target_position"]["y"] == 0.2
        assert result.result["target_position"]["z"] == 0.1
        assert result.result["status"] == "command_sent"
        patch_command_broadcaster.send_command.assert_called_once()

    def test_move_with_speed_parameter(self, patch_command_broadcaster):

        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=0.1, speed=0.5)

        assert result.success is True
        assert result.result is not None
        assert result.result["speed"] == 0.5

    def test_move_with_approach_offset(self, patch_command_broadcaster):

        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=0.1, approach_offset=0.05)

        assert result.success is True
        assert result.result is not None
        assert result.result["approach_offset"] == 0.05
        # Y coordinate should be offset - approach_offset lifts along Unity Y (up-axis)
        assert result.result["target_position"]["y"] == pytest.approx(
            0.25
        )  # 0.2 + 0.05
        assert result.result["target_position"]["z"] == pytest.approx(0.1)  # unchanged

    def test_move_command_structure(self, patch_command_broadcaster):

        result = move_to_coordinate(
            "Robot1", x=0.3, y=0.2, z=0.1, speed=1.5, request_id=123
        )

        # Verify command was sent
        patch_command_broadcaster.send_command.assert_called_once()
        call_args = patch_command_broadcaster.send_command.call_args

        # Check command structure
        command = call_args[0][0]
        assert command["command_type"] == "move_to_coordinate"
        assert command["robot_id"] == "Robot1"
        assert command["parameters"]["target_position"]["x"] == 0.3
        assert command["parameters"]["target_position"]["y"] == 0.2
        assert command["parameters"]["target_position"]["z"] == 0.1
        assert command["parameters"]["speed_multiplier"] == 1.5
        assert "timestamp" in command

        # Check request_id parameter
        request_id = call_args[0][1]
        assert request_id == 123


# Test Class: Coordinate Validation


class TestMoveCoordinateValidation:

    def test_move_invalid_x_coordinate_too_high(self, patch_command_broadcaster):

        result = move_to_coordinate("Robot1", x=1.5, y=0.0, z=0.1)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_X_COORDINATE"

    def test_move_invalid_x_coordinate_too_low(self, patch_command_broadcaster):

        result = move_to_coordinate("Robot1", x=-1.5, y=0.0, z=0.1)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_X_COORDINATE"

    def test_move_invalid_y_coordinate(self, patch_command_broadcaster):

        result = move_to_coordinate("Robot1", x=0.3, y=2.0, z=0.1)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_Y_COORDINATE"

    def test_move_invalid_z_coordinate_too_high(self, patch_command_broadcaster):

        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=1.0)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_Z_COORDINATE"

    def test_move_invalid_z_coordinate_too_low(self, patch_command_broadcaster):

        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=-1.0)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_Z_COORDINATE"

    def test_move_with_negative_z_valid(self, patch_command_broadcaster):

        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=-0.3)

        # Z can be negative (below robot base level)
        assert result.success is True


# Test Class: Parameter Validation


class TestMoveParameterValidation:

    def test_move_invalid_speed_too_low(self, patch_command_broadcaster):

        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=0.1, speed=0.05)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_SPEED"

    def test_move_invalid_speed_too_high(self, patch_command_broadcaster):

        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=0.1, speed=5.0)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_SPEED"

    def test_move_invalid_approach_offset_negative(self, patch_command_broadcaster):

        result = move_to_coordinate(
            "Robot1", x=0.3, y=0.2, z=0.1, approach_offset=-0.05
        )

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_APPROACH_OFFSET"

    def test_move_invalid_approach_offset_too_large(self, patch_command_broadcaster):

        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=0.1, approach_offset=0.5)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_APPROACH_OFFSET"

    def test_move_invalid_robot_id(self, patch_command_broadcaster):

        result = move_to_coordinate("", x=0.3, y=0.2, z=0.1)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"


# Test Class: Error Handling


class TestMoveErrors:

    def test_move_communication_failed(self, patch_command_broadcaster):
        patch_command_broadcaster.send_command = Mock(return_value=False)

        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=0.1)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "COMMUNICATION_FAILED"

    def test_move_network_error(self, patch_command_broadcaster):
        patch_command_broadcaster.send_command = Mock(
            side_effect=Exception("Network error")
        )

        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=0.1)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "UNEXPECTED_ERROR"


# Test Class: allow_parallel_ros routing to per-robot ROS connection


class TestMoveToCoordinateAllowParallelRos:
    def test_uses_parallel_instance_and_flag_when_allow_parallel_ros_true(self):
        mock_default_bridge = MagicMock()
        mock_default_bridge.is_connected = True
        mock_parallel_bridge = MagicMock()
        mock_parallel_bridge.plan_and_execute.return_value = {
            "success": True,
            "planning_time": 1.0,
        }

        with patch(
            "ros2.ROSBridge.ROSBridge.get_instance", return_value=mock_default_bridge
        ), patch(
            "ros2.ROSBridge.ROSBridge.get_parallel_instance",
            return_value=mock_parallel_bridge,
        ) as mock_get_parallel:
            result = move_to_coordinate(
                "Robot1",
                x=0.3,
                y=0.2,
                z=0.1,
                use_ros=True,
                allow_parallel_ros=True,
            )

        assert result.success is True
        mock_get_parallel.assert_called_once_with("Robot1")
        mock_parallel_bridge.plan_and_execute.assert_called_once()
        assert (
            mock_parallel_bridge.plan_and_execute.call_args[1]["allow_parallel"] is True
        )

    def test_uses_default_instance_when_allow_parallel_ros_false(self):
        mock_default_bridge = MagicMock()
        mock_default_bridge.is_connected = True
        mock_default_bridge.plan_and_execute.return_value = {
            "success": True,
            "planning_time": 1.0,
        }

        with patch(
            "ros2.ROSBridge.ROSBridge.get_instance", return_value=mock_default_bridge
        ), patch("ros2.ROSBridge.ROSBridge.get_parallel_instance") as mock_get_parallel:
            result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=0.1, use_ros=True)

        assert result.success is True
        mock_get_parallel.assert_not_called()
        assert (
            mock_default_bridge.plan_and_execute.call_args[1]["allow_parallel"] is False
        )


class TestAdjustOrientationAllowParallelRos:
    def test_uses_parallel_instance_and_flag_when_allow_parallel_ros_true(self):
        mock_default_bridge = MagicMock()
        mock_default_bridge.is_connected = True
        mock_parallel_bridge = MagicMock()
        mock_parallel_bridge.plan_orientation_change.return_value = {
            "success": True,
            "planning_time": 1.0,
        }

        with patch(
            "ros2.ROSBridge.ROSBridge.get_instance", return_value=mock_default_bridge
        ), patch(
            "ros2.ROSBridge.ROSBridge.get_parallel_instance",
            return_value=mock_parallel_bridge,
        ) as mock_get_parallel:
            result = adjust_end_effector_orientation(
                "Robot2",
                roll=90.0,
                use_ros=True,
                allow_parallel_ros=True,
            )

        assert result.success is True
        mock_get_parallel.assert_called_once_with("Robot2")
        assert (
            mock_parallel_bridge.plan_orientation_change.call_args[1]["allow_parallel"]
            is True
        )

    def test_uses_default_instance_when_allow_parallel_ros_false(self):
        mock_default_bridge = MagicMock()
        mock_default_bridge.is_connected = True
        mock_default_bridge.plan_orientation_change.return_value = {
            "success": True,
            "planning_time": 1.0,
        }

        with patch(
            "ros2.ROSBridge.ROSBridge.get_instance", return_value=mock_default_bridge
        ), patch("ros2.ROSBridge.ROSBridge.get_parallel_instance") as mock_get_parallel:
            result = adjust_end_effector_orientation("Robot2", roll=90.0, use_ros=True)

        assert result.success is True
        mock_get_parallel.assert_not_called()
        assert (
            mock_default_bridge.plan_orientation_change.call_args[1]["allow_parallel"]
            is False
        )


# Test Class: Operation Definition


class TestMoveOperationDefinition:

    def test_operation_definition_exists(self):
        assert MOVE_TO_COORDINATE_OPERATION is not None
        assert MOVE_TO_COORDINATE_OPERATION.name == "move_to_coordinate"
        assert MOVE_TO_COORDINATE_OPERATION.operation_id == "motion_move_to_coord_001"

    def test_operation_has_metadata(self):
        op = MOVE_TO_COORDINATE_OPERATION

        assert op.description is not None
        assert len(op.parameters) >= 3  # robot_id, x, y, z at minimum
        assert op.preconditions is not None
        assert op.postconditions is not None
        assert op.implementation is not None

    def test_operation_execution_through_definition(self, patch_command_broadcaster):

        result = MOVE_TO_COORDINATE_OPERATION.execute(
            robot_id="Robot1", x=0.3, y=0.2, z=0.1
        )

        assert result.success is True
