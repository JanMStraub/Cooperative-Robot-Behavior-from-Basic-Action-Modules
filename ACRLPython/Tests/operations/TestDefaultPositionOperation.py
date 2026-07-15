import time
from unittest.mock import Mock, MagicMock, patch

from operations.DefaultPositionOperation import (
    return_to_start_position,
    RETURN_TO_START_POSITION_OPERATION,
)


class TestReturnToStartPosition:

    def test_return_to_start_position_success(self, patch_command_broadcaster):
        result = return_to_start_position("Robot1")

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "Robot1"
        assert result.result["speed"] == 1.0
        assert result.result["status"] == "command_sent"
        assert "timestamp" in result.result
        patch_command_broadcaster.send_command.assert_called_once()

    def test_return_to_start_position_with_custom_speed(
        self, patch_command_broadcaster
    ):
        result = return_to_start_position("Robot1", speed=0.5)

        assert result.success is True
        assert result.result is not None
        assert result.result["speed"] == 0.5

    def test_return_to_start_position_slow_speed(self, patch_command_broadcaster):
        result = return_to_start_position("Robot1", speed=0.3)

        assert result.success is True
        assert result.result is not None
        assert result.result["speed"] == 0.3

    def test_return_to_start_position_fast_speed(self, patch_command_broadcaster):
        result = return_to_start_position("Robot1", speed=1.5)

        assert result.success is True
        assert result.result is not None
        assert result.result["speed"] == 1.5

    def test_return_to_start_position_invalid_robot_id_empty(self):
        result = return_to_start_position("")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_return_to_start_position_invalid_robot_id_none(self):
        result = return_to_start_position(None)  # type: ignore[arg-type]

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_return_to_start_position_invalid_robot_id_number(self):
        result = return_to_start_position(123)  # type: ignore[arg-type]

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_return_to_start_position_invalid_speed_too_low(self):
        result = return_to_start_position("Robot1", speed=0.05)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_SPEED"

    def test_return_to_start_position_invalid_speed_too_high(self):
        result = return_to_start_position("Robot1", speed=3.0)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_SPEED"

    def test_return_to_start_position_command_structure(
        self, patch_command_broadcaster
    ):
        result = return_to_start_position("Robot1", speed=0.8, request_id=555)

        patch_command_broadcaster.send_command.assert_called_once()
        call_args = patch_command_broadcaster.send_command.call_args

        command = call_args[0][0]
        assert command["command_type"] == "return_to_start_position"
        assert command["robot_id"] == "Robot1"
        assert command["parameters"]["speed_multiplier"] == 0.8
        assert "timestamp" in command

        request_id = call_args[0][1]
        assert request_id == 555

    def test_return_to_start_position_communication_failed(
        self, patch_command_broadcaster
    ):
        patch_command_broadcaster.send_command = Mock(return_value=False)

        result = return_to_start_position("Robot1")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "COMMUNICATION_FAILED"

    def test_return_to_start_position_network_error(self, patch_command_broadcaster):
        patch_command_broadcaster.send_command = Mock(
            side_effect=Exception("Network error")
        )

        result = return_to_start_position("Robot1")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "UNEXPECTED_ERROR"


class TestReturnAllowParallelRos:
    def test_uses_parallel_instance_and_flag_when_allow_parallel_ros_true(self):
        mock_default_bridge = MagicMock()
        mock_default_bridge.is_connected = True
        mock_parallel_bridge = MagicMock()
        mock_parallel_bridge.plan_return_to_start.return_value = {
            "success": True,
            "planning_time": 1.0,
        }

        with patch(
            "ros2.ROSBridge.ROSBridge.get_instance", return_value=mock_default_bridge
        ), patch(
            "ros2.ROSBridge.ROSBridge.get_parallel_instance",
            return_value=mock_parallel_bridge,
        ) as mock_get_parallel:
            result = return_to_start_position(
                "Robot1", use_ros=True, allow_parallel_ros=True
            )

        assert result.success is True
        mock_get_parallel.assert_called_once_with("Robot1")
        assert (
            mock_parallel_bridge.plan_return_to_start.call_args[1]["allow_parallel"]
            is True
        )

    def test_uses_default_instance_when_allow_parallel_ros_false(self):
        mock_default_bridge = MagicMock()
        mock_default_bridge.is_connected = True
        mock_default_bridge.plan_return_to_start.return_value = {
            "success": True,
            "planning_time": 1.0,
        }

        with patch(
            "ros2.ROSBridge.ROSBridge.get_instance", return_value=mock_default_bridge
        ), patch("ros2.ROSBridge.ROSBridge.get_parallel_instance") as mock_get_parallel:
            result = return_to_start_position("Robot1", use_ros=True)

        assert result.success is True
        mock_get_parallel.assert_not_called()
        assert (
            mock_default_bridge.plan_return_to_start.call_args[1]["allow_parallel"]
            is False
        )


class TestReturnSpeedValidation:

    def test_return_minimum_valid_speed(self, patch_command_broadcaster):
        result = return_to_start_position("Robot1", speed=0.1)

        assert result.success is True
        assert result.result is not None
        assert result.result["speed"] == 0.1

    def test_return_maximum_valid_speed(self, patch_command_broadcaster):
        result = return_to_start_position("Robot1", speed=2.0)

        assert result.success is True
        assert result.result is not None
        assert result.result["speed"] == 2.0

    def test_return_speed_boundary_below_minimum(self):
        result = return_to_start_position("Robot1", speed=0.099)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_SPEED"

    def test_return_speed_boundary_above_maximum(self):
        result = return_to_start_position("Robot1", speed=2.001)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_SPEED"

    def test_return_speed_typical_values(self, patch_command_broadcaster):
        typical_speeds = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8]

        for speed in typical_speeds:
            result = return_to_start_position("Robot1", speed=speed)
            assert result.success is True
            assert result.result is not None
            assert result.result["speed"] == speed


class TestReturnDifferentRobots:

    def test_return_standard_robot_id(self, patch_command_broadcaster):
        result = return_to_start_position("Robot1")

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "Robot1"

    def test_return_ar4_robot_id(self, patch_command_broadcaster):
        result = return_to_start_position("AR4_Robot")

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "AR4_Robot"

    def test_return_numbered_robot_id(self, patch_command_broadcaster):
        result = return_to_start_position("Robot2")

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "Robot2"

    def test_return_custom_robot_id(self, patch_command_broadcaster):
        result = return_to_start_position("CustomRobot_123")

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "CustomRobot_123"


class TestReturnOperationDefinition:

    def test_return_operation_definition(self):
        assert RETURN_TO_START_POSITION_OPERATION is not None
        assert RETURN_TO_START_POSITION_OPERATION.name == "return_to_start_position"
        assert (
            RETURN_TO_START_POSITION_OPERATION.operation_id
            == "motion_return_to_start_001"
        )

    def test_return_operation_has_metadata(self):
        op = RETURN_TO_START_POSITION_OPERATION

        assert op.description is not None
        assert len(op.parameters) >= 1  # robot_id minimum
        assert op.preconditions is not None
        assert op.postconditions is not None
        assert op.implementation is not None
        assert op.average_duration_ms is not None
        assert op.success_rate is not None

    def test_return_operation_has_usage_examples(self):
        op = RETURN_TO_START_POSITION_OPERATION

        assert op.usage_examples is not None
        assert len(op.usage_examples) > 0

    def test_return_operation_has_failure_modes(self):
        op = RETURN_TO_START_POSITION_OPERATION

        assert op.failure_modes is not None
        assert len(op.failure_modes) > 0

    def test_return_operation_execution_through_definition(
        self, patch_command_broadcaster
    ):
        """Test executing return operation through BasicOperation.execute()."""
        result = RETURN_TO_START_POSITION_OPERATION.execute(
            robot_id="Robot1", speed=1.0
        )

        assert result.success is True

    def test_return_operation_preconditions(self):
        op = RETURN_TO_START_POSITION_OPERATION

        # Should have preconditions about robot registration
        preconditions_text = " ".join(op.preconditions).lower()
        assert "register" in preconditions_text or "initialized" in preconditions_text

    def test_return_operation_postconditions(self):
        op = RETURN_TO_START_POSITION_OPERATION
        assert isinstance(op.postconditions, list)

    def test_return_operation_category(self):
        op = RETURN_TO_START_POSITION_OPERATION

        assert op.category.value == "navigation"

    def test_return_operation_complexity(self):
        op = RETURN_TO_START_POSITION_OPERATION

        assert op.complexity.value == "basic"


class TestReturnConcurrency:

    def test_concurrent_returns_different_robots(self, patch_command_broadcaster):
        """Test concurrent return operations for different robots."""
        import threading

        results = []

        def return_worker(robot_id):
            result = return_to_start_position(robot_id)
            results.append(result)

        threads = [
            threading.Thread(target=return_worker, args=(f"Robot{i}",))
            for i in range(1, 4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_concurrent_returns_same_robot(self, patch_command_broadcaster):
        """Test concurrent return operations for same robot."""
        import threading

        results = []

        def return_worker():
            result = return_to_start_position("Robot1")
            results.append(result)

        threads = [threading.Thread(target=return_worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 3
        assert all(r.success for r in results)


class TestReturnEdgeCases:

    def test_return_with_minimal_parameters(self, patch_command_broadcaster):
        result = return_to_start_position("Robot1")

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "Robot1"
        assert result.result["speed"] == 1.0

    def test_return_with_all_parameters(self, patch_command_broadcaster):
        result = return_to_start_position(
            robot_id="AR4_Robot", speed=0.7, request_id=999
        )

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "AR4_Robot"
        assert result.result["speed"] == 0.7

    def test_return_robot_id_with_special_characters(self, patch_command_broadcaster):
        result = return_to_start_position("Robot_Test-123")

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "Robot_Test-123"

    def test_return_very_slow_speed(self, patch_command_broadcaster):
        result = return_to_start_position("Robot1", speed=0.1)

        assert result.success is True
        assert result.result is not None
        assert result.result["speed"] == 0.1

    def test_return_very_fast_speed(self, patch_command_broadcaster):
        result = return_to_start_position("Robot1", speed=2.0)

        assert result.success is True
        assert result.result is not None
        assert result.result["speed"] == 2.0

    def test_return_default_speed_value(self, patch_command_broadcaster):
        """Test that default speed is 1.0 (normal)."""
        result = return_to_start_position("Robot1")

        assert result.success is True
        assert result.result is not None
        assert result.result["speed"] == 1.0

    def test_return_timestamp_accuracy(self, patch_command_broadcaster):
        before_time = time.time()
        result = return_to_start_position("Robot1")
        after_time = time.time()

        assert result.success is True
        assert result.result is not None
        timestamp = result.result["timestamp"]
        assert before_time <= timestamp <= after_time


class TestReturnErrorHandling:

    def test_return_invalid_robot_id_has_suggestions(self):
        """Test that invalid robot ID error provides recovery suggestions."""
        result = return_to_start_position("")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"
        assert "recovery_suggestions" in result.error
        assert len(result.error["recovery_suggestions"]) > 0

    def test_return_invalid_speed_has_suggestions(self):
        """Test that invalid speed error provides recovery suggestions."""
        result = return_to_start_position("Robot1", speed=5.0)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_SPEED"
        assert "recovery_suggestions" in result.error
        assert len(result.error["recovery_suggestions"]) > 0

    def test_return_communication_failed_has_suggestions(
        self, patch_command_broadcaster
    ):
        """Test that communication failure provides recovery suggestions."""
        patch_command_broadcaster.send_command = Mock(return_value=False)

        result = return_to_start_position("Robot1")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "COMMUNICATION_FAILED"
        assert "recovery_suggestions" in result.error
        assert len(result.error["recovery_suggestions"]) > 0

    def test_return_unexpected_error_has_suggestions(self, patch_command_broadcaster):
        """Test that unexpected error provides recovery suggestions."""
        patch_command_broadcaster.send_command = Mock(
            side_effect=Exception("Test error")
        )

        result = return_to_start_position("Robot1")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "UNEXPECTED_ERROR"
        assert "recovery_suggestions" in result.error
        assert len(result.error["recovery_suggestions"]) > 0

    def test_return_error_messages_are_descriptive(self):
        # Invalid robot ID
        result = return_to_start_position("")
        assert result.error is not None
        assert result.error["message"]
        assert len(result.error["message"]) > 10

        # Invalid speed
        result = return_to_start_position("Robot1", speed=10.0)
        assert result.error is not None
        assert result.error["message"]
        assert len(result.error["message"]) > 10


class TestReturnIntegration:

    def test_return_operation_registered_correctly(self):
        op = RETURN_TO_START_POSITION_OPERATION

        assert op.operation_id is not None
        assert op.name is not None
        assert op.implementation is not None
        assert callable(op.implementation)

    def test_return_operation_parameters_match_function(self):
        op = RETURN_TO_START_POSITION_OPERATION

        param_names = [p.name for p in op.parameters]
        assert "robot_id" in param_names
        assert "speed" in param_names

    def test_return_operation_required_parameters(self):
        op = RETURN_TO_START_POSITION_OPERATION

        required_params = [p for p in op.parameters if p.required]
        assert len(required_params) >= 1  # At least robot_id

        robot_id_param = next((p for p in op.parameters if p.name == "robot_id"), None)
        assert robot_id_param is not None
        assert robot_id_param.required is True

    def test_return_operation_optional_parameters(self):
        op = RETURN_TO_START_POSITION_OPERATION

        speed_param = next((p for p in op.parameters if p.name == "speed"), None)
        assert speed_param is not None
        assert speed_param.required is False
        assert speed_param.default == 1.0
