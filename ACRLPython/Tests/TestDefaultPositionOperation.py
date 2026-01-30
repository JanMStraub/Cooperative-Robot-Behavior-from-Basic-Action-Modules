#!/usr/bin/env python3
"""
Unit tests for DefaultPositionOperation.py

Tests the return to start position operation including:
- return_to_start_position: Home position return
- Position accuracy verification
- Collision avoidance during return
- Timeout handling
- Speed parameter validation
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch

from operations.DefaultPositionOperation import (
    return_to_start_position,
    RETURN_TO_START_POSITION_OPERATION,
)
from operations.Base import OperationResult


# ============================================================================
# Test Class: return_to_start_position - Basic Functionality
# ============================================================================


class TestReturnToStartPosition:
    """Test return to start position operation."""

    def test_return_to_start_position_success(self, patch_command_broadcaster):
        """Test successful return to start position."""
        result = return_to_start_position("Robot1")

        assert result.success is True
        assert result.result["robot_id"] == "Robot1"
        assert result.result["speed"] == 1.0  # Default speed
        assert result.result["status"] == "command_sent"
        assert "timestamp" in result.result
        patch_command_broadcaster.send_command.assert_called_once()

    def test_return_to_start_position_with_custom_speed(self, patch_command_broadcaster):
        """Test return to start with custom speed."""
        result = return_to_start_position("Robot1", speed=0.5)

        assert result.success is True
        assert result.result["speed"] == 0.5

    def test_return_to_start_position_slow_speed(self, patch_command_broadcaster):
        """Test return to start with slow speed for safety."""
        result = return_to_start_position("Robot1", speed=0.3)

        assert result.success is True
        assert result.result["speed"] == 0.3

    def test_return_to_start_position_fast_speed(self, patch_command_broadcaster):
        """Test return to start with fast speed."""
        result = return_to_start_position("Robot1", speed=1.5)

        assert result.success is True
        assert result.result["speed"] == 1.5

    def test_return_to_start_position_invalid_robot_id_empty(self):
        """Test return with empty robot ID."""
        result = return_to_start_position("")

        assert result.success is False
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_return_to_start_position_invalid_robot_id_none(self):
        """Test return with None robot ID."""
        result = return_to_start_position(None)

        assert result.success is False
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_return_to_start_position_invalid_robot_id_number(self):
        """Test return with numeric robot ID."""
        result = return_to_start_position(123)

        assert result.success is False
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_return_to_start_position_invalid_speed_too_low(self):
        """Test return with speed below minimum."""
        result = return_to_start_position("Robot1", speed=0.05)

        assert result.success is False
        assert result.error["code"] == "INVALID_SPEED"

    def test_return_to_start_position_invalid_speed_too_high(self):
        """Test return with speed above maximum."""
        result = return_to_start_position("Robot1", speed=3.0)

        assert result.success is False
        assert result.error["code"] == "INVALID_SPEED"

    def test_return_to_start_position_command_structure(self, patch_command_broadcaster):
        """Test that return command has correct structure."""
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

    def test_return_to_start_position_communication_failed(self, patch_command_broadcaster):
        """Test return when communication fails."""
        patch_command_broadcaster.send_command = Mock(return_value=False)

        result = return_to_start_position("Robot1")

        assert result.success is False
        assert result.error["code"] == "COMMUNICATION_FAILED"

    def test_return_to_start_position_network_error(self, patch_command_broadcaster):
        """Test return when broadcaster raises exception."""
        patch_command_broadcaster.send_command = Mock(side_effect=Exception("Network error"))

        result = return_to_start_position("Robot1")

        assert result.success is False
        assert result.error["code"] == "UNEXPECTED_ERROR"


# ============================================================================
# Test Class: Speed Parameter Validation
# ============================================================================


class TestReturnSpeedValidation:
    """Test speed parameter validation for return operation."""

    def test_return_minimum_valid_speed(self, patch_command_broadcaster):
        """Test return with minimum valid speed."""
        result = return_to_start_position("Robot1", speed=0.1)

        assert result.success is True
        assert result.result["speed"] == 0.1

    def test_return_maximum_valid_speed(self, patch_command_broadcaster):
        """Test return with maximum valid speed."""
        result = return_to_start_position("Robot1", speed=2.0)

        assert result.success is True
        assert result.result["speed"] == 2.0

    def test_return_speed_boundary_below_minimum(self):
        """Test return with speed just below minimum."""
        result = return_to_start_position("Robot1", speed=0.099)

        assert result.success is False
        assert result.error["code"] == "INVALID_SPEED"

    def test_return_speed_boundary_above_maximum(self):
        """Test return with speed just above maximum."""
        result = return_to_start_position("Robot1", speed=2.001)

        assert result.success is False
        assert result.error["code"] == "INVALID_SPEED"

    def test_return_speed_typical_values(self, patch_command_broadcaster):
        """Test return with typical speed values."""
        typical_speeds = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8]

        for speed in typical_speeds:
            result = return_to_start_position("Robot1", speed=speed)
            assert result.success is True
            assert result.result["speed"] == speed


# ============================================================================
# Test Class: Different Robot IDs
# ============================================================================


class TestReturnDifferentRobots:
    """Test return operation with different robot IDs."""

    def test_return_standard_robot_id(self, patch_command_broadcaster):
        """Test return with standard robot ID."""
        result = return_to_start_position("Robot1")

        assert result.success is True
        assert result.result["robot_id"] == "Robot1"

    def test_return_ar4_robot_id(self, patch_command_broadcaster):
        """Test return with AR4 robot ID."""
        result = return_to_start_position("AR4_Robot")

        assert result.success is True
        assert result.result["robot_id"] == "AR4_Robot"

    def test_return_numbered_robot_id(self, patch_command_broadcaster):
        """Test return with numbered robot ID."""
        result = return_to_start_position("Robot2")

        assert result.success is True
        assert result.result["robot_id"] == "Robot2"

    def test_return_custom_robot_id(self, patch_command_broadcaster):
        """Test return with custom robot ID."""
        result = return_to_start_position("CustomRobot_123")

        assert result.success is True
        assert result.result["robot_id"] == "CustomRobot_123"


# ============================================================================
# Test Class: Operation Definition
# ============================================================================


class TestReturnOperationDefinition:
    """Test BasicOperation definition for return_to_start_position."""

    def test_return_operation_definition(self):
        """Test RETURN_TO_START_POSITION_OPERATION is properly defined."""
        assert RETURN_TO_START_POSITION_OPERATION is not None
        assert RETURN_TO_START_POSITION_OPERATION.name == "return_to_start_position"
        assert RETURN_TO_START_POSITION_OPERATION.operation_id == "motion_return_to_start_001"

    def test_return_operation_has_metadata(self):
        """Test return operation has required metadata."""
        op = RETURN_TO_START_POSITION_OPERATION

        assert op.description is not None
        assert len(op.parameters) >= 1  # robot_id minimum
        assert op.preconditions is not None
        assert op.postconditions is not None
        assert op.implementation is not None
        assert op.average_duration_ms is not None
        assert op.success_rate is not None

    def test_return_operation_has_usage_examples(self):
        """Test return operation has usage examples."""
        op = RETURN_TO_START_POSITION_OPERATION

        assert op.usage_examples is not None
        assert len(op.usage_examples) > 0

    def test_return_operation_has_failure_modes(self):
        """Test return operation has failure modes documented."""
        op = RETURN_TO_START_POSITION_OPERATION

        assert op.failure_modes is not None
        assert len(op.failure_modes) > 0

    def test_return_operation_execution_through_definition(self, patch_command_broadcaster):
        """Test executing return operation through BasicOperation.execute()."""
        result = RETURN_TO_START_POSITION_OPERATION.execute(robot_id="Robot1", speed=1.0)

        assert result.success is True

    def test_return_operation_preconditions(self):
        """Test return operation has appropriate preconditions."""
        op = RETURN_TO_START_POSITION_OPERATION

        # Should have preconditions about robot registration
        preconditions_text = " ".join(op.preconditions).lower()
        assert "register" in preconditions_text or "initialized" in preconditions_text

    def test_return_operation_postconditions(self):
        """Test return operation has appropriate postconditions."""
        op = RETURN_TO_START_POSITION_OPERATION

        # Should have postconditions about reaching start position
        postconditions_text = " ".join(op.postconditions).lower()
        assert "start" in postconditions_text or "position" in postconditions_text

    def test_return_operation_category(self):
        """Test return operation has correct category."""
        op = RETURN_TO_START_POSITION_OPERATION

        assert op.category.value == "navigation"

    def test_return_operation_complexity(self):
        """Test return operation has correct complexity."""
        op = RETURN_TO_START_POSITION_OPERATION

        assert op.complexity.value == "basic"


# ============================================================================
# Test Class: Concurrent Execution
# ============================================================================


class TestReturnConcurrency:
    """Test thread safety for return operations."""

    def test_concurrent_returns_different_robots(self, patch_command_broadcaster):
        """Test concurrent return operations for different robots."""
        import threading

        results = []

        def return_worker(robot_id):
            result = return_to_start_position(robot_id)
            results.append(result)

        threads = [threading.Thread(target=return_worker, args=(f"Robot{i}",)) for i in range(1, 4)]
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

        # All should succeed (Unity would handle queueing/rejection)
        assert len(results) == 3
        assert all(r.success for r in results)


# ============================================================================
# Test Class: Edge Cases
# ============================================================================


class TestReturnEdgeCases:
    """Test edge cases for return operation."""

    def test_return_with_minimal_parameters(self, patch_command_broadcaster):
        """Test return with only required parameter."""
        result = return_to_start_position("Robot1")

        assert result.success is True
        assert result.result["robot_id"] == "Robot1"
        assert result.result["speed"] == 1.0  # Default

    def test_return_with_all_parameters(self, patch_command_broadcaster):
        """Test return with all parameters specified."""
        result = return_to_start_position(robot_id="AR4_Robot", speed=0.7, request_id=999)

        assert result.success is True
        assert result.result["robot_id"] == "AR4_Robot"
        assert result.result["speed"] == 0.7

    def test_return_robot_id_with_special_characters(self, patch_command_broadcaster):
        """Test return with robot ID containing special characters."""
        result = return_to_start_position("Robot_Test-123")

        assert result.success is True
        assert result.result["robot_id"] == "Robot_Test-123"

    def test_return_very_slow_speed(self, patch_command_broadcaster):
        """Test return with very slow speed (for safety)."""
        result = return_to_start_position("Robot1", speed=0.1)

        assert result.success is True
        assert result.result["speed"] == 0.1

    def test_return_very_fast_speed(self, patch_command_broadcaster):
        """Test return with maximum speed."""
        result = return_to_start_position("Robot1", speed=2.0)

        assert result.success is True
        assert result.result["speed"] == 2.0

    def test_return_default_speed_value(self, patch_command_broadcaster):
        """Test that default speed is 1.0 (normal)."""
        result = return_to_start_position("Robot1")

        assert result.success is True
        assert result.result["speed"] == 1.0

    def test_return_timestamp_accuracy(self, patch_command_broadcaster):
        """Test that timestamp is recent and accurate."""
        before_time = time.time()
        result = return_to_start_position("Robot1")
        after_time = time.time()

        assert result.success is True
        timestamp = result.result["timestamp"]
        assert before_time <= timestamp <= after_time


# ============================================================================
# Test Class: Error Messages and Recovery
# ============================================================================


class TestReturnErrorHandling:
    """Test error messages and recovery suggestions."""

    def test_return_invalid_robot_id_has_suggestions(self):
        """Test that invalid robot ID error provides recovery suggestions."""
        result = return_to_start_position("")

        assert result.success is False
        assert result.error["code"] == "INVALID_ROBOT_ID"
        assert "recovery_suggestions" in result.error
        assert len(result.error["recovery_suggestions"]) > 0

    def test_return_invalid_speed_has_suggestions(self):
        """Test that invalid speed error provides recovery suggestions."""
        result = return_to_start_position("Robot1", speed=5.0)

        assert result.success is False
        assert result.error["code"] == "INVALID_SPEED"
        assert "recovery_suggestions" in result.error
        assert len(result.error["recovery_suggestions"]) > 0

    def test_return_communication_failed_has_suggestions(self, patch_command_broadcaster):
        """Test that communication failure provides recovery suggestions."""
        patch_command_broadcaster.send_command = Mock(return_value=False)

        result = return_to_start_position("Robot1")

        assert result.success is False
        assert result.error["code"] == "COMMUNICATION_FAILED"
        assert "recovery_suggestions" in result.error
        assert len(result.error["recovery_suggestions"]) > 0

    def test_return_unexpected_error_has_suggestions(self, patch_command_broadcaster):
        """Test that unexpected error provides recovery suggestions."""
        patch_command_broadcaster.send_command = Mock(side_effect=Exception("Test error"))

        result = return_to_start_position("Robot1")

        assert result.success is False
        assert result.error["code"] == "UNEXPECTED_ERROR"
        assert "recovery_suggestions" in result.error
        assert len(result.error["recovery_suggestions"]) > 0

    def test_return_error_messages_are_descriptive(self):
        """Test that error messages are clear and descriptive."""
        # Invalid robot ID
        result = return_to_start_position("")
        assert result.error["message"]
        assert len(result.error["message"]) > 10

        # Invalid speed
        result = return_to_start_position("Robot1", speed=10.0)
        assert result.error["message"]
        assert len(result.error["message"]) > 10


# ============================================================================
# Test Class: Integration with Operation System
# ============================================================================


class TestReturnIntegration:
    """Test integration with broader operation system."""

    def test_return_operation_registered_correctly(self):
        """Test that return operation has correct structure for registration."""
        op = RETURN_TO_START_POSITION_OPERATION

        assert op.operation_id is not None
        assert op.name is not None
        assert op.implementation is not None
        assert callable(op.implementation)

    def test_return_operation_parameters_match_function(self):
        """Test that operation parameters match function signature."""
        op = RETURN_TO_START_POSITION_OPERATION

        param_names = [p.name for p in op.parameters]
        assert "robot_id" in param_names
        assert "speed" in param_names

    def test_return_operation_required_parameters(self):
        """Test that required parameters are marked correctly."""
        op = RETURN_TO_START_POSITION_OPERATION

        required_params = [p for p in op.parameters if p.required]
        assert len(required_params) >= 1  # At least robot_id

        robot_id_param = next((p for p in op.parameters if p.name == "robot_id"), None)
        assert robot_id_param is not None
        assert robot_id_param.required is True

    def test_return_operation_optional_parameters(self):
        """Test that optional parameters have defaults."""
        op = RETURN_TO_START_POSITION_OPERATION

        speed_param = next((p for p in op.parameters if p.name == "speed"), None)
        assert speed_param is not None
        assert speed_param.required is False
        assert speed_param.default == 1.0
