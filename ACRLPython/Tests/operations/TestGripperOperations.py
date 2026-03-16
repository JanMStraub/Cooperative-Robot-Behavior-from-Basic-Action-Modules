#!/usr/bin/env python3
"""
Unit tests for GripperOperations.py

Tests the gripper control operations including:
- Open/close gripper commands
- Timeout handling
- Command broadcasting to Unity
- Failure recovery
- Invalid robot ID handling
- State validation
- Parameter validation
- Error handling
"""

import pytest
from unittest.mock import Mock

from operations.GripperOperations import control_gripper, CONTROL_GRIPPER_OPERATION


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_broadcaster():
    """
    Create a mock CommandBroadcaster for testing.

    Returns:
        Mock CommandBroadcaster with send_command method
    """
    broadcaster = Mock()
    broadcaster.send_command = Mock(return_value=True)
    return broadcaster


# ============================================================================
# Test Class: Basic Gripper Operations
# ============================================================================


class TestGripperOperations:
    """Test basic gripper control operations."""

    def test_open_gripper_success(self, patch_command_broadcaster):
        """Test opening gripper successfully."""

        result = control_gripper("Robot1", open_gripper=True)

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "Robot1"
        assert result.result["open_gripper"] is True
        assert result.result["status"] == "command_sent"
        patch_command_broadcaster.send_command.assert_called_once()

    def test_close_gripper_success(self, patch_command_broadcaster):
        """Test closing gripper successfully."""

        result = control_gripper("Robot1", open_gripper=False)

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "Robot1"
        assert result.result["open_gripper"] is False
        patch_command_broadcaster.send_command.assert_called_once()

    def test_gripper_command_structure(self, patch_command_broadcaster):
        """Test that gripper command has correct structure."""

        result = control_gripper("Robot1", open_gripper=True, request_id=123)

        # Verify command was sent
        patch_command_broadcaster.send_command.assert_called_once()
        call_args = patch_command_broadcaster.send_command.call_args

        # Check command structure
        command = call_args[0][0]
        assert command["command_type"] == "control_gripper"
        assert command["robot_id"] == "Robot1"
        assert command["parameters"]["open_gripper"] is True
        assert "timestamp" in command

        # Check request_id parameter
        request_id = call_args[0][1]
        assert request_id == 123


# ============================================================================
# Test Class: Error Handling
# ============================================================================


class TestGripperErrors:
    """Test error handling for gripper operations."""

    def test_gripper_invalid_robot_id_empty(self, patch_command_broadcaster):
        """Test gripper control with empty robot ID."""

        result = control_gripper("", open_gripper=True)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_gripper_invalid_robot_id_none(self, patch_command_broadcaster):
        """Test gripper control with None robot ID."""

        result = control_gripper(None, open_gripper=True)  # type: ignore[arg-type]

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_gripper_invalid_parameter_type(self, patch_command_broadcaster):
        """Test gripper control with invalid open_gripper parameter type."""

        result = control_gripper("Robot1", open_gripper="yes")  # type: ignore[arg-type]

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_OPEN_GRIPPER_PARAMETER"

    def test_gripper_communication_failed(self, patch_command_broadcaster):
        """Test gripper control when communication fails."""
        patch_command_broadcaster.send_command = Mock(return_value=False)

        result = control_gripper("Robot1", open_gripper=True)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "COMMUNICATION_FAILED"

    def test_gripper_network_failure(self, patch_command_broadcaster):
        """Test gripper control when broadcaster raises exception."""
        patch_command_broadcaster.send_command = Mock(
            side_effect=Exception("Network error")
        )

        result = control_gripper("Robot1", open_gripper=True)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "UNEXPECTED_ERROR"


# ============================================================================
# Test Class: Command Broadcasting
# ============================================================================


class TestGripperBroadcasting:
    """Test command broadcasting to Unity."""

    def test_gripper_command_broadcast_open(self, patch_command_broadcaster):
        """Test opening gripper broadcasts correct command to Unity."""

        control_gripper("CustomRobot", open_gripper=True, request_id=999)

        call_args = patch_command_broadcaster.send_command.call_args
        command = call_args[0][0]
        request_id = call_args[0][1]

        assert command["robot_id"] == "CustomRobot"
        assert command["parameters"]["open_gripper"] is True
        assert request_id == 999

    def test_gripper_command_broadcast_close(self, patch_command_broadcaster):
        """Test closing gripper broadcasts correct command to Unity."""

        control_gripper("AR4_Robot", open_gripper=False, request_id=555)

        call_args = patch_command_broadcaster.send_command.call_args
        command = call_args[0][0]
        request_id = call_args[0][1]

        assert command["robot_id"] == "AR4_Robot"
        assert command["parameters"]["open_gripper"] is False
        assert request_id == 555


# ============================================================================
# Test Class: Operation Definition
# ============================================================================


class TestGripperOperationDefinition:
    """Test the BasicOperation definition for gripper control."""

    def test_operation_definition_exists(self):
        """Test that CONTROL_GRIPPER_OPERATION is properly defined."""
        assert CONTROL_GRIPPER_OPERATION is not None
        assert CONTROL_GRIPPER_OPERATION.name == "control_gripper"
        assert (
            CONTROL_GRIPPER_OPERATION.operation_id == "manipulation_control_gripper_001"
        )

    def test_operation_has_metadata(self):
        """Test that operation has required metadata."""
        op = CONTROL_GRIPPER_OPERATION

        assert op.description is not None
        assert len(op.parameters) >= 2  # robot_id, open_gripper
        assert op.preconditions is not None
        assert op.postconditions is not None
        assert op.implementation is not None

    def test_operation_execution_through_definition(self, patch_command_broadcaster):
        """Test executing operation through BasicOperation.execute()."""

        result = CONTROL_GRIPPER_OPERATION.execute(robot_id="Robot1", open_gripper=True)

        assert result.success is True
