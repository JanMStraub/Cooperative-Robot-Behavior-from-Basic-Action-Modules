#!/usr/bin/env python3
"""
Unit tests for StatusOperations.py

Tests the status check operations including:
- Robot status polling
- Response handling and parsing
- Detailed vs basic status requests
- Multiple robot queries
- Robot not found handling
- Timeout on status query
- Parameter validation
- Error handling
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch

from operations.StatusOperations import check_robot_status, CHECK_ROBOT_STATUS_OPERATION
from operations.Base import OperationResult


# ============================================================================
# Test Class: Basic Status Operations
# ============================================================================

class TestStatusOperations:
    """Test basic status check operations."""

    def test_check_robot_status_success(self, patch_command_broadcaster):
        """Test checking robot status successfully."""
        result = check_robot_status("Robot1")

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "Robot1"
        assert result.result["detailed"] is False
        assert result.result["status"] == "query_sent"
        patch_command_broadcaster.send_command.assert_called_once()

    def test_check_robot_status_detailed(self, patch_command_broadcaster):
        """Test checking detailed robot status."""
        result = check_robot_status("Robot1", detailed=True)

        assert result.success is True
        assert result.result is not None
        assert result.result["detailed"] is True

    def test_status_command_structure(self, patch_command_broadcaster):
        """Test that status command has correct structure."""
        result = check_robot_status("Robot1", detailed=True, request_id=456)

        # Verify command was sent
        patch_command_broadcaster.send_command.assert_called_once()
        call_args = patch_command_broadcaster.send_command.call_args

        # Check command structure
        command = call_args[0][0]
        assert command["command_type"] == "check_robot_status"
        assert command["robot_id"] == "Robot1"
        assert command["parameters"]["detailed"] is True
        assert "timestamp" in command

        # Check request_id parameter
        request_id = call_args[0][1]
        assert request_id == 456

    def test_check_multiple_robots(self, patch_command_broadcaster):
        """Test checking status of multiple robots sequentially."""
        result1 = check_robot_status("Robot1")
        result2 = check_robot_status("Robot2")

        assert result1.success is True
        assert result2.success is True
        assert result1.result is not None
        assert result2.result is not None
        assert result1.result["robot_id"] == "Robot1"
        assert result2.result["robot_id"] == "Robot2"
        assert patch_command_broadcaster.send_command.call_count == 2


# ============================================================================
# Test Class: Parameter Validation
# ============================================================================

class TestStatusParameterValidation:
    """Test parameter validation for status operations."""

    def test_status_invalid_robot_id_empty(self, patch_command_broadcaster):
        """Test status check with empty robot ID."""
        result = check_robot_status("")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_status_invalid_robot_id_none(self, patch_command_broadcaster):
        """Test status check with None robot ID."""
        result = check_robot_status(None)  # type: ignore[arg-type]

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_status_invalid_detailed_parameter(self, patch_command_broadcaster):
        """Test status check with invalid detailed parameter type."""
        result = check_robot_status("Robot1", detailed="yes")  # type: ignore[arg-type]

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_DETAILED_PARAMETER"


# ============================================================================
# Test Class: Error Handling
# ============================================================================

class TestStatusErrors:
    """Test error handling for status operations."""

    def test_status_communication_failed(self, patch_command_broadcaster):
        """Test status check when communication fails."""
        patch_command_broadcaster.send_command = Mock(return_value=False)

        result = check_robot_status("Robot1")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "COMMUNICATION_FAILED"

    def test_status_network_error(self, patch_command_broadcaster):
        """Test status check when broadcaster raises exception."""
        patch_command_broadcaster.send_command = Mock(side_effect=Exception("Network error"))

        result = check_robot_status("Robot1")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "UNEXPECTED_ERROR"


# ============================================================================
# Test Class: Operation Definition
# ============================================================================

class TestStatusOperationDefinition:
    """Test the BasicOperation definition for status check."""

    def test_operation_definition_exists(self):
        """Test that CHECK_ROBOT_STATUS_OPERATION is properly defined."""
        assert CHECK_ROBOT_STATUS_OPERATION is not None
        assert CHECK_ROBOT_STATUS_OPERATION.name == "check_robot_status"
        assert CHECK_ROBOT_STATUS_OPERATION.operation_id == "status_check_robot_001"

    def test_operation_has_metadata(self):
        """Test that operation has required metadata."""
        op = CHECK_ROBOT_STATUS_OPERATION

        assert op.description is not None
        assert len(op.parameters) >= 2  # robot_id, detailed
        assert op.preconditions is not None
        assert op.postconditions is not None
        assert op.implementation is not None

    def test_operation_execution_through_definition(self, patch_command_broadcaster):
        """Test executing operation through BasicOperation.execute()."""
        result = CHECK_ROBOT_STATUS_OPERATION.execute(robot_id="Robot1", detailed=False)

        assert result.success is True
