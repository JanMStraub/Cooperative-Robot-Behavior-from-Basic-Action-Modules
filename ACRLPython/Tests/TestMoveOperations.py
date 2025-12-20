#!/usr/bin/env python3
"""
Unit tests for MoveOperations.py

Tests the movement operations including:
- Coordinate validation (in bounds, reachable)
- Speed parameter handling
- Approach offset calculation
- Command broadcasting
- Movement timeout
- Invalid coordinate handling
- Parameter validation
- Error handling and recovery
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch

from operations.MoveOperations import move_to_coordinate, MOVE_TO_COORDINATE_OPERATION
from operations.Base import OperationResult


# ============================================================================
# Test Class: Basic Movement Operations
# ============================================================================

class TestMoveOperations:
    """Test basic movement operations."""

    def test_move_to_coordinate_success(self, patch_command_broadcaster):
        """Test moving to coordinate successfully."""
        
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
        """Test moving with custom speed parameter."""
        
        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=0.1, speed=0.5)

        assert result.success is True
        assert result.result is not None
        assert result.result["speed"] == 0.5

    def test_move_with_approach_offset(self, patch_command_broadcaster):
        """Test moving with approach offset."""
        
        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=0.1, approach_offset=0.05)

        assert result.success is True
        assert result.result is not None
        assert result.result["approach_offset"] == 0.05
        # Z coordinate should be offset (use approx for floating point comparison)
        assert result.result["target_position"]["z"] == pytest.approx(0.15)  # 0.1 + 0.05

    def test_move_command_structure(self, patch_command_broadcaster):
        """Test that move command has correct structure."""
        
        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=0.1, speed=1.5, request_id=123)

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


# ============================================================================
# Test Class: Coordinate Validation
# ============================================================================

class TestMoveCoordinateValidation:
    """Test coordinate validation and bounds checking."""

    def test_move_invalid_x_coordinate_too_high(self, patch_command_broadcaster):
        """Test movement with X coordinate above maximum."""
        
        result = move_to_coordinate("Robot1", x=1.5, y=0.0, z=0.1)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_X_COORDINATE"

    def test_move_invalid_x_coordinate_too_low(self, patch_command_broadcaster):
        """Test movement with X coordinate below minimum."""
        
        result = move_to_coordinate("Robot1", x=-1.5, y=0.0, z=0.1)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_X_COORDINATE"

    def test_move_invalid_y_coordinate(self, patch_command_broadcaster):
        """Test movement with Y coordinate out of range."""
        
        result = move_to_coordinate("Robot1", x=0.3, y=2.0, z=0.1)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_Y_COORDINATE"

    def test_move_invalid_z_coordinate_too_high(self, patch_command_broadcaster):
        """Test movement with Z coordinate above maximum."""
        
        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=1.0)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_Z_COORDINATE"

    def test_move_invalid_z_coordinate_too_low(self, patch_command_broadcaster):
        """Test movement with Z coordinate below minimum."""
        
        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=-1.0)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_Z_COORDINATE"

    def test_move_with_negative_z_valid(self, patch_command_broadcaster):
        """Test movement with negative Z coordinate (valid, below base)."""
        
        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=-0.3)

        # Z can be negative (below robot base level)
        assert result.success is True


# ============================================================================
# Test Class: Parameter Validation
# ============================================================================

class TestMoveParameterValidation:
    """Test parameter validation for movement operations."""

    def test_move_invalid_speed_too_low(self, patch_command_broadcaster):
        """Test movement with speed below minimum."""
        
        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=0.1, speed=0.05)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_SPEED"

    def test_move_invalid_speed_too_high(self, patch_command_broadcaster):
        """Test movement with speed above maximum."""
        
        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=0.1, speed=5.0)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_SPEED"

    def test_move_invalid_approach_offset_negative(self, patch_command_broadcaster):
        """Test movement with negative approach offset."""
        
        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=0.1, approach_offset=-0.05)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_APPROACH_OFFSET"

    def test_move_invalid_approach_offset_too_large(self, patch_command_broadcaster):
        """Test movement with approach offset above maximum."""
        
        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=0.1, approach_offset=0.5)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_APPROACH_OFFSET"

    def test_move_invalid_robot_id(self, patch_command_broadcaster):
        """Test movement with invalid robot ID."""
        
        result = move_to_coordinate("", x=0.3, y=0.2, z=0.1)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"


# ============================================================================
# Test Class: Error Handling
# ============================================================================

class TestMoveErrors:
    """Test error handling for movement operations."""

    def test_move_communication_failed(self, patch_command_broadcaster):
        """Test movement when communication fails."""
        patch_command_broadcaster.send_command = Mock(return_value=False)

        
        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=0.1)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "COMMUNICATION_FAILED"

    def test_move_network_error(self, patch_command_broadcaster):
        """Test movement when broadcaster raises exception."""
        patch_command_broadcaster.send_command = Mock(side_effect=Exception("Network error"))

        result = move_to_coordinate("Robot1", x=0.3, y=0.2, z=0.1)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "UNEXPECTED_ERROR"


# ============================================================================
# Test Class: Operation Definition
# ============================================================================

class TestMoveOperationDefinition:
    """Test the BasicOperation definition for movement."""

    def test_operation_definition_exists(self):
        """Test that MOVE_TO_COORDINATE_OPERATION is properly defined."""
        assert MOVE_TO_COORDINATE_OPERATION is not None
        assert MOVE_TO_COORDINATE_OPERATION.name == "move_to_coordinate"
        assert MOVE_TO_COORDINATE_OPERATION.operation_id == "motion_move_to_coord_001"

    def test_operation_has_metadata(self):
        """Test that operation has required metadata."""
        op = MOVE_TO_COORDINATE_OPERATION

        assert op.description is not None
        assert len(op.parameters) >= 3  # robot_id, x, y, z at minimum
        assert op.preconditions is not None
        assert op.postconditions is not None
        assert op.implementation is not None

    def test_operation_execution_through_definition(self, patch_command_broadcaster):
        """Test executing operation through BasicOperation.execute()."""
        
        result = MOVE_TO_COORDINATE_OPERATION.execute(robot_id="Robot1", x=0.3, y=0.2, z=0.1)

        assert result.success is True
