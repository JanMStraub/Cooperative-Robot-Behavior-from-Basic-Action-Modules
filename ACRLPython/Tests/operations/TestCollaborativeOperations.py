#!/usr/bin/env python3
"""
Unit tests for CollaborativeOperations.py

Tests the collaborative manipulation operations including:
- stabilize_object: Object stabilization for dual-arm tasks
- Dual-arm force coordination
- Stability verification under disturbances
- Handoff scenarios with stabilization
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch

from operations.CollaborativeOperations import (
    stabilize_object,
    STABILIZE_OBJECT_OPERATION,
)
from operations.Base import OperationResult


# ============================================================================
# Test Class: stabilize_object - Object Stabilization
# ============================================================================


class TestStabilizeObject:
    """Test object stabilization operation."""

    def test_stabilize_object_success(self, patch_command_broadcaster):
        """Test successful object stabilization."""
        result = stabilize_object("Robot1", "LargeCube", duration_ms=5000, force_limit=10.0)

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "Robot1"
        assert result.result["object_id"] == "LargeCube"
        assert result.result["duration_ms"] == 5000
        assert result.result["force_limit"] == 10.0
        assert result.result["status"] == "stabilizing"
        patch_command_broadcaster.send_command.assert_called_once()

    def test_stabilize_object_default_parameters(self, patch_command_broadcaster):
        """Test stabilization with default duration and force limit."""
        result = stabilize_object("Robot1", "Cube01")

        assert result.success is True
        assert result.result is not None
        assert result.result["duration_ms"] == 5000  # Default
        assert result.result["force_limit"] == 10.0  # Default

    def test_stabilize_object_custom_duration(self, patch_command_broadcaster):
        """Test stabilization with custom duration."""
        result = stabilize_object("Robot1", "AssemblyPart", duration_ms=10000)

        assert result.success is True
        assert result.result is not None
        assert result.result["duration_ms"] == 10000

    def test_stabilize_object_custom_force_limit(self, patch_command_broadcaster):
        """Test stabilization with custom force limit."""
        result = stabilize_object("Robot1", "FragileObject", force_limit=5.0)

        assert result.success is True
        assert result.result is not None
        assert result.result["force_limit"] == 5.0

    def test_stabilize_object_invalid_robot_id(self):
        """Test stabilization with invalid robot ID."""
        result = stabilize_object("", "Cube01")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_ROBOT_ID"

    def test_stabilize_object_invalid_object_id(self):
        """Test stabilization with invalid object ID."""
        result = stabilize_object("Robot1", "")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_OBJECT_ID"

    def test_stabilize_object_invalid_duration_too_low(self):
        """Test stabilization with duration below minimum."""
        result = stabilize_object("Robot1", "Cube01", duration_ms=50)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_DURATION"

    def test_stabilize_object_invalid_duration_too_high(self):
        """Test stabilization with duration above maximum."""
        result = stabilize_object("Robot1", "Cube01", duration_ms=40000)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_DURATION"

    def test_stabilize_object_invalid_force_limit_too_low(self):
        """Test stabilization with force limit below minimum."""
        result = stabilize_object("Robot1", "Cube01", force_limit=0.5)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_FORCE_LIMIT"

    def test_stabilize_object_invalid_force_limit_too_high(self):
        """Test stabilization with force limit above maximum."""
        result = stabilize_object("Robot1", "Cube01", force_limit=60.0)

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "INVALID_FORCE_LIMIT"

    def test_stabilize_object_boundary_duration_values(self, patch_command_broadcaster):
        """Test stabilization with boundary duration values."""
        # Minimum valid duration
        result = stabilize_object("Robot1", "Cube01", duration_ms=100)
        assert result.success is True

        # Maximum valid duration
        result = stabilize_object("Robot1", "Cube01", duration_ms=30000)
        assert result.success is True

    def test_stabilize_object_boundary_force_values(self, patch_command_broadcaster):
        """Test stabilization with boundary force limit values."""
        # Minimum valid force
        result = stabilize_object("Robot1", "Cube01", force_limit=1.0)
        assert result.success is True

        # Maximum valid force
        result = stabilize_object("Robot1", "Cube01", force_limit=50.0)
        assert result.success is True

    def test_stabilize_object_command_structure(self, patch_command_broadcaster):
        """Test that stabilize command has correct structure."""
        result = stabilize_object(
            "Robot1", "AssemblyPart", duration_ms=8000, force_limit=15.0, request_id=789
        )

        patch_command_broadcaster.send_command.assert_called_once()
        call_args = patch_command_broadcaster.send_command.call_args

        command = call_args[0][0]
        assert command["command_type"] == "stabilize_object"
        assert command["robot_id"] == "Robot1"
        assert command["parameters"]["object_id"] == "AssemblyPart"
        assert command["parameters"]["duration_ms"] == 8000
        assert command["parameters"]["force_limit"] == 15.0
        assert "timestamp" in command

        request_id = call_args[0][1]
        assert request_id == 789

    def test_stabilize_object_communication_failed(self, patch_command_broadcaster):
        """Test stabilization when communication fails."""
        patch_command_broadcaster.send_command = Mock(return_value=False)

        result = stabilize_object("Robot1", "Cube01")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "COMMUNICATION_FAILED"

    def test_stabilize_object_network_error(self, patch_command_broadcaster):
        """Test stabilization when broadcaster raises exception."""
        patch_command_broadcaster.send_command = Mock(side_effect=Exception("Network error"))

        result = stabilize_object("Robot1", "Cube01")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "UNEXPECTED_ERROR"


# ============================================================================
# Test Class: Dual-Arm Coordination Scenarios
# ============================================================================


class TestDualArmStabilization:
    """Test dual-arm coordination scenarios with stabilization."""

    def test_stabilize_for_partner_manipulation(self, patch_command_broadcaster):
        """Test stabilization while partner robot manipulates object."""
        # Robot1 stabilizes
        result1 = stabilize_object("Robot1", "LargeBoard", duration_ms=10000, force_limit=20.0)
        assert result1.success is True

        # This would be followed by Robot2 manipulation in real scenario
        # (tested separately in integration tests)

    def test_short_duration_stabilization(self, patch_command_broadcaster):
        """Test brief stabilization for quick handoff."""
        result = stabilize_object("Robot1", "SmallPart", duration_ms=1000, force_limit=5.0)

        assert result.success is True
        assert result.result is not None
        assert result.result["duration_ms"] == 1000

    def test_long_duration_stabilization(self, patch_command_broadcaster):
        """Test extended stabilization for complex assembly."""
        result = stabilize_object("Robot1", "AssemblyBase", duration_ms=25000, force_limit=30.0)

        assert result.success is True
        assert result.result is not None
        assert result.result["duration_ms"] == 25000

    def test_low_force_fragile_object(self, patch_command_broadcaster):
        """Test stabilization with low force for fragile objects."""
        result = stabilize_object("Robot1", "GlassVial", duration_ms=5000, force_limit=2.0)

        assert result.success is True
        assert result.result is not None
        assert result.result["force_limit"] == 2.0

    def test_high_force_heavy_object(self, patch_command_broadcaster):
        """Test stabilization with high force for heavy objects."""
        result = stabilize_object("Robot1", "HeavyBlock", duration_ms=5000, force_limit=45.0)

        assert result.success is True
        assert result.result is not None
        assert result.result["force_limit"] == 45.0


# ============================================================================
# Test Class: Operation Definition
# ============================================================================


class TestStabilizeOperationDefinition:
    """Test BasicOperation definition for stabilize_object."""

    def test_stabilize_operation_definition(self):
        """Test STABILIZE_OBJECT_OPERATION is properly defined."""
        assert STABILIZE_OBJECT_OPERATION is not None
        assert STABILIZE_OBJECT_OPERATION.name == "stabilize_object"
        assert STABILIZE_OBJECT_OPERATION.operation_id == "collaborative_stabilize_001"

    def test_stabilize_operation_has_metadata(self):
        """Test stabilize operation has required metadata."""
        op = STABILIZE_OBJECT_OPERATION

        assert op.description is not None
        assert len(op.parameters) >= 2  # robot_id, object_id minimum
        assert op.preconditions is not None
        assert op.postconditions is not None
        assert op.implementation is not None
        assert op.average_duration_ms is not None
        assert op.success_rate is not None

    def test_stabilize_operation_has_relationships(self):
        """Test stabilize operation has relationship metadata."""
        op = STABILIZE_OBJECT_OPERATION

        assert op.relationships is not None
        assert op.relationships.required_operations is not None
        assert len(op.relationships.required_operations) > 0
        # Should require gripper control and movement
        assert "manipulation_control_gripper_001" in op.relationships.required_operations
        assert "motion_move_to_coord_001" in op.relationships.required_operations

    def test_stabilize_operation_execution_through_definition(self, patch_command_broadcaster):
        """Test executing stabilize operation through BasicOperation.execute()."""
        result = STABILIZE_OBJECT_OPERATION.execute(
            robot_id="Robot1", object_id="TestCube", duration_ms=5000, force_limit=10.0
        )

        assert result.success is True

    def test_stabilize_operation_preconditions(self):
        """Test stabilize operation has appropriate preconditions."""
        op = STABILIZE_OBJECT_OPERATION

        # Should have at least one predicate-format precondition
        assert len(op.preconditions) >= 1
        assert any("robot_is_initialized" in pre for pre in op.preconditions)

    def test_stabilize_operation_postconditions(self):
        """Test stabilize operation has appropriate postconditions."""
        op = STABILIZE_OBJECT_OPERATION

        # Postconditions are empty (side-effects not verifiable as predicates)
        assert isinstance(op.postconditions, list)


# ============================================================================
# Test Class: Concurrent Execution
# ============================================================================


class TestStabilizationConcurrency:
    """Test thread safety for stabilization operations."""

    def test_concurrent_stabilization_different_objects(self, patch_command_broadcaster):
        """Test concurrent stabilization of different objects by different robots."""
        import threading

        results = []

        def stabilize_worker(robot_id, object_id):
            result = stabilize_object(robot_id, object_id, duration_ms=3000)
            results.append(result)

        threads = [
            threading.Thread(target=stabilize_worker, args=(f"Robot{i}", f"Object{i}"))
            for i in range(1, 4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_concurrent_stabilization_same_robot(self, patch_command_broadcaster):
        """Test sequential stabilization requests for same robot don't interfere."""
        import threading

        results = []

        def stabilize_worker(object_id):
            result = stabilize_object("Robot1", object_id, duration_ms=2000)
            results.append(result)

        threads = [
            threading.Thread(target=stabilize_worker, args=(f"Object{i}",)) for i in range(1, 4)
        ]
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


class TestStabilizationEdgeCases:
    """Test edge cases for stabilization operations."""

    def test_stabilize_with_minimal_parameters(self, patch_command_broadcaster):
        """Test stabilization with only required parameters."""
        result = stabilize_object("Robot1", "Object1")

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "Robot1"
        assert result.result["object_id"] == "Object1"

    def test_stabilize_with_all_parameters(self, patch_command_broadcaster):
        """Test stabilization with all parameters specified."""
        result = stabilize_object(
            robot_id="Robot1",
            object_id="ComplexPart",
            duration_ms=15000,
            force_limit=25.0,
            request_id=999,
        )

        assert result.success is True
        assert result.result is not None
        assert result.result["duration_ms"] == 15000
        assert result.result["force_limit"] == 25.0

    def test_stabilize_object_id_with_special_characters(self, patch_command_broadcaster):
        """Test stabilization with object ID containing special characters."""
        result = stabilize_object("Robot1", "Object_123-ABC")

        assert result.success is True
        assert result.result is not None
        assert result.result["object_id"] == "Object_123-ABC"

    def test_stabilize_robot_id_with_numbers(self, patch_command_broadcaster):
        """Test stabilization with robot ID containing numbers."""
        result = stabilize_object("AR4_Robot_2", "Cube01")

        assert result.success is True
        assert result.result is not None
        assert result.result["robot_id"] == "AR4_Robot_2"

    def test_stabilize_minimum_duration_maximum_force(self, patch_command_broadcaster):
        """Test stabilization with minimum duration and maximum force."""
        result = stabilize_object("Robot1", "HeavyObject", duration_ms=100, force_limit=50.0)

        assert result.success is True
        assert result.result is not None
        assert result.result["duration_ms"] == 100
        assert result.result["force_limit"] == 50.0

    def test_stabilize_maximum_duration_minimum_force(self, patch_command_broadcaster):
        """Test stabilization with maximum duration and minimum force."""
        result = stabilize_object("Robot1", "DelicateObject", duration_ms=30000, force_limit=1.0)

        assert result.success is True
        assert result.result is not None
        assert result.result["duration_ms"] == 30000
        assert result.result["force_limit"] == 1.0


# ============================================================================
# Test Class: Parameter Validation
# ============================================================================


class TestStabilizationParameterValidation:
    """Test comprehensive parameter validation for stabilization."""

    def test_robot_id_type_validation(self):
        """Test robot_id must be a string."""
        result = stabilize_object(None, "Cube01")
        assert result.success is False

        result = stabilize_object(123, "Cube01")
        assert result.success is False

    def test_object_id_type_validation(self):
        """Test object_id must be a string."""
        result = stabilize_object("Robot1", None)
        assert result.success is False

        result = stabilize_object("Robot1", 456)
        assert result.success is False

    def test_duration_type_validation(self, patch_command_broadcaster):
        """Test duration_ms accepts integer values."""
        result = stabilize_object("Robot1", "Cube01", duration_ms=5000)
        assert result.success is True

    def test_force_limit_type_validation(self, patch_command_broadcaster):
        """Test force_limit accepts float values."""
        result = stabilize_object("Robot1", "Cube01", force_limit=10.0)
        assert result.success is True

        # Integer should also work (converted to float)
        result = stabilize_object("Robot1", "Cube01", force_limit=10)
        assert result.success is True

    def test_duration_range_boundaries(self, patch_command_broadcaster):
        """Test duration range boundaries precisely."""
        # Just below minimum
        result = stabilize_object("Robot1", "Cube01", duration_ms=99)
        assert result.success is False

        # Exactly minimum
        result = stabilize_object("Robot1", "Cube01", duration_ms=100)
        assert result.success is True

        # Exactly maximum
        result = stabilize_object("Robot1", "Cube01", duration_ms=30000)
        assert result.success is True

        # Just above maximum
        result = stabilize_object("Robot1", "Cube01", duration_ms=30001)
        assert result.success is False

    def test_force_limit_range_boundaries(self, patch_command_broadcaster):
        """Test force limit range boundaries precisely."""
        # Just below minimum
        result = stabilize_object("Robot1", "Cube01", force_limit=0.9)
        assert result.success is False

        # Exactly minimum
        result = stabilize_object("Robot1", "Cube01", force_limit=1.0)
        assert result.success is True

        # Exactly maximum
        result = stabilize_object("Robot1", "Cube01", force_limit=50.0)
        assert result.success is True

        # Just above maximum
        result = stabilize_object("Robot1", "Cube01", force_limit=50.1)
        assert result.success is False
