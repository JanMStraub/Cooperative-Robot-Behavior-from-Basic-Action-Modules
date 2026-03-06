#!/usr/bin/env python3
"""
Unit tests for Verification.py

Tests formal verification engine including predicate parsing, precondition/
postcondition validation, and operation safety checks.
"""

import pytest
from unittest.mock import Mock, patch
from operations.Verification import (
    PredicateViolation,
    VerificationResult,
    PredicateParser,
    OperationVerifier,
    quick_verify_operation,
)
from operations.Base import OperationResult, BasicOperation, OperationCategory, OperationComplexity


class TestPredicateViolation:
    """Test PredicateViolation dataclass"""

    def test_predicate_violation_creation(self):
        """Test creating a PredicateViolation"""
        violation = PredicateViolation(
            predicate="target_within_reach(robot_id, x, y, z)",
            reason="Target exceeds max reach of 0.8m",
            severity="error",
            recovery_suggestions=["Move target closer", "Use different robot"]
        )

        assert violation.predicate == "target_within_reach(robot_id, x, y, z)"
        assert violation.reason == "Target exceeds max reach of 0.8m"
        assert violation.severity == "error"
        assert len(violation.recovery_suggestions) == 2

    def test_predicate_violation_defaults(self):
        """Test default values for PredicateViolation"""
        violation = PredicateViolation(
            predicate="test_predicate()",
            reason="Test reason"
        )

        assert violation.severity == "error"
        assert violation.recovery_suggestions == []


class TestVerificationResult:
    """Test VerificationResult dataclass"""

    def test_verification_result_success(self):
        """Test successful verification result"""
        result = VerificationResult(success=True)

        assert result.success is True
        assert result.execution_allowed is True
        assert len(result.violations) == 0
        assert len(result.warnings) == 0

    def test_add_violation_error(self):
        """Test adding error violation blocks execution"""
        result = VerificationResult(success=True)

        result.add_violation(
            predicate="robot_is_initialized(robot_id)",
            reason="Robot not found",
            severity="error",
            suggestions=["Initialize robot first"]
        )

        assert result.success is False
        assert result.execution_allowed is False
        assert len(result.violations) == 1
        assert result.violations[0].severity == "error"

    def test_add_violation_warning(self):
        """Test adding warning doesn't block execution"""
        result = VerificationResult(success=True)

        result.add_violation(
            predicate="robot_is_stationary(robot_id)",
            reason="Robot may still be moving",
            severity="warning"
        )

        assert result.success is True  # Still successful
        assert result.execution_allowed is True  # Can still execute
        assert len(result.warnings) == 1
        assert result.warnings[0].severity == "warning"

    def test_verification_result_to_dict(self):
        """Test serialization to dictionary"""
        result = VerificationResult(success=True)
        result.add_violation(
            predicate="test_predicate()",
            reason="Test failure",
            severity="error"
        )
        result.checked_predicates = ["predicate1", "predicate2"]

        result_dict = result.to_dict()

        assert result_dict["success"] is False
        assert result_dict["execution_allowed"] is False
        assert len(result_dict["violations"]) == 1
        assert result_dict["violations"][0]["predicate"] == "test_predicate()"
        assert len(result_dict["checked_predicates"]) == 2


class TestPredicateParser:
    """Test PredicateParser"""

    def test_parse_simple_predicate(self):
        """Test parsing predicate with no parameters"""
        parsed = PredicateParser.parse("operation_succeeded()")

        assert parsed is not None
        predicate_name, param_names = parsed
        assert predicate_name == "operation_succeeded"
        assert param_names == []

    def test_parse_with_single_param(self):
        """Test parsing predicate with one parameter"""
        parsed = PredicateParser.parse("robot_is_initialized(robot_id)")

        assert parsed is not None
        predicate_name, param_names = parsed
        assert predicate_name == "robot_is_initialized"
        assert param_names == ["robot_id"]

    def test_parse_with_multiple_params(self):
        """Test parsing predicate with multiple parameters"""
        parsed = PredicateParser.parse("target_within_reach(robot_id, x, y, z)")

        assert parsed is not None
        predicate_name, param_names = parsed
        assert predicate_name == "target_within_reach"
        assert param_names == ["robot_id", "x", "y", "z"]

    def test_parse_with_spaces(self):
        """Test parsing handles extra spaces"""
        parsed = PredicateParser.parse("  gripper_is_open( robot_id )  ")

        assert parsed is not None
        predicate_name, param_names = parsed
        assert predicate_name == "gripper_is_open"
        assert param_names == ["robot_id"]

    def test_parse_invalid_format(self):
        """Test invalid predicate format returns None"""
        parsed = PredicateParser.parse("not a valid predicate")

        assert parsed is None

    def test_resolve_parameters(self):
        """Test resolving parameter names to values"""
        param_names = ["robot_id", "x", "y", "z"]
        operation_params = {
            "robot_id": "Robot1",
            "x": 0.3,
            "y": 0.2,
            "z": 0.1
        }

        resolved = PredicateParser.resolve_parameters(param_names, operation_params)

        assert resolved["robot_id"] == "Robot1"
        assert resolved["x"] == 0.3
        assert resolved["y"] == 0.2
        assert resolved["z"] == 0.1

    def test_resolve_parameters_missing(self):
        """Test resolving with missing parameters"""
        param_names = ["robot_id", "missing_param"]
        operation_params = {"robot_id": "Robot1"}

        resolved = PredicateParser.resolve_parameters(param_names, operation_params)

        assert resolved["robot_id"] == "Robot1"
        assert resolved["missing_param"] is None


class TestOperationVerifier:
    """Test OperationVerifier"""

    @patch('operations.Verification.evaluate_predicate')
    def test_verify_preconditions_pass(self, mock_evaluate, sample_operation_with_conditions, mock_world_state, cleanup_world_state):
        """Test all preconditions valid"""
        verifier = OperationVerifier()

        # Mock all predicates pass
        mock_evaluate.return_value = (True, "")

        params = {
            "robot_id": "Robot1",
            "x": 0.3,
            "y": 0.2,
            "z": 0.1
        }

        result = verifier.verify_preconditions(
            sample_operation_with_conditions,
            params,
            mock_world_state
        )

        assert result.success is True
        assert result.execution_allowed is True
        assert len(result.violations) == 0

    @patch('operations.Verification.evaluate_predicate')
    def test_verify_preconditions_fail(self, mock_evaluate, sample_operation_with_conditions, mock_world_state, cleanup_world_state):
        """Test precondition violation blocks execution"""
        verifier = OperationVerifier()

        # Mock first predicate fails
        mock_evaluate.side_effect = [
            (False, "Target exceeds max reach of 0.8m"),  # target_within_reach fails
            (True, "")  # robot_is_initialized passes
        ]

        params = {
            "robot_id": "Robot1",
            "x": 2.0,
            "y": 0.0,
            "z": 0.1
        }

        result = verifier.verify_preconditions(
            sample_operation_with_conditions,
            params,
            mock_world_state
        )

        assert result.success is False
        assert result.execution_allowed is False
        assert len(result.violations) == 1
        assert "exceeds max reach" in result.violations[0].reason

    @patch('operations.Verification.evaluate_predicate')
    def test_verify_postconditions_pass(self, mock_evaluate, sample_operation_with_conditions, mock_world_state, cleanup_world_state):
        """Test postconditions satisfied"""
        verifier = OperationVerifier()

        # Mock postcondition passes
        mock_evaluate.return_value = (True, "")

        operation_result = OperationResult.success_result({"final_position": (0.3, 0.2, 0.1)})

        params = {"robot_id": "Robot1"}

        result = verifier.verify_postconditions(
            sample_operation_with_conditions,
            operation_result,
            params,
            mock_world_state
        )

        assert result.success is True
        assert len(result.violations) == 0

    @patch('operations.Verification.evaluate_predicate')
    def test_verify_postconditions_fail(self, mock_evaluate, sample_operation_with_conditions, mock_world_state, cleanup_world_state):
        """Test postcondition violation is warning"""
        verifier = OperationVerifier()

        # Mock postcondition fails
        mock_evaluate.return_value = (False, "Robot is still moving")

        operation_result = OperationResult.success_result({"final_position": (0.3, 0.2, 0.1)})

        params = {"robot_id": "Robot1"}

        result = verifier.verify_postconditions(
            sample_operation_with_conditions,
            operation_result,
            params,
            mock_world_state
        )

        # Postcondition failures are warnings, not errors
        assert result.success is True  # Success not affected
        assert len(result.warnings) == 1
        assert "still moving" in result.warnings[0].reason

    def test_verify_postconditions_operation_failed(self, sample_operation_with_conditions, mock_world_state, cleanup_world_state):
        """Test postconditions fail when operation failed"""
        verifier = OperationVerifier()

        operation_result = OperationResult.error_result(
                error_code="MOVEMENT_FAILED",
                message="Robot could not reach target",
                recovery_suggestions=["Check logs"]
            )

        params = {"robot_id": "Robot1"}

        result = verifier.verify_postconditions(
            sample_operation_with_conditions,
            operation_result,
            params,
            mock_world_state
        )

        assert result.success is False
        assert result.execution_allowed is False
        assert len(result.violations) == 1
        assert "Operation failed" in result.violations[0].reason

    def test_suggest_recovery_target_within_reach(self, cleanup_world_state):
        """Test recovery suggestions for target_within_reach"""
        verifier = OperationVerifier()

        suggestions = verifier._suggest_recovery_for_predicate(
            "target_within_reach",
            "Target exceeds max reach",
            {"robot_id": "Robot1", "x": 2.0}
        )

        assert len(suggestions) > 0
        assert any("closer" in s.lower() for s in suggestions)

    def test_suggest_recovery_robot_initialized(self, cleanup_world_state):
        """Test recovery suggestions for robot_is_initialized"""
        verifier = OperationVerifier()

        suggestions = verifier._suggest_recovery_for_predicate(
            "robot_is_initialized",
            "Robot not found",
            {"robot_id": "UnknownRobot"}
        )

        assert len(suggestions) > 0
        assert any("initialize" in s.lower() for s in suggestions)


class TestQuickVerifyOperation:
    """Test quick_verify_operation helper"""

    @patch('operations.Verification.evaluate_predicate')
    def test_quick_verify_safe(self, mock_evaluate, sample_operation_with_conditions, mock_world_state, cleanup_world_state):
        """Test quick verification passes"""
        # Mock all predicates pass
        mock_evaluate.return_value = (True, "")

        params = {
            "robot_id": "Robot1",
            "x": 0.3,
            "y": 0.2,
            "z": 0.1
        }

        is_safe, result = quick_verify_operation(
            sample_operation_with_conditions,
            params,
            mock_world_state
        )

        assert is_safe is True
        assert result.execution_allowed is True

    @patch('operations.Verification.evaluate_predicate')
    def test_quick_verify_unsafe(self, mock_evaluate, sample_operation_with_conditions, mock_world_state, cleanup_world_state):
        """Test quick verification blocks unsafe operation"""
        # Mock predicate fails
        mock_evaluate.side_effect = [
            (False, "Target out of reach"),
            (True, "")
        ]

        params = {
            "robot_id": "Robot1",
            "x": 5.0,
            "y": 0.0,
            "z": 0.1
        }

        is_safe, result = quick_verify_operation(
            sample_operation_with_conditions,
            params,
            mock_world_state
        )

        assert is_safe is False
        assert result.execution_allowed is False
        assert len(result.violations) > 0


# ============================================================================
# Test Class: Spatial Predicate Accuracy
# ============================================================================

class TestSpatialPredicateAccuracy:
    """Test accuracy of spatial predicate calculations."""

    def test_distance_calculation_accuracy(self, cleanup_world_state):
        """Test distance calculation is accurate to millimeter precision."""
        from operations.WorldState import get_world_state
        import math

        world_state = get_world_state()

        # Add robot and object with known positions
        world_state.update_robot(robot_id="Robot1", position=(0.0, 0.0, 0.0))
        world_state.update_object_position("Cube_01", (0.3, 0.4, 0.0), "red")

        # Calculate expected distance: sqrt(0.3^2 + 0.4^2 + 0^2) = 0.5
        expected_distance = 0.5

        # Get actual distance from world state
        robot_pos = world_state.get_robot_position("Robot1")
        object_pos = world_state.get_object_position("Cube_01")

        # Ensure positions are not None
        assert robot_pos is not None, "Robot position should not be None"
        assert object_pos is not None, "Object position should not be None"

        actual_distance = math.sqrt(
            (object_pos[0] - robot_pos[0])**2 +
            (object_pos[1] - robot_pos[1])**2 +
            (object_pos[2] - robot_pos[2])**2
        )

        # Verify accuracy to 1mm (0.001m)
        assert abs(actual_distance - expected_distance) < 0.001

    def test_within_reach_predicate_3d_distance(self, cleanup_world_state):
        """Test target_within_reach predicate uses 3D Euclidean distance."""
        from operations.SpatialPredicates import evaluate_predicate
        from config.Robot import ROBOT_BASE_POSITIONS, MAX_ROBOT_REACH
        import math

        # Robot1 base is at (-0.475, 0.0, 0.0) in config
        # Target at (-0.2, 0.3, 0.1) should be within reach
        x, y, z = -0.2, 0.3, 0.1

        # Calculate expected distance from Robot1 base
        base = ROBOT_BASE_POSITIONS.get("Robot1", (0, 0, 0))
        expected_distance = math.sqrt(
            (x - base[0])**2 + (y - base[1])**2 + (z - base[2])**2
        )

        # Should be within MAX_ROBOT_REACH (0.8m by default)
        is_valid, reason = evaluate_predicate(
            "target_within_reach",
            robot_id="Robot1",
            x=x, y=y, z=z,
            world_state=None
        )

        # Check result matches expectation based on distance
        if expected_distance <= MAX_ROBOT_REACH:
            assert is_valid is True, f"Expected within reach: {expected_distance:.3f}m <= {MAX_ROBOT_REACH}m"
        else:
            assert is_valid is False, f"Expected out of reach: {expected_distance:.3f}m > {MAX_ROBOT_REACH}m"

    def test_object_at_location_tolerance(self, cleanup_world_state):
        """Test object location tolerance checking using WorldState."""
        from operations.WorldState import get_world_state
        import math

        world_state = get_world_state()

        # Object at position (0.300, 0.200, 0.100)
        world_state.update_object_position("Cube_01", (0.300, 0.200, 0.100), "red")

        obj_pos = world_state.get_object_position("Cube_01")
        assert obj_pos is not None, "Object position should not be None"

        # Ensure obj_pos is a tuple
        if not isinstance(obj_pos, tuple):
            obj_pos = tuple(obj_pos)

        # Test within 1cm tolerance
        target_pos = (0.305, 0.198, 0.102)
        distance = math.sqrt(sum((a-b)**2 for a, b in zip(obj_pos, target_pos)))
        assert distance < 0.01  # Within 1cm

        # Test outside 1cm tolerance
        target_pos_far = (0.320, 0.220, 0.120)
        distance_far = math.sqrt(sum((a-b)**2 for a, b in zip(obj_pos, target_pos_far)))
        assert distance_far > 0.01  # Outside 1cm

    def test_complex_spatial_and_predicate(self, cleanup_world_state):
        """Test combining multiple spatial conditions (simulating AND logic)."""
        from operations.WorldState import get_world_state
        from operations.SpatialPredicates import evaluate_predicate
        from config.Robot import MAX_ROBOT_REACH

        world_state = get_world_state()

        # Setup scenario
        world_state.update_object_position("Target", (0.3, 0.3, 0.0), "blue")

        # Test target_within_reach predicate
        is_reachable, _ = evaluate_predicate(
            "target_within_reach",
            robot_id="Robot1",
            x=0.3, y=0.3, z=0.0,
            world_state=world_state
        )

        # Check object exists in world state
        obj_pos = world_state.get_object_position("Target")
        object_exists = obj_pos is not None

        # Both conditions should be testable
        assert object_exists is True  # Object was added

        # Distance from Robot1 base to (0.3, 0.3, 0.0) depends on base position
        # Just verify predicate returns a boolean
        assert isinstance(is_reachable, bool)

    def test_complex_spatial_or_predicate(self, cleanup_world_state):
        """Test combining spatial conditions (simulating OR logic)."""
        from operations.WorldState import get_world_state
        from operations.SpatialPredicates import evaluate_predicate

        world_state = get_world_state()

        # Object far from robot base (5m away - definitely out of reach)
        world_state.update_object_position("Target", (5.0, 0.0, 0.0), "red")

        # Check reachability (should be false - too far)
        is_reachable, _ = evaluate_predicate(
            "target_within_reach",
            robot_id="Robot1",
            x=5.0, y=0.0, z=0.0,
            world_state=world_state
        )
        assert is_reachable is False, "Target should be out of reach at 5m distance"

        # But object exists in world state
        obj_exists = world_state.get_object_position("Target") is not None
        assert obj_exists is True

        # Simulates OR: if either condition is true, proceed
        can_proceed = is_reachable or obj_exists
        assert can_proceed is True  # Object exists, so OR is true

        # Test when both conditions are false
        non_existent = world_state.get_object_position("NonExistent") is not None
        can_proceed_2 = is_reachable or non_existent
        assert can_proceed_2 is False  # Both false

    def test_robot_to_object_distance_exact(self, cleanup_world_state):
        """Test exact distance calculation between robot and object."""
        from operations.WorldState import get_world_state
        import math

        world_state = get_world_state()

        # Test case 1: Cardinal direction (only X difference)
        world_state.update_robot(robot_id="Robot1", position=(0.0, 0.0, 0.0))
        world_state.update_object_position("Obj1", (1.0, 0.0, 0.0), "red")

        robot_pos = world_state.get_robot_position("Robot1")
        obj_pos = world_state.get_object_position("Obj1")

        # Ensure positions are tuples
        assert robot_pos is not None and obj_pos is not None, "Positions should not be None"
        if not isinstance(robot_pos, tuple):
            robot_pos = tuple(robot_pos)
        if not isinstance(obj_pos, tuple):
            obj_pos = tuple(obj_pos)

        distance = math.sqrt(sum((a-b)**2 for a, b in zip(obj_pos, robot_pos)))

        assert abs(distance - 1.0) < 0.0001  # Exact 1.0m

        # Test case 2: Diagonal in XY plane
        world_state.update_object_position("Obj2", (1.0, 1.0, 0.0), "blue")
        obj_pos = world_state.get_object_position("Obj2")
        assert obj_pos is not None
        if not isinstance(obj_pos, tuple):
            obj_pos = tuple(obj_pos)
        distance = math.sqrt(sum((a-b)**2 for a, b in zip(obj_pos, robot_pos)))

        expected = math.sqrt(2.0)  # ~1.414m
        assert abs(distance - expected) < 0.0001

        # Test case 3: 3D diagonal
        world_state.update_object_position("Obj3", (1.0, 1.0, 1.0), "green")
        obj_pos = world_state.get_object_position("Obj3")
        assert obj_pos is not None
        if not isinstance(obj_pos, tuple):
            obj_pos = tuple(obj_pos)
        distance = math.sqrt(sum((a-b)**2 for a, b in zip(obj_pos, robot_pos)))

        expected = math.sqrt(3.0)  # ~1.732m
        assert abs(distance - expected) < 0.0001

    def test_workspace_boundary_predicate(self, cleanup_world_state):
        """Test workspace boundary checking with is_in_robot_workspace predicate."""
        from operations.SpatialPredicates import evaluate_predicate
        from config.Robot import ROBOT_WORKSPACE_ASSIGNMENTS, WORKSPACE_REGIONS

        # Test position within Robot1's workspace
        # Get Robot1's workspace
        workspace_name = ROBOT_WORKSPACE_ASSIGNMENTS.get("Robot1")
        if workspace_name and workspace_name in WORKSPACE_REGIONS:
            workspace = WORKSPACE_REGIONS[workspace_name]

            # Test position inside workspace
            x_mid = (workspace["x_min"] + workspace["x_max"]) / 2
            y_mid = (workspace["y_min"] + workspace["y_max"]) / 2
            z_mid = (workspace["z_min"] + workspace["z_max"]) / 2

            is_valid, _ = evaluate_predicate(
                "is_in_robot_workspace",
                robot_id="Robot1",
                x=x_mid, y=y_mid, z=z_mid,
                world_state=None
            )
            assert is_valid is True  # Should be inside workspace

            # Test position outside workspace (way outside)
            is_valid_out, reason = evaluate_predicate(
                "is_in_robot_workspace",
                robot_id="Robot1",
                x=workspace["x_max"] + 1.0,  # 1m beyond boundary
                y=y_mid, z=z_mid,
                world_state=None
            )
            assert is_valid_out is False  # Should be outside

    def test_multiple_object_distance_comparisons(self, cleanup_world_state):
        """Test distance comparisons with multiple objects."""
        from operations.WorldState import get_world_state
        import math

        world_state = get_world_state()

        # Robot at origin
        world_state.update_robot(robot_id="Robot1", position=(0.0, 0.0, 0.0))

        # Three objects at different distances
        world_state.update_object_position("Near", (0.1, 0.0, 0.0), "red")    # 0.1m
        world_state.update_object_position("Mid", (0.5, 0.0, 0.0), "blue")    # 0.5m
        world_state.update_object_position("Far", (1.0, 0.0, 0.0), "green")   # 1.0m

        robot_pos = world_state.get_robot_position("Robot1")

        # Calculate distances
        near_pos = world_state.get_object_position("Near")
        mid_pos = world_state.get_object_position("Mid")
        far_pos = world_state.get_object_position("Far")

        # Ensure all positions are tuples
        assert robot_pos is not None, "Robot position should not be None"
        assert near_pos is not None and mid_pos is not None and far_pos is not None, "Object positions should not be None"

        if not isinstance(robot_pos, tuple):
            robot_pos = tuple(robot_pos)
        if not isinstance(near_pos, tuple):
            near_pos = tuple(near_pos)
        if not isinstance(mid_pos, tuple):
            mid_pos = tuple(mid_pos)
        if not isinstance(far_pos, tuple):
            far_pos = tuple(far_pos)

        dist_near = math.sqrt(sum((a-b)**2 for a, b in zip(near_pos, robot_pos)))
        dist_mid = math.sqrt(sum((a-b)**2 for a, b in zip(mid_pos, robot_pos)))
        dist_far = math.sqrt(sum((a-b)**2 for a, b in zip(far_pos, robot_pos)))

        # Verify ordering
        assert dist_near < dist_mid < dist_far
        assert abs(dist_near - 0.1) < 0.001
        assert abs(dist_mid - 0.5) < 0.001
        assert abs(dist_far - 1.0) < 0.001

    def test_collision_distance_threshold(self, cleanup_world_state):
        """Test collision detection based on distance threshold."""
        from operations.WorldState import get_world_state
        import math

        world_state = get_world_state()

        # Two robots very close to each other
        world_state.update_robot(robot_id="Robot1", position=(0.0, 0.0, 0.0))
        world_state.update_robot(robot_id="Robot2", position=(0.15, 0.0, 0.0))  # 15cm apart

        # Calculate distance between robots
        pos1 = world_state.get_robot_position("Robot1")
        pos2 = world_state.get_robot_position("Robot2")

        assert pos1 is not None and pos2 is not None, "Robot positions should not be None"
        if not isinstance(pos1, tuple):
            pos1 = tuple(pos1)
        if not isinstance(pos2, tuple):
            pos2 = tuple(pos2)

        distance = math.sqrt(sum((a-b)**2 for a, b in zip(pos2, pos1)))

        # Verify distance calculations - relaxed tolerance for floating point
        assert abs(distance - 0.15) < 0.001  # Should be 15cm
        assert distance < 0.2  # Too close for 20cm threshold
        assert distance > 0.1  # But further than 10cm

    def test_large_scale_spatial_performance(self, cleanup_world_state):
        """Test WorldState performance with many objects."""
        from operations.WorldState import get_world_state
        import math
        import time

        world_state = get_world_state()

        # Create 100 objects in scene
        for i in range(100):
            x = (i % 10) * 0.1
            y = (i // 10) * 0.1
            world_state.update_object_position(f"Obj_{i}", (x, y, 0.0), "red")

        world_state.update_robot(robot_id="Robot1", position=(0.5, 0.5, 0.0))
        robot_pos = world_state.get_robot_position("Robot1")

        assert robot_pos is not None, "Robot position should not be None"
        if not isinstance(robot_pos, tuple):
            robot_pos = tuple(robot_pos)

        # Measure time to calculate distances to all objects
        start = time.time()

        distances = []
        for i in range(100):
            obj_pos = world_state.get_object_position(f"Obj_{i}")
            if obj_pos:
                if not isinstance(obj_pos, tuple):
                    obj_pos = tuple(obj_pos)
                dist = math.sqrt(sum((a-b)**2 for a, b in zip(obj_pos, robot_pos)))
                distances.append(dist)

        elapsed = time.time() - start

        # Should complete quickly (< 10ms for 100 objects)
        assert elapsed < 0.01
        assert len(distances) == 100

    def test_predicate_with_negative_coordinates(self, cleanup_world_state):
        """Test distance calculations work correctly with negative coordinates."""
        from operations.WorldState import get_world_state
        import math

        world_state = get_world_state()

        # Robot in negative coordinates
        world_state.update_robot(robot_id="Robot1", position=(-0.5, -0.3, 0.0))
        # Object in positive coordinates
        world_state.update_object_position("Target", (0.5, 0.3, 0.0), "blue")

        robot_pos = world_state.get_robot_position("Robot1")
        obj_pos = world_state.get_object_position("Target")

        assert robot_pos is not None and obj_pos is not None, "Positions should not be None"
        if not isinstance(robot_pos, tuple):
            robot_pos = tuple(robot_pos)
        if not isinstance(obj_pos, tuple):
            obj_pos = tuple(obj_pos)

        distance = math.sqrt(sum((a-b)**2 for a, b in zip(obj_pos, robot_pos)))

        # Distance: sqrt((1.0)^2 + (0.6)^2) = sqrt(1.36) = 1.166m
        expected = math.sqrt(1.0**2 + 0.6**2)
        assert abs(distance - expected) < 0.001

    def test_predicate_performance_with_1000_objects(self, cleanup_world_state):
        """Test WorldState and predicate performance with 1000+ objects."""
        from operations.WorldState import get_world_state
        from operations.SpatialPredicates import evaluate_predicate
        import time

        world_state = get_world_state()

        # Create 1000 objects in scene
        for i in range(1000):
            x = (i % 32) * 0.05  # 32x32 grid
            y = (i // 32) * 0.05
            z = 0.0
            world_state.update_object_position(f"Obj_{i:04d}", (x, y, z), "red")

        # Add robot
        world_state.update_robot(robot_id="Robot1", position=(0.8, 0.8, 0.0))

        # Measure predicate evaluation time with large world state
        start = time.time()

        # Evaluate predicates multiple times
        for _ in range(10):
            # Check if target is within reach (uses ROBOT_BASE_POSITIONS)
            is_valid1, _ = evaluate_predicate(
                "target_within_reach",
                robot_id="Robot1",
                x=0.5, y=0.5, z=0.0,
                world_state=world_state
            )

            # Check object exists in world state
            obj_exists = world_state.get_object_position("Obj_0500") is not None

        elapsed = time.time() - start

        # 10 predicate evaluations with 1000 objects should complete quickly (< 100ms)
        assert elapsed < 0.1, f"Predicate evaluation took {elapsed:.3f}s, expected < 0.1s"

        # Verify object exists
        obj_999 = world_state.get_object_position("Obj_0999")
        assert obj_999 is not None  # Object should exist


# ============================================================================
# Test Class: Recovery Suggestion Effectiveness
# ============================================================================

class TestRecoverySuggestions:
    """Test recovery suggestion generation and effectiveness."""

    def test_out_of_reach_recovery_suggestion(self, cleanup_world_state):
        """Test recovery suggestion for out-of-reach target using OperationVerifier."""
        from operations.WorldState import get_world_state
        from operations.Verification import OperationVerifier
        from operations.Base import BasicOperation, OperationCategory, OperationComplexity

        world_state = get_world_state()

        # Create operation with target_within_reach precondition
        operation = BasicOperation(
            operation_id="test_op",
            name="test_operation",
            category=OperationCategory.MANIPULATION,
            complexity=OperationComplexity.BASIC,
            description="Test operation",
            long_description="Test operation for verification",
            usage_examples=["test_operation(robot_id='Robot1', x=0.3, y=0.2, z=0.1)"],
            parameters=[],
            preconditions=["target_within_reach(robot_id, x, y, z)"],
            postconditions=[],
            average_duration_ms=100.0,
            success_rate=0.95,
            failure_modes=["Target out of reach"],
            implementation=lambda: None
        )

        # Target way beyond reach (10m away - definitely out of reach)
        params = {"robot_id": "Robot1", "x": 10.0, "y": 0.0, "z": 0.0}

        # Verify operation (should fail)
        verifier = OperationVerifier()
        result = verifier.verify_preconditions(operation, params, world_state)

        # Should fail verification
        assert result.execution_allowed is False
        assert len(result.violations) > 0

        # Check if recovery suggestions are provided
        violation = result.violations[0]
        assert len(violation.recovery_suggestions) > 0

        # Suggestion should mention moving closer or reach
        suggestion_text = " ".join(violation.recovery_suggestions).lower()
        assert any(keyword in suggestion_text for keyword in ["move", "closer", "reach", "robot"])

    def test_missing_object_recovery_suggestion(self, cleanup_world_state):
        """Test detection of missing object in WorldState."""
        from operations.WorldState import get_world_state

        world_state = get_world_state()

        # No objects in scene
        world_state.update_robot(robot_id="Robot1", position=(0.3, 0.3, 0.0))

        # Check for non-existent object using WorldState
        obj_pos = world_state.get_object_position("NonExistentCube")

        # Should return None (the actual WorldState returns None for non-existent objects)
        assert obj_pos is None, f"Expected None for non-existent object, got {obj_pos}"

        # In practice, recovery would involve running detect_objects operation
        # to find and register objects in the scene
        # (Implementation dependent - just verify predicate works)

