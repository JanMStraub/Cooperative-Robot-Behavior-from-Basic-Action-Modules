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

    @patch('operations.SpatialPredicates.evaluate_predicate')
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

    @patch('operations.SpatialPredicates.evaluate_predicate')
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

    @patch('operations.SpatialPredicates.evaluate_predicate')
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

    @patch('operations.SpatialPredicates.evaluate_predicate')
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

    @patch('operations.SpatialPredicates.evaluate_predicate')
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
