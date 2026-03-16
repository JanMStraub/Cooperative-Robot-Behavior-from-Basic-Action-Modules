#!/usr/bin/env python3
"""
Formal Verification System for Robot Operations
================================================

This module provides formal verification of operation preconditions and
postconditions using predicate logic. It ensures operations are safe to
execute before they run and validates expected outcomes after execution.

Key Components:
- VerificationResult: Structured verification result with violations
- PredicateParser: Parses predicate strings into callable checks
- OperationVerifier: Main verification engine
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from .SpatialPredicates import evaluate_predicate
from .WorldState import get_world_state
from .Base import BasicOperation, OperationResult

# Configure logging
from core.LoggingSetup import get_logger

logger = get_logger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class PredicateViolation:
    """
    Record of a predicate that failed verification.

    Attributes:
        predicate: The predicate expression that failed
        reason: Why it failed
        severity: "error" (blocks execution) or "warning" (allowed but risky)
        recovery_suggestions: List of actions to fix the violation
    """

    predicate: str
    reason: str
    severity: str = "error"  # "error" or "warning"
    recovery_suggestions: List[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    """
    Result of verification check.

    Attributes:
        success: True if all checks passed
        violations: List of failed predicates
        warnings: List of risky but allowed conditions
        checked_predicates: List of all predicates that were checked
        execution_allowed: True if operation can proceed despite warnings
    """

    success: bool
    violations: List[PredicateViolation] = field(default_factory=list)
    warnings: List[PredicateViolation] = field(default_factory=list)
    checked_predicates: List[str] = field(default_factory=list)
    execution_allowed: bool = True

    def add_violation(
        self,
        predicate: str,
        reason: str,
        severity: str = "error",
        suggestions: Optional[List[str]] = None,
    ):
        """Add a violation to the result."""
        violation = PredicateViolation(
            predicate=predicate,
            reason=reason,
            severity=severity,
            recovery_suggestions=suggestions or [],
        )
        if severity == "error":
            self.violations.append(violation)
            self.success = False
            self.execution_allowed = False
        else:  # warning
            self.warnings.append(violation)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "execution_allowed": self.execution_allowed,
            "violations": [
                {
                    "predicate": v.predicate,
                    "reason": v.reason,
                    "severity": v.severity,
                    "recovery_suggestions": v.recovery_suggestions,
                }
                for v in self.violations
            ],
            "warnings": [
                {
                    "predicate": w.predicate,
                    "reason": w.reason,
                    "severity": w.severity,
                    "recovery_suggestions": w.recovery_suggestions,
                }
                for w in self.warnings
            ],
            "checked_predicates": self.checked_predicates,
        }


# ============================================================================
# Predicate Parser
# ============================================================================


class PredicateParser:
    """
    Parser for predicate expressions in preconditions/postconditions.

    Supports expressions like:
    - "target_within_reach(robot_id, x, y, z)"
    - "robot_is_initialized(robot_id)"
    - "gripper_is_open(robot_id)"

    Parameter values are resolved from operation parameters.
    """

    # Regex pattern to parse predicate calls
    PREDICATE_PATTERN = re.compile(r"(\w+)\((.*?)\)")

    @staticmethod
    def parse(predicate_str: str) -> Optional[Tuple[str, List[str]]]:
        """
        Parse predicate string into (predicate_name, parameter_names).

        Args:
            predicate_str: Predicate expression like "target_within_reach(robot_id, x, y, z)"

        Returns:
            (predicate_name, [param1_name, param2_name, ...]) or None if invalid

        Example:
            >>> PredicateParser.parse("target_within_reach(robot_id, x, y, z)")
            ("target_within_reach", ["robot_id", "x", "y", "z"])
        """
        match = PredicateParser.PREDICATE_PATTERN.match(predicate_str.strip())
        if not match:
            return None

        predicate_name = match.group(1)
        params_str = match.group(2).strip()

        # Parse parameters (comma-separated, may have spaces)
        if params_str:
            param_names = [p.strip() for p in params_str.split(",")]
        else:
            param_names = []

        return predicate_name, param_names

    @staticmethod
    def resolve_parameters(
        param_names: List[str], operation_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve parameter names to actual values from operation parameters.

        Args:
            param_names: List of parameter names from predicate
            operation_params: Actual operation parameters

        Returns:
            Dictionary mapping parameter names to values

        Example:
            >>> param_names = ["robot_id", "x", "y", "z"]
            >>> operation_params = {"robot_id": "Robot1", "x": 0.3, "y": 0.2, "z": 0.1}
            >>> PredicateParser.resolve_parameters(param_names, operation_params)
            {"robot_id": "Robot1", "x": 0.3, "y": 0.2, "z": 0.1}
        """
        resolved = {}
        for param_name in param_names:
            if param_name in operation_params:
                resolved[param_name] = operation_params[param_name]
            else:
                # Special case: calculated parameters (e.g., calculated_x from spatial operations)
                # These will be None if not provided
                resolved[param_name] = None

        return resolved


# ============================================================================
# Operation Verifier
# ============================================================================


class OperationVerifier:
    """
    Main verification engine for checking operation safety.

    Verifies preconditions before execution and postconditions after completion.
    Uses predicate system for formal verification.
    """

    def __init__(self):
        """Initialize the verifier."""
        self.world_state = get_world_state()

    def verify_preconditions(
        self, operation: BasicOperation, params: Dict[str, Any], world_state=None
    ) -> VerificationResult:
        """
        Verify operation preconditions before execution.

        Args:
            operation: The operation to verify
            params: Operation parameters
            world_state: Optional WorldState instance (uses global if None)

        Returns:
            VerificationResult with success status and any violations

        Example:
            >>> verifier = OperationVerifier()
            >>> result = verifier.verify_preconditions(move_op, {"robot_id": "Robot1", "x": 0.3, ...})
            >>> if not result.success:
            ...     print("Preconditions failed:", result.violations)
        """
        if world_state is None:
            world_state = self.world_state

        result = VerificationResult(success=True)

        # Check each precondition
        for precondition in operation.preconditions:
            result.checked_predicates.append(precondition)

            # Parse the precondition
            parsed = PredicateParser.parse(precondition)
            if parsed is None:
                # Malformed predicate — log warning and record as a violation so callers
                # are aware that a precondition was skipped rather than evaluated.
                logger.warning(
                    f"Skipping malformed precondition (cannot parse): '{precondition}'"
                )
                result.add_violation(
                    predicate=precondition,
                    reason="Predicate could not be parsed — verify syntax (expected: 'predicate_name(param1, ...)')",
                    severity="warning",
                    suggestions=[
                        "Check precondition syntax: use predicate_name(param) format",
                        "Verify the predicate name is registered in SpatialPredicates",
                    ],
                )
                continue

            predicate_name, param_names = parsed

            # Resolve parameter values
            predicate_params = PredicateParser.resolve_parameters(param_names, params)

            # Add world_state to parameters
            predicate_params["world_state"] = world_state

            # Evaluate the predicate
            try:
                is_valid, reason = evaluate_predicate(
                    predicate_name, **predicate_params
                )

                if not is_valid:
                    # Precondition failed
                    result.add_violation(
                        predicate=precondition,
                        reason=reason,
                        severity="error",
                        suggestions=self._suggest_recovery_for_predicate(
                            predicate_name, reason, params
                        ),
                    )
                    logger.warning(f"Precondition failed: {precondition} - {reason}")
                else:
                    logger.debug(f"Precondition passed: {precondition}")

            except Exception as e:
                logger.error(f"Error evaluating precondition '{precondition}': {e}")
                result.add_violation(
                    predicate=precondition,
                    reason=f"Evaluation error: {str(e)}",
                    severity="error",
                    suggestions=[
                        "Check predicate parameters",
                        "Verify world state is accessible",
                    ],
                )

        return result

    def verify_postconditions(
        self,
        operation: BasicOperation,
        operation_result: OperationResult,
        params: Dict[str, Any],
        world_state=None,
    ) -> VerificationResult:
        """
        Verify operation postconditions after execution.

        Args:
            operation: The operation that was executed
            operation_result: Result from operation execution
            params: Operation parameters that were used
            world_state: Optional WorldState instance

        Returns:
            VerificationResult with success status and any violations

        Example:
            >>> result = verifier.verify_postconditions(move_op, op_result, params)
            >>> if not result.success:
            ...     print("Postconditions violated:", result.violations)
        """
        if world_state is None:
            world_state = self.world_state

        result = VerificationResult(success=True)

        # If operation failed, postconditions are expected to fail
        if not operation_result.success:
            result.add_violation(
                predicate="operation_succeeded",
                reason=f"Operation failed: {operation_result.error}",
                severity="error",
                suggestions=(
                    operation_result.error.get("recovery_suggestions", [])
                    if operation_result.error
                    and isinstance(operation_result.error, dict)
                    else []
                ),
            )
            return result

        # Check each postcondition
        for postcondition in operation.postconditions:
            result.checked_predicates.append(postcondition)

            # Parse the postcondition
            parsed = PredicateParser.parse(postcondition)
            if parsed is None:
                # Malformed predicate — log warning and record as a violation so callers
                # are aware that a postcondition was skipped rather than evaluated.
                logger.warning(
                    f"Skipping malformed postcondition (cannot parse): '{postcondition}'"
                )
                result.add_violation(
                    predicate=postcondition,
                    reason="Predicate could not be parsed — verify syntax (expected: 'predicate_name(param1, ...)')",
                    severity="warning",
                    suggestions=[
                        "Check postcondition syntax: use predicate_name(param) format",
                        "Verify the predicate name is registered in SpatialPredicates",
                    ],
                )
                continue

            predicate_name, param_names = parsed

            # Resolve parameter values
            predicate_params = PredicateParser.resolve_parameters(param_names, params)
            predicate_params["world_state"] = world_state

            # Evaluate the predicate
            try:
                is_valid, reason = evaluate_predicate(
                    predicate_name, **predicate_params
                )

                if not is_valid:
                    # Postcondition failed - may indicate operation didn't complete as expected
                    result.add_violation(
                        predicate=postcondition,
                        reason=reason,
                        severity="warning",  # Postconditions are usually warnings, not blockers
                        suggestions=[
                            "Operation may not have completed fully",
                            "Check robot status",
                            "Consider retrying operation",
                        ],
                    )
                    logger.warning(f"Postcondition failed: {postcondition} - {reason}")

            except Exception as e:
                logger.error(f"Error evaluating postcondition '{postcondition}': {e}")
                result.add_violation(
                    predicate=postcondition,
                    reason=f"Evaluation error: {str(e)}",
                    severity="warning",
                    suggestions=["Check world state", "Verify operation completed"],
                )

        return result

    def _suggest_recovery_for_predicate(
        self, predicate_name: str, failure_reason: str, params: Dict[str, Any]
    ) -> List[str]:
        """
        Generate recovery suggestions for a failed predicate.

        Uses WorldState to provide context-aware suggestions based on current
        robot positions, object locations, and workspace allocations.

        Args:
            predicate_name: Name of failed predicate
            failure_reason: Why it failed
            params: Operation parameters

        Returns:
            List of recovery suggestions
        """
        suggestions = []

        if predicate_name == "target_within_reach":
            # Query WorldState for which robots CAN reach the target
            x, y, z = params.get("x"), params.get("y"), params.get("z")
            if x is not None and y is not None and z is not None:
                from .SpatialPredicates import target_within_reach

                # Check which other robots can reach this target
                for robot_id, state in self.world_state._robot_states.items():
                    if robot_id != params.get("robot_id"):
                        is_valid, _ = target_within_reach(
                            robot_id, x, y, z, world_state=self.world_state
                        )
                        if is_valid:
                            suggestions.append(
                                f"Use {robot_id} instead (target is within reach)"
                            )

            # Add generic suggestions if no specific robot found
            if not suggestions:
                suggestions.extend(
                    [
                        "Move target closer to robot base",
                        "Use a different robot closer to target",
                        f"Current robot: {params.get('robot_id')}",
                        "Consider breaking movement into multiple steps",
                    ]
                )

        elif predicate_name == "robot_is_initialized":
            suggestions.extend(
                [
                    "Initialize robot before commanding movement",
                    "Check Unity RobotManager has robot registered",
                    "Verify robot is powered on and connected",
                ]
            )

        elif predicate_name == "robot_is_stationary":
            suggestions.extend(
                [
                    "Wait for current movement to complete",
                    "Cancel current movement before starting new one",
                    "Check robot is not stuck in motion",
                ]
            )

        elif predicate_name == "is_in_robot_workspace":
            suggestions.extend(
                [
                    "Target position outside robot workspace",
                    "Use move_to_region to navigate to correct workspace",
                    "Consider using shared_zone for handoff operations",
                ]
            )

        elif predicate_name == "object_accessible_by_robot":
            # Suggest alternative nearby accessible objects
            x, y, z = params.get("x"), params.get("y"), params.get("z")
            if x is None or y is None or z is None:
                # Try to get from object_position tuple
                obj_pos = params.get("object_position")
                if obj_pos and len(obj_pos) == 3:
                    x, y, z = obj_pos

            if x is not None and y is not None and z is not None:
                from .SpatialPredicates import object_accessible_by_robot

                # Find nearby objects that ARE accessible
                nearby = self.world_state.find_objects_near((x, y, z), radius=0.15)
                robot_id = params.get("robot_id")

                for obj in nearby:
                    is_valid, _ = object_accessible_by_robot(
                        robot_id, obj.position, world_state=self.world_state
                    )
                    if is_valid:
                        pos = obj.position
                        suggestions.append(
                            f"Try {obj.object_id} at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) instead (accessible)"
                        )

            # Add generic suggestions if no alternatives found
            if not suggestions:
                suggestions.extend(
                    [
                        "Target object not accessible from robot workspace",
                        "Move robot to shared zone for better access",
                        "Consider handoff from another robot",
                    ]
                )

        elif (
            predicate_name == "gripper_is_open" or predicate_name == "gripper_is_closed"
        ):
            suggestions.extend(
                [
                    "Send gripper command to change state",
                    "Use control_gripper operation",
                    "Check gripper is not obstructed",
                ]
            )

        else:
            suggestions.append(f"Address issue: {failure_reason}")

        return suggestions


# ============================================================================
# Utility Functions
# ============================================================================


def quick_verify_operation(
    operation: BasicOperation, params: Dict[str, Any], world_state=None
) -> Tuple[bool, VerificationResult]:
    """
    Quick verification helper for checking if operation is safe to execute.

    Args:
        operation: Operation to verify
        params: Operation parameters
        world_state: Optional WorldState

    Returns:
        (is_safe, verification_result)

    Example:
        >>> is_safe, result = quick_verify_operation(move_op, {"robot_id": "Robot1", ...})
        >>> if is_safe:
        ...     operation.execute(**params)
    """
    verifier = OperationVerifier()
    result = verifier.verify_preconditions(operation, params, world_state)
    return result.execution_allowed, result
