#!/usr/bin/env python3
"""Formal pre/postcondition verification using predicate logic — checks operation safety before execution."""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from .SpatialPredicates import evaluate_predicate
from .WorldState import get_world_state
from .Base import BasicOperation, OperationResult

from core.LoggingSetup import get_logger

logger = get_logger(__name__)


@dataclass
class PredicateViolation:
    predicate: str
    reason: str
    severity: str = "error"  # "error" or "warning"
    recovery_suggestions: List[str] = field(default_factory=list)


@dataclass
class VerificationResult:
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


class PredicateParser:
    """Parse predicate strings like "target_within_reach(robot_id, x, y, z)" into name + param list."""

    PREDICATE_PATTERN = re.compile(r"(\w+)\((.*?)\)")

    @staticmethod
    def parse(predicate_str: str) -> Optional[Tuple[str, List[str]]]:
        match = PredicateParser.PREDICATE_PATTERN.match(predicate_str.strip())
        if not match:
            return None

        predicate_name = match.group(1)
        params_str = match.group(2).strip()

        if params_str:
            param_names = [p.strip() for p in params_str.split(",")]
        else:
            param_names = []

        return predicate_name, param_names

    @staticmethod
    def resolve_parameters(
        param_names: List[str], operation_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        resolved = {}
        for param_name in param_names:
            resolved[param_name] = operation_params.get(param_name, None)

        return resolved


class OperationVerifier:
    """Checks pre/postconditions via predicate system before and after operation execution."""

    def __init__(self):
        self.world_state = get_world_state()

    def verify_preconditions(
        self, operation: BasicOperation, params: Dict[str, Any], world_state=None
    ) -> VerificationResult:
        if world_state is None:
            world_state = self.world_state

        result = VerificationResult(success=True)

        for precondition in operation.preconditions:
            result.checked_predicates.append(precondition)

            parsed = PredicateParser.parse(precondition)
            if parsed is None:
                # record violation so callers know a precondition was skipped, not passed
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

            predicate_params = PredicateParser.resolve_parameters(param_names, params)

            # skip if params unresolved (runtime variable refs like "$target.x")
            required_names = [p for p in param_names if p != "world_state"]
            missing_params = [p for p in required_names if p not in predicate_params]
            unresolved_params = [
                p
                for p in required_names
                if p in predicate_params
                and isinstance(predicate_params[p], str)
                and predicate_params[p].startswith("$")
            ]
            if missing_params or unresolved_params:
                logger.warning(
                    f"Skipping precondition '{precondition}': params "
                    f"{missing_params + unresolved_params} not yet resolved "
                    f"(runtime variable references)"
                )
                continue

            predicate_params["world_state"] = world_state

            try:
                is_valid, reason = evaluate_predicate(
                    predicate_name, **predicate_params
                )

                if not is_valid:
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
                logger.error(f"Error evaluating predicate '{precondition}': {e}")
                result.add_violation(
                    predicate=precondition,
                    reason=f"Predicate evaluation error: {str(e)}",
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
        if world_state is None:
            world_state = self.world_state

        result = VerificationResult(success=True)

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

        for postcondition in operation.postconditions:
            result.checked_predicates.append(postcondition)

            parsed = PredicateParser.parse(postcondition)
            if parsed is None:
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

            predicate_params = PredicateParser.resolve_parameters(param_names, params)

            required_names = [p for p in param_names if p != "world_state"]
            missing_params = [p for p in required_names if p not in predicate_params]
            if missing_params:
                logger.warning(
                    f"Skipping postcondition '{postcondition}': params {missing_params} "
                    f"not yet resolved (may be runtime variable references)"
                )
                continue

            predicate_params["world_state"] = world_state

            try:
                is_valid, reason = evaluate_predicate(
                    predicate_name, **predicate_params
                )

                if not is_valid:
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
        suggestions = []

        if predicate_name == "target_within_reach":
            # Query WorldState for which robots CAN reach the target
            x, y, z = params.get("x"), params.get("y"), params.get("z")
            if x is not None and y is not None and z is not None:
                from .SpatialPredicates import target_within_reach

                # Check which other robots can reach this target
                for robot_id, _state in self.world_state._robot_states.items():
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


# Utility Functions


def quick_verify_operation(
    operation: BasicOperation, params: Dict[str, Any], world_state=None
) -> Tuple[bool, VerificationResult]:
    """Returns (is_safe, verification_result) for operation precondition check."""
    verifier = OperationVerifier()
    result = verifier.verify_preconditions(operation, params, world_state)
    return result.execution_allowed, result
