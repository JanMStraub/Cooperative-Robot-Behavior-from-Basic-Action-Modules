#!/usr/bin/env python3
"""
Negotiation Plan Verifier
==========================

Plan-level verification for negotiated multi-robot plans.
Checks structural validity (signal/wait pairs, variable flow,
parallel group ordering) and spatial safety before execution.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

from .WorldState import get_world_state
from .CoordinationVerifier import CoordinationVerifier
from config.Robot import ROBOT_BASE_POSITIONS, MAX_ROBOT_REACH, MIN_ROBOT_SEPARATION

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class PlanVerificationResult:
    """
    Result of verifying a negotiated plan.

    Attributes:
        valid: True if plan passes all structural checks
        errors: Blocking issues that prevent execution
        warnings: Non-blocking concerns
        safety_check: True if spatial safety checks passed
    """

    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    safety_check: bool = True

    def add_error(self, msg: str):
        """Add an error and mark plan as invalid."""
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str):
        """Add a non-blocking warning."""
        self.warnings.append(msg)


# ============================================================================
# Negotiation Verifier
# ============================================================================


class NegotiationVerifier:
    """
    Verifies structural correctness and spatial safety of negotiated plans.

    Checks:
    - All operations exist in registry
    - Signal/wait_for_signal pairs are matched
    - Variable definitions precede variable usage
    - Parallel groups are ordered correctly
    - Spatial safety (reach, collision) for all robots
    """

    def __init__(self):
        """Initialize the verifier."""
        self._coordination_verifier = CoordinationVerifier()

    def verify_plan(
        self,
        commands: List[Dict[str, Any]],
        world_state=None,
    ) -> PlanVerificationResult:
        """
        Verify a negotiated plan before execution.

        Args:
            commands: List of command dicts with operation, params, etc.
            world_state: Optional WorldState instance

        Returns:
            PlanVerificationResult with errors and warnings
        """
        if world_state is None:
            world_state = get_world_state()

        result = PlanVerificationResult()

        if not commands:
            result.add_error("Empty plan (no commands)")
            return result

        # Run all structural checks
        op_errors = self._verify_operations_exist(commands)
        for err in op_errors:
            result.add_error(err)

        signal_errors = self._verify_signal_wait_pairs(commands)
        for err in signal_errors:
            result.add_error(err)

        var_errors = self._verify_variable_flow(commands)
        for err in var_errors:
            result.add_error(err)

        group_errors = self._verify_parallel_group_ordering(commands)
        for err in group_errors:
            result.add_error(err)

        # Run spatial safety checks
        safety_errors, safety_warnings = self._verify_spatial_safety(
            commands, world_state
        )
        for err in safety_errors:
            result.add_error(err)
            result.safety_check = False
        for warn in safety_warnings:
            result.add_warning(warn)

        return result

    def _verify_operations_exist(self, commands: List[Dict[str, Any]]) -> List[str]:
        """
        Verify all operations in the plan exist in the registry.

        Args:
            commands: Plan commands

        Returns:
            List of error messages for missing operations
        """
        from core.Imports import get_global_registry

        errors = []
        try:
            registry = get_global_registry()
        except Exception:
            logger.warning("Cannot access operation registry for verification")
            return errors

        for i, cmd in enumerate(commands):
            operation = cmd.get("operation", "")
            if not operation:
                errors.append(f"Command {i}: missing 'operation' field")
                continue
            if registry.get_operation_by_name(operation) is None:
                errors.append(f"Command {i}: unknown operation '{operation}'")

        return errors

    def _verify_signal_wait_pairs(self, commands: List[Dict[str, Any]]) -> List[str]:
        """
        Verify every wait_for_signal has a matching signal.

        Args:
            commands: Plan commands

        Returns:
            List of error messages for unmatched signals
        """
        errors = []
        defined_signals = set()
        waited_signals = set()

        for cmd in commands:
            operation = cmd.get("operation", "")
            params = cmd.get("params", {})

            if operation == "signal":
                event = params.get("event_name")
                if event:
                    defined_signals.add(event)

            elif operation == "wait_for_signal":
                event = params.get("event_name")
                if event:
                    waited_signals.add(event)

        # Check all waited signals have a definition
        unmatched = waited_signals - defined_signals
        for event in unmatched:
            errors.append(f"wait_for_signal('{event}') has no matching signal")

        # Warn about signals nobody waits for (not an error)
        unused = defined_signals - waited_signals
        for event in unused:
            logger.debug(
                f"signal('{event}') has no matching wait_for_signal (harmless)"
            )

        return errors

    def _verify_variable_flow(self, commands: List[Dict[str, Any]]) -> List[str]:
        """
        Verify variables are defined before they are used.

        Args:
            commands: Plan commands

        Returns:
            List of error messages for undefined variable usage
        """
        errors = []
        defined_vars = set()

        # Process commands in execution order (by parallel_group, then index)
        sorted_commands = sorted(
            enumerate(commands),
            key=lambda x: (x[1].get("parallel_group", x[0]), x[0]),
        )

        current_group = None
        group_captures = set()

        for idx, cmd in sorted_commands:
            group = cmd.get("parallel_group", idx)

            # When we enter a new group, commit previous group's captures
            if group != current_group:
                defined_vars.update(group_captures)
                group_captures = set()
                current_group = group

            # Check variable usage in params
            params = cmd.get("params", {})
            for key, val in params.items():
                if isinstance(val, str) and val.startswith("$"):
                    var_name = val[1:].split(".")[0]
                    if var_name not in defined_vars:
                        errors.append(
                            f"Command {idx}: variable ${var_name} used before definition "
                            f"(in {cmd.get('operation', '?')}.{key})"
                        )

            # Track captures
            if "capture_var" in cmd:
                group_captures.add(cmd["capture_var"])

        return errors

    def _verify_parallel_group_ordering(
        self, commands: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Verify parallel group numbers are valid.

        Args:
            commands: Plan commands

        Returns:
            List of error messages for ordering issues
        """
        errors = []
        groups_seen = set()

        for i, cmd in enumerate(commands):
            group = cmd.get("parallel_group")
            if group is not None:
                if not isinstance(group, int):
                    errors.append(
                        f"Command {i}: parallel_group must be an integer, got {type(group).__name__}"
                    )
                else:
                    groups_seen.add(group)

        # Check for gaps (warn only)
        if groups_seen:
            min_g = min(groups_seen)
            max_g = max(groups_seen)
            expected = set(range(min_g, max_g + 1))
            gaps = expected - groups_seen
            if gaps:
                logger.debug(f"Parallel group gaps (non-blocking): {sorted(gaps)}")

        return errors

    def _verify_spatial_safety(
        self,
        commands: List[Dict[str, Any]],
        world_state,
    ) -> Tuple[List[str], List[str]]:
        """
        Verify spatial safety of the plan (reach, collision).

        Args:
            commands: Plan commands
            world_state: WorldState instance

        Returns:
            (errors, warnings) tuple
        """
        errors = []
        warnings = []

        # Collect all move targets per parallel group to check for collisions
        from collections import defaultdict

        group_targets: Dict[int, List[Tuple[str, Tuple[float, float, float]]]] = (
            defaultdict(list)
        )

        for i, cmd in enumerate(commands):
            operation = cmd.get("operation", "")
            params = cmd.get("params", {})
            robot_id = params.get("robot_id", "")
            group = cmd.get("parallel_group", i)

            if operation == "move_to_coordinate":
                x = params.get("x")
                y = params.get("y")
                z = params.get("z")

                # Also handle position as a list/tuple (e.g. "position": [x, y, z])
                if x is None or y is None or z is None:
                    pos_param = params.get("position")
                    if pos_param is not None and len(pos_param) >= 3:
                        x, y, z = pos_param[0], pos_param[1], pos_param[2]

                if x is not None and y is not None and z is not None:
                    # Skip variable references
                    if any(isinstance(v, str) and v.startswith("$") for v in [x, y, z]):
                        continue

                    try:
                        pos = (float(x), float(y), float(z))
                    except (ValueError, TypeError):
                        continue

                    # Check reachability from robot base
                    base = ROBOT_BASE_POSITIONS.get(robot_id)
                    if base:
                        dx = pos[0] - base[0]
                        dy = pos[1] - base[1]
                        dz = pos[2] - base[2]
                        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                        if dist > MAX_ROBOT_REACH:
                            errors.append(
                                f"Command {i}: target {pos} is {dist:.3f}m from "
                                f"{robot_id} base (max reach: {MAX_ROBOT_REACH}m)"
                            )

                    group_targets[group].append((robot_id, pos))

        # Check for concurrent moves that are too close
        for group, targets in group_targets.items():
            if len(targets) < 2:
                continue

            for a_idx in range(len(targets)):
                for b_idx in range(a_idx + 1, len(targets)):
                    r_a, pos_a = targets[a_idx]
                    r_b, pos_b = targets[b_idx]
                    if r_a == r_b:
                        continue

                    dx = pos_a[0] - pos_b[0]
                    dy = pos_a[1] - pos_b[1]
                    dz = pos_a[2] - pos_b[2]
                    dist = (dx * dx + dy * dy + dz * dz) ** 0.5

                    if dist < MIN_ROBOT_SEPARATION:
                        errors.append(
                            f"Parallel group {group}: {r_a} and {r_b} targets "
                            f"are {dist:.3f}m apart (min: {MIN_ROBOT_SEPARATION}m)"
                        )
                    elif dist < MIN_ROBOT_SEPARATION * 2:
                        warnings.append(
                            f"Parallel group {group}: {r_a} and {r_b} targets "
                            f"are close ({dist:.3f}m), consider adding safety margin"
                        )

        return errors, warnings
