#!/usr/bin/env python3
"""Plan-level verification for negotiated multi-robot plans (structural + spatial safety)."""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

from .WorldState import get_world_state
from .CoordinationVerifier import CoordinationVerifier
from config.Robot import ROBOT_BASE_POSITIONS, MAX_ROBOT_REACH, MIN_ROBOT_SEPARATION

logger = logging.getLogger(__name__)


@dataclass
class PlanVerificationResult:
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    safety_check: bool = True

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)


class NegotiationVerifier:
    """Verifies structural correctness and spatial safety of negotiated plans."""

    def __init__(self):
        self._coordination_verifier = CoordinationVerifier()

    def verify_plan(
        self,
        commands: List[Dict[str, Any]],
        world_state=None,
    ) -> PlanVerificationResult:
        if world_state is None:
            world_state = get_world_state()

        result = PlanVerificationResult()

        if not commands:
            result.add_error("Empty plan (no commands)")
            return result

        for err in self._verify_operations_exist(commands):
            result.add_error(err)
        for err in self._verify_signal_wait_pairs(commands):
            result.add_error(err)
        for err in self._verify_coordinate_params(commands):
            result.add_error(err)
        for err in self._verify_variable_flow(commands):
            result.add_error(err)
        for err in self._verify_parallel_group_ordering(commands):
            result.add_error(err)

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

    def _verify_coordinate_params(self, commands: List[Dict[str, Any]]) -> List[str]:
        """Reject coordinate-requiring operations that have None or missing x/y/z."""
        errors = []
        COORD_OPS = {"move_to_coordinate", "place_object"}
        for i, cmd in enumerate(commands):
            operation = cmd.get("operation", "")
            if operation not in COORD_OPS:
                continue
            params = cmd.get("params", {})
            missing = [
                k
                for k in ("x", "y", "z")
                if params.get(k) is None and not isinstance(params.get(k), (int, float))
            ]
            # Also treat unresolved $vars as missing (negotiation runs before variable substitution)
            unresolved = [
                k
                for k in ("x", "y", "z")
                if isinstance(params.get(k), str) and params[k].startswith("$")
            ]
            if missing or unresolved:
                errors.append(
                    f"Command {i} ('{operation}'): missing or unresolved coordinate params "
                    f"{missing + unresolved} - use numeric values from world state"
                )
        return errors

    def _verify_signal_wait_pairs(self, commands: List[Dict[str, Any]]) -> List[str]:
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
                else:
                    errors.append(
                        f"signal command missing required 'event_name' parameter"
                    )

            elif operation == "wait_for_signal":
                event = params.get("event_name")
                if event:
                    waited_signals.add(event)
                else:
                    errors.append(
                        f"wait_for_signal command missing required 'event_name' parameter"
                    )

        unmatched = waited_signals - defined_signals
        for event in unmatched:
            errors.append(f"wait_for_signal('{event}') has no matching signal")

        unused = defined_signals - waited_signals
        for event in unused:
            logger.warning(
                f"signal('{event}') has no matching wait_for_signal - "
                f"potential coordination gap"
            )

        return errors

    def _verify_variable_flow(self, commands: List[Dict[str, Any]]) -> List[str]:
        errors = []
        defined_vars = set()

        sorted_commands = sorted(
            enumerate(commands),
            key=lambda x: (x[1].get("parallel_group", x[0]), x[0]),
        )

        current_group = None
        group_captures = set()

        for idx, cmd in sorted_commands:
            group = cmd.get("parallel_group", idx)

            if group != current_group:
                defined_vars.update(group_captures)
                group_captures = set()
                current_group = group

            params = cmd.get("params", {})
            for key, val in params.items():
                if isinstance(val, str) and val.startswith("$"):
                    var_name = val[1:].split(".")[0]
                    if var_name not in defined_vars:
                        errors.append(
                            f"Command {idx}: variable ${var_name} used before definition "
                            f"(in {cmd.get('operation', '?')}.{key})"
                        )

            if "capture_var" in cmd:
                group_captures.add(cmd["capture_var"])

        return errors

    def _verify_parallel_group_ordering(
        self, commands: List[Dict[str, Any]]
    ) -> List[str]:
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
        _world_state,
    ) -> Tuple[List[str], List[str]]:
        errors = []
        warnings = []

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
                    if any(isinstance(v, str) and v.startswith("$") for v in [x, y, z]):
                        continue

                    try:
                        pos = (float(x), float(y), float(z))
                    except (ValueError, TypeError):
                        continue

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
