#!/usr/bin/env python3
"""Parameter validators — return None on success, OperationResult error on failure (walrus-operator friendly)."""

import math
from typing import Optional

from .Base import OperationResult
from config.Robot import ROBOT_BASE_POSITIONS

# Workspace bounds — single source of truth

WORKSPACE_X: tuple = (-0.65, 0.65)
WORKSPACE_Y: tuple = (0.0, 0.7)
WORKSPACE_Z: tuple = (-0.5, 0.5)
SPEED_RANGE: tuple = (0.1, 2.0)
APPROACH_OFFSET_RANGE: tuple = (0.0, 0.1)

# Minimum allowed distance from any robot base (meters). Targets closer than
# this are near-singularity and cause violent joint motion when IK tries to
# solve them — as observed when the LLM sent Robot2 to its own base (0.475,0,0).
BASE_EXCLUSION_RADIUS: float = 0.15

# Validators


def validate_robot_id(robot_id: str) -> Optional[OperationResult]:
    if not robot_id or not isinstance(robot_id, str):
        return OperationResult.error_result(
            "INVALID_ROBOT_ID",
            f"Robot ID must be a non-empty string, got: {robot_id!r}",
            [
                "Provide a valid robot ID (e.g., 'Robot1', 'AR4_Robot')",
                "Check RobotManager in Unity for available robot IDs",
            ],
        )
    return None


def validate_xyz(x: float, y: float, z: float) -> Optional[OperationResult]:
    if not (WORKSPACE_X[0] <= x <= WORKSPACE_X[1]):
        return OperationResult.error_result(
            "INVALID_X_COORDINATE",
            f"X coordinate {x} out of range [{WORKSPACE_X[0]}, {WORKSPACE_X[1]}]",
            [
                f"Adjust X to be within robot workspace [{WORKSPACE_X[0]}, {WORKSPACE_X[1]}]",
                "Use detect_object to get valid coordinates",
            ],
        )
    if not (WORKSPACE_Y[0] <= y <= WORKSPACE_Y[1]):
        return OperationResult.error_result(
            "INVALID_Y_COORDINATE",
            f"Y coordinate {y} out of range [{WORKSPACE_Y[0]}, {WORKSPACE_Y[1]}]",
            [
                f"Adjust Y to be within robot workspace [{WORKSPACE_Y[0]}, {WORKSPACE_Y[1]}]",
                "Use detect_object to get valid coordinates",
            ],
        )
    if not (WORKSPACE_Z[0] <= z <= WORKSPACE_Z[1]):
        return OperationResult.error_result(
            "INVALID_Z_COORDINATE",
            f"Z coordinate {z} out of range [{WORKSPACE_Z[0]}, {WORKSPACE_Z[1]}]",
            [
                f"Adjust Z to be within robot workspace [{WORKSPACE_Z[0]}, {WORKSPACE_Z[1]}]",
                "Z can be negative (below robot base level)",
            ],
        )
    return None


def validate_speed(speed: float) -> Optional[OperationResult]:
    if not (SPEED_RANGE[0] <= speed <= SPEED_RANGE[1]):
        return OperationResult.error_result(
            "INVALID_SPEED",
            f"Speed {speed} out of range [{SPEED_RANGE[0]}, {SPEED_RANGE[1]}]",
            [
                f"Use speed between {SPEED_RANGE[0]} (very slow) and {SPEED_RANGE[1]} (fast)",
                "Typical values: 0.2 (precise), 1.0 (normal), 1.5 (fast)",
            ],
        )
    return None


def validate_approach_offset(offset: float) -> Optional[OperationResult]:
    if not (APPROACH_OFFSET_RANGE[0] <= offset <= APPROACH_OFFSET_RANGE[1]):
        return OperationResult.error_result(
            "INVALID_APPROACH_OFFSET",
            f"Approach offset {offset} out of range "
            f"[{APPROACH_OFFSET_RANGE[0]}, {APPROACH_OFFSET_RANGE[1]}]",
            [
                f"Use offset between {APPROACH_OFFSET_RANGE[0]} (exact position) "
                f"and {APPROACH_OFFSET_RANGE[1]} (10 cm before)",
                "Typical approach offset: 0.05 (5 cm)",
            ],
        )
    return None


def validate_not_near_base(x: float, y: float, z: float) -> Optional[OperationResult]:
    for name, (bx, by, bz) in ROBOT_BASE_POSITIONS.items():
        dist = math.sqrt((x - bx) ** 2 + (y - by) ** 2 + (z - bz) ** 2)
        if dist < BASE_EXCLUSION_RADIUS:
            return OperationResult.error_result(
                "TARGET_NEAR_ROBOT_BASE",
                f"Target ({x:.3f}, {y:.3f}, {z:.3f}) is {dist:.3f}m from {name} base — "
                f"near-singularity, minimum distance is {BASE_EXCLUSION_RADIUS}m",
                [
                    "Use return_to_start_position instead of move_to_coordinate for homing",
                    "Target an object or field coordinate, not a robot base position",
                ],
            )
    return None
