#!/usr/bin/env python3
"""
Validators - Parameter validation helpers for robot operations.

Single source of truth for workspace bounds and common validation logic.
Each function returns ``None`` on success or an ``OperationResult`` error
on failure, enabling the walrus-operator pattern::

    if err := validate_robot_id(robot_id):
        return err
    if err := validate_xyz(x, y, z):
        return err

Workspace coordinate system (robot base frame, metres):
    X — forward / back  : [-0.65, 0.65]
    Y — height          : [0.0,   0.7 ]
    Z — left / right    : [-0.5,  0.5 ]
"""

from typing import Optional

from .Base import OperationResult

# ---------------------------------------------------------------------------
# Workspace bounds — single source of truth
# ---------------------------------------------------------------------------

WORKSPACE_X: tuple = (-0.65, 0.65)
WORKSPACE_Y: tuple = (0.0, 0.7)
WORKSPACE_Z: tuple = (-0.5, 0.5)
SPEED_RANGE: tuple = (0.1, 2.0)
APPROACH_OFFSET_RANGE: tuple = (0.0, 0.1)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_robot_id(robot_id: str) -> Optional[OperationResult]:
    """
    Validate that robot_id is a non-empty string.

    Args:
        robot_id: Robot identifier to validate.

    Returns:
        None if valid, OperationResult error otherwise.
    """
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
    """
    Validate that x/y/z coordinates are within the robot workspace.

    Args:
        x: X coordinate in metres.
        y: Y coordinate in metres.
        z: Z coordinate in metres.

    Returns:
        None if valid, OperationResult error for the first out-of-range axis.
    """
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
    """
    Validate that speed is within the legal multiplier range.

    Args:
        speed: Speed multiplier to validate.

    Returns:
        None if valid, OperationResult error otherwise.
    """
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
    """
    Validate that approach_offset is within the legal range.

    Args:
        offset: Approach offset in metres to validate.

    Returns:
        None if valid, OperationResult error otherwise.
    """
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
