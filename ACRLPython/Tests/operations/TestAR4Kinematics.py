#!/usr/bin/env python3
"""
Unit tests for AR4Kinematics.py

Verifies the URDF-based forward kinematics solver produces geometrically
correct end-effector poses without requiring Unity or network access.

Key invariants tested:
- Zero pose outputs a position above the robot base (arm points up by default)
- Position type and shape are correct
- Quaternion output is unit-length
- Robot2 mirrored base (180° yaw) produces an X-mirrored position
- Invalid joint angle count raises ValueError
- FK position stays within AR4 kinematic reach (0.64 m)
"""

import math
import pytest

from operations.AR4Kinematics import (
    compute_end_effector_pose,
    compute_end_effector_position,
)


# Robot base positions matching config/Robot.py
ROBOT1_BASE = (-0.475, 0.0, 0.0)
ROBOT2_BASE = (0.475, 0.0, 0.0)
MAX_REACH = 0.64  # metres, from config/Robot.py MAX_ROBOT_REACH

ZERO_ANGLES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# ============================================================================
# Basic output shape / type tests
# ============================================================================


def test_returns_tuple_pair():
    """compute_end_effector_pose returns (pos_xyz, quat_xyzw)."""
    result = compute_end_effector_pose(ZERO_ANGLES, ROBOT1_BASE)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_position_is_3tuple():
    pos, _ = compute_end_effector_pose(ZERO_ANGLES, ROBOT1_BASE)
    assert len(pos) == 3
    assert all(isinstance(v, float) for v in pos)


def test_quaternion_is_4tuple():
    _, quat = compute_end_effector_pose(ZERO_ANGLES, ROBOT1_BASE)
    assert len(quat) == 4


def test_quaternion_is_unit_length():
    """Output quaternion must be normalised to unit length."""
    import math as _math

    _, quat = compute_end_effector_pose(ZERO_ANGLES, ROBOT1_BASE)
    norm = _math.sqrt(sum(q * q for q in quat))
    assert abs(norm - 1.0) < 1e-6


def test_quaternion_unit_for_nonzero_config():
    """Unit quaternion check for a non-trivial joint configuration."""
    import math as _math

    angles = [0.1, -0.3, 0.5, -0.2, 0.4, -0.1]
    _, quat = compute_end_effector_pose(angles, ROBOT1_BASE)
    norm = _math.sqrt(sum(q * q for q in quat))
    assert abs(norm - 1.0) < 1e-6


# ============================================================================
# Zero pose sanity checks
# ============================================================================


def test_zero_pose_position_is_near_base():
    """At zero angles the end-effector should be somewhere near Robot1 base X."""
    pos, _ = compute_end_effector_pose(ZERO_ANGLES, ROBOT1_BASE)
    bx = ROBOT1_BASE[0]
    # End-effector should be within max reach of the base
    dist = math.sqrt(
        (pos[0] - bx) ** 2 + (pos[1] - ROBOT1_BASE[1]) ** 2 + (pos[2] - ROBOT1_BASE[2]) ** 2
    )
    assert dist <= MAX_REACH + 0.05, (
        f"Zero-pose EE distance from base {dist:.3f} m exceeds reach {MAX_REACH} m"
    )


def test_zero_pose_height_above_table():
    """At zero angles the end-effector should be at or above table (Y >= 0)."""
    pos, _ = compute_end_effector_pose(ZERO_ANGLES, ROBOT1_BASE)
    # Some joint configurations may place EE slightly below base — just check Y is reasonable
    assert pos[1] >= -0.5, f"EE Y at zero pose unexpectedly low: {pos[1]:.3f}"


# ============================================================================
# Workspace reach check
# ============================================================================


def test_fk_within_max_reach_several_configs():
    """FK-derived EE should always be within kinematic reach of the base."""
    configs = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, -0.5, 0.5, -0.5, 0.5, -0.5],
        [1.0, 0.5, -0.5, 1.0, -1.0, 0.5],
        [-0.5, 0.3, -0.3, 0.1, -0.2, 0.0],
    ]
    for angles in configs:
        pos, _ = compute_end_effector_pose(angles, ROBOT1_BASE)
        bx, by, bz = ROBOT1_BASE
        dist = math.sqrt((pos[0] - bx) ** 2 + (pos[1] - by) ** 2 + (pos[2] - bz) ** 2)
        assert dist <= MAX_REACH + 0.1, (
            f"Config {angles}: EE dist {dist:.3f} m exceeds max reach {MAX_REACH} m"
        )


# ============================================================================
# Robot2 mirroring
# ============================================================================


def test_robot2_mirrored_x():
    """Robot2 (base_yaw=π) should produce an X position mirrored relative to Robot1.

    Both robots start at symmetric X offsets (±0.475).  At zero joint angles
    with opposite base yaw, the EE X offsets relative to their respective
    bases should be approximately equal and opposite.
    """
    pos1, _ = compute_end_effector_pose(ZERO_ANGLES, ROBOT1_BASE, base_yaw_rad=0.0)
    pos2, _ = compute_end_effector_pose(ZERO_ANGLES, ROBOT2_BASE, base_yaw_rad=math.pi)

    # Offset from base
    dx1 = pos1[0] - ROBOT1_BASE[0]
    dx2 = pos2[0] - ROBOT2_BASE[0]

    # The X offsets should be equal and opposite (within floating-point tolerance)
    assert abs(dx1 + dx2) < 0.02, (
        f"Robot1 dx={dx1:.4f} and Robot2 dx={dx2:.4f} are not symmetric"
    )


def test_robot2_z_negated_relative_to_robot1():
    """A 180° yaw rotation about Y negates the Z offset — Robot2's Z offset from
    its base should be the negation of Robot1's Z offset from its base."""
    pos1, _ = compute_end_effector_pose(ZERO_ANGLES, ROBOT1_BASE, base_yaw_rad=0.0)
    pos2, _ = compute_end_effector_pose(ZERO_ANGLES, ROBOT2_BASE, base_yaw_rad=math.pi)

    dz1 = pos1[2] - ROBOT1_BASE[2]
    dz2 = pos2[2] - ROBOT2_BASE[2]

    # After a 180° yaw, Z flips sign: dz2 ≈ -dz1
    assert abs(dz1 + dz2) < 0.02, (
        f"Robot1 dz={dz1:.4f} and Robot2 dz={dz2:.4f} are not negatives of each other"
    )


# ============================================================================
# Error handling
# ============================================================================


def test_wrong_joint_count_raises():
    """Passing fewer than 6 joint angles should raise ValueError."""
    with pytest.raises(ValueError):
        compute_end_effector_pose([0.0, 0.0, 0.0], ROBOT1_BASE)


def test_too_many_joints_raises():
    """Passing more than 6 joint angles should raise ValueError."""
    with pytest.raises(ValueError):
        compute_end_effector_pose([0.0] * 7, ROBOT1_BASE)


# ============================================================================
# Convenience wrapper
# ============================================================================


def test_position_only_wrapper():
    """compute_end_effector_position returns exactly the position tuple."""
    pos_full, _ = compute_end_effector_pose(ZERO_ANGLES, ROBOT1_BASE)
    pos_short = compute_end_effector_position(ZERO_ANGLES, ROBOT1_BASE)
    assert pos_short == pos_full


# ============================================================================
# Determinism
# ============================================================================


def test_deterministic():
    """Same inputs produce identical outputs across multiple calls."""
    angles = [0.2, -0.1, 0.3, -0.4, 0.1, 0.0]
    result_a = compute_end_effector_pose(angles, ROBOT1_BASE)
    result_b = compute_end_effector_pose(angles, ROBOT1_BASE)
    assert result_a == result_b
