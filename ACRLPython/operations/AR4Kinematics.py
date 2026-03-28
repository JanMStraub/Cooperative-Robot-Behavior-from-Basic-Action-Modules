#!/usr/bin/env python3
"""
AR4 Forward Kinematics
======================

Pure-NumPy FK solver for the AR4 6-DOF arm derived from the URDF joint chain:
  ACRLUnity/Assets/Prefabs/ar4_urdf/ar4.urdf

Each revolute joint transform is computed as:
    T_i = Translate(xyz_i) @ RPY_to_matrix(rpy_i) @ AxisAngle(axis_i, θ_i)

The full chain:
    T_world_ee = T_base * T_j1(θ1) * T_j2(θ2) * ... * T_j6(θ6) * T_gripper_fixed

Coordinate frames:
- URDF/ROS: right-handed (X-right, Y-up/forward per joint convention)
- Unity:    left-handed  (X-right, Y-up, Z-forward)

After computing FK in ROS frame, negate X of position and quat.x to convert
to Unity frame — same pattern as GraspFrameTransform.py:202-204.
"""

import math
import logging
from typing import Tuple

import numpy as np

from operations.GraspFrameTransform import _quat_multiply, _normalise_quat
from core.LoggingSetup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# URDF joint definitions (verified from ar4.urdf)
# Each entry: (origin_xyz, origin_rpy, axis_xyz)
# Fixed joints have axis=None.
# ---------------------------------------------------------------------------

_JOINT_PARAMS = [
    # joint_1: revolute
    ((0.0, 0.0, 0.0), (math.pi, 0.0, 0.0), (0.0, 0.0, 1.0)),
    # joint_2: revolute
    ((0.0, 0.0642, -0.16977), (math.pi / 2, 0.0, -math.pi / 2), (0.0, 0.0, -1.0)),
    # joint_3: revolute
    ((0.0, -0.305, 0.007), (0.0, 0.0, math.pi), (0.0, 0.0, -1.0)),
    # joint_4: revolute
    ((0.0, 0.0, 0.0), (math.pi / 2, 0.0, -math.pi / 2), (0.0, 0.0, -1.0)),
    # joint_5: revolute
    ((0.0, 0.0, -0.22263), (math.pi, 0.0, -math.pi / 2), (1.0, 0.0, 0.0)),
    # joint_6: revolute
    ((0.0, 0.0, 0.041), (0.0, 0.0, math.pi), (0.0, 0.0, 1.0)),
    # ee_joint: fixed (origin rpy="0 0 0" xyz="0 0 0")
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None),
    # gripper_base_joint: fixed (origin rpy="-1.5708 0 0" xyz="0 0 0")
    ((0.0, 0.0, 0.0), (-math.pi / 2, 0.0, 0.0), None),
]

# Number of revolute joints (first 6 entries)
_NUM_JOINTS = 6


# ---------------------------------------------------------------------------
# Low-level transform helpers
# ---------------------------------------------------------------------------


def _translation_matrix(xyz: Tuple[float, float, float]) -> np.ndarray:
    """Build a 4×4 homogeneous translation matrix.

    Args:
        xyz: Translation (x, y, z).

    Returns:
        4×4 homogeneous translation matrix.
    """
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = xyz[0]
    T[1, 3] = xyz[1]
    T[2, 3] = xyz[2]
    return T


def _rpy_to_matrix(rpy: Tuple[float, float, float]) -> np.ndarray:
    """Convert intrinsic fixed-frame RPY angles (rad) to a 4×4 rotation matrix.

    URDF uses fixed-axis (extrinsic) RPY: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).

    Args:
        rpy: (roll, pitch, yaw) in radians.

    Returns:
        4×4 homogeneous rotation matrix.
    """
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # Rz @ Ry @ Rx  (extrinsic, which is the URDF convention)
    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def _axis_angle_matrix(axis: Tuple[float, float, float], theta: float) -> np.ndarray:
    """Build a 4×4 rotation matrix for rotation of *theta* rad around *axis*.

    Uses Rodrigues' formula.

    Args:
        axis: Unit rotation axis (x, y, z).
        theta: Rotation angle in radians.

    Returns:
        4×4 homogeneous rotation matrix.
    """
    ax, ay, az = axis
    c, s = math.cos(theta), math.sin(theta)
    t = 1.0 - c
    R = np.array(
        [
            [t * ax * ax + c, t * ax * ay - s * az, t * ax * az + s * ay],
            [t * ax * ay + s * az, t * ay * ay + c, t * ay * az - s * ax],
            [t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def _mat_to_quaternion(R3: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a quaternion [x, y, z, w].

    Uses the Shepperd method for numerical stability.

    Args:
        R3: 3×3 rotation matrix.

    Returns:
        Quaternion [x, y, z, w].
    """
    trace = R3[0, 0] + R3[1, 1] + R3[2, 2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R3[2, 1] - R3[1, 2]) * s
        y = (R3[0, 2] - R3[2, 0]) * s
        z = (R3[1, 0] - R3[0, 1]) * s
    elif R3[0, 0] > R3[1, 1] and R3[0, 0] > R3[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R3[0, 0] - R3[1, 1] - R3[2, 2])
        w = (R3[2, 1] - R3[1, 2]) / s
        x = 0.25 * s
        y = (R3[0, 1] + R3[1, 0]) / s
        z = (R3[0, 2] + R3[2, 0]) / s
    elif R3[1, 1] > R3[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R3[1, 1] - R3[0, 0] - R3[2, 2])
        w = (R3[0, 2] - R3[2, 0]) / s
        x = (R3[0, 1] + R3[1, 0]) / s
        y = 0.25 * s
        z = (R3[1, 2] + R3[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R3[2, 2] - R3[0, 0] - R3[1, 1])
        w = (R3[1, 0] - R3[0, 1]) / s
        x = (R3[0, 2] + R3[2, 0]) / s
        y = (R3[1, 2] + R3[2, 1]) / s
        z = 0.25 * s
    return _normalise_quat(np.array([x, y, z, w], dtype=np.float64))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_end_effector_pose(
    joint_angles: list,
    base_position: Tuple[float, float, float],
    base_yaw_rad: float = 0.0,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """Compute the end-effector pose using AR4 forward kinematics.

    Traverses the full URDF chain (joint_1 → joint_6 → ee_joint →
    gripper_base_joint) and returns the gripper-base frame pose in Unity's
    left-handed world frame.

    Args:
        joint_angles: 6 joint angles in radians, ROS joint convention
            (joint_1 … joint_6).
        base_position: World position of the robot base in Unity frame (x, y, z).
        base_yaw_rad: Additional yaw rotation of the whole robot base about the
            world Y-axis (radians).  Pass ``math.pi`` for Robot2 which is
            mounted mirrored (180° yaw from Robot1).

    Returns:
        Tuple of:
        - position (x, y, z) in Unity world frame (metres)
        - rotation quaternion (x, y, z, w) in Unity world frame

    Raises:
        ValueError: If ``joint_angles`` does not have exactly 6 elements.
    """
    if len(joint_angles) != _NUM_JOINTS:
        raise ValueError(
            f"Expected {_NUM_JOINTS} joint angles, got {len(joint_angles)}"
        )

    # Start with identity
    T = np.eye(4, dtype=np.float64)

    # Traverse all joints (6 revolute + 2 fixed)
    for i, (xyz, rpy, axis) in enumerate(_JOINT_PARAMS):
        T_trans = _translation_matrix(xyz)
        T_rpy = _rpy_to_matrix(rpy)
        if axis is not None and i < _NUM_JOINTS:
            T_rot = _axis_angle_matrix(axis, joint_angles[i])
        else:
            T_rot = np.eye(4, dtype=np.float64)
        T = T @ T_trans @ T_rpy @ T_rot

    # Extract ROS-frame position and rotation
    pos_ros = T[:3, 3]
    rot_ros = _mat_to_quaternion(T[:3, :3])

    # Apply base yaw rotation (to handle Robot2 mirroring)
    if abs(base_yaw_rad) > 1e-9:
        # Rotation around Y-axis by base_yaw_rad
        cy, sy = math.cos(base_yaw_rad / 2), math.sin(base_yaw_rad / 2)
        q_base_yaw = np.array([0.0, sy, 0.0, cy], dtype=np.float64)
        # Rotate position
        # Use Rodrigues directly: p' = R_y(base_yaw) * p
        yaw = base_yaw_rad
        c_yaw, s_yaw = math.cos(yaw), math.sin(yaw)
        px, py, pz = pos_ros
        pos_ros = np.array(
            [c_yaw * px + s_yaw * pz, py, -s_yaw * px + c_yaw * pz],
            dtype=np.float64,
        )
        # Compose rotation
        rot_ros = _quat_multiply(q_base_yaw, rot_ros)
        rot_ros = _normalise_quat(rot_ros)

    # Add base offset (already in Unity frame since ROBOT_BASE_POSITIONS is Unity)
    bx, by, bz = base_position

    # Convert ROS right-handed frame → Unity left-handed frame:
    # negate X of position and quat.x  (same as GraspFrameTransform.py:202-204)
    pos_unity = (
        -pos_ros[0] + bx,
        pos_ros[1] + by,
        pos_ros[2] + bz,
    )
    quat_unity = (
        -rot_ros[0],  # x negated
        rot_ros[1],
        rot_ros[2],
        rot_ros[3],
    )

    return pos_unity, quat_unity


def compute_end_effector_position(
    joint_angles: list,
    base_position: Tuple[float, float, float],
    base_yaw_rad: float = 0.0,
) -> Tuple[float, float, float]:
    """Convenience wrapper returning only the end-effector position (x, y, z).

    Args:
        joint_angles: 6 joint angles in radians, ROS joint convention.
        base_position: Robot base position in Unity world frame.
        base_yaw_rad: Base yaw offset in radians.

    Returns:
        End-effector position (x, y, z) in Unity world frame.
    """
    pos, _ = compute_end_effector_pose(joint_angles, base_position, base_yaw_rad)
    return pos
