#!/usr/bin/env python3
"""Pure-NumPy FK solver for the AR4 6-DOF arm (from ar4.urdf). Outputs Unity left-handed world frame poses."""

import math
from typing import List, Tuple

import numpy as np


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _normalise_quat(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


# URDF joint definitions (verified from ar4.urdf)
# Each entry: (origin_xyz, origin_rpy, axis_xyz)
# Fixed joints have axis=None.

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

# Chain indices at which to snapshot T for collision link poses (link_2, link_3, link_5, link_6)
_LINK_CAPTURE_INDICES = {1, 2, 4, 5}

# Low-level transform helpers


def _translation_matrix(xyz: Tuple[float, float, float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = xyz[0]
    T[1, 3] = xyz[1]
    T[2, 3] = xyz[2]
    return T


def _rpy_to_matrix(rpy: Tuple[float, float, float]) -> np.ndarray:
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


def compute_end_effector_pose(
    joint_angles: list,
    base_position: Tuple[float, float, float],
    base_yaw_rad: float = 0.0,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """Compute EE pose via FK. base_yaw_rad=π for Robot2 (180° mount). Returns Unity world frame."""
    if len(joint_angles) != _NUM_JOINTS:
        raise ValueError(
            f"Expected {_NUM_JOINTS} joint angles, got {len(joint_angles)}"
        )

    T = np.eye(4, dtype=np.float64)

    for i, (xyz, rpy, axis) in enumerate(_JOINT_PARAMS):
        T_trans = _translation_matrix(xyz)
        T_rpy = _rpy_to_matrix(rpy)
        if axis is not None and i < _NUM_JOINTS:
            T_rot = _axis_angle_matrix(axis, joint_angles[i])
        else:
            T_rot = np.eye(4, dtype=np.float64)
        T = T @ T_trans @ T_rpy @ T_rot

    pos_ros = T[:3, 3]
    rot_ros = _mat_to_quaternion(T[:3, :3])

    # Robot2 mirroring: rotate position and compose quaternion
    if abs(base_yaw_rad) > 1e-9:
        cy, sy = math.cos(base_yaw_rad / 2), math.sin(base_yaw_rad / 2)
        q_base_yaw = np.array([0.0, sy, 0.0, cy], dtype=np.float64)
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

    bx, by, bz = base_position

    # ROS right-handed → Unity left-handed: negate X of position and quat.x
    # (same pattern as GraspFrameTransform.py:202-204)
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


def compute_link_poses(
    joint_angles: list,
    base_position: Tuple[float, float, float],
    base_yaw_rad: float = 0.0,
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
    """Snapshots link_2/3/5/6 transforms for collision geometry. Returns Unity world frame poses."""
    if len(joint_angles) != _NUM_JOINTS:
        raise ValueError(
            f"Expected {_NUM_JOINTS} joint angles, got {len(joint_angles)}"
        )

    T = np.eye(4, dtype=np.float64)
    snapshots: List[np.ndarray] = []

    for i, (xyz, rpy, axis) in enumerate(_JOINT_PARAMS):
        T_trans = _translation_matrix(xyz)
        T_rpy = _rpy_to_matrix(rpy)
        if axis is not None and i < _NUM_JOINTS:
            T_rot = _axis_angle_matrix(axis, joint_angles[i])
        else:
            T_rot = np.eye(4, dtype=np.float64)
        T = T @ T_trans @ T_rpy @ T_rot
        if i in _LINK_CAPTURE_INDICES:
            snapshots.append(T.copy())

    # Hoist yaw constants out of per-snapshot loop
    apply_yaw = abs(base_yaw_rad) > 1e-9
    c_yaw, s_yaw = 1.0, 0.0
    q_base_yaw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if apply_yaw:
        cy, sy = math.cos(base_yaw_rad / 2), math.sin(base_yaw_rad / 2)
        q_base_yaw = np.array([0.0, sy, 0.0, cy], dtype=np.float64)
        c_yaw, s_yaw = math.cos(base_yaw_rad), math.sin(base_yaw_rad)
    bx, by, bz = base_position

    result = []
    for snap in snapshots:
        pos_ros = snap[:3, 3]
        rot_ros = _mat_to_quaternion(snap[:3, :3])

        if apply_yaw:
            px, py, pz = pos_ros
            pos_ros = np.array(
                [c_yaw * px + s_yaw * pz, py, -s_yaw * px + c_yaw * pz],
                dtype=np.float64,
            )
            rot_ros = _normalise_quat(_quat_multiply(q_base_yaw, rot_ros))

        pos_unity = (-pos_ros[0] + bx, pos_ros[1] + by, pos_ros[2] + bz)
        quat_unity = (-rot_ros[0], rot_ros[1], rot_ros[2], rot_ros[3])
        result.append((pos_unity, quat_unity))

    return result


def compute_end_effector_position(
    joint_angles: list,
    base_position: Tuple[float, float, float],
    base_yaw_rad: float = 0.0,
) -> Tuple[float, float, float]:
    """Convenience wrapper - returns only position (x, y, z) in Unity world frame."""
    pos, _ = compute_end_effector_pose(joint_angles, base_position, base_yaw_rad)
    return pos
