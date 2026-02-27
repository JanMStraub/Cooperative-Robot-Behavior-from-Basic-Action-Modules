"""
Quaternion mathematical operations for 3D rotations.

This module provides core quaternion operations needed for grasp planning,
implementing industry-standard formulas compatible with Unity's quaternion system.

All quaternions are represented as tuples (x, y, z, w) where w is the scalar component.
"""

import numpy as np
from typing import Tuple


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """
    Convert Euler angles (in radians) to quaternion.

    Uses the ZYX convention (yaw-pitch-roll) to match Unity's Quaternion.Euler behavior.

    Args:
        roll: Rotation around X-axis (radians)
        pitch: Rotation around Y-axis (radians)
        yaw: Rotation around Z-axis (radians)

    Returns:
        Quaternion as (x, y, z, w) tuple
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    # ZYX convention
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (x, y, z, w)


def euler_from_quaternion(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (in radians).

    Returns Euler angles in ZYX convention (yaw-pitch-roll) to match Unity.

    Args:
        x, y, z, w: Quaternion components

    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)


def quaternion_multiply(q1: Tuple[float, float, float, float],
                       q2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Multiply two quaternions.

    Performs Hamilton product: q1 * q2. The order matters - this represents
    applying q2 first, then q1 (like matrix multiplication).

    Args:
        q1: First quaternion (x, y, z, w)
        q2: Second quaternion (x, y, z, w)

    Returns:
        Resulting quaternion (x, y, z, w)
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return (x, y, z, w)


def quaternion_rotate_vector(quat: Tuple[float, float, float, float],
                             vec: np.ndarray) -> np.ndarray:
    """
    Rotate a 3D vector by a quaternion.

    Uses the formula: v' = q * v * q^(-1)
    Optimized implementation avoids explicit quaternion multiplication.

    Args:
        quat: Rotation quaternion (x, y, z, w)
        vec: 3D vector as numpy array [x, y, z]

    Returns:
        Rotated vector as numpy array [x, y, z]
    """
    x, y, z, w = quat

    # Extract vector part of quaternion
    qv = np.array([x, y, z])

    # Rodrigues' rotation formula optimized for quaternions
    # v' = v + 2 * qv × (qv × v + w * v)
    t = 2.0 * np.cross(qv, vec)
    rotated = vec + w * t + np.cross(qv, t)

    return rotated


def quaternion_angle(q1: Tuple[float, float, float, float],
                     q2: Tuple[float, float, float, float]) -> float:
    """
    Calculate the angular distance between two quaternions in degrees.

    Returns the minimum angle needed to rotate from q1 to q2.

    Args:
        q1: First quaternion (x, y, z, w)
        q2: Second quaternion (x, y, z, w)

    Returns:
        Angle in degrees [0, 180]
    """
    # Normalize quaternions
    q1_norm = np.array(q1) / np.linalg.norm(q1)
    q2_norm = np.array(q2) / np.linalg.norm(q2)

    # Dot product
    dot = np.abs(np.dot(q1_norm, q2_norm))

    # Clamp to avoid numerical errors with arccos
    dot = np.clip(dot, -1.0, 1.0)

    # Angle = 2 * arccos(|dot|)
    angle_rad = 2.0 * np.arccos(dot)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def quaternion_inverse(quat: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Calculate the inverse (conjugate for unit quaternions) of a quaternion.

    For a unit quaternion, the inverse equals the conjugate: (x, y, z, w)^(-1) = (-x, -y, -z, w)
    This function assumes the input is a unit quaternion (or normalizes it).

    Args:
        quat: Quaternion (x, y, z, w)

    Returns:
        Inverse quaternion (x, y, z, w)
    """
    x, y, z, w = quat

    # Normalize to ensure unit quaternion
    norm = np.sqrt(x*x + y*y + z*z + w*w)

    if norm < 1e-8:
        # Degenerate case, return identity
        return (0.0, 0.0, 0.0, 1.0)

    # Conjugate and scale
    scale = 1.0 / (norm * norm)
    return (-x * scale, -y * scale, -z * scale, w * scale)


def quaternion_normalize(quat: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Normalize a quaternion to unit length.

    Args:
        quat: Quaternion (x, y, z, w)

    Returns:
        Normalized quaternion (x, y, z, w)
    """
    x, y, z, w = quat
    norm = np.sqrt(x*x + y*y + z*z + w*w)

    if norm < 1e-8:
        # Degenerate case, return identity
        return (0.0, 0.0, 0.0, 1.0)

    return (x / norm, y / norm, z / norm, w / norm)


def quaternion_identity() -> Tuple[float, float, float, float]:
    """
    Return the identity quaternion (no rotation).

    Returns:
        Identity quaternion (0, 0, 0, 1)
    """
    return (0.0, 0.0, 0.0, 1.0)


def quaternion_from_axis_angle(axis: np.ndarray, angle_rad: float) -> Tuple[float, float, float, float]:
    """
    Create a quaternion from an axis-angle representation.

    Args:
        axis: 3D rotation axis (will be normalized)
        angle_rad: Rotation angle in radians

    Returns:
        Quaternion (x, y, z, w)
    """
    # Normalize axis — guard against zero-length axis
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return (0.0, 0.0, 0.0, 1.0)
    axis_norm = axis / norm

    half_angle = angle_rad * 0.5
    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)

    x = axis_norm[0] * sin_half
    y = axis_norm[1] * sin_half
    z = axis_norm[2] * sin_half
    w = cos_half

    return (x, y, z, w)
