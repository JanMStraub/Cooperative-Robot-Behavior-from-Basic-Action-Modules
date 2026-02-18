"""
Vector mathematical operations for spatial calculations.

This module provides core vector operations needed for grasp planning,
implementing standard linear algebra operations with NumPy.

All vectors are represented as NumPy arrays of shape (3,) for 3D operations.
"""

import numpy as np
from typing import Tuple


def vector_normalize(vec: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Args:
        vec: Input vector as numpy array

    Returns:
        Normalized vector (unit length). Returns zero vector if input is zero.
    """
    norm = np.linalg.norm(vec)

    if norm < 1e-8:
        # Return zero vector for degenerate case
        return np.zeros_like(vec)

    return vec / norm


def vector_dot(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate dot product of two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Scalar dot product
    """
    return float(np.dot(v1, v2))


def vector_cross(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calculate cross product of two 3D vectors.

    Args:
        v1: First 3D vector
        v2: Second 3D vector

    Returns:
        Cross product vector perpendicular to both inputs
    """
    return np.cross(v1, v2)


def vector_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        p1: First point as vector
        p2: Second point as vector

    Returns:
        Distance as float
    """
    return float(np.linalg.norm(p2 - p1))


def vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate angle between two vectors in degrees.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Angle in degrees [0, 180]
    """
    # Normalize vectors
    v1_norm = vector_normalize(v1)
    v2_norm = vector_normalize(v2)

    # Calculate dot product
    dot = np.dot(v1_norm, v2_norm)

    # Clamp to avoid numerical errors with arccos
    dot = np.clip(dot, -1.0, 1.0)

    # Calculate angle
    angle_rad = np.arccos(dot)
    angle_deg = np.degrees(angle_rad)

    return float(angle_deg)


def vector_project(v: np.ndarray, onto: np.ndarray) -> np.ndarray:
    """
    Project vector v onto vector 'onto'.

    Args:
        v: Vector to project
        onto: Vector to project onto

    Returns:
        Projected vector component along 'onto'
    """
    onto_norm = vector_normalize(onto)
    projection_length = np.dot(v, onto_norm)
    return projection_length * onto_norm


def vector_reject(v: np.ndarray, from_direction: np.ndarray) -> np.ndarray:
    """
    Reject (remove) component of v along from_direction.

    Returns the component of v perpendicular to from_direction.

    Args:
        v: Vector to reject from
        from_direction: Direction to reject

    Returns:
        Rejected vector component perpendicular to from_direction
    """
    projection = vector_project(v, from_direction)
    return v - projection


def vector_lerp(v1: np.ndarray, v2: np.ndarray, t: float) -> np.ndarray:
    """
    Linear interpolation between two vectors.

    Args:
        v1: Start vector
        v2: End vector
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated vector
    """
    t = np.clip(t, 0.0, 1.0)
    return v1 + (v2 - v1) * t


def vector_slerp(v1: np.ndarray, v2: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two vectors.

    Maintains constant magnitude during interpolation, unlike lerp.

    Args:
        v1: Start vector
        v2: End vector
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated vector
    """
    t = np.clip(t, 0.0, 1.0)

    # Normalize input vectors
    v1_norm = vector_normalize(v1)
    v2_norm = vector_normalize(v2)

    # Calculate angle
    dot = np.dot(v1_norm, v2_norm)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)

    # Handle near-parallel case
    if abs(theta) < 1e-6:
        return vector_lerp(v1, v2, t)

    # Slerp formula
    sin_theta = np.sin(theta)
    a = np.sin((1.0 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta

    # Maintain magnitude
    magnitude = np.linalg.norm(v1) * (1.0 - t) + np.linalg.norm(v2) * t

    return (a * v1_norm + b * v2_norm) * magnitude


def vector_clamp_magnitude(v: np.ndarray, max_magnitude: float) -> np.ndarray:
    """
    Clamp vector magnitude to maximum value.

    Args:
        v: Input vector
        max_magnitude: Maximum allowed magnitude

    Returns:
        Vector with magnitude <= max_magnitude
    """
    magnitude = np.linalg.norm(v)

    if magnitude > max_magnitude:
        return (v / magnitude) * max_magnitude

    return v


def vectors_orthonormal_basis(forward: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate an orthonormal basis from a forward vector.

    Creates right and up vectors perpendicular to forward.

    Args:
        forward: Forward direction vector

    Returns:
        Tuple of (forward_normalized, right, up) forming orthonormal basis
    """
    forward_norm = vector_normalize(forward)

    # Choose a reference vector not parallel to forward
    if abs(forward_norm[1]) < 0.9:
        reference = np.array([0.0, 1.0, 0.0])  # Up vector
    else:
        reference = np.array([1.0, 0.0, 0.0])  # Right vector

    # Generate right vector
    right = vector_normalize(vector_cross(reference, forward_norm))

    # Generate up vector
    up = vector_normalize(vector_cross(forward_norm, right))

    return forward_norm, right, up
