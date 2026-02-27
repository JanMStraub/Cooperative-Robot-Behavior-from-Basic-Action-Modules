"""
Coordinate transformation utilities for Unity-ROS integration.

This module provides coordinate frame transformations between:
- Unity world coordinates (Y-up, left-handed)
- Robot-local coordinates (relative to robot base)
- ROS coordinates (Z-up, right-handed)

Robot base positions and rotations are configured to match the Unity scene setup.
"""

import math
import numpy as np
from typing import Dict, Tuple, Optional


# Robot base positions and rotations in Unity world coordinates
# These match the positions in Unity Environment.prefab
# Robot1: Unity rotation (0, 0, 0, -1) = 360° = 0° effective rotation - facing forward (+Z in Unity)
# Robot2: Unity rotation (0, 1, 0, 0) = 180° around Y - facing backward (-Z in Unity)
ROBOT_BASE_TRANSFORMS = {
    "Robot1": {"position": (-0.475, 0.0, 0.0), "y_rotation": 0.0},
    "Robot2": {"position": (0.475, 0.0, 0.0), "y_rotation": 180.0},
}


def world_to_robot_frame(world_pos: Dict[str, float], robot_id: str) -> Dict[str, float]:
    """
    Transform a position from Unity world coordinates to robot-local ROS frame.

    This performs a three-step transformation:
    1. Translate to robot base (Unity world -> Unity local)
    2. Apply robot's Y-rotation (for Robot2's 180° facing direction)
    3. Convert Unity axes to ROS axes (Y-up left-handed -> Z-up right-handed)

    Args:
        world_pos: Position in Unity world frame with keys 'x', 'y', 'z'
        robot_id: Robot identifier ('Robot1' or 'Robot2')

    Returns:
        Position in robot-local ROS frame with keys 'x', 'y', 'z'

    Raises:
        ValueError: If robot_id is not configured
    """
    if robot_id not in ROBOT_BASE_TRANSFORMS:
        raise ValueError(
            f"Unknown robot '{robot_id}'. "
            f"Available: {list(ROBOT_BASE_TRANSFORMS.keys())}"
        )

    # Get robot base transform
    transform = ROBOT_BASE_TRANSFORMS[robot_id]
    base_x, base_y, base_z = transform["position"]
    y_rotation_deg = transform["y_rotation"]

    # Step 1: Translate to robot base (Unity world -> Unity local)
    unity_local_x = world_pos.get("x", 0.0) - base_x
    unity_local_y = world_pos.get("y", 0.0) - base_y
    unity_local_z = world_pos.get("z", 0.0) - base_z

    # Step 2: Apply robot's Y-rotation in Unity space (if any)
    y_rotation_rad = math.radians(y_rotation_deg)
    cos_theta = math.cos(y_rotation_rad)
    sin_theta = math.sin(y_rotation_rad)

    rotated_x = cos_theta * unity_local_x + sin_theta * unity_local_z
    rotated_y = unity_local_y
    rotated_z = -sin_theta * unity_local_x + cos_theta * unity_local_z

    # Step 3: Convert from Unity (Y-up, left-handed) to ROS (Z-up, right-handed)
    # Unity axes: X=right, Y=up, Z=forward
    # ROS axes:   X=forward, Y=left, Z=up
    # Conversion: Unity (X, Y, Z) → ROS (Z, -X, Y)
    ros_x = rotated_z  # Unity Z (forward) → ROS X (forward)
    ros_y = -rotated_x  # Unity X (right) → ROS -Y (since ROS Y is left)
    ros_z = rotated_y  # Unity Y (up) → ROS Z (up)

    return {"x": ros_x, "y": ros_y, "z": ros_z}


def robot_to_world_frame(local_pos: Dict[str, float], robot_id: str) -> Dict[str, float]:
    """
    Transform a position from robot-local ROS frame to Unity world coordinates.

    This is the inverse of world_to_robot_frame, performing:
    1. Convert ROS axes to Unity axes (Z-up right-handed -> Y-up left-handed)
    2. Apply inverse robot Y-rotation
    3. Translate to world origin

    Args:
        local_pos: Position in robot-local ROS frame with keys 'x', 'y', 'z'
        robot_id: Robot identifier ('Robot1' or 'Robot2')

    Returns:
        Position in Unity world frame with keys 'x', 'y', 'z'

    Raises:
        ValueError: If robot_id is not configured
    """
    if robot_id not in ROBOT_BASE_TRANSFORMS:
        raise ValueError(
            f"Unknown robot '{robot_id}'. "
            f"Available: {list(ROBOT_BASE_TRANSFORMS.keys())}"
        )

    # Get robot base transform
    transform = ROBOT_BASE_TRANSFORMS[robot_id]
    base_x, base_y, base_z = transform["position"]
    y_rotation_deg = transform["y_rotation"]

    # Step 1: Convert from ROS (Z-up, right-handed) to Unity (Y-up, left-handed)
    # Inverse of: Unity (X, Y, Z) → ROS (Z, -X, Y)
    # Therefore: ROS (X, Y, Z) → Unity (-Y, Z, X)
    ros_x = local_pos.get("x", 0.0)
    ros_y = local_pos.get("y", 0.0)
    ros_z = local_pos.get("z", 0.0)

    rotated_x = -ros_y  # ROS Y (left) → Unity -X (since Unity X is right)
    rotated_y = ros_z  # ROS Z (up) → Unity Y (up)
    rotated_z = ros_x  # ROS X (forward) → Unity Z (forward)

    # Step 2: Apply inverse robot Y-rotation
    y_rotation_rad = math.radians(-y_rotation_deg)  # Inverse rotation
    cos_theta = math.cos(y_rotation_rad)
    sin_theta = math.sin(y_rotation_rad)

    unity_local_x = cos_theta * rotated_x + sin_theta * rotated_z
    unity_local_y = rotated_y
    unity_local_z = -sin_theta * rotated_x + cos_theta * rotated_z

    # Step 3: Translate to world origin
    world_x = unity_local_x + base_x
    world_y = unity_local_y + base_y
    world_z = unity_local_z + base_z

    return {"x": world_x, "y": world_y, "z": world_z}


def _build_np_transform_cache() -> Dict[str, Dict]:
    """
    Pre-compute NumPy transform matrices for each robot at module load time.

    For each robot the combined rotation+axis-conversion matrix maps Unity-local
    coordinates directly to ROS coordinates (and its transpose maps back).

    The full forward pipeline is:
        translate → Y-rotate (Unity local) → axis-permute to ROS
    Rotation matrix (Y-axis, angle θ):
        [[cos θ,  0, sin θ],
         [0,      1, 0    ],
         [-sin θ, 0, cos θ]]
    Axis permutation (Unity XYZ → ROS ZYX with sign flip on Y):
        ros_x = rotated_z
        ros_y = -rotated_x
        ros_z = rotated_y
    Combined into a 3×3 matrix applied after translation.

    Returns:
        Dict mapping robot_id to {'base': np.ndarray, 'fwd': np.ndarray, 'inv': np.ndarray}
    """
    cache = {}
    for robot_id, transform in ROBOT_BASE_TRANSFORMS.items():
        base = np.array(transform["position"], dtype=float)
        theta = math.radians(transform["y_rotation"])
        c, s = math.cos(theta), math.sin(theta)

        # Y-rotation then axis-permute, composed into one matrix.
        # Row i of fwd gives the ROS axis-i expression in Unity-local coords.
        # ros_x = rotated_z = -s*ux + c*uz
        # ros_y = -rotated_x = -(c*ux + s*uz) = -c*ux - s*uz
        # ros_z = rotated_y = uy
        fwd = np.array([
            [-s,  0.0,  c],   # ros_x row
            [-c,  0.0, -s],   # ros_y row
            [0.0, 1.0,  0.0], # ros_z row
        ], dtype=float)

        cache[robot_id] = {
            "base": base,
            "fwd": fwd,
            "inv": fwd.T,  # Orthogonal matrix: inverse == transpose
        }
    return cache


_NP_TRANSFORM_CACHE: Dict[str, Dict] = _build_np_transform_cache()


def world_to_robot_frame_np(
    world_pos: np.ndarray, robot_id: str
) -> np.ndarray:
    """
    Transform a position from Unity world to robot ROS frame (NumPy version).

    Uses pre-computed matrices to avoid dict round-trips. Numerically equivalent
    to world_to_robot_frame() but operates entirely on NumPy arrays.

    Args:
        world_pos: Position as [x, y, z] NumPy array in Unity world frame
        robot_id: Robot identifier ('Robot1' or 'Robot2')

    Returns:
        Position in robot-local ROS frame as [x, y, z] NumPy array

    Raises:
        ValueError: If robot_id is not configured
    """
    if robot_id not in _NP_TRANSFORM_CACHE:
        raise ValueError(
            f"Unknown robot '{robot_id}'. "
            f"Available: {list(_NP_TRANSFORM_CACHE.keys())}"
        )
    entry = _NP_TRANSFORM_CACHE[robot_id]
    unity_local = world_pos - entry["base"]
    return entry["fwd"] @ unity_local


def robot_to_world_frame_np(
    local_pos: np.ndarray, robot_id: str
) -> np.ndarray:
    """
    Transform a position from robot ROS frame to Unity world (NumPy version).

    Inverse of world_to_robot_frame_np. Uses the transpose of the forward
    matrix (valid because fwd is orthogonal) plus translation.

    Args:
        local_pos: Position as [x, y, z] NumPy array in robot-local ROS frame
        robot_id: Robot identifier ('Robot1' or 'Robot2')

    Returns:
        Position in Unity world frame as [x, y, z] NumPy array

    Raises:
        ValueError: If robot_id is not configured
    """
    if robot_id not in _NP_TRANSFORM_CACHE:
        raise ValueError(
            f"Unknown robot '{robot_id}'. "
            f"Available: {list(_NP_TRANSFORM_CACHE.keys())}"
        )
    entry = _NP_TRANSFORM_CACHE[robot_id]
    unity_local = entry["inv"] @ local_pos
    return unity_local + entry["base"]


def get_robot_base_position(robot_id: str) -> Tuple[float, float, float]:
    """
    Get the robot base position in Unity world coordinates.

    Args:
        robot_id: Robot identifier

    Returns:
        Base position as (x, y, z) tuple

    Raises:
        ValueError: If robot_id is not configured
    """
    if robot_id not in ROBOT_BASE_TRANSFORMS:
        raise ValueError(
            f"Unknown robot '{robot_id}'. "
            f"Available: {list(ROBOT_BASE_TRANSFORMS.keys())}"
        )

    return ROBOT_BASE_TRANSFORMS[robot_id]["position"]


def get_robot_base_rotation(robot_id: str) -> float:
    """
    Get the robot base Y-rotation in Unity world coordinates.

    Args:
        robot_id: Robot identifier

    Returns:
        Y-rotation in degrees

    Raises:
        ValueError: If robot_id is not configured
    """
    if robot_id not in ROBOT_BASE_TRANSFORMS:
        raise ValueError(
            f"Unknown robot '{robot_id}'. "
            f"Available: {list(ROBOT_BASE_TRANSFORMS.keys())}"
        )

    return ROBOT_BASE_TRANSFORMS[robot_id]["y_rotation"]


def add_robot_transform(
    robot_id: str, position: Tuple[float, float, float], y_rotation: float
) -> None:
    """
    Register a new robot base transform.

    This allows dynamic robot configurations beyond the default Robot1/Robot2.

    Args:
        robot_id: Unique robot identifier
        position: Base position in Unity world as (x, y, z)
        y_rotation: Base rotation around Y-axis in degrees
    """
    ROBOT_BASE_TRANSFORMS[robot_id] = {
        "position": position,
        "y_rotation": y_rotation,
    }
