"""
Spatial Predicates for Robot Operations
========================================

This module provides boolean predicate functions for checking spatial
relationships and robot state. Predicates return (is_valid, reason_if_invalid)
tuples and are used by the verification system to check operation preconditions.

All predicates are registered in PREDICATE_REGISTRY for dynamic lookup.
"""

import math
import logging
from typing import Tuple, Dict, Callable, Any, Optional
import LLMConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global predicate registry
PREDICATE_REGISTRY: Dict[str, Callable] = {}


def register_predicate(name: str):
    """
    Decorator to register a predicate function.

    Args:
        name: Predicate name for lookup

    Example:
        @register_predicate("target_within_reach")
        def target_within_reach(robot_id, x, y, z, world_state=None):
            ...
    """

    def decorator(func: Callable) -> Callable:
        PREDICATE_REGISTRY[name] = func
        return func

    return decorator


def get_predicate(name: str) -> Optional[Callable]:
    """
    Get a registered predicate function by name.

    Args:
        name: Predicate name

    Returns:
        Predicate function or None if not found
    """
    return PREDICATE_REGISTRY.get(name)


# ============================================================================
# Core Spatial Predicates (Week 1)
# ============================================================================


@register_predicate("target_within_reach")
def target_within_reach(
    robot_id: str, x: float, y: float, z: float, world_state=None
) -> Tuple[bool, str]:
    """
    Check if a target position is within the robot's maximum reach distance.

    Args:
        robot_id: Robot identifier (e.g., "Robot1")
        x, y, z: Target position in world coordinates
        world_state: Optional WorldState instance for querying robot info

    Returns:
        (is_valid, reason_if_invalid)
    """
    try:
        # Get robot base position
        base_pos = LLMConfig.ROBOT_BASE_POSITIONS.get(robot_id)
        if base_pos is None:
            return False, f"Unknown robot '{robot_id}' - not in ROBOT_BASE_POSITIONS"

        # Calculate distance from base to target
        dx = x - base_pos[0]
        dy = y - base_pos[1]
        dz = z - base_pos[2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Check against maximum reach
        max_reach = LLMConfig.MAX_ROBOT_REACH
        if distance > max_reach:
            return (
                False,
                f"Target at ({x:.3f}, {y:.3f}, {z:.3f}) is {distance:.3f}m from robot base, exceeds max reach {max_reach}m",
            )

        return True, ""

    except Exception as e:
        logger.error(f"Error in target_within_reach: {e}")
        return False, f"Error checking reach: {str(e)}"


@register_predicate("is_in_robot_workspace")
def is_in_robot_workspace(
    robot_id: str, x: float, y: float, z: float, world_state=None
) -> Tuple[bool, str]:
    """
    Check if a position is within the robot's assigned workspace region.

    Args:
        robot_id: Robot identifier
        x, y, z: Position in world coordinates
        world_state: Optional WorldState instance

    Returns:
        (is_valid, reason_if_invalid)
    """
    try:
        # Get robot's assigned workspace
        workspace_name = LLMConfig.ROBOT_WORKSPACE_ASSIGNMENTS.get(robot_id)
        if workspace_name is None:
            return False, f"Robot '{robot_id}' has no assigned workspace"

        # Get workspace bounds
        workspace = LLMConfig.WORKSPACE_REGIONS.get(workspace_name)
        if workspace is None:
            return (
                False,
                f"Workspace '{workspace_name}' not defined in WORKSPACE_REGIONS",
            )

        # Check bounds
        if not (workspace["x_min"] <= x <= workspace["x_max"]):
            return (
                False,
                f"X coordinate {x:.3f} outside workspace X range [{workspace['x_min']}, {workspace['x_max']}]",
            )

        if not (workspace["y_min"] <= y <= workspace["y_max"]):
            return (
                False,
                f"Y coordinate {y:.3f} outside workspace Y range [{workspace['y_min']}, {workspace['y_max']}]",
            )

        if not (workspace["z_min"] <= z <= workspace["z_max"]):
            return (
                False,
                f"Z coordinate {z:.3f} outside workspace Z range [{workspace['z_min']}, {workspace['z_max']}]",
            )

        return True, ""

    except Exception as e:
        logger.error(f"Error in is_in_robot_workspace: {e}")
        return False, f"Error checking workspace: {str(e)}"


@register_predicate("is_in_shared_zone")
def is_in_shared_zone(x: float, y: float, z: float) -> Tuple[bool, str]:
    """
    Check if a position is in the shared workspace zone.

    Args:
        x, y, z: Position in world coordinates

    Returns:
        (is_valid, reason_if_invalid)
    """
    try:
        shared_zone = LLMConfig.WORKSPACE_REGIONS.get("shared_zone")
        if shared_zone is None:
            return False, "Shared zone not defined in WORKSPACE_REGIONS"

        # Check bounds
        in_zone = (
            shared_zone["x_min"] <= x <= shared_zone["x_max"]
            and shared_zone["y_min"] <= y <= shared_zone["y_max"]
            and shared_zone["z_min"] <= z <= shared_zone["z_max"]
        )

        if in_zone:
            return True, ""
        else:
            return False, f"Position ({x:.3f}, {y:.3f}, {z:.3f}) is outside shared zone"

    except Exception as e:
        logger.error(f"Error in is_in_shared_zone: {e}")
        return False, f"Error checking shared zone: {str(e)}"


@register_predicate("robot_is_initialized")
def robot_is_initialized(robot_id: str, world_state=None) -> Tuple[bool, str]:
    """
    Check if a robot is initialized and ready for commands.

    Args:
        robot_id: Robot identifier
        world_state: WorldState instance for querying robot status

    Returns:
        (is_valid, reason_if_invalid)
    """
    try:
        # Check if robot exists in configuration
        if robot_id not in LLMConfig.ROBOT_BASE_POSITIONS:
            return False, f"Robot '{robot_id}' not found in system configuration"

        # If world_state is provided, query actual status
        if world_state is not None:
            try:
                status = world_state.get_robot_status(robot_id)
                if status is None:
                    return False, f"Robot '{robot_id}' status unavailable"

                # Check if robot is active/initialized
                if not status.get("is_initialized", False):
                    return False, f"Robot '{robot_id}' is not initialized"

                return True, ""
            except Exception as e:
                logger.warning(f"Could not query robot status: {e}")
                # Fall through to basic check

        # Basic check: robot exists in config
        return True, ""

    except Exception as e:
        logger.error(f"Error in robot_is_initialized: {e}")
        return False, f"Error checking initialization: {str(e)}"


@register_predicate("robot_is_stationary")
def robot_is_stationary(robot_id: str, world_state=None) -> Tuple[bool, str]:
    """
    Check if a robot is stationary (not currently moving).

    Args:
        robot_id: Robot identifier
        world_state: WorldState instance for querying robot status

    Returns:
        (is_valid, reason_if_invalid)
    """
    try:
        if world_state is None:
            return False, "WorldState required to check robot movement"

        # Query robot status
        status = world_state.get_robot_status(robot_id)
        if status is None:
            return False, f"Robot '{robot_id}' status unavailable"

        # Check if robot is moving
        is_moving = status.get("is_moving", False)
        if is_moving:
            return False, f"Robot '{robot_id}' is currently moving"

        return True, ""

    except Exception as e:
        logger.error(f"Error in robot_is_stationary: {e}")
        return False, f"Error checking movement: {str(e)}"


@register_predicate("gripper_is_open")
def gripper_is_open(robot_id: str, world_state=None) -> Tuple[bool, str]:
    """
    Check if a robot's gripper is in the open state.

    Args:
        robot_id: Robot identifier
        world_state: WorldState instance for querying gripper state

    Returns:
        (is_valid, reason_if_invalid)
    """
    try:
        if world_state is None:
            return False, "WorldState required to check gripper state"

        # Query robot status
        status = world_state.get_robot_status(robot_id)
        if status is None:
            return False, f"Robot '{robot_id}' status unavailable"

        # Check gripper state
        gripper_state = status.get("gripper_state", "unknown")
        if gripper_state == "open":
            return True, ""
        elif gripper_state == "closed":
            return False, f"Robot '{robot_id}' gripper is closed"
        else:
            return False, f"Robot '{robot_id}' gripper state unknown: {gripper_state}"

    except Exception as e:
        logger.error(f"Error in gripper_is_open: {e}")
        return False, f"Error checking gripper: {str(e)}"


@register_predicate("gripper_is_closed")
def gripper_is_closed(robot_id: str, world_state=None) -> Tuple[bool, str]:
    """
    Check if a robot's gripper is in the closed state.

    Args:
        robot_id: Robot identifier
        world_state: WorldState instance for querying gripper state

    Returns:
        (is_valid, reason_if_invalid)
    """
    try:
        if world_state is None:
            return False, "WorldState required to check gripper state"

        # Query robot status
        status = world_state.get_robot_status(robot_id)
        if status is None:
            return False, f"Robot '{robot_id}' status unavailable"

        # Check gripper state
        gripper_state = status.get("gripper_state", "unknown")
        if gripper_state == "closed":
            return True, ""
        elif gripper_state == "open":
            return False, f"Robot '{robot_id}' gripper is open"
        else:
            return False, f"Robot '{robot_id}' gripper state unknown: {gripper_state}"

    except Exception as e:
        logger.error(f"Error in gripper_is_closed: {e}")
        return False, f"Error checking gripper: {str(e)}"


@register_predicate("object_accessible_by_robot")
def object_accessible_by_robot(
    robot_id: str, object_position: Tuple[float, float, float], world_state=None
) -> Tuple[bool, str]:
    """
    Check if an object at a given position is accessible by the robot.

    This checks both reachability (within max reach) and workspace constraints.

    Args:
        robot_id: Robot identifier
        object_position: Tuple of (x, y, z) world coordinates
        world_state: Optional WorldState instance

    Returns:
        (is_valid, reason_if_invalid)
    """
    try:
        x, y, z = object_position

        # Check reachability
        is_reachable, reach_reason = target_within_reach(robot_id, x, y, z, world_state)
        if not is_reachable:
            return False, f"Object not reachable: {reach_reason}"

        # Check workspace (objects in shared zone are accessible by all robots)
        is_shared, _ = is_in_shared_zone(x, y, z)
        if is_shared:
            return True, ""  # Shared zone objects accessible by all

        # Check if in robot's workspace
        is_in_workspace, workspace_reason = is_in_robot_workspace(
            robot_id, x, y, z, world_state
        )
        if not is_in_workspace:
            return False, f"Object not in workspace: {workspace_reason}"

        return True, ""

    except Exception as e:
        logger.error(f"Error in object_accessible_by_robot: {e}")
        return False, f"Error checking accessibility: {str(e)}"


# ============================================================================
# Multi-Robot Predicates (Week 2)
# ============================================================================


@register_predicate("robots_will_collide")
def robots_will_collide(
    robot1_id: str,
    target1: Tuple[float, float, float],
    robot2_id: str,
    target2: Tuple[float, float, float],
    world_state=None,
) -> Tuple[bool, str]:
    """
    Check if two robots will collide if they move to their respective targets.

    This performs a simplified collision check based on:
    1. Distance between target positions
    2. Workspace overlap
    3. Path intersection (simplified linear path assumption)

    Args:
        robot1_id: First robot identifier
        target1: Target position for robot1 (x, y, z)
        robot2_id: Second robot identifier
        target2: Target position for robot2 (x, y, z)
        world_state: WorldState instance for current positions

    Returns:
        (will_collide, reason)
        Note: Returns (True, reason) if collision detected, (False, "") if safe
    """
    try:
        # Check 1: Target positions too close
        dx = target1[0] - target2[0]
        dy = target1[1] - target2[1]
        dz = target1[2] - target2[2]
        target_distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        min_separation = LLMConfig.MIN_ROBOT_SEPARATION
        if target_distance < min_separation:
            return True, (
                f"Target positions too close: {target_distance:.3f}m "
                f"(minimum separation: {min_separation}m)"
            )

        # Check 2: Path intersection (simplified - assume linear paths)
        if world_state is not None:
            pos1 = world_state.get_robot_position(robot1_id)
            pos2 = world_state.get_robot_position(robot2_id)

            if pos1 is not None and pos2 is not None:
                # Calculate minimum distance between two line segments
                # Line 1: pos1 -> target1
                # Line 2: pos2 -> target2
                min_path_distance = _calculate_segment_distance(
                    pos1, target1, pos2, target2
                )

                safety_margin = LLMConfig.COLLISION_SAFETY_MARGIN
                if min_path_distance < safety_margin:
                    return True, (
                        f"Paths will intersect: minimum distance {min_path_distance:.3f}m "
                        f"(safety margin: {safety_margin}m)"
                    )

        # Check 3: Both targets in same restricted workspace
        # (Shared zone is allowed, but not exclusive workspaces)
        is_shared1, _ = is_in_shared_zone(*target1)
        is_shared2, _ = is_in_shared_zone(*target2)

        if not is_shared1 and not is_shared2:
            # Both in exclusive workspaces - check if same workspace
            workspace1 = _get_workspace_containing_point(*target1)
            workspace2 = _get_workspace_containing_point(*target2)

            if workspace1 == workspace2 and workspace1 not in [
                "shared_zone",
                "center",
                None,
            ]:
                return True, (
                    f"Both robots targeting same exclusive workspace: {workspace1}"
                )

        # No collision detected
        return False, ""

    except Exception as e:
        logger.error(f"Error in robots_will_collide: {e}")
        return True, f"Error checking collision (assuming unsafe): {str(e)}"


def _calculate_segment_distance(
    p1_start: Tuple[float, float, float],
    p1_end: Tuple[float, float, float],
    p2_start: Tuple[float, float, float],
    p2_end: Tuple[float, float, float],
) -> float:
    """
    Calculate minimum distance between two line segments in 3D.

    Uses parametric line representation and finds minimum distance
    between any two points on the segments.

    Args:
        p1_start: Start point of segment 1
        p1_end: End point of segment 1
        p2_start: Start point of segment 2
        p2_end: End point of segment 2

    Returns:
        Minimum distance between segments
    """
    # Direction vectors
    d1 = tuple(p1_end[i] - p1_start[i] for i in range(3))
    d2 = tuple(p2_end[i] - p2_start[i] for i in range(3))

    # Vector between start points
    r = tuple(p1_start[i] - p2_start[i] for i in range(3))

    # Dot products
    a = sum(d1[i] * d1[i] for i in range(3))
    b = sum(d1[i] * d2[i] for i in range(3))
    c = sum(d2[i] * d2[i] for i in range(3))
    d = sum(d1[i] * r[i] for i in range(3))
    e = sum(d2[i] * r[i] for i in range(3))

    # Avoid division by zero
    denominator = a * c - b * b
    if abs(denominator) < 1e-10:
        # Segments are parallel
        # Calculate distance from p2_start to segment 1
        t1 = max(0.0, min(1.0, -d / a if a > 1e-10 else 0.0))
        point1 = tuple(p1_start[i] + t1 * d1[i] for i in range(3))
        dist = math.sqrt(sum((point1[i] - p2_start[i]) ** 2 for i in range(3)))
        return dist

    # Calculate parameters for closest points
    t1 = (b * e - c * d) / denominator
    t2 = (a * e - b * d) / denominator

    # Clamp to segment bounds [0, 1]
    t1 = max(0.0, min(1.0, t1))
    t2 = max(0.0, min(1.0, t2))

    # Calculate closest points
    point1 = tuple(p1_start[i] + t1 * d1[i] for i in range(3))
    point2 = tuple(p2_start[i] + t2 * d2[i] for i in range(3))

    # Calculate distance
    dist = math.sqrt(sum((point1[i] - point2[i]) ** 2 for i in range(3)))
    return dist


def _get_workspace_containing_point(x: float, y: float, z: float) -> Optional[str]:
    """
    Find which workspace region contains a given point.

    Args:
        x, y, z: Point coordinates

    Returns:
        Workspace region name or None if not in any region
    """
    for region_name, bounds in LLMConfig.WORKSPACE_REGIONS.items():
        if (
            bounds["x_min"] <= x <= bounds["x_max"]
            and bounds["y_min"] <= y <= bounds["y_max"]
            and bounds["z_min"] <= z <= bounds["z_max"]
        ):
            return region_name
    return None


# ============================================================================
# Utility Functions
# ============================================================================


def evaluate_predicate(predicate_name: str, **kwargs) -> Tuple[bool, str]:
    """
    Evaluate a predicate by name with given arguments.

    Args:
        predicate_name: Name of registered predicate
        **kwargs: Arguments to pass to predicate

    Returns:
        (is_valid, reason_if_invalid)

    Example:
        is_valid, reason = evaluate_predicate(
            "target_within_reach",
            robot_id="Robot1",
            x=0.3, y=0.2, z=0.1,
            world_state=world_state
        )
    """
    predicate = get_predicate(predicate_name)
    if predicate is None:
        return False, f"Unknown predicate: {predicate_name}"

    try:
        return predicate(**kwargs)
    except Exception as e:
        logger.error(f"Error evaluating predicate '{predicate_name}': {e}")
        return False, f"Predicate evaluation error: {str(e)}"


def list_predicates() -> list[str]:
    """
    List all registered predicate names.

    Returns:
        List of predicate names
    """
    return list(PREDICATE_REGISTRY.keys())
