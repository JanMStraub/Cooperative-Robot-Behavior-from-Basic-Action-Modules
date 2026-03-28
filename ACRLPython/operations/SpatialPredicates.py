#!/usr/bin/env python3
"""
Spatial Predicates for Robot Operations
========================================

This module provides boolean predicate functions for checking spatial
relationships and robot state. Predicates return (is_valid, reason_if_invalid)
tuples and are used by the verification system to check operation preconditions.

All predicates are registered in PREDICATE_REGISTRY for dynamic lookup.
"""

import math
from typing import Tuple, Dict, Callable, Optional

try:
    from config.Robot import (
        ROBOT_BASE_POSITIONS,
        ROBOT_WORKSPACE_ASSIGNMENTS,
        WORKSPACE_REGIONS,
        MAX_ROBOT_REACH,
        MIN_ROBOT_SEPARATION,
        COLLISION_SAFETY_MARGIN,
    )
except ImportError:
    from ..config.Robot import (
        ROBOT_BASE_POSITIONS,
        ROBOT_WORKSPACE_ASSIGNMENTS,
        WORKSPACE_REGIONS,
        MAX_ROBOT_REACH,
        MIN_ROBOT_SEPARATION,
        COLLISION_SAFETY_MARGIN,
    )

# Configure logging
from core.LoggingSetup import get_logger

logger = get_logger(__name__)

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
        # Guard against None coordinates (e.g. from LLM that omits values)
        if x is None or y is None or z is None:
            return False, f"Target coordinates contain None: ({x}, {y}, {z})"

        # Get robot base position
        base_pos = ROBOT_BASE_POSITIONS.get(robot_id)
        if base_pos is None:
            return False, f"Unknown robot '{robot_id}' - not in ROBOT_BASE_POSITIONS"

        # Calculate distance from base to target
        dx = x - base_pos[0]
        dy = y - base_pos[1]
        dz = z - base_pos[2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Check against maximum reach
        max_reach = MAX_ROBOT_REACH
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
        workspace_name = ROBOT_WORKSPACE_ASSIGNMENTS.get(robot_id)
        if workspace_name is None:
            return False, f"Robot '{robot_id}' has no assigned workspace"

        # Get workspace bounds
        workspace = WORKSPACE_REGIONS.get(workspace_name)
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
        shared_zone = WORKSPACE_REGIONS.get("shared_zone")
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
        if robot_id not in ROBOT_BASE_POSITIONS:
            return False, f"Robot '{robot_id}' not found in system configuration"

        # If world_state is provided, query actual status
        if world_state is not None:
            try:
                status = world_state.get_robot_status(robot_id)
                if status is None:
                    # Status unavailable, fall back to basic check
                    logger.debug(
                        f"Robot '{robot_id}' status unavailable, using basic check"
                    )
                elif "is_initialized" in status:
                    # We have actual status information from Unity
                    if not status.get("is_initialized"):
                        return False, f"Robot '{robot_id}' is not initialized"
                    return True, ""
                else:
                    # Status query was sent but no response yet (e.g., status="query_sent")
                    # Fall back to basic check
                    logger.debug(
                        f"Robot '{robot_id}' status pending, using basic check"
                    )
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

        min_separation = MIN_ROBOT_SEPARATION
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

                safety_margin = COLLISION_SAFETY_MARGIN
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
        # Segments are parallel — evaluate all 4 endpoint-to-opposite-segment distances
        # and return the minimum, since a single endpoint projection is not sufficient
        # when segments are adjacent (end-to-end) or overlapping.
        def _point_to_seg1_dist(pt: Tuple[float, float, float]) -> float:
            t = max(
                0.0,
                min(
                    1.0,
                    (
                        sum((pt[i] - p1_start[i]) * d1[i] for i in range(3)) / a
                        if a > 1e-10
                        else 0.0
                    ),
                ),
            )
            closest = tuple(p1_start[i] + t * d1[i] for i in range(3))
            return math.sqrt(sum((closest[i] - pt[i]) ** 2 for i in range(3)))

        def _point_to_seg2_dist(pt: Tuple[float, float, float]) -> float:
            t = max(
                0.0,
                min(
                    1.0,
                    (
                        sum((pt[i] - p2_start[i]) * d2[i] for i in range(3)) / c
                        if c > 1e-10
                        else 0.0
                    ),
                ),
            )
            closest = tuple(p2_start[i] + t * d2[i] for i in range(3))
            return math.sqrt(sum((closest[i] - pt[i]) ** 2 for i in range(3)))

        return min(
            _point_to_seg1_dist(p2_start),
            _point_to_seg1_dist(p2_end),
            _point_to_seg2_dist(p1_start),
            _point_to_seg2_dist(p1_end),
        )

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
    for region_name, bounds in WORKSPACE_REGIONS.items():
        if (
            bounds["x_min"] <= x <= bounds["x_max"]
            and bounds["y_min"] <= y <= bounds["y_max"]
            and bounds["z_min"] <= z <= bounds["z_max"]
        ):
            return region_name
    return None


# ============================================================================
# Object Liveness and Grasp Predicates (Knowledge Graph Stage A)
# ============================================================================


@register_predicate("object_not_stale")
def object_not_stale(object_id: str, world_state=None) -> Tuple[bool, str]:
    """
    Check if object is still fresh (confidence above stale threshold).

    An object becomes stale when its detection confidence drops below
    the threshold due to not being seen in recent frames.

    Args:
        object_id: Object identifier
        world_state: Optional WorldState instance

    Returns:
        (is_valid, reason_if_invalid)
    """
    if world_state is None:
        try:
            from .WorldState import get_world_state
        except ImportError:
            from operations.WorldState import get_world_state
        world_state = get_world_state()

    # Get object from world state
    obj = world_state._objects.get(object_id)
    if obj is None:
        return False, f"Object '{object_id}' not found in world state"

    if obj.stale:
        return (
            False,
            f"Object '{object_id}' is stale (confidence: {obj.confidence:.2f})",
        )

    return True, ""


@register_predicate("object_not_grasped_by_other")
def object_not_grasped_by_other(
    object_id: str, robot_id: str, world_state=None
) -> Tuple[bool, str]:
    """
    Check that object is not currently grasped by a different robot.

    Args:
        object_id: Object identifier
        robot_id: Robot that wants to grasp the object
        world_state: Optional WorldState instance

    Returns:
        (is_valid, reason_if_invalid)
    """
    if world_state is None:
        try:
            from .WorldState import get_world_state
        except ImportError:
            from operations.WorldState import get_world_state
        world_state = get_world_state()

    # Get object from world state
    obj = world_state._objects.get(object_id)
    if obj is None:
        return False, f"Object '{object_id}' not found in world state"

    if obj.grasped_by is not None and obj.grasped_by != robot_id:
        return (
            False,
            f"Object '{object_id}' is already grasped by {obj.grasped_by}",
        )

    return True, ""


@register_predicate("region_available_for_robot")
def region_available_for_robot(
    region: str, robot_id: str, world_state=None
) -> Tuple[bool, str]:
    """
    Check that a workspace region is not allocated to a different robot.

    Regions can be unallocated (available to all) or allocated to the
    requesting robot (available), but not allocated to another robot.

    Args:
        region: Region name (e.g., "left_workspace", "shared_zone")
        robot_id: Robot that wants to use the region
        world_state: Optional WorldState instance

    Returns:
        (is_valid, reason_if_invalid)
    """
    if world_state is None:
        try:
            from .WorldState import get_world_state
        except ImportError:
            from operations.WorldState import get_world_state
        world_state = get_world_state()

    # Check if region exists
    if region not in WORKSPACE_REGIONS:
        return False, f"Unknown workspace region: '{region}'"

    # Get current allocation
    owner = world_state.get_workspace_owner(region)

    # Region is available if unallocated or allocated to this robot
    if owner is None or owner == robot_id:
        return True, ""

    return False, f"Region '{region}' is allocated to {owner}"


# ============================================================================
# Gripper and Image Availability Predicates
# ============================================================================


@register_predicate("gripper_holding_object")
def gripper_holding_object(robot_id: str, world_state=None) -> Tuple[bool, str]:
    """
    Check if the robot's gripper is currently closed (holding an object).

    Uses the gripper_state field from WorldState RobotState, which is updated
    from Unity via WorldStateServer messages.

    Args:
        robot_id: Robot identifier
        world_state: Optional WorldState instance

    Returns:
        (is_valid, reason_if_invalid)
    """
    if world_state is None:
        try:
            from .WorldState import get_world_state
        except ImportError:
            from operations.WorldState import get_world_state
        world_state = get_world_state()

    state = world_state._robot_states.get(robot_id)
    if state is None:
        return False, f"Robot '{robot_id}' not found in world state"

    if state.gripper_state == "closed":
        return True, ""

    return False, f"Robot '{robot_id}' gripper is '{state.gripper_state}', expected 'closed'"


@register_predicate("stereo_images_available")
def stereo_images_available(max_age_seconds: float = 30.0, world_state=None) -> Tuple[bool, str]:
    """
    Check if a recent stereo image pair is available in UnifiedImageStorage.

    Queries the image storage singleton for the latest stereo pair timestamp
    and rejects pairs older than max_age_seconds.

    Args:
        max_age_seconds: Maximum acceptable age of stereo images in seconds
        world_state: Unused; accepted for predicate system compatibility

    Returns:
        (is_valid, reason_if_invalid)
    """
    import time

    try:
        from servers.ImageStorageCore import UnifiedImageStorage
    except ImportError:
        return False, "ImageStorageCore unavailable — cannot verify stereo images"

    storage = UnifiedImageStorage()
    if storage is None:
        return False, "UnifiedImageStorage not initialized"

    latest_ts = storage.get_latest_stereo_timestamp()
    if latest_ts == 0.0:
        return False, "No stereo images in storage"

    # Guard against None, 0 or negative values (e.g. from LLM or missing params).
    effective_max_age = max_age_seconds if (max_age_seconds is not None and max_age_seconds > 0) else 30.0

    age = time.time() - latest_ts
    if age > effective_max_age:
        return False, f"Stereo images are stale ({age:.1f}s old, max {effective_max_age}s)"

    return True, ""


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
