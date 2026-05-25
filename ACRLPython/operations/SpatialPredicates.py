#!/usr/bin/env python3
"""Boolean predicates for spatial relationships and robot state — used by the verification system to check preconditions."""

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

from core.LoggingSetup import get_logger

logger = get_logger(__name__)

# Global predicate registry
PREDICATE_REGISTRY: Dict[str, Callable] = {}


def register_predicate(name: str):
    def decorator(func: Callable) -> Callable:
        PREDICATE_REGISTRY[name] = func
        return func

    return decorator


def get_predicate(name: str) -> Optional[Callable]:
    return PREDICATE_REGISTRY.get(name)


@register_predicate("target_within_reach")
def target_within_reach(
    robot_id: str, x: float, y: float, z: float, world_state=None
) -> Tuple[bool, str]:
    try:
        if x is None or y is None or z is None:
            return False, f"Target coordinates contain None: ({x}, {y}, {z})"

        base_pos = ROBOT_BASE_POSITIONS.get(robot_id)
        if base_pos is None:
            return False, f"Unknown robot '{robot_id}' - not in ROBOT_BASE_POSITIONS"

        dx = x - base_pos[0]
        dy = y - base_pos[1]
        dz = z - base_pos[2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

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
    try:
        workspace_name = ROBOT_WORKSPACE_ASSIGNMENTS.get(robot_id)
        if workspace_name is None:
            return False, f"Robot '{robot_id}' has no assigned workspace"

        workspace = WORKSPACE_REGIONS.get(workspace_name)
        if workspace is None:
            return (
                False,
                f"Workspace '{workspace_name}' not defined in WORKSPACE_REGIONS",
            )

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
    try:
        shared_zone = WORKSPACE_REGIONS.get("shared_zone")
        if shared_zone is None:
            return False, "Shared zone not defined in WORKSPACE_REGIONS"

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
    try:
        if robot_id not in ROBOT_BASE_POSITIONS:
            return False, f"Robot '{robot_id}' not found in system configuration"

        if world_state is not None:
            try:
                status = world_state.get_robot_status(robot_id)
                if status is None:
                    logger.debug(
                        f"Robot '{robot_id}' status unavailable, using basic check"
                    )
                elif "is_initialized" in status:
                    if not status.get("is_initialized"):
                        return False, f"Robot '{robot_id}' is not initialized"
                    return True, ""
                else:
                    # query_sent but no Unity response yet — basic check fallback
                    logger.debug(
                        f"Robot '{robot_id}' status pending, using basic check"
                    )
            except Exception as e:
                logger.warning(f"Could not query robot status: {e}")
                # Fall through to basic check

        return True, ""

    except Exception as e:
        logger.error(f"Error in robot_is_initialized: {e}")
        return False, f"Error checking initialization: {str(e)}"


@register_predicate("robot_is_stationary")
def robot_is_stationary(robot_id: str, world_state=None) -> Tuple[bool, str]:
    try:
        if world_state is None:
            return False, "WorldState required to check robot movement"

        status = world_state.get_robot_status(robot_id)
        if status is None:
            return False, f"Robot '{robot_id}' status unavailable"

        is_moving = status.get("is_moving", False)
        if is_moving:
            return False, f"Robot '{robot_id}' is currently moving"

        return True, ""

    except Exception as e:
        logger.error(f"Error in robot_is_stationary: {e}")
        return False, f"Error checking movement: {str(e)}"


@register_predicate("gripper_is_open")
def gripper_is_open(robot_id: str, world_state=None) -> Tuple[bool, str]:
    try:
        if world_state is None:
            return False, "WorldState required to check gripper state"

        status = world_state.get_robot_status(robot_id)
        if status is None:
            return False, f"Robot '{robot_id}' status unavailable"

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
    try:
        if world_state is None:
            return False, "WorldState required to check gripper state"

        status = world_state.get_robot_status(robot_id)
        if status is None:
            return False, f"Robot '{robot_id}' status unavailable"

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
    try:
        x, y, z = object_position

        is_reachable, reach_reason = target_within_reach(robot_id, x, y, z, world_state)
        if not is_reachable:
            return False, f"Object not reachable: {reach_reason}"

        is_shared, _ = is_in_shared_zone(x, y, z)
        if is_shared:
            return True, ""

        is_in_workspace, workspace_reason = is_in_robot_workspace(
            robot_id, x, y, z, world_state
        )
        if not is_in_workspace:
            return False, f"Object not in workspace: {workspace_reason}"

        return True, ""

    except Exception as e:
        logger.error(f"Error in object_accessible_by_robot: {e}")
        return False, f"Error checking accessibility: {str(e)}"


@register_predicate("robots_will_collide")
def robots_will_collide(
    robot1_id: str,
    target1: Tuple[float, float, float],
    robot2_id: str,
    target2: Tuple[float, float, float],
    world_state=None,
) -> Tuple[bool, str]:
    """Simplified collision check: target separation, linear path intersection, exclusive workspace overlap."""
    try:
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

        if world_state is not None:
            pos1 = world_state.get_robot_position(robot1_id)
            pos2 = world_state.get_robot_position(robot2_id)

            if pos1 is not None and pos2 is not None:
                min_path_distance = _calculate_segment_distance(
                    pos1, target1, pos2, target2
                )

                safety_margin = COLLISION_SAFETY_MARGIN
                if min_path_distance < safety_margin:
                    return True, (
                        f"Paths will intersect: minimum distance {min_path_distance:.3f}m "
                        f"(safety margin: {safety_margin}m)"
                    )

        is_shared1, _ = is_in_shared_zone(*target1)
        is_shared2, _ = is_in_shared_zone(*target2)

        if not is_shared1 and not is_shared2:
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
    d1 = tuple(p1_end[i] - p1_start[i] for i in range(3))
    d2 = tuple(p2_end[i] - p2_start[i] for i in range(3))
    r = tuple(p1_start[i] - p2_start[i] for i in range(3))

    a = sum(d1[i] * d1[i] for i in range(3))
    b = sum(d1[i] * d2[i] for i in range(3))
    c = sum(d2[i] * d2[i] for i in range(3))
    d = sum(d1[i] * r[i] for i in range(3))
    e = sum(d2[i] * r[i] for i in range(3))

    denominator = a * c - b * b
    if abs(denominator) < 1e-10:
        # parallel segments — need all 4 endpoint distances; single projection fails for adjacent/overlapping
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

    t1 = max(0.0, min(1.0, (b * e - c * d) / denominator))
    t2 = max(0.0, min(1.0, (a * e - b * d) / denominator))

    point1 = tuple(p1_start[i] + t1 * d1[i] for i in range(3))
    point2 = tuple(p2_start[i] + t2 * d2[i] for i in range(3))

    dist = math.sqrt(sum((point1[i] - point2[i]) ** 2 for i in range(3)))
    return dist


def _get_workspace_containing_point(x: float, y: float, z: float) -> Optional[str]:
    for region_name, bounds in WORKSPACE_REGIONS.items():
        if (
            bounds["x_min"] <= x <= bounds["x_max"]
            and bounds["y_min"] <= y <= bounds["y_max"]
            and bounds["z_min"] <= z <= bounds["z_max"]
        ):
            return region_name
    return None


@register_predicate("object_not_stale")
def object_not_stale(object_id: str, world_state=None) -> Tuple[bool, str]:
    if world_state is None:
        try:
            from .WorldState import get_world_state
        except ImportError:
            from operations.WorldState import get_world_state
        world_state = get_world_state()

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
    if world_state is None:
        try:
            from .WorldState import get_world_state
        except ImportError:
            from operations.WorldState import get_world_state
        world_state = get_world_state()

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
    if world_state is None:
        try:
            from .WorldState import get_world_state
        except ImportError:
            from operations.WorldState import get_world_state
        world_state = get_world_state()

    if region not in WORKSPACE_REGIONS:
        return False, f"Unknown workspace region: '{region}'"

    owner = world_state.get_workspace_owner(region)

    if owner is None or owner == robot_id:
        return True, ""

    return False, f"Region '{region}' is allocated to {owner}"


@register_predicate("gripper_holding_object")
def gripper_holding_object(robot_id: str, world_state=None) -> Tuple[bool, str]:
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

    return (
        False,
        f"Robot '{robot_id}' gripper is '{state.gripper_state}', expected 'closed'",
    )


@register_predicate("stereo_images_available")
def stereo_images_available(
    max_age_seconds: float = 30.0, world_state=None
) -> Tuple[bool, str]:
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

    # LLM sometimes passes None or 0 — clamp to sensible default
    effective_max_age = (
        max_age_seconds
        if (max_age_seconds is not None and max_age_seconds > 0)
        else 30.0
    )

    age = time.time() - latest_ts
    if age > effective_max_age:
        return (
            False,
            f"Stereo images are stale ({age:.1f}s old, max {effective_max_age}s)",
        )

    return True, ""


def evaluate_predicate(predicate_name: str, **kwargs) -> Tuple[bool, str]:
    predicate = get_predicate(predicate_name)
    if predicate is None:
        return False, f"Unknown predicate: {predicate_name}"

    try:
        return predicate(**kwargs)
    except Exception as e:
        logger.error(f"Error evaluating predicate '{predicate_name}': {e}")
        return False, f"Predicate evaluation error: {str(e)}"


def list_predicates() -> list[str]:
    return list(PREDICATE_REGISTRY.keys())
