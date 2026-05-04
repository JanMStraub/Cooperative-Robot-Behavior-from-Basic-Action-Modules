#!/usr/bin/env python3
"""
Grasp Operations for Advanced Grasp Planning
=============================================

This module implements MoveIt2-inspired grasp planning operations that use
the full grasp planning pipeline with candidate generation, IK validation,
collision checking, and scoring.
"""

import logging
import math
import time
from typing import List, Optional

import numpy as np

from core.LoggingSetup import setup_logging

from .Base import (
    BasicOperation,
    OperationCategory,
    OperationComplexity,
    OperationParameter,
    OperationRelationship,
    OperationResult,
)

from .ROSDispatcher import _get_control_mode

setup_logging(__name__)
logger = logging.getLogger(__name__)

# Import from centralized lazy import system (prevents circular dependencies)
try:
    from ..core.Imports import get_command_broadcaster as _get_command_broadcaster
except ImportError:
    from core.Imports import get_command_broadcaster as _get_command_broadcaster  # type: ignore[no-redef]

try:
    from ..config.Vision import YOLO_MODEL_PATH as _YOLO_MODEL_PATH
except ImportError:
    from config.Vision import YOLO_MODEL_PATH as _YOLO_MODEL_PATH  # type: ignore[no-redef]

try:
    from ..config.Robot import (
        FOLLOW_TARGET_DRIFT_THRESHOLD,
        FOLLOW_TARGET_ENABLED,
        FOLLOW_TARGET_MAX_CORRECTIONS,
        GRASP_DESCENT_ACCELERATION_SCALING,
        GRASP_DESCENT_VELOCITY_SCALING,
        GRASP_TCP_OFFSET,
        PRE_GRASP_CLEARANCE_Y,
        PRE_GRASP_HOVER_OFFSET,
        PREGRASP_ACCELERATION_SCALING,
        PREGRASP_VELOCITY_SCALING,
    )
except ImportError:
    from config.Robot import (  # type: ignore[no-redef]
        FOLLOW_TARGET_DRIFT_THRESHOLD,
        FOLLOW_TARGET_ENABLED,
        FOLLOW_TARGET_MAX_CORRECTIONS,
        GRASP_DESCENT_ACCELERATION_SCALING,
        GRASP_DESCENT_VELOCITY_SCALING,
        GRASP_TCP_OFFSET,
        PRE_GRASP_CLEARANCE_Y,
        PRE_GRASP_HOVER_OFFSET,
        PREGRASP_ACCELERATION_SCALING,
        PREGRASP_VELOCITY_SCALING,
    )


def _execute_grasp_with_follow_target(
    bridge,
    robot_id: str,
    object_id: str,
    planned_position: dict,
    orientation: dict,
    tcp_y_offset: float = 0.0,
    world_state=None,
) -> bool:
    """Move arm to object, optionally correct for drift, then close gripper.

    After MoveIt delivers the arm to the initially-planned position, the target
    cube may have been pushed by the other robot's open fingers.  When
    FOLLOW_TARGET_ENABLED is True this function re-queries the live object
    position from WorldState and issues a corrective plan_and_execute if the
    object has drifted more than FOLLOW_TARGET_DRIFT_THRESHOLD metres.  Up to
    FOLLOW_TARGET_MAX_CORRECTIONS correction moves are attempted before the
    gripper is closed regardless.

    The gripper is only closed *after* the arm has settled at (or near) the
    final object position, so it never closes during the approach phase.

    Args:
        bridge: Connected ROSBridge instance.
        robot_id: Robot namespace (e.g. "Robot1").
        object_id: Object identifier used to look up live position in WorldState.
        planned_position: Initial target position dict with x/y/z keys.
        orientation: Gripper orientation quaternion dict with x/y/z/w keys.
        tcp_y_offset: Additional Y offset applied to live positions (e.g. 0.05m
            for top-down grasps so ee_link lands above the object centre).
        world_state: WorldState instance (obtained lazily if None).

    Returns:
        True if the gripper was closed successfully, False otherwise.
    """
    import math

    current_position = dict(planned_position)

    if FOLLOW_TARGET_ENABLED and world_state is not None:
        for correction in range(FOLLOW_TARGET_MAX_CORRECTIONS):
            live_pos = world_state.get_object_position(object_id)
            if live_pos is None:
                logger.debug(
                    f"[follow_target] {robot_id}: object '{object_id}' not in WorldState, skipping correction"
                )
                break

            # Compute drift distance in XZ plane (Y is vertical, cube stays on table)
            dx = live_pos[0] - (current_position["x"])
            dz = live_pos[2] - (current_position["z"])
            drift = math.sqrt(dx * dx + dz * dz)

            if drift <= FOLLOW_TARGET_DRIFT_THRESHOLD:
                logger.info(
                    f"[follow_target] {robot_id}: object drift {drift * 100:.1f} cm — within threshold, ready to close"
                )
                break

            logger.info(
                f"[follow_target] {robot_id}: object drifted {drift * 100:.1f} cm "
                f"(correction {correction + 1}/{FOLLOW_TARGET_MAX_CORRECTIONS}), re-planning"
            )

            # Build corrected target from live object position
            corrected = _vec_to_pos(live_pos, tcp_y_offset)
            current_position = corrected

            correction_result = bridge.plan_and_execute(
                position=corrected,
                orientation=orientation,
                planning_time=8.0,
                robot_id=robot_id,
                max_velocity_scaling=0.3,
                max_acceleration_scaling=0.3,
            )

            if not correction_result or not correction_result.get("success"):
                logger.warning(
                    f"[follow_target] {robot_id}: corrective move failed — "
                    f"{correction_result.get('error') if correction_result else 'no response'}"
                )
                break
    else:
        if not FOLLOW_TARGET_ENABLED:
            logger.debug(
                f"[follow_target] disabled — closing gripper at planned position"
            )

    # Arm is at (corrected) grasp position.
    # Wait for the ArticulationBody PD controller to settle before closing so the
    # gripper doesn't fire while the arm is still oscillating at the target pose.
    logger.info(
        f"[follow_target] {robot_id}: waiting for arm to settle before closing gripper"
    )
    time.sleep(0.5)

    # Close gripper
    logger.info(f"[follow_target] {robot_id}: closing gripper")
    gripper_result = bridge.control_gripper(0.0, robot_id=robot_id)
    if gripper_result and gripper_result.get("success"):
        # Give Unity physics time to register the contact before returning
        time.sleep(0.8)
        logger.info(f"[follow_target] {robot_id}: gripper closed")
        return True
    else:
        logger.warning(f"[follow_target] {robot_id}: gripper close command failed")
        return False


# ============================================================================
# Private Helpers
# ============================================================================


def _vec_to_pos(seq, y_offset: float = 0.0) -> dict:
    """Convert a 3-element sequence to a position dict, optionally shifting Y.

    Used for converting WorldState position tuples and grasp planner position
    arrays to the ``{x, y, z}`` dicts that ROSBridge methods expect.

    Args:
        seq: Any 3-element sequence with indices 0 (x), 1 (y), 2 (z).
        y_offset: Value added to the Y component (positive = up in Unity).

    Returns:
        Dict with float keys ``x``, ``y``, ``z``.
    """
    return {"x": seq[0], "y": seq[1] + y_offset, "z": seq[2]}


def _yaw_toward_object(robot_id: str, object_position) -> float:
    """Return Unity-frame yaw (radians) that orients the gripper jaw perpendicular
    to the robot-to-object vector so the fingers straddle the object from the sides.

    For a top-down grasp the jaw (open/close axis) must be perpendicular to the
    approach direction so the fingers land on either side of the object rather than
    on top of it.  We compute the robot→object azimuth and rotate +90°.

    Args:
        robot_id: Robot identifier used to look up base position.
        object_position: (x, y, z) object centre in Unity world space.

    Returns:
        Yaw angle in radians, normalised to (-π/2, π/2] for gripper symmetry.
    """
    try:
        from config.Robot import ROBOT_BASE_POSITIONS

        base = ROBOT_BASE_POSITIONS.get(robot_id, (0.0, 0.0, 0.0))
    except Exception:
        base = (0.0, 0.0, 0.0)

    dx = object_position[0] - base[0]
    dz = object_position[2] - base[2]
    # +90° rotation makes jaw perpendicular to approach so fingers straddle the object.
    yaw = math.atan2(dz, dx) + math.pi / 2

    # Normalise to (-π/2, π/2] to exploit 180° gripper symmetry.
    if yaw > math.pi / 2:
        yaw -= math.pi
    elif yaw <= -math.pi / 2:
        yaw += math.pi

    return yaw


def _get_control_mode() -> str:
    """Return DEFAULT_CONTROL_MODE from config, or 'ros' as fallback.

    Centralises the repeated try/except import pattern that was scattered
    across six nested except blocks in grasp_object.

    Returns:
        The configured control mode string (e.g., 'ros', 'hybrid', 'unity').
    """
    try:
        from config.ROS import DEFAULT_CONTROL_MODE

        return DEFAULT_CONTROL_MODE
    except ImportError:
        return "ros"


def _handle_ros_failure(error_msg: str, context: str):
    """Decide whether to fall back to TCP or return an error result.

    In hybrid mode a ROS planning failure is non-fatal: we log a warning
    and signal the caller to retry via the TCP path.  In any other mode we
    return an error OperationResult so the caller can surface it immediately.

    Args:
        error_msg: Human-readable description of the failure.
        context: Short label used in the warning log (e.g., function name).

    Returns:
        (should_fallback, error_result_or_None) tuple.
        When should_fallback is True the caller should proceed with TCP.
        When should_fallback is False the caller should return error_result.
    """
    if _get_control_mode() == "hybrid":
        logger.warning(f"{context}: {error_msg}, falling back to TCP")
        return True, None
    return False, OperationResult.error_result(
        "ROS_PLANNING_FAILED",
        error_msg,
        ["Check MoveIt logs", "Verify object is reachable"],
    )


def _yaw_from_world_state_or_robot(
    robot_id: str,
    object_id: str,
    object_position,
    world_state,
) -> tuple:
    """Return (yaw_unity_radians, yaw_source_str) for a top-down grasp.

    Priority:
    1. WorldState object rotation: jaw = object_Y_rotation + 90° (across short axis).
       Used when the object has a meaningful Y rotation (elongated objects).
    2. Approach-perpendicular fallback (robot-to-object + 90°) for symmetric/unknown objects.

    Normalised to (-π/2, π/2] for gripper symmetry.
    """
    if world_state is not None:
        try:
            with world_state._lock:
                _obj = world_state._objects.get(object_id)
                if _obj is None:
                    _norm = object_id.lower().replace(" ", "_").replace("-", "_")
                    for k, v in world_state._objects.items():
                        if _norm in k.lower() or k.lower() in _norm:
                            _obj = v
                            break
                if _obj is not None and _obj.rotation is not None:
                    # rotation[1] = Unity Y (up-axis) rotation in degrees.
                    # Unity Y is left-handed (CW from above) → negate for ROS.
                    # The jaw must align with the long axis of the object.
                    # If local X is longer: long axis is at -rotation[1] in ROS frame.
                    # If local Z is longer: long axis is at -rotation[1] + 90° (local Z is 90° from local X).
                    yaw_deg = _obj.rotation[1]
                    z_longer = (
                        _obj.dimensions is not None
                        and _obj.dimensions[2] > _obj.dimensions[0]
                    )
                    offset = math.pi / 2 if z_longer else 0.0
                    yaw_unity = -math.radians(yaw_deg) + offset
                    if yaw_unity > math.pi / 2:
                        yaw_unity -= math.pi
                    elif yaw_unity <= -math.pi / 2:
                        yaw_unity += math.pi
                    axis = "Z" if z_longer else "X"
                    return yaw_unity, f"WorldState long-{axis} (obj_yaw={yaw_deg:.1f}°→jaw={math.degrees(yaw_unity):.1f}°)"
        except Exception as _e:
            logger.warning(f"WorldState rotation lookup failed: {_e}")

    yaw_unity = _yaw_toward_object(robot_id, object_position)
    return yaw_unity, f"approach-perpendicular ({math.degrees(yaw_unity):.1f}°)"


def _grasp_via_ros_planned(
    bridge,
    robot_id: str,
    object_id: str,
    object_position,
    object_dimensions,
    robot_state,
    preferred_approach: str,
    request_id: int,
    world_state,
    grasp_yaw_override: "Optional[float]" = None,
    pre_grasp_distance: float = 0.0,
):
    """Execute grasp using the full GraspPlanner pipeline (ROS path).

    Generates grasp candidates, picks the highest-scoring one, moves to the
    pre-grasp hover position, descends via a Cartesian path, then closes the
    gripper.  Reports an error if the gripper close fails (Fix 5).

    Args:
        bridge: Connected ROSBridge instance.
        robot_id: Robot namespace (e.g. "Robot1").
        object_id: Name of the object to grasp.
        object_position: (x, y, z) tuple from WorldState.
        object_dimensions: Object size from WorldState (required for GraspPlanner).
        robot_state: RobotState with position from WorldState.
        preferred_approach: Approach hint ('top', 'front', 'side', or None for auto).
        request_id: Request tracking ID.
        world_state: WorldState instance for follow-target drift correction.
        grasp_yaw_override: Optional yaw in radians (Unity Y-axis). When set,
            overrides the WorldState object yaw passed to GraspPlanner so the
            computed grasp rotation aligns the jaw to the requested axis.

    Returns:
        (result, should_fallback) tuple.
        If should_fallback is True the caller should retry via TCP.
        Otherwise result is the definitive OperationResult.
    """
    try:
        from grasp_planning.GraspPlanner import GraspPlanner
    except ImportError as e:
        logger.warning(f"Grasp planning not available ({e}), using position-only")
        return None, True  # signal caller to try position-only path

    # Build object rotation quaternion for GraspPlanner.
    # Priority: explicit yaw override > WorldState object rotation > robot-to-object axis.
    # GraspPlanner expects a Unity-frame quaternion (x, y, z, w).
    obj_rot_quat = (0.0, 0.0, 0.0, 1.0)
    if grasp_yaw_override is not None:
        yaw_unity = grasp_yaw_override
    else:
        yaw_unity, yaw_source = _yaw_from_world_state_or_robot(
            robot_id, object_id, object_position, world_state
        )
        logger.info(f"[ROS planned] gripper yaw from {yaw_source}")

    # Build a Unity Y-axis rotation quaternion: (0, sin(θ/2), 0, cos(θ/2))
    half_y = yaw_unity / 2.0
    obj_rot_quat = (0.0, math.sin(half_y), 0.0, math.cos(half_y))

    try:
        planner = GraspPlanner()
        best_grasp = planner.plan_grasp(
            object_position=object_position,
            object_rotation=obj_rot_quat,
            object_size=object_dimensions,
            robot_id=robot_id,
            gripper_position=robot_state.position,
            gripper_rotation=None,
            use_moveit_ik=True,
            preferred_approach=(
                preferred_approach if preferred_approach != "auto" else None
            ),
            min_score=0.3,
        )
    except Exception as e:
        logger.error(f"Error during grasp planning: {e}", exc_info=True)
        fallback, err = _handle_ros_failure(
            f"Grasp planning failed: {str(e)}", "_grasp_via_ros_planned"
        )
        if fallback:
            return None, True
        return err, False

    if best_grasp is None:
        logger.warning(
            "Grasp planning found no valid candidates, falling back to position-only"
        )
        return None, True  # caller tries position-only

    logger.info(
        f"Grasp planning succeeded: {best_grasp.approach_type} approach, "
        f"score={best_grasp.total_score:.3f}"
    )

    # Build the MoveIt (ROS) orientation for the chosen grasp approach.
    #
    # The general Unity→ROS swap (x,y,z,w)→(z,-x,y,w) breaks for top-down grasps:
    # GraspCandidateGenerator produces Euler(180°,0°,90°) in Unity ZYX, which gives
    # w=0 exactly.  After the axis swap the result is a pure rotation around ROS -Y
    # (horizontal), so link5/6 end up horizontal instead of pointing down.
    #
    # For top: build the ROS quaternion directly from the already-computed yaw.
    # In ROS (Z-up), top-down = tool-Z along -Z_ros = 180° around ROS X composed
    # with yaw θ around ROS Z.  q_x180=(1,0,0,0); q_yaw=(0,0,sin(θ/2),cos(θ/2)).
    # Product (q_yaw * q_x180): x=cos(θ/2), y=-sin(θ/2), z=0, w=0.
    # w=0 is unavoidable for any 180° rotation; MoveIt normalises internally so the
    # axis direction is what matters — and now it is correct (pointing down).
    if best_grasp.approach_type == "top":
        # Fold yaw into (-π/2, π/2] (jaw 180° symmetry).
        yaw_ros = yaw_unity  # Unity Y-axis == ROS Z-axis
        while yaw_ros > math.pi / 2:
            yaw_ros -= math.pi
        while yaw_ros < -math.pi / 2:
            yaw_ros += math.pi
        s = math.sin(yaw_ros / 2.0)
        c = math.cos(yaw_ros / 2.0)
        ros_x, ros_y, ros_z, ros_w = c, -s, 0.0, 0.0
        logger.debug(
            f"[_grasp_via_ros_planned] Top-down ROS orientation: "
            f"yaw_ros={math.degrees(yaw_ros):.1f}° → q=({ros_x:.3f},{ros_y:.3f},{ros_z:.3f},{ros_w:.3f})"
        )
    else:
        # Front/side: standard Unity→ROS axis swap (x,y,z,w)→(z,-x,y,w).
        unity_q = best_grasp.grasp_rotation
        ros_x = unity_q[2]
        ros_y = -unity_q[0]
        ros_z = unity_q[1]
        ros_w = unity_q[3]
        if ros_w < 0.0:
            ros_x, ros_y, ros_z, ros_w = -ros_x, -ros_y, -ros_z, -ros_w
        # Fold yaw into (-π/2, π/2] for jaw symmetry.
        _yaw = math.atan2(
            2.0 * (ros_w * ros_z + ros_x * ros_y),
            1.0 - 2.0 * (ros_y * ros_y + ros_z * ros_z),
        )
        if _yaw > math.pi / 2 or _yaw < -math.pi / 2:
            ros_x, ros_y, ros_z, ros_w = -ros_y, ros_x, ros_w, -ros_z
            if ros_w < 0.0:
                ros_x, ros_y, ros_z, ros_w = -ros_x, -ros_y, -ros_z, -ros_w
            logger.debug(
                f"[_grasp_via_ros_planned] Yaw {math.degrees(_yaw):.1f}° outside (-90°,90°] — "
                f"applied 180° wrist flip to minimise joint-6 travel."
            )
    grasp_orientation = {"x": ros_x, "y": ros_y, "z": ros_z, "w": ros_w}

    grasp_pos = _vec_to_pos(best_grasp.grasp_position)

    # Override pre-grasp position if caller specified an explicit distance.
    # GraspCandidateGenerator derives it from object size (min 5cm), which is too
    # small for 2cm cubes — the caller's pre_grasp_distance (e.g. 0.15m) is used instead.
    if pre_grasp_distance > 0.0 and best_grasp.approach_direction is not None:
        pre_grasp_pt = (
            np.array(best_grasp.grasp_position)
            + np.array(best_grasp.approach_direction) * pre_grasp_distance
        )
        pre_grasp_pos = _vec_to_pos(tuple(pre_grasp_pt))
    else:
        pre_grasp_pos = _vec_to_pos(best_grasp.pre_grasp_position)

    # Step 1: Pre-grasp hover.
    # Omit orientation constraint for non-top approaches: constraining orientation at
    # pre-grasp shrinks the IK solution space and causes OMPL to fail at borderline
    # reach distances (side/front poses near the edge of the workspace).
    # Orientation is enforced at Step 2 (Cartesian descent) where it matters.
    _pre_grasp_orientation = grasp_orientation if preferred_approach == "top" else None
    logger.info(f"Moving to pre-grasp position for {robot_id}")
    pre_result = bridge.plan_and_execute(
        position=pre_grasp_pos,
        orientation=_pre_grasp_orientation,
        planning_time=10.0,
        robot_id=robot_id,
        max_velocity_scaling=PREGRASP_VELOCITY_SCALING,
        max_acceleration_scaling=PREGRASP_ACCELERATION_SCALING,
        constrain_joint4=True,
    )
    if not pre_result or not pre_result.get("success"):
        pre_err = pre_result.get("error", "Unknown") if pre_result else "No response"
        logger.warning(f"Pre-grasp move failed ({pre_err}), attempting direct grasp")

    # Brief pause so /joint_states has the settled pose before MoveIt samples start state.
    time.sleep(0.3)

    # Step 2: Cartesian descent to grasp position.
    # plan_cartesian_descent (not plan_and_execute) is required here: it constrains
    # the ee_link to a straight-line path so wrist joints cannot rotate to an
    # alternate IK solution and offset the gripper laterally during descent.
    logger.info(f"Descending to grasp position for {robot_id}")
    result = bridge.plan_cartesian_descent(
        position=grasp_pos,
        orientation=grasp_orientation,
        robot_id=robot_id,
        max_velocity_scaling=GRASP_DESCENT_VELOCITY_SCALING,
        max_acceleration_scaling=GRASP_DESCENT_ACCELERATION_SCALING,
    )

    if not result or not result.get("success"):
        error_msg = result.get("error", "Unknown") if result else "No response"
        fallback, err = _handle_ros_failure(
            f"MoveIt motion planning failed: {error_msg}", "_grasp_via_ros_planned"
        )
        if fallback:
            return None, True
        return err, False

    # Step 3: Follow-target drift correction + gripper close
    logger.info(f"Arm at grasp position, starting follow-target for {robot_id}")
    gripper_ok = _execute_grasp_with_follow_target(
        bridge=bridge,
        robot_id=robot_id,
        object_id=object_id,
        planned_position=grasp_pos,
        orientation=grasp_orientation,
        tcp_y_offset=0.0,
        world_state=world_state,
    )
    if not gripper_ok:
        return (
            OperationResult.error_result(
                "GRIPPER_CLOSE_FAILED",
                f"Arm reached grasp position but gripper close command failed for {robot_id}",
                [
                    "Check gripper hardware/simulation state",
                    "Verify GripperContactSensor is active",
                ],
            ),
            False,
        )

    return (
        OperationResult.success_result(
            {
                "robot_id": robot_id,
                "object_id": object_id,
                "position": object_position,
                "grasp_approach": best_grasp.approach_type,
                "grasp_score": best_grasp.total_score,
                "request_id": request_id,
                "status": "ros_executed_with_grasp_planning",
                "planning_time": result.get("planning_time", 0),
                "timestamp": time.time(),
            }
        ),
        False,
    )


def _grasp_via_ros_position_only(
    bridge,
    robot_id: str,
    object_id: str,
    object_position,
    request_id: int,
    world_state,
    grasp_yaw_override: "Optional[float]" = None,
):
    """Execute grasp using position-only ROS planning (no GraspPlanner).

    Used when object dimensions are unavailable or GraspPlanner is not
    installed.  Moves to a clearance height, then a top-down hover position,
    then descends via a Cartesian path, then closes the gripper.

    Args:
        bridge: Connected ROSBridge instance.
        robot_id: Robot namespace.
        object_id: Object name (used for drift correction lookup).
        object_position: (x, y, z) tuple from WorldState.
        request_id: Request tracking ID.
        world_state: WorldState instance for follow-target drift correction.
        grasp_yaw_override: Optional yaw in radians (Unity Y-axis). When set,
            overrides the WorldState object yaw so the gripper jaw aligns to
            the requested axis (e.g. handoff orientation).

    Returns:
        (result, should_fallback) tuple.  See _grasp_via_ros_planned for contract.
    """
    # Build top-down orientation with yaw baked in.
    # Priority: explicit override > WorldState object rotation > zero yaw.
    # Follows the same logic as the VGN+ROS path (_grasp_via_vgn_with_ros).
    #
    # Top-down base: q_topdown ≈ (x=0.9999, y=0, z=0, w=0.0087) — 179° around
    # ROS X axis, which flips ee_link Z downward.  A small positive w bias keeps
    # the quaternion in the w>0 hemisphere so MoveIt's IK solver always picks the
    # same wrist configuration and avoids the ±360° flip that occurs at w=0.
    # Priority: explicit yaw override > WorldState object rotation > robot-to-object axis.
    if grasp_yaw_override is not None:
        yaw_unity = grasp_yaw_override
        if yaw_unity > math.pi / 2:
            yaw_unity -= math.pi
        elif yaw_unity <= -math.pi / 2:
            yaw_unity += math.pi
    else:
        yaw_unity, yaw_source = _yaw_from_world_state_or_robot(
            robot_id, object_id, object_position, world_state
        )
        logger.info(f"[ROS pos-only] gripper yaw from {yaw_source}")

    # Compose q_yaw_ros * q_topdown (same formula as _grasp_via_vgn_with_ros).
    half = yaw_unity / 2.0
    qy_z = math.sin(half)
    qy_w = math.cos(half)
    bx, by, bz, bw = 0.9999, 0.0, 0.0, 0.0087
    ox = qy_w * bx - qy_z * by
    oy = qy_w * by + qy_z * bx
    oz = qy_w * bz + qy_z * bw
    ow = qy_w * bw - qy_z * bz
    mag = math.sqrt(ox * ox + oy * oy + oz * oz + ow * ow)
    top_down_orientation = {
        "x": ox / mag,
        "y": oy / mag,
        "z": oz / mag,
        "w": ow / mag,
    }
    logger.info(
        f"[ROS pos-only] top-down orientation yaw={math.degrees(yaw_unity):.1f}° "
        f"orientation={top_down_orientation}"
    )

    pre_grasp_position = _vec_to_pos(object_position, PRE_GRASP_HOVER_OFFSET)
    grasp_position = _vec_to_pos(object_position, GRASP_TCP_OFFSET)

    logger.info(
        f"[GRASP_DEBUG] {robot_id} object_position={object_position}, "
        f"pre_grasp_y={pre_grasp_position['y']:.3f}, grasp_y={grasp_position['y']:.3f}"
    )

    # Step 1: Clearance waypoint — move to safe height before descending toward object.
    # Mirrors VGN+ROS path (lines ~1460-1484): only insert when pre-grasp hover is below
    # PRE_GRASP_CLEARANCE_Y so the arm sweeps over obstacles before committing to approach.
    clearance_pos = {
        "x": pre_grasp_position["x"],
        "y": PRE_GRASP_CLEARANCE_Y,
        "z": pre_grasp_position["z"],
    }
    if pre_grasp_position["y"] < PRE_GRASP_CLEARANCE_Y:
        logger.info(f"[ROS pos-only] Clearance waypoint for {robot_id}: {clearance_pos}")
        clearance_result = bridge.plan_and_execute(
            position=clearance_pos,
            orientation=top_down_orientation,
            planning_time=10.0,
            robot_id=robot_id,
            max_velocity_scaling=PREGRASP_VELOCITY_SCALING,
            max_acceleration_scaling=PREGRASP_ACCELERATION_SCALING,
        )
        if not clearance_result or not clearance_result.get("success"):
            cl_err = (
                clearance_result.get("error", "Unknown")
                if clearance_result
                else "No response"
            )
            logger.warning(
                f"[ROS pos-only] Clearance waypoint failed ({cl_err}) — "
                "proceeding directly to pre-grasp"
            )
        else:
            time.sleep(0.2)

    # Step 2: Move to pre-grasp hover position.
    # TODO: remove orientation=top_down_orientation once VGN is implemented — VGN will
    #       supply approach-aligned orientations, making this heuristic unnecessary.
    #       The top-down constraint shrinks the IK solution space at borderline reach
    #       distances and can cause OMPL to fail before planning even starts.
    logger.info(f"Moving to pre-grasp position for {robot_id}")
    pre_result = bridge.plan_and_execute(
        position=pre_grasp_position,
        orientation=top_down_orientation,
        planning_time=10.0,
        robot_id=robot_id,
        max_velocity_scaling=PREGRASP_VELOCITY_SCALING,
        max_acceleration_scaling=PREGRASP_ACCELERATION_SCALING,
    )
    if not pre_result or not pre_result.get("success"):
        error_msg = pre_result.get("error", "Unknown") if pre_result else "No response"
        logger.warning(f"Pre-grasp move failed ({error_msg})")
        fallback, err = _handle_ros_failure(
            f"MoveIt pre-grasp planning failed: {error_msg}",
            "_grasp_via_ros_position_only",
        )
        if fallback:
            return None, True
        return err, False

    # Brief pause so /joint_states has the settled pose before MoveIt samples start state.
    time.sleep(0.3)

    # Step 3: Straight-line Cartesian descent to grasp position
    logger.info(
        f"[GRASP_DEBUG] {robot_id} starting Cartesian descent to "
        f"Unity y={grasp_position['y']:.3f} (object_y + {GRASP_TCP_OFFSET}m)"
    )
    result = bridge.plan_cartesian_descent(
        position=grasp_position,
        orientation=top_down_orientation,
        robot_id=robot_id,
        max_velocity_scaling=GRASP_DESCENT_VELOCITY_SCALING,
        max_acceleration_scaling=GRASP_DESCENT_ACCELERATION_SCALING,
    )

    if not result or not result.get("success"):
        error_msg = result.get("error", "Unknown") if result else "No response"
        fallback, err = _handle_ros_failure(
            f"MoveIt motion planning failed: {error_msg}",
            "_grasp_via_ros_position_only",
        )
        if fallback:
            return None, True
        return err, False

    # Step 4: Follow-target drift correction + gripper close
    logger.info(f"Arm at grasp position, starting follow-target for {robot_id}")
    gripper_ok = _execute_grasp_with_follow_target(
        bridge=bridge,
        robot_id=robot_id,
        object_id=object_id,
        planned_position=grasp_position,
        orientation=top_down_orientation,
        tcp_y_offset=GRASP_TCP_OFFSET,
        world_state=world_state,
    )
    if not gripper_ok:
        return (
            OperationResult.error_result(
                "GRIPPER_CLOSE_FAILED",
                f"Arm reached grasp position but gripper close command failed for {robot_id}",
                [
                    "Check gripper hardware/simulation state",
                    "Verify GripperContactSensor is active",
                ],
            ),
            False,
        )

    return (
        OperationResult.success_result(
            {
                "robot_id": robot_id,
                "object_id": object_id,
                "position": object_position,
                "request_id": request_id,
                "status": "ros_executed",
                "planning_time": result.get("planning_time", 0),
                "timestamp": time.time(),
            }
        ),
        False,
    )


# ============================================================================
# VGN helpers
# ============================================================================

# Import from shared module to avoid circular dependency with VGNClient
from operations.GraspUtils import _build_segmentation_mask  # noqa: F401


def _grasp_via_vgn(
    robot_id: str,
    object_id: str,
    preferred_approach: str,
    use_advanced_planning: bool,
    pre_grasp_distance: float,
    enable_retreat: bool,
    retreat_distance: float,
    request_id: int,
    custom_approach_vector: "Optional[List[float]]" = None,
) -> "Optional[OperationResult]":
    """Attempt to grasp using the local VGN neural grasp prediction pipeline.

    This is the VGN fast-path inside ``grasp_object``.  It:

    1. Generates a fresh point cloud via ``generate_point_cloud``.
    2. Detects the target object bounding box with ``detect_objects``.
    3. Calls VGNClient which internally: refines bbox via VLM, masks point cloud,
       builds TSDF, runs VGN inference, returns 6-DOF poses in camera frame.
    4. Transforms poses from camera frame to Unity world frame.
    5. Computes pre-grasp positions (approach hover).
    6. Sends the ``grasp_object`` command with ``precomputed_candidates`` to Unity.

    Returns:
        OperationResult on success or definitive failure; None if VGN is
        unavailable or produced no candidates (triggers geometric fallback).
    """
    import numpy as np

    # Lazy imports to respect layered architecture
    try:
        from config.Servers import VGN_TOP_K
    except ImportError:
        VGN_TOP_K = 20

    from operations.GraspFrameTransform import transform_grasp_poses_to_unity
    from operations.PointCloudOperations import generate_point_cloud
    from operations.VGNClient import VGNClient

    client = VGNClient()
    if not client.is_available():
        logger.info("[VGN] Model unavailable — will use geometric fallback")
        return None

    # 1. Generate point cloud
    pc_result = generate_point_cloud(robot_id=robot_id, request_id=request_id)
    if not pc_result.success:
        logger.warning(
            f"[VGN] generate_point_cloud failed ({pc_result.error}), using geometric fallback"
        )
        return None

    pc = pc_result.result
    assert pc is not None
    points_list = pc["points"]
    colors_list = pc["colors"]
    cam_pos = pc["camera_position"]
    cam_rot = pc["camera_rotation"]
    fov = pc["fov"]

    points_np = np.array(points_list, dtype=np.float32)
    colors_np = np.array(colors_list, dtype=np.uint8) if colors_list else None

    # 2. Detect target object to get YOLO bounding box and image for VGN
    yolo_bbox: tuple = (0, 0, 0, 0)
    image_np: "Optional[np.ndarray]" = None
    img_w = 640
    img_h = 480
    try:
        from core.Imports import get_unified_image_storage
        from vision.YOLODetector import YOLODetector

        _storage = get_unified_image_storage()
        _stereo = _storage.get_latest_stereo()
        if _stereo is not None:
            _, _left_img, _, _ = _stereo
            if _left_img is not None:
                img_h, img_w = _left_img.shape[:2]
                _detector = YOLODetector(model_path=_YOLO_MODEL_PATH)
                _det_result = _detector.detect_objects(_left_img, camera_id="main")
                obj_id_lower = object_id.lower().replace("_", " ")
                for _obj in _det_result.detections:
                    color_field = getattr(_obj, "color", "").lower()
                    if obj_id_lower in color_field or color_field in obj_id_lower:
                        yolo_bbox = (
                            int(_obj.bbox_x),
                            int(_obj.bbox_y),
                            int(_obj.bbox_w),
                            int(_obj.bbox_h),
                        )
                        logger.debug(f"[VGN] YOLO bbox for {object_id}: {yolo_bbox}")
                        break
    except Exception as exc:
        logger.debug(f"[VGN] Could not get YOLO bbox (non-fatal): {exc}")

    # 3. Retrieve left stereo image for VLM
    try:
        from core.Imports import get_unified_image_storage

        storage = get_unified_image_storage()
        stereo = storage.get_latest_stereo()
        if stereo is not None:
            _, left_img, _, _ = stereo
            image_np = left_img
    except Exception as exc:
        logger.debug(
            f"[VGN] Could not retrieve stereo image for VLM (non-fatal): {exc}"
        )

    if image_np is None:
        image_np = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    grasps = client.predict_grasps(
        points=points_np,
        colors=colors_np,
        image=image_np,
        yolo_bbox=yolo_bbox,
        object_label=object_id,
        image_width=img_w,
        image_height=img_h,
        fov=fov,
        top_k=VGN_TOP_K,
        cam_pos=cam_pos,
        cam_rot=cam_rot,
    )
    if not grasps:
        logger.info("[VGN] Returned no candidates — using geometric fallback")
        return None

    logger.info(f"[VGN] Candidates received: {len(grasps)}")

    # 5. Transform to Unity world frame (skip if VGN already output world-frame grasps)
    if grasps and grasps[0].get("_world_frame"):
        world_grasps = grasps
        logger.info("[VGN] Grasps already in Unity world frame — skipping transform")
    else:
        world_grasps = transform_grasp_poses_to_unity(grasps, cam_pos, cam_rot)
    if not world_grasps:
        logger.warning(
            "[VGN] Frame transform produced no valid poses — using geometric fallback"
        )
        return None

    # 5b. Filter and re-rank by custom approach vector when provided
    if custom_approach_vector is not None:
        cav = np.array(custom_approach_vector, dtype=np.float64)
        mag = np.linalg.norm(cav)
        if mag > 1e-6:
            cav_unit = cav / mag
            aligned = [
                g
                for g in world_grasps
                if np.dot(np.array(g["approach_direction"]), cav_unit) > 0.0
            ]
            world_grasps = aligned if aligned else world_grasps
            world_grasps.sort(
                key=lambda g: (
                    g.get("score", 0.0)
                    * np.dot(np.array(g["approach_direction"]), cav_unit)
                ),
                reverse=True,
            )
            logger.info(
                f"[VGN] custom_approach_vector filtered {len(world_grasps)} candidates "
                f"(from {len(grasps)} raw)"
            )

    # 6. Build precomputed_candidates list for Unity PlanGraspWithExternalCandidates
    hover = pre_grasp_distance if pre_grasp_distance > 0 else PRE_GRASP_HOVER_OFFSET
    candidates = []
    for g in world_grasps:
        pos = g["position"]
        rot = g["rotation"]
        approach = g["approach_direction"]

        # Pre-grasp: step back AGAINST the approach direction by hover distance.
        # approach_direction points toward the object (VGN convention), so we
        # subtract to place the pre-grasp behind the grasp point, not through it.
        pre_pos = [
            pos[0] - approach[0] * hover,
            pos[1] - approach[1] * hover,
            pos[2] - approach[2] * hover,
        ]

        candidates.append(
            {
                "pre_grasp_position": {
                    "x": pre_pos[0],
                    "y": pre_pos[1],
                    "z": pre_pos[2],
                },
                "pre_grasp_rotation": {
                    "x": rot[0],
                    "y": rot[1],
                    "z": rot[2],
                    "w": rot[3],
                },
                "grasp_position": {"x": pos[0], "y": pos[1], "z": pos[2]},
                "grasp_rotation": {"x": rot[0], "y": rot[1], "z": rot[2], "w": rot[3]},
                "approach_direction": {
                    "x": approach[0],
                    "y": approach[1],
                    "z": approach[2],
                },
                "grasp_depth": 0.5,
                "antipodal_score": g.get("score", 0.0),
                "vgn_score": g.get("score", 0.0),
                "approach_type": preferred_approach,
            }
        )

    # 7. Build and send grasp command with precomputed_candidates
    parameters = {
        "object_id": object_id,
        "use_advanced_planning": use_advanced_planning,
        "preferred_approach": preferred_approach.lower(),
        "pre_grasp_distance": pre_grasp_distance,
        "enable_retreat": enable_retreat,
        "retreat_distance": retreat_distance,
        "precomputed_candidates": candidates,
    }

    command = {
        "command_type": "grasp_object",
        "target_type": "robot",
        "robot_id": robot_id,
        "parameters": parameters,
        "request_id": request_id,
    }

    broadcaster = _get_command_broadcaster()
    if broadcaster is None:
        return OperationResult.error_result(
            "COMMUNICATION_ERROR",
            "CommandBroadcaster not available",
            ["Ensure CommandServer is running"],
        )

    logger.info(
        f"[VGN] Sending grasp_object: {robot_id} -> {object_id} "
        f"({len(candidates)} candidates)"
    )
    success = broadcaster.send_command(command, request_id)
    if success:
        return OperationResult.success_result(
            {
                "command_sent": True,
                "robot_id": robot_id,
                "object_id": object_id,
                "request_id": request_id,
                "vgn_candidates": len(candidates),
            }
        )
    return OperationResult.error_result(
        "COMMUNICATION_ERROR",
        "Failed to send VGN grasp command to Unity",
        ["Check Unity is connected to CommandServer"],
    )


def _grasp_via_vgn_with_ros(
    bridge,
    robot_id: str,
    object_id: str,
    preferred_approach: str,
    pre_grasp_distance: float,
    request_id: int,
    world_state,
    custom_approach_vector: "Optional[List[float]]" = None,
    grasp_yaw_override: "Optional[float]" = None,
) -> "Optional[OperationResult]":
    """Attempt grasp using VGN pose selection with MoveIt trajectory execution.

    This is the highest-priority path when both VGN and ROS are enabled.
    It combines VGN's 6-DOF pose quality (via local Apple Silicon inference)
    with MoveIt's collision-free trajectory planning.

    Steps:
    1. Check VGN model availability; return None immediately if unavailable.
    2. Generate a fresh stereo point cloud.
    3. Detect the target object for bbox + image retrieval.
    4. Query VGNClient (includes VLM bbox refinement) for ranked grasp poses.
    5. Transform poses from camera frame to Unity world frame.
    6. Pick the top-scoring candidate and compute its pre-grasp hover position.
    7. Move to pre-grasp via MoveIt plan_and_execute.
    8. Cartesian descent to the grasp position via plan_cartesian_descent.
    9. Follow-target drift correction + gripper close.

    Returns:
        None — VGN unavailable OR MoveIt planning failed before arm moved
               (arm has not moved; caller falls back to geometric ROS planning).
        OperationResult (error) — Arm descended but gripper close failed.
        OperationResult (success) — Full grasp executed successfully.
    """
    import numpy as np

    # Lazy imports — respect layered architecture
    try:
        from config.Servers import VGN_TOP_K
    except ImportError:
        VGN_TOP_K = 20

    from operations.GraspFrameTransform import transform_grasp_poses_to_unity
    from operations.PointCloudOperations import generate_point_cloud
    from operations.VGNClient import VGNClient

    # 1. Availability check
    client = VGNClient()
    if not client.is_available():
        logger.info("[VGN+ROS] Model unavailable — falling back to geometric ROS")
        return None

    # 2. Generate point cloud
    pc_result = generate_point_cloud(robot_id=robot_id, request_id=request_id)
    if not pc_result.success:
        logger.warning(
            f"[VGN+ROS] generate_point_cloud failed ({pc_result.error}), "
            "falling back to geometric ROS"
        )
        return None

    pc = pc_result.result
    assert pc is not None
    points_np = np.array(pc["points"], dtype=np.float32)
    colors_np = np.array(pc["colors"], dtype=np.uint8) if pc.get("colors") else None
    cam_pos = pc["camera_position"]
    cam_rot = pc["camera_rotation"]
    fov = pc["fov"]
    # Use the stereo image dimensions that were actually used for reconstruction
    img_w = pc.get("image_width", 640)
    img_h = pc.get("image_height", 480)

    # 3. Detect target object — get YOLO bbox for VGN masking AND world position.
    #
    # SGBM stereo is unreliable on flat synthetic Unity surfaces and systematically
    # overestimates depth (~1.8x), so VGN's stereo-derived centroid position is wrong.
    # WorldState (populated by a preceding detect_object_stereo / VisionProcessor run)
    # stores DepthEstimator positions which are well-calibrated.  Use that as the
    # grasp centre; rely on VGN only for orientation/approach direction.
    yolo_bbox: tuple = (0, 0, 0, 0)
    image_np: "Optional[np.ndarray]" = None
    det_img_w = img_w
    det_img_h = img_h
    _detection_depth_m: "Optional[float]" = None  # stereo bbox depth for VGN hint

    # Pull world position and dimensions from WorldState.
    _detected_world_pos: "Optional[List[float]]" = None
    _object_dimensions = None
    try:
        ws_pos = world_state.get_object_position(object_id)
        if ws_pos is not None:
            _detected_world_pos = list(ws_pos)
            logger.info(
                f"[VGN+ROS] Using WorldState position for '{object_id}': "
                f"{[round(v, 3) for v in _detected_world_pos]}"
            )
    except Exception:
        pass
    try:
        _object_dimensions = world_state.get_object_dimensions(object_id)
    except Exception:
        pass

    # Always run YOLO detection to get the bbox for VGN point-cloud masking.
    try:
        from core.Imports import get_unified_image_storage
        from vision.YOLODetector import YOLODetector

        _storage = get_unified_image_storage()
        _stereo = _storage.get_latest_stereo()
        if _stereo is not None:
            _, _left_img, _, _ = _stereo
            if _left_img is not None:
                det_img_h, det_img_w = _left_img.shape[:2]
                _detector = YOLODetector(model_path=_YOLO_MODEL_PATH)
                _det_result = _detector.detect_objects(_left_img, camera_id="main")
                obj_id_norm = object_id.lower().replace(" ", "_")
                for _obj in _det_result.detections:
                    color_field = getattr(_obj, "color", "").lower().replace(" ", "_")
                    if obj_id_norm in color_field or color_field in obj_id_norm:
                        yolo_bbox = (
                            int(_obj.bbox_x),
                            int(_obj.bbox_y),
                            int(_obj.bbox_w),
                            int(_obj.bbox_h),
                        )
                        _dm = getattr(_obj, "depth_m", None)
                        if _dm is not None:
                            _detection_depth_m = float(_dm)
                        break
    except Exception as exc:
        logger.debug(f"[VGN] Could not get YOLO bbox (non-fatal): {exc}")

    # Scale bbox from detection resolution to stereo/point-cloud resolution
    if yolo_bbox != (0, 0, 0, 0) and (det_img_w != img_w or det_img_h != img_h):
        scale_x = img_w / det_img_w
        scale_y = img_h / det_img_h
        bx, by, bw, bh = yolo_bbox
        yolo_bbox = (
            int(bx * scale_x),
            int(by * scale_y),
            int(bw * scale_x),
            int(bh * scale_y),
        )
        logger.info(
            f"[VGN] Scaled bbox {det_img_w}x{det_img_h}→{img_w}x{img_h}: {yolo_bbox}"
        )
    if yolo_bbox == (0, 0, 0, 0):
        logger.warning(
            f"[VGN] No valid bbox found for '{object_id}' — masking will use all points"
        )

    try:
        from core.Imports import get_unified_image_storage

        storage = get_unified_image_storage()
        stereo = storage.get_latest_stereo()
        if stereo is not None:
            _, left_img, _, _ = stereo
            image_np = left_img
            pass  # img_w/img_h already set from point cloud result
    except Exception as exc:
        logger.debug(f"[VGN+ROS] Stereo image retrieval (non-fatal): {exc}")

    if image_np is None:
        image_np = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # 4. Query VGN (VLM bbox refinement + TSDF + inference internal to VGNClient)
    logger.info(
        f"[VGN] Calling predict_grasps: image_width={img_w}, image_height={img_h}, fov={fov}, bbox={yolo_bbox}, points_shape={points_np.shape}"
    )
    import numpy as _np

    logger.info(
        f"[VGN] Point cloud sample (first 3): {points_np[:3].tolist()}, X range=[{points_np[:,0].min():.3f},{points_np[:,0].max():.3f}], Y=[{points_np[:,1].min():.3f},{points_np[:,1].max():.3f}], Z=[{points_np[:,2].min():.3f},{points_np[:,2].max():.3f}]"
    )
    grasps = client.predict_grasps(
        points=points_np,
        colors=colors_np,
        image=image_np,
        yolo_bbox=yolo_bbox,
        object_label=object_id,
        image_width=img_w,
        image_height=img_h,
        fov=fov,
        top_k=VGN_TOP_K,
        cam_pos=cam_pos,
        cam_rot=cam_rot,
        object_world_pos=_detected_world_pos,
        detection_depth_m=_detection_depth_m,
        object_dimensions=_object_dimensions,
    )
    if not grasps:
        logger.info("[VGN+ROS] No candidates returned — falling back to geometric ROS")
        return None

    # 5. Transform to Unity world frame (skip if VGN already output world-frame grasps)
    if grasps and grasps[0].get("_world_frame"):
        world_grasps = grasps
        logger.info(
            "[VGN+ROS] Grasps already in Unity world frame - skipping transform"
        )
    else:
        world_grasps = transform_grasp_poses_to_unity(grasps, cam_pos, cam_rot)
    if not world_grasps:
        logger.warning(
            "[VGN+ROS] Frame transform produced no valid poses — falling back"
        )
        return None

    # 5b. Filter and re-rank by custom approach vector when provided
    if custom_approach_vector is not None:
        cav = np.array(custom_approach_vector, dtype=np.float64)
        mag = np.linalg.norm(cav)
        if mag > 1e-6:
            cav_unit = cav / mag
            aligned = [
                g
                for g in world_grasps
                if np.dot(np.array(g["approach_direction"]), cav_unit) > 0.0
            ]
            world_grasps = aligned if aligned else world_grasps
            world_grasps.sort(
                key=lambda g: (
                    g.get("score", 0.0)
                    * np.dot(np.array(g["approach_direction"]), cav_unit)
                ),
                reverse=True,
            )
            logger.info(
                f"[VGN+ROS] custom_approach_vector filtered {len(world_grasps)} candidates "
                f"(from {len(grasps)} raw)"
            )

    # 6. Pick top candidate based on preferred_approach direction.
    _y_approaches = sorted(
        [g["approach_direction"][1] for g in world_grasps], reverse=True
    )
    logger.info(
        f"[VGN+ROS] Approach Y distribution (top 5): {[round(v,2) for v in _y_approaches[:5]]}"
    )
    _approach_lower = preferred_approach.lower() if preferred_approach else "top"
    if _approach_lower == "side":
        # Side approach: prefer candidates with high horizontal (X) component, low Y.
        _MIN_X_APPROACH = 0.3
        side_candidates = [
            g
            for g in world_grasps
            if abs(g.get("approach_direction", [0, 0, 0])[0]) >= _MIN_X_APPROACH
        ]
        if side_candidates:
            top = max(side_candidates, key=lambda g: g.get("score", 0.0))
            logger.info(
                f"[VGN+ROS] Selected side grasp "
                f"(X_approach={top['approach_direction'][0]:.2f}) from "
                f"{len(side_candidates)}/{len(world_grasps)} candidates"
            )
        else:
            top = max(
                world_grasps,
                key=lambda g: abs(g.get("approach_direction", [0, 0, 0])[0]),
            )
            logger.warning(
                f"[VGN+ROS] No grasp with |X_approach| >= {_MIN_X_APPROACH} — "
                f"using most-horizontal candidate (X_approach={top['approach_direction'][0]:.2f})"
            )
    elif _approach_lower == "front":
        _MIN_Z_APPROACH = 0.3
        front_candidates = [
            g
            for g in world_grasps
            if abs(g.get("approach_direction", [0, 0, 0])[2]) >= _MIN_Z_APPROACH
        ]
        if front_candidates:
            top = max(front_candidates, key=lambda g: g.get("score", 0.0))
            logger.info(
                f"[VGN+ROS] Selected front grasp "
                f"(Z_approach={top['approach_direction'][2]:.2f}) from "
                f"{len(front_candidates)}/{len(world_grasps)} candidates"
            )
        else:
            top = max(
                world_grasps,
                key=lambda g: abs(g.get("approach_direction", [0, 0, 0])[2]),
            )
            logger.warning(
                f"[VGN+ROS] No grasp with |Z_approach| >= {_MIN_Z_APPROACH} — "
                f"using most-frontal candidate"
            )
    else:
        # Default: top-down — prefer upward Y approach to avoid table collisions.
        _MIN_Y_APPROACH = 0.2
        top_down_candidates = [
            g
            for g in world_grasps
            if g.get("approach_direction", [0, 0, 0])[1] >= _MIN_Y_APPROACH
        ]
        if top_down_candidates:
            top = max(top_down_candidates, key=lambda g: g.get("score", 0.0))
            logger.info(
                f"[VGN+ROS] Selected top-down-feasible grasp "
                f"(Y_approach={top['approach_direction'][1]:.2f}) from "
                f"{len(top_down_candidates)}/{len(world_grasps)} candidates"
            )
        else:
            top = max(
                world_grasps, key=lambda g: g.get("approach_direction", [0, 0, 0])[1]
            )
            logger.warning(
                f"[VGN+ROS] No grasp with Y_approach >= {_MIN_Y_APPROACH} — "
                f"using most-top-down candidate (Y_approach={top['approach_direction'][1]:.2f})"
            )
    pos = top["position"]
    approach = top["approach_direction"]
    logger.info(
        f"[VGN+ROS] Top grasp world_pos={[round(v,3) for v in pos]}, approach={[round(v,3) for v in approach]}, cam_pos={cam_pos}, cam_rot={cam_rot}"
    )

    # Position anchor override: stereo reconstruction has a depth scale error for
    # synthetic Unity scenes (~1.8x overestimate).  DepthEstimator (WorldState) is
    # better calibrated, so substitute its world position as the grasp centre.
    # VGN's orientation/approach are still used.
    if _detected_world_pos:
        dp = _detected_world_pos
        logger.info(
            f"[VGN+ROS] Overriding VGN pos {[round(v,3) for v in pos]} with "
            f"DepthEstimator pos {[round(v,3) for v in dp]} for '{object_id}'"
        )
        pos = dp

    hover = pre_grasp_distance if pre_grasp_distance > 0 else PRE_GRASP_HOVER_OFFSET
    _approach_lower = preferred_approach.lower() if preferred_approach else "top"
    _is_top_down_approach = _approach_lower not in ("side", "front")

    # Orientation selection:
    # VGN rarely predicts near-vertical (top-down) grasps for table-top cubes —
    # the best approach Y in this scene is typically ~0.45, producing a twisted
    # wrist orientation that looks wrong.  For table grasps, use the proven
    # top_down_orientation (~179° around ROS X: gripper straight down).
    # For non-table scenarios (handoff, elevated object) where VGN predicts a
    # genuinely top-down approach (Y >= 0.7), trust VGN's full 6-DOF orientation.
    _TOP_DOWN_Y_THRESHOLD = 0.7
    _vgn_approach_y = approach[1]  # Unity Y = up
    if abs(_vgn_approach_y) >= _TOP_DOWN_Y_THRESHOLD:
        # VGN approach is sufficiently top-down — use VGN direction for pre-grasp offset,
        # but use approach-perpendicular yaw so the jaw straddles the object from the sides.
        pre_approach = approach

        if grasp_yaw_override is not None:
            yaw_unity = grasp_yaw_override
            if yaw_unity > math.pi / 2:
                yaw_unity -= math.pi
            elif yaw_unity < -math.pi / 2:
                yaw_unity += math.pi
            yaw_source = f"override ({math.degrees(yaw_unity):.1f}°)"
        else:
            yaw_unity, yaw_source = _yaw_from_world_state_or_robot(
                robot_id, object_id, pos, world_state
            )

        # Compose: q_yaw_ros * q_topdown (same as else branch)
        half = yaw_unity / 2.0
        qy_z = math.sin(half)
        qy_w = math.cos(half)
        bx, by, bz, bw = 0.9999, 0.0, 0.0, 0.0087
        ox = qy_w * bx - qy_z * by
        oy = qy_w * by + qy_z * bx
        oz = qy_w * bz + qy_z * bw
        ow = qy_w * bw - qy_z * bz
        mag = math.sqrt(ox * ox + oy * oy + oz * oz + ow * ow)
        orientation = {
            "x": ox / mag,
            "y": oy / mag,
            "z": oz / mag,
            "w": ow / mag,
        }
        logger.info(
            f"[VGN+ROS] Top-down + yaw={math.degrees(yaw_unity):.1f}° "
            f"from {yaw_source} "
            f"(VGN approach |Y|={abs(_vgn_approach_y):.2f} >= {_TOP_DOWN_Y_THRESHOLD}), "
            f"orientation={orientation}"
        )
    else:
        # For explicit side/front approach: use VGN approach direction for pre-grasp offset.
        # For table top-down fallback (shallow VGN): hover directly above.
        if not _is_top_down_approach:
            pre_approach = approach  # VGN approach direction (points toward object)
        else:
            pre_approach = [0.0, 1.0, 0.0]  # straight up in Unity world frame

        # Yaw: explicit override > WorldState short-axis > approach-perpendicular.
        if grasp_yaw_override is not None:
            yaw_unity = grasp_yaw_override
            yaw_source = f"override ({math.degrees(grasp_yaw_override):.1f}°)"
        else:
            yaw_unity, yaw_source = _yaw_from_world_state_or_robot(
                robot_id, object_id, pos, world_state
            )

        # Exploit 180° gripper symmetry: normalise to (-π/2, π/2] to minimise
        # wrist travel (grasping at θ and θ+π are physically identical).
        if yaw_unity > math.pi / 2:
            yaw_unity -= math.pi
        elif yaw_unity < -math.pi / 2:
            yaw_unity += math.pi
        # Compose: q_final = q_yaw_ros * q_topdown
        # q_topdown ≈ {x:0.9999, y:0, z:0, w:0.0087} (179° around ROS X = gripper down)
        # q_yaw_ros = {x:0, y:0, z:sin(θ/2), w:cos(θ/2)} (yaw around ROS Z = up)
        half = yaw_unity / 2.0
        qy_z = math.sin(half)
        qy_w = math.cos(half)
        # Full quaternion multiply q_yaw * q_topdown.
        # a = q_yaw = (ax=0, ay=0, az=qy_z, aw=qy_w)
        # b = q_topdown = (bx=0.9999, by=0, bz=0, bw=0.0087)
        # Formula: (aw*bx+ax*bw+ay*bz-az*by,
        #           aw*by+ay*bw+az*bx-ax*bz,
        #           aw*bz+az*bw+ax*by-ay*bx,
        #           aw*bw-ax*bx-ay*by-az*bz)
        bx, by, bz, bw = 0.9999, 0.0, 0.0, 0.0087
        ox = qy_w * bx - qy_z * by  # qy_w*bx + 0*bw + 0*bz - qy_z*by
        oy = qy_w * by + qy_z * bx  # qy_w*by + 0*bw + qy_z*bx - 0*bz
        oz = qy_w * bz + qy_z * bw  # qy_w*bz + qy_z*bw + 0*by - 0*bx
        ow = qy_w * bw - qy_z * bz  # qy_w*bw - 0*bx - 0*by - qy_z*bz
        # Normalise
        mag = math.sqrt(ox * ox + oy * oy + oz * oz + ow * ow)
        orientation = {
            "x": ox / mag,
            "y": oy / mag,
            "z": oz / mag,
            "w": ow / mag,
        }
        logger.info(
            f"[VGN+ROS] Top-down + yaw={math.degrees(yaw_unity):.1f}° "
            f"from {yaw_source} "
            f"(VGN approach |Y|={abs(_vgn_approach_y):.2f} < {_TOP_DOWN_Y_THRESHOLD}), "
            f"orientation={orientation}"
        )

    pre_grasp_pos = {
        "x": pos[0] + pre_approach[0] * hover,
        "y": pos[1] + pre_approach[1] * hover,
        "z": pos[2] + pre_approach[2] * hover,
    }
    # Apply GRASP_TCP_OFFSET along the approach direction so fingers stop at the
    # object surface rather than driving through it.
    grasp_pos = {
        "x": pos[0] + pre_approach[0] * GRASP_TCP_OFFSET,
        "y": pos[1] + pre_approach[1] * GRASP_TCP_OFFSET,
        "z": pos[2] + pre_approach[2] * GRASP_TCP_OFFSET,
    }

    # 7a. Clearance waypoint: only for top-down approaches where the arm might sweep
    #     through table-height space. Side/front approaches come from the side so no
    #     clearance above the object is needed.
    clearance_pos = {"x": pos[0], "y": PRE_GRASP_CLEARANCE_Y, "z": pos[2]}
    _is_top_down_approach = _approach_lower not in ("side", "front")
    if _is_top_down_approach and pre_grasp_pos["y"] < PRE_GRASP_CLEARANCE_Y:
        # Pre-grasp is below clearance height — insert the waypoint.
        logger.info(f"[VGN+ROS] Clearance waypoint for {robot_id}: {clearance_pos}")
        clearance_result = bridge.plan_and_execute(
            position=clearance_pos,
            orientation=orientation,
            planning_time=10.0,
            robot_id=robot_id,
            max_velocity_scaling=PREGRASP_VELOCITY_SCALING,
            max_acceleration_scaling=PREGRASP_ACCELERATION_SCALING,
            constrain_joint4=True,
        )
        if not clearance_result or not clearance_result.get("success"):
            cl_err = (
                clearance_result.get("error", "Unknown")
                if clearance_result
                else "No response"
            )
            logger.warning(
                f"[VGN+ROS] Clearance waypoint failed ({cl_err}) — "
                "proceeding directly to pre-grasp"
            )
        else:
            time.sleep(0.2)

    # 7b. MoveIt pre-grasp move.
    # Omit orientation constraint for side/front: constraining orientation at pre-grasp
    # shrinks the IK solution space and causes OMPL to fail near workspace boundaries.
    _pre_grasp_orientation = orientation if _is_top_down_approach else None
    logger.info(f"[VGN+ROS] Moving to pre-grasp for {robot_id}: {pre_grasp_pos}")
    # Attempt 1: with chosen orientation (top-down) or unconstrained (side/front)
    pre_result = bridge.plan_and_execute(
        position=pre_grasp_pos,
        orientation=_pre_grasp_orientation,
        planning_time=10.0,
        robot_id=robot_id,
        max_velocity_scaling=PREGRASP_VELOCITY_SCALING,
        max_acceleration_scaling=PREGRASP_ACCELERATION_SCALING,
        constrain_joint4=True,
    )
    if not pre_result or not pre_result.get("success"):
        pre_err = pre_result.get("error", "Unknown") if pre_result else "No response"
        logger.info(
            f"[VGN+ROS] Pre-grasp with orientation failed ({pre_err}) — "
            "retrying without orientation constraint"
        )
        # Attempt 2: position-only. constrain_joint6 prevents free-spin when orientation
        # is unconstrained; constrain_joint4 prevents long-arc IK solution.
        pre_result = bridge.plan_and_execute(
            position=pre_grasp_pos,
            orientation=None,
            planning_time=10.0,
            robot_id=robot_id,
            max_velocity_scaling=PREGRASP_VELOCITY_SCALING,
            max_acceleration_scaling=PREGRASP_ACCELERATION_SCALING,
            constrain_joint6=True,
            constrain_joint4=True,
        )
    if not pre_result or not pre_result.get("success"):
        pre_err = pre_result.get("error", "Unknown") if pre_result else "No response"
        logger.warning(
            f"[VGN+ROS] Pre-grasp planning failed ({pre_err}) — "
            "falling back to geometric ROS"
        )
        return None

    # 8. Settle pause (let /joint_states stabilise before MoveIt samples start state)
    time.sleep(0.3)

    # 9. Cartesian descent to grasp position
    logger.info(f"[VGN+ROS] Cartesian descent for {robot_id}: {grasp_pos}")
    descent_result = bridge.plan_cartesian_descent(
        position=grasp_pos,
        orientation=orientation,
        robot_id=robot_id,
        max_velocity_scaling=GRASP_DESCENT_VELOCITY_SCALING,
        max_acceleration_scaling=GRASP_DESCENT_ACCELERATION_SCALING,
    )
    if not descent_result or not descent_result.get("success"):
        descent_err = (
            descent_result.get("error", "Unknown") if descent_result else "No response"
        )
        logger.warning(
            f"[VGN+ROS] Cartesian descent failed ({descent_err}) — "
            "falling back to geometric ROS"
        )
        return None

    # 10. Follow-target drift correction + gripper close
    # Arm has descended — do NOT return None from here; return an error result.
    gripper_ok = _execute_grasp_with_follow_target(
        bridge=bridge,
        robot_id=robot_id,
        object_id=object_id,
        planned_position=grasp_pos,
        orientation=orientation,
        tcp_y_offset=0.0,
        world_state=world_state,
    )
    if not gripper_ok:
        return OperationResult.error_result(
            "GRIPPER_CLOSE_FAILED",
            f"Arm reached VGN pose but gripper close failed for {robot_id}",
            [
                "Check gripper hardware/simulation state",
                "Verify GripperContactSensor is active",
            ],
        )

    return OperationResult.success_result(
        {
            "robot_id": robot_id,
            "object_id": object_id,
            "request_id": request_id,
            "vgn_candidates": len(world_grasps),
            "status": "vgn_ros_executed",
            "timestamp": time.time(),
        }
    )


# ============================================================================
# Implementation: Grasp Object Operation
# ============================================================================


def grasp_object(
    robot_id: str,
    object_id: str,
    use_advanced_planning: bool = True,
    preferred_approach: str = "top",  # "top", "front", "side", "auto"
    pre_grasp_distance: float = 0.0,  # 0 = use config default
    enable_retreat: bool = True,
    retreat_distance: float = 0.0,  # 0 = use config default
    custom_approach_vector: Optional[List[float]] = None,  # [x, y, z] or None
    grasp_yaw_override: Optional[float] = None,  # radians; bypasses WorldState/VGN yaw
    request_id: int = 0,
    use_ros: Optional[bool] = None,
) -> OperationResult:
    """
    Plan and execute grasp on detected object using MoveIt2-inspired pipeline.

    This operation uses the full advanced grasp planning pipeline to:
    1. Generate multiple grasp candidates per approach type
    2. Filter candidates by IK reachability
    3. Filter candidates by collision-free approach paths
    4. Score candidates by weighted criteria (IK quality, approach, depth, stability)
    5. Execute best grasp with three-waypoint sequence (pre-grasp → grasp → retreat)

    The operation leverages GraspPlanningPipeline in Unity which implements:
    - Candidate generation with object-size adaptive distances
    - IK validation using damped least-squares solver
    - SphereCast collision checking along approach trajectories
    - Multi-criteria scoring with configurable weights
    - Safe retreat motion after grasping

    Args:
        robot_id: ID of the robot to control (e.g., "Robot1", "AR4_Robot")
        object_id: ID or name of the object to grasp (must be detected/tracked)
        use_advanced_planning: Use full pipeline (True) or simple planner (False)
        preferred_approach: Preferred grasp approach direction
            - "top": Approach from above, gripper pointing down (default)
            - "auto": Let pipeline determine best approach
            - "front": Approach from front/back
            - "side": Approach from left/right
        pre_grasp_distance: Custom pre-grasp distance in meters (0 = use config)
        enable_retreat: Whether to retreat after grasping
        retreat_distance: Custom retreat distance in meters (0 = use config)
        custom_approach_vector: Custom approach direction [x, y, z] (overrides preferred_approach)
        request_id: Request ID for tracking (optional)
        use_ros: Whether to use ROS for motion planning (None = auto-detect from config)

    Returns:
        Dict with the following structure:
        {
            "success": bool,           # True if grasp was planned and executed
            "result": dict or None,    # Result data if successful
            "error": dict or None      # Error information if failed
        }

        Success result structure:
        {
            "robot_id": str,
            "object_id": str,
            "approach_type": str,
            "score": float,            # Quality score of selected grasp
            "status": str,
            "timestamp": float
        }

        Error structure:
        {
            "code": str,                    # Error code (e.g., "PLANNING_FAILED")
            "message": str,                 # Human-readable error message
            "recovery_suggestions": list    # List of suggested actions
        }

    Example:
        >>> # Grasp object with automatic approach selection
        >>> result = grasp_object("Robot1", "Cube_01")
        >>> if result["success"]:
        ...     print(f"Grasped {result['result']['object_id']}")

        >>> # Grasp with specific approach direction
        >>> result = grasp_object("Robot1", "Cube_01", preferred_approach="top")

        >>> # Grasp without retreat motion
        >>> result = grasp_object("Robot1", "Cube_01", enable_retreat=False)

        >>> # Custom approach vector (approach from specific direction)
        >>> result = grasp_object("Robot1", "Cube_01",
        ...                       custom_approach_vector=[0, 1, 0.5])
    """
    try:
        # --- Input validation ---
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string, got: {robot_id}",
                [
                    "Provide a valid robot ID (e.g., 'Robot1', 'AR4_Robot')",
                    "Check RobotManager in Unity for available robot IDs",
                ],
            )

        if not object_id or not isinstance(object_id, str):
            return OperationResult.error_result(
                "INVALID_OBJECT_ID",
                f"Object ID must be a non-empty string, got: {object_id}",
                [
                    "Provide a valid object ID or name",
                    "Ensure object is detected and tracked in the scene",
                ],
            )

        valid_approaches = ["auto", "top", "front", "side"]
        if preferred_approach.lower() not in valid_approaches:
            return OperationResult.error_result(
                "INVALID_APPROACH",
                f"Preferred approach must be one of {valid_approaches}, got: {preferred_approach}",
                [
                    "Use 'top' for standard top-down approach (default)",
                    "Or specify 'front', 'side', or 'auto' explicitly",
                ],
            )

        if custom_approach_vector is not None:
            if (
                not isinstance(custom_approach_vector, (list, tuple))
                or len(custom_approach_vector) != 3
            ):
                return OperationResult.error_result(
                    "INVALID_APPROACH_VECTOR",
                    f"Custom approach vector must be a 3-element list [x, y, z], got: {custom_approach_vector}",
                    [
                        "Provide a valid 3D vector: [x, y, z]",
                        "Example: [0, 1, 0] for upward approach",
                    ],
                )

        # --- Determine execution path ---
        _use_ros = use_ros
        if _use_ros is None:
            try:
                from config.ROS import DEFAULT_CONTROL_MODE, ROS_ENABLED

                _use_ros = ROS_ENABLED and DEFAULT_CONTROL_MODE in ("ros", "hybrid")
            except ImportError:
                _use_ros = False

        try:
            from config.Servers import VGN_ENABLED as _vgn_enabled
        except ImportError:
            _vgn_enabled = False

        # --- ROS path ---
        bridge = None
        if _use_ros:
            try:
                from ros2.ROSBridge import ROSBridge

                bridge = ROSBridge.get_instance()
                if not bridge.is_connected and not bridge.connect():
                    should_fallback, err = _handle_ros_failure(
                        "Failed to connect to ROS bridge (port 5020)", "grasp_object"
                    )
                    if not should_fallback:
                        assert err is not None
                        return err
                    _use_ros = False

            except ImportError:
                logger.warning("ros2 module not available, falling back to TCP")
                _use_ros = False

        if _use_ros:
            try:
                from core.Imports import get_world_state

                world_state = get_world_state()
                object_position = world_state.get_object_position(object_id)

                logger.info(
                    f"[GRASP_DEBUG] {object_id} WorldState position: {object_position}"
                )

                if object_position is None:
                    # WorldState is empty — detection either wasn't run or failed silently.
                    # Log stored keys so the mismatch (e.g. "red" vs "red_cube") is visible.
                    stored_keys = list(world_state._objects.keys())
                    logger.warning(
                        f"Object '{object_id}' not found in WorldState "
                        f"(stored keys: {stored_keys}). "
                        f"Attempting inline detect_object_stereo before proceeding."
                    )

                    # Inline detection: derive color from object_id (e.g. "red_cube" → "red").
                    # This makes grasp_object self-healing when the LLM omits the detection step.
                    color_hint = (
                        object_id.split("_")[0] if "_" in object_id else object_id
                    )
                    try:
                        from operations.VisionOperations import detect_object_stereo

                        det_result = detect_object_stereo(
                            robot_id=robot_id,
                            color=color_hint,
                            camera_id="stereo",
                            request_id=request_id,
                        )
                        if det_result and det_result["success"]:
                            # Re-query WorldState — detection should have written the position
                            object_position = world_state.get_object_position(object_id)
                            if object_position is not None:
                                logger.info(
                                    f"Inline detection succeeded: '{object_id}' now at {object_position}"
                                )
                            else:
                                logger.warning(
                                    f"Inline detection returned success but '{object_id}' still "
                                    f"not in WorldState (stored: {list(world_state._objects.keys())})"
                                )
                        else:
                            err_msg = (
                                det_result["error"] if det_result else "no response"
                            )
                            logger.warning(f"Inline detection failed: {err_msg}")
                    except Exception as det_err:
                        logger.error(
                            f"Inline detection raised: {det_err}", exc_info=True
                        )

                if object_position is None:
                    should_fallback, err = _handle_ros_failure(
                        f"Object {object_id} not in WorldState after detection attempt — "
                        f"verify detect_object_stereo precedes grasp_object",
                        "grasp_object",
                    )
                    if not should_fallback:
                        assert err is not None
                        return err
                    _use_ros = False
                else:
                    logger.info(
                        f"Resolved {object_id} to position {object_position}, planning with ROS"
                    )

                    object_dimensions = world_state.get_object_dimensions(object_id)
                    robot_state = world_state.get_robot_state(robot_id)

                    # PATH 1: VGN pose + MoveIt execution (highest priority when both enabled)
                    if _vgn_enabled:
                        assert bridge is not None
                        result = _grasp_via_vgn_with_ros(
                            bridge=bridge,
                            robot_id=robot_id,
                            object_id=object_id,
                            preferred_approach=preferred_approach,
                            pre_grasp_distance=pre_grasp_distance,
                            request_id=request_id,
                            world_state=world_state,
                            custom_approach_vector=custom_approach_vector,
                            grasp_yaw_override=grasp_yaw_override,
                        )
                        if result is not None:
                            return result

                    # PATH 2: Geometric ROS planning (existing code, unchanged)
                    # Try full GraspPlanner pipeline when dimensions + robot pose are available
                    if (
                        object_dimensions is not None
                        and robot_state is not None
                        and robot_state.position is not None
                    ):
                        logger.info(f"Using grasp planning pipeline for {object_id}")
                        assert bridge is not None
                        ros_result, fallback = _grasp_via_ros_planned(
                            bridge=bridge,
                            robot_id=robot_id,
                            object_id=object_id,
                            object_position=object_position,
                            object_dimensions=object_dimensions,
                            robot_state=robot_state,
                            preferred_approach=preferred_approach.lower(),
                            request_id=request_id,
                            world_state=world_state,
                            grasp_yaw_override=grasp_yaw_override,
                            pre_grasp_distance=pre_grasp_distance,
                        )
                        if not fallback:
                            assert ros_result is not None
                            return ros_result
                        # fallback=True: fall through to position-only ROS

                    # Position-only ROS path (no dimensions or GraspPlanner fallback)
                    logger.info(f"Using position-only planning for {object_id}")
                    assert bridge is not None
                    ros_result, fallback = _grasp_via_ros_position_only(
                        bridge=bridge,
                        robot_id=robot_id,
                        object_id=object_id,
                        object_position=object_position,
                        request_id=request_id,
                        world_state=world_state,
                        grasp_yaw_override=grasp_yaw_override,
                    )
                    if not fallback:
                        assert ros_result is not None
                        return ros_result
                    _use_ros = False  # TCP fallback

            except Exception as e:
                logger.error(f"Error resolving object position for ROS: {e}")
                should_fallback, err = _handle_ros_failure(
                    f"Error preparing ROS grasp: {str(e)}", "grasp_object"
                )
                if not should_fallback:
                    assert err is not None
                    return err
                _use_ros = False

        # --- VGN neural path (optional, falls back to geometric on failure) ---
        if _vgn_enabled:
            vgn_result = _grasp_via_vgn(
                robot_id=robot_id,
                object_id=object_id,
                preferred_approach=preferred_approach,
                use_advanced_planning=use_advanced_planning,
                pre_grasp_distance=pre_grasp_distance,
                enable_retreat=enable_retreat,
                retreat_distance=retreat_distance,
                request_id=request_id,
                custom_approach_vector=custom_approach_vector,
            )
            if vgn_result is not None:
                return vgn_result
            logger.info(
                "[VGN] Unavailable or returned no candidates — "
                "falling back to geometric pipeline"
            )

        # --- TCP path (Unity grasp pipeline) ---
        parameters = {
            "object_id": object_id,
            "use_advanced_planning": use_advanced_planning,
            "preferred_approach": preferred_approach.lower(),
            "pre_grasp_distance": pre_grasp_distance,
            "enable_retreat": enable_retreat,
            "retreat_distance": retreat_distance,
        }

        # Add custom approach vector if provided
        if custom_approach_vector is not None:
            parameters["custom_approach_vector"] = {
                "x": custom_approach_vector[0],
                "y": custom_approach_vector[1],
                "z": custom_approach_vector[2],
            }

        command = {
            "command_type": "grasp_object",  # Fixed: Unity expects "command_type" not "command"
            "target_type": "robot",  # Added: Required field for routing
            "robot_id": robot_id,
            "parameters": parameters,  # Nest grasp parameters under "parameters" key
            "request_id": request_id,
        }

        # Send command to Unity via CommandBroadcaster
        broadcaster = _get_command_broadcaster()
        if broadcaster is None:
            return OperationResult.error_result(
                "COMMUNICATION_ERROR",
                "CommandBroadcaster not available",
                [
                    "Ensure CommandServer is running",
                    "Check server initialization in orchestrator",
                ],
            )

        # Send command (don't wait - SequenceExecutor handles completion waiting)
        logger.info(f"Sending grasp_object command: {robot_id} -> {object_id}")
        success = broadcaster.send_command(command, request_id)

        if success:
            # Return success immediately - SequenceExecutor will wait for Unity completion
            logger.debug(f"Grasp command sent successfully, request_id={request_id}")
            return OperationResult.success_result(
                {
                    "command_sent": True,
                    "robot_id": robot_id,
                    "object_id": object_id,
                    "request_id": request_id,
                }
            )
        else:
            logger.error(f"Failed to send grasp command")
            return OperationResult.error_result(
                "COMMUNICATION_ERROR",
                "Failed to send grasp command to Unity",
                [
                    "Check Unity is connected to CommandServer",
                    "Verify network connectivity",
                ],
            )

    except Exception as e:
        logger.exception(f"Exception in grasp_object operation: {e}")
        return OperationResult.error_result(
            "EXCEPTION",
            f"Exception during grasp operation: {str(e)}",
            [
                "Check stack trace in logs",
                "Verify all parameters are correct",
                "Ensure Unity is running and responsive",
            ],
        )


# ============================================================================
# Implementation: Orient Gripper For Handoff Receive Operation (helper)
# ============================================================================


def _compute_handoff_approach_vector(
    object_position: tuple,
    object_dimensions: tuple,
    receiving_robot_position: tuple,
) -> list:
    """Compute a grasp approach vector that leaves the handoff end accessible.

    Determines the object's longest horizontal axis, then returns a unit vector
    pointing toward the end of the object that is *furthest* from the receiving
    robot.  Robot A grasps from that far end so Robot B can approach from the
    near end without the two grippers colliding.

    Args:
        object_position: (x, y, z) world-space centre of the object.
        object_dimensions: (width, height, depth) of the object in metres.
            Corresponds to (x, y, z) extents.
        receiving_robot_position: (x, y, z) base/ee position of the robot that
            will receive the object.

    Returns:
        Normalised approach vector [x, y, z] for use as custom_approach_vector.
        Falls back to [0, 1, 0] (straight down) if geometry is degenerate.
    """
    import math

    ox = object_position[0]
    oz = object_position[2]
    ow = object_dimensions[0]
    od = object_dimensions[2]
    rx = receiving_robot_position[0]
    rz = receiving_robot_position[2]

    # Vector from object centre to receiving robot (horizontal plane only).
    # Y is vertical in Unity, so we ignore it to avoid biasing the axis
    # selection toward a tall thin object being grabbed from above.
    dx = rx - ox
    dz = rz - oz

    # Choose the dominant horizontal axis of the object (longest of x/z extent).
    # We compare width (x-extent) vs depth (z-extent) to find the elongated axis.
    if ow >= od:
        # Object is wider along X — the handoff axis is X.
        # Sign: approach from the end pointing AWAY from the receiving robot.
        sign = -1.0 if dx >= 0 else 1.0
        approach = [sign, 0.0, 0.0]
    else:
        # Object is longer along Z — the handoff axis is Z.
        sign = -1.0 if dz >= 0 else 1.0
        approach = [0.0, 0.0, sign]

    # Normalise (already unit length for axis-aligned vectors, but be safe).
    mag = math.sqrt(sum(v * v for v in approach))
    if mag < 1e-6:
        logger.warning("Handoff approach vector degenerate, falling back to top-down")
        return [0.0, 1.0, 0.0]

    return [v / mag for v in approach]


# ============================================================================
# receive_handoff — high-level receive: orient + move to approach + close
# ============================================================================


def receive_handoff(
    robot_id: str,
    object_id: str,
    source_robot_id: str,
    request_id: int = 0,
    use_ros: Optional[bool] = None,
    release_signal: Optional[str] = None,
) -> OperationResult:
    """Execute the full receive side of a handoff: orient gripper, move to approach, close.

    Autonomously computes approach position and orientation from WorldState
    geometry — no hardcoded coordinates required.  Mirrors the autonomy of
    ``grasp_object`` for the receiving robot.

    Steps (internal):
      1. Fetch object position + dimensions from WorldState.
      2. Compute approach position: object centre offset toward receiver by
         half object width + ``HANDOFF_GRIPPER_CLEARANCE``.
      3. Compute yaw toward object using ``_yaw_toward_object``.  pitch=0 so
         gripper stays horizontal (Robot1 presents at pitch=0).
      4. ``plan_and_execute`` (ROS) or ``move_to_coordinate`` (Unity) — move directly to
         approach position with orientation set, so OMPL finds the correct wrist config.
      5. ``control_gripper(open_gripper=False)``.
      8. Emit ``release_signal`` event (if provided) so source robot releases
         immediately — prevents Unity physics conflict from dual-gripper hold.

    Args:
        robot_id: ID of the receiving robot (e.g. ``"Robot2"``).
        object_id: ID of the object being handed off (must be in WorldState).
        source_robot_id: ID of the robot currently holding the object.
        request_id: Request tracking ID (optional).
        use_ros: Whether to use ROS motion planning (None = auto from config).
        release_signal: If provided, emit this event name immediately after
            gripper close so the source robot can release in parallel rather
            than waiting for this operation to fully return.

    Returns:
        OperationResult with ``approach_position`` and ``orientation`` in result dict.

    Example:
        >>> result = receive_handoff("Robot2", "red_cube", "Robot1")
    """
    try:
        for param, value in (
            ("robot_id", robot_id),
            ("object_id", object_id),
            ("source_robot_id", source_robot_id),
        ):
            if not value or not isinstance(value, str):
                return OperationResult.error_result(
                    f"INVALID_{param.upper()}",
                    f"{param} must be a non-empty string, got: {value}",
                    [f"Provide a valid {param} such as 'Robot2'"],
                )

        # ── 1. Fetch geometry ────────────────────────────────────────────────
        try:
            from config.Robot import (
                DEFAULT_HANDOFF_OBJECT_DIMENSIONS,
                HANDOFF_GRIPPER_CLEARANCE,
                ROBOT_BASE_POSITIONS,
            )
        except ImportError:
            DEFAULT_HANDOFF_OBJECT_DIMENSIONS = (0.02, 0.02, 0.02)
            HANDOFF_GRIPPER_CLEARANCE = 0.02
            ROBOT_BASE_POSITIONS = {}

        object_position = None
        object_dimensions = None
        receiver_pos = None
        world_state = None

        try:
            from core.Imports import get_world_state

            world_state = get_world_state()
            object_position = world_state.get_object_position(object_id)
            object_dimensions = world_state.get_object_dimensions(object_id)
            robot_state = world_state.get_robot_state(robot_id)
            if robot_state is not None and robot_state.position is not None:
                receiver_pos = robot_state.position
        except Exception:
            pass

        if object_position is None:
            return OperationResult.error_result(
                "OBJECT_NOT_IN_WORLD_STATE",
                f"Object '{object_id}' position not found in WorldState",
                ["Ensure detect_object_stereo was run before receive_handoff"],
            )

        if object_dimensions is None:
            object_dimensions = DEFAULT_HANDOFF_OBJECT_DIMENSIONS
            logger.warning(
                f"receive_handoff: dimensions for '{object_id}' unavailable, "
                f"using default {DEFAULT_HANDOFF_OBJECT_DIMENSIONS}"
            )

        if receiver_pos is None:
            receiver_pos = ROBOT_BASE_POSITIONS.get(robot_id, (0.0, 0.0, 0.0))
            logger.info(
                f"receive_handoff: robot state unavailable, using base position {receiver_pos}"
            )

        # ── 2. Compute approach position ─────────────────────────────────────
        # Robot1 holds object sideways, long axis along Y, bottom face toward Robot2.
        # ap_x: ee_link stops so fingertips land at the near face —
        #   near_face + GRASP_TCP_OFFSET (finger reach) + clearance.
        # ap_y: object centre — Robot1 grips from top, Robot2 approaches from side at same height.
        #
        # object_dimensions are local-frame (from BoxCollider.size * lossyScale).
        # Robot1 grips across the short axis and presents the short face toward Robot2.
        # The X half-extent Robot2 must clear = short axis / 2.
        lx = object_dimensions[0]
        lz = object_dimensions[2] if len(object_dimensions) > 2 else lx
        obj_x = min(lx, lz)
        approach_sign = 1.0 if receiver_pos[0] > object_position[0] else -1.0
        near_face_x = object_position[0] + approach_sign * obj_x * 0.5
        ap_x = near_face_x + approach_sign * HANDOFF_GRIPPER_CLEARANCE
        obj_height = object_dimensions[1] if len(object_dimensions) > 1 else 0.02
        ap_y = object_position[1] + obj_height * 0.25
        ap_z = object_position[2]
        logger.info(
            f"receive_handoff: approach_position=({ap_x:.3f}, {ap_y:.3f}, {ap_z:.3f})"
        )

        # ── 3. Static orientation ─────────────────────────────────────────────
        # Unity applies yaw relative to robot base (baseRotation * Euler(pitch,yaw,roll)).
        # Robot2 base is already 180° — yaw=0 in local frame points toward -X (handoff).
        # Mirrors Robot1's pitch=0,yaw=0,roll=0 lock exactly.
        static_yaw_deg = 0.0
        logger.info(
            "receive_handoff: using robot-local yaw=0° (base rotation handles world facing)"
        )

        from .MoveOperations import move_to_coordinate

        # ── 4–6. Move to final approach position with orientation ─────────────
        # Mirrors the top-down grasp pattern: single plan_and_execute with both
        # position AND orientation so OMPL finds a solution in the correct wrist
        # configuration from the start — no separate orient step, no Cartesian
        # complexity. pitch=0,yaw=0,roll=0 in Robot2's base_link = identity {w:1},
        # which points the gripper toward -X (handoff side) because Robot2's
        # base is already rotated 180° in Unity world space.
        _use_ros_approach = use_ros
        if _use_ros_approach is None:
            try:
                from config.ROS import DEFAULT_CONTROL_MODE, ROS_ENABLED

                _use_ros_approach = ROS_ENABLED and DEFAULT_CONTROL_MODE in (
                    "ros",
                    "hybrid",
                )
            except ImportError:
                _use_ros_approach = False

        # Side-approach orientation: roll=90° around ROS X-axis so fingers are
        # horizontal. q = (sin45, 0, 0, cos45). Single plan_and_execute with
        # position+orientation so OMPL finds one consistent IK solution — same
        # pattern as top-down grasp, no separate orient step needed.
        import math as _math

        _half = _math.radians(90.0) / 2.0
        handoff_orientation = {
            "x": _math.sin(_half),
            "y": 0.0,
            "z": 0.0,
            "w": _math.cos(_half),
        }

        if _use_ros_approach:
            from ros2.ROSBridge import ROSBridge

            bridge = ROSBridge.get_instance()
            approach_result = bridge.plan_and_execute(
                position={"x": ap_x, "y": ap_y, "z": ap_z},
                orientation=handoff_orientation,
                robot_id=robot_id,
                max_velocity_scaling=0.4,
                max_acceleration_scaling=0.3,
                constrain_joint4=True,
            )
            approach_success = approach_result and approach_result.get("success")
            approach_error = (approach_result or {}).get(
                "error", "No response from ROS bridge"
            )
        else:
            _move = move_to_coordinate(
                robot_id=robot_id,
                x=ap_x,
                y=ap_y,
                z=ap_z,
                request_id=request_id,
                use_ros=False,
            )
            approach_success = _move.success
            approach_error = _move.error

        if not approach_success:
            return OperationResult.error_result(
                "MOVE_FAILED",
                f"receive_handoff: approach failed — {approach_error}",
                [
                    "Check for workspace collision",
                    "Verify approach_position is reachable",
                ],
            )

        # ── 7. Close gripper ──────────────────────────────────────────────────
        from .GripperOperations import control_gripper

        gripper_result = control_gripper(
            robot_id=robot_id,
            open_gripper=False,
            request_id=request_id,
        )
        if not gripper_result.success:
            return OperationResult.error_result(
                "GRIPPER_FAILED",
                f"receive_handoff: gripper close failed — {gripper_result.error}",
                ["Check gripper state", "Verify object is within gripper reach"],
            )

        # ── 8. Signal source robot to release immediately ─────────────────────
        if release_signal:
            try:
                from .SyncOperations import EventBus

                EventBus().signal(release_signal)
                logger.info(
                    f"receive_handoff: emitted release signal '{release_signal}'"
                )
            except Exception as e:
                logger.warning(f"receive_handoff: failed to emit release signal — {e}")

        return OperationResult.success_result(
            {
                "robot_id": robot_id,
                "object_id": object_id,
                "approach_position": {"x": ap_x, "y": ap_y, "z": ap_z},
                "orientation": {"pitch": 0.0, "yaw": static_yaw_deg, "roll": 0.0},
                "status": "handoff_received",
            }
        )

    except Exception as e:
        logger.exception(f"Exception in receive_handoff: {e}")
        return OperationResult.error_result(
            "EXCEPTION",
            f"Exception during receive_handoff: {str(e)}",
            ["Check stack trace in logs", "Verify WorldState is populated"],
        )


RECEIVE_HANDOFF_OPERATION = BasicOperation(
    operation_id="coordination_receive_handoff_001",
    name="receive_handoff",
    category=OperationCategory.COORDINATION,
    complexity=OperationComplexity.COMPLEX,
    description=(
        "Full receive side of a handoff: orient gripper toward object, move to approach "
        "position, and close gripper. Approach geometry computed autonomously from WorldState."
    ),
    long_description="""
        High-level operation that handles the complete receive side of a robot-to-robot
        handoff.  Equivalent to grasp_object for the receiving robot.

        Autonomously derives approach position (object centre ± half-width + clearance)
        and gripper yaw (robot-base → object vector) from WorldState.  No hardcoded
        coordinates required.

        Internal sequence:
          1. Fetch object position + dimensions from WorldState.
          2. Compute approach position offset toward receiver.
          3. Compute yaw from _yaw_toward_object(); pitch=0 (horizontal approach).
          4. adjust_end_effector_orientation(pitch=0, yaw=yaw_deg, roll=0).
          5. move_to_coordinate(approach_position).
          6. control_gripper(open_gripper=False).

        Must be called AFTER the source robot has signalled readiness (r1_at_handoff)
        and AFTER detect_object_stereo so WorldState has current object geometry.
    """,
    usage_examples=[
        "Robot2 receives red cube from Robot1: "
        "receive_handoff(robot_id='Robot2', object_id='red_cube', source_robot_id='Robot1')",
    ],
    parameters=[
        OperationParameter(
            name="robot_id",
            type="str",
            description="ID of the receiving robot",
            required=True,
        ),
        OperationParameter(
            name="object_id",
            type="str",
            description="ID of the object being handed off (must be in WorldState)",
            required=True,
        ),
        OperationParameter(
            name="source_robot_id",
            type="str",
            description="ID of the robot currently holding the object",
            required=True,
        ),
        OperationParameter(
            name="release_signal",
            type="str",
            description="Event name to emit immediately after gripper close so source robot can release in parallel (optional)",
            required=False,
        ),
    ],
    preconditions=[
        "robot_is_initialized(robot_id)",
    ],
    postconditions=["gripper_closed(robot_id)", "robot_at_approach_position(robot_id)"],
    average_duration_ms=6000.0,
    success_rate=0.90,
    failure_modes=[
        "Object not in WorldState (run detect_object_stereo first)",
        "Approach position unreachable (workspace collision)",
        "Gripper close fails (object not in reach)",
    ],
    relationships=OperationRelationship(
        operation_id="coordination_receive_handoff_001",
        required_operations=["manipulation_grasp_object_001"],
        required_reasons={
            "manipulation_grasp_object_001": (
                "Source robot must hold the object so WorldState has current geometry"
            ),
        },
        commonly_paired_with=[
            "sync_wait_for_signal_001",
            "perception_detect_object_stereo_001",
            "manipulation_release_object_001",
        ],
        pairing_reasons={
            "sync_wait_for_signal_001": "Wait for source robot signal before receiving",
            "perception_detect_object_stereo_001": "Re-detect object position at presentation point",
            "manipulation_release_object_001": "Source robot releases after receive_handoff succeeds",
        },
        typical_after=[
            "sync_wait_for_signal_001",
            "perception_detect_object_stereo_001",
        ],
        typical_before=["manipulation_release_object_001"],
        coordination_requirements={
            "requires_peer_robot": True,
            "peer_robot_param": "source_robot_id",
            "coordination_pattern": "handoff",
        },
    ),
    implementation=receive_handoff,
)


# ============================================================================
# Operation Definition for Registry — Standard Grasp
# ============================================================================


GRASP_OBJECT_OPERATION = BasicOperation(
    operation_id="manipulation_grasp_object_001",
    name="grasp_object",
    category=OperationCategory.MANIPULATION,
    complexity=OperationComplexity.COMPLEX,
    description="Plan and execute grasp using MoveIt2-inspired pipeline with candidate generation, IK validation, collision checking, and scoring",
    long_description="""
        This operation uses a MoveIt2-inspired grasp planning pipeline to execute robust grasps.

        The pipeline includes:
        1. Candidate Generation: Generate multiple grasp poses per approach type (top, front, side)
        2. IK Filtering: Validate reachability using inverse kinematics solver
        3. Collision Filtering: Check approach paths for obstacles using SphereCast
        4. Multi-Criteria Scoring: Rank candidates by IK quality, approach preference, depth, and stability
        5. Three-Waypoint Execution: Pre-grasp → Grasp → Retreat sequence

        This operation provides superior grasp success rates compared to simple planning by:
        - Testing multiple approach directions and selecting the best
        - Validating reachability before execution
        - Avoiding collision paths
        - Adapting pre-grasp distances to object size
        - Including safe retreat motions after grasping
    """,
    usage_examples=[
        "Grasp object top-down (default): grasp_object(robot_id='Robot1', object_id='Cube_01')",
        "Grasp from specific direction: grasp_object(robot_id='Robot1', object_id='Cube_01', preferred_approach='side')",
        "Grasp with custom distances: grasp_object(robot_id='Robot1', object_id='Cube_01', pre_grasp_distance=0.12, retreat_distance=0.15)",
        "Grasp without retreat: grasp_object(robot_id='Robot1', object_id='Cube_01', enable_retreat=False)",
        "Grasp with custom approach vector: grasp_object(robot_id='Robot1', object_id='Cube_01', custom_approach_vector=[0, 1, 0.5])",
    ],
    parameters=[
        OperationParameter(
            name="robot_id",
            type="str",
            description="ID of the robot to control (e.g., 'Robot1', 'AR4_Robot')",
            required=True,
        ),
        OperationParameter(
            name="object_id",
            type="str",
            description="ID or name of the object to grasp (must be detected/tracked)",
            required=True,
        ),
        OperationParameter(
            name="use_advanced_planning",
            type="bool",
            description="Use full planning pipeline (True) or simple planner (False)",
            required=False,
            default=True,
        ),
        OperationParameter(
            name="preferred_approach",
            type="str",
            description="Preferred grasp approach: 'top' (default, top-down), 'front', 'side', 'auto'",
            required=False,
            default="top",
            valid_values=["auto", "top", "front", "side"],
        ),
        OperationParameter(
            name="pre_grasp_distance",
            type="float",
            description="Custom pre-grasp distance in meters (0 = use config default)",
            required=False,
            default=0.0,
            valid_range=(0.0, 0.3),
        ),
        OperationParameter(
            name="enable_retreat",
            type="bool",
            description="Whether to retreat after grasping",
            required=False,
            default=True,
        ),
        OperationParameter(
            name="retreat_distance",
            type="float",
            description="Custom retreat distance in meters (0 = use config default)",
            required=False,
            default=0.0,
            valid_range=(0.0, 0.5),
        ),
        OperationParameter(
            name="custom_approach_vector",
            type="list",
            description="Custom approach direction [x, y, z] (overrides preferred_approach)",
            required=False,
            default=None,
        ),
    ],
    preconditions=[
        "robot_is_initialized(robot_id)",
    ],
    postconditions=[],
    average_duration_ms=150.0,
    success_rate=0.92,
    failure_modes=[
        "No valid grasp candidates found (all filtered out)",
        "IK validation failed for all candidates",
        "All approach paths have collisions",
        "Object not found in scene",
        "Object outside robot reach",
    ],
    relationships=OperationRelationship(
        operation_id="manipulation_grasp_object_001",
        required_operations=[
            "perception_stereo_detect_001",
            "status_check_robot_001",
        ],
        required_reasons={
            "perception_stereo_detect_001": "Object must be detected with 3D world coordinates before grasp planning can begin",
            "status_check_robot_001": "Robot must be initialized and responsive before executing complex grasp pipeline",
        },
        commonly_paired_with=[
            "manipulation_control_gripper_001",
            "motion_move_to_coord_001",
            "motion_return_to_start_001",
        ],
        pairing_reasons={
            "manipulation_control_gripper_001": "Open gripper before approach, close after grasping for controlled pickup",
            "motion_move_to_coord_001": "Move to pre-grasp position before executing final grasp",
            "motion_return_to_start_001": "Return to safe position after successful grasp with object in hand",
        },
        typical_after=[
            "perception_stereo_detect_001",
            "status_check_robot_001",
            "motion_move_to_coord_001",
        ],
        typical_before=[
            "motion_move_to_coord_001",
            "manipulation_control_gripper_001",
            "motion_return_to_start_001",
        ],
    ),
    implementation=grasp_object,
)
