#!/usr/bin/env python3
"""Small helpers shared across grasp sub-modules."""

import logging
import math
import time

from core.LoggingSetup import setup_logging

from ..Base import OperationResult

setup_logging(__name__)
logger = logging.getLogger(__name__)

try:
    from ...config.Robot import (
        FOLLOW_TARGET_DRIFT_THRESHOLD,
        FOLLOW_TARGET_ENABLED,
        FOLLOW_TARGET_MAX_CORRECTIONS,
        FOLLOW_TARGET_RETRACT_HEIGHT,
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
        FOLLOW_TARGET_RETRACT_HEIGHT,
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
    approach_offset_xz: tuple = (0.0, 0.0),
) -> "tuple[bool, str]":
    """Move to planned position, optionally correct for object drift, then close gripper.

    approach_offset_xz: (dx, dz) from object centre to the planned grasp position
    (e.g. left_side Z offset). Applied to live_pos during drift correction so the
    corrected target stays at the correct approach side of the object instead of
    drifting to the raw centre (which may collide with a bracing robot).

    Returns (True, "") on success, (False, reason) on failure.
    """
    import math

    try:
        from core.Imports import is_sequence_aborted as _is_aborted
    except ImportError:
        def _is_aborted() -> bool:
            return False

    current_position = dict(planned_position)

    if FOLLOW_TARGET_ENABLED and world_state is not None:
        for correction in range(FOLLOW_TARGET_MAX_CORRECTIONS):
            if _is_aborted():
                logger.info(f"[follow_target] {robot_id}: sequence aborted — stopping correction")
                return False, "sequence aborted"

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

            # Retract straight up before replanning so the gripper doesn't drag
            # along the table surface on the way to the new object position.
            retract_pos = dict(current_position)
            retract_pos["y"] = current_position["y"] + FOLLOW_TARGET_RETRACT_HEIGHT
            logger.info(
                f"[follow_target] {robot_id}: retracting {FOLLOW_TARGET_RETRACT_HEIGHT * 100:.0f} cm before replan"
            )
            retract_result = bridge.plan_and_execute(
                position=retract_pos,
                orientation=orientation,
                planning_time=5.0,
                robot_id=robot_id,
                max_velocity_scaling=0.8,
                max_acceleration_scaling=0.7,
            )
            if not retract_result or not retract_result.get("success"):
                logger.warning(
                    f"[follow_target] {robot_id}: retract failed — "
                    f"{retract_result.get('error') if retract_result else 'no response'}, aborting correction"
                )
                break

            # Apply the same XZ approach offset as the original grasp so the corrected
            # target stays at the correct side of the object (e.g. left_side) rather
            # than the raw centre — which could be occupied by a bracing robot.
            corrected = {
                "x": live_pos[0] + approach_offset_xz[0],
                "y": live_pos[1] + tcp_y_offset,
                "z": live_pos[2] + approach_offset_xz[1],
            }
            hover_pos = {
                "x": live_pos[0] + approach_offset_xz[0],
                "y": live_pos[1] + PRE_GRASP_HOVER_OFFSET,
                "z": live_pos[2] + approach_offset_xz[1],
            }
            current_position = corrected

            # Step A: move to pre-grasp hover above the new object position.
            # No orientation constraint here — constraining at hover shrinks the IK
            # solution space and causes OMPL to fail at borderline reach distances
            # (same reasoning as _grasp_via_ros_planned for non-top approaches).
            # Orientation is enforced at descent (Step B) where it matters.
            logger.info(
                f"[follow_target] {robot_id}: moving to hover above corrected position"
            )
            hover_result = bridge.plan_and_execute(
                position=hover_pos,
                orientation=None,
                planning_time=5.0,
                robot_id=robot_id,
                max_velocity_scaling=PREGRASP_VELOCITY_SCALING,
                max_acceleration_scaling=PREGRASP_ACCELERATION_SCALING,
            )
            if not hover_result or not hover_result.get("success"):
                hover_err = hover_result.get("error") if hover_result else "no response"
                logger.warning(
                    f"[follow_target] {robot_id}: hover move failed — {hover_err}"
                )
                return False, f"hover move failed: {hover_err}"

            time.sleep(0.1)

            if _is_aborted():
                logger.info(f"[follow_target] {robot_id}: sequence aborted after hover — stopping")
                return False, "sequence aborted"

            # Step B: Move to corrected grasp position.
            # Uses free-space planner (OMPL) rather than Cartesian descent — the retract
            # in Step A already lifted the arm clear of the table, so dragging is not a risk.
            # OMPL is more robust than Cartesian descent at workspace-edge configurations.
            logger.info(
                f"[follow_target] {robot_id}: moving to corrected grasp position"
            )
            correction_result = bridge.plan_and_execute(
                position=corrected,
                orientation=orientation,
                planning_time=8.0,
                robot_id=robot_id,
                max_velocity_scaling=GRASP_DESCENT_VELOCITY_SCALING,
                max_acceleration_scaling=GRASP_DESCENT_ACCELERATION_SCALING,
            )
            if not correction_result or not correction_result.get("success"):
                corr_err = correction_result.get("error") if correction_result else "no response"
                logger.warning(
                    f"[follow_target] {robot_id}: corrective move failed — {corr_err}"
                    " — retrying without orientation constraint"
                )
                correction_result = bridge.plan_and_execute(
                    position=corrected,
                    orientation=None,
                    planning_time=8.0,
                    robot_id=robot_id,
                    max_velocity_scaling=GRASP_DESCENT_VELOCITY_SCALING,
                    max_acceleration_scaling=GRASP_DESCENT_ACCELERATION_SCALING,
                )
            if not correction_result or not correction_result.get("success"):
                corr_err = correction_result.get("error") if correction_result else "no response"
                logger.warning(
                    f"[follow_target] {robot_id}: corrective move failed — {corr_err}"
                )
                return False, f"corrective move failed: {corr_err}"
    else:
        if not FOLLOW_TARGET_ENABLED:
            logger.debug(
                f"[follow_target] disabled — closing gripper at planned position"
            )

    # Arm is at (corrected) grasp position. ROS plan_and_execute is synchronous —
    # arm velocity is already ~0 on return. Close immediately.
    logger.info(f"[follow_target] {robot_id}: closing gripper")
    gripper_result = bridge.control_gripper(0.0, robot_id=robot_id)
    if gripper_result and gripper_result.get("success"):
        # Give Unity physics time to register the contact before returning.
        # GripperContactSensor needs 100ms min contact + 167ms force average ≈ 270ms;
        # 0.3s provides 30ms margin above the physical minimum.
        time.sleep(0.3)
        logger.info(f"[follow_target] {robot_id}: gripper closed")
        return True, ""
    else:
        logger.warning(f"[follow_target] {robot_id}: gripper close command failed")
        return False, "gripper close command failed"


def _vec_to_pos(seq, y_offset: float = 0.0) -> dict:
    return {"x": seq[0], "y": seq[1] + y_offset, "z": seq[2]}


def _yaw_toward_object(robot_id: str, object_position) -> float:
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
    try:
        from config.ROS import DEFAULT_CONTROL_MODE

        return DEFAULT_CONTROL_MODE
    except ImportError:
        return "ros"


def _handle_ros_failure(error_msg: str, context: str):
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
                    return (
                        yaw_unity,
                        f"WorldState long-{axis} (obj_yaw={yaw_deg:.1f}°→jaw={math.degrees(yaw_unity):.1f}°)",
                    )
        except Exception as _e:
            logger.warning(f"WorldState rotation lookup failed: {_e}")

    yaw_unity = _yaw_toward_object(robot_id, object_position)
    return yaw_unity, f"approach-perpendicular ({math.degrees(yaw_unity):.1f}°)"
