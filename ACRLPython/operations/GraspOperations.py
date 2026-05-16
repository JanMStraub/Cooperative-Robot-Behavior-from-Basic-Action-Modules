#!/usr/bin/env python3
"""MoveIt2-inspired grasp operations: candidate generation, IK validation, collision checking, scoring."""

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
) -> bool:
    """Move to planned position, optionally correct for object drift, then close gripper. Returns True if gripper closed."""
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

            corrected = _vec_to_pos(live_pos, tcp_y_offset)
            hover_pos = _vec_to_pos(live_pos, PRE_GRASP_HOVER_OFFSET)
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
                logger.warning(
                    f"[follow_target] {robot_id}: hover move failed — "
                    f"{hover_result.get('error') if hover_result else 'no response'}"
                )
                return False  # arm at retract height — closing gripper here would miss the object
            time.sleep(0.1)

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
                logger.warning(
                    f"[follow_target] {robot_id}: corrective move failed — "
                    f"{correction_result.get('error') if correction_result else 'no response'}"
                )
                return False  # arm at hover height — closing gripper here would miss the object
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
        return True
    else:
        logger.warning(f"[follow_target] {robot_id}: gripper close command failed")
        return False


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
    """Full GraspPlanner pipeline (ROS path). Returns (result, should_fallback)."""
    try:
        from grasp_planning.GraspPlanner import GraspPlanner
    except ImportError as e:
        logger.warning(f"Grasp planning not available ({e}), using position-only")
        return None, True  # signal caller to try position-only path

    # Priority: yaw override > WorldState rotation > robot-to-object axis. GraspPlanner expects Unity quaternion.
    obj_rot_quat = (0.0, 0.0, 0.0, 1.0)
    if grasp_yaw_override is not None:
        yaw_unity = grasp_yaw_override
    else:
        yaw_unity, yaw_source = _yaw_from_world_state_or_robot(
            robot_id, object_id, object_position, world_state
        )
        logger.info(f"[ROS planned] gripper yaw from {yaw_source}")

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

    # GRASP_TCP_OFFSET lifts ee_link above contact surface — without it, fingers penetrate table.
    grasp_pos = _vec_to_pos(best_grasp.grasp_position, GRASP_TCP_OFFSET)

    # Caller's pre_grasp_distance overrides GraspCandidateGenerator's min (5cm too small for 2cm cubes).
    if pre_grasp_distance > 0.0 and best_grasp.approach_direction is not None:
        pre_grasp_pt = (
            np.array(best_grasp.grasp_position)
            + np.array(best_grasp.approach_direction) * pre_grasp_distance
        )
        pre_grasp_pos = _vec_to_pos(tuple(pre_grasp_pt))
    else:
        pre_grasp_pos = _vec_to_pos(best_grasp.pre_grasp_position)

    # Step 1: Clearance waypoint — always approach pre-grasp from above for reproducible joint config.
    _pre_grasp_orientation = grasp_orientation if preferred_approach == "top" else None
    if pre_grasp_pos["y"] < PRE_GRASP_CLEARANCE_Y:
        clearance_pos = {
            "x": pre_grasp_pos["x"],
            "y": PRE_GRASP_CLEARANCE_Y,
            "z": pre_grasp_pos["z"],
        }
        logger.info(f"[ROS planned] Clearance waypoint for {robot_id}: {clearance_pos}")
        clearance_result = bridge.plan_and_execute(
            position=clearance_pos,
            orientation=None,  # no orientation needed at clearance height — constraint causes slow planning
            planning_time=3.0,
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
                f"[ROS planned] Clearance waypoint failed ({cl_err}) — proceeding to pre-grasp"
            )
        else:
            time.sleep(0.2)

    # Step 2: Pre-grasp hover — no orientation for side/front (shrinks IK space at borderline reach).
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

    # Settle pause: ROSJointStatePublisher 50Hz → 0.1s = 5 ticks before MoveIt samples start state.
    time.sleep(0.1)

    # Step 3: Cartesian descent — constrains ee_link to straight-line so wrist can't flip to alternate IK.
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

    # Follow-target drift correction + gripper close
    logger.info(f"Arm at grasp position, starting follow-target for {robot_id}")
    gripper_ok = _execute_grasp_with_follow_target(
        bridge=bridge,
        robot_id=robot_id,
        object_id=object_id,
        planned_position=grasp_pos,
        orientation=grasp_orientation,
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
    """Position-only ROS grasp fallback when GraspPlanner unavailable. Returns (result, should_fallback)."""
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
        logger.info(
            f"[ROS pos-only] Clearance waypoint for {robot_id}: {clearance_pos}"
        )
        clearance_result = bridge.plan_and_execute(
            position=clearance_pos,
            orientation=None,  # no orientation needed at clearance height — constraint causes slow planning
            planning_time=3.0,
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
    # constrain_joint4=True: ensures joint_4 arrives in the short-arc configuration so
    # the subsequent Cartesian descent's ±90° joint_4 window includes valid grasp configs.
    logger.info(f"Moving to pre-grasp position for {robot_id}")
    pre_result = bridge.plan_and_execute(
        position=pre_grasp_position,
        orientation=top_down_orientation,
        planning_time=10.0,
        robot_id=robot_id,
        max_velocity_scaling=PREGRASP_VELOCITY_SCALING,
        max_acceleration_scaling=PREGRASP_ACCELERATION_SCALING,
        constrain_joint4=True,
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

    # Settle pause: 50Hz publisher → 0.1s = 5 ticks.
    time.sleep(0.1)

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


# Shared module import avoids circular dependency with VGNClient
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
    """VGN TCP path: point cloud → YOLO bbox → VGNClient → Unity precomputed_candidates. None if unavailable."""
    import numpy as np

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

    hover = pre_grasp_distance if pre_grasp_distance > 0 else PRE_GRASP_HOVER_OFFSET
    candidates = []
    for g in world_grasps:
        pos = g["position"]
        rot = g["rotation"]
        approach = g["approach_direction"]

        # approach_direction points toward object (VGN convention) → subtract to place hover behind grasp.
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
    """VGN 6-DOF poses + MoveIt (highest-priority path). None if unavailable; error result if arm descended but gripper failed."""
    import numpy as np

    try:
        from config.Servers import VGN_TOP_K
    except ImportError:
        VGN_TOP_K = 20

    from operations.GraspFrameTransform import transform_grasp_poses_to_unity
    from operations.PointCloudOperations import generate_point_cloud
    from operations.VGNClient import VGNClient

    client = VGNClient()
    if not client.is_available():
        logger.info("[VGN+ROS] Model unavailable — falling back to geometric ROS")
        return None

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

    # SGBM depth unreliable on Unity surfaces (~1.8x overestimate) → use WorldState for position.
    # VGN used only for orientation/approach direction.
    yolo_bbox: tuple = (0, 0, 0, 0)
    image_np: "Optional[np.ndarray]" = None
    det_img_w = img_w
    det_img_h = img_h
    _detection_depth_m: "Optional[float]" = None  # stereo bbox depth for VGN hint

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

    logger.info(
        f"[VGN] Calling predict_grasps: image_width={img_w}, image_height={img_h}, fov={fov}, bbox={yolo_bbox}, points_shape={points_np.shape}"
    )
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

    # Stereo depth scale error ~1.8x on synthetic Unity surfaces → WorldState position is more accurate.
    if _detected_world_pos:
        dp = _detected_world_pos
        logger.info(
            f"[VGN+ROS] Overriding VGN pos {[round(v,3) for v in pos]} with "
            f"DepthEstimator pos {[round(v,3) for v in dp]} for '{object_id}'"
        )
        pos = dp

    # Fail fast before MoveIt if grasp position outside robot reach.
    try:
        from .SpatialPredicates import target_within_reach as _twr

        _reachable, _reach_reason = _twr(robot_id, pos[0], pos[1], pos[2])
        if not _reachable:
            logger.warning(
                f"[VGN+ROS] Grasp position {[round(v,3) for v in pos]} unreachable "
                f"for {robot_id}: {_reach_reason} — falling back to geometric ROS"
            )
            return None
    except Exception:
        pass  # non-fatal: SpatialPredicates unavailable

    hover = pre_grasp_distance if pre_grasp_distance > 0 else PRE_GRASP_HOVER_OFFSET
    _approach_lower = preferred_approach.lower() if preferred_approach else "top"
    _is_top_down_approach = _approach_lower not in ("side", "front")

    # VGN rarely predicts near-vertical grasps for table cubes (typical Y~0.45 → twisted wrist).
    # Use proven top_down_orientation unless VGN approach |Y| >= 0.7 (genuinely top-down).
    _TOP_DOWN_Y_THRESHOLD = 0.7
    _vgn_approach_y = approach[1]  # Unity Y = up
    if abs(_vgn_approach_y) >= _TOP_DOWN_Y_THRESHOLD:
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
        # Side/front: use VGN approach direction. Table top-down fallback: straight up.
        if not _is_top_down_approach:
            pre_approach = approach
        else:
            pre_approach = [0.0, 1.0, 0.0]

        if grasp_yaw_override is not None:
            yaw_unity = grasp_yaw_override
            yaw_source = f"override ({math.degrees(grasp_yaw_override):.1f}°)"
        else:
            yaw_unity, yaw_source = _yaw_from_world_state_or_robot(
                robot_id, object_id, pos, world_state
            )

        # 180° gripper symmetry: grasping at θ and θ+π identical → minimise wrist travel.
        if yaw_unity > math.pi / 2:
            yaw_unity -= math.pi
        elif yaw_unity < -math.pi / 2:
            yaw_unity += math.pi
        # q_final = q_yaw_ros * q_topdown
        # q_topdown ≈ (0.9999, 0, 0, 0.0087): 179° around ROS X = gripper down
        # q_yaw_ros = (0, 0, sin(θ/2), cos(θ/2)): yaw around ROS Z
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
            f"(VGN approach |Y|={abs(_vgn_approach_y):.2f} < {_TOP_DOWN_Y_THRESHOLD}), "
            f"orientation={orientation}"
        )

    pre_grasp_pos = {
        "x": pos[0] + pre_approach[0] * hover,
        "y": pos[1] + pre_approach[1] * hover,
        "z": pos[2] + pre_approach[2] * hover,
    }
    # GRASP_TCP_OFFSET along approach: fingers stop at surface, don't drive through.
    grasp_pos = {
        "x": pos[0] + pre_approach[0] * GRASP_TCP_OFFSET,
        "y": pos[1] + pre_approach[1] * GRASP_TCP_OFFSET,
        "z": pos[2] + pre_approach[2] * GRASP_TCP_OFFSET,
    }

    # Clearance waypoint: top-down only — side/front don't sweep through table-height space.
    clearance_pos = {"x": pos[0], "y": PRE_GRASP_CLEARANCE_Y, "z": pos[2]}
    _is_top_down_approach = _approach_lower not in ("side", "front")
    if _is_top_down_approach and pre_grasp_pos["y"] < PRE_GRASP_CLEARANCE_Y:
        # Pre-grasp is below clearance height — insert the waypoint.
        logger.info(f"[VGN+ROS] Clearance waypoint for {robot_id}: {clearance_pos}")
        clearance_result = bridge.plan_and_execute(
            position=clearance_pos,
            orientation=None,  # no orientation needed at clearance height — constraint causes slow planning
            planning_time=3.0,
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

    # No orientation for side/front pre-grasp — shrinks IK solution space near workspace boundaries.
    _pre_grasp_orientation = orientation if _is_top_down_approach else None
    logger.info(f"[VGN+ROS] Moving to pre-grasp for {robot_id}: {pre_grasp_pos}")
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
        # Position-only: constrain_joint6 prevents free-spin; constrain_joint4 prevents long-arc IK.
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

    # Settle pause: 50Hz publisher → 0.15s = 7 ticks.
    time.sleep(0.15)

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

    # Arm has descended — do NOT return None from here; return error result.
    gripper_ok = _execute_grasp_with_follow_target(
        bridge=bridge,
        robot_id=robot_id,
        object_id=object_id,
        planned_position=grasp_pos,
        orientation=orientation,
        tcp_y_offset=GRASP_TCP_OFFSET,
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


# Implementation: Grasp Object Operation


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
    """Plan and execute grasp via VGN+ROS → geometric ROS → TCP fallback chain."""
    try:
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

        # Signal intent for joint-attention and anticipatory refresh
        try:
            from core.Imports import get_world_state

            _intent_ws = get_world_state()
            if _intent_ws is not None:
                _intent_ws.update_robot_state(
                    robot_id, {"moving_toward_object": object_id}
                )
            try:
                from core.Imports import get_perception_refresh_daemon

                _daemon = get_perception_refresh_daemon()
                if _daemon is not None:
                    _daemon.trigger_anticipatory_refresh([object_id])
            except Exception:
                pass
        except Exception:
            pass

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

                    # PATH 1: VGN+MoveIt (highest priority — provides 6-DOF orientation for angled targets)
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

                    # PATH 2: Geometric ROS (GraspPlanner when dimensions + robot pose available)
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

        # VGN TCP path (falls back to geometric on unavailability/failure)
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

        # TCP path: resolve canonical WorldState key (Unity obj.name) so FindObjectFlexible
        # receives "Red Cube" not LLM-generated "redCube".
        try:
            from core.Imports import get_world_state as _get_ws

            _ws = _get_ws()
            if _ws is not None:
                _canonical = _ws.resolve_canonical_id(object_id)
                if _canonical and _canonical != object_id:
                    logger.debug(
                        f"grasp_object: resolved object_id '{object_id}' → '{_canonical}' via WorldState"
                    )
                    object_id = _canonical
        except Exception as _resolve_err:
            logger.debug(f"grasp_object: canonical resolution skipped: {_resolve_err}")

        parameters = {
            "object_id": object_id,
            "use_advanced_planning": use_advanced_planning,
            "preferred_approach": preferred_approach.lower(),
            "pre_grasp_distance": pre_grasp_distance,
            "enable_retreat": enable_retreat,
            "retreat_distance": retreat_distance,
        }

        if custom_approach_vector is not None:
            parameters["custom_approach_vector"] = {
                "x": custom_approach_vector[0],
                "y": custom_approach_vector[1],
                "z": custom_approach_vector[2],
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
                [
                    "Ensure CommandServer is running",
                    "Check server initialization in orchestrator",
                ],
            )

        logger.info(f"Sending grasp_object command: {robot_id} -> {object_id}")
        success = broadcaster.send_command(command, request_id)

        if success:
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
    finally:
        # Clear movement intent regardless of outcome
        try:
            from core.Imports import get_world_state as _get_ws

            _ws = _get_ws()
            if _ws is not None:
                _ws.update_robot_state(robot_id, {"moving_toward_object": None})
        except Exception:
            pass


def _compute_handoff_approach_vector(
    object_position: tuple,
    object_dimensions: tuple,
    receiving_robot_position: tuple,
) -> list:
    import math

    ox = object_position[0]
    oz = object_position[2]
    ow = object_dimensions[0]
    od = object_dimensions[2]
    rx = receiving_robot_position[0]
    rz = receiving_robot_position[2]

    # Use horizontal plane only; Y is vertical — ignoring avoids bias toward tall thin objects.
    dx = rx - ox
    dz = rz - oz

    # Dominant horizontal axis (x-extent vs z-extent).
    if ow >= od:
        # Wider along X: approach from end away from receiving robot.
        sign = -1.0 if dx >= 0 else 1.0
        approach = [sign, 0.0, 0.0]
    else:
        # Longer along Z: the handoff axis is Z.
        sign = -1.0 if dz >= 0 else 1.0
        approach = [0.0, 0.0, sign]

    # Normalise (already unit length for axis-aligned vectors, but be safe).
    mag = math.sqrt(sum(v * v for v in approach))
    if mag < 1e-6:
        logger.warning("Handoff approach vector degenerate, falling back to top-down")
        return [0.0, 1.0, 0.0]

    return [v / mag for v in approach]


def receive_handoff(
    robot_id: str,
    object_id: str,
    source_robot_id: str,
    request_id: int = 0,
    use_ros: Optional[bool] = None,
    release_signal: Optional[str] = None,
) -> OperationResult:
    """Receive side of handoff: compute approach from WorldState, move, close gripper. release_signal prevents dual-gripper physics conflict."""
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

        try:
            from config.Robot import (
                DEFAULT_HANDOFF_OBJECT_DIMENSIONS,
                ROBOT_BASE_POSITIONS,
            )
        except ImportError:
            DEFAULT_HANDOFF_OBJECT_DIMENSIONS = (0.02, 0.02, 0.02)
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

        # Robot1 presents object sideways; Robot2 approaches along X to the near face.
        # object_dimensions are BoxCollider local-frame (size * lossyScale).
        lx = object_dimensions[0]
        approach_sign = 1.0 if receiver_pos[0] > object_position[0] else -1.0
        near_face_x = object_position[0] + approach_sign * lx * 0.5
        ap_x = near_face_x  # TCP stops at near face; fingers extend and wrap
        obj_height = object_dimensions[1] if len(object_dimensions) > 1 else 0.02
        logger.info(
            f"receive_handoff: object_dimensions={object_dimensions}, obj_height={obj_height:.4f}m"
        )
        # Grip above center (40% height, min 4cm) so receiver clears source robot's fingers.
        ap_y = object_position[1] + max(obj_height * 0.4, 0.04)
        ap_z = object_position[2]
        logger.info(
            f"receive_handoff: approach_position=({ap_x:.3f}, {ap_y:.3f}, {ap_z:.3f})"
        )

        # Robot2 base is 180° → yaw=0 local = toward -X (handoff). Mirrors Robot1's lock.
        static_yaw_deg = 0.0
        logger.info(
            "receive_handoff: using robot-local yaw=0° (base rotation handles world facing)"
        )

        from .MoveOperations import move_to_coordinate

        # Two-step approach: pre-waypoint (free-space, no orientation) → Cartesian move (locked).
        # Free-space first so OMPL finds correct joint config; Cartesian prevents joint_6 spin.
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

        # Side-approach: roll=90° around ROS X → fingers horizontal. q=(sin45,0,0,cos45).
        import math as _math

        _half = _math.radians(90.0) / 2.0
        handoff_orientation = {
            "x": _math.sin(_half),
            "y": 0.0,
            "z": 0.0,
            "w": _math.cos(_half),
        }

        if _use_ros_approach:
            import time as _time

            from ros2.ROSBridge import ROSBridge

            bridge = ROSBridge.get_instance()

            # Step A: pre-waypoint 0.10m back, no orientation constraint.
            pre_x = ap_x + approach_sign * 0.10
            logger.info(
                f"receive_handoff: pre-waypoint=({pre_x:.3f}, {ap_y:.3f}, {ap_z:.3f})"
            )
            pre_result = bridge.plan_and_execute(
                position={"x": pre_x, "y": ap_y, "z": ap_z},
                orientation=handoff_orientation,
                robot_id=robot_id,
                max_velocity_scaling=0.5,
                max_acceleration_scaling=0.4,
            )
            if not pre_result or not pre_result.get("success"):
                logger.warning(
                    f"receive_handoff: pre-waypoint failed ({(pre_result or {}).get('error', 'no response')}) — proceeding to final position"
                )

            # Locked-orientation Cartesian needs more start-state margin than free-space.
            _time.sleep(0.2)

            approach_result = bridge.plan_cartesian_move(
                position={"x": ap_x, "y": ap_y, "z": ap_z},
                orientation=handoff_orientation,
                robot_id=robot_id,
                max_velocity_scaling=0.2,
                max_acceleration_scaling=0.15,
                lock_orientation=True,
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

        # GripperContactSensor: 100ms contact + 167ms force avg ≈ 270ms. 0.5s for side-grasp margin.
        import time as _time_grip

        _time_grip.sleep(0.5)

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
    postconditions=[],
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

# Operation Definition for Registry — Standard Grasp

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
