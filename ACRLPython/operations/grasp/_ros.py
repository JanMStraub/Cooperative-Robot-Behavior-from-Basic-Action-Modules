#!/usr/bin/env python3
"""ROS-based grasp paths: planned (GraspPlanner) and position-only fallback."""

import logging
import math
import time
from typing import Optional

from core.LoggingSetup import setup_logging

from ..Base import OperationResult
from ._helpers import (
    _execute_grasp_with_follow_target,
    _handle_ros_failure,
    _vec_to_pos,
    _yaw_from_world_state_or_robot,
)

setup_logging(__name__)
logger = logging.getLogger(__name__)

try:
    from ...config.Robot import (
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
        GRASP_DESCENT_ACCELERATION_SCALING,
        GRASP_DESCENT_VELOCITY_SCALING,
        GRASP_TCP_OFFSET,
        PRE_GRASP_CLEARANCE_Y,
        PRE_GRASP_HOVER_OFFSET,
        PREGRASP_ACCELERATION_SCALING,
        PREGRASP_VELOCITY_SCALING,
    )


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
    import numpy as np

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
    grasp_ok, grasp_fail_reason = _execute_grasp_with_follow_target(
        bridge=bridge,
        robot_id=robot_id,
        object_id=object_id,
        planned_position=grasp_pos,
        orientation=grasp_orientation,
        tcp_y_offset=GRASP_TCP_OFFSET,
        world_state=world_state,
    )
    if not grasp_ok:
        return (
            OperationResult.error_result(
                "GRASP_EXECUTION_FAILED",
                f"Grasp execution failed for {robot_id}: {grasp_fail_reason}",
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
    grasp_ok, grasp_fail_reason = _execute_grasp_with_follow_target(
        bridge=bridge,
        robot_id=robot_id,
        object_id=object_id,
        planned_position=grasp_position,
        orientation=top_down_orientation,
        tcp_y_offset=GRASP_TCP_OFFSET,
        world_state=world_state,
    )
    if not grasp_ok:
        return (
            OperationResult.error_result(
                "GRASP_EXECUTION_FAILED",
                f"Grasp execution failed for {robot_id}: {grasp_fail_reason}",
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
