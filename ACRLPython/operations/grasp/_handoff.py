#!/usr/bin/env python3
"""Handoff operations: receive_handoff and its BasicOperation definition."""

import logging
from typing import Optional

from core.LoggingSetup import setup_logging

from ..Base import (
    BasicOperation,
    OperationCategory,
    OperationComplexity,
    OperationParameter,
    OperationRelationship,
    OperationResult,
)

setup_logging(__name__)
logger = logging.getLogger(__name__)


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

        # Re-detect from the receiver's perspective so WorldState has the object's
        # current position (it may have moved with the source robot since initial detection).
        try:
            from ..VisionOperations import detect_object_stereo as _detect

            _color = object_id.split("_")[0] if "_" in object_id else object_id
            _det_result = _detect(
                color=_color, robot_id=robot_id, request_id=request_id
            )
            if _det_result.success:
                _dp = _det_result.result or {}
                logger.info(
                    f"receive_handoff: re-detected '{object_id}' via {robot_id} at "
                    f"({_dp.get('x', 0.0):.3f}, {_dp.get('y', 0.0):.3f}, {_dp.get('z', 0.0):.3f})"
                )
            else:
                logger.warning(
                    f"receive_handoff: re-detect failed ({_det_result.error}) - "
                    f"falling back to WorldState"
                )
        except Exception as _det_err:
            logger.warning(f"receive_handoff: re-detect skipped ({_det_err})")

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

        # Use live WorldState object position (50 Hz from Unity) for X/Z/Y.
        # EE position is NOT used for X/Z: Robot1's EE is at its grasp point, which
        # may be at an object face rather than the center, causing center_x to be offset
        # by lx/2 and the insertion to overshoot. EE is still read for Y correction only
        # (to detect the "object floats in air but detection returned table-level Y" case).
        source_ee_pos = None
        try:
            source_state = (
                world_state.get_robot_state(source_robot_id) if world_state else None
            )
            if source_state is not None and source_state.position is not None:
                source_ee_pos = source_state.position
                logger.info(
                    f"receive_handoff: source {source_robot_id} EE at "
                    f"({source_ee_pos[0]:.3f}, {source_ee_pos[1]:.3f}, {source_ee_pos[2]:.3f})"
                    " - used for Y correction only"
                )
        except Exception as _src_e:
            logger.warning(f"receive_handoff: could not read source EE - {_src_e}")

        ref_x = object_position[0]
        ref_z = object_position[2]
        ref_y = object_position[1]
        logger.info(
            f"receive_handoff: initial ref from detected position "
            f"({ref_x:.3f}, {ref_y:.3f}, {ref_z:.3f})"
        )

        # Robot1 presents object sideways; Robot2 approaches along X to the near face.
        # object_dimensions are BoxCollider local-frame (size * lossyScale).
        lx = object_dimensions[0]
        approach_sign = 1.0 if receiver_pos[0] > ref_x else -1.0
        near_face_x = ref_x + approach_sign * lx * 0.5
        # ap_x = near_face_x: approach target for reach check, pre-waypoint, and gripper close.
        # With roll=90° (jaws vertical in Y), no X insertion needed - jaws straddle at the face.
        ap_x = near_face_x
        obj_height = object_dimensions[1] if len(object_dimensions) > 1 else 0.02
        logger.info(
            f"receive_handoff: object_dimensions={object_dimensions}, obj_height={obj_height:.4f}m"
        )
        # TCP at object bottom-face Y so the open jaws bracket the object symmetrically.
        # offset = obj_height/2 places the TCP at the bottom face; the upper jaw
        # (at TCP + jaw_half_gap) only needs jaw_half_gap > obj_height/2 to clear the top.
        # The old max(height*0.4, 4cm) formula always clipped to 4cm for sub-10cm objects,
        # putting the upper jaw inside the object for any gripper with < 11cm total opening.
        _Y_FLOOR_CLEARANCE = 0.03
        ap_y = max(ref_y - obj_height * 0.5, _Y_FLOOR_CLEARANCE)
        ap_z = ref_z
        logger.info(
            f"receive_handoff: approach_position=({ap_x:.3f}, {ap_y:.3f}, {ap_z:.3f})"
            f" (floor clearance {_Y_FLOOR_CLEARANCE}m, "
            f"source={'EE' if source_ee_pos is not None else 'detection'})"
        )

        # Validate reachability before wasting time on MoveIt planning calls.
        try:
            from ..SpatialPredicates import target_within_reach

            reachable, reach_reason = target_within_reach(robot_id, ap_x, ap_y, ap_z)
            if not reachable:
                return OperationResult.error_result(
                    "APPROACH_UNREACHABLE",
                    f"receive_handoff: approach position ({ap_x:.3f}, {ap_y:.3f}, {ap_z:.3f}) "
                    f"is outside {robot_id}'s reach - {reach_reason}",
                    [
                        "Ensure source robot moves object to shared zone before signalling",
                        "Check handoff position is within receiving robot's workspace",
                    ],
                )
        except Exception as _e:
            logger.warning(
                f"receive_handoff: reach pre-check failed ({_e}), continuing"
            )

        # Robot2 base is 180° → yaw=0 local = toward -X (handoff). Mirrors Robot1's lock.
        static_yaw_deg = 0.0
        logger.info(
            "receive_handoff: using robot-local yaw=0° (base rotation handles world facing)"
        )

        # Guarantee maximum jaw opening before approach - avoids a partially-closed gripper
        # (from a prior failed attempt) causing the jaw tips to contact the object face.
        from ..GripperOperations import control_gripper as _open_gripper

        _open_gripper(robot_id=robot_id, open_gripper=True, request_id=request_id)

        from ..MoveOperations import move_to_coordinate

        # Two-step approach: pre-waypoint (free-space, orientation + joint_6 pinned) → Cartesian move (locked).
        # Free-space first so OMPL finds a joint config near the current one; Cartesian slides in without joint_6 spin.
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

            # Step A: pre-waypoint 0.10m back with orientation constraint.
            # Locking orientation here seeds OMPL with the correct joint config so
            # the subsequent Cartesian slide-in stays on the approach axis.
            pre_x = ap_x + approach_sign * 0.10
            logger.info(
                f"receive_handoff: pre-waypoint=({pre_x:.3f}, {ap_y:.3f}, {ap_z:.3f})"
            )
            # Pin joint_6 near its current value so RRTConnect cannot satisfy the
            # roll=90° orientation goal by jumping to a far IK branch - that branch
            # flip is what makes the arm swing all the way around on approach even
            # though the target is right in front. ±90° window keeps the wrist close
            # while still giving OMPL room to reach the locked orientation.
            pre_result = bridge.plan_and_execute(
                position={"x": pre_x, "y": ap_y, "z": ap_z},
                orientation=handoff_orientation,
                robot_id=robot_id,
                max_velocity_scaling=0.5,
                max_acceleration_scaling=0.4,
                constrain_joint6=True,
                joint6_window_rad=_math.radians(90.0),
            )
            if not pre_result or not pre_result.get("success"):
                logger.warning(
                    f"receive_handoff: pre-waypoint failed ({(pre_result or {}).get('error', 'no response')}) - trying hover pre-waypoint"
                )
                # Fallback: approach from above (no orientation constraint so OMPL
                # can find a path from any joint config), then descend in free-space.
                hover_y = ap_y + 0.18
                logger.info(
                    f"receive_handoff: hover pre-waypoint=({ap_x:.3f}, {hover_y:.3f}, {ap_z:.3f})"
                )
                hover_result = bridge.plan_and_execute(
                    position={"x": ap_x, "y": hover_y, "z": ap_z},
                    robot_id=robot_id,
                    max_velocity_scaling=0.5,
                    max_acceleration_scaling=0.4,
                )
                if not hover_result or not hover_result.get("success"):
                    logger.warning(
                        f"receive_handoff: hover pre-waypoint also failed ({(hover_result or {}).get('error', 'no response')}) - proceeding to final position"
                    )

            # Re-read object position and source EE after pre-waypoint completes.
            # By now (~8s elapsed) Robot1 has settled at its handoff position.
            # Re-read WorldState at approach time (Robot1 has settled).
            # If EE is >10cm above detected Y the object has been lifted -
            # WorldState Y is stale. Use EE-derived Y instead.
            try:
                _src2 = (
                    world_state.get_robot_state(source_robot_id)
                    if world_state
                    else None
                )
                _obj2 = (
                    world_state.get_object_position(object_id) if world_state else None
                )
                if _obj2 is not None:
                    _nr_x = _obj2[0]
                    _nr_z = _obj2[2]
                    _det_y2 = _obj2[1]
                else:
                    _nr_x = object_position[0]
                    _nr_z = object_position[2]
                    _det_y2 = object_position[1]
                if _src2 is not None and _src2.position is not None:
                    _ee2 = _src2.position
                    if _ee2[1] - _det_y2 > 0.10:
                        _nr_y = _ee2[1] - obj_height / 2
                        logger.info(
                            f"receive_handoff: detected Y={_det_y2:.3f}m is "
                            f"{_ee2[1] - _det_y2:.3f}m below EE - "
                            f"using EE-derived Y={_nr_y:.3f}m"
                        )
                    else:
                        _nr_y = _det_y2
                else:
                    _nr_y = _det_y2
                ap_x = _nr_x + approach_sign * lx * 0.5
                ap_y = max(_nr_y - obj_height * 0.5, _Y_FLOOR_CLEARANCE)
                ap_z = _nr_z
                logger.info(
                    f"receive_handoff: refreshed approach from settled WorldState "
                    f"obj=({_nr_x:.3f}, {_det_y2:.3f}, {_nr_z:.3f}) → "
                    f"approach=({ap_x:.3f}, {ap_y:.3f}, {ap_z:.3f})"
                )
            except Exception as _ee2_err:
                logger.warning(
                    f"receive_handoff: position re-read failed ({_ee2_err}) - using initial coords"
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

            # Fallback: Cartesian move can fail (0% complete) when MoveIt cannot
            # find a straight-line path to the approach position. Retry with
            # free-space planning which lets OMPL route around the constraint.
            if not approach_success:
                logger.warning(
                    f"receive_handoff: Cartesian approach failed ({approach_error})"
                    " - retrying with free-space planning"
                )
                approach_result = bridge.plan_and_execute(
                    position={"x": ap_x, "y": ap_y, "z": ap_z},
                    orientation=handoff_orientation,
                    robot_id=robot_id,
                    max_velocity_scaling=0.3,
                    max_acceleration_scaling=0.25,
                )
                approach_success = approach_result and approach_result.get("success")
                approach_error = (approach_result or {}).get(
                    "error", "No response from ROS bridge"
                )

            # Approach stops at near_face_x. With roll=90° (jaws vertical in Y),
            # the jaws straddle the object in Y - no X insertion needed.
            # Inserting to center_x pushed the object away.

        else:
            # Disable ProximityGuard so the receiver can approach past the EE_STOP_THRESHOLD
            # (0.25m). Without this, the guard freezes both robots before the gripper reaches
            # the object. Re-enabled after gripper close regardless of outcome.
            _broadcaster = None
            try:
                from .._imports import get_command_broadcaster

                _broadcaster = get_command_broadcaster()
                _broadcaster.send_command(
                    {
                        "command_type": "set_proximity_guard",
                        "parameters": {"enabled": False},
                    },
                    request_id,
                )
                logger.info("receive_handoff: ProximityGuard disabled for approach")
            except Exception as _pg_err:
                logger.warning(
                    f"receive_handoff: could not disable ProximityGuard - {_pg_err}"
                )

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

            if approach_success:
                # move_to_coordinate (TCP) is fire-and-forget - Robot2 hasn't reached
                # the position yet when it returns. Poll WorldState until the arm
                # actually stops before issuing the gripper close. Without this wait,
                # control_gripper races the movement and creates a FixedJoint at the
                # wrong position; the IK solver then fights the misplaced joint → flapping.
                try:
                    from ..MoveOperations import _tcp_wait_for_not_moving

                    _tcp_wait_for_not_moving(robot_id, timeout=20.0)
                except Exception as _wait_err:
                    logger.warning(
                        f"receive_handoff: move-completion poll failed ({_wait_err})"
                        " - falling back to 4s fixed delay"
                    )
                    import time as _t_move

                    _t_move.sleep(4.0)

        if not approach_success:
            return OperationResult.error_result(
                "MOVE_FAILED",
                f"receive_handoff: approach failed - {approach_error}",
                [
                    "Check for workspace collision",
                    "Verify approach_position is reachable",
                ],
            )

        from ..GripperOperations import control_gripper

        gripper_result = control_gripper(
            robot_id=robot_id,
            open_gripper=False,
            request_id=request_id,
        )
        if not gripper_result.success:
            return OperationResult.error_result(
                "GRIPPER_FAILED",
                f"receive_handoff: gripper close failed - {gripper_result.error}",
                ["Check gripper state", "Verify object is within gripper reach"],
            )

        # Destroy source robot's FixedJoint immediately after closing ours.
        # The dual-FixedJoint window is now only the round-trip of these two commands
        # (~ms) rather than the seconds it takes for the group-9 release_object to run.
        # Without this, two conflicting rigid constraints on the same body produce an
        # impulse that snaps the receiving robot's arm back uncontrollably.
        import time as _t_rel

        from ..GripperOperations import control_gripper as _release_source_gripper

        try:
            _rel = _release_source_gripper(
                robot_id=source_robot_id,
                open_gripper=True,
                request_id=request_id,
            )
            if _rel.success:
                logger.info(
                    f"receive_handoff: source {source_robot_id} released - "
                    "dual-FixedJoint window closed"
                )
            else:
                logger.warning(f"receive_handoff: source release failed ({_rel.error})")
        except Exception as _rel_err:
            logger.warning(
                f"receive_handoff: could not release source gripper - {_rel_err}"
            )

        if release_signal:
            try:
                from ..SyncOperations import EventBus

                EventBus().signal(release_signal)
                logger.info(
                    f"receive_handoff: emitted release signal '{release_signal}'"
                )
            except Exception as e:
                logger.warning(f"receive_handoff: failed to emit release signal - {e}")

        # Re-enable ProximityGuard now that the gripper has closed and transfer is done.
        # Only needed for the TCP path (ROS path never disabled it).
        if not _use_ros_approach:
            try:
                from .._imports import get_command_broadcaster as _gcb

                _gcb().send_command(
                    {
                        "command_type": "set_proximity_guard",
                        "parameters": {"enabled": True},
                    },
                    request_id,
                )
                logger.info("receive_handoff: ProximityGuard re-enabled")
            except Exception as _pg_re_err:
                logger.warning(
                    f"receive_handoff: could not re-enable ProximityGuard - {_pg_re_err}"
                )

        # GripperContactSensor: 100ms contact + 167ms force avg ≈ 270ms. 0.5s for side-grasp margin.
        import time as _time_grip

        _time_grip.sleep(0.5)

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
            str(e),
            ["Check stack trace in logs", "Verify WorldState is populated"],
        )


RECEIVE_HANDOFF_OPERATION = BasicOperation(
    operation_id="coordination_receive_handoff_001",
    name="receive_handoff",
    category=OperationCategory.COORDINATION,
    complexity=OperationComplexity.COMPLEX,
    description=(
        "Full receive side of a handoff: approach position computed from WorldState, open gripper, move, close. "
        "Trigger phrases: 'take the object from Robot1', 'accept the handoff', 'receive what Robot1 is offering', "
        "'grab it from Robot1', 'collect from partner'. "
        "Call after source robot signals readiness and after detect_object_stereo populates WorldState."
    ),
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
