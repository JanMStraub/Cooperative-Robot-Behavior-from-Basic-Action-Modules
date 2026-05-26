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
        # Move 2 cm past the near face so the gripper actually contacts the object.
        # Previously stopping exactly at the face left a gap and the grasp missed.
        ap_x = near_face_x - approach_sign * 0.02
        obj_height = object_dimensions[1] if len(object_dimensions) > 1 else 0.02
        logger.info(
            f"receive_handoff: object_dimensions={object_dimensions}, obj_height={obj_height:.4f}m"
        )
        # Grip below center (40% height, min 4cm) so receiver clears source robot's fingers.
        ap_y = object_position[1] - max(obj_height * 0.4, 0.04)
        ap_z = object_position[2]
        logger.info(
            f"receive_handoff: approach_position=({ap_x:.3f}, {ap_y:.3f}, {ap_z:.3f})"
        )

        # Validate reachability before wasting time on MoveIt planning calls.
        try:
            from ..SpatialPredicates import target_within_reach

            reachable, reach_reason = target_within_reach(robot_id, ap_x, ap_y, ap_z)
            if not reachable:
                return OperationResult.error_result(
                    "APPROACH_UNREACHABLE",
                    f"receive_handoff: approach position ({ap_x:.3f}, {ap_y:.3f}, {ap_z:.3f}) "
                    f"is outside {robot_id}'s reach — {reach_reason}. "
                    "Source robot should present object closer to the shared workspace.",
                    [
                        "Ensure source robot moves object to shared zone before signalling",
                        "Check handoff position is within receiving robot's workspace",
                    ],
                )
        except Exception as _e:
            logger.warning(f"receive_handoff: reach pre-check failed ({_e}), continuing")

        # Robot2 base is 180° → yaw=0 local = toward -X (handoff). Mirrors Robot1's lock.
        static_yaw_deg = 0.0
        logger.info(
            "receive_handoff: using robot-local yaw=0° (base rotation handles world facing)"
        )

        from ..MoveOperations import move_to_coordinate

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

            # Step A: pre-waypoint 0.10m back with orientation constraint.
            # Locking orientation here seeds OMPL with the correct joint config so
            # the subsequent Cartesian slide-in stays on the approach axis.
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

            # Fallback: Cartesian move can fail (0% complete) when MoveIt cannot
            # find a straight-line path to the approach position. Retry with
            # free-space planning which lets OMPL route around the constraint.
            if not approach_success:
                logger.warning(
                    f"receive_handoff: Cartesian approach failed ({approach_error})"
                    " — retrying with free-space planning"
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

        from ..GripperOperations import control_gripper

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
                from ..SyncOperations import EventBus

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
