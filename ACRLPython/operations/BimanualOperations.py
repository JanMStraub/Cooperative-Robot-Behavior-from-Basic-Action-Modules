#!/usr/bin/env python3
"""
Bimanual Operations (Level 5): synchronized_grasp, joint_transport.

Both robots cooperate on a single large object — simultaneous approach/grasp
and rigid cooperative transport.
"""

import time
import logging
import math
from typing import Optional

from ._imports import (
    get_command_broadcaster as _get_command_broadcaster,
    get_world_state as _get_world_state,
)
from .ROSDispatcher import execute_with_ros_fallback
from .Base import (
    BasicOperation,
    OperationCategory,
    OperationComplexity,
    OperationParameter,
    OperationResult,
    OperationRelationship,
)

logger = logging.getLogger(__name__)


def _require_str(val, code, label):
    if not val or not isinstance(val, str):
        return OperationResult.error_result(code, f"{label} must be a non-empty string")
    return None


def _field(state, key, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def synchronized_grasp(
    robot_id: str,
    partner_robot_id: str,
    object_id: str,
    approach_axis: str = "x",
    timeout_ms: int = 15000,
    request_id: int = 0,
    use_ros: Optional[bool] = None,
) -> OperationResult:
    """Both robots simultaneously approach and grasp the same large object from opposite sides."""
    try:
        err = _require_str(robot_id, "INVALID_ROBOT_ID", "robot_id")
        if err:
            return err
        err = _require_str(
            partner_robot_id, "INVALID_PARTNER_ROBOT_ID", "partner_robot_id"
        )
        if err:
            return err
        if robot_id == partner_robot_id:
            return OperationResult.error_result(
                "INVALID_PARTNER_ROBOT_ID",
                f"partner_robot_id must differ from robot_id, got: '{partner_robot_id}'",
            )
        err = _require_str(object_id, "INVALID_OBJECT_ID", "object_id")
        if err:
            return err

        if approach_axis == "auto":
            approach_axis = "x"
        if approach_axis not in ("x", "z"):
            return OperationResult.error_result(
                "INVALID_APPROACH_AXIS",
                f"approach_axis must be 'x', 'z', or 'auto', got: '{approach_axis}'",
            )

        if not (3000 <= timeout_ms <= 60000):
            return OperationResult.error_result(
                "INVALID_TIMEOUT",
                f"timeout_ms must be in [3000, 60000], got: {timeout_ms}",
            )

        def _ros_path() -> Optional[OperationResult]:
            try:
                from ros2.ROSBridge import ROSBridge
            except ImportError:
                return None

            bridge = ROSBridge.get_instance()
            if not bridge.is_connected and not bridge.connect():
                return None

            ws = _get_world_state()
            obj_pos = ws.get_object_position(object_id) if ws else None
            if not obj_pos:
                logger.debug(
                    "synchronized_grasp ROS: object position unknown, falling back to TCP"
                )
                return None

            ox, oy, oz = obj_pos
            OFFSET = 0.15
            if approach_axis == "x":
                r1_pos = {"x": ox - OFFSET, "y": oy, "z": oz}
                r2_pos = {"x": ox + OFFSET, "y": oy, "z": oz}
            else:
                r1_pos = {"x": ox, "y": oy, "z": oz - OFFSET}
                r2_pos = {"x": ox, "y": oy, "z": oz + OFFSET}

            logger.info(
                f"synchronized_grasp ROS: {robot_id} -> {r1_pos}, {partner_robot_id} -> {r2_pos}"
            )

            r1 = bridge.plan_and_execute(
                position=r1_pos, robot_id=robot_id, coordinate_space="unity_world"
            )
            if not (r1 and r1.get("success")):
                return None

            r2 = bridge.plan_and_execute(
                position=r2_pos,
                robot_id=partner_robot_id,
                coordinate_space="unity_world",
            )
            if not (r2 and r2.get("success")):
                return OperationResult.error_result(
                    "ROS_PARTNER_APPROACH_FAILED",
                    f"ROS approach succeeded for {robot_id} but failed for {partner_robot_id}",
                    ["Check partner robot reachability", "Use TCP mode instead"],
                )

            bridge.control_gripper(0.0, robot_id=robot_id)
            bridge.control_gripper(0.0, robot_id=partner_robot_id)

            return OperationResult.success_result(
                {
                    "robot_id": robot_id,
                    "partner_robot_id": partner_robot_id,
                    "object_id": object_id,
                    "approach_axis": approach_axis,
                    "status": "synchronized_grasp_complete",
                    "mode": "ros",
                    "timestamp": time.time(),
                }
            )

        def _tcp_path() -> OperationResult:
            command = {
                "command_type": "synchronized_grasp",
                "robot_id": robot_id,
                "parameters": {
                    "partner_robot_id": partner_robot_id,
                    "object_id": object_id,
                    "approach_axis": approach_axis,
                    "timeout_ms": timeout_ms,
                },
                "request_id": request_id,
                "timestamp": time.time(),
            }

            logger.info(
                f"Sending synchronized_grasp command: {robot_id} + {partner_robot_id} -> {object_id} "
                f"(axis={approach_axis}, timeout={timeout_ms}ms)"
            )

            success = _get_command_broadcaster().send_command(command, request_id)
            if not success:
                return OperationResult.error_result(
                    "COMMUNICATION_FAILED",
                    "Failed to send command to Unity",
                    ["Ensure Unity is running"],
                )

            return OperationResult.success_result(
                {
                    "robot_id": robot_id,
                    "partner_robot_id": partner_robot_id,
                    "object_id": object_id,
                    "approach_axis": approach_axis,
                    "status": "synchronized_grasp_complete",
                    "timestamp": time.time(),
                }
            )

        return execute_with_ros_fallback(_ros_path, _tcp_path, use_ros)

    except Exception as e:
        logger.error(f"Unexpected error in synchronized_grasp: {e}", exc_info=True)
        return OperationResult.error_result(
            "UNEXPECTED_ERROR", f"synchronized_grasp failed: {e}"
        )


def create_synchronized_grasp_operation() -> BasicOperation:
    return BasicOperation(
        operation_id="collaborative_synchronized_grasp_001",
        name="synchronized_grasp",
        category=OperationCategory.MANIPULATION,
        complexity=OperationComplexity.COMPLEX,
        description="Both robots simultaneously approach and grasp the same large object from opposite sides (bimanual grasping)",
        long_description=(
            "Coordinates both robots to approach and grasp a single large object simultaneously "
            "from opposite sides — one robot takes the left side, the other the right. "
            "Supports ROS (MoveIt) and Unity TCP paths. "
            "Follow with joint_transport to move the object, then release_object on both robots."
        ),
        usage_examples=[
            "synchronized_grasp('Robot1', 'Robot2', 'red_cube')  # both robots grasp red cube from left/right",
            "synchronized_grasp('Robot1', 'Robot2', 'LargeBox', approach_axis='z')  # front/back approach",
            "# Robot1 grasps left side, Robot2 grasps right side of the cube:",
            "synchronized_grasp('Robot1', 'Robot2', '$target.color')",
            "# After this, use joint_transport to lift and move the object together",
        ],
        preconditions=[
            "robot_is_initialized(robot_id)",
            "robot_is_initialized(partner_robot_id)",
            "robot_is_idle(robot_id)",
            "robot_is_idle(partner_robot_id)",
        ],
        postconditions=[
            "both_robots_grasping(object_id)",
        ],
        success_rate=0.78,
        failure_modes=[
            "One robot fails to reach approach position",
            "Gripper contact not confirmed",
            "Object moves during approach",
            "Timeout waiting for both robots",
        ],
        parameters=[
            OperationParameter(
                name="robot_id",
                type="str",
                description="Primary robot ID",
                required=True,
            ),
            OperationParameter(
                name="partner_robot_id",
                type="str",
                description="Partner robot ID (must differ from robot_id)",
                required=True,
            ),
            OperationParameter(
                name="object_id",
                type="str",
                description="Object to grasp together",
                required=True,
            ),
            OperationParameter(
                name="approach_axis",
                type="str",
                description="Axis along which robots approach from opposite sides ('x', 'z', or 'auto'). 'auto' resolves to 'x' (default robot layout).",
                required=False,
                default="auto",
                valid_range=None,
            ),
            OperationParameter(
                name="timeout_ms",
                type="int",
                description="Maximum time to complete the synchronized grasp (ms)",
                required=False,
                default=15000,
                valid_range=(3000, 60000),
            ),
        ],
        average_duration_ms=8000.0,
        relationships=OperationRelationship(
            operation_id="collaborative_synchronized_grasp_001",
            required_operations=["coordination_check_partner_001"],
            commonly_paired_with=[
                "collaborative_joint_transport_001",
                "sync_signal_001",
                "coordination_check_partner_001",
            ],
        ),
        implementation=synchronized_grasp,
    )


SYNCHRONIZED_GRASP_OPERATION = create_synchronized_grasp_operation()


def joint_transport(
    robot_id: str,
    partner_robot_id: str,
    target_x: float,
    target_y: float,
    target_z: float,
    lift_height: float = 0.05,
    timeout_ms: int = 20000,
    request_id: int = 0,
    use_ros: Optional[bool] = None,
) -> OperationResult:
    """Both robots (already grasping the same object) move it together to a target position."""
    try:
        err = _require_str(robot_id, "INVALID_ROBOT_ID", "robot_id")
        if err:
            return err
        err = _require_str(
            partner_robot_id, "INVALID_PARTNER_ROBOT_ID", "partner_robot_id"
        )
        if err:
            return err
        if robot_id == partner_robot_id:
            return OperationResult.error_result(
                "INVALID_PARTNER_ROBOT_ID",
                f"partner_robot_id must differ from robot_id, got: '{partner_robot_id}'",
            )

        for name, val in (
            ("target_x", target_x),
            ("target_y", target_y),
            ("target_z", target_z),
        ):
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                return OperationResult.error_result(
                    f"INVALID_{name.upper()}",
                    f"{name} must be a finite float, got: {val!r}",
                )

        if not (0.0 <= lift_height <= 0.3):
            return OperationResult.error_result(
                "INVALID_LIFT_HEIGHT",
                f"lift_height must be in [0.0, 0.3]m, got: {lift_height}",
            )

        if not (5000 <= timeout_ms <= 120000):
            return OperationResult.error_result(
                "INVALID_TIMEOUT",
                f"timeout_ms must be in [5000, 120000], got: {timeout_ms}",
            )

        try:
            from .WorldState import WorldState

            ws = WorldState()
            r1_state = ws.get_robot_state(robot_id)
            r2_state = ws.get_robot_state(partner_robot_id)

            r1_gripper = (
                _field(r1_state, "gripper_state", "unknown") if r1_state else "unknown"
            )
            r2_gripper = (
                _field(r2_state, "gripper_state", "unknown") if r2_state else "unknown"
            )

            if r1_gripper != "closed" or r2_gripper != "closed":
                return OperationResult.error_result(
                    "PRECONDITION_FAILED",
                    f"Both robots must be grasping the object. {robot_id}: {r1_gripper}, {partner_robot_id}: {r2_gripper}",
                    [
                        "Use synchronized_grasp first to have both robots grasp the object"
                    ],
                )
        except Exception as e:
            logger.debug(f"WorldState precondition check skipped: {e}")

        def _ros_path() -> Optional[OperationResult]:
            try:
                from ros2.ROSBridge import ROSBridge
            except ImportError:
                return None

            bridge = ROSBridge.get_instance()
            if not bridge.is_connected and not bridge.connect():
                return None

            r1_cur = bridge.get_ee_pose(robot_id=robot_id)
            r2_cur = bridge.get_ee_pose(robot_id=partner_robot_id)
            if not (r1_cur and r2_cur):
                return None

            logger.info(
                f"joint_transport ROS: lift {lift_height}m then move to "
                f"({target_x:.3f}, {target_y:.3f}, {target_z:.3f})"
            )

            lift1 = bridge.plan_cartesian_move(
                position={
                    "x": r1_cur["position"]["x"],
                    "y": r1_cur["position"]["y"] + lift_height,
                    "z": r1_cur["position"]["z"],
                },
                robot_id=robot_id,
            )
            if not (lift1 and lift1.get("success")):
                return None

            lift2 = bridge.plan_cartesian_move(
                position={
                    "x": r2_cur["position"]["x"],
                    "y": r2_cur["position"]["y"] + lift_height,
                    "z": r2_cur["position"]["z"],
                },
                robot_id=partner_robot_id,
            )
            if not (lift2 and lift2.get("success")):
                return OperationResult.error_result(
                    "ROS_PARTNER_LIFT_FAILED",
                    f"Lift succeeded for {robot_id} but failed for {partner_robot_id}",
                    [
                        "Check partner robot reachability",
                        "Reduce lift_height",
                        "Use TCP mode",
                    ],
                )

            target = {"x": target_x, "y": target_y + lift_height, "z": target_z}

            mv1 = bridge.plan_and_execute(
                position=target, robot_id=robot_id, coordinate_space="unity_world"
            )
            if not (mv1 and mv1.get("success")):
                return OperationResult.error_result(
                    "ROS_TRANSPORT_R1_FAILED",
                    f"Transport failed for {robot_id} after lift phase",
                    ["Verify target reachability", "Use TCP mode"],
                )

            mv2 = bridge.plan_and_execute(
                position=target,
                robot_id=partner_robot_id,
                coordinate_space="unity_world",
            )
            if not (mv2 and mv2.get("success")):
                return OperationResult.error_result(
                    "ROS_TRANSPORT_R2_FAILED",
                    f"Transport failed for {partner_robot_id} after {robot_id} reached target",
                    ["Verify target reachability", "Use TCP mode"],
                )

            return OperationResult.success_result(
                {
                    "robot_id": robot_id,
                    "partner_robot_id": partner_robot_id,
                    "target_position": {"x": target_x, "y": target_y, "z": target_z},
                    "lift_height": lift_height,
                    "status": "joint_transport_complete",
                    "mode": "ros",
                    "timestamp": time.time(),
                }
            )

        def _tcp_path() -> OperationResult:
            command = {
                "command_type": "joint_transport",
                "robot_id": robot_id,
                "parameters": {
                    "partner_robot_id": partner_robot_id,
                    "target_x": target_x,
                    "target_y": target_y,
                    "target_z": target_z,
                    "lift_height": lift_height,
                    "timeout_ms": timeout_ms,
                },
                "request_id": request_id,
                "timestamp": time.time(),
            }

            logger.info(
                f"Sending joint_transport command: {robot_id} + {partner_robot_id} -> "
                f"({target_x:.3f}, {target_y:.3f}, {target_z:.3f}), lift={lift_height}m, timeout={timeout_ms}ms"
            )

            success = _get_command_broadcaster().send_command(command, request_id)
            if not success:
                return OperationResult.error_result(
                    "COMMUNICATION_FAILED",
                    "Failed to send command to Unity",
                    ["Ensure Unity is running"],
                )

            return OperationResult.success_result(
                {
                    "robot_id": robot_id,
                    "partner_robot_id": partner_robot_id,
                    "target_position": {"x": target_x, "y": target_y, "z": target_z},
                    "lift_height": lift_height,
                    "status": "joint_transport_complete",
                    "timestamp": time.time(),
                }
            )

        return execute_with_ros_fallback(_ros_path, _tcp_path, use_ros)

    except Exception as e:
        logger.error(f"Unexpected error in joint_transport: {e}", exc_info=True)
        return OperationResult.error_result(
            "UNEXPECTED_ERROR", f"joint_transport failed: {e}"
        )


def create_joint_transport_operation() -> BasicOperation:
    return BasicOperation(
        operation_id="collaborative_joint_transport_001",
        name="joint_transport",
        category=OperationCategory.MANIPULATION,
        complexity=OperationComplexity.COMPLEX,
        description="Both robots cooperatively transport a jointly-grasped object to a target position (rigid cooperative transport)",
        long_description=(
            "Moves a jointly-held object to a target position. Both robots must already be "
            "grasping it (synchronized_grasp first). Lifts by lift_height before translating "
            "to reduce surface friction. Supports ROS and TCP paths. "
            "Release with release_object on both robots when done."
        ),
        usage_examples=[
            "# After synchronized_grasp, lift both to y=0.15:",
            "joint_transport('Robot1', 'Robot2', 0.0, 0.15, 0.3)",
            "# Lift higher before moving:",
            "joint_transport('Robot1', 'Robot2', -0.1, 0.2, 0.1, lift_height=0.1)",
            "# Full cooperative sequence: grasp → transport → release",
            "# 1. synchronized_grasp('Robot1', 'Robot2', 'red_cube')",
            "# 2. joint_transport('Robot1', 'Robot2', 0.0, 0.15, 0.0)",
            "# 3. release_object('Robot1') + release_object('Robot2')",
        ],
        preconditions=[
            "robot_is_initialized(robot_id)",
            "robot_is_initialized(partner_robot_id)",
            "gripper_state(robot_id) == 'closed'",
            "gripper_state(partner_robot_id) == 'closed'",
        ],
        postconditions=[
            "object_at_target_position(target_x, target_y, target_z)",
        ],
        success_rate=0.72,
        failure_modes=[
            "One robot loses grasp during transport",
            "Target position unreachable for one robot",
            "Collision during transport",
            "Timeout",
        ],
        parameters=[
            OperationParameter(
                name="robot_id",
                type="str",
                description="Primary robot ID",
                required=True,
            ),
            OperationParameter(
                name="partner_robot_id",
                type="str",
                description="Partner robot ID (must differ from robot_id)",
                required=True,
            ),
            OperationParameter(
                name="target_x",
                type="float",
                description="Target X coordinate (Unity world space, meters)",
                required=True,
            ),
            OperationParameter(
                name="target_y",
                type="float",
                description="Target Y coordinate (Unity world space, meters)",
                required=True,
            ),
            OperationParameter(
                name="target_z",
                type="float",
                description="Target Z coordinate (Unity world space, meters)",
                required=True,
            ),
            OperationParameter(
                name="lift_height",
                type="float",
                description="Height to lift object before translating (meters)",
                required=False,
                default=0.05,
                valid_range=(0.0, 0.3),
            ),
            OperationParameter(
                name="timeout_ms",
                type="int",
                description="Maximum time to complete the transport (ms)",
                required=False,
                default=20000,
                valid_range=(5000, 120000),
            ),
        ],
        average_duration_ms=12000.0,
        relationships=OperationRelationship(
            operation_id="collaborative_joint_transport_001",
            required_operations=["collaborative_synchronized_grasp_001"],
            commonly_paired_with=[
                "collaborative_synchronized_grasp_001",
                "manipulation_release_object_001",
                "sync_signal_001",
            ],
        ),
        implementation=joint_transport,
    )


JOINT_TRANSPORT_OPERATION = create_joint_transport_operation()
