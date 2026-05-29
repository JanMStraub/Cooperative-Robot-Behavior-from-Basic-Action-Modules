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

from ._imports import get_command_broadcaster as _get_command_broadcaster
from .Base import (
    BasicOperation,
    OperationCategory,
    OperationComplexity,
    OperationParameter,
    OperationResult,
    OperationRelationship,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# synchronized_grasp
# ---------------------------------------------------------------------------


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
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                "Robot ID must be a non-empty string",
                ["Provide a valid robot ID"],
            )

        if not partner_robot_id or not isinstance(partner_robot_id, str):
            return OperationResult.error_result(
                "INVALID_PARTNER_ROBOT_ID",
                "Partner robot ID must be a non-empty string",
                ["Provide a valid partner robot ID"],
            )

        if robot_id == partner_robot_id:
            return OperationResult.error_result(
                "INVALID_PARTNER_ROBOT_ID",
                f"Partner robot ID must differ from robot_id, got: '{partner_robot_id}'",
                ["Provide a different robot ID for the partner"],
            )

        if not object_id or not isinstance(object_id, str):
            return OperationResult.error_result(
                "INVALID_OBJECT_ID",
                "Object ID must be a non-empty string",
                ["Provide a valid object ID"],
            )

        if approach_axis not in ("x", "z"):
            return OperationResult.error_result(
                "INVALID_APPROACH_AXIS",
                f"Approach axis must be 'x' or 'z', got: '{approach_axis}'",
                ["Use 'x' for left/right approach or 'z' for front/back approach"],
            )

        if not (3000 <= timeout_ms <= 60000):
            return OperationResult.error_result(
                "INVALID_TIMEOUT",
                f"Timeout must be in range [3000, 60000]ms, got: {timeout_ms}",
                ["Use a timeout between 3000ms (3s) and 60000ms (60s)"],
            )

        _use_ros = use_ros
        if _use_ros is None:
            try:
                from config.ROS import ROS_ENABLED, DEFAULT_CONTROL_MODE

                _use_ros = ROS_ENABLED and DEFAULT_CONTROL_MODE in ("ros", "hybrid")
            except ImportError:
                _use_ros = False

        if _use_ros:
            logger.info(
                "Bimanual operation not yet supported via ROS — using Unity direct control"
            )
            _use_ros = False

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

        logger.info(
            f"Successfully initiated synchronized_grasp for {robot_id} + {partner_robot_id}"
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

    except Exception as e:
        logger.error(f"Unexpected error in synchronized_grasp: {e}", exc_info=True)
        return OperationResult.error_result(
            "UNEXPECTED_ERROR",
            f"Unexpected error occurred: {str(e)}",
            ["Check logs"],
        )


def create_synchronized_grasp_operation() -> BasicOperation:
    return BasicOperation(
        operation_id="collaborative_synchronized_grasp_001",
        name="synchronized_grasp",
        category=OperationCategory.MANIPULATION,
        complexity=OperationComplexity.COMPLEX,
        description="Both robots simultaneously approach and grasp the same large object from opposite sides (bimanual grasping)",
        long_description="""
            Commands both robots to coordinate a simultaneous approach and grasp of a single
            large object from opposite sides along the specified axis.

            Requires:
            - Both robots to be idle and positioned within reach of the object
            - Coordination timing so neither robot collides with the other during approach
            - Gripper contact confirmation from both robots before declaring success

            Suitable for objects too large or unstable for a single-arm grasp.
        """,
        usage_examples=[
            "synchronized_grasp('Robot1', 'Robot2', 'LargeBox')",
            "synchronized_grasp('Robot1', 'Robot2', 'LargeBox', approach_axis='z')",
            "Bimanual grasp: both robots approach from opposite X sides",
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
                description="Axis along which robots approach from opposite sides ('x' or 'z')",
                required=False,
                default="x",
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
        preconditions=[
            "robot_is_initialized(robot_id)",
            "robot_is_initialized(partner_robot_id)",
            "robot_is_idle(robot_id)",
            "robot_is_idle(partner_robot_id)",
        ],
        postconditions=[
            "both_robots_grasping(object_id)",
        ],
        average_duration_ms=8000.0,
        success_rate=78.0,
        failure_modes=[
            "One robot fails to reach approach position",
            "Gripper contact not confirmed",
            "Object moves during approach",
            "Timeout waiting for both robots",
        ],
        relationships=OperationRelationship(
            operation_id="collaborative_synchronized_grasp_001",
            required_operations=["coordination_check_partner_001"],
            required_reasons={
                "coordination_check_partner_001": "Verify partner is idle before starting synchronized bimanual grasp"
            },
            commonly_paired_with=[
                "collaborative_joint_transport_001",
                "sync_signal_001",
                "coordination_check_partner_001",
            ],
            pairing_reasons={
                "collaborative_joint_transport_001": "After synchronized grasp, transport object together",
                "sync_signal_001": "Signal partners after grasp completes",
                "coordination_check_partner_001": "Check partner availability before bimanual task",
            },
            typical_before=["collaborative_joint_transport_001"],
            typical_after=["coordination_check_partner_001"],
        ),
        implementation=synchronized_grasp,
    )


SYNCHRONIZED_GRASP_OPERATION = create_synchronized_grasp_operation()


# ---------------------------------------------------------------------------
# joint_transport
# ---------------------------------------------------------------------------


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
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                "Robot ID must be a non-empty string",
                ["Provide a valid robot ID"],
            )

        if not partner_robot_id or not isinstance(partner_robot_id, str):
            return OperationResult.error_result(
                "INVALID_PARTNER_ROBOT_ID",
                "Partner robot ID must be a non-empty string",
                ["Provide a valid partner robot ID"],
            )

        if robot_id == partner_robot_id:
            return OperationResult.error_result(
                "INVALID_PARTNER_ROBOT_ID",
                f"Partner robot ID must differ from robot_id, got: '{partner_robot_id}'",
                ["Provide a different robot ID for the partner"],
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
                    [f"Provide a valid finite number for {name}"],
                )

        if not (0.0 <= lift_height <= 0.3):
            return OperationResult.error_result(
                "INVALID_LIFT_HEIGHT",
                f"Lift height must be in range [0.0, 0.3]m, got: {lift_height}",
                ["Use a lift height between 0.0m and 0.3m"],
            )

        if not (5000 <= timeout_ms <= 120000):
            return OperationResult.error_result(
                "INVALID_TIMEOUT",
                f"Timeout must be in range [5000, 120000]ms, got: {timeout_ms}",
                ["Use a timeout between 5000ms (5s) and 120000ms (120s)"],
            )

        # Precondition check: both robots must have closed grippers
        try:
            try:
                from .WorldState import WorldState
            except ImportError:
                from operations.WorldState import WorldState  # type: ignore[no-redef]

            ws = WorldState()
            r1_state = ws.get_robot_state(robot_id)
            r2_state = ws.get_robot_state(partner_robot_id)

            def _gripper(s):
                if s is None:
                    return "unknown"
                return (
                    s.get("gripper_state", "unknown")
                    if isinstance(s, dict)
                    else getattr(s, "gripper_state", "unknown")
                )

            r1_gripper = _gripper(r1_state)
            r2_gripper = _gripper(r2_state)

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

        _use_ros = use_ros
        if _use_ros is None:
            try:
                from config.ROS import ROS_ENABLED, DEFAULT_CONTROL_MODE

                _use_ros = ROS_ENABLED and DEFAULT_CONTROL_MODE in ("ros", "hybrid")
            except ImportError:
                _use_ros = False

        if _use_ros:
            logger.info(
                "Bimanual operation not yet supported via ROS — using Unity direct control"
            )
            _use_ros = False

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

        logger.info(
            f"Successfully initiated joint_transport for {robot_id} + {partner_robot_id}"
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

    except Exception as e:
        logger.error(f"Unexpected error in joint_transport: {e}", exc_info=True)
        return OperationResult.error_result(
            "UNEXPECTED_ERROR",
            f"Unexpected error occurred: {str(e)}",
            ["Check logs"],
        )


def create_joint_transport_operation() -> BasicOperation:
    return BasicOperation(
        operation_id="collaborative_joint_transport_001",
        name="joint_transport",
        category=OperationCategory.MANIPULATION,
        complexity=OperationComplexity.COMPLEX,
        description="Both robots cooperatively transport a jointly-grasped object to a target position (rigid cooperative transport)",
        long_description="""
            Commands both robots to move a jointly-held object in a coordinated fashion
            to the specified target position. Both robots lift the object by lift_height
            before translating to minimise surface friction and collision risk.

            Requires:
            - Both robots already grasping the object (gripper_state == 'closed')
            - Target position reachable by both robots simultaneously
            - Synchronised velocity profiles to avoid tearing the object

            Typical usage follows synchronized_grasp; afterwards both robots release
            via control_gripper / release_object.
        """,
        usage_examples=[
            "joint_transport('Robot1', 'Robot2', 0.0, 0.15, 0.3)",
            "joint_transport('Robot1', 'Robot2', -0.1, 0.2, 0.1, lift_height=0.1)",
            "Cooperative carry: both arms move large tray to target",
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
        preconditions=[
            "robot_is_initialized(robot_id)",
            "robot_is_initialized(partner_robot_id)",
            "gripper_state(robot_id) == 'closed'",
            "gripper_state(partner_robot_id) == 'closed'",
        ],
        postconditions=[
            "object_at_target_position(target_x, target_y, target_z)",
        ],
        average_duration_ms=12000.0,
        success_rate=72.0,
        failure_modes=[
            "One robot loses grasp during transport",
            "Target position unreachable for one robot",
            "Collision during transport",
            "Timeout",
        ],
        relationships=OperationRelationship(
            operation_id="collaborative_joint_transport_001",
            required_operations=["collaborative_synchronized_grasp_001"],
            required_reasons={
                "collaborative_synchronized_grasp_001": "Both robots must be grasping the object before joint transport"
            },
            commonly_paired_with=[
                "collaborative_synchronized_grasp_001",
                "manipulation_release_object_001",
                "sync_signal_001",
            ],
            pairing_reasons={
                "collaborative_synchronized_grasp_001": "Grasp object together before transporting",
                "manipulation_release_object_001": "Release after transport completes",
                "sync_signal_001": "Signal transport complete to coordinate release",
            },
            typical_before=["manipulation_release_object_001"],
            typical_after=["collaborative_synchronized_grasp_001"],
        ),
        implementation=joint_transport,
    )


JOINT_TRANSPORT_OPERATION = create_joint_transport_operation()
