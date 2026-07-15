#!/usr/bin/env python3
"""Return-to-start-position operation: restores saved start joint targets from RobotManager registration."""

import time
from typing import Optional

from .Base import (
    BasicOperation,
    OperationCategory,
    OperationComplexity,
    OperationParameter,
    OperationRelationship,
    OperationResult,
)
from .ROSDispatcher import execute_with_ros_fallback

from core.LoggingSetup import get_logger

logger = get_logger(__name__)

try:
    from ..core.Imports import get_command_broadcaster as _get_command_broadcaster
except ImportError:
    from core.Imports import get_command_broadcaster as _get_command_broadcaster


def return_to_start_position(
    robot_id: str,
    speed: float = 1.0,
    request_id: int = 0,
    use_ros: Optional[bool] = None,
    allow_parallel_ros: bool = False,
) -> OperationResult:
    """Return robot to saved start joint configuration (exact joint restore, not IK).

    ROS path reads start angles from WorldState so MoveIt targets the same pose as TCP, not URDF all-zeros.
    """
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

        if not (0.1 <= speed <= 2.0):
            return OperationResult.error_result(
                "INVALID_SPEED",
                f"Speed {speed} out of range [0.1, 2.0]",
                [
                    "Use speed between 0.1 (very slow) and 2.0 (fast)",
                    "Typical values: 0.3 (safe), 1.0 (normal), 1.5 (fast)",
                ],
            )

        def _ros_path():
            from ros2.ROSBridge import ROSBridge

            bridge = (
                ROSBridge.get_parallel_instance(robot_id)
                if allow_parallel_ros
                else ROSBridge.get_instance()
            )
            start_joint_angles = None
            try:
                from core.Imports import get_world_state

                ws = get_world_state()
                robot_state = ws.get_robot_state(robot_id) if ws else None
                if robot_state and robot_state.start_joint_angles:
                    start_joint_angles = robot_state.start_joint_angles
            except Exception:
                pass
            result = bridge.plan_return_to_start(
                robot_id=robot_id,
                target_joint_angles=start_joint_angles,
                speed=speed,
                allow_parallel=allow_parallel_ros,
            )
            if result and result.get("success"):
                logger.info(f"ROS return to start completed for {robot_id}")
                return OperationResult.success_result(
                    {
                        "robot_id": robot_id,
                        "speed": speed,
                        "status": "ros_executed",
                        "planning_time": result.get("planning_time", 0),
                        "timestamp": time.time(),
                    }
                )
            return None  # signal failure to ROSDispatcher

        def _tcp_path():
            command = {
                "command_type": "return_to_start_position",
                "robot_id": robot_id,
                "parameters": {
                    "speed_multiplier": speed,
                },
                "timestamp": time.time(),
                "request_id": request_id,
            }
            logger.info(f"Sending return_to_start_position command to {robot_id}")
            success = _get_command_broadcaster().send_command(command, request_id)
            if not success:
                return OperationResult.error_result(
                    "COMMUNICATION_FAILED",
                    "Failed to send command to Unity - no clients connected",
                    [
                        "Ensure Unity is running with UnifiedPythonReceiver active",
                        "Verify CommandServer is running (port 5007)",
                        "Check Unity console for connection errors",
                    ],
                )
            logger.info(
                f"Successfully sent return_to_start_position command to {robot_id}"
            )
            return OperationResult.success_result(
                {
                    "robot_id": robot_id,
                    "speed": speed,
                    "status": "command_sent",
                    "timestamp": time.time(),
                }
            )

        return execute_with_ros_fallback(_ros_path, _tcp_path, use_ros)

    except Exception as e:
        logger.error(
            f"Unexpected error in return_to_start_position: {e}", exc_info=True
        )
        return OperationResult.error_result(
            "UNEXPECTED_ERROR",
            str(e),
            ["Check logs", "Verify parameters", "Retry"],
        )


def create_return_to_start_position_operation() -> BasicOperation:
    return BasicOperation(
        operation_id="motion_return_to_start_001",
        name="return_to_start_position",
        category=OperationCategory.NAVIGATION,
        complexity=OperationComplexity.BASIC,
        description="Return robot to initial start position using saved joint targets",
        usage_examples=[
            "Return to home after picking object: return_to_start_position(robot_id='Robot1')",
            "Slow return for safety: return_to_start_position(robot_id='Robot1', speed=0.3)",
            "Reset robot between tasks",
            "Return to known position after collision detection",
        ],
        parameters=[
            OperationParameter(
                name="robot_id",
                type="str",
                description="ID of the robot to move (e.g., 'Robot1', 'AR4_Robot')",
                required=True,
            ),
            OperationParameter(
                name="speed",
                type="float",
                description="Movement speed multiplier (0.1=slow, 1.0=normal, 2.0=fast)",
                required=False,
                default=1.0,
                valid_range=(0.1, 2.0),
            ),
        ],
        preconditions=[
            "robot_is_initialized(robot_id)",
        ],
        postconditions=[],
        average_duration_ms=1500.0,
        success_rate=0.98,
        failure_modes=[
            "Start joint targets not saved (robot not properly registered)",
            "Joint limits exceeded during movement",
            "Collision detected during return movement",
            "Communication failed - Unity not connected to CommandServer",
            "Robot ID not found in RobotManager",
        ],
        relationships=OperationRelationship(
            operation_id="motion_return_to_start_001",
            required_operations=[],
            commonly_paired_with=[
                "motion_move_to_coord_001",
                "manipulation_control_gripper_001",
            ],
            pairing_reasons={
                "motion_move_to_coord_001": "Return to home after completing task at target position",
                "manipulation_control_gripper_001": "Release object then return to home position",
            },
            typical_after=[
                "manipulation_grasp_object_001",
                "motion_move_to_coord_001",
            ],
        ),
        implementation=return_to_start_position,
    )


RETURN_TO_START_POSITION_OPERATION = create_return_to_start_position_operation()
