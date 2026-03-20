#!/usr/bin/env python3
"""
Default Position Operations for Robot Control
==============================================

This module implements operations for returning the robot to its initial/default position
by restoring the start joint targets saved when the robot was registered.
"""

import time
from typing import Optional

# Lazy import to avoid circular dependency with servers module
from .Base import (
    BasicOperation,
    OperationCategory,
    OperationComplexity,
    OperationParameter,
    OperationRelationship,
    OperationResult,
)
from .ROSDispatcher import execute_with_ros_fallback

# Configure logging
from core.LoggingSetup import get_logger

logger = get_logger(__name__)


# Import from centralized lazy import system (prevents circular dependencies)
try:
    from ..core.Imports import get_command_broadcaster as _get_command_broadcaster
except ImportError:
    from core.Imports import get_command_broadcaster as _get_command_broadcaster

# ============================================================================
# Implementation: Return to Start Position
# ============================================================================


def return_to_start_position(
    robot_id: str,
    speed: float = 1.0,
    request_id: int = 0,
    use_ros: Optional[bool] = None,
) -> OperationResult:
    """
    Return robot to its initial start position using saved joint targets.

    This operation commands the robot arm to return to the exact joint configuration
    it had when first registered with RobotManager. This ensures the robot returns
    to a known, safe position regardless of current configuration.

    Args:
        robot_id: ID of the robot to move (e.g., "AR4_Robot", "Robot1")
        speed: Speed multiplier (0.1=slow, 1.0=normal, 2.0=fast), range: [0.1, 2.0]
        use_ros: Whether to use ROS for motion planning (None = auto-detect from config)

    Returns:
        Dict with the following structure:
        {
            "success": bool,           # True if command was sent successfully
            "result": dict or None,    # Result data if successful
            "error": dict or None      # Error information if failed
        }

        Success result structure:
        {
            "robot_id": str,
            "speed": float,
            "status": "command_sent",
            "timestamp": float
        }

        Error structure:
        {
            "code": str,                    # Error code (e.g., "INVALID_ROBOT_ID")
            "message": str,                 # Human-readable error message
            "recovery_suggestions": list    # List of suggested actions
        }

    Example:
        >>> # Return robot to start position at normal speed
        >>> result = return_to_start_position("Robot1")
        >>> if result["success"]:
        ...     print(f"Command sent at {result['result']['timestamp']}")

        >>> # Return slowly for safety
        >>> result = return_to_start_position("Robot1", speed=0.3)

    Note:
        This operation uses the joint targets saved when the robot was first
        registered, providing exact joint angle restoration rather than IK-based
        end-effector positioning.
    """
    try:
        # Validate robot_id
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string, got: {robot_id}",
                [
                    "Provide a valid robot ID (e.g., 'Robot1', 'AR4_Robot')",
                    "Check RobotManager in Unity for available robot IDs",
                ],
            )

        # Validate speed
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

            bridge = ROSBridge.get_instance()
            # Read the Unity start joint targets from WorldState so MoveIt plans
            # to the same pose the TCP path uses, not the URDF all-zeros pose.
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
                robot_id=robot_id, target_joint_angles=start_joint_angles
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
            logger.info(
                f"Sending return_to_start_position command to {robot_id}"
            )
            success = _get_command_broadcaster().send_command(command, request_id)
            if not success:
                return OperationResult.error_result(
                    "COMMUNICATION_FAILED",
                    "Failed to send command to Unity - no clients connected",
                    [
                        "Ensure Unity is running with UnifiedPythonReceiver active",
                        "Verify CommandServer is running (port 5010)",
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
            f"Unexpected error occurred: {str(e)}",
            [
                "Check logs for detailed error information",
                "Verify all parameters are correct types",
                "Retry the operation",
                "Report bug if error persists",
            ],
        )


# ============================================================================
# BasicOperation Definition - For RAG System
# ============================================================================


def create_return_to_start_position_operation() -> BasicOperation:
    """
    Create the BasicOperation definition for return_to_start_position.

    This provides rich metadata for RAG retrieval and LLM task planning.
    """
    return BasicOperation(
        operation_id="motion_return_to_start_001",
        name="return_to_start_position",
        category=OperationCategory.NAVIGATION,
        complexity=OperationComplexity.BASIC,
        description="Return the robot to its initial start position using saved joint targets",
        long_description="""
            This operation commands the robot arm to return to the exact joint configuration
            it had when first registered with RobotManager. Unlike move_to_coordinate which
            uses inverse kinematics to reach an end-effector position, this operation directly
            sets joint angles to restore the robot to a known, safe configuration.

            This is useful for:
            - Returning to a safe home position after completing a task
            - Resetting the robot state between operations
            - Clearing joint configurations after failed movements
            - Returning to a known good state for calibration

            The start joint targets are automatically saved when the robot is registered
            in RobotManager during scene initialization.
        """,
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


# Create the operation instance for export
RETURN_TO_START_POSITION_OPERATION = create_return_to_start_position_operation()
