"""
Status Check Operations for Robot Control
==========================================

This module implements simple status checking operations that query
robot state without causing any movement or changes.
"""

import time

# Lazy import to avoid circular dependency with servers module
from .Base import (
    BasicOperation,
    OperationCategory,
    OperationComplexity,
    OperationParameter,
    OperationRelationship,
    OperationResult,
)

# Configure logging
from core.LoggingSetup import get_logger
logger = get_logger(__name__)


# Import from centralized lazy import system (prevents circular dependencies)
try:
    from ..core.Imports import get_command_broadcaster as _get_command_broadcaster
except ImportError:
    from core.Imports import get_command_broadcaster as _get_command_broadcaster


# ============================================================================
# Implementation: Check Robot Status
# ============================================================================


def check_robot_status(
    robot_id: str, detailed: bool = False, request_id: int = 0
) -> OperationResult:
    """
    Query the current status of a robot.

    This operation sends a status check request to Unity to retrieve
    the current state of the robot including position, joint angles,
    and operational status. This is a read-only operation that does
    not cause any movement or state changes.

    Args:
        robot_id: ID of the robot to check (e.g., "Robot1", "AR4_Robot")
        detailed: If True, return detailed joint information. If False, return basic status only.

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
            "detailed": bool,
            "status": "query_sent",
            "timestamp": float
        }

        Error structure:
        {
            "code": str,                    # Error code (e.g., "INVALID_ROBOT_ID")
            "message": str,                 # Human-readable error message
            "recovery_suggestions": list    # List of suggested actions
        }

    Example:
        >>> # Check basic robot status
        >>> result = check_robot_status("Robot1")
        >>> if result["success"]:
        ...     print("Status check sent")

        >>> # Get detailed status with joint information
        >>> result = check_robot_status("Robot1", detailed=True)

    Note:
        This operation is asynchronous. It sends the query to Unity and returns
        immediately. Unity will respond with robot status via the same TCP connection.
        Listen for status response events in Unity's LLMResultsReceiver.
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

        # Validate detailed parameter
        if not isinstance(detailed, bool):
            return OperationResult.error_result(
                "INVALID_DETAILED_PARAMETER",
                f"detailed must be a boolean, got: {type(detailed).__name__}",
                [
                    "Use detailed=True for full status or detailed=False for basic status",
                ],
            )

        # Construct status query command
        command = {
            "command_type": "check_robot_status",
            "robot_id": robot_id,
            "parameters": {
                "detailed": detailed,
            },
            "timestamp": time.time(),
            "request_id": request_id,
        }

        # Send to Unity via CommandBroadcaster
        logger.info(f"Sending status check to {robot_id} (detailed={detailed})")

        success = _get_command_broadcaster().send_command(command, request_id)

        if not success:
            return OperationResult.error_result(
                "COMMUNICATION_FAILED",
                "Failed to send command to Unity - no clients connected",
                [
                    "Ensure Unity is running with UnifiedPythonReceiver active",
                    "Verify CommandServer is running (port 5010)",
                    "Check Unity console for connection errors",
                    "Restart backend: python -m orchestrators.RunRobotController",
                ],
            )

        # Return success
        logger.info(f"Successfully sent status check to {robot_id}")

        return OperationResult.success_result(
            {
                "robot_id": robot_id,
                "detailed": detailed,
                "status": "query_sent",
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.error(f"Unexpected error in check_robot_status: {e}", exc_info=True)
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


def create_check_robot_status_operation() -> BasicOperation:
    """
    Create the BasicOperation definition for check_robot_status.

    This provides rich metadata for RAG retrieval and LLM task planning.
    """
    return BasicOperation(
        # Identity
        operation_id="status_check_robot_001",
        name="check_robot_status",
        category=OperationCategory.STATE_CHECK,
        complexity=OperationComplexity.ATOMIC,
        # Descriptions for RAG
        description="Query the current status and state of a robot without causing any movement",
        long_description="""
            This operation sends a status check request to Unity to retrieve
            the current state of the robot including position, joint angles,
            and operational status. This is a read-only operation that does
            not cause any movement or state changes.

            The operation can return either basic status (position, active state)
            or detailed status (including all joint angles, velocities, and targets).

            This is useful for:
            - Verifying robot is ready before starting a task
            - Checking current position before planning movement
            - Debugging robot behavior
            - Monitoring robot health during operation
        """,
        usage_examples=[
            "Before moving robot: check_robot_status('Robot1') to ensure it's ready",
            "check_robot_status('Robot1', detailed=True) to get full joint information",
            "Verify robot reached target by checking status after movement",
            "Monitor robot state during multi-step task execution",
        ],
        # Parameters
        parameters=[
            OperationParameter(
                name="robot_id",
                type="str",
                description="ID of the robot to check (e.g., 'Robot1', 'AR4_Robot')",
                required=True,
            ),
            OperationParameter(
                name="detailed",
                type="bool",
                description="If True, return detailed joint information. If False, basic status only.",
                required=False,
                default=False,
            ),
        ],
        # Conditions
        preconditions=[
            "robot_is_initialized(robot_id)",
        ],
        postconditions=[
        ],
        # Performance
        average_duration_ms=50.0,  # Very fast - just a query
        success_rate=0.99,  # Very reliable
        failure_modes=[
            "Robot ID not found in RobotManager",
            "Communication failed - Unity not connected to CommandServer",
            "Invalid parameter type for 'detailed'",
        ],
        # Relationships
        relationships=OperationRelationship(
            operation_id="status_check_robot_001",
            required_operations=[],
            commonly_paired_with=["motion_move_to_coord_001"],
            pairing_reasons={
                "motion_move_to_coord_001": "Verify robot is ready and not in error state before commanding movement",
            },
            typical_before=[
                "motion_move_to_coord_001",
                "manipulation_control_gripper_001",
                "manipulation_grasp_object_001",
            ],
        ),
        # Implementation
        implementation=check_robot_status,
    )


# Create the operation instance for export
CHECK_ROBOT_STATUS_OPERATION = create_check_robot_status_operation()
