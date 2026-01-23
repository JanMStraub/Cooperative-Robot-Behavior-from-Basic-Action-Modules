"""
Gripper Operations for Gripper Control
==========================================

This module implements open and close operations for the robot gripper
through Unity's GripperController via TCP communication.
"""

import time
import logging
from typing import Any, Dict, Optional
# Lazy import to avoid circular dependency with servers module
from .Base import (
    BasicOperation,
    OperationCategory,
    OperationComplexity,
    OperationParameter,
    OperationResult,
    OperationRelationship,
)

# Configure logging
from core.LoggingSetup import setup_logging
setup_logging(__name__)
logger = logging.getLogger(__name__)


# Import from centralized lazy import system (prevents circular dependencies)
try:
    from ..core.Imports import get_command_broadcaster as _get_command_broadcaster
except ImportError:
    from core.Imports import get_command_broadcaster as _get_command_broadcaster


# ============================================================================
# Implementation: Control Gripper Operation
# ============================================================================


def control_gripper(
    robot_id: str, open_gripper: bool, request_id: int = 0, object_id: Optional[str] = None
) -> OperationResult:
    """
    Open or close the robot gripper.

    This operation commands the robot gripper to open or close completely.
    The robot uses the GripperController component to execute this command.

    Args:
        robot_id: ID of the robot to control (e.g., "Robot1", "AR4_Robot")
        open_gripper: Boolean value - True to open gripper, False to close it
        request_id: Optional request ID for tracking
        object_id: Optional object ID to attach when closing (enables handoff)

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
            "open_gripper": bool,
            "status": str,
            "timestamp": float
        }

        Error structure:
        {
            "code": str,                    # Error code (e.g., "INVALID_PARAMETER")
            "message": str,                 # Human-readable error message
            "recovery_suggestions": list    # List of suggested actions
        }

    Example:
        >>> # Open the gripper completely
        >>> result = control_gripper("Robot1", True)
        >>> if result["success"]:
        ...     print(f"Operation completed: {result['result']}")

        >>> # Close the gripper completely
        >>> result = control_gripper("Robot1", False)
        >>> if result["success"]:
        ...     print(f"Operation completed: {result['result']}")

        >>> # Close gripper and attach specific object (for handoff)
        >>> result = control_gripper("Robot1", False, object_id="RedCube")
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

        # Validate open_gripper parameter
        if not isinstance(open_gripper, bool):
            return OperationResult.error_result(
                "INVALID_OPEN_GRIPPER_PARAMETER",
                f"open_gripper must be a boolean, got: {type(open_gripper).__name__}",
                [
                    "Use open_gripper=True to open the gripper or open_gripper=False to close it",
                ],
            )

        # Construct command for Unity
        params: Dict[str, Any] = {
            "open_gripper": open_gripper,
        }
        # Add object_id if specified (for handoff scenarios)
        if object_id:
            params["object_id"] = object_id

        command = {
            "command_type": "control_gripper",
            "robot_id": robot_id,
            "parameters": params,
            "timestamp": time.time(),
            "request_id": request_id,
        }

        # Send to Unity via CommandBroadcaster
        logger.info(
            f"Sending control_gripper command to {robot_id} (open_gripper={open_gripper})"
        )

        success = _get_command_broadcaster().send_command(command, request_id)

        if not success:
            return OperationResult.error_result(
                "COMMUNICATION_FAILED",
                "Failed to send command to Unity - no clients connected",
                [
                    "Ensure Unity is running with UnifiedPythonReceiver active",
                    "Verify ResultsServer is running (port 5010)",
                    "Check Unity console for connection errors",
                    "Restart ResultsServer or run analyzer",
                ],
            )

        # Return success
        logger.info(f"Successfully sent control_gripper command to {robot_id}")

        return OperationResult.success_result(
            {
                "robot_id": robot_id,
                "open_gripper": open_gripper,
                "status": "command_sent",
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.error(f"Unexpected error in control_gripper: {e}", exc_info=True)
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


def create_control_gripper_operation() -> BasicOperation:
    """
    Create the BasicOperation definition for control_gripper.

    This provides rich metadata for RAG retrieval and LLM task planning.
    """
    return BasicOperation(
        # Identity
        operation_id="manipulation_control_gripper_001",
        name="control_gripper",
        category=OperationCategory.MANIPULATION,
        complexity=OperationComplexity.ATOMIC,
        # Descriptions for RAG
        description="Controls the robot gripper to either open or close it completely.",
        long_description="""
            This operation commands the robot gripper to open or close completely.
            The operation uses the GripperController component to control the gripper movement.

            This operation is useful for grasping and releasing objects. When closing the gripper,
            it will grip any object currently between the gripper jaws. When opening, it will
            release any held object.

            This operation is asynchronous - it sends the command to Unity and returns immediately.
            Unity executes the movement in the background using GripperController.
        """,
        usage_examples=[
            "After navigating to an object: control_gripper(robot_id='Robot1', open_gripper=False) # Close gripper to grasp object",
            "After navigating to a drop-off location: control_gripper(robot_id='Robot1', open_gripper=True) # Open gripper to release object",
            "Open gripper before approaching object to prepare for grasping",
            "Close gripper after positioning at target coordinates to secure object",
            "Handoff: control_gripper(robot_id='Robot2', open_gripper=False, object_id='RedCube') # Close gripper and attach object held by another robot",
        ],
        parameters=[
            OperationParameter(
                name="robot_id",
                type="str",
                description="ID of the robot to control (e.g., 'Robot1', 'AR4_Robot')",
                required=True,
            ),
            OperationParameter(
                name="open_gripper",
                type="bool",
                description="True to open gripper completely, False to close gripper completely",
                required=True,
            ),
            OperationParameter(
                name="object_id",
                type="str",
                description="Optional object ID to attach when closing (for handoff scenarios)",
                required=False,
            ),
        ],
        preconditions=[
            "Robot arm is initialized and calibrated",
            "Robot is not currently executing another gripper command",
            "Robot exists in Unity's RobotManager",
            "GripperController component is attached to robot",
            "Unity is running with UnifiedPythonReceiver active",
            "ResultsServer is running on port 5010",
        ],
        postconditions=[
            "Command has been sent to Unity via TCP",
            "Unity GripperController will open or close the gripper",
            "Gripper will reach target state (open or closed) within 500ms",
            "Ready to execute next operation",
            "If closing: object between jaws will be grasped (if present)",
            "If opening: previously held object will be released (if present)",
        ],
        average_duration_ms=500,  # Time for gripper to fully open or close
        success_rate=0.98,
        failure_modes=[
            "Robot ID not found in RobotManager",
            "Communication failed - Unity not connected to ResultsServer",
            "Invalid parameter type for 'open_gripper'",
            "GripperController component not found on robot",
            "Gripper mechanism jammed or obstructed",
        ],
        relationships=OperationRelationship(
            operation_id="manipulation_control_gripper_001",
            required_operations=["motion_move_to_coord_001"],
            required_reasons={
                "motion_move_to_coord_001": "Must position gripper at target location before grasping or releasing",
            },
            commonly_paired_with=[
                "motion_move_to_coord_001",
                "perception_stereo_detect_001",
                "status_check_robot_001",
            ],
            pairing_reasons={
                "motion_move_to_coord_001": "Position at target before gripper action, sequence: move → grasp or move → release",
                "perception_stereo_detect_001": "Detect object before moving to grasp it",
                "status_check_robot_001": "Verify gripper reached target position before closing to grasp",
            },
            typical_before=[],
            typical_after=["motion_move_to_coord_001"],
        ),
        implementation=control_gripper,
    )


# Create the operation instance for export
CONTROL_GRIPPER_OPERATION = create_control_gripper_operation()


# ============================================================================
# Implementation: Release Object
# ============================================================================


def release_object(
    robot_id: str,
    request_id: int = 0,
) -> OperationResult:
    """
    Open gripper to release held object.

    This is an ATOMIC operation - ONLY opens gripper at current position.
    For positioning before release, chain with move_to_coordinate.

    IMPORTANT: This operation does NOT move the robot. If you need to
    move to a specific position before releasing, you must call
    move_to_coordinate BEFORE calling this operation.

    Args:
        robot_id: ID of the robot to control
        request_id: Optional request ID for tracking

    Returns:
        OperationResult with release confirmation or error details

    Example:
        >>> # Chain operations for positioned release:
        >>> # Step 1: Move to drop-off position
        >>> result = move_to_coordinate("Robot1", x=0.3, y=0.0, z=0.1)
        >>> # Step 2: Release object
        >>> result = release_object("Robot1")

        >>> # Release at current position (no movement)
        >>> result = release_object("Robot1")
    """
    try:
        # Validate robot_id
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string, got: {robot_id}",
                ["Provide a valid robot ID (e.g., 'Robot1', 'AR4_Robot')"],
            )

        # Construct command for Unity - ONLY gripper opening, NO positioning
        params: Dict[str, Any] = {
            "open_gripper": True,  # Release means open
        }

        command = {
            "command_type": "release_object",
            "robot_id": robot_id,
            "parameters": params,
            "timestamp": time.time(),
            "request_id": request_id,
        }

        # Send to Unity
        logger.info(f"Sending release_object command to {robot_id} (atomic - gripper only)")

        success = _get_command_broadcaster().send_command(command, request_id)

        if not success:
            return OperationResult.error_result(
                "COMMUNICATION_FAILED",
                "Failed to send command to Unity - no clients connected",
                [
                    "Ensure Unity is running with UnifiedPythonReceiver active",
                    "Verify CommandServer is running (port 5010)",
                ],
            )

        logger.info(f"Successfully sent release_object command to {robot_id}")

        return OperationResult.success_result(
            {
                "robot_id": robot_id,
                "status": "command_sent",
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.error(f"Unexpected error in release_object: {e}", exc_info=True)
        return OperationResult.error_result(
            "UNEXPECTED_ERROR",
            f"Unexpected error occurred: {str(e)}",
            ["Check logs for detailed error information", "Retry the operation"],
        )


def create_release_object_operation() -> BasicOperation:
    """Create the BasicOperation definition for release_object."""
    return BasicOperation(
        operation_id="manipulation_release_object_002",
        name="release_object",
        category=OperationCategory.MANIPULATION,
        complexity=OperationComplexity.ATOMIC,
        description="Open gripper to release held object (atomic operation)",
        long_description="""
            This ATOMIC operation opens the gripper to release any held object.

            IMPORTANT: This operation does NOT move the robot. It ONLY opens
            the gripper at the current position.

            For positioned release, you must chain operations:
            1. move_to_coordinate(robot_id, target_position)
            2. release_object(robot_id)

            This atomicity is critical for LLM-driven control, as it allows
            the LLM to see and control each step of a complex workflow.
        """,
        usage_examples=[
            "release_object('Robot1') - Release at current position",
            "Chain: move_to_coordinate('Robot1', x=0.3, y=0, z=0.1) → release_object('Robot1')",
            "After positioning: release_object('Robot1')",
        ],
        parameters=[
            OperationParameter(
                name="robot_id",
                type="str",
                description="Robot ID",
                required=True,
            ),
        ],
        preconditions=[
            "Robot gripper is closed (holding object or empty)",
            "Unity connected",
        ],
        postconditions=[
            "Gripper opened",
            "Any held object released from gripper",
            "Ready for next operation",
        ],
        average_duration_ms=500.0,
        success_rate=0.98,
        failure_modes=[
            "Communication failed",
            "Gripper mechanism jammed",
        ],
        relationships=OperationRelationship(
            operation_id="manipulation_release_object_002",
            required_operations=[],
            required_reasons={},
            commonly_paired_with=[
                "motion_move_to_coord_001",
                "manipulation_control_gripper_001",
            ],
            pairing_reasons={
                "motion_move_to_coord_001": "Typically position before releasing: move → release",
                "manipulation_control_gripper_001": "Alternative atomic gripper control",
            },
            typical_before=[],
            typical_after=["motion_move_to_coord_001"],
        ),
        implementation=release_object,
    )


RELEASE_OBJECT_OPERATION = create_release_object_operation()
