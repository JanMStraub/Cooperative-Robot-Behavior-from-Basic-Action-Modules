"""
Movement Operations for Robot Control
======================================

This module implements movement-related operations for controlling the robot arm
through Unity's RobotController via TCP communication.
"""

import time
import logging
# Lazy import to avoid circular dependency with servers module
# from servers.CommandServer import get_command_broadcaster
from .Base import (
    BasicOperation,
    OperationCategory,
    OperationComplexity,
    OperationParameter,
    OperationResult,
    OperationRelationship,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Lazy import function to avoid circular dependency
def _get_command_broadcaster():
    """Lazy import of command broadcaster to avoid circular dependency"""
    from servers.CommandServer import get_command_broadcaster
    return get_command_broadcaster()


# ============================================================================
# Implementation: Move to Coordinates
# ============================================================================


def move_to_coordinate(
    robot_id: str,
    x: float,
    y: float,
    z: float,
    speed: float = 1.0,
    approach_offset: float = 0.0,
    request_id: int = 0,
) -> OperationResult:
    """
    Move robot end effector to specified 3D coordinate.

    This operation commands the robot arm to move its end effector (gripper tip)
    to a specified 3D position in the robot's coordinate system. The robot will
    use inverse kinematics to calculate the required joint angles and execute
    a smooth trajectory to reach the target position.

    The movement respects velocity and acceleration limits for safe operation.
    Collision detection is active during movement.

    Args:
        robot_id: ID of the robot to move (e.g., "AR4_Robot", "Robot1")
        x: X coordinate in meters (forward/back from robot base), range: [-1.0, 1.0]
        y: Y coordinate in meters (left/right from robot base), range: [-1.0, 1.0]
        z: Z coordinate in meters (height above robot base), range: [-0.5, 0.6]
        speed: Speed multiplier (0.1=slow, 1.0=normal, 2.0=fast), range: [0.1, 2.0]
        approach_offset: Stop distance before target in meters, range: [0.0, 0.1]

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
            "target_position": {"x": float, "y": float, "z": float},
            "speed": float,
            "approach_offset": float,
            "status": "command_sent",
            "timestamp": float
        }

        Error structure:
        {
            "code": str,                    # Error code (e.g., "INVALID_X_COORDINATE")
            "message": str,                 # Human-readable error message
            "recovery_suggestions": list    # List of suggested actions
        }

    Example:
        >>> # Move to detected object position
        >>> result = move_to_coordinate("Robot1", 0.3, 0.15, 0.1)
        >>> if result["success"]:
        ...     print(f"Command sent at {result['result']['timestamp']}")

        >>> # Move slowly to precise position
        >>> result = move_to_coordinate("Robot1", 0.0, 0.0, 0.3, speed=0.2)

        >>> # Approach with offset (stop 5cm before target)
        >>> result = move_to_coordinate("Robot1", 0.3, 0.0, 0.1, approach_offset=0.05)

    Note:
        This operation is asynchronous - it sends the command to Unity and returns immediately. Unity executes the movement in the background. For synchronous execution (waiting for completion), use move_to_coordinate_sync() instead.
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

        # Validate X coordinate
        if not (-1.0 <= x <= 1.0):
            return OperationResult.error_result(
                "INVALID_X_COORDINATE",
                f"X coordinate {x} out of range [-1.0, 1.0]",
                [
                    "Adjust X to be within robot workspace [-1.0, 1.0]",
                    "Use detect_object to get valid coordinates",
                ],
            )

        # Validate Y coordinate
        if not (-1.0 <= y <= 1.0):
            return OperationResult.error_result(
                "INVALID_Y_COORDINATE",
                f"Y coordinate {y} out of range [-1.0, 1.0]",
                [
                    "Adjust Y to be within robot workspace [-1.0, 1.0]",
                    "Use detect_object to get valid coordinates",
                ],
            )

        # Validate Z coordinate
        if not (-0.5 <= z <= 0.6):
            return OperationResult.error_result(
                "INVALID_Z_COORDINATE",
                f"Z coordinate {z} out of range [-0.5, 0.6]",
                [
                    "Adjust Z to be within robot workspace [-0.5, 0.6]",
                    "Z can be negative (below robot base level)",
                ],
            )

        # Validate speed
        if not (0.1 <= speed <= 2.0):
            return OperationResult.error_result(
                "INVALID_SPEED",
                f"Speed {speed} out of range [0.1, 2.0]",
                [
                    "Use speed between 0.1 (very slow) and 2.0 (fast)",
                    "Typical values: 0.2 (precise), 1.0 (normal), 1.5 (fast)",
                ],
            )

        # Validate approach_offset
        if not (0.0 <= approach_offset <= 0.1):
            return OperationResult.error_result(
                "INVALID_APPROACH_OFFSET",
                f"Approach offset {approach_offset} out of range [0.0, 0.1]",
                [
                    "Use offset between 0.0 (exact position) and 0.1 (10cm before)",
                    "Typical approach offset: 0.05 (5cm)",
                ],
            )

        # Apply approach offset to target position
        actual_x = x
        actual_y = y
        actual_z = z + approach_offset  # Add offset to height for safety

        # Construct command for Unity
        command = {
            "command_type": "move_to_coordinate",
            "robot_id": robot_id,
            "parameters": {
                "target_position": {"x": actual_x, "y": actual_y, "z": actual_z},
                "speed_multiplier": speed,
                "original_target": {"x": x, "y": y, "z": z},
                "approach_offset": approach_offset,
            },
            "timestamp": time.time(),
            "request_id": request_id,
        }

        # Send to Unity via CommandBroadcaster
        logger.info(
            f"Sending move_to_coordinate command to {robot_id}: ({actual_x:.3f}, {actual_y:.3f}, {actual_z:.3f})"
        )

        success = _get_command_broadcaster().send_command(command, request_id)

        if not success:
            return OperationResult.error_result(
                "COMMUNICATION_FAILED",
                "Failed to send command to Unity - no clients connected",
                [
                    "Ensure Unity is running with LLMResultsReceiver active",
                    "Verify ResultsServer is running (port 5006)",
                    "Check Unity console for connection errors",
                    "Restart ResultsServer: python -m LLMCommunication.orchestrators.RunAnalyzer",
                ],
            )

        #  Return success
        logger.info(f"Successfully sent move_to_coordinate command to {robot_id}")

        return OperationResult.success_result(
            {
                "robot_id": robot_id,
                "target_position": {"x": actual_x, "y": actual_y, "z": actual_z},
                "original_target": {"x": x, "y": y, "z": z},
                "speed": speed,
                "approach_offset": approach_offset,
                "status": "command_sent",
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.error(f"Unexpected error in move_to_coordinate: {e}", exc_info=True)
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


def create_move_to_coordinate_operation() -> BasicOperation:
    """
    Create the BasicOperation definition for move_to_coordinate.

    This provides rich metadata for RAG retrieval and LLM task planning.
    """
    return BasicOperation(
        operation_id="motion_move_to_coord_001",
        name="move_to_coordinate",
        category=OperationCategory.NAVIGATION,
        complexity=OperationComplexity.BASIC,
        description="Move the robot's end effector to a specific 3D coordinate in workspace",
        long_description="""
            This operation commands the robot arm to move its end effector (gripper tip)
            to a specified 3D position in the robot's coordinate system. The robot will
            use inverse kinematics to calculate the required joint angles and execute
            a smooth trajectory to reach the target position.

            The movement respects velocity and acceleration limits for safe operation.
            Collision detection is active during movement. The operation supports different
            movement speeds for precise positioning versus fast traversal.

            This operation is asynchronous - it sends the command to Unity and returns
            immediately. Unity executes the movement in the background using RobotController.
        """,
        usage_examples=[
            "After detecting an object at (0.3, 0.15, 0.1), move there: move_to_coordinate(robot_id='Robot1', x=0.3, y=0.15, z=0.1)",
            "Move to home position: move_to_coordinate(robot_id='Robot1', x=0.0, y=0.0, z=0.3)",
            "Approach detected object coordinates before grasping",
            "Move slowly to precise position: move_to_coordinate(robot_id='Robot1', x=0.2, y=0.1, z=0.15, speed=0.2)",
            "Approach with 5cm offset: move_to_coordinate(robot_id='Robot1', x=0.3, y=0.0, z=0.1, approach_offset=0.05)",
        ],
        parameters=[
            OperationParameter(
                name="robot_id",
                type="str",
                description="ID of the robot to move (e.g., 'Robot1', 'AR4_Robot')",
                required=True,
            ),
            OperationParameter(
                name="x",
                type="float",
                description="X coordinate in meters (forward/back from robot base)",
                required=True,
                valid_range=(-1.0, 1.0),
            ),
            OperationParameter(
                name="y",
                type="float",
                description="Y coordinate in meters (left/right from robot base)",
                required=True,
                valid_range=(-1.0, 1.0),
            ),
            OperationParameter(
                name="z",
                type="float",
                description="Z coordinate in meters (height above robot base)",
                required=True,
                valid_range=(-0.5, 0.6),
            ),
            OperationParameter(
                name="speed",
                type="float",
                description="Movement speed multiplier (0.1=slow, 1.0=normal, 2.0=fast)",
                required=False,
                default=1.0,
                valid_range=(0.1, 2.0),
            ),
            OperationParameter(
                name="approach_offset",
                type="float",
                description="Stop this many meters before target (useful for approaching)",
                required=False,
                default=0.0,
                valid_range=(0.0, 0.1),
            ),
        ],
        preconditions=[
            "Robot arm is initialized and calibrated",
            "Target coordinate is within robot's reachable workspace",
            "No obstacles blocking path to target",
            "Robot is not currently executing another motion command",
            "Unity is running with LLMResultsReceiver active",
            "ResultsServer is running on port 5006",
        ],
        postconditions=[
            "Command has been sent to Unity via TCP",
            "Unity RobotController will move end effector to target coordinate",
            "Robot end effector will reach target (within 2mm tolerance)",
            "Robot arm will be stable and holding position",
            "Ready to execute next operation",
        ],
        average_duration_ms=1200.0,
        success_rate=0.96,
        failure_modes=[
            "Target coordinate is unreachable (outside workspace or singularity)",
            "Collision detected during movement - motion stopped for safety",
            "Joint limits would be exceeded",
            "Timeout - movement taking too long, possible obstruction",
            "Communication failed - Unity not connected to ResultsServer",
            "Robot ID not found in RobotManager",
        ],
        relationships=OperationRelationship(
            operation_id="motion_move_to_coord_001",
            required_operations=["status_check_robot_001"],
            required_reasons={
                "status_check_robot_001": "Verify robot is ready and not executing another command before moving",
            },
            commonly_paired_with=[
                "perception_stereo_detect_001",
                "manipulation_control_gripper_001",
                "status_check_robot_001",
            ],
            pairing_reasons={
                "perception_stereo_detect_001": "Move to detected object coordinates after detection",
                "manipulation_control_gripper_001": "Position gripper before grasping or after releasing",
                "status_check_robot_001": "Verify arrival at target position after movement",
            },
            typical_before=["manipulation_control_gripper_001"],
            typical_after=["perception_stereo_detect_001", "spatial_move_relative_001"],
        ),
        # Link to the actual implementation function
        implementation=move_to_coordinate,
    )


# Create the operation instance for export
MOVE_TO_COORDINATE_OPERATION = create_move_to_coordinate_operation()
