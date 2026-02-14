"""
Multi-Robot Coordination Operations (Level 4)
==============================================

This module implements ATOMIC Level 4 operations from the thesis exposé:
- detect_other_robot: Robot-robot detection
- mirror_movement_of_other_robot: Movement mirroring

REMOVED (non-atomic):
- hand_over_object_to_another_robot: Use WorkflowPatterns.HANDOFF_PATTERN instead

These operations enable multi-robot collaboration and coordination.
All operations are atomic - the LLM chains them to create complex workflows.
"""

import time
import logging
from typing import Optional, Dict, Any

# Import from centralized lazy import system
try:
    from ..core.Imports import get_command_broadcaster as _get_command_broadcaster
    from ..core.Imports import get_unified_image_storage
except ImportError:
    from core.Imports import get_command_broadcaster as _get_command_broadcaster
    from core.Imports import get_unified_image_storage

# Handle both direct execution and package import
try:
    from .Base import (
        BasicOperation,
        OperationCategory,
        OperationComplexity,
        OperationParameter,
        OperationResult,
        OperationRelationship,
    )
except ImportError:
    from operations.Base import (
        BasicOperation,
        OperationCategory,
        OperationComplexity,
        OperationParameter,
        OperationResult,
        OperationRelationship,
    )

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Implementation: detect_other_robot - Robot-robot detection
# ============================================================================


def detect_other_robot(
    robot_id: str,
    target_robot_id: str,
    camera_id: str = "main",
    request_id: int = 0,
) -> OperationResult:
    """
    Detect another robot in the workspace using vision.

    This operation uses computer vision to detect and locate another robot
    in the shared workspace, enabling spatial awareness for coordination.

    Args:
        robot_id: ID of the detecting robot
        target_robot_id: ID of the robot to detect
        camera_id: Camera ID for detection

    Returns:
        OperationResult with detection data including position and distance

    Example:
        >>> # Robot1 detects Robot2
        >>> result = detect_other_robot("Robot1", "Robot2")
        >>> if result["success"]:
        ...     position = result["result"]["position"]
        ...     distance = result["result"]["distance"]
    """
    try:
        # Validate robot IDs
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string",
                ["Provide a valid robot ID"],
            )

        if not target_robot_id or not isinstance(target_robot_id, str):
            return OperationResult.error_result(
                "INVALID_TARGET_ROBOT_ID",
                f"Target robot ID must be a non-empty string",
                ["Provide a valid target robot ID"],
            )

        # Get WorldState for robot positions
        try:
            from .WorldState import WorldState
        except ImportError:
            from operations.WorldState import WorldState

        world_state = WorldState()

        # Get target robot position from WorldState
        target_state = world_state.get_robot_state(target_robot_id)
        if not target_state:
            return OperationResult.error_result(
                "TARGET_ROBOT_NOT_FOUND",
                f"Target robot '{target_robot_id}' not found in world state",
                [
                    "Verify target robot is active in Unity",
                    "Check WorldStatePublisher is sending data",
                ],
            )

        # Get detecting robot position
        detector_state = world_state.get_robot_state(robot_id)
        if not detector_state:
            return OperationResult.error_result(
                "DETECTOR_ROBOT_NOT_FOUND",
                f"Detecting robot '{robot_id}' not found in world state",
                ["Verify robot is active"],
            )

        # Calculate distance between robots
        import math

        detector_pos = detector_state.get("end_effector_position", detector_state.get("position"))
        target_pos = target_state.get("end_effector_position", target_state.get("position"))

        if not detector_pos or not target_pos:
            return OperationResult.error_result(
                "POSITION_DATA_MISSING",
                "Robot position data missing",
                ["Ensure WorldStatePublisher is active"],
            )

        distance = math.sqrt(
            (detector_pos["x"] - target_pos["x"]) ** 2
            + (detector_pos["y"] - target_pos["y"]) ** 2
            + (detector_pos["z"] - target_pos["z"]) ** 2
        )

        logger.info(
            f"Robot {robot_id} detected {target_robot_id} at distance {distance:.3f}m"
        )

        return OperationResult.success_result(
            {
                "robot_id": robot_id,
                "target_robot_id": target_robot_id,
                "position": target_pos,
                "distance": distance,
                "detected": True,
                "camera_id": camera_id,
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.error(f"Error in detect_other_robot: {e}", exc_info=True)
        return OperationResult.error_result(
            "DETECTION_ERROR",
            f"Robot detection failed: {str(e)}",
            ["Check logs for details"],
        )


# ============================================================================
# Implementation: mirror_movement_of_other_robot - Movement mirroring
# ============================================================================


def mirror_movement_of_other_robot(
    robot_id: str,
    target_robot_id: str,
    mirror_axis: str = "x",
    scale_factor: float = 1.0,
    request_id: int = 0,
    use_ros: bool = None,
) -> OperationResult:
    """
    Mirror the movement of another robot.

    This operation commands a robot to mirror (copy/reflect) the movements
    of another robot, useful for synchronized tasks or demonstration.

    Args:
        robot_id: ID of the mirroring robot
        target_robot_id: ID of the robot to mirror
        mirror_axis: Axis to mirror across ("x", "y", "z", or "none" for direct copy)
        scale_factor: Scale factor for mirrored movement (1.0 = same size)
        request_id: Optional request ID for tracking
        use_ros: Whether to use ROS for motion planning (None = auto-detect from config)

    Returns:
        OperationResult with mirroring activation confirmation

    Example:
        >>> # Robot2 mirrors Robot1's movements across X axis
        >>> result = mirror_movement_of_other_robot("Robot2", "Robot1", "x")

        >>> # Robot2 copies Robot1's movements exactly
        >>> result = mirror_movement_of_other_robot("Robot2", "Robot1", "none")
    """
    try:
        # Validate robot IDs
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string",
                ["Provide a valid robot ID"],
            )

        if not target_robot_id or not isinstance(target_robot_id, str):
            return OperationResult.error_result(
                "INVALID_TARGET_ROBOT_ID",
                f"Target robot ID must be a non-empty string",
                ["Provide a valid target robot ID"],
            )

        # Validate mirror_axis
        valid_axes = ["x", "y", "z", "none"]
        if mirror_axis not in valid_axes:
            return OperationResult.error_result(
                "INVALID_MIRROR_AXIS",
                f"mirror_axis must be one of {valid_axes}, got: {mirror_axis}",
                [f"Use one of: {', '.join(valid_axes)}"],
            )

        # Validate scale_factor
        if not (0.1 <= scale_factor <= 2.0):
            return OperationResult.error_result(
                "INVALID_SCALE_FACTOR",
                f"scale_factor must be in range [0.1, 2.0], got: {scale_factor}",
                ["Use scale between 0.1 (10%) and 2.0 (200%)"],
            )

        # Determine whether to use ROS or TCP path
        _use_ros = use_ros
        if _use_ros is None:
            try:
                from config.ROS import ROS_ENABLED, DEFAULT_CONTROL_MODE
                _use_ros = ROS_ENABLED and DEFAULT_CONTROL_MODE in ("ros", "hybrid")
            except ImportError:
                _use_ros = False

        # Note: Mirror movement is a continuous tracking operation
        # ROS support would require real-time trajectory tracking via ROS topics
        # For now, this is best handled by Unity directly (TCP path)
        if _use_ros:
            logger.info("Mirror movement via ROS not yet implemented - using Unity direct control")
            _use_ros = False

        # Construct command (TCP path)
        command = {
            "command_type": "mirror_movement",
            "robot_id": robot_id,
            "parameters": {
                "target_robot_id": target_robot_id,
                "mirror_axis": mirror_axis,
                "scale_factor": scale_factor,
            },
            "timestamp": time.time(),
            "request_id": request_id,
        }

        logger.info(
            f"Sending mirror_movement command: {robot_id} mirrors {target_robot_id}"
        )

        success = _get_command_broadcaster().send_command(command, request_id)

        if not success:
            return OperationResult.error_result(
                "COMMUNICATION_FAILED",
                "Failed to send command to Unity",
                ["Ensure Unity is running"],
            )

        logger.info(f"Successfully activated mirroring for {robot_id}")

        return OperationResult.success_result(
            {
                "robot_id": robot_id,
                "target_robot_id": target_robot_id,
                "mirror_axis": mirror_axis,
                "scale_factor": scale_factor,
                "status": "mirroring_active",
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.error(f"Unexpected error in mirror_movement: {e}", exc_info=True)
        return OperationResult.error_result(
            "UNEXPECTED_ERROR",
            f"Unexpected error occurred: {str(e)}",
            ["Check logs"],
        )


# ============================================================================
# REMOVED: hand_over_object_to_another_robot
# ============================================================================
# This operation was REMOVED because it is non-atomic (combines 5 steps).
# For handoff workflows, see operations/WorkflowPatterns.py for the
# HANDOFF_PATTERN showing how to chain atomic operations.
#
# The LLM should chain operations:
# 1. move_to_coordinate(robot_from, handoff_position)
# 2. signal(robot_from, "ready_for_handoff")
# 3. wait_for_signal(robot_to, "ready_for_handoff")
# 4. move_to_coordinate(robot_to, handoff_position)
# 5. control_gripper(robot_to, open=False, object_id=object)
# 6. control_gripper(robot_from, open=True)
# 7. signal completion and move away
# ============================================================================


# ============================================================================
# BasicOperation Definitions
# ============================================================================


def create_detect_other_robot_operation() -> BasicOperation:
    """Create the BasicOperation definition for detect_other_robot."""
    return BasicOperation(
        operation_id="coordination_detect_robot_001",
        name="detect_other_robot",
        category=OperationCategory.PERCEPTION,
        complexity=OperationComplexity.INTERMEDIATE,
        description="Detect and locate another robot in shared workspace",
        long_description="""
            This operation uses vision and WorldState to detect another robot
            in the workspace, providing spatial awareness for coordination tasks.

            Returns robot position and distance for coordination planning.
        """,
        usage_examples=[
            "detect_other_robot('Robot1', 'Robot2')",
            "Check distance before coordination: if distance < 0.3, coordinate movements",
        ],
        parameters=[
            OperationParameter(
                name="robot_id",
                type="str",
                description="Detecting robot ID",
                required=True,
            ),
            OperationParameter(
                name="target_robot_id",
                type="str",
                description="Robot to detect",
                required=True,
            ),
            OperationParameter(
                name="camera_id",
                type="str",
                description="Camera for detection",
                required=False,
                default="main",
            ),
        ],
        preconditions=["Both robots active", "WorldState updated"],
        postconditions=["Target robot detected", "Position and distance returned"],
        average_duration_ms=80,
        success_rate=0.96,
        failure_modes=["Target robot not in view", "WorldState not updated"],
        implementation=detect_other_robot,
    )


def create_mirror_movement_operation() -> BasicOperation:
    """Create the BasicOperation definition for mirror_movement_of_other_robot."""
    return BasicOperation(
        operation_id="coordination_mirror_movement_002",
        name="mirror_movement_of_other_robot",
        category=OperationCategory.NAVIGATION,
        complexity=OperationComplexity.COMPLEX,
        description="Mirror the movements of another robot",
        long_description="""
            This operation enables one robot to mirror (copy/reflect) the
            movements of another robot in real-time.

            Useful for synchronized tasks, demonstration, or coordinated
            manipulation where movements should be symmetric.
        """,
        usage_examples=[
            "mirror_movement_of_other_robot('Robot2', 'Robot1', 'x')",
            "Synchronized bimanual manipulation with mirrored movements",
        ],
        parameters=[
            OperationParameter(
                name="robot_id",
                type="str",
                description="Mirroring robot ID",
                required=True,
            ),
            OperationParameter(
                name="target_robot_id",
                type="str",
                description="Robot to mirror",
                required=True,
            ),
            OperationParameter(
                name="mirror_axis",
                type="str",
                description="Axis to mirror across ('x', 'y', 'z', 'none')",
                required=False,
                default="x",
            ),
            OperationParameter(
                name="scale_factor",
                type="float",
                description="Scale factor for movements (0.1-2.0)",
                required=False,
                default=1.0,
            ),
        ],
        preconditions=["Both robots active", "Sufficient workspace clearance"],
        postconditions=["Mirroring active", "Movements synchronized"],
        average_duration_ms=50,  # Activation time
        success_rate=0.89,
        failure_modes=["Workspace collision", "Robot limits exceeded"],
        implementation=mirror_movement_of_other_robot,
    )


# ============================================================================
# REMOVED: create_hand_over_object_operation
# ============================================================================
# The hand_over_object_to_another_robot operation was removed because it
# is non-atomic. See WorkflowPatterns.py for HANDOFF_PATTERN.
# ============================================================================


# ============================================================================
# Create operation instances for export
# ============================================================================

DETECT_OTHER_ROBOT_OPERATION = create_detect_other_robot_operation()
MIRROR_MOVEMENT_OPERATION = create_mirror_movement_operation()
# HAND_OVER_OBJECT_OPERATION removed - use WorkflowPatterns.HANDOFF_PATTERN instead
