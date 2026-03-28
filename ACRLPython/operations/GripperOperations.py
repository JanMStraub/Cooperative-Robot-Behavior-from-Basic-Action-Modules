#!/usr/bin/env python3
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
from .Validators import validate_robot_id
from .ROSDispatcher import execute_with_ros_fallback

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
    robot_id: str,
    open_gripper: bool,
    request_id: int = 0,
    object_id: Optional[str] = None,
    use_ros: Optional[bool] = None,
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
        if err := validate_robot_id(robot_id):
            return err

        # Validate open_gripper parameter
        if not isinstance(open_gripper, bool):
            return OperationResult.error_result(
                "INVALID_OPEN_GRIPPER_PARAMETER",
                f"open_gripper must be a boolean, got: {type(open_gripper).__name__}",
                [
                    "Use open_gripper=True to open the gripper or open_gripper=False to close it"
                ],
            )

        def _update_gripper_world_state():
            """Optimistic gripper state update — no Unity stream needed."""
            try:
                from core.Imports import get_world_state

                get_world_state().update_robot_state(
                    robot_id,
                    {"gripper_state": "open" if open_gripper else "closed"},
                )
            except Exception as _exc:
                logger.debug(f"Could not update gripper WorldState for {robot_id}: {_exc}")

        def _ros_path():
            from ros2.ROSBridge import ROSBridge

            bridge = ROSBridge.get_instance()
            gripper_position = 1.0 if open_gripper else 0.0
            result = bridge.control_gripper(gripper_position, robot_id=robot_id)
            if result and result.get("success"):
                logger.info(f"ROS gripper command sent for {robot_id}")
                _update_gripper_world_state()
                return OperationResult.success_result(
                    {
                        "robot_id": robot_id,
                        "open_gripper": open_gripper,
                        "status": "ros_command_sent",
                        "timestamp": time.time(),
                    }
                )
            return None

        def _tcp_path():
            params: Dict[str, Any] = {"open_gripper": open_gripper}
            if object_id:
                params["object_id"] = object_id
            command = {
                "command_type": "control_gripper",
                "robot_id": robot_id,
                "parameters": params,
                "timestamp": time.time(),
                "request_id": request_id,
            }
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
                        "Verify CommandServer is running (port 5010)",
                        "Check Unity console for connection errors",
                        "Restart backend: python -m orchestrators.RunRobotController",
                    ],
                )
            logger.info(f"Successfully sent control_gripper command to {robot_id}")
            _update_gripper_world_state()
            return OperationResult.success_result(
                {
                    "robot_id": robot_id,
                    "open_gripper": open_gripper,
                    "status": "command_sent",
                    "timestamp": time.time(),
                }
            )

        return execute_with_ros_fallback(_ros_path, _tcp_path, use_ros)

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
            "robot_is_initialized(robot_id)",
        ],
        postconditions=[],
        average_duration_ms=500,  # Time for gripper to fully open or close
        success_rate=0.98,
        failure_modes=[
            "Robot ID not found in RobotManager",
            "Communication failed - Unity not connected to CommandServer",
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
    use_ros: Optional[bool] = None,
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

        def _ros_path():
            from ros2.ROSBridge import ROSBridge

            bridge = ROSBridge.get_instance()
            # Gripper: 1.0 = fully open (normalized value)
            result = bridge.control_gripper(1.0, robot_id=robot_id)
            if result and result.get("success"):
                logger.info(f"ROS release_object command sent for {robot_id}")
                return OperationResult.success_result(
                    {
                        "robot_id": robot_id,
                        "status": "ros_command_sent",
                        "timestamp": time.time(),
                    }
                )
            return None  # signal failure to ROSDispatcher

        def _tcp_path():
            command = {
                "command_type": "release_object",
                "robot_id": robot_id,
                "parameters": {"open_gripper": True},
                "timestamp": time.time(),
                "request_id": request_id,
            }
            logger.info(
                f"Sending release_object command to {robot_id} (atomic - gripper only)"
            )
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

        return execute_with_ros_fallback(_ros_path, _tcp_path, use_ros)

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
            "robot_is_initialized(robot_id)",
        ],
        postconditions=[],
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


# ============================================================================
# Implementation: Place Object
# ============================================================================

# Height above the target surface the arm hovers at before descending.
PLACE_HOVER_OFFSET: float = 0.15  # 15 cm above target

# Height above target position where gripper opens.
# Keeps a small gap so the object rests on the surface rather than being
# forced into it, while still being close enough not to drop freely.
PLACE_TCP_OFFSET: float = 0.055  # 5.5 cm above target (matches GRASP_TCP_OFFSET)


def place_object(
    robot_id: str,
    x: float,
    y: float,
    z: float,
    use_ros: Optional[bool] = None,
    request_id: int = 0,
) -> OperationResult:
    """
    Carefully place a held object at the specified world position.

    This is the inverse of grasp_object.  It performs a controlled
    three-step sequence:
      1. Move (via MoveIt or Unity IK) to a hover position PLACE_HOVER_OFFSET
         above the target.
      2. Cartesian descent to PLACE_TCP_OFFSET above the target surface so
         the object lands gently rather than dropping.
      3. Open the gripper to release the object.
      4. Cartesian ascent back to the hover height so the arm clears the
         placed object.

    The ROS path uses plan_and_execute for the hover move and
    plan_cartesian_descent for the final lowering, mirroring the grasp
    approach.  The TCP fallback sends a single ``place_object`` command to
    Unity which executes the same sequence inside a coroutine.

    Args:
        robot_id: ID of the robot performing the placement.
        x: Target X coordinate in Unity world space (metres).
        y: Target Y coordinate in Unity world space (metres).
        z: Target Z coordinate in Unity world space (metres).
        use_ros: Override ROS/TCP path selection.  None = use config default.
        request_id: Optional request ID for tracking.

    Returns:
        OperationResult with placement confirmation or error details.

    Example:
        >>> result = place_object("Robot1", x=-0.18, y=0.06, z=0.05)
        >>> if result.success:
        ...     print("Object placed successfully")
    """
    try:
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string, got: {robot_id}",
                ["Provide a valid robot ID (e.g., 'Robot1')"],
            )

        def _ros_path():
            from ros2.ROSBridge import ROSBridge

            bridge = ROSBridge.get_instance()

            hover_pos = {"x": x, "y": y + PLACE_HOVER_OFFSET, "z": z}
            place_pos = {"x": x, "y": y + PLACE_TCP_OFFSET, "z": z}

            # Step 1: Move to hover above target.
            logger.info(f"place_object: moving to hover above target for {robot_id}")
            hover_result = bridge.plan_and_execute(
                position=hover_pos,
                robot_id=robot_id,
            )
            if not hover_result or not hover_result.get("success"):
                err = hover_result.get("error", "Unknown") if hover_result else "No response"
                logger.warning(f"place_object: hover move failed ({err}), attempting direct descent")

            # Brief settle pause so /joint_states reflects the hover pose before
            # MoveIt samples the start state for the descent plan.
            time.sleep(0.3)

            # Step 2: Descend to place height using free-space planning.
            # plan_cartesian_descent is NOT used here: MoveIt's collision model does
            # not include the held object, so a straight-line path through the
            # object's swept volume frequently fails at 0% completion.
            # Free-space planning (plan_and_execute) finds a collision-free path.
            logger.info(f"place_object: descending to place position for {robot_id}")
            descent_result = bridge.plan_and_execute(
                position=place_pos,
                robot_id=robot_id,
            )
            if not descent_result or not descent_result.get("success"):
                err = descent_result.get("error", "Unknown") if descent_result else "No response"
                logger.warning(f"place_object: descent failed ({err}), releasing at current height")

            # Step 3: Open gripper and wait long enough for Unity to fully process
            # the detach before the ascent trajectory is published.  The gripper
            # command is fire-and-forget (ROS topic publish) so we must sleep to
            # let ROSGripperSubscriber detach the object before we move away.
            logger.info(f"place_object: releasing object for {robot_id}")
            bridge.control_gripper(1.0, robot_id=robot_id)
            time.sleep(1.0)

            # Step 4: Ascend back to hover height to clear the placed object.
            logger.info(f"place_object: ascending after place for {robot_id}")
            bridge.plan_and_execute(position=hover_pos, robot_id=robot_id)

            return OperationResult.success_result(
                {
                    "robot_id": robot_id,
                    "placed_at": {"x": x, "y": y, "z": z},
                    # "ros_executed" tells SequenceExecutor to skip the Unity
                    # completion-signal wait — the ROS path is fully synchronous
                    # and Unity never sends a TCP completion message for it.
                    "status": "ros_executed",
                    "timestamp": time.time(),
                }
            )

        def _tcp_path():
            command = {
                "command_type": "place_object",
                "robot_id": robot_id,
                "parameters": {
                    "target_position": {"x": x, "y": y, "z": z},
                    "hover_offset": PLACE_HOVER_OFFSET,
                    "tcp_offset": PLACE_TCP_OFFSET,
                },
                "timestamp": time.time(),
                "request_id": request_id,
            }
            logger.info(f"Sending place_object command to {robot_id} at ({x}, {y}, {z})")
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
            return OperationResult.success_result(
                {
                    "robot_id": robot_id,
                    "placed_at": {"x": x, "y": y, "z": z},
                    "status": "command_sent",
                    "timestamp": time.time(),
                }
            )

        return execute_with_ros_fallback(_ros_path, _tcp_path, use_ros)

    except Exception as e:
        logger.error(f"Unexpected error in place_object: {e}", exc_info=True)
        return OperationResult.error_result(
            "UNEXPECTED_ERROR",
            f"Unexpected error occurred: {str(e)}",
            ["Check logs for detailed error information", "Retry the operation"],
        )


PLACE_OBJECT_OPERATION = BasicOperation(
    operation_id="manipulation_place_object_003",
    name="place_object",
    category=OperationCategory.MANIPULATION,
    complexity=OperationComplexity.INTERMEDIATE,
    description=(
        "Carefully place a held object at a target position with controlled descent and ascent"
    ),
    long_description="""
        Performs a controlled place sequence that is the inverse of grasp_object:
        1. Move to a hover position above the target (PLACE_HOVER_OFFSET = 15 cm).
        2. Cartesian descent to just above the surface (PLACE_TCP_OFFSET = 5.5 cm).
        3. Open gripper to release the object gently onto the surface.
        4. Cartesian ascent back to hover height to clear the placed object.

        Use this instead of release_object whenever you need the object to
        land at a specific position (e.g. placing on a field, on a workbench,
        or into a container).  release_object is only appropriate for an
        explicit immediate gripper open at the current position.
    """,
    usage_examples=[
        "place_object('Robot1', x=-0.18, y=0.06, z=0.05) — place at field G center",
        "Typical sequence: detect_field → move_to_coordinate (hover) → place_object",
    ],
    parameters=[
        OperationParameter(
            name="robot_id",
            type="str",
            description="ID of the robot performing the placement",
            required=True,
        ),
        OperationParameter(
            name="x",
            type="float",
            description="Target X coordinate in Unity world space (metres)",
            required=True,
        ),
        OperationParameter(
            name="y",
            type="float",
            description="Target Y coordinate in Unity world space (metres)",
            required=True,
        ),
        OperationParameter(
            name="z",
            type="float",
            description="Target Z coordinate in Unity world space (metres)",
            required=True,
        ),
    ],
    preconditions=[],
    postconditions=[],
    average_duration_ms=8000.0,
    success_rate=0.90,
    failure_modes=[
        "IK infeasible for hover or descent position",
        "Cartesian descent fraction too low (workspace boundary)",
        "Object slips before gripper opens",
    ],
    relationships=OperationRelationship(
        operation_id="manipulation_place_object_003",
        required_operations=[],
        commonly_paired_with=[
            "perception_detect_field_004",
            "manipulation_grasp_object_001",
        ],
        pairing_reasons={
            "perception_detect_field_004": "Detect field position to supply x/y/z for placement",
            "manipulation_grasp_object_001": "Grasp precedes place in a pick-and-place sequence",
        },
        typical_after=[
            "manipulation_grasp_object_001",
            "motion_move_to_coord_001",
            "perception_detect_field_004",
        ],
        typical_before=["manipulation_release_object_002"],
    ),
    implementation=place_object,
)
