#!/usr/bin/env python3
"""
Movement Operations for Robot Control
======================================

This module implements movement-related operations for controlling the robot arm
through Unity's RobotController via TCP communication.
"""

import time
import logging
from typing import Optional

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
from .Validators import (
    validate_robot_id,
    validate_xyz,
    validate_speed,
    validate_approach_offset,
)
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
# Implementation: Move to Coordinates
# ============================================================================


def move_to_coordinate(
    robot_id: str,
    x: float,
    y: float,
    z: float,
    speed: float = 1.0,
    approach_offset: float = 0.0,
    use_advanced_planning: bool = True,
    request_id: int = 0,
    use_ros: Optional[bool] = None,
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
        approach_offset: Lift above target in meters along Unity Y (up-axis), range: [0.0, 0.1]
        use_advanced_planning: Use full grasp planning pipeline (generates 15 candidates), default: True

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
        if err := validate_robot_id(robot_id):
            return err
        if err := validate_xyz(x, y, z):
            return err
        if err := validate_speed(speed):
            return err
        if err := validate_approach_offset(approach_offset):
            return err

        # Apply approach offset to target position
        # Unity uses Y-up coordinate system; approach_offset lifts the target above
        # the object along Unity's Y axis (height), not Z (depth/forward).
        actual_x = x
        actual_y = y + approach_offset  # Add offset to height (Unity Y = up)
        actual_z = z

        def _ros_path():
            from ros2.ROSBridge import ROSBridge

            bridge = ROSBridge.get_instance()
            result = bridge.plan_and_execute(
                position={"x": actual_x, "y": actual_y, "z": actual_z},
                robot_id=robot_id,
            )
            if result and result.get("success"):
                logger.info(f"ROS motion completed for {robot_id}")
                return OperationResult.success_result(
                    {
                        "robot_id": robot_id,
                        "target_position": {
                            "x": actual_x,
                            "y": actual_y,
                            "z": actual_z,
                        },
                        "speed": speed,
                        "approach_offset": approach_offset,
                        "status": "ros_executed",
                        "planning_time": result.get("planning_time", 0),
                        "timestamp": time.time(),
                    }
                )
            return None  # signal failure to ROSDispatcher

        def _tcp_path():
            command = {
                "command_type": "move_to_coordinate",
                "robot_id": robot_id,
                "parameters": {
                    "target_position": {"x": actual_x, "y": actual_y, "z": actual_z},
                    "speed_multiplier": speed,
                    "original_target": {"x": x, "y": y, "z": z},
                    "approach_offset": approach_offset,
                    "use_advanced_planning": use_advanced_planning,
                },
                "timestamp": time.time(),
                "request_id": request_id,
            }
            logger.info(
                f"Sending move_to_coordinate command to {robot_id}: "
                f"({actual_x:.3f}, {actual_y:.3f}, {actual_z:.3f})"
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

        return execute_with_ros_fallback(_ros_path, _tcp_path, use_ros)

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
                valid_range=(-0.65, 0.65),
            ),
            OperationParameter(
                name="y",
                type="float",
                description="Y coordinate in meters (left/right from robot base)",
                required=True,
                valid_range=(0.0, 0.7),
            ),
            OperationParameter(
                name="z",
                type="float",
                description="Z coordinate in meters (height above robot base)",
                required=True,
                valid_range=(-0.5, 0.5),
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
                description="Lift above target by this many meters along Unity Y (up-axis) before grasping",
                required=False,
                default=0.0,
                valid_range=(0.0, 0.1),
            ),
        ],
        preconditions=[
            "robot_is_initialized(robot_id)",
            "target_within_reach(robot_id, x, y, z)",
        ],
        postconditions=[],
        average_duration_ms=1200.0,
        success_rate=0.96,
        failure_modes=[
            "Target coordinate is unreachable (outside workspace or singularity)",
            "Collision detected during movement - motion stopped for safety",
            "Joint limits would be exceeded",
            "Timeout - movement taking too long, possible obstruction",
            "Communication failed - Unity not connected to CommandServer",
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
                "motion_pick_at_coord_004",
            ],
            pairing_reasons={
                "perception_stereo_detect_001": "Move to detected object coordinates after detection",
                "manipulation_control_gripper_001": (
                    "IMPORTANT: Never call control_gripper immediately after move_to_coordinate "
                    "with approach_offset > 0 — the gripper will close in mid-air. "
                    "Always add a second move_to_coordinate(approach_offset=0) descent step first, "
                    "or use pick_object_at_coordinate which encodes the full hover→descent→grasp pattern."
                ),
                "status_check_robot_001": "Verify arrival at target position after movement",
                "motion_pick_at_coord_004": "Use pick_object_at_coordinate when the goal is grasping at known coords",
            },
            typical_before=["manipulation_control_gripper_001"],
            typical_after=["perception_stereo_detect_001", "spatial_move_relative_001"],
        ),
        # Link to the actual implementation function
        implementation=move_to_coordinate,
    )


# Create the operation instance for export
MOVE_TO_COORDINATE_OPERATION = create_move_to_coordinate_operation()


# ============================================================================
# Implementation: Move from A to B (Explicit Waypoint Movement)
# ============================================================================


def move_from_a_to_b(
    robot_id: str,
    point_a: dict,
    point_b: dict,
    speed: float = 1.0,
    request_id: int = 0,
    use_ros: Optional[bool] = None,
) -> OperationResult:
    """
    Move robot end effector from point A to point B through explicit waypoints.

    This operation provides explicit A→B waypoint movement with validation
    of both start and end positions. Unlike move_to_coordinate which moves
    from current position, this operation validates the starting position.

    Args:
        robot_id: ID of the robot to move (e.g., "Robot1", "AR4_Robot")
        point_a: Start position dict with keys 'x', 'y', 'z' (meters)
        point_b: End position dict with keys 'x', 'y', 'z' (meters)
        speed: Speed multiplier (0.1=slow, 1.0=normal, 2.0=fast), range: [0.1, 2.0]
        request_id: Optional request ID for tracking
        use_ros: Whether to use ROS for motion planning (None = auto-detect from config)

    Returns:
        OperationResult with movement confirmation or error details

    Example:
        >>> # Move from home to object
        >>> result = move_from_a_to_b(
        ...     "Robot1",
        ...     {"x": 0.0, "y": 0.0, "z": 0.3},  # home position
        ...     {"x": 0.3, "y": 0.15, "z": 0.1}  # object position
        ... )
    """
    try:
        if err := validate_robot_id(robot_id):
            return err

        # Validate point_a
        if not isinstance(point_a, dict) or not all(
            k in point_a for k in ["x", "y", "z"]
        ):
            return OperationResult.error_result(
                "INVALID_POINT_A",
                f"Point A must be dict with x, y, z keys, got: {point_a}",
                ["Provide point_a as: {'x': 0.0, 'y': 0.0, 'z': 0.3}"],
            )

        # Validate point_b
        if not isinstance(point_b, dict) or not all(
            k in point_b for k in ["x", "y", "z"]
        ):
            return OperationResult.error_result(
                "INVALID_POINT_B",
                f"Point B must be dict with x, y, z keys, got: {point_b}",
                ["Provide point_b as: {'x': 0.3, 'y': 0.15, 'z': 0.1}"],
            )

        def _ros_path():
            from ros2.ROSBridge import ROSBridge

            bridge = ROSBridge.get_instance()
            result = bridge.plan_multi_waypoint(
                waypoints=[point_a, point_b], robot_id=robot_id
            )
            if result and result.get("success"):
                logger.info(f"ROS waypoint motion completed for {robot_id}")
                return OperationResult.success_result(
                    {
                        "robot_id": robot_id,
                        "point_a": point_a,
                        "point_b": point_b,
                        "speed": speed,
                        "status": "ros_executed",
                        "planning_time": result.get("planning_time", 0),
                        "timestamp": time.time(),
                    }
                )
            return None

        def _tcp_path():
            command = {
                "command_type": "move_from_a_to_b",
                "robot_id": robot_id,
                "parameters": {
                    "point_a": point_a,
                    "point_b": point_b,
                    "speed_multiplier": speed,
                },
                "timestamp": time.time(),
                "request_id": request_id,
            }
            logger.info(
                f"Sending move_from_a_to_b command to {robot_id}: A{point_a} → B{point_b}"
            )
            success = _get_command_broadcaster().send_command(command, request_id)
            if not success:
                return OperationResult.error_result(
                    "COMMUNICATION_FAILED",
                    "Failed to send command to Unity - no clients connected",
                    [
                        "Ensure Unity is running",
                        "Verify CommandServer is running (port 5010)",
                    ],
                )
            logger.info(f"Successfully sent move_from_a_to_b command to {robot_id}")
            return OperationResult.success_result(
                {
                    "robot_id": robot_id,
                    "point_a": point_a,
                    "point_b": point_b,
                    "speed": speed,
                    "status": "command_sent",
                    "timestamp": time.time(),
                }
            )

        return execute_with_ros_fallback(_ros_path, _tcp_path, use_ros)

    except Exception as e:
        logger.error(f"Unexpected error in move_from_a_to_b: {e}", exc_info=True)
        return OperationResult.error_result(
            "UNEXPECTED_ERROR",
            f"Unexpected error occurred: {str(e)}",
            ["Check logs for detailed error information", "Retry the operation"],
        )


def create_move_from_a_to_b_operation() -> BasicOperation:
    """Create the BasicOperation definition for move_from_a_to_b."""
    return BasicOperation(
        operation_id="motion_move_a_to_b_002",
        name="move_from_a_to_b",
        category=OperationCategory.NAVIGATION,
        complexity=OperationComplexity.BASIC,
        description="Move robot end effector from point A to point B through explicit waypoints",
        long_description="""
            This operation provides explicit A→B waypoint movement with validation
            of both start and end positions. Unlike move_to_coordinate which moves
            from current position, this operation validates the starting position.

            Useful for planned trajectories where both waypoints are known in advance.
        """,
        usage_examples=[
            "move_from_a_to_b('Robot1', {'x': 0.0, 'y': 0.0, 'z': 0.3}, {'x': 0.3, 'y': 0.15, 'z': 0.1})",
            "Move from home to detected object with explicit path validation",
        ],
        parameters=[
            OperationParameter(
                name="robot_id", type="str", description="Robot ID", required=True
            ),
            OperationParameter(
                name="point_a",
                type="dict",
                description="Start position with x, y, z keys (meters)",
                required=True,
            ),
            OperationParameter(
                name="point_b",
                type="dict",
                description="End position with x, y, z keys (meters)",
                required=True,
            ),
            OperationParameter(
                name="speed",
                type="float",
                description="Speed multiplier (0.1-2.0)",
                required=False,
                default=1.0,
            ),
        ],
        preconditions=[
            "robot_is_initialized(robot_id)",
        ],
        postconditions=[],
        average_duration_ms=1500.0,
        success_rate=0.95,
        failure_modes=["Path obstructed", "Point unreachable", "Communication failed"],
        implementation=move_from_a_to_b,
    )


MOVE_FROM_A_TO_B_OPERATION = create_move_from_a_to_b_operation()


# ============================================================================
# Implementation: Adjust End Effector Orientation
# ============================================================================


def adjust_end_effector_orientation(
    robot_id: str,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    request_id: int = 0,
    use_ros: Optional[bool] = None,
) -> OperationResult:
    """
    Adjust the end effector orientation without changing position.

    This operation modifies only the gripper orientation (roll, pitch, yaw)
    while maintaining the current position.

    Args:
        robot_id: ID of the robot to control
        roll: Roll angle in degrees (rotation around X axis), range: [-180, 180]
        pitch: Pitch angle in degrees (rotation around Y axis), range: [-180, 180]
        yaw: Yaw angle in degrees (rotation around Z axis), range: [-180, 180]
        request_id: Optional request ID for tracking
        use_ros: Whether to use ROS for motion planning (None = auto-detect from config)

    Returns:
        OperationResult with orientation adjustment confirmation

    Example:
        >>> # Rotate gripper 90 degrees for side grasp
        >>> result = adjust_end_effector_orientation("Robot1", roll=90.0)

        >>> # Adjust pitch for top-down grasp
        >>> result = adjust_end_effector_orientation("Robot1", pitch=-90.0)
    """
    try:
        if err := validate_robot_id(robot_id):
            return err

        # Validate angles
        for angle_name, angle_value in [("roll", roll), ("pitch", pitch), ("yaw", yaw)]:
            if not isinstance(angle_value, (int, float)):
                return OperationResult.error_result(
                    "INVALID_ANGLE",
                    f"{angle_name} must be a number, got: {type(angle_value).__name__}",
                    ["Provide angles as floats in degrees"],
                )
            if not (-180.0 <= angle_value <= 180.0):
                return OperationResult.error_result(
                    "ANGLE_OUT_OF_RANGE",
                    f"{angle_name}={angle_value} out of range [-180, 180]",
                    ["Keep angles within [-180, 180] degrees"],
                )

        def _ros_path():
            from ros2.ROSBridge import ROSBridge

            bridge = ROSBridge.get_instance()
            result = bridge.plan_orientation_change(
                orientation={"roll": roll, "pitch": pitch, "yaw": yaw},
                robot_id=robot_id,
            )
            if result and result.get("success"):
                logger.info(f"ROS orientation adjustment completed for {robot_id}")
                return OperationResult.success_result(
                    {
                        "robot_id": robot_id,
                        "orientation": {"roll": roll, "pitch": pitch, "yaw": yaw},
                        "status": "ros_executed",
                        "planning_time": result.get("planning_time", 0),
                        "timestamp": time.time(),
                    }
                )
            return None

        def _tcp_path():
            command = {
                "command_type": "adjust_end_effector_orientation",
                "robot_id": robot_id,
                "parameters": {"roll": roll, "pitch": pitch, "yaw": yaw},
                "timestamp": time.time(),
                "request_id": request_id,
            }
            logger.info(
                f"Sending adjust_end_effector_orientation to {robot_id}: "
                f"roll={roll}, pitch={pitch}, yaw={yaw}"
            )
            success = _get_command_broadcaster().send_command(command, request_id)
            if not success:
                return OperationResult.error_result(
                    "COMMUNICATION_FAILED",
                    "Failed to send command to Unity",
                    ["Ensure Unity is running", "Verify CommandServer is running"],
                )
            logger.info(f"Successfully sent orientation adjustment to {robot_id}")
            return OperationResult.success_result(
                {
                    "robot_id": robot_id,
                    "orientation": {"roll": roll, "pitch": pitch, "yaw": yaw},
                    "status": "command_sent",
                    "timestamp": time.time(),
                }
            )

        return execute_with_ros_fallback(_ros_path, _tcp_path, use_ros)

    except Exception as e:
        logger.error(
            f"Unexpected error in adjust_end_effector_orientation: {e}", exc_info=True
        )
        return OperationResult.error_result(
            "UNEXPECTED_ERROR",
            f"Unexpected error occurred: {str(e)}",
            ["Check logs", "Retry operation"],
        )


def create_adjust_end_effector_orientation_operation() -> BasicOperation:
    """Create the BasicOperation definition for adjust_end_effector_orientation."""
    return BasicOperation(
        operation_id="motion_adjust_orientation_003",
        name="adjust_end_effector_orientation",
        category=OperationCategory.NAVIGATION,
        complexity=OperationComplexity.BASIC,
        description="Adjust end effector orientation (roll, pitch, yaw) without changing position",
        long_description="""
            This operation modifies only the gripper orientation while maintaining
            the current position. Useful for adjusting grasp approach angle or
            tool orientation.

            Rotation order: Roll (X) → Pitch (Y) → Yaw (Z)
        """,
        usage_examples=[
            "adjust_end_effector_orientation('Robot1', roll=90.0) - Side grasp",
            "adjust_end_effector_orientation('Robot1', pitch=-90.0) - Top-down grasp",
            "adjust_end_effector_orientation('Robot1', yaw=45.0) - Angled approach",
        ],
        parameters=[
            OperationParameter(
                name="robot_id", type="str", description="Robot ID", required=True
            ),
            OperationParameter(
                name="roll",
                type="float",
                description="Roll angle in degrees (X axis)",
                required=False,
                default=0.0,
                valid_range=(-180.0, 180.0),
            ),
            OperationParameter(
                name="pitch",
                type="float",
                description="Pitch angle in degrees (Y axis)",
                required=False,
                default=0.0,
                valid_range=(-180.0, 180.0),
            ),
            OperationParameter(
                name="yaw",
                type="float",
                description="Yaw angle in degrees (Z axis)",
                required=False,
                default=0.0,
                valid_range=(-180.0, 180.0),
            ),
        ],
        preconditions=[
            "robot_is_initialized(robot_id)",
        ],
        postconditions=[],
        average_duration_ms=800.0,
        success_rate=0.96,
        failure_modes=["Unreachable orientation", "Joint limits exceeded"],
        implementation=adjust_end_effector_orientation,
    )


ADJUST_END_EFFECTOR_ORIENTATION_OPERATION = (
    create_adjust_end_effector_orientation_operation()
)


# ============================================================================
# Implementation: Pick Object at Coordinate (Approach → Descent → Grasp)
# ============================================================================


def pick_object_at_coordinate(
    robot_id: str,
    x: float,
    y: float,
    z: float,
    approach_height: float = 0.10,
    speed: float = 0.5,
    request_id: int = 0,
    use_ros: Optional[bool] = None,
) -> OperationResult:
    """
    Pick an object at a known 3D coordinate using a hover → descent → grasp sequence.

    This operation encodes the correct three-step pick pattern that prevents the
    gripper from closing while still above the object:

    1. Open gripper (ensure fingers clear the object during approach).
    2. Move to hover position (target + approach_height above along Unity Y).
    3. Descend straight down to contact position (approach_offset=0).
    4. Close gripper.

    Use this instead of manually chaining move_to_coordinate + control_gripper,
    which causes the gripper to fire 10 cm above the cube.  For object-name-based
    grasping use grasp_object instead (it runs the full GraspPlanningPipeline).

    Args:
        robot_id: ID of the robot to control (e.g., "Robot1", "AR4_Robot").
        x: X coordinate of the object centre in metres.
        y: Y coordinate of the object centre in metres (Unity Y = up).
        z: Z coordinate of the object centre in metres.
        approach_height: Distance above the object for the hover position in metres,
            range: [0.02, 0.20].  Default 0.10 m (10 cm).
        speed: Speed multiplier used for both moves, range: [0.1, 2.0].
        request_id: Optional request ID for tracking.
        use_ros: Whether to use ROS for motion planning (None = auto-detect).

    Returns:
        OperationResult indicating success or the first failure in the sequence.

    Example:
        >>> # Pick a cube sitting at world position (0.3, 0.05, 0.1)
        >>> result = pick_object_at_coordinate("Robot1", 0.3, 0.05, 0.1)

        >>> # Pick with a taller approach clearance
        >>> result = pick_object_at_coordinate("Robot1", 0.3, 0.05, 0.1, approach_height=0.15)
    """
    try:
        if err := validate_robot_id(robot_id):
            return err
        if err := validate_xyz(x, y, z):
            return err
        if err := validate_speed(speed):
            return err
        if not isinstance(approach_height, (int, float)) or not (
            0.02 <= approach_height <= 0.20
        ):
            return OperationResult.error_result(
                "INVALID_APPROACH_HEIGHT",
                f"approach_height must be between 0.02 and 0.20 m, got: {approach_height}",
                ["Use a value between 0.02 m (2 cm) and 0.20 m (20 cm)"],
            )

        # Import here to avoid circular dependency — GripperOperations → Base only.
        try:
            from operations.GripperOperations import control_gripper
        except ImportError:
            from .GripperOperations import control_gripper

        # Step 1: Open gripper so fingers don't knock the object during approach.
        open_result = control_gripper(
            robot_id=robot_id,
            open_gripper=True,
            request_id=request_id,
            use_ros=use_ros,
        )
        if not open_result["success"]:
            return open_result

        # Step 2: Move to hover position (approach_height above object).
        hover_result = move_to_coordinate(
            robot_id=robot_id,
            x=x,
            y=y,
            z=z,
            speed=speed,
            approach_offset=approach_height,
            request_id=request_id,
            use_ros=use_ros,
        )
        if not hover_result["success"]:
            return hover_result

        # Step 3: Descend straight to contact position (approach_offset=0).
        descent_result = move_to_coordinate(
            robot_id=robot_id,
            x=x,
            y=y,
            z=z,
            speed=min(speed, 0.3),  # Slow down for the final contact move.
            approach_offset=0.0,
            request_id=request_id,
            use_ros=use_ros,
        )
        if not descent_result["success"]:
            return descent_result

        # Step 4: Close gripper to grasp the object.
        close_result = control_gripper(
            robot_id=robot_id,
            open_gripper=False,
            request_id=request_id,
            use_ros=use_ros,
        )
        if not close_result["success"]:
            return close_result

        logger.info(
            f"pick_object_at_coordinate: {robot_id} successfully picked at "
            f"({x:.3f}, {y:.3f}, {z:.3f})"
        )
        return OperationResult.success_result(
            {
                "robot_id": robot_id,
                "target_position": {"x": x, "y": y, "z": z},
                "approach_height": approach_height,
                "speed": speed,
                "status": "picked",
                "timestamp": __import__("time").time(),
            }
        )

    except Exception as e:
        logger.error(f"Unexpected error in pick_object_at_coordinate: {e}", exc_info=True)
        return OperationResult.error_result(
            "UNEXPECTED_ERROR",
            f"Unexpected error occurred: {str(e)}",
            [
                "Check logs for detailed error information",
                "Verify all parameters are correct types",
                "Retry the operation",
            ],
        )


def create_pick_object_at_coordinate_operation() -> BasicOperation:
    """Create the BasicOperation definition for pick_object_at_coordinate."""
    return BasicOperation(
        operation_id="motion_pick_at_coord_004",
        name="pick_object_at_coordinate",
        category=OperationCategory.NAVIGATION,
        complexity=OperationComplexity.INTERMEDIATE,
        description=(
            "Pick an object at a known 3D coordinate using hover → descent → grasp sequence"
        ),
        long_description="""
            Encodes the correct three-step pick pattern:
            1. Open gripper (clear fingers during approach)
            2. Move to hover position (approach_height above the object)
            3. Descend straight down to contact position
            4. Close gripper

            Use this instead of manually chaining move_to_coordinate + control_gripper.
            That naive pattern closes the gripper while the arm is still approach_height
            above the object, missing the cube entirely.

            For picking by object name (with full GraspPlanningPipeline, IK validation,
            and collision filtering) use grasp_object instead.
        """,
        usage_examples=[
            "pick_object_at_coordinate('Robot1', 0.3, 0.05, 0.1) - Pick cube at known coords",
            "pick_object_at_coordinate('Robot1', x, y, z, approach_height=0.15) - Taller clearance",
            "Use after detect_object_stereo returns a position to pick without object-name lookup",
        ],
        parameters=[
            OperationParameter(
                name="robot_id", type="str", description="Robot ID", required=True
            ),
            OperationParameter(
                name="x",
                type="float",
                description="X coordinate of object centre in metres",
                required=True,
                valid_range=(-0.65, 0.65),
            ),
            OperationParameter(
                name="y",
                type="float",
                description="Y coordinate of object centre in metres (Unity Y = up)",
                required=True,
                valid_range=(0.0, 0.7),
            ),
            OperationParameter(
                name="z",
                type="float",
                description="Z coordinate of object centre in metres",
                required=True,
                valid_range=(-0.5, 0.5),
            ),
            OperationParameter(
                name="approach_height",
                type="float",
                description="Height above object for hover position in metres (default 0.10)",
                required=False,
                default=0.10,
                valid_range=(0.02, 0.20),
            ),
            OperationParameter(
                name="speed",
                type="float",
                description="Speed multiplier (0.1=slow, 1.0=normal)",
                required=False,
                default=0.5,
                valid_range=(0.1, 2.0),
            ),
        ],
        preconditions=[
            "robot_is_initialized(robot_id)",
            "target_within_reach(robot_id, x, y, z)",
        ],
        postconditions=["gripper_holding_object(robot_id)"],
        average_duration_ms=3600.0,
        success_rate=0.90,
        failure_modes=[
            "Object not at specified coordinates (use grasp_object for name-based picking)",
            "Descent position unreachable",
            "Gripper fails to close (object not in contact)",
            "Communication failed - Unity not connected",
        ],
        relationships=OperationRelationship(
            operation_id="motion_pick_at_coord_004",
            required_operations=["status_check_robot_001"],
            required_reasons={
                "status_check_robot_001": "Verify robot is ready before executing multi-step pick",
            },
            commonly_paired_with=[
                "perception_stereo_detect_001",
                "motion_move_to_coord_001",
            ],
            pairing_reasons={
                "perception_stereo_detect_001": "Detect object position first, then pick at detected coords",
                "motion_move_to_coord_001": "Use move_to_coordinate for navigation; pick_object_at_coordinate for grasping",
            },
            typical_before=["motion_move_to_coord_001"],
            typical_after=["perception_stereo_detect_001"],
        ),
        implementation=pick_object_at_coordinate,
    )


PICK_OBJECT_AT_COORDINATE_OPERATION = create_pick_object_at_coordinate_operation()
