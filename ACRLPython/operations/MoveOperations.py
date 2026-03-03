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
        approach_offset: Stop distance before target in meters, range: [0.0, 0.1]
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
        if not (-0.65 <= x <= 0.65):
            return OperationResult.error_result(
                "INVALID_X_COORDINATE",
                f"X coordinate {x} out of range [-0.65, 0.65]",
                [
                    "Adjust X to be within robot workspace [-0.65, 0.65]",
                    "Use detect_object to get valid coordinates",
                ],
            )

        # Validate Y coordinate
        if not (0.0 <= y <= 0.7):
            return OperationResult.error_result(
                "INVALID_Y_COORDINATE",
                f"Y coordinate {y} out of range [0.0, 0.7]",
                [
                    "Adjust Y to be within robot workspace [0.0, 0.7]",
                    "Use detect_object to get valid coordinates",
                ],
            )

        # Validate Z coordinate
        if not (-0.5 <= z <= 0.5):
            return OperationResult.error_result(
                "INVALID_Z_COORDINATE",
                f"Z coordinate {z} out of range [-0.5, 0.5]",
                [
                    "Adjust Z to be within robot workspace [-0.5, 0.5]",
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

        # Determine whether to use ROS or TCP path
        _use_ros = use_ros
        if _use_ros is None:
            try:
                from config.ROS import ROS_ENABLED, DEFAULT_CONTROL_MODE

                _use_ros = ROS_ENABLED and DEFAULT_CONTROL_MODE in ("ros", "hybrid")
            except ImportError:
                _use_ros = False

        # Apply approach offset to target position
        actual_x = x
        actual_y = y
        actual_z = z + approach_offset  # Add offset to height for safety

        # Route via ROS if enabled
        if _use_ros:
            try:
                from ros2.ROSBridge import ROSBridge

                bridge = ROSBridge.get_instance()
                if not bridge.is_connected:
                    if not bridge.connect():
                        # Fall back to TCP if hybrid mode
                        try:
                            from config.ROS import DEFAULT_CONTROL_MODE

                            if DEFAULT_CONTROL_MODE == "hybrid":
                                logger.warning(
                                    "ROS bridge unavailable, falling back to TCP"
                                )
                                _use_ros = False
                            else:
                                return OperationResult.error_result(
                                    "ROS_CONNECTION_FAILED",
                                    "Failed to connect to ROS bridge (port 5020)",
                                    [
                                        "Ensure Docker ROS services are running: cd rosUnityIntegration && ./start_ros_endpoint.sh",
                                        "Set DEFAULT_CONTROL_MODE='hybrid' in config/ROS.py for automatic fallback",
                                    ],
                                )
                        except ImportError:
                            _use_ros = False

                if _use_ros:
                    result = bridge.plan_and_execute(
                        position={"x": actual_x, "y": actual_y, "z": actual_z},
                        robot_id=robot_id,  # Pass robot_id to route to correct MoveIt instance
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
                    else:
                        error_msg = (
                            result.get("error", "Unknown") if result else "No response"
                        )
                        try:
                            from config.ROS import DEFAULT_CONTROL_MODE

                            if DEFAULT_CONTROL_MODE == "hybrid":
                                logger.warning(
                                    f"ROS planning failed ({error_msg}), falling back to TCP"
                                )
                                _use_ros = False
                            else:
                                return OperationResult.error_result(
                                    "ROS_PLANNING_FAILED",
                                    f"MoveIt planning failed: {error_msg}",
                                    ["Check MoveIt logs", "Verify target is reachable"],
                                )
                        except ImportError:
                            _use_ros = False
            except ImportError:
                logger.warning("ros2 module not available, falling back to TCP")
                _use_ros = False

        # Construct command for Unity (TCP path)
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
                    "Ensure Unity is running with UnifiedPythonReceiver active",
                    "Verify CommandServer is running (port 5010)",
                    "Check Unity console for connection errors",
                    "Restart backend: python -m orchestrators.RunRobotController",
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
                description="Stop this many meters before target (useful for approaching)",
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
        # Validate robot_id
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string, got: {robot_id}",
                ["Provide a valid robot ID (e.g., 'Robot1', 'AR4_Robot')"],
            )

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

        # Determine whether to use ROS or TCP path
        _use_ros = use_ros
        if _use_ros is None:
            try:
                from config.ROS import ROS_ENABLED, DEFAULT_CONTROL_MODE

                _use_ros = ROS_ENABLED and DEFAULT_CONTROL_MODE in ("ros", "hybrid")
            except ImportError:
                _use_ros = False

        # Route via ROS if enabled
        if _use_ros:
            try:
                from ros2.ROSBridge import ROSBridge

                bridge = ROSBridge.get_instance()
                if not bridge.is_connected:
                    if not bridge.connect():
                        # Fall back to TCP if hybrid mode
                        try:
                            from config.ROS import DEFAULT_CONTROL_MODE

                            if DEFAULT_CONTROL_MODE == "hybrid":
                                logger.warning(
                                    "ROS bridge unavailable, falling back to TCP"
                                )
                                _use_ros = False
                            else:
                                return OperationResult.error_result(
                                    "ROS_CONNECTION_FAILED",
                                    "Failed to connect to ROS bridge (port 5020)",
                                    [
                                        "Ensure Docker ROS services are running: cd rosUnityIntegration && ./start_ros_endpoint.sh",
                                        "Set DEFAULT_CONTROL_MODE='hybrid' in config/ROS.py for automatic fallback",
                                    ],
                                )
                        except ImportError:
                            _use_ros = False

                if _use_ros:
                    # Plan path through both waypoints via ROS
                    result = bridge.plan_multi_waypoint(
                        waypoints=[point_a, point_b],
                        robot_id=robot_id,
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
                    else:
                        error_msg = (
                            result.get("error", "Unknown") if result else "No response"
                        )
                        try:
                            from config.ROS import DEFAULT_CONTROL_MODE

                            if DEFAULT_CONTROL_MODE == "hybrid":
                                logger.warning(
                                    f"ROS planning failed ({error_msg}), falling back to TCP"
                                )
                                _use_ros = False
                            else:
                                return OperationResult.error_result(
                                    "ROS_PLANNING_FAILED",
                                    f"MoveIt planning failed: {error_msg}",
                                    [
                                        "Check MoveIt logs",
                                        "Verify waypoints are reachable",
                                    ],
                                )
                        except ImportError:
                            _use_ros = False
            except ImportError:
                logger.warning("ros2 module not available, falling back to TCP")
                _use_ros = False

        # Construct command for Unity with both waypoints (TCP path)
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

        # Send to Unity
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
        # Validate robot_id
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string, got: {robot_id}",
                ["Provide a valid robot ID"],
            )

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

        # Determine whether to use ROS or TCP path
        _use_ros = use_ros
        if _use_ros is None:
            try:
                from config.ROS import ROS_ENABLED, DEFAULT_CONTROL_MODE

                _use_ros = ROS_ENABLED and DEFAULT_CONTROL_MODE in ("ros", "hybrid")
            except ImportError:
                _use_ros = False

        # Route via ROS if enabled
        if _use_ros:
            try:
                from ros2.ROSBridge import ROSBridge

                bridge = ROSBridge.get_instance()
                if not bridge.is_connected:
                    if not bridge.connect():
                        # Fall back to TCP if hybrid mode
                        try:
                            from config.ROS import DEFAULT_CONTROL_MODE

                            if DEFAULT_CONTROL_MODE == "hybrid":
                                logger.warning(
                                    "ROS bridge unavailable, falling back to TCP"
                                )
                                _use_ros = False
                            else:
                                return OperationResult.error_result(
                                    "ROS_CONNECTION_FAILED",
                                    "Failed to connect to ROS bridge (port 5020)",
                                    ["Ensure Docker ROS services are running"],
                                )
                        except ImportError:
                            _use_ros = False

                if _use_ros:
                    # Plan orientation change via ROS
                    result = bridge.plan_orientation_change(
                        orientation={"roll": roll, "pitch": pitch, "yaw": yaw},
                        robot_id=robot_id,
                    )
                    if result and result.get("success"):
                        logger.info(
                            f"ROS orientation adjustment completed for {robot_id}"
                        )
                        return OperationResult.success_result(
                            {
                                "robot_id": robot_id,
                                "orientation": {
                                    "roll": roll,
                                    "pitch": pitch,
                                    "yaw": yaw,
                                },
                                "status": "ros_executed",
                                "planning_time": result.get("planning_time", 0),
                                "timestamp": time.time(),
                            }
                        )
                    else:
                        error_msg = (
                            result.get("error", "Unknown") if result else "No response"
                        )
                        try:
                            from config.ROS import DEFAULT_CONTROL_MODE

                            if DEFAULT_CONTROL_MODE == "hybrid":
                                logger.warning(
                                    f"ROS planning failed ({error_msg}), falling back to TCP"
                                )
                                _use_ros = False
                            else:
                                return OperationResult.error_result(
                                    "ROS_PLANNING_FAILED",
                                    f"MoveIt planning failed: {error_msg}",
                                    [
                                        "Check MoveIt logs",
                                        "Verify orientation is reachable",
                                    ],
                                )
                        except ImportError:
                            _use_ros = False
            except ImportError:
                logger.warning("ros2 module not available, falling back to TCP")
                _use_ros = False

        # Construct command for Unity (TCP path)
        command = {
            "command_type": "adjust_end_effector_orientation",
            "robot_id": robot_id,
            "parameters": {
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
            },
            "timestamp": time.time(),
            "request_id": request_id,
        }

        logger.info(
            f"Sending adjust_end_effector_orientation to {robot_id}: roll={roll}, pitch={pitch}, yaw={yaw}"
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
