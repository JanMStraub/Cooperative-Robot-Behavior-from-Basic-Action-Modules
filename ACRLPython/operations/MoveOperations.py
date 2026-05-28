#!/usr/bin/env python3
"""Movement operations for controlling the robot arm through Unity's RobotController via TCP."""

import time
import logging
from typing import Optional

# Lazy import to avoid circular dependency with servers module
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

from core.LoggingSetup import setup_logging

setup_logging(__name__)
logger = logging.getLogger(__name__)


from ._imports import get_command_broadcaster as _get_command_broadcaster


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
    """Move robot EE to specified 3D coordinate. Async — returns immediately, Unity executes in background. approach_offset lifts target along Y."""
    try:
        if err := validate_robot_id(robot_id):
            return err
        if err := validate_xyz(x, y, z):
            return err
        if err := validate_speed(speed):
            return err
        if err := validate_approach_offset(approach_offset):
            return err

        actual_x = x
        actual_y = y + approach_offset  # Add offset to height (Unity Y = up)
        actual_z = z

        def _ros_path():
            from ros2.ROSBridge import ROSBridge

            bridge = ROSBridge.get_instance()
            result = bridge.plan_and_execute(
                position={"x": actual_x, "y": actual_y, "z": actual_z},
                robot_id=robot_id,
                coordinate_space="unity_world",  # LLM generates Unity world coords (Y-up); transform to base_link applied in ROSMotionClient
            )
            if result and result.get("success"):
                logger.info(f"ROS motion completed for {robot_id}")
                # ROS path bypasses Unity's SetTarget, so _targetTransform stays null
                # and WorldState would publish (0,0,0) as target_position.  Write the
                # confirmed target explicitly so operations like receive_handoff that
                # read get_robot_target() get the correct position.
                try:
                    from ._imports import get_world_state

                    ws = get_world_state()
                    if ws is not None:
                        ws.update_robot_state(
                            robot_id,
                            {
                                "target_position": {
                                    "x": actual_x,
                                    "y": actual_y,
                                    "z": actual_z,
                                }
                            },
                        )
                except Exception as _e:
                    logger.debug(
                        f"move_to_coordinate: could not write target to WorldState: {_e}"
                    )
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
                        "Verify CommandServer is running (port 5007)",
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


def create_move_to_coordinate_operation() -> BasicOperation:
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
            "Navigate to detected object coordinates (no gripper): move_to_coordinate(robot_id='Robot1', x=0.3, y=0.15, z=0.1)",
            "Move slowly to precise position: move_to_coordinate(robot_id='Robot1', x=0.2, y=0.1, z=0.15, speed=0.2)",
            "Hover 5cm above a target without grasping: move_to_coordinate(robot_id='Robot1', x=0.3, y=0.0, z=0.1, approach_offset=0.05)",
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
                description="Lift above target by this many meters along Unity Y (up-axis). Only set this when an explicit hover-then-descend pattern is needed; leave at 0.0 for pure navigation.",
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
                    "Only pair with control_gripper when the task EXPLICITLY asks to open or close the gripper. "
                    "Pure navigation tasks ('move to X', 'detect and move to it') must NOT include any gripper operation. "
                    "If grasping is needed, use pick_object_at_coordinate (coords) or grasp_object (by name) instead — "
                    "never manually chain move_to_coordinate + control_gripper for picking."
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


MOVE_TO_COORDINATE_OPERATION = create_move_to_coordinate_operation()


def adjust_end_effector_orientation(
    robot_id: str,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    request_id: int = 0,
    use_ros: Optional[bool] = None,
) -> OperationResult:
    """Adjust end effector orientation (roll/pitch/yaw) without changing position."""
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
    """Pick at known coords: open → hover → descend → close gripper. Don't manually chain move+control_gripper — use this instead."""
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

        # GripperOperations → Base only (no circular dep).
        try:
            from operations.GripperOperations import control_gripper
        except ImportError:
            from .GripperOperations import control_gripper

        open_result = control_gripper(
            robot_id=robot_id,
            open_gripper=True,
            request_id=request_id,
            use_ros=use_ros,
        )
        if not open_result["success"]:
            return open_result

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
        logger.error(
            f"Unexpected error in pick_object_at_coordinate: {e}", exc_info=True
        )
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
