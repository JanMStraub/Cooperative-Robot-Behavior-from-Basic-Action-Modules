"""
Intermediate Operations (Level 3)
==================================

This module implements Level 3 operations from the thesis exposé:
- grip_object: Simplified grasping wrapper
- align_object: Object/gripper alignment
- follow_path: Multi-waypoint trajectory following
- draw_with_pen: Tool manipulation for drawing

These operations provide higher-level manipulation capabilities building
on the basic atomic operations.
"""

import time
import logging
from typing import List, Dict, Any, Optional

# Import from centralized lazy import system
try:
    from ..core.Imports import get_command_broadcaster as _get_command_broadcaster
except ImportError:
    from core.Imports import get_command_broadcaster as _get_command_broadcaster

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
# Implementation: grip_object - Simplified grasping
# ============================================================================


def grip_object(
    robot_id: str,
    object_position: Dict[str, float],
    approach_direction: str = "top",
    request_id: int = 0,
    use_ros: bool = None,
) -> OperationResult:
    """
    Simplified grasping operation - wrapper around grasp_object.

    This provides a simpler interface for grasping compared to the full
    grasp_object operation which includes grasp planning pipeline.

    Args:
        robot_id: Robot identifier
        object_position: Target position dict with 'x', 'y', 'z' keys (meters)
        approach_direction: Approach direction ("top", "front", "side")
        request_id: Optional request ID for tracking
        use_ros: Whether to use ROS for motion planning (None = auto-detect from config)

    Returns:
        OperationResult with grasp confirmation

    Example:
        >>> # Grip object at detected position
        >>> result = grip_object("Robot1", {"x": 0.3, "y": 0.15, "z": 0.1})
    """
    try:
        # Validate robot_id
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string, got: {robot_id}",
                ["Provide a valid robot ID"],
            )

        # Validate object_position
        if not isinstance(object_position, dict) or not all(
            k in object_position for k in ["x", "y", "z"]
        ):
            return OperationResult.error_result(
                "INVALID_OBJECT_POSITION",
                f"object_position must have x, y, z keys, got: {object_position}",
                ["Provide as: {'x': 0.3, 'y': 0.15, 'z': 0.1}"],
            )

        # Validate approach_direction
        valid_directions = ["top", "front", "side"]
        if approach_direction not in valid_directions:
            return OperationResult.error_result(
                "INVALID_APPROACH_DIRECTION",
                f"approach_direction must be one of {valid_directions}, got: {approach_direction}",
                [f"Use one of: {', '.join(valid_directions)}"],
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
                                logger.warning("ROS bridge unavailable, falling back to TCP")
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
                    result = bridge.plan_and_execute(
                        position=object_position,
                        robot_id=robot_id,
                    )
                    if result and result.get("success"):
                        logger.info(f"ROS grip motion completed for {robot_id}")
                        return OperationResult.success_result({
                            "robot_id": robot_id,
                            "object_position": object_position,
                            "approach_direction": approach_direction,
                            "status": "ros_executed",
                            "planning_time": result.get("planning_time", 0),
                            "timestamp": time.time(),
                        })
                    else:
                        error_msg = result.get("error", "Unknown") if result else "No response"
                        try:
                            from config.ROS import DEFAULT_CONTROL_MODE
                            if DEFAULT_CONTROL_MODE == "hybrid":
                                logger.warning(f"ROS planning failed ({error_msg}), falling back to TCP")
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

        # Construct simplified grasp command (TCP path)
        command = {
            "command_type": "grip_object",
            "robot_id": robot_id,
            "parameters": {
                "object_position": object_position,
                "approach_direction": approach_direction,
            },
            "timestamp": time.time(),
            "request_id": request_id,
        }

        logger.info(
            f"Sending grip_object command to {robot_id} at position {object_position}"
        )

        success = _get_command_broadcaster().send_command(command, request_id)

        if not success:
            return OperationResult.error_result(
                "COMMUNICATION_FAILED",
                "Failed to send command to Unity",
                ["Ensure Unity is running", "Verify CommandServer is running"],
            )

        logger.info(f"Successfully sent grip_object command to {robot_id}")

        return OperationResult.success_result(
            {
                "robot_id": robot_id,
                "object_position": object_position,
                "approach_direction": approach_direction,
                "status": "command_sent",
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.error(f"Unexpected error in grip_object: {e}", exc_info=True)
        return OperationResult.error_result(
            "UNEXPECTED_ERROR",
            f"Unexpected error occurred: {str(e)}",
            ["Check logs", "Retry operation"],
        )


# ============================================================================
# Implementation: align_object - Object/gripper alignment
# ============================================================================


def align_object(
    robot_id: str,
    target_orientation: Dict[str, float],
    alignment_type: str = "gripper",
    request_id: int = 0,
    use_ros: bool = None,
) -> OperationResult:
    """
    Align object or gripper to target orientation.

    This operation adjusts the gripper orientation to align with an object
    or aligns a held object to a target orientation.

    Args:
        robot_id: Robot identifier
        target_orientation: Target orientation dict with 'roll', 'pitch', 'yaw' (degrees)
        alignment_type: Type of alignment ("gripper" or "object")
        request_id: Optional request ID for tracking
        use_ros: Whether to use ROS for motion planning (None = auto-detect from config)

    Returns:
        OperationResult with alignment confirmation

    Example:
        >>> # Align gripper for vertical insertion
        >>> result = align_object("Robot1", {"roll": 0, "pitch": -90, "yaw": 0})

        >>> # Align held object to horizontal orientation
        >>> result = align_object("Robot1", {"roll": 0, "pitch": 0, "yaw": 0}, "object")
    """
    try:
        # Validate robot_id
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string",
                ["Provide a valid robot ID"],
            )

        # Validate target_orientation
        if not isinstance(target_orientation, dict) or not all(
            k in target_orientation for k in ["roll", "pitch", "yaw"]
        ):
            return OperationResult.error_result(
                "INVALID_ORIENTATION",
                f"target_orientation must have roll, pitch, yaw keys",
                ["Provide as: {'roll': 0, 'pitch': -90, 'yaw': 0}"],
            )

        # Validate alignment_type
        valid_types = ["gripper", "object"]
        if alignment_type not in valid_types:
            return OperationResult.error_result(
                "INVALID_ALIGNMENT_TYPE",
                f"alignment_type must be 'gripper' or 'object', got: {alignment_type}",
                ["Use 'gripper' to align gripper, 'object' to align held object"],
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
                                logger.warning("ROS bridge unavailable, falling back to TCP")
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
                    result = bridge.plan_orientation_change(
                        orientation=target_orientation,
                        robot_id=robot_id,
                    )
                    if result and result.get("success"):
                        logger.info(f"ROS alignment completed for {robot_id}")
                        return OperationResult.success_result({
                            "robot_id": robot_id,
                            "target_orientation": target_orientation,
                            "alignment_type": alignment_type,
                            "status": "ros_executed",
                            "planning_time": result.get("planning_time", 0),
                            "timestamp": time.time(),
                        })
                    else:
                        error_msg = result.get("error", "Unknown") if result else "No response"
                        try:
                            from config.ROS import DEFAULT_CONTROL_MODE
                            if DEFAULT_CONTROL_MODE == "hybrid":
                                logger.warning(f"ROS planning failed ({error_msg}), falling back to TCP")
                                _use_ros = False
                            else:
                                return OperationResult.error_result(
                                    "ROS_PLANNING_FAILED",
                                    f"MoveIt planning failed: {error_msg}",
                                    ["Check MoveIt logs", "Verify orientation is reachable"],
                                )
                        except ImportError:
                            _use_ros = False
            except ImportError:
                logger.warning("ros2 module not available, falling back to TCP")
                _use_ros = False

        # Construct command (TCP path)
        command = {
            "command_type": "align_object",
            "robot_id": robot_id,
            "parameters": {
                "target_orientation": target_orientation,
                "alignment_type": alignment_type,
            },
            "timestamp": time.time(),
            "request_id": request_id,
        }

        logger.info(
            f"Sending align_object command to {robot_id}: {target_orientation}"
        )

        success = _get_command_broadcaster().send_command(command, request_id)

        if not success:
            return OperationResult.error_result(
                "COMMUNICATION_FAILED",
                "Failed to send command to Unity",
                ["Ensure Unity is running"],
            )

        logger.info(f"Successfully sent align_object command to {robot_id}")

        return OperationResult.success_result(
            {
                "robot_id": robot_id,
                "target_orientation": target_orientation,
                "alignment_type": alignment_type,
                "status": "command_sent",
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.error(f"Unexpected error in align_object: {e}", exc_info=True)
        return OperationResult.error_result(
            "UNEXPECTED_ERROR",
            f"Unexpected error occurred: {str(e)}",
            ["Check logs"],
        )


# ============================================================================
# Implementation: follow_path - Multi-waypoint trajectory
# ============================================================================


def follow_path(
    robot_id: str,
    waypoints: List[Dict[str, float]],
    speed: float = 1.0,
    request_id: int = 0,
    use_ros: bool = None,
) -> OperationResult:
    """
    Follow a multi-waypoint trajectory.

    This operation commands the robot to move through a series of waypoints
    in sequence, creating a smooth trajectory.

    Args:
        robot_id: Robot identifier
        waypoints: List of waypoint dicts, each with 'x', 'y', 'z' keys (meters)
        speed: Speed multiplier (0.1-2.0)
        request_id: Optional request ID for tracking
        use_ros: Whether to use ROS for motion planning (None = auto-detect from config)

    Returns:
        OperationResult with path execution confirmation

    Example:
        >>> # Follow path with 3 waypoints
        >>> waypoints = [
        ...     {"x": 0.0, "y": 0.0, "z": 0.3},  # Start
        ...     {"x": 0.2, "y": 0.1, "z": 0.2},  # Middle
        ...     {"x": 0.3, "y": 0.15, "z": 0.1}, # End
        ... ]
        >>> result = follow_path("Robot1", waypoints)
    """
    try:
        # Validate robot_id
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string",
                ["Provide a valid robot ID"],
            )

        # Validate waypoints
        if not isinstance(waypoints, list) or len(waypoints) < 2:
            return OperationResult.error_result(
                "INVALID_WAYPOINTS",
                f"waypoints must be a list with at least 2 points, got: {len(waypoints) if isinstance(waypoints, list) else 'not a list'}",
                ["Provide at least 2 waypoints for a valid path"],
            )

        # Validate each waypoint
        for i, wp in enumerate(waypoints):
            if not isinstance(wp, dict) or not all(k in wp for k in ["x", "y", "z"]):
                return OperationResult.error_result(
                    "INVALID_WAYPOINT",
                    f"Waypoint {i} must have x, y, z keys, got: {wp}",
                    ["Each waypoint must be: {'x': float, 'y': float, 'z': float}"],
                )

        # Validate speed
        if not (0.1 <= speed <= 2.0):
            return OperationResult.error_result(
                "INVALID_SPEED",
                f"Speed must be in range [0.1, 2.0], got: {speed}",
                ["Use speed between 0.1 (slow) and 2.0 (fast)"],
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
                                logger.warning("ROS bridge unavailable, falling back to TCP")
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
                    result = bridge.plan_multi_waypoint(
                        waypoints=waypoints,
                        robot_id=robot_id,
                    )
                    if result and result.get("success"):
                        logger.info(f"ROS multi-waypoint path completed for {robot_id}")
                        return OperationResult.success_result({
                            "robot_id": robot_id,
                            "waypoints": waypoints,
                            "waypoint_count": len(waypoints),
                            "speed": speed,
                            "status": "ros_executed",
                            "planning_time": result.get("planning_time", 0),
                            "timestamp": time.time(),
                        })
                    else:
                        error_msg = result.get("error", "Unknown") if result else "No response"
                        try:
                            from config.ROS import DEFAULT_CONTROL_MODE
                            if DEFAULT_CONTROL_MODE == "hybrid":
                                logger.warning(f"ROS planning failed ({error_msg}), falling back to TCP")
                                _use_ros = False
                            else:
                                return OperationResult.error_result(
                                    "ROS_PLANNING_FAILED",
                                    f"MoveIt planning failed: {error_msg}",
                                    ["Check MoveIt logs", "Verify all waypoints are reachable"],
                                )
                        except ImportError:
                            _use_ros = False
            except ImportError:
                logger.warning("ros2 module not available, falling back to TCP")
                _use_ros = False

        # Construct command (TCP path)
        command = {
            "command_type": "follow_path",
            "robot_id": robot_id,
            "parameters": {
                "waypoints": waypoints,
                "speed_multiplier": speed,
            },
            "timestamp": time.time(),
            "request_id": request_id,
        }

        logger.info(
            f"Sending follow_path command to {robot_id} with {len(waypoints)} waypoints"
        )

        success = _get_command_broadcaster().send_command(command, request_id)

        if not success:
            return OperationResult.error_result(
                "COMMUNICATION_FAILED",
                "Failed to send command to Unity",
                ["Ensure Unity is running"],
            )

        logger.info(f"Successfully sent follow_path command to {robot_id}")

        return OperationResult.success_result(
            {
                "robot_id": robot_id,
                "waypoints": waypoints,
                "waypoint_count": len(waypoints),
                "speed": speed,
                "status": "command_sent",
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.error(f"Unexpected error in follow_path: {e}", exc_info=True)
        return OperationResult.error_result(
            "UNEXPECTED_ERROR",
            f"Unexpected error occurred: {str(e)}",
            ["Check logs"],
        )


# ============================================================================
# Implementation: draw_with_pen - Tool manipulation for drawing
# ============================================================================


def draw_with_pen(
    robot_id: str,
    pen_position: Dict[str, float],
    paper_position: Dict[str, float],
    shape: str = "line",
    shape_params: Optional[Dict[str, Any]] = None,
    request_id: int = 0,
    use_ros: bool = None,
) -> OperationResult:
    """
    Draw with a pen tool on paper surface.

    This operation commands the robot to use a pen tool to draw shapes
    on a paper surface. Demonstrates tool manipulation capability.

    Args:
        robot_id: Robot identifier
        pen_position: Position of pen tool with 'x', 'y', 'z' keys
        paper_position: Position of paper surface with 'x', 'y', 'z' keys
        shape: Shape to draw ("line", "circle", "square", "custom")
        shape_params: Optional shape-specific parameters
        request_id: Optional request ID for tracking
        use_ros: Whether to use ROS for motion planning (None = auto-detect from config)

    Returns:
        OperationResult with drawing execution confirmation

    Example:
        >>> # Draw a line on paper
        >>> result = draw_with_pen(
        ...     "Robot1",
        ...     {"x": 0.2, "y": 0.0, "z": 0.3},  # Pen location
        ...     {"x": 0.3, "y": 0.0, "z": 0.0},  # Paper location
        ...     shape="line",
        ...     shape_params={"length": 0.1, "angle": 0}
        ... )

        >>> # Draw a circle
        >>> result = draw_with_pen(
        ...     "Robot1",
        ...     pen_pos, paper_pos,
        ...     shape="circle",
        ...     shape_params={"radius": 0.05}
        ... )
    """
    try:
        # Validate robot_id
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string",
                ["Provide a valid robot ID"],
            )

        # Validate pen_position
        if not isinstance(pen_position, dict) or not all(
            k in pen_position for k in ["x", "y", "z"]
        ):
            return OperationResult.error_result(
                "INVALID_PEN_POSITION",
                f"pen_position must have x, y, z keys",
                ["Provide as: {'x': 0.2, 'y': 0.0, 'z': 0.3}"],
            )

        # Validate paper_position
        if not isinstance(paper_position, dict) or not all(
            k in paper_position for k in ["x", "y", "z"]
        ):
            return OperationResult.error_result(
                "INVALID_PAPER_POSITION",
                f"paper_position must have x, y, z keys",
                ["Provide as: {'x': 0.3, 'y': 0.0, 'z': 0.0}"],
            )

        # Validate shape
        valid_shapes = ["line", "circle", "square", "custom"]
        if shape not in valid_shapes:
            return OperationResult.error_result(
                "INVALID_SHAPE",
                f"shape must be one of {valid_shapes}, got: {shape}",
                [f"Use one of: {', '.join(valid_shapes)}"],
            )

        # Determine whether to use ROS or TCP path
        _use_ros = use_ros
        if _use_ros is None:
            try:
                from config.ROS import ROS_ENABLED, DEFAULT_CONTROL_MODE
                _use_ros = ROS_ENABLED and DEFAULT_CONTROL_MODE in ("ros", "hybrid")
            except ImportError:
                _use_ros = False

        # Route via ROS if enabled - draw operations involve complex paths
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
                                logger.warning("ROS bridge unavailable, falling back to TCP")
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
                    # For drawing, generate waypoints based on shape and use multi-waypoint planning
                    # This is a simplified approach - full implementation would generate shape-specific paths
                    waypoints = [pen_position, paper_position]  # Simplified: pick up pen, move to paper
                    result = bridge.plan_multi_waypoint(
                        waypoints=waypoints,
                        robot_id=robot_id,
                    )
                    if result and result.get("success"):
                        logger.info(f"ROS drawing motion completed for {robot_id}")
                        return OperationResult.success_result({
                            "robot_id": robot_id,
                            "pen_position": pen_position,
                            "paper_position": paper_position,
                            "shape": shape,
                            "shape_params": shape_params,
                            "status": "ros_executed",
                            "planning_time": result.get("planning_time", 0),
                            "timestamp": time.time(),
                        })
                    else:
                        error_msg = result.get("error", "Unknown") if result else "No response"
                        try:
                            from config.ROS import DEFAULT_CONTROL_MODE
                            if DEFAULT_CONTROL_MODE == "hybrid":
                                logger.warning(f"ROS planning failed ({error_msg}), falling back to TCP")
                                _use_ros = False
                            else:
                                return OperationResult.error_result(
                                    "ROS_PLANNING_FAILED",
                                    f"MoveIt planning failed: {error_msg}",
                                    ["Check MoveIt logs", "Verify positions are reachable"],
                                )
                        except ImportError:
                            _use_ros = False
            except ImportError:
                logger.warning("ros2 module not available, falling back to TCP")
                _use_ros = False

        # Construct command (TCP path)
        command = {
            "command_type": "draw_with_pen",
            "robot_id": robot_id,
            "parameters": {
                "pen_position": pen_position,
                "paper_position": paper_position,
                "shape": shape,
                "shape_params": shape_params or {},
            },
            "timestamp": time.time(),
            "request_id": request_id,
        }

        logger.info(
            f"Sending draw_with_pen command to {robot_id}: shape={shape}"
        )

        success = _get_command_broadcaster().send_command(command, request_id)

        if not success:
            return OperationResult.error_result(
                "COMMUNICATION_FAILED",
                "Failed to send command to Unity",
                ["Ensure Unity is running"],
            )

        logger.info(f"Successfully sent draw_with_pen command to {robot_id}")

        return OperationResult.success_result(
            {
                "robot_id": robot_id,
                "pen_position": pen_position,
                "paper_position": paper_position,
                "shape": shape,
                "shape_params": shape_params,
                "status": "command_sent",
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.error(f"Unexpected error in draw_with_pen: {e}", exc_info=True)
        return OperationResult.error_result(
            "UNEXPECTED_ERROR",
            f"Unexpected error occurred: {str(e)}",
            ["Check logs"],
        )


# ============================================================================
# BasicOperation Definitions
# ============================================================================


def create_grip_object_operation() -> BasicOperation:
    """Create the BasicOperation definition for grip_object."""
    return BasicOperation(
        operation_id="manipulation_grip_object_003",
        name="grip_object",
        category=OperationCategory.MANIPULATION,
        complexity=OperationComplexity.INTERMEDIATE,
        description="Simplified grasping operation - wrapper around full grasp planning",
        long_description="""
            This operation provides a simpler interface for grasping objects
            compared to the full grasp_object operation which includes the
            complete grasp planning pipeline.

            Useful for quick grasping when full grasp planning is not needed.
        """,
        usage_examples=[
            "grip_object('Robot1', {'x': 0.3, 'y': 0.15, 'z': 0.1})",
            "grip_object('Robot1', detected_position, approach_direction='side')",
        ],
        parameters=[
            OperationParameter(
                name="robot_id", type="str", description="Robot ID", required=True
            ),
            OperationParameter(
                name="object_position",
                type="dict",
                description="Object position with x, y, z keys",
                required=True,
            ),
            OperationParameter(
                name="approach_direction",
                type="str",
                description="Approach direction ('top', 'front', 'side')",
                required=False,
                default="top",
            ),
        ],
        preconditions=["Object detected", "Position reachable"],
        postconditions=["Object grasped", "Gripper closed on object"],
        average_duration_ms=1500.0,
        success_rate=0.93,
        failure_modes=["Object not reachable", "Grasp failed"],
        implementation=grip_object,
    )


def create_align_object_operation() -> BasicOperation:
    """Create the BasicOperation definition for align_object."""
    return BasicOperation(
        operation_id="manipulation_align_object_004",
        name="align_object",
        category=OperationCategory.MANIPULATION,
        complexity=OperationComplexity.INTERMEDIATE,
        description="Align object or gripper to target orientation",
        long_description="""
            This operation adjusts gripper or held object orientation to
            match a target orientation.

            Useful for precise placement or insertion tasks requiring
            specific orientations.
        """,
        usage_examples=[
            "align_object('Robot1', {'roll': 0, 'pitch': -90, 'yaw': 0})",
            "Align for vertical insertion before placing object",
        ],
        parameters=[
            OperationParameter(
                name="robot_id", type="str", description="Robot ID", required=True
            ),
            OperationParameter(
                name="target_orientation",
                type="dict",
                description="Target orientation (roll, pitch, yaw in degrees)",
                required=True,
            ),
            OperationParameter(
                name="alignment_type",
                type="str",
                description="'gripper' or 'object'",
                required=False,
                default="gripper",
            ),
        ],
        preconditions=["Robot at stable position", "Orientation reachable"],
        postconditions=["Orientation adjusted", "Alignment complete"],
        average_duration_ms=900.0,
        success_rate=0.94,
        failure_modes=["Unreachable orientation", "Joint limits"],
        implementation=align_object,
    )


def create_follow_path_operation() -> BasicOperation:
    """Create the BasicOperation definition for follow_path."""
    return BasicOperation(
        operation_id="navigation_follow_path_004",
        name="follow_path",
        category=OperationCategory.NAVIGATION,
        complexity=OperationComplexity.INTERMEDIATE,
        description="Follow multi-waypoint trajectory",
        long_description="""
            This operation commands the robot to move through a series of
            waypoints, creating a smooth continuous trajectory.

            Useful for complex paths avoiding obstacles or following
            predefined trajectories.
        """,
        usage_examples=[
            "follow_path('Robot1', [{'x': 0, 'y': 0, 'z': 0.3}, {'x': 0.3, 'y': 0, 'z': 0.1}])",
            "Navigate around obstacles using waypoint path",
        ],
        parameters=[
            OperationParameter(
                name="robot_id", type="str", description="Robot ID", required=True
            ),
            OperationParameter(
                name="waypoints",
                type="list",
                description="List of waypoint dicts with x, y, z keys",
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
        preconditions=["All waypoints reachable", "Path collision-free"],
        postconditions=["All waypoints reached", "End at final waypoint"],
        average_duration_ms=2000.0,
        success_rate=0.91,
        failure_modes=["Waypoint unreachable", "Path obstructed"],
        implementation=follow_path,
    )


def create_draw_with_pen_operation() -> BasicOperation:
    """Create the BasicOperation definition for draw_with_pen."""
    return BasicOperation(
        operation_id="manipulation_draw_with_pen_005",
        name="draw_with_pen",
        category=OperationCategory.MANIPULATION,
        complexity=OperationComplexity.COMPLEX,
        description="Draw shapes with pen tool on paper surface",
        long_description="""
            This operation demonstrates tool manipulation by drawing shapes
            with a pen on paper.

            Requires precise position control and tool interaction.
        """,
        usage_examples=[
            "draw_with_pen('Robot1', pen_pos, paper_pos, 'line', {'length': 0.1})",
            "draw_with_pen('Robot1', pen_pos, paper_pos, 'circle', {'radius': 0.05})",
        ],
        parameters=[
            OperationParameter(
                name="robot_id", type="str", description="Robot ID", required=True
            ),
            OperationParameter(
                name="pen_position",
                type="dict",
                description="Pen position (x, y, z)",
                required=True,
            ),
            OperationParameter(
                name="paper_position",
                type="dict",
                description="Paper position (x, y, z)",
                required=True,
            ),
            OperationParameter(
                name="shape",
                type="str",
                description="Shape to draw ('line', 'circle', 'square', 'custom')",
                required=False,
                default="line",
            ),
            OperationParameter(
                name="shape_params",
                type="dict",
                description="Shape-specific parameters",
                required=False,
            ),
        ],
        preconditions=["Pen tool equipped", "Paper surface detected"],
        postconditions=["Shape drawn on paper", "Pen returned to start"],
        average_duration_ms=3000.0,
        success_rate=0.85,
        failure_modes=["Pen not detected", "Paper position inaccurate"],
        implementation=draw_with_pen,
    )


# ============================================================================
# Create operation instances for export
# ============================================================================

GRIP_OBJECT_OPERATION = create_grip_object_operation()
ALIGN_OBJECT_OPERATION = create_align_object_operation()
FOLLOW_PATH_OPERATION = create_follow_path_operation()
DRAW_WITH_PEN_OPERATION = create_draw_with_pen_operation()
