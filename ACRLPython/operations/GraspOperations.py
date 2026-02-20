"""
Grasp Operations for Advanced Grasp Planning
=============================================

This module implements MoveIt2-inspired grasp planning operations that use
the full grasp planning pipeline with candidate generation, IK validation,
collision checking, and scoring.
"""

import logging
import time
from typing import Optional, List
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
# Follow-Target Configuration
# ============================================================================

# Toggle: when True the arm will re-plan to the live object position after each
# trajectory if the cube has drifted (e.g. pushed by the other robot's approach).
# Set to False to disable correction moves and always close at the planned position.
FOLLOW_TARGET_ENABLED: bool = True

# Maximum number of corrective moves before giving up and closing anyway.
FOLLOW_TARGET_MAX_CORRECTIONS: int = 3

# Minimum object drift (metres) that triggers a corrective plan_and_execute.
# Smaller values react to tiny vibrations; larger values only correct real pushes.
FOLLOW_TARGET_DRIFT_THRESHOLD: float = 0.015  # 1.5 cm


def _execute_grasp_with_follow_target(
    bridge,
    robot_id: str,
    object_id: str,
    planned_position: dict,
    orientation: dict,
    tcp_y_offset: float = 0.0,
    world_state=None,
) -> bool:
    """Move arm to object, optionally correct for drift, then close gripper.

    After MoveIt delivers the arm to the initially-planned position, the target
    cube may have been pushed by the other robot's open fingers.  When
    FOLLOW_TARGET_ENABLED is True this function re-queries the live object
    position from WorldState and issues a corrective plan_and_execute if the
    object has drifted more than FOLLOW_TARGET_DRIFT_THRESHOLD metres.  Up to
    FOLLOW_TARGET_MAX_CORRECTIONS correction moves are attempted before the
    gripper is closed regardless.

    The gripper is only closed *after* the arm has settled at (or near) the
    final object position, so it never closes during the approach phase.

    Args:
        bridge: Connected ROSBridge instance.
        robot_id: Robot namespace (e.g. "Robot1").
        object_id: Object identifier used to look up live position in WorldState.
        planned_position: Initial target position dict with x/y/z keys.
        orientation: Gripper orientation quaternion dict with x/y/z/w keys.
        tcp_y_offset: Additional Y offset applied to live positions (e.g. 0.05m
            for top-down grasps so ee_link lands above the object centre).
        world_state: WorldState instance (obtained lazily if None).

    Returns:
        True if the gripper was closed successfully, False otherwise.
    """
    import math

    current_position = dict(planned_position)

    if FOLLOW_TARGET_ENABLED and world_state is not None:
        for correction in range(FOLLOW_TARGET_MAX_CORRECTIONS):
            live_pos = world_state.get_object_position(object_id)
            if live_pos is None:
                logger.debug(f"[follow_target] {robot_id}: object '{object_id}' not in WorldState, skipping correction")
                break

            # Compute drift distance in XZ plane (Y is vertical, cube stays on table)
            dx = live_pos[0] - (current_position["x"])
            dz = live_pos[2] - (current_position["z"])
            drift = math.sqrt(dx * dx + dz * dz)

            if drift <= FOLLOW_TARGET_DRIFT_THRESHOLD:
                logger.info(
                    f"[follow_target] {robot_id}: object drift {drift*100:.1f} cm — within threshold, ready to close"
                )
                break

            logger.info(
                f"[follow_target] {robot_id}: object drifted {drift*100:.1f} cm "
                f"(correction {correction + 1}/{FOLLOW_TARGET_MAX_CORRECTIONS}), re-planning"
            )

            # Build corrected target from live object position
            corrected = {
                "x": live_pos[0],
                "y": live_pos[1] + tcp_y_offset,
                "z": live_pos[2],
            }
            current_position = corrected

            correction_result = bridge.plan_and_execute(
                position=corrected,
                orientation=orientation,
                planning_time=8.0,
                robot_id=robot_id,
                max_velocity_scaling=0.3,
                max_acceleration_scaling=0.3,
            )

            if not correction_result or not correction_result.get("success"):
                logger.warning(
                    f"[follow_target] {robot_id}: corrective move failed — "
                    f"{correction_result.get('error') if correction_result else 'no response'}"
                )
                break
    else:
        if not FOLLOW_TARGET_ENABLED:
            logger.debug(f"[follow_target] disabled — closing gripper at planned position")

    # Arm is at (corrected) grasp position.
    # Wait for the ArticulationBody PD controller to settle before closing so the
    # gripper doesn't fire while the arm is still oscillating at the target pose.
    logger.info(f"[follow_target] {robot_id}: waiting for arm to settle before closing gripper")
    time.sleep(0.5)

    # Close gripper
    logger.info(f"[follow_target] {robot_id}: closing gripper")
    gripper_result = bridge.control_gripper(0.0, robot_id=robot_id)
    if gripper_result and gripper_result.get("success"):
        # Give Unity physics time to register the contact before returning
        time.sleep(0.8)
        logger.info(f"[follow_target] {robot_id}: gripper closed")
        return True
    else:
        logger.warning(f"[follow_target] {robot_id}: gripper close command failed")
        return False


# ============================================================================
# Implementation: Grasp Object Operation
# ============================================================================


def grasp_object(
    robot_id: str,
    object_id: str,
    use_advanced_planning: bool = True,
    preferred_approach: str = "auto",  # "top", "front", "side", "auto"
    pre_grasp_distance: float = 0.0,   # 0 = use config default
    enable_retreat: bool = True,
    retreat_distance: float = 0.0,     # 0 = use config default
    custom_approach_vector: Optional[List[float]] = None,  # [x, y, z] or None
    request_id: int = 0,
    use_ros: Optional[bool] = None,
) -> OperationResult:
    """
    Plan and execute grasp on detected object using MoveIt2-inspired pipeline.

    This operation uses the full advanced grasp planning pipeline to:
    1. Generate multiple grasp candidates per approach type
    2. Filter candidates by IK reachability
    3. Filter candidates by collision-free approach paths
    4. Score candidates by weighted criteria (IK quality, approach, depth, stability)
    5. Execute best grasp with three-waypoint sequence (pre-grasp → grasp → retreat)

    The operation leverages GraspPlanningPipeline in Unity which implements:
    - Candidate generation with object-size adaptive distances
    - IK validation using damped least-squares solver
    - SphereCast collision checking along approach trajectories
    - Multi-criteria scoring with configurable weights
    - Safe retreat motion after grasping

    Args:
        robot_id: ID of the robot to control (e.g., "Robot1", "AR4_Robot")
        object_id: ID or name of the object to grasp (must be detected/tracked)
        use_advanced_planning: Use full pipeline (True) or simple planner (False)
        preferred_approach: Preferred grasp approach direction
            - "auto": Let pipeline determine best approach (recommended)
            - "top": Approach from above (gripper pointing down)
            - "front": Approach from front/back
            - "side": Approach from left/right
        pre_grasp_distance: Custom pre-grasp distance in meters (0 = use config)
        enable_retreat: Whether to retreat after grasping
        retreat_distance: Custom retreat distance in meters (0 = use config)
        custom_approach_vector: Custom approach direction [x, y, z] (overrides preferred_approach)
        request_id: Request ID for tracking (optional)
        use_ros: Whether to use ROS for motion planning (None = auto-detect from config)

    Returns:
        Dict with the following structure:
        {
            "success": bool,           # True if grasp was planned and executed
            "result": dict or None,    # Result data if successful
            "error": dict or None      # Error information if failed
        }

        Success result structure:
        {
            "robot_id": str,
            "object_id": str,
            "approach_type": str,
            "score": float,            # Quality score of selected grasp
            "status": str,
            "timestamp": float
        }

        Error structure:
        {
            "code": str,                    # Error code (e.g., "PLANNING_FAILED")
            "message": str,                 # Human-readable error message
            "recovery_suggestions": list    # List of suggested actions
        }

    Example:
        >>> # Grasp object with automatic approach selection
        >>> result = grasp_object("Robot1", "Cube_01")
        >>> if result["success"]:
        ...     print(f"Grasped {result['result']['object_id']}")

        >>> # Grasp with specific approach direction
        >>> result = grasp_object("Robot1", "Cube_01", preferred_approach="top")

        >>> # Grasp without retreat motion
        >>> result = grasp_object("Robot1", "Cube_01", enable_retreat=False)

        >>> # Custom approach vector (approach from specific direction)
        >>> result = grasp_object("Robot1", "Cube_01",
        ...                       custom_approach_vector=[0, 1, 0.5])
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

        # Validate object_id
        if not object_id or not isinstance(object_id, str):
            return OperationResult.error_result(
                "INVALID_OBJECT_ID",
                f"Object ID must be a non-empty string, got: {object_id}",
                [
                    "Provide a valid object ID or name",
                    "Ensure object is detected and tracked in the scene",
                ],
            )

        # Validate preferred_approach
        valid_approaches = ["auto", "top", "front", "side"]
        if preferred_approach.lower() not in valid_approaches:
            return OperationResult.error_result(
                "INVALID_APPROACH",
                f"Preferred approach must be one of {valid_approaches}, got: {preferred_approach}",
                [
                    "Use 'auto' to let pipeline determine best approach",
                    "Or specify 'top', 'front', or 'side' explicitly",
                ],
            )

        # Validate custom_approach_vector if provided
        if custom_approach_vector is not None:
            if not isinstance(custom_approach_vector, (list, tuple)) or len(custom_approach_vector) != 3:
                return OperationResult.error_result(
                    "INVALID_APPROACH_VECTOR",
                    f"Custom approach vector must be a 3-element list [x, y, z], got: {custom_approach_vector}",
                    [
                        "Provide a valid 3D vector: [x, y, z]",
                        "Example: [0, 1, 0] for upward approach",
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

        # Route via ROS if enabled (grasp planning is complex, best handled by MoveIt for collision-free grasps)
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
                    # ROS grasp planning requires object position - query from WorldState
                    # WorldState is updated by detect_object_stereo and other detection operations
                    try:
                        from core.Imports import get_world_state
                        world_state = get_world_state()
                        object_position = world_state.get_object_position(object_id)

                        logger.info(
                            f"[GRASP_DEBUG] {object_id} WorldState position: {object_position}"
                        )
                        if object_position is None:
                            logger.warning(f"Object {object_id} not found in WorldState - cannot use ROS planning")
                            try:
                                from config.ROS import DEFAULT_CONTROL_MODE
                                if DEFAULT_CONTROL_MODE == "hybrid":
                                    logger.info("Falling back to TCP for grasp execution")
                                    _use_ros = False
                                else:
                                    return OperationResult.error_result(
                                        "OBJECT_NOT_FOUND",
                                        f"Object {object_id} not found in WorldState - run detect_object_stereo first",
                                        [
                                            "Use detect_object_stereo to locate the object before grasping",
                                            "Ensure object detection completed successfully",
                                        ],
                                    )
                            except ImportError:
                                _use_ros = False
                        else:
                            # We have object position - check for dimensions for grasp planning
                            logger.info(f"Resolved {object_id} to position {object_position}, planning with ROS")

                            # Get object dimensions and current gripper state for grasp planning
                            object_dimensions = world_state.get_object_dimensions(object_id)
                            robot_state = world_state.get_robot_state(robot_id)

                            # If dimensions are available, use grasp planning pipeline
                            if object_dimensions is not None and robot_state is not None and robot_state.position is not None:
                                try:
                                    from grasp_planning.GraspPlanner import GraspPlanner

                                    logger.info(f"Using grasp planning pipeline for {object_id}")

                                    # Initialize grasp planner
                                    planner = GraspPlanner()

                                    # Plan grasp with full pipeline
                                    best_grasp = planner.plan_grasp(
                                        object_position=object_position,
                                        object_rotation=(0.0, 0.0, 0.0, 1.0),  # Identity rotation for now
                                        object_size=object_dimensions,
                                        robot_id=robot_id,
                                        gripper_position=robot_state.position,
                                        gripper_rotation=None,  # Could use robot_state.rotation if available
                                        use_moveit_ik=True,
                                        preferred_approach=preferred_approach.lower() if preferred_approach.lower() != "auto" else None,
                                        min_score=0.3,
                                    )

                                    if best_grasp is None:
                                        logger.warning("Grasp planning found no valid candidates, falling back to position-only")
                                        # Fall through to position-only planning
                                    else:
                                        # Use planned grasp pose with orientation
                                        logger.info(
                                            f"Grasp planning succeeded: {best_grasp.approach_type} approach, "
                                            f"score={best_grasp.total_score:.3f}"
                                        )

                                        # Execute grasp: pre-grasp hover → descend to grasp.
                                        # Using pre_grasp_position avoids approaching the cube
                                        # from the side (which would knock it away).
                                        grasp_orientation = {
                                            "x": best_grasp.grasp_rotation[0],
                                            "y": best_grasp.grasp_rotation[1],
                                            "z": best_grasp.grasp_rotation[2],
                                            "w": best_grasp.grasp_rotation[3],
                                        }
                                        logger.info(f"Moving to pre-grasp position for {robot_id}")
                                        pre_result = bridge.plan_and_execute(
                                            position={
                                                "x": best_grasp.pre_grasp_position[0],
                                                "y": best_grasp.pre_grasp_position[1],
                                                "z": best_grasp.pre_grasp_position[2],
                                            },
                                            orientation=grasp_orientation,
                                            planning_time=10.0,
                                            robot_id=robot_id,
                                        )
                                        if not pre_result or not pre_result.get("success"):
                                            pre_err = pre_result.get("error", "Unknown") if pre_result else "No response"
                                            logger.warning(f"Pre-grasp move failed ({pre_err}), attempting direct grasp")

                                        # Brief pause after pre-grasp arrival so MoveIt samples
                                        # settled joint states as the start state for the descent.
                                        # ROSTrajectorySubscriber already waits for physics settle
                                        # before firing "completed", but a small extra delay ensures
                                        # the /joint_states topic has published the resting pose.
                                        time.sleep(0.3)

                                        logger.info(f"Descending to grasp position for {robot_id}")
                                        result = bridge.plan_cartesian_descent(
                                            position={
                                                "x": best_grasp.grasp_position[0],
                                                "y": best_grasp.grasp_position[1],
                                                "z": best_grasp.grasp_position[2],
                                            },
                                            orientation=grasp_orientation,
                                            robot_id=robot_id,
                                            max_velocity_scaling=0.3,
                                            max_acceleration_scaling=0.3,
                                        )

                                        if result and result.get("success"):
                                            # Arm reached planned grasp position.
                                            # Follow-target: correct for any object drift caused
                                            # by the other robot's approach, then close gripper.
                                            logger.info(f"Arm at grasp position, starting follow-target for {robot_id}")
                                            try:
                                                from core.Imports import get_world_state
                                                _ws = get_world_state()
                                            except Exception:
                                                _ws = None
                                            _execute_grasp_with_follow_target(
                                                bridge=bridge,
                                                robot_id=robot_id,
                                                object_id=object_id,
                                                planned_position={
                                                    "x": best_grasp.grasp_position[0],
                                                    "y": best_grasp.grasp_position[1],
                                                    "z": best_grasp.grasp_position[2],
                                                },
                                                orientation=grasp_orientation,
                                                tcp_y_offset=0.0,
                                                world_state=_ws,
                                            )

                                            return OperationResult.success_result({
                                                "robot_id": robot_id,
                                                "object_id": object_id,
                                                "position": object_position,
                                                "grasp_approach": best_grasp.approach_type,
                                                "grasp_score": best_grasp.total_score,
                                                "request_id": request_id,
                                                "status": "ros_executed_with_grasp_planning",
                                                "planning_time": result.get("planning_time", 0),
                                                "timestamp": time.time(),
                                            })
                                        else:
                                            error_msg = result.get("error", "Unknown") if result else "No response"
                                            try:
                                                from config.ROS import DEFAULT_CONTROL_MODE
                                                if DEFAULT_CONTROL_MODE == "hybrid":
                                                    logger.warning(f"ROS motion planning failed ({error_msg}), falling back to TCP")
                                                    _use_ros = False
                                                else:
                                                    return OperationResult.error_result(
                                                        "ROS_PLANNING_FAILED",
                                                        f"MoveIt motion planning failed: {error_msg}",
                                                        ["Check MoveIt logs", "Verify object is reachable"],
                                                    )
                                            except ImportError:
                                                _use_ros = False

                                        # If we got here, planning succeeded but execution failed
                                        # If hybrid mode disabled fallback, skip to TCP path
                                        if not _use_ros:
                                            pass  # Will fall through to TCP path
                                        else:
                                            # Return error (execution failed in non-hybrid mode)
                                            return OperationResult.error_result(
                                                "ROS_EXECUTION_FAILED",
                                                f"Grasp planning succeeded but execution failed",
                                                ["Check robot state", "Verify trajectory is collision-free"],
                                            )

                                except ImportError as e:
                                    logger.warning(f"Grasp planning not available ({e}), using position-only")
                                    # Fall through to position-only planning
                                except Exception as e:
                                    logger.error(f"Error during grasp planning: {e}", exc_info=True)
                                    try:
                                        from config.ROS import DEFAULT_CONTROL_MODE
                                        if DEFAULT_CONTROL_MODE == "hybrid":
                                            logger.warning(f"Falling back to TCP due to grasp planning error")
                                            _use_ros = False
                                        else:
                                            return OperationResult.error_result(
                                                "GRASP_PLANNING_ERROR",
                                                f"Grasp planning failed: {str(e)}",
                                                ["Check logs for details", "Verify object dimensions are available"],
                                            )
                                    except ImportError:
                                        _use_ros = False

                            # Fallback: position-only planning (no dimensions or grasp planning failed)
                            if _use_ros:
                                logger.info(f"Using position-only planning for {object_id}")

                                # Convert position tuple to dict
                                position_dict = {
                                    "x": object_position[0],
                                    "y": object_position[1],
                                    "z": object_position[2],
                                }

                                # Top-down grasp orientation in ROS base_link frame.
                                # roll=pi (180deg around X) flips ee_link Z to point down
                                # (-ROS Z = gravity direction). MoveIt must reach the target
                                # with this orientation so the TCP offset below is valid.
                                #
                                # Unity uses Euler(180,0,90) which includes 90deg Z to
                                # compensate for the URDF gripper_base_joint rpy=(-pi/2,0,0)
                                # mount. MoveIt's tip_link is ee_link (before that mount),
                                # so no Z compensation is needed here.
                                top_down_orientation = {
                                    "x": 1.0,
                                    "y": 0.0,
                                    "z": 0.0,
                                    "w": 0.0,
                                }

                                # TCP offset: shift ee_link target 5cm above object center
                                # so gripper fingers (which extend 5cm from ee_link in -Z)
                                # land at the object. Only valid with top-down orientation.
                                # Matches gripperCenterOffset.z = 0.05m in GraspCandidate.cs.
                                # NOTE: offset is applied in Unity world Y (up), not Unity Z
                                # (forward). Unity Y=up maps to ROS Z=up after the transform
                                # in _transform_world_to_local, so +0.05 here places ee_link
                                # 5cm above the object in the world vertical axis.
                                #
                                # PRE-GRASP → GRASP two-step: first move to a safe hover
                                # position 15cm above the object, then descend straight down
                                # 10cm to the grasp position. This keeps the approach vector
                                # vertical so the arm never nudges the cube sideways.
                                PRE_GRASP_Y_OFFSET = 0.15  # 15cm above object center
                                GRASP_Y_OFFSET = 0.03      # 3cm above object center (lower = fingers wrap more of object)

                                pre_grasp_position = {
                                    "x": position_dict["x"],
                                    "y": position_dict["y"] + PRE_GRASP_Y_OFFSET,
                                    "z": position_dict["z"],
                                }
                                position_dict_with_offset = {
                                    "x": position_dict["x"],
                                    "y": position_dict["y"] + GRASP_Y_OFFSET,
                                    "z": position_dict["z"],
                                }

                                # Step 1: Move to pre-grasp hover position
                                logger.info(f"Moving to pre-grasp position for {robot_id}")
                                pre_result = bridge.plan_and_execute(
                                    position=pre_grasp_position,
                                    orientation=top_down_orientation,
                                    planning_time=10.0,
                                    robot_id=robot_id,
                                )
                                if not pre_result or not pre_result.get("success"):
                                    error_msg = pre_result.get("error", "Unknown") if pre_result else "No response"
                                    logger.warning(f"Pre-grasp move failed ({error_msg})")
                                    try:
                                        from config.ROS import DEFAULT_CONTROL_MODE
                                        if DEFAULT_CONTROL_MODE == "hybrid":
                                            _use_ros = False
                                        else:
                                            return OperationResult.error_result(
                                                "ROS_PLANNING_FAILED",
                                                f"MoveIt pre-grasp planning failed: {error_msg}",
                                                ["Check MoveIt logs", "Verify object is reachable"],
                                            )
                                    except ImportError:
                                        _use_ros = False

                                if _use_ros:
                                    # Brief pause after pre-grasp arrival to let /joint_states
                                    # publish the settled pose before MoveIt samples start state.
                                    time.sleep(0.3)

                                    # Step 2: Descend straight down to grasp position.
                                    # Use Cartesian path to guarantee a vertical descent —
                                    # free-space planning allows wrist joints to rotate to an
                                    # alternate IK solution, offsetting the gripper laterally.
                                    logger.info(f"Descending to grasp position for {robot_id}")
                                    result = bridge.plan_cartesian_descent(
                                        position=position_dict_with_offset,
                                        orientation=top_down_orientation,
                                        robot_id=robot_id,
                                        max_velocity_scaling=0.3,
                                        max_acceleration_scaling=0.3,
                                    )
                                else:
                                    result = None

                                if result and result.get("success"):
                                    # Arm reached planned grasp position.
                                    # Follow-target: correct for any object drift, then close gripper.
                                    logger.info(f"Arm at grasp position, starting follow-target for {robot_id}")
                                    try:
                                        from core.Imports import get_world_state
                                        _ws = get_world_state()
                                    except Exception:
                                        _ws = None
                                    _execute_grasp_with_follow_target(
                                        bridge=bridge,
                                        robot_id=robot_id,
                                        object_id=object_id,
                                        planned_position=position_dict_with_offset,
                                        orientation=top_down_orientation,
                                        tcp_y_offset=GRASP_Y_OFFSET,
                                        world_state=_ws,
                                    )

                                    return OperationResult.success_result({
                                        "robot_id": robot_id,
                                        "object_id": object_id,
                                        "position": object_position,
                                        "request_id": request_id,
                                        "status": "ros_executed",
                                        "planning_time": result.get("planning_time", 0),
                                        "timestamp": time.time(),
                                    })
                                else:
                                    error_msg = result.get("error", "Unknown") if result else "No response"
                                    try:
                                        from config.ROS import DEFAULT_CONTROL_MODE
                                        if DEFAULT_CONTROL_MODE == "hybrid":
                                            logger.warning(f"ROS motion planning failed ({error_msg}), falling back to TCP")
                                            _use_ros = False
                                        else:
                                            return OperationResult.error_result(
                                                "ROS_PLANNING_FAILED",
                                                f"MoveIt motion planning failed: {error_msg}",
                                                ["Check MoveIt logs", "Verify object is reachable"],
                                            )
                                    except ImportError:
                                        _use_ros = False
                    except Exception as e:
                        logger.error(f"Error resolving object position for ROS: {e}")
                        try:
                            from config.ROS import DEFAULT_CONTROL_MODE
                            if DEFAULT_CONTROL_MODE == "hybrid":
                                logger.warning(f"Falling back to TCP due to error: {e}")
                                _use_ros = False
                            else:
                                return OperationResult.error_result(
                                    "ROS_PLANNING_ERROR",
                                    f"Error preparing ROS grasp: {str(e)}",
                                    ["Check logs for details", "Verify WorldState is initialized"],
                                )
                        except ImportError:
                            _use_ros = False
            except ImportError:
                logger.warning("ros2 module not available, falling back to TCP")
                _use_ros = False

        # Build command payload with parameters nested (to match Unity's RobotCommand structure) (TCP path)
        parameters = {
            "object_id": object_id,
            "use_advanced_planning": use_advanced_planning,
            "preferred_approach": preferred_approach.lower(),
            "pre_grasp_distance": pre_grasp_distance,
            "enable_retreat": enable_retreat,
            "retreat_distance": retreat_distance,
        }

        # Add custom approach vector if provided
        if custom_approach_vector is not None:
            parameters["custom_approach_vector"] = {
                "x": custom_approach_vector[0],
                "y": custom_approach_vector[1],
                "z": custom_approach_vector[2],
            }

        command = {
            "command_type": "grasp_object",  # Fixed: Unity expects "command_type" not "command"
            "target_type": "robot",  # Added: Required field for routing
            "robot_id": robot_id,
            "parameters": parameters,  # Nest grasp parameters under "parameters" key
            "request_id": request_id,
        }

        # Send command to Unity via CommandBroadcaster
        broadcaster = _get_command_broadcaster()
        if broadcaster is None:
            return OperationResult.error_result(
                "COMMUNICATION_ERROR",
                "CommandBroadcaster not available",
                [
                    "Ensure CommandServer is running",
                    "Check server initialization in orchestrator",
                ],
            )

        # Send command (don't wait - SequenceExecutor handles completion waiting)
        logger.info(f"Sending grasp_object command: {robot_id} -> {object_id}")
        success = broadcaster.send_command(command, request_id)

        if success:
            # Return success immediately - SequenceExecutor will wait for Unity completion
            logger.debug(f"Grasp command sent successfully, request_id={request_id}")
            return OperationResult.success_result({
                "command_sent": True,
                "robot_id": robot_id,
                "object_id": object_id,
                "request_id": request_id,
            })
        else:
            logger.error(f"Failed to send grasp command")
            return OperationResult.error_result(
                "COMMUNICATION_ERROR",
                "Failed to send grasp command to Unity",
                [
                    "Check Unity is connected to CommandServer",
                    "Verify network connectivity",
                ],
            )

    except Exception as e:
        logger.exception(f"Exception in grasp_object operation: {e}")
        return OperationResult.error_result(
            "EXCEPTION",
            f"Exception during grasp operation: {str(e)}",
            [
                "Check stack trace in logs",
                "Verify all parameters are correct",
                "Ensure Unity is running and responsive",
            ],
        )


# ============================================================================
# Operation Definition for Registry
# ============================================================================


GRASP_OBJECT_OPERATION = BasicOperation(
    operation_id="manipulation_grasp_object_001",
    name="grasp_object",
    category=OperationCategory.MANIPULATION,
    complexity=OperationComplexity.COMPLEX,
    description="Plan and execute grasp using MoveIt2-inspired pipeline with candidate generation, IK validation, collision checking, and scoring",
    long_description="""
        This operation uses a MoveIt2-inspired grasp planning pipeline to execute robust grasps.

        The pipeline includes:
        1. Candidate Generation: Generate multiple grasp poses per approach type (top, front, side)
        2. IK Filtering: Validate reachability using inverse kinematics solver
        3. Collision Filtering: Check approach paths for obstacles using SphereCast
        4. Multi-Criteria Scoring: Rank candidates by IK quality, approach preference, depth, and stability
        5. Three-Waypoint Execution: Pre-grasp → Grasp → Retreat sequence

        This operation provides superior grasp success rates compared to simple planning by:
        - Testing multiple approach directions and selecting the best
        - Validating reachability before execution
        - Avoiding collision paths
        - Adapting pre-grasp distances to object size
        - Including safe retreat motions after grasping
    """,
    usage_examples=[
        "Grasp object with automatic approach selection: grasp_object(robot_id='Robot1', object_id='Cube_01')",
        "Grasp from specific direction: grasp_object(robot_id='Robot1', object_id='Cube_01', preferred_approach='top')",
        "Grasp with custom distances: grasp_object(robot_id='Robot1', object_id='Cube_01', pre_grasp_distance=0.12, retreat_distance=0.15)",
        "Grasp without retreat: grasp_object(robot_id='Robot1', object_id='Cube_01', enable_retreat=False)",
        "Grasp with custom approach vector: grasp_object(robot_id='Robot1', object_id='Cube_01', custom_approach_vector=[0, 1, 0.5])",
    ],
    parameters=[
        OperationParameter(
            name="robot_id",
            type="str",
            description="ID of the robot to control (e.g., 'Robot1', 'AR4_Robot')",
            required=True,
        ),
        OperationParameter(
            name="object_id",
            type="str",
            description="ID or name of the object to grasp (must be detected/tracked)",
            required=True,
        ),
        OperationParameter(
            name="use_advanced_planning",
            type="bool",
            description="Use full planning pipeline (True) or simple planner (False)",
            required=False,
            default=True,
        ),
        OperationParameter(
            name="preferred_approach",
            type="str",
            description="Preferred grasp approach: 'auto', 'top', 'front', 'side'",
            required=False,
            default="auto",
            valid_values=["auto", "top", "front", "side"],
        ),
        OperationParameter(
            name="pre_grasp_distance",
            type="float",
            description="Custom pre-grasp distance in meters (0 = use config default)",
            required=False,
            default=0.0,
            valid_range=(0.0, 0.3),
        ),
        OperationParameter(
            name="enable_retreat",
            type="bool",
            description="Whether to retreat after grasping",
            required=False,
            default=True,
        ),
        OperationParameter(
            name="retreat_distance",
            type="float",
            description="Custom retreat distance in meters (0 = use config default)",
            required=False,
            default=0.0,
            valid_range=(0.0, 0.5),
        ),
        OperationParameter(
            name="custom_approach_vector",
            type="list",
            description="Custom approach direction [x, y, z] (overrides preferred_approach)",
            required=False,
            default=None,
        ),
    ],
    preconditions=[
        "Object must be detected and tracked in the scene",
        "Robot must be initialized and responsive",
        "Target object must be within robot's workspace",
    ],
    postconditions=[
        "Robot gripper positioned at grasp location with object grasped",
        "If retreat enabled: robot lifted object to safe height",
        "Grasp quality score available for verification",
    ],
    average_duration_ms=150.0,
    success_rate=0.92,
    failure_modes=[
        "No valid grasp candidates found (all filtered out)",
        "IK validation failed for all candidates",
        "All approach paths have collisions",
        "Object not found in scene",
        "Object outside robot reach",
    ],
    implementation=grasp_object,
)
