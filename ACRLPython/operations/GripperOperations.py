#!/usr/bin/env python3
"""Open/close operations for the robot gripper through Unity's GripperController via TCP."""

import time
import logging
from typing import Any, Dict, Optional

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

from core.LoggingSetup import setup_logging

setup_logging(__name__)
logger = logging.getLogger(__name__)


from ._imports import (
    get_command_broadcaster as _get_command_broadcaster,
    PLACE_HOVER_OFFSET,
    PLACE_MIN_Y,
    PLACE_TCP_OFFSET,
)


def control_gripper(
    robot_id: str,
    open_gripper: bool,
    request_id: int = 0,
    object_id: Optional[str] = None,
    use_ros: Optional[bool] = None,
) -> OperationResult:
    """Open or close the robot gripper. Pass object_id when closing for handoff attachment."""
    try:
        if err := validate_robot_id(robot_id):
            return err

        if not isinstance(
            open_gripper, bool
        ):  # pyright: ignore[reportUnnecessaryIsInstance]
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
                logger.debug(
                    f"Could not update gripper WorldState for {robot_id}: {_exc}"
                )

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
                        "Verify CommandServer is running (port 5007)",
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


def create_control_gripper_operation() -> BasicOperation:
    return BasicOperation(
        operation_id="manipulation_control_gripper_001",
        name="control_gripper",
        category=OperationCategory.MANIPULATION,
        complexity=OperationComplexity.ATOMIC,
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
        average_duration_ms=500,
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


CONTROL_GRIPPER_OPERATION = create_control_gripper_operation()


def release_object(
    robot_id: str,
    request_id: int = 0,
    use_ros: Optional[bool] = None,
) -> OperationResult:
    """
    Open gripper at current position (ATOMIC — does NOT move the robot).

    For positioned release, chain: move_to_coordinate → release_object.
    """
    try:
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string, got: {robot_id}",
                ["Provide a valid robot ID (e.g., 'Robot1', 'AR4_Robot')"],
            )

        def _ros_path():
            from ros2.ROSBridge import ROSBridge

            bridge = ROSBridge.get_instance()
            # 1.0 = fully open (normalized value)
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
                        "Verify CommandServer is running (port 5007)",
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


def _check_placement_reachability(robot_id: str, x: float, y: float, z: float) -> str:
    """Non-blocking KG reachability check for the placement position.

    Returns a short note string included in the OperationResult payload.
    Never raises — if KG is disabled or unavailable the check is silently skipped.
    """
    try:
        from config.KnowledgeGraph import KNOWLEDGE_GRAPH_ENABLED

        if not KNOWLEDGE_GRAPH_ENABLED:
            return "kg_disabled"

        from core.Imports import get_graph_query_engine

        qe = get_graph_query_engine()
        if qe is None:
            return "kg_unavailable"

        result = qe.can_reach_position(robot_id, (x, y, z))
        if not result["reachable"]:
            logger.warning(
                f"place_object: KG reachability check failed for {robot_id} "
                f"→ ({x:.3f}, {y:.3f}, {z:.3f}): {result['reason']}"
            )
            return f"unreachable:{result['reason']}"

        return "reachable"
    except Exception as e:
        logger.debug(f"place_object: KG reachability check skipped: {e}")
        return "kg_check_skipped"


def _resolve_placement_y(
    on_top_of: str,
    placed_object_height: float,
    fallback_y: float,
) -> tuple[float, str]:
    """Compute placement Y when stacking on another object via WorldState lookup."""
    try:
        from ._imports import get_world_state
    except ImportError:
        from operations._imports import get_world_state  # type: ignore

    ws = get_world_state()
    canonical_id = ws.resolve_canonical_id(on_top_of)
    if canonical_id is None:
        logger.warning(
            f"place_object: on_top_of='{on_top_of}' not in WorldState; fallback to explicit y"
        )
        return fallback_y, "fallback_object_not_found"

    obj_pos = ws.get_object_position(canonical_id)
    obj_dims = ws.get_object_dimensions(canonical_id)

    if obj_pos is None:
        logger.warning(
            f"place_object: '{canonical_id}' has no position; fallback to explicit y"
        )
        return fallback_y, "fallback_no_position"
    if obj_dims is None:
        logger.warning(
            f"place_object: '{canonical_id}' has no dimensions (vision-only detection); fallback to explicit y"
        )
        return fallback_y, "fallback_no_dimensions"

    _, obj_height, _ = obj_dims
    computed_y = obj_pos[1] + obj_height / 2.0 + placed_object_height / 2.0
    logger.info(
        f"place_object: stacking on '{canonical_id}' — "
        f"obj_center_y={obj_pos[1]:.4f}, obj_height={obj_height:.4f}, "
        f"placed_obj_height={placed_object_height:.4f} → computed_y={computed_y:.4f}"
    )
    return computed_y, f"stacked_on:{canonical_id}"


def place_object(
    robot_id: str,
    x: float,
    y: float,
    z: float,
    on_top_of: Optional[str] = None,
    placed_object_height: float = 0.0,
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
        y: Target Y coordinate in Unity world space (metres).  Ignored when
           on_top_of resolves successfully; used as fallback otherwise.
        z: Target Z coordinate in Unity world space (metres).
        on_top_of: Optional name or ID of a WorldState object to stack on.
           When provided and resolved, placement Y is computed automatically
           from target object position + dimensions.  Falls back to explicit y
           if the object is not found or lacks dimension data.
        placed_object_height: Height of the held object (metres).  Used with
           on_top_of so the held object lands flush on the target surface.
           Defaults to 0.0.
        use_ros: Override ROS/TCP path selection.  None = use config default.
        request_id: Optional request ID for tracking.

    Returns:
        OperationResult with placement confirmation or error details.
        Includes a ``resolution`` key: ``"explicit_coords"``,
        ``"stacked_on:<id>"``, or a ``"fallback_*"`` reason.

    Example:
        >>> result = place_object("Robot1", x=-0.18, y=0.06, z=0.05)
        >>> result = place_object("Robot1", x=0.0, y=0.0, z=0.05, on_top_of="blue_cube")
    """
    try:
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string, got: {robot_id}",
                ["Provide a valid robot ID (e.g., 'Robot1')"],
            )

        effective_y = y
        resolution_note = "explicit_coords"
        if on_top_of:
            effective_y, resolution_note = _resolve_placement_y(
                on_top_of, placed_object_height, fallback_y=y
            )

        # Stereo depth on flat Unity surfaces is unreliable and returns near-zero Y.
        # Clamp to table surface height so the arm never descends into the table.
        if effective_y < PLACE_MIN_Y:
            logger.warning(
                f"place_object: effective_y={effective_y:.4f} below PLACE_MIN_Y={PLACE_MIN_Y} "
                f"(stereo underestimate) — clamping to {PLACE_MIN_Y}"
            )
            effective_y = PLACE_MIN_Y
            resolution_note += "+y_clamped"

        reachability_note = _check_placement_reachability(robot_id, x, effective_y, z)

        def _ros_path():
            from ros2.ROSBridge import ROSBridge

            bridge = ROSBridge.get_instance()

            hover_pos = {"x": x, "y": effective_y + PLACE_HOVER_OFFSET, "z": z}
            place_pos = {"x": x, "y": effective_y + PLACE_TCP_OFFSET, "z": z}

            # Top-down orientation: ~179° around ROS X axis (ee_link Z points down).
            # w=0.0087 (not 0.0) matches grasp planner's base quaternion — keeps MoveIt
            # IK in the w>0 hemisphere so the solver always picks the same wrist config
            # and avoids the ±360° flip that occurs at the w=0 singularity.
            top_down_orientation = {"x": 0.9999, "y": 0.0, "z": 0.0, "w": 0.0087}

            # Step 0: Orient gripper to top-down at current position before moving.
            # plan_and_execute with an orientation constraint can fail to find a plan
            # if the robot starts far from the desired orientation (e.g. after a yawed
            # grasp). Pre-orienting makes subsequent constrained moves much more reliable.
            logger.info(f"place_object: orienting to top-down for {robot_id}")
            orient_result = bridge.plan_orientation_change(
                {"roll": 180, "pitch": 0, "yaw": 0},
                robot_id=robot_id,
            )
            if not orient_result or not orient_result.get("success"):
                logger.warning(
                    f"place_object: pre-orient failed for {robot_id}, continuing anyway"
                )

            time.sleep(0.1)

            # Step 1: Move to hover above target with top-down gripper orientation.
            logger.info(f"place_object: moving to hover above target for {robot_id}")
            hover_result = bridge.plan_and_execute(
                position=hover_pos,
                orientation=top_down_orientation,
                robot_id=robot_id,
                constrain_joint4=True,
            )
            if not hover_result or not hover_result.get("success"):
                err = (
                    hover_result.get("error", "Unknown")
                    if hover_result
                    else "No response"
                )
                logger.warning(
                    f"place_object: hover move failed ({err}), attempting direct descent"
                )

            # Brief settle pause so /joint_states reflects the hover pose before
            # MoveIt samples the start state for the descent plan.
            time.sleep(0.1)

            # Step 2: Descend to place height using free-space planning.
            # plan_cartesian_descent is NOT used here: MoveIt's collision model does
            # not include the held object, so a straight-line path through the
            # object's swept volume frequently fails at 0% completion.
            # Free-space planning (plan_and_execute) finds a collision-free path.
            logger.info(f"place_object: descending to place position for {robot_id}")
            descent_result = bridge.plan_and_execute(
                position=place_pos,
                orientation=top_down_orientation,
                robot_id=robot_id,
                constrain_joint4=True,
            )
            if not descent_result or not descent_result.get("success"):
                err = (
                    descent_result.get("error", "Unknown")
                    if descent_result
                    else "No response"
                )
                logger.warning(
                    f"place_object: descent failed ({err}), releasing at current height"
                )

            # Step 3: Open gripper and wait for Unity to detach the object before
            # the ascent trajectory is published. The gripper command is a fire-and-forget
            # ROS topic publish; 0.5s is well above the ~50ms ROS topic round-trip.
            logger.info(f"place_object: releasing object for {robot_id}")
            bridge.control_gripper(1.0, robot_id=robot_id)
            time.sleep(0.5)

            # Step 4: Ascend back to hover height to clear the placed object.
            logger.info(f"place_object: ascending after place for {robot_id}")
            bridge.plan_and_execute(
                position=hover_pos,
                robot_id=robot_id,
            )

            return OperationResult.success_result(
                {
                    "robot_id": robot_id,
                    "placed_at": {"x": x, "y": effective_y, "z": z},
                    # "ros_executed" tells SequenceExecutor to skip the Unity
                    # completion-signal wait — the ROS path is fully synchronous
                    # and Unity never sends a TCP completion message for it.
                    "status": "ros_executed",
                    "resolution": resolution_note,
                    "reachability": reachability_note,
                    "timestamp": time.time(),
                }
            )

        def _tcp_path():
            command = {
                "command_type": "place_object",
                "robot_id": robot_id,
                "parameters": {
                    "target_position": {"x": x, "y": effective_y, "z": z},
                    "hover_offset": PLACE_HOVER_OFFSET,
                    "tcp_offset": PLACE_TCP_OFFSET,
                },
                "timestamp": time.time(),
                "request_id": request_id,
            }
            logger.info(
                f"Sending place_object command to {robot_id} at ({x}, {effective_y}, {z})"
            )
            success = _get_command_broadcaster().send_command(command, request_id)
            if not success:
                return OperationResult.error_result(
                    "COMMUNICATION_FAILED",
                    "Failed to send command to Unity - no clients connected",
                    [
                        "Ensure Unity is running with UnifiedPythonReceiver active",
                        "Verify CommandServer is running (port 5007)",
                    ],
                )
            return OperationResult.success_result(
                {
                    "robot_id": robot_id,
                    "placed_at": {"x": x, "y": effective_y, "z": z},
                    "status": "command_sent",
                    "resolution": resolution_note,
                    "reachability": reachability_note,
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
        OperationParameter(
            name="on_top_of",
            type="str",
            description=(
                "Optional: name or ID of a WorldState object to stack on. "
                "When provided, placement Y is computed from target object "
                "position + dimensions. x and z still control horizontal alignment. "
                "Falls back to explicit y if object not found or lacks dimensions."
            ),
            required=False,
        ),
        OperationParameter(
            name="placed_object_height",
            type="float",
            description=(
                "Height of the held object (metres). Used with on_top_of so "
                "the held object lands flush on the target surface. Default 0.0."
            ),
            required=False,
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


def place_between_objects(
    robot_id: str,
    object_id_1: str,
    object_id_2: str,
    y: float = 0.06,
    on_top_of: Optional[str] = None,
    placed_object_height: float = 0.0,
    use_ros: Optional[bool] = None,
    request_id: int = 0,
) -> OperationResult:
    """
    Place a held object at the XZ midpoint between two WorldState objects.

    Delegates to place_object; on_top_of and placed_object_height pass through
    for flush stacking. Y defaults to caller-supplied value when on_top_of not used.
    """
    try:
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string, got: {robot_id}",
                ["Provide a valid robot ID (e.g., 'Robot1')"],
            )

        try:
            from ._imports import get_world_state
        except ImportError:
            from operations._imports import get_world_state  # type: ignore

        ws = get_world_state()

        def _resolve_pos(obj_id: str):
            canonical = ws.resolve_canonical_id(obj_id)
            if canonical is None:
                return None, obj_id
            pos = ws.get_object_position(canonical)
            return pos, canonical

        pos1, id1 = _resolve_pos(object_id_1)
        pos2, id2 = _resolve_pos(object_id_2)

        if pos1 is None:
            return OperationResult.error_result(
                "OBJECT_NOT_FOUND",
                f"Object '{object_id_1}' not found in WorldState",
                [
                    f"Detect '{object_id_1}' with detect_object_stereo first",
                    "Check object name matches WorldState entry",
                ],
            )
        if pos2 is None:
            return OperationResult.error_result(
                "OBJECT_NOT_FOUND",
                f"Object '{object_id_2}' not found in WorldState",
                [
                    f"Detect '{object_id_2}' with detect_object_stereo first",
                    "Check object name matches WorldState entry",
                ],
            )

        mid_x = (pos1[0] + pos2[0]) / 2.0
        mid_z = (pos1[2] + pos2[2]) / 2.0
        logger.info(
            f"place_between_objects: midpoint of '{id1}' {pos1} and '{id2}' {pos2}"
            f" → x={mid_x:.4f}, z={mid_z:.4f}"
        )

        result = place_object(
            robot_id=robot_id,
            x=mid_x,
            y=y,
            z=mid_z,
            on_top_of=on_top_of,
            placed_object_height=placed_object_height,
            use_ros=use_ros,
            request_id=request_id,
        )

        if result.success and result.result is not None:
            result.result["midpoint"] = {
                "x": mid_x,
                "y": result.result["placed_at"]["y"],
                "z": mid_z,
            }
            result.result["reference_objects"] = [id1, id2]

        return result

    except Exception as e:
        logger.error(f"Unexpected error in place_between_objects: {e}", exc_info=True)
        return OperationResult.error_result(
            "UNEXPECTED_ERROR",
            f"Unexpected error occurred: {str(e)}",
            ["Check logs for detailed error information", "Retry the operation"],
        )


PLACE_BETWEEN_OBJECTS_OPERATION = BasicOperation(
    operation_id="manipulation_place_between_objects_004",
    name="place_between_objects",
    category=OperationCategory.MANIPULATION,
    complexity=OperationComplexity.INTERMEDIATE,
    description=("Place a held object at the midpoint between two reference objects"),
    long_description="""
        Resolves both reference objects from WorldState, computes the XZ midpoint,
        and executes a controlled place sequence (hover, descent, release, ascent).
        Use this instead of manually averaging coordinates in the prompt.

        Supports the same on_top_of and placed_object_height stacking options
        as place_object for precise vertical placement.
    """,
    usage_examples=[
        "place_between_objects('Robot1', 'blue_cube', 'red_cube') — place at XZ midpoint, default Y",
        "place_between_objects('Robot1', 'blue_cube', 'red_cube', on_top_of='blue_cube') — stack height from blue_cube",
    ],
    parameters=[
        OperationParameter(
            name="robot_id",
            type="str",
            description="ID of the robot performing the placement",
            required=True,
        ),
        OperationParameter(
            name="object_id_1",
            type="str",
            description="Name or ID of the first reference object in WorldState",
            required=True,
        ),
        OperationParameter(
            name="object_id_2",
            type="str",
            description="Name or ID of the second reference object in WorldState",
            required=True,
        ),
        OperationParameter(
            name="y",
            type="float",
            description="Placement surface Y coordinate (metres). Default 0.06 (typical table surface).",
            required=False,
        ),
        OperationParameter(
            name="on_top_of",
            type="str",
            description=(
                "Optional: name or ID of a WorldState object whose top surface sets Y. "
                "Overrides the y parameter when resolvable."
            ),
            required=False,
        ),
        OperationParameter(
            name="placed_object_height",
            type="float",
            description="Height of the held object (metres). Used with on_top_of for flush stacking.",
            required=False,
        ),
    ],
    preconditions=[],
    postconditions=[],
    average_duration_ms=8000.0,
    success_rate=0.88,
    failure_modes=[
        "Either reference object not found in WorldState",
        "Midpoint outside robot reach envelope",
        "IK infeasible at computed midpoint",
    ],
    relationships=OperationRelationship(
        operation_id="manipulation_place_between_objects_004",
        required_operations=[],
        commonly_paired_with=[
            "manipulation_grasp_object_001",
            "vision_detect_object_stereo_001",
        ],
        pairing_reasons={
            "manipulation_grasp_object_001": "Grasp precedes between-placement in a pick-and-place sequence",
            "vision_detect_object_stereo_001": "Detect both reference objects before placing between them",
        },
        typical_after=["manipulation_grasp_object_001"],
        typical_before=["manipulation_release_object_002"],
    ),
    implementation=place_between_objects,
)
