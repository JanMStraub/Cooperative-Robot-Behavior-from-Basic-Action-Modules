#!/usr/bin/env python3
"""Public grasp_object dispatcher and its BasicOperation definition."""

import logging
from typing import List, Optional

from core.LoggingSetup import setup_logging

from ..Base import (
    BasicOperation,
    OperationCategory,
    OperationComplexity,
    OperationParameter,
    OperationRelationship,
    OperationResult,
)

try:
    from ._helpers import _handle_ros_failure  # type: ignore[import]
    from ._ros import _grasp_via_ros_planned, _grasp_via_ros_position_only  # type: ignore[import]
    from ._vgn import _grasp_via_vgn, _grasp_via_vgn_with_ros  # type: ignore[import]
except ImportError:
    from operations.grasp._helpers import _handle_ros_failure  # type: ignore[no-redef]
    from operations.grasp._ros import _grasp_via_ros_planned, _grasp_via_ros_position_only  # type: ignore[no-redef]
    from operations.grasp._vgn import _grasp_via_vgn, _grasp_via_vgn_with_ros  # type: ignore[no-redef]

setup_logging(__name__)
logger = logging.getLogger(__name__)

try:
    from ...core.Imports import get_command_broadcaster as _get_command_broadcaster
except ImportError:
    from core.Imports import get_command_broadcaster as _get_command_broadcaster  # type: ignore[no-redef]


def grasp_object(
    robot_id: str,
    object_id: str,
    use_advanced_planning: bool = True,
    preferred_approach: str = "top",  # "top", "front", "side", "auto"
    pre_grasp_distance: float = 0.0,  # 0 = use config default
    enable_retreat: bool = True,
    retreat_distance: float = 0.0,  # 0 = use config default
    custom_approach_vector: Optional[List[float]] = None,  # [x, y, z] or None
    grasp_yaw_override: Optional[float] = None,  # radians; bypasses WorldState/VGN yaw
    request_id: int = 0,
    use_ros: Optional[bool] = None,
) -> OperationResult:
    """Plan and execute grasp via VGN+ROS → geometric ROS → TCP fallback chain."""
    try:
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string, got: {robot_id}",
                [
                    "Provide a valid robot ID (e.g., 'Robot1', 'AR4_Robot')",
                    "Check RobotManager in Unity for available robot IDs",
                ],
            )

        if not object_id or not isinstance(object_id, str):
            return OperationResult.error_result(
                "INVALID_OBJECT_ID",
                f"Object ID must be a non-empty string, got: {object_id}",
                [
                    "Provide a valid object ID or name",
                    "Ensure object is detected and tracked in the scene",
                ],
            )

        if object_id.startswith("$"):
            return OperationResult.error_result(
                "UNRESOLVED_VARIABLE",
                f"object_id '{object_id}' is an unresolved variable reference - "
                "run detect_object_stereo before grasp_object to capture the target",
                [
                    "Add a detect_object_stereo step before grasp_object",
                    "Check that capture_var is set on the detection step",
                ],
            )

        _approach_aliases = {"left": "left_side", "right": "right_side"}
        preferred_approach = _approach_aliases.get(
            preferred_approach.lower(), preferred_approach.lower()
        )
        valid_approaches = ["auto", "top", "front", "side", "left_side", "right_side"]
        if preferred_approach not in valid_approaches:
            return OperationResult.error_result(
                "INVALID_APPROACH",
                f"Preferred approach must be one of {valid_approaches}, got: {preferred_approach}",
                [
                    "Use 'top' for standard top-down approach (default)",
                    "Or specify 'front', 'side', or 'auto' explicitly",
                ],
            )

        if (
            isinstance(custom_approach_vector, (list, tuple))
            and len(custom_approach_vector) == 0
        ):
            custom_approach_vector = None
        if custom_approach_vector is not None:
            if (
                not isinstance(custom_approach_vector, (list, tuple))
                or len(custom_approach_vector) != 3
            ):
                return OperationResult.error_result(
                    "INVALID_APPROACH_VECTOR",
                    f"Custom approach vector must be a 3-element list [x, y, z], got: {custom_approach_vector}",
                    [
                        "Provide a valid 3D vector: [x, y, z]",
                        "Example: [0, 1, 0] for upward approach",
                    ],
                )

        # Signal intent for joint-attention and anticipatory refresh
        try:
            from core.Imports import get_world_state

            _intent_ws = get_world_state()
            if _intent_ws is not None:
                _intent_ws.update_robot_state(
                    robot_id, {"moving_toward_object": object_id}
                )
            try:
                from core.Imports import get_perception_refresh_daemon

                _daemon = get_perception_refresh_daemon()
                if _daemon is not None:
                    _daemon.trigger_anticipatory_refresh([object_id])
            except Exception:
                pass
        except Exception:
            pass

        _use_ros = use_ros
        if _use_ros is None:
            try:
                from config.ROS import DEFAULT_CONTROL_MODE, ROS_ENABLED

                _use_ros = ROS_ENABLED and DEFAULT_CONTROL_MODE in ("ros", "hybrid")
            except ImportError:
                _use_ros = False

        try:
            from config.Servers import VGN_ENABLED as _vgn_enabled
        except ImportError:
            _vgn_enabled = False

        bridge = None
        if _use_ros:
            try:
                from ros2.ROSBridge import ROSBridge

                bridge = ROSBridge.get_instance()
                if not bridge.is_connected and not bridge.connect():
                    should_fallback, err = _handle_ros_failure(
                        "Failed to connect to ROS bridge (port 5020)", "grasp_object"
                    )
                    if not should_fallback:
                        assert err is not None
                        return err
                    _use_ros = False

            except ImportError:
                logger.warning("ros2 module not available, falling back to TCP")
                _use_ros = False

        if _use_ros:
            try:
                from core.Imports import get_world_state

                world_state = get_world_state()
                object_position = world_state.get_object_position(object_id)

                logger.info(
                    f"[GRASP_DEBUG] {object_id} WorldState position: {object_position}"
                )

                if object_position is None:
                    # WorldState is empty - detection either wasn't run or failed silently.
                    # Log stored keys so the mismatch (e.g. "red" vs "red_cube") is visible.
                    stored_keys = list(world_state._objects.keys())
                    logger.warning(
                        f"Object '{object_id}' not found in WorldState "
                        f"(stored keys: {stored_keys}). "
                        f"Attempting inline detect_object_stereo before proceeding."
                    )

                    # Inline detection: derive color from object_id (e.g. "red_cube" → "red").
                    # This makes grasp_object self-healing when the LLM omits the detection step.
                    color_hint = (
                        object_id.split("_")[0] if "_" in object_id else object_id
                    )
                    try:
                        from operations.VisionOperations import detect_object_stereo

                        det_result = detect_object_stereo(
                            robot_id=robot_id,
                            color=color_hint,
                            camera_id="stereo",
                            request_id=request_id,
                        )
                        if det_result and det_result["success"]:
                            # Re-query WorldState - detection should have written the position
                            object_position = world_state.get_object_position(object_id)
                            if object_position is not None:
                                logger.info(
                                    f"Inline detection succeeded: '{object_id}' now at {object_position}"
                                )
                            else:
                                logger.warning(
                                    f"Inline detection returned success but '{object_id}' still "
                                    f"not in WorldState (stored: {list(world_state._objects.keys())})"
                                )
                        else:
                            err_msg = (
                                det_result["error"] if det_result else "no response"
                            )
                            logger.warning(f"Inline detection failed: {err_msg}")
                    except Exception as det_err:
                        logger.error(
                            f"Inline detection raised: {det_err}", exc_info=True
                        )

                if object_position is None:
                    should_fallback, err = _handle_ros_failure(
                        f"Object {object_id} not in WorldState after detection attempt - "
                        f"verify detect_object_stereo precedes grasp_object",
                        "grasp_object",
                    )
                    if not should_fallback:
                        assert err is not None
                        return err
                    _use_ros = False
                else:
                    logger.info(
                        f"Resolved {object_id} to position {object_position}, planning with ROS"
                    )

                    object_dimensions = world_state.get_object_dimensions(object_id)
                    robot_state = world_state.get_robot_state(robot_id)

                    # PATH 1: VGN+MoveIt (highest priority - provides 6-DOF orientation for angled targets)
                    if _vgn_enabled:
                        assert bridge is not None
                        result = _grasp_via_vgn_with_ros(
                            bridge=bridge,
                            robot_id=robot_id,
                            object_id=object_id,
                            preferred_approach=preferred_approach,
                            pre_grasp_distance=pre_grasp_distance,
                            request_id=request_id,
                            world_state=world_state,
                            custom_approach_vector=custom_approach_vector,
                            grasp_yaw_override=grasp_yaw_override,
                        )
                        if result is not None:
                            return result

                    # PATH 2: Geometric ROS (GraspPlanner when dimensions + robot pose available)
                    if (
                        object_dimensions is not None
                        and robot_state is not None
                        and robot_state.position is not None
                    ):
                        logger.info(f"Using grasp planning pipeline for {object_id}")
                        assert bridge is not None
                        ros_result, fallback = _grasp_via_ros_planned(
                            bridge=bridge,
                            robot_id=robot_id,
                            object_id=object_id,
                            object_position=object_position,
                            object_dimensions=object_dimensions,
                            robot_state=robot_state,
                            preferred_approach=preferred_approach.lower(),
                            request_id=request_id,
                            world_state=world_state,
                            grasp_yaw_override=grasp_yaw_override,
                            pre_grasp_distance=pre_grasp_distance,
                        )
                        if not fallback:
                            assert ros_result is not None
                            return ros_result
                        # fallback=True: fall through to position-only ROS

                    # Position-only ROS path (no dimensions or GraspPlanner fallback)
                    logger.info(f"Using position-only planning for {object_id}")
                    assert bridge is not None
                    ros_result, fallback = _grasp_via_ros_position_only(
                        bridge=bridge,
                        robot_id=robot_id,
                        object_id=object_id,
                        object_position=object_position,
                        request_id=request_id,
                        world_state=world_state,
                        grasp_yaw_override=grasp_yaw_override,
                    )
                    if not fallback:
                        assert ros_result is not None
                        return ros_result
                    _use_ros = False  # TCP fallback

            except Exception as e:
                logger.error(f"Error resolving object position for ROS: {e}")
                should_fallback, err = _handle_ros_failure(
                    f"Error preparing ROS grasp: {str(e)}", "grasp_object"
                )
                if not should_fallback:
                    assert err is not None
                    return err
                _use_ros = False

        # VGN TCP path (falls back to geometric on unavailability/failure)
        if _vgn_enabled:
            vgn_result = _grasp_via_vgn(
                robot_id=robot_id,
                object_id=object_id,
                preferred_approach=preferred_approach,
                use_advanced_planning=use_advanced_planning,
                pre_grasp_distance=pre_grasp_distance,
                enable_retreat=enable_retreat,
                retreat_distance=retreat_distance,
                request_id=request_id,
                custom_approach_vector=custom_approach_vector,
            )
            if vgn_result is not None:
                return vgn_result
            logger.info(
                "[VGN] Unavailable or returned no candidates - "
                "falling back to geometric pipeline"
            )

        # TCP path: resolve canonical WorldState key (Unity obj.name) so FindObjectFlexible
        # receives "Red Cube" not LLM-generated "redCube".
        try:
            from core.Imports import get_world_state as _get_ws

            _ws = _get_ws()
            if _ws is not None:
                _canonical = _ws.resolve_canonical_id(object_id)
                if _canonical and _canonical != object_id:
                    logger.debug(
                        f"grasp_object: resolved object_id '{object_id}' → '{_canonical}' via WorldState"
                    )
                    object_id = _canonical
        except Exception as _resolve_err:
            logger.debug(f"grasp_object: canonical resolution skipped: {_resolve_err}")

        parameters = {
            "object_id": object_id,
            "use_advanced_planning": use_advanced_planning,
            "preferred_approach": preferred_approach.lower(),
            "pre_grasp_distance": pre_grasp_distance,
            "enable_retreat": enable_retreat,
            "retreat_distance": retreat_distance,
        }

        if custom_approach_vector is not None:
            parameters["custom_approach_vector"] = {
                "x": custom_approach_vector[0],
                "y": custom_approach_vector[1],
                "z": custom_approach_vector[2],
            }

        command = {
            "command_type": "grasp_object",
            "target_type": "robot",
            "robot_id": robot_id,
            "parameters": parameters,
            "request_id": request_id,
        }

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

        logger.info(f"Sending grasp_object command: {robot_id} -> {object_id}")
        completion = broadcaster.send_command_and_wait(command, timeout=45.0)

        if completion is None:
            logger.error(f"grasp_object timed out or send failed: {robot_id} -> {object_id}")
            return OperationResult.error_result(
                "TIMEOUT",
                "grasp_object did not complete within timeout (Unity unreachable or grasp stalled)",
                [
                    "Check Unity is connected to CommandServer",
                    "Verify the robot is not blocked or frozen by ProximityGuard",
                ],
            )

        if not completion.get("success", False):
            logger.warning(f"grasp_object failed (object not held): {robot_id} -> {object_id}")
            return OperationResult.error_result(
                "GRASP_FAILED",
                f"Grasp executed but object '{object_id}' is not held",
                [
                    "Verify object position is within robot reach (0.6 m)",
                    "Run detect_object_stereo before grasp_object to refresh pose",
                    "Try a different preferred_approach (top/front/side)",
                ],
            )

        logger.info(f"grasp_object succeeded: {robot_id} holding {object_id}")
        return OperationResult.success_result(
            {
                "status": "tcp_executed",
                "robot_id": robot_id,
                "object_id": object_id,
                "request_id": request_id,
                "is_holding_object": True,
            }
        )

    except Exception as e:
        logger.exception(f"Exception in grasp_object operation: {e}")
        return OperationResult.error_result(
            "EXCEPTION",
            str(e),
            [
                "Check stack trace in logs",
                "Verify all parameters are correct",
                "Ensure Unity is running and responsive",
            ],
        )
    finally:
        # Clear movement intent regardless of outcome
        try:
            from core.Imports import get_world_state as _get_ws

            _ws = _get_ws()
            if _ws is not None:
                _ws.update_robot_state(robot_id, {"moving_toward_object": None})
        except Exception:
            pass


GRASP_OBJECT_OPERATION = BasicOperation(
    operation_id="manipulation_grasp_object_001",
    name="grasp_object",
    category=OperationCategory.MANIPULATION,
    complexity=OperationComplexity.COMPLEX,
    description="Plan and execute grasp using MoveIt2-inspired pipeline with candidate generation, IK validation, collision checking, and scoring",
    usage_examples=[
        "Grasp object top-down (default): grasp_object(robot_id='Robot1', object_id='Cube_01')",
        "Grasp from specific direction: grasp_object(robot_id='Robot1', object_id='Cube_01', preferred_approach='side')",
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
            description="Preferred grasp approach: 'top' (default, top-down), 'front', 'side' (any horizontal), 'left_side' (left end of longest object axis), 'right_side' (right end of longest object axis), 'auto'",
            required=False,
            default="top",
            valid_values=[
                "auto",
                "top",
                "front",
                "side",
                "left_side",
                "right_side",
                "left",
                "right",
            ],
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
            description="Custom approach direction [x, y, z] (overrides preferred_approach). OMIT entirely when not needed - do NOT pass an empty list [].",
            required=False,
            default=None,
        ),
    ],
    preconditions=[
        "robot_is_initialized(robot_id)",
    ],
    postconditions=[],
    average_duration_ms=150.0,
    success_rate=0.92,
    failure_modes=[
        "No valid grasp candidates found (all filtered out)",
        "IK validation failed for all candidates",
        "All approach paths have collisions",
        "Object not found in scene",
        "Object outside robot reach",
    ],
    relationships=OperationRelationship(
        operation_id="manipulation_grasp_object_001",
        required_operations=[
            "perception_stereo_detect_001",
            "status_check_robot_001",
        ],
        required_reasons={
            "perception_stereo_detect_001": "Object must be detected with 3D world coordinates before grasp planning can begin",
            "status_check_robot_001": "Robot must be initialized and responsive before executing complex grasp pipeline",
        },
        commonly_paired_with=[
            "manipulation_control_gripper_001",
            "motion_move_to_coord_001",
            "motion_return_to_start_001",
        ],
        pairing_reasons={
            "manipulation_control_gripper_001": "Open gripper before approach, close after grasping for controlled pickup",
            "motion_move_to_coord_001": "Move to pre-grasp position before executing final grasp",
            "motion_return_to_start_001": "Return to safe position after successful grasp with object in hand",
        },
        typical_after=[
            "perception_stereo_detect_001",
            "status_check_robot_001",
            "motion_move_to_coord_001",
        ],
        typical_before=[
            "motion_move_to_coord_001",
            "manipulation_control_gripper_001",
            "motion_return_to_start_001",
        ],
    ),
    implementation=grasp_object,
)
