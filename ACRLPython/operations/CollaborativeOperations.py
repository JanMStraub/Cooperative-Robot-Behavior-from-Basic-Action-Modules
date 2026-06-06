#!/usr/bin/env python3
"""
Collaborative Manipulation Operations (Level 5): stabilize_object (ATOMIC).

Non-atomic stabilize_and_manipulate_collaboratively removed — use WorkflowPatterns.STABILIZE_MANIPULATE_PATTERN.
"""

import time
import logging
from typing import Optional

from ._imports import get_command_broadcaster as _get_command_broadcaster
from .Base import (
    BasicOperation,
    OperationCategory,
    OperationComplexity,
    OperationParameter,
    OperationResult,
    OperationRelationship,
)

logger = logging.getLogger(__name__)


def _kg_both_robots_can_reach(object_id: str, _robot_id: str) -> Optional[str]:
    try:
        from config.KnowledgeGraph import KNOWLEDGE_GRAPH_ENABLED

        if not KNOWLEDGE_GRAPH_ENABLED:
            return None

        from core.Imports import get_graph_query_engine

        qe = get_graph_query_engine()
        if qe is None:
            return None

        reachable = qe.find_reachable_robots(object_id)
        if reachable and len(reachable) < 2:
            return (
                f"KG: only {len(reachable)} robot(s) can reach '{object_id}' "
                f"(reachable: {reachable}); stabilize_object requires both arms"
            )
        return None

    except Exception as e:
        logger.debug(f"KG reachability check skipped: {e}")
        return None


def stabilize_object(
    robot_id: str,
    object_id: str,
    duration_ms: int = 5000,
    force_limit: float = 10.0,
    request_id: int = 0,
    use_ros: Optional[bool] = None,
) -> OperationResult:
    """Hold object stable while partner robot manipulates. Force control handled Unity-side."""
    try:
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string",
                ["Provide a valid robot ID"],
            )

        if not object_id or not isinstance(object_id, str):
            return OperationResult.error_result(
                "INVALID_OBJECT_ID",
                f"Object ID must be a non-empty string",
                ["Provide a valid object ID"],
            )

        if not (100 <= duration_ms <= 30000):
            return OperationResult.error_result(
                "INVALID_DURATION",
                f"Duration must be in range [100, 30000]ms, got: {duration_ms}",
                ["Use duration between 100ms and 30000ms (30s)"],
            )

        if not (1.0 <= force_limit <= 50.0):
            return OperationResult.error_result(
                "INVALID_FORCE_LIMIT",
                f"Force limit must be in range [1.0, 50.0]N, got: {force_limit}",
                ["Use force limit between 1N and 50N"],
            )

        kg_warning = _kg_both_robots_can_reach(object_id, robot_id)
        if kg_warning:
            logger.warning(kg_warning)

        _use_ros = use_ros
        if _use_ros is None:
            try:
                from config.ROS import ROS_ENABLED, DEFAULT_CONTROL_MODE

                _use_ros = ROS_ENABLED and DEFAULT_CONTROL_MODE in ("ros", "hybrid")
            except ImportError:
                _use_ros = False

        # Stabilisation needs continuous force feedback loops → Unity TCP path always
        if _use_ros:
            logger.info(
                "Stabilization force control via ROS not yet implemented - using Unity direct control"
            )
            _use_ros = False

        command = {
            "command_type": "stabilize_object",
            "robot_id": robot_id,
            "parameters": {
                "object_id": object_id,
                "duration_ms": duration_ms,
                "force_limit": force_limit,
            },
            "timestamp": time.time(),
            "request_id": request_id,
        }

        logger.info(
            f"Sending stabilize_object command to {robot_id}: {object_id} for {duration_ms}ms"
        )

        success = _get_command_broadcaster().send_command(command, request_id)

        if not success:
            return OperationResult.error_result(
                "COMMUNICATION_FAILED",
                "Failed to send command to Unity",
                ["Ensure Unity is running"],
            )

        logger.info(f"Successfully activated stabilization for {robot_id}")

        return OperationResult.success_result(
            {
                "robot_id": robot_id,
                "object_id": object_id,
                "duration_ms": duration_ms,
                "force_limit": force_limit,
                "status": "stabilizing",
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.error(f"Unexpected error in stabilize_object: {e}", exc_info=True)
        return OperationResult.error_result(
            "UNEXPECTED_ERROR",
            f"Unexpected error occurred: {str(e)}",
            ["Check logs"],
        )


# REMOVED: stabilize_and_manipulate_collaboratively
# This operation was REMOVED because it is non-atomic (combines grasp + hold + manipulate).
# For collaborative manipulation workflows, see operations/WorkflowPatterns.py for the
# STABILIZE_MANIPULATE_PATTERN showing how to chain atomic operations.
#
# The LLM should chain operations:
# Robot1 (Stabilizer):
# 1. move_to_coordinate(robot1, object_position)
# 2. grasp_object(robot1, object_coords)
# 3. stabilize_object(robot1, object_id, duration_ms=10000)
#
# Robot2 (Manipulator) - runs in parallel:
# 1. wait_for_signal(robot2, "stabilization_active")
# 2. move_to_coordinate(robot2, manipulation_position)
# 3. [perform manipulation operation]
# 4. signal(robot2, "manipulation_complete")
#
# Robot1 releases when done:
# 5. wait_for_signal(robot1, "manipulation_complete")
# 6. control_gripper(robot1, open=True)


def create_stabilize_object_operation() -> BasicOperation:
    return BasicOperation(
        operation_id="collaborative_stabilize_001",
        name="stabilize_object",
        category=OperationCategory.MANIPULATION,
        complexity=OperationComplexity.COMPLEX,
        description="Hold object stable while partner robot manipulates it",
        long_description="""
            Commands a robot to grasp and hold an object stable, providing support
            while a partner robot performs manipulation on the same object.

            Trigger phrases: "keep it stable", "hold the object still", "keep it steady
            while Robot2 works", "stabilize while partner manipulates", "brace the object",
            "hold it in place", "support the object for the other robot", "don't move it".

            Requires force control to maintain grip without crushing. The holding robot
            stays stationary for the full duration while the partner manipulates.
            Critical for assembly, insertion, and precision placement tasks.
        """,
        usage_examples=[
            "stabilize_object('Robot1', 'LargeCube', duration_ms=5000)",
            "Robot1 holds board while Robot2 inserts pegs",
            "Bimanual assembly: one robot stabilizes, other assembles",
        ],
        parameters=[
            OperationParameter(
                name="robot_id",
                type="str",
                description="Stabilizing robot ID",
                required=True,
            ),
            OperationParameter(
                name="object_id",
                type="str",
                description="Object to stabilize",
                required=True,
            ),
            OperationParameter(
                name="duration_ms",
                type="int",
                description="Stabilization duration (ms)",
                required=False,
                default=5000,
                valid_range=(100, 30000),
            ),
            OperationParameter(
                name="force_limit",
                type="float",
                description="Maximum grip force (Newtons)",
                required=False,
                default=10.0,
                valid_range=(1.0, 50.0),
            ),
        ],
        preconditions=[
            "robot_is_initialized(robot_id)",
        ],
        postconditions=[],
        average_duration_ms=5000.0,  # Depends on duration parameter
        success_rate=0.88,
        failure_modes=[
            "Object slips during stabilization",
            "Insufficient grip force",
            "Position drift during hold",
        ],
        relationships=OperationRelationship(
            operation_id="collaborative_stabilize_001",
            required_operations=[
                "manipulation_control_gripper_001",
                "motion_move_to_coord_001",
            ],
            required_reasons={
                "manipulation_control_gripper_001": "Must grip object to stabilize",
                "motion_move_to_coord_001": "Must position at object before gripping",
            },
            commonly_paired_with=[
                "sync_signal_001",
                "sync_wait_for_signal_001",
            ],
            pairing_reasons={
                "sync_signal_001": "Signal partner when stabilization active",
                "sync_wait_for_signal_001": "Wait for partner to complete manipulation",
            },
            typical_before=[],
            typical_after=["manipulation_control_gripper_001"],
        ),
        implementation=stabilize_object,
    )


STABILIZE_OBJECT_OPERATION = create_stabilize_object_operation()
# STABILIZE_AND_MANIPULATE_OPERATION removed - use WorkflowPatterns.STABILIZE_MANIPULATE_PATTERN instead


def place_for_partner(
    robot_id: str,
    zone_id: str = "shared_zone",
    signal_name: Optional[str] = None,
    request_id: int = 0,
    use_ros: Optional[bool] = None,
) -> OperationResult:
    """Place a held object at a shared zone for the partner robot to pick up, then signal readiness."""
    try:
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                "Robot ID must be a non-empty string",
                ["Provide a valid robot ID"],
            )

        if not zone_id or not isinstance(zone_id, str):
            return OperationResult.error_result(
                "INVALID_ZONE_ID",
                "Zone ID must be a non-empty string",
                ["Provide a valid zone ID"],
            )

        try:
            from ._imports import WORKSPACE_REGIONS, PLACE_MIN_Y
        except ImportError:
            from operations._imports import WORKSPACE_REGIONS, PLACE_MIN_Y  # type: ignore[no-redef]

        try:
            from ..config.Robot import HANDOFF_PRESENTATION_POSITION
        except ImportError:
            from config.Robot import HANDOFF_PRESENTATION_POSITION  # type: ignore[no-redef]

        if zone_id == "shared_zone":
            x, y, z = HANDOFF_PRESENTATION_POSITION
        else:
            zone = WORKSPACE_REGIONS.get(zone_id)
            if zone is None:
                return OperationResult.error_result(
                    "UNKNOWN_ZONE",
                    f"Zone '{zone_id}' not found in WORKSPACE_REGIONS",
                    [f"Valid zones: {list(WORKSPACE_REGIONS.keys())}"],
                )
            x = (zone["x_min"] + zone["x_max"]) / 2
            z = (zone["z_min"] + zone["z_max"]) / 2
            y = PLACE_MIN_Y

        try:
            from .GripperOperations import place_object
        except ImportError:
            from operations.GripperOperations import place_object  # type: ignore[no-redef]

        result = place_object(robot_id, x, y, z, request_id=request_id, use_ros=use_ros)

        if result.success:
            effective_signal = signal_name or f"object_ready_at_{zone_id}"

            try:
                from .SyncOperations import EventBus
            except ImportError:
                from operations.SyncOperations import EventBus  # type: ignore[no-redef]

            EventBus().signal(effective_signal)

            result_data = dict(result.result or {})
            result_data["partner_signal"] = effective_signal
            result_data["zone_id"] = zone_id
            return OperationResult.success_result(result_data)

        return result

    except Exception as e:
        logger.error(f"Unexpected error in place_for_partner: {e}", exc_info=True)
        return OperationResult.error_result(
            "UNEXPECTED_ERROR",
            f"Unexpected error occurred: {str(e)}",
            ["Check logs"],
        )


def create_place_for_partner_operation() -> BasicOperation:
    return BasicOperation(
        operation_id="collaborative_place_for_partner_001",
        name="place_for_partner",
        category=OperationCategory.MANIPULATION,
        complexity=OperationComplexity.INTERMEDIATE,
        description="Place a held object at a shared zone for the partner robot to pick up, then signal readiness",
        long_description="""
            Moves to the target shared zone, places the held object there, then emits a
            signal (default: object_ready_at_<zone_id>) so the partner robot knows it can
            proceed to pick the object up. Avoids the need for a live simultaneous handoff.
        """,
        usage_examples=[
            "place_for_partner('Robot1') — places at shared_zone center, signals object_ready_at_shared_zone",
            "place_for_partner('Robot1', zone_id='center', signal_name='cube_dropped')",
        ],
        preconditions=[
            "robot_is_initialized(robot_id)",
            "robot_is_holding_object(robot_id)",
        ],
        postconditions=[
            "object_at_zone_center(zone_id)",
            "partner_notified_via_signal",
        ],
        parameters=[
            OperationParameter(
                name="robot_id",
                type="str",
                description="Robot ID performing the placement",
                required=True,
            ),
            OperationParameter(
                name="zone_id",
                type="str",
                description="Target shared zone ID from WORKSPACE_REGIONS",
                required=False,
                default="shared_zone",
            ),
            OperationParameter(
                name="signal_name",
                type="str",
                description="Override signal name (default: object_ready_at_<zone_id>)",
                required=False,
                default=None,
            ),
        ],
        average_duration_ms=3000.0,
        success_rate=0.87,
        failure_modes=[
            "Place operation fails",
            "Robot not holding an object",
            "Zone out of reach",
        ],
        relationships=OperationRelationship(
            operation_id="collaborative_place_for_partner_001",
            required_operations=["manipulation_grasp_object_001"],
            required_reasons={
                "manipulation_grasp_object_001": "Robot must be holding object before placing for partner",
            },
            commonly_paired_with=[
                "sync_wait_for_signal_001",
                "coordination_yield_workspace_002",
                "coordination_check_partner_001",
            ],
            pairing_reasons={
                "sync_wait_for_signal_001": "Partner waits for object_ready_at_<zone> signal",
                "coordination_yield_workspace_002": "Yield shared zone before placing",
                "coordination_check_partner_001": "Check partner is idle before placing for pickup",
            },
            typical_before=["sync_signal_001"],
            typical_after=["manipulation_grasp_object_001"],
        ),
        implementation=place_for_partner,
    )


PLACE_FOR_PARTNER_OPERATION = create_place_for_partner_operation()
