#!/usr/bin/env python3
"""
Multi-Robot Coordination Operations (Level 4): detect_other_robot, mirror_movement_of_other_robot (ATOMIC).

hand_over_object_to_another_robot removed (non-atomic) — use WorkflowPatterns.HANDOFF_PATTERN.
"""

import time
import logging

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


def _require_str(val, code, label):
    if not val or not isinstance(val, str):
        return OperationResult.error_result(code, f"{label} must be a non-empty string")
    return None


def _field(state, key, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _xyz(pos):
    if isinstance(pos, dict):
        return pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)
    return pos[0], pos[1], pos[2]


def detect_other_robot(
    robot_id: str,
    target_robot_id: str,
    camera_id: str = "main",
    request_id: int = 0,
) -> OperationResult:
    try:
        err = _require_str(robot_id, "INVALID_ROBOT_ID", "robot_id")
        if err:
            return err
        err = _require_str(
            target_robot_id, "INVALID_TARGET_ROBOT_ID", "target_robot_id"
        )
        if err:
            return err

        from .WorldState import WorldState

        world_state = WorldState()

        target_state = world_state.get_robot_state(target_robot_id)
        if not target_state:
            return OperationResult.error_result(
                "TARGET_ROBOT_NOT_FOUND",
                f"Target robot '{target_robot_id}' not found in world state",
            )

        detector_state = world_state.get_robot_state(robot_id)
        if not detector_state:
            return OperationResult.error_result(
                "DETECTOR_ROBOT_NOT_FOUND",
                f"Detecting robot '{robot_id}' not found in world state",
            )

        import math

        detector_pos = _field(detector_state, "end_effector_position") or _field(
            detector_state, "position"
        )
        target_pos = _field(target_state, "end_effector_position") or _field(
            target_state, "position"
        )

        if not detector_pos or not target_pos:
            return OperationResult.error_result(
                "POSITION_DATA_MISSING",
                "Robot position data missing",
            )

        dx, dy, dz = tuple(a - b for a, b in zip(_xyz(detector_pos), _xyz(target_pos)))
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        logger.info(
            f"Robot {robot_id} detected {target_robot_id} at distance {distance:.3f}m"
        )

        # KG enrichment is additive — safe to skip if disabled
        kg_proximity = None
        try:
            from config.KnowledgeGraph import KNOWLEDGE_GRAPH_ENABLED

            if KNOWLEDGE_GRAPH_ENABLED:
                from core.Imports import get_graph_query_engine

                qe = get_graph_query_engine()
                if qe is not None:
                    nearby = qe.find_robots_near(robot_id, max_distance=distance + 0.05)
                    kg_proximity = [
                        r for r in nearby if r["robot_id"] == target_robot_id
                    ]
        except Exception as e:
            logger.debug(f"KG proximity enrichment skipped: {e}")

        result_data = {
            "robot_id": robot_id,
            "target_robot_id": target_robot_id,
            "position": target_pos,
            "distance": distance,
            "detected": True,
            "camera_id": camera_id,
            "timestamp": time.time(),
        }
        if kg_proximity is not None:
            result_data["kg_proximity"] = kg_proximity

        return OperationResult.success_result(result_data)

    except Exception as e:
        logger.error(f"Error in detect_other_robot: {e}", exc_info=True)
        return OperationResult.error_result(
            "DETECTION_ERROR", f"Robot detection failed: {e}"
        )


def mirror_movement_of_other_robot(
    robot_id: str,
    target_robot_id: str,
    mirror_axis: str = "x",
    scale_factor: float = 1.0,
    duration_ms: int = 10000,
    request_id: int = 0,
) -> OperationResult:
    try:
        err = _require_str(robot_id, "INVALID_ROBOT_ID", "robot_id")
        if err:
            return err
        err = _require_str(
            target_robot_id, "INVALID_TARGET_ROBOT_ID", "target_robot_id"
        )
        if err:
            return err

        valid_axes = ["x", "y", "z", "none"]
        if mirror_axis not in valid_axes:
            return OperationResult.error_result(
                "INVALID_MIRROR_AXIS",
                f"mirror_axis must be one of {valid_axes}, got: {mirror_axis}",
            )

        # negative values invert direction (reflection)
        if not (0.1 <= abs(scale_factor) <= 2.0):
            return OperationResult.error_result(
                "INVALID_SCALE_FACTOR",
                f"scale_factor must be in [0.1, 2.0] (negative inverts), got: {scale_factor}",
            )

        if not (1000 <= duration_ms <= 60000):
            return OperationResult.error_result(
                "INVALID_DURATION",
                f"duration_ms must be in [1000, 60000], got: {duration_ms}",
            )

        command = {
            "command_type": "mirror_movement",
            "robot_id": robot_id,
            "parameters": {
                "target_robot_id": target_robot_id,
                "mirror_axis": mirror_axis,
                "scale_factor": scale_factor,
                "duration_ms": duration_ms,
            },
            "timestamp": time.time(),
            "request_id": request_id,
        }

        logger.info(
            f"Sending mirror_movement command: {robot_id} mirrors {target_robot_id} for {duration_ms}ms"
        )

        success = _get_command_broadcaster().send_command(command, request_id)

        if not success:
            return OperationResult.error_result(
                "COMMUNICATION_FAILED",
                "Failed to send command to Unity",
            )

        logger.info(f"Successfully activated mirroring for {robot_id}")

        return OperationResult.success_result(
            {
                "robot_id": robot_id,
                "target_robot_id": target_robot_id,
                "mirror_axis": mirror_axis,
                "scale_factor": scale_factor,
                "duration_ms": duration_ms,
                "status": "mirroring_active",
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.error(f"Unexpected error in mirror_movement: {e}", exc_info=True)
        return OperationResult.error_result(
            "UNEXPECTED_ERROR", f"mirror_movement failed: {e}"
        )


def create_detect_other_robot_operation() -> BasicOperation:
    return BasicOperation(
        operation_id="coordination_detect_robot_001",
        name="detect_other_robot",
        category=OperationCategory.PERCEPTION,
        complexity=OperationComplexity.INTERMEDIATE,
        description="Detect and locate another robot in shared workspace",
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
        preconditions=["robot_is_initialized(robot_id)"],
        postconditions=[],
        average_duration_ms=80,
        success_rate=0.96,
        failure_modes=["Target robot not in view", "WorldState not updated"],
        relationships=OperationRelationship(
            operation_id="coordination_detect_robot_001",
            required_operations=["status_check_robot_001"],
            required_reasons={
                "status_check_robot_001": "Verify this robot is active before attempting inter-robot detection",
            },
            commonly_paired_with=[
                "coordination_mirror_movement_002",
                "motion_move_to_coord_001",
                "sync_signal_001",
            ],
            pairing_reasons={
                "coordination_mirror_movement_002": "Detect robot position before enabling mirrored movement",
                "motion_move_to_coord_001": "Move relative to other robot's detected position",
                "sync_signal_001": "Signal readiness for coordination after detecting peer",
            },
            typical_before=["coordination_mirror_movement_002", "sync_signal_001"],
            typical_after=["status_check_robot_001"],
        ),
        implementation=detect_other_robot,
    )


def create_mirror_movement_operation() -> BasicOperation:
    return BasicOperation(
        operation_id="coordination_mirror_movement_002",
        name="mirror_movement_of_other_robot",
        category=OperationCategory.NAVIGATION,
        complexity=OperationComplexity.COMPLEX,
        description="Mirror the movements of another robot for a configurable duration",
        usage_examples=[
            "mirror_movement_of_other_robot('Robot2', 'Robot1', 'x', duration_ms=10000)",
            "mirror_movement_of_other_robot('Robot2', 'Robot1', 'none', duration_ms=5000)",
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
            OperationParameter(
                name="duration_ms",
                type="int",
                description="Duration to run mirroring in milliseconds (1000-60000)",
                required=False,
                default=10000,
                valid_range=(1000, 60000),
            ),
        ],
        preconditions=["robot_is_initialized(robot_id)"],
        postconditions=[],
        average_duration_ms=10000,
        success_rate=0.92,
        failure_modes=["Workspace collision", "Robot limits exceeded"],
        relationships=OperationRelationship(
            operation_id="coordination_mirror_movement_002",
            required_operations=["coordination_detect_robot_001"],
            required_reasons={
                "coordination_detect_robot_001": "Must know other robot's position and proximity before enabling mirroring",
            },
            commonly_paired_with=[
                "coordination_detect_robot_001",
                "sync_signal_001",
                "sync_wait_for_signal_001",
            ],
            pairing_reasons={
                "coordination_detect_robot_001": "Detect target robot first to establish baseline position",
                "sync_signal_001": "Signal when mirroring is active so other robot can proceed",
                "sync_wait_for_signal_001": "Wait for peer readiness before starting synchronized movement",
            },
            typical_before=["sync_signal_001"],
            typical_after=["coordination_detect_robot_001", "sync_wait_for_signal_001"],
        ),
        implementation=mirror_movement_of_other_robot,
    )


def check_partner_status(
    robot_id: str,
    partner_robot_id: str,
    request_id: int = 0,
) -> OperationResult:
    """Query a partner robot's full state before planning a joint task.

    Pure WorldState read — no Unity command sent.
    """
    import math

    try:
        err = _require_str(robot_id, "INVALID_ROBOT_ID", "robot_id")
        if err:
            return err
        err = _require_str(
            partner_robot_id, "INVALID_PARTNER_ROBOT_ID", "partner_robot_id"
        )
        if err:
            return err
        if robot_id == partner_robot_id:
            return OperationResult.error_result(
                "INVALID_PARTNER_ROBOT_ID",
                "partner_robot_id must differ from robot_id",
            )

        from .WorldState import WorldState

        world_state = WorldState()

        partner_state = world_state.get_robot_state(partner_robot_id)
        if not partner_state:
            return OperationResult.error_result(
                "PARTNER_NOT_FOUND",
                f"Partner robot '{partner_robot_id}' not found in world state",
            )

        gripper_state = _field(partner_state, "gripper_state", "unknown")
        is_moving = _field(partner_state, "is_moving", False)
        is_initialized = _field(partner_state, "is_initialized", False)
        position = _field(partner_state, "position")
        moving_toward_object = _field(partner_state, "moving_toward_object")
        workspace_intent = _field(partner_state, "workspace_intent")
        proximity_frozen = _field(partner_state, "proximity_frozen", False)

        distance = None
        self_state = world_state.get_robot_state(robot_id)
        if self_state is not None:
            self_pos = _field(self_state, "position")
            if self_pos is not None and position is not None:
                sx, sy, sz = _xyz(self_pos)
                px, py, pz = _xyz(position)
                distance = math.sqrt((sx - px) ** 2 + (sy - py) ** 2 + (sz - pz) ** 2)

        if is_moving:
            status = "busy"
        elif gripper_state == "closed":
            status = "holding_object"
        else:
            status = "idle"

        logger.info(
            f"check_partner_status: {robot_id} queried {partner_robot_id} — "
            f"status={status}, distance={distance}"
        )

        return OperationResult.success_result(
            {
                "partner_robot_id": partner_robot_id,
                "gripper_state": gripper_state,
                "is_moving": is_moving,
                "is_initialized": is_initialized,
                "position": list(position) if position is not None else None,
                "moving_toward_object": moving_toward_object,
                "workspace_intent": workspace_intent,
                "proximity_frozen": proximity_frozen,
                "distance": distance,
                "status": status,
                "has_object": gripper_state == "closed",
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.error(f"Error in check_partner_status: {e}", exc_info=True)
        return OperationResult.error_result(
            "CHECK_PARTNER_STATUS_ERROR", f"Failed to check partner status: {e}"
        )


def create_check_partner_status_operation() -> BasicOperation:
    return BasicOperation(
        operation_id="coordination_check_partner_001",
        name="check_partner_status",
        category=OperationCategory.COORDINATION,
        complexity=OperationComplexity.INTERMEDIATE,
        description=(
            "Query a partner robot's full state (gripper, motion, position) before planning a joint task. "
            "Trigger phrases: 'check if partner is ready', 'is Robot2 ready', 'verify partner before starting', "
            "'confirm partner status', 'is the other robot free'."
        ),
        usage_examples=[
            "check_partner_status('Robot1', 'Robot2')",
            "if result['status'] == 'idle': proceed with joint task",
            "if result['has_object']: plan handoff receive workflow",
        ],
        parameters=[
            OperationParameter(
                name="robot_id",
                type="str",
                description="ID of the querying robot",
                required=True,
            ),
            OperationParameter(
                name="partner_robot_id",
                type="str",
                description="ID of the partner robot to query",
                required=True,
            ),
        ],
        average_duration_ms=20.0,
        relationships=OperationRelationship(
            operation_id="coordination_check_partner_001",
            required_operations=["status_check_robot_001"],
            commonly_paired_with=[
                "coordination_detect_robot_001",
                "sync_signal_001",
                "sync_wait_for_signal_001",
            ],
        ),
        implementation=check_partner_status,
    )


DETECT_OTHER_ROBOT_OPERATION = create_detect_other_robot_operation()
MIRROR_MOVEMENT_OPERATION = create_mirror_movement_operation()
CHECK_PARTNER_STATUS_OPERATION = create_check_partner_status_operation()
