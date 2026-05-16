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


def detect_other_robot(
    robot_id: str,
    target_robot_id: str,
    camera_id: str = "main",
    request_id: int = 0,
) -> OperationResult:
    try:
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string",
                ["Provide a valid robot ID"],
            )

        if not target_robot_id or not isinstance(target_robot_id, str):
            return OperationResult.error_result(
                "INVALID_TARGET_ROBOT_ID",
                f"Target robot ID must be a non-empty string",
                ["Provide a valid target robot ID"],
            )

        try:
            from .WorldState import WorldState
        except ImportError:
            from operations.WorldState import WorldState

        world_state = WorldState()

        target_state = world_state.get_robot_state(target_robot_id)
        if not target_state:
            return OperationResult.error_result(
                "TARGET_ROBOT_NOT_FOUND",
                f"Target robot '{target_robot_id}' not found in world state",
                [
                    "Verify target robot is active in Unity",
                    "Check WorldStatePublisher is sending data",
                ],
            )

        detector_state = world_state.get_robot_state(robot_id)
        if not detector_state:
            return OperationResult.error_result(
                "DETECTOR_ROBOT_NOT_FOUND",
                f"Detecting robot '{robot_id}' not found in world state",
                ["Verify robot is active"],
            )

        import math

        # dict mocks for tests, RobotState dataclass in production
        if isinstance(detector_state, dict):
            detector_pos = detector_state.get(
                "end_effector_position"
            ) or detector_state.get("position")
        else:
            detector_pos = getattr(detector_state, "position", None)

        if isinstance(target_state, dict):
            target_pos = target_state.get("end_effector_position") or target_state.get(
                "position"
            )
        else:
            target_pos = getattr(target_state, "position", None)

        if not detector_pos or not target_pos:
            return OperationResult.error_result(
                "POSITION_DATA_MISSING",
                "Robot position data missing",
                ["Ensure WorldStatePublisher is active"],
            )

        # handle both tuple coords and dict mocks
        def _xyz(pos):
            if isinstance(pos, dict):
                return pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)
            return pos[0], pos[1], pos[2]

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
            "DETECTION_ERROR",
            f"Robot detection failed: {str(e)}",
            ["Check logs for details"],
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
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string",
                ["Provide a valid robot ID"],
            )

        if not target_robot_id or not isinstance(target_robot_id, str):
            return OperationResult.error_result(
                "INVALID_TARGET_ROBOT_ID",
                f"Target robot ID must be a non-empty string",
                ["Provide a valid target robot ID"],
            )

        valid_axes = ["x", "y", "z", "none"]
        if mirror_axis not in valid_axes:
            return OperationResult.error_result(
                "INVALID_MIRROR_AXIS",
                f"mirror_axis must be one of {valid_axes}, got: {mirror_axis}",
                [f"Use one of: {', '.join(valid_axes)}"],
            )

        # negative values invert direction (reflection)
        if not (0.1 <= abs(scale_factor) <= 2.0):
            return OperationResult.error_result(
                "INVALID_SCALE_FACTOR",
                f"scale_factor magnitude must be in range [0.1, 2.0], got: {scale_factor}",
                [
                    "Use scale between 0.1 (10%) and 2.0 (200%), negative values invert direction"
                ],
            )

        if not (1000 <= duration_ms <= 60000):
            return OperationResult.error_result(
                "INVALID_DURATION",
                f"duration_ms must be in range [1000, 60000], got: {duration_ms}",
                ["Use duration between 1000ms (1s) and 60000ms (60s)"],
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
                ["Ensure Unity is running"],
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
            "UNEXPECTED_ERROR",
            f"Unexpected error occurred: {str(e)}",
            ["Check logs"],
        )


# REMOVED: hand_over_object_to_another_robot
# This operation was REMOVED because it is non-atomic (combines 5 steps).
# For handoff workflows, see operations/WorkflowPatterns.py for the
# HANDOFF_PATTERN showing how to chain atomic operations.
#
# The LLM should chain operations:
# 1. move_to_coordinate(robot_from, handoff_position)
# 2. signal(robot_from, "ready_for_handoff")
# 3. wait_for_signal(robot_to, "ready_for_handoff")
# 4. move_to_coordinate(robot_to, handoff_position)
# 5. control_gripper(robot_to, open=False, object_id=object)
# 6. control_gripper(robot_from, open=True)
# 7. signal completion and move away


def create_detect_other_robot_operation() -> BasicOperation:
    return BasicOperation(
        operation_id="coordination_detect_robot_001",
        name="detect_other_robot",
        category=OperationCategory.PERCEPTION,
        complexity=OperationComplexity.INTERMEDIATE,
        description="Detect and locate another robot in shared workspace",
        long_description="""
            This operation uses vision and WorldState to detect another robot
            in the workspace, providing spatial awareness for coordination tasks.

            Returns robot position and distance for coordination planning.
        """,
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
        long_description="""
            Activates Unity's mirroring coroutine: one robot copies/reflects the
            movements of another in real-time for the specified duration.

            Unity executes the tracking loop internally. Use duration_ms to control
            how long the mirroring runs (default 10s). Useful for synchronized tasks,
            demonstration, or coordinated bimanual manipulation.
        """,
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


DETECT_OTHER_ROBOT_OPERATION = create_detect_other_robot_operation()
MIRROR_MOVEMENT_OPERATION = create_mirror_movement_operation()
