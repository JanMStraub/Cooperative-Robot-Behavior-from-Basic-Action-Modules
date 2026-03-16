#!/usr/bin/env python3
"""
Collaborative Manipulation Operations (Level 5)
================================================

This module implements ATOMIC Level 5 operations from the thesis exposé:
- stabilize_object: Hold object stable while partner manipulates (ATOMIC)

REMOVED (non-atomic):
- stabilize_and_manipulate_collaboratively: Use WorkflowPatterns.STABILIZE_MANIPULATE_PATTERN

These operations enable the most advanced multi-robot collaboration
requiring tight coordination and force control.
All operations are atomic - the LLM chains them to create complex workflows.
"""

import time
import logging
from typing import Optional

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
# Knowledge Graph helpers
# ============================================================================


def _kg_both_robots_can_reach(object_id: str, robot_id: str) -> Optional[str]:
    """
    Check whether the KG reports that at least two robots can reach the object.

    Returns a warning string if the reachable set is populated but contains
    fewer than two robots (suggesting the stabilizing robot's partner cannot
    reach). Returns None when the check passes or the KG is unavailable/empty
    (empty list = KG not yet populated = no false negatives at startup).

    Args:
        object_id: The object to check reachability for.
        robot_id: The robot requesting the check (for log context).

    Returns:
        Warning string if fewer than 2 robots can reach the object, else None.
    """
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


# ============================================================================
# Implementation: stabilize_object - Object stabilization
# ============================================================================


def stabilize_object(
    robot_id: str,
    object_id: str,
    duration_ms: int = 5000,
    force_limit: float = 10.0,
    request_id: int = 0,
    use_ros: Optional[bool] = None,
) -> OperationResult:
    """
    Stabilize (hold) an object while another robot manipulates it.

    This operation commands a robot to grasp and hold an object stable,
    providing support for another robot to perform manipulation tasks
    (e.g., insertion, assembly, precision placement).

    Args:
        robot_id: ID of the stabilizing robot
        object_id: ID of the object to stabilize
        duration_ms: Duration to hold stable (milliseconds)
        force_limit: Maximum grip force (Newtons)
        request_id: Optional request ID for tracking
        use_ros: Whether to use ROS for motion planning (None = auto-detect from config)

    Returns:
        OperationResult with stabilization activation confirmation

    Example:
        >>> # Robot1 holds object stable for 5 seconds
        >>> result = stabilize_object("Robot1", "LargeCube", duration_ms=5000)

        >>> # Robot1 stabilizes while Robot2 manipulates
        >>> stabilize_object("Robot1", "AssemblyPart", duration_ms=10000)
    """
    try:
        # Validate robot_id
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string",
                ["Provide a valid robot ID"],
            )

        # Validate object_id
        if not object_id or not isinstance(object_id, str):
            return OperationResult.error_result(
                "INVALID_OBJECT_ID",
                f"Object ID must be a non-empty string",
                ["Provide a valid object ID"],
            )

        # Validate duration_ms
        if not (100 <= duration_ms <= 30000):
            return OperationResult.error_result(
                "INVALID_DURATION",
                f"Duration must be in range [100, 30000]ms, got: {duration_ms}",
                ["Use duration between 100ms and 30000ms (30s)"],
            )

        # Validate force_limit
        if not (1.0 <= force_limit <= 50.0):
            return OperationResult.error_result(
                "INVALID_FORCE_LIMIT",
                f"Force limit must be in range [1.0, 50.0]N, got: {force_limit}",
                ["Use force limit between 1N and 50N"],
            )

        # KG reachability advisory (non-blocking — graph may be stale at startup)
        kg_warning = _kg_both_robots_can_reach(object_id, robot_id)
        if kg_warning:
            logger.warning(kg_warning)

        # Determine whether to use ROS or TCP path
        _use_ros = use_ros
        if _use_ros is None:
            try:
                from config.ROS import ROS_ENABLED, DEFAULT_CONTROL_MODE

                _use_ros = ROS_ENABLED and DEFAULT_CONTROL_MODE in ("ros", "hybrid")
            except ImportError:
                _use_ros = False

        # Note: Stabilization requires force control and continuous holding
        # This is best handled by Unity's force control system (TCP path)
        # ROS could handle initial positioning, but force feedback loops are Unity-side
        if _use_ros:
            logger.info(
                "Stabilization force control via ROS not yet implemented - using Unity direct control"
            )
            _use_ros = False

        # Construct command (TCP path)
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


# ============================================================================
# REMOVED: stabilize_and_manipulate_collaboratively
# ============================================================================
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
# ============================================================================


# ============================================================================
# BasicOperation Definitions
# ============================================================================


def create_stabilize_object_operation() -> BasicOperation:
    """Create the BasicOperation definition for stabilize_object."""
    return BasicOperation(
        operation_id="collaborative_stabilize_001",
        name="stabilize_object",
        category=OperationCategory.MANIPULATION,
        complexity=OperationComplexity.COMPLEX,
        description="Hold object stable while partner robot manipulates it",
        long_description="""
            This operation commands a robot to grasp and hold an object stable,
            providing support for another robot to perform manipulation tasks.

            Requires:
            - Force control to maintain stable grip without crushing object
            - Position stability to prevent movement during partner manipulation
            - Coordination with partner robot timing

            Critical for tasks requiring dual-arm support: assembly, insertion,
            precision placement.
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


# ============================================================================
# REMOVED: create_stabilize_and_manipulate_operation
# ============================================================================
# The stabilize_and_manipulate_collaboratively operation was removed because it
# is non-atomic. See WorkflowPatterns.py for STABILIZE_MANIPULATE_PATTERN.
# ============================================================================


# ============================================================================
# Create operation instances for export
# ============================================================================

STABILIZE_OBJECT_OPERATION = create_stabilize_object_operation()
# STABILIZE_AND_MANIPULATE_OPERATION removed - use WorkflowPatterns.STABILIZE_MANIPULATE_PATTERN instead
