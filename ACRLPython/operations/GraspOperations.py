"""
Grasp Operations for Advanced Grasp Planning
=============================================

This module implements MoveIt2-inspired grasp planning operations that use
the full grasp planning pipeline with candidate generation, IK validation,
collision checking, and scoring.
"""

import logging
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
    request_id: int = 0
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

        # Build command payload with parameters nested (to match Unity's RobotCommand structure)
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
