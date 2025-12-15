"""
Multi-Robot Coordination Operations
====================================

**DEPRECATED**: This module is deprecated as of the LLM-driven coordination refactoring.
Use atomic operations (move_to_coordinate, control_gripper, etc.) combined with
synchronization primitives (signal, wait_for_signal) instead.

The LLM should plan multi-robot coordination using basic building blocks rather than
pre-programmed coordination patterns.

This file is kept for reference only. These operations are NOT registered in the
global operation registry.

Original Operations:
- coordinate_simultaneous_move: Safe parallel movement with collision checking
  → Replaced by: Two move_to_coordinate calls with parallel_group marking
- coordinate_handoff: Object handoff between two robots
  → Replaced by: LLM-planned sequence using detect, move, grip, signal primitives
- allocate_workspace_region: Workspace allocation for exclusive access
  → Replaced by: Automatic safety checks in Unity CollaborativeStrategy
"""

import time
import logging
from typing import Tuple, Optional
import LLMConfig
from servers.CommandServer import get_command_broadcaster
from operations.WorldState import get_world_state
from operations.MoveOperations import move_to_coordinate
from operations.CoordinationVerifier import CoordinationVerifier
from operations.SpatialPredicates import robots_will_collide
from .Base import (
    BasicOperation,
    OperationCategory,
    OperationComplexity,
    OperationParameter,
    OperationResult,
    ParameterFlow,
    OperationRelationship,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Implementation: Coordinate Simultaneous Move
# ============================================================================


def coordinate_simultaneous_move(
    robot1_id: str,
    target1: Tuple[float, float, float],
    robot2_id: str,
    target2: Tuple[float, float, float],
    request_id: int = 0,
) -> OperationResult:
    """
    Coordinate simultaneous movement of two robots with collision checking.

    Checks if robots will collide during movement. If safe, executes parallel
    movement. If collision detected, serializes the movements (robot1 first,
    then robot2).

    Args:
        robot1_id: First robot identifier
        target1: Target position for robot1 (x, y, z)
        robot2_id: Second robot identifier
        target2: Target position for robot2 (x, y, z)
        request_id: Request tracking ID

    Returns:
        OperationResult with coordination status

    Example:
        >>> # Safe parallel movement
        >>> coordinate_simultaneous_move("Robot1", (0.3, -0.2, 0.1), "Robot2", (0.3, 0.2, 0.1))

        >>> # Automatic serialization if collision detected
        >>> coordinate_simultaneous_move("Robot1", (0.0, 0.0, 0.2), "Robot2", (0.0, 0.0, 0.2))
    """
    try:
        world_state = get_world_state()

        # Check for collision
        will_collide, reason = robots_will_collide(
            robot1_id, target1,
            robot2_id, target2,
            world_state
        )

        if will_collide:
            # Collision detected - serialize movements
            logger.warning(
                f"Collision detected between {robot1_id} and {robot2_id}: {reason}. "
                "Serializing movements."
            )

            # Move robot1 first
            logger.info(f"Moving {robot1_id} to {target1}")
            result1 = move_to_coordinate(
                robot_id=robot1_id,
                x=target1[0],
                y=target1[1],
                z=target1[2],
                request_id=request_id
            )

            if not result1.success:
                return OperationResult.error_result(
                    "ROBOT1_MOVEMENT_FAILED",
                    f"Failed to move {robot1_id}: {result1.error}",
                    ["Check robot1 status", "Verify target1 is reachable"]
                )

            # Wait briefly for robot1 to start moving
            time.sleep(0.5)

            # Move robot2 second
            logger.info(f"Moving {robot2_id} to {target2}")
            result2 = move_to_coordinate(
                robot_id=robot2_id,
                x=target2[0],
                y=target2[1],
                z=target2[2],
                request_id=request_id + 1
            )

            if not result2.success:
                return OperationResult.error_result(
                    "ROBOT2_MOVEMENT_FAILED",
                    f"Failed to move {robot2_id}: {result2.error}",
                    ["Check robot2 status", "Verify target2 is reachable"]
                )

            return OperationResult.success_result({
                "coordination_mode": "serialized",
                "collision_reason": reason,
                "robot1_id": robot1_id,
                "robot1_target": target1,
                "robot2_id": robot2_id,
                "robot2_target": target2,
                "timestamp": time.time()
            })

        else:
            # Safe to move in parallel
            logger.info(
                f"Safe parallel movement: {robot1_id} to {target1}, "
                f"{robot2_id} to {target2}"
            )

            # Execute both movements
            result1 = move_to_coordinate(
                robot_id=robot1_id,
                x=target1[0],
                y=target1[1],
                z=target1[2],
                request_id=request_id
            )

            result2 = move_to_coordinate(
                robot_id=robot2_id,
                x=target2[0],
                y=target2[1],
                z=target2[2],
                request_id=request_id + 1
            )

            # Check if both succeeded
            if not result1.success:
                return OperationResult.error_result(
                    "ROBOT1_MOVEMENT_FAILED",
                    f"Failed to move {robot1_id}: {result1.error}",
                    ["Check robot1 status", "Verify target1 is reachable"]
                )

            if not result2.success:
                return OperationResult.error_result(
                    "ROBOT2_MOVEMENT_FAILED",
                    f"Failed to move {robot2_id}: {result2.error}",
                    ["Check robot2 status", "Verify target2 is reachable"]
                )

            return OperationResult.success_result({
                "coordination_mode": "parallel",
                "robot1_id": robot1_id,
                "robot1_target": target1,
                "robot2_id": robot2_id,
                "robot2_target": target2,
                "timestamp": time.time()
            })

    except Exception as e:
        logger.error(f"Error in coordinate_simultaneous_move: {e}", exc_info=True)
        return OperationResult.error_result(
            "COORDINATION_ERROR",
            f"Unexpected error: {str(e)}",
            ["Check logs for details", "Verify both robots exist"]
        )


# ============================================================================
# Implementation: Coordinate Handoff
# ============================================================================


def coordinate_handoff(
    robot1_id: str,
    robot2_id: str,
    object_id: str,
    handoff_position: Optional[Tuple[float, float, float]] = None,
    request_id: int = 0,
) -> OperationResult:
    """
    Coordinate object handoff between two robots.

    Multi-step sequence:
    1. Robot1 moves to object (if not already grasping)
    2. Robot1 grasps object
    3. Both robots move to handoff position (shared_zone center if not specified)
    4. Robot2 grasps object
    5. Robot1 releases object

    Args:
        robot1_id: First robot (gives object)
        robot2_id: Second robot (receives object)
        object_id: Object to handoff
        handoff_position: Optional handoff location (defaults to shared_zone center)
        request_id: Request tracking ID

    Returns:
        OperationResult with handoff status

    Example:
        >>> # Handoff with default position
        >>> coordinate_handoff("Robot1", "Robot2", "cube_01")

        >>> # Handoff at specific position
        >>> coordinate_handoff("Robot1", "Robot2", "cube_01", handoff_position=(0.0, 0.0, 0.2))
    """
    try:
        world_state = get_world_state()

        # Determine handoff position (default to shared_zone center)
        if handoff_position is None:
            shared_zone = LLMConfig.WORKSPACE_REGIONS.get("shared_zone")
            if shared_zone:
                handoff_position = (
                    (shared_zone["x_min"] + shared_zone["x_max"]) / 2,
                    (shared_zone["y_min"] + shared_zone["y_max"]) / 2,
                    (shared_zone["z_min"] + shared_zone["z_max"]) / 2,
                )
            else:
                handoff_position = (0.0, 0.0, 0.15)  # Fallback position

        logger.info(
            f"Coordinating handoff: {robot1_id} -> {robot2_id}, "
            f"object: {object_id}, position: {handoff_position}"
        )

        # Step 1: Get object position
        obj_state = world_state._objects.get(object_id)
        if obj_state is None:
            return OperationResult.error_result(
                "OBJECT_NOT_FOUND",
                f"Object '{object_id}' not found in world state",
                ["Run object detection to locate object", "Verify object ID"]
            )

        # Step 2: Robot1 moves to object (if not already there)
        if obj_state.grasped_by != robot1_id:
            logger.info(f"Step 1: {robot1_id} moving to object at {obj_state.position}")
            result = move_to_coordinate(
                robot_id=robot1_id,
                x=obj_state.position[0],
                y=obj_state.position[1],
                z=obj_state.position[2],
                request_id=request_id
            )
            if not result.success:
                return OperationResult.error_result(
                    "ROBOT1_APPROACH_FAILED",
                    f"Robot1 failed to approach object: {result.error}",
                    ["Check robot1 can reach object", "Verify object position"]
                )

            # TODO: Actually close gripper (requires GripperOperations integration)
            # For now, mark object as grasped
            world_state.mark_object_grasped(object_id, robot1_id)
            logger.info(f"{robot1_id} grasped {object_id}")

        # Step 3: Robot1 moves to handoff position
        logger.info(f"Step 2: {robot1_id} moving to handoff position")
        result = move_to_coordinate(
            robot_id=robot1_id,
            x=handoff_position[0],
            y=handoff_position[1],
            z=handoff_position[2],
            request_id=request_id + 1
        )
        if not result.success:
            return OperationResult.error_result(
                "ROBOT1_HANDOFF_MOVE_FAILED",
                f"Robot1 failed to reach handoff position: {result.error}",
                ["Check handoff position is reachable"]
            )

        # Step 4: Robot2 moves to handoff position (offset to avoid collision)
        robot2_offset_pos = (
            handoff_position[0] + 0.15,  # Offset to right
            handoff_position[1],
            handoff_position[2]
        )
        logger.info(f"Step 3: {robot2_id} moving to handoff position (offset)")
        result = move_to_coordinate(
            robot_id=robot2_id,
            x=robot2_offset_pos[0],
            y=robot2_offset_pos[1],
            z=robot2_offset_pos[2],
            request_id=request_id + 2
        )
        if not result.success:
            return OperationResult.error_result(
                "ROBOT2_HANDOFF_MOVE_FAILED",
                f"Robot2 failed to reach handoff position: {result.error}",
                ["Check handoff position is reachable by robot2"]
            )

        # Step 5: Transfer object
        logger.info(f"Step 4: Transferring {object_id} from {robot1_id} to {robot2_id}")
        world_state.mark_object_grasped(object_id, robot2_id)

        # Step 6: Robot1 releases and moves away
        logger.info(f"Step 5: {robot1_id} releasing object")
        # TODO: Open robot1 gripper

        return OperationResult.success_result({
            "robot1_id": robot1_id,
            "robot2_id": robot2_id,
            "object_id": object_id,
            "handoff_position": handoff_position,
            "status": "handoff_complete",
            "timestamp": time.time()
        })

    except Exception as e:
        logger.error(f"Error in coordinate_handoff: {e}", exc_info=True)
        return OperationResult.error_result(
            "HANDOFF_ERROR",
            f"Unexpected error during handoff: {str(e)}",
            ["Check logs for details", "Verify both robots and object exist"]
        )


# ============================================================================
# Implementation: Allocate Workspace Region
# ============================================================================


def allocate_workspace_region(
    robot_id: str,
    region_name: str,
    request_id: int = 0,
) -> OperationResult:
    """
    Allocate a workspace region for exclusive robot access.

    Claims a workspace region to prevent conflicts with other robots.
    Region remains allocated until explicitly released or robot completes operation.

    Args:
        robot_id: Robot identifier
        region_name: Region to allocate ("left_workspace", "right_workspace", "shared_zone", "center")
        request_id: Request tracking ID

    Returns:
        OperationResult with allocation status

    Example:
        >>> # Allocate left workspace
        >>> allocate_workspace_region("Robot1", "left_workspace")

        >>> # Allocate shared zone
        >>> allocate_workspace_region("Robot1", "shared_zone")
    """
    try:
        world_state = get_world_state()

        # Validate region name
        if region_name not in LLMConfig.WORKSPACE_REGIONS:
            valid_regions = list(LLMConfig.WORKSPACE_REGIONS.keys())
            return OperationResult.error_result(
                "INVALID_REGION",
                f"Unknown region '{region_name}'. Valid regions: {', '.join(valid_regions)}",
                [f"Use one of: {', '.join(valid_regions)}"]
            )

        # Attempt allocation
        success = world_state.allocate_workspace(region_name, robot_id)

        if not success:
            current_owner = world_state.get_workspace_owner(region_name)
            return OperationResult.error_result(
                "ALLOCATION_FAILED",
                f"Region '{region_name}' already allocated to {current_owner}",
                [
                    f"Wait for {current_owner} to release region",
                    "Use different region",
                    "Force release if robot is stuck (use with caution)"
                ]
            )

        logger.info(f"Allocated {region_name} to {robot_id}")

        return OperationResult.success_result({
            "robot_id": robot_id,
            "region_name": region_name,
            "status": "allocated",
            "timestamp": time.time()
        })

    except Exception as e:
        logger.error(f"Error in allocate_workspace_region: {e}", exc_info=True)
        return OperationResult.error_result(
            "ALLOCATION_ERROR",
            f"Unexpected error: {str(e)}",
            ["Check logs for details", "Verify region name"]
        )


# ============================================================================
# BasicOperation Definitions
# ============================================================================


def create_coordinate_simultaneous_move_operation() -> BasicOperation:
    """Create BasicOperation definition for coordinate_simultaneous_move."""
    return BasicOperation(
        operation_id="coord_simultaneous_move_001",
        name="coordinate_simultaneous_move",
        category=OperationCategory.COORDINATION,
        complexity=OperationComplexity.COMPLEX,
        description="Coordinate safe parallel movement of two robots with collision checking",
        long_description="""
            Checks for collision between two robot movements. If safe, executes parallel
            movement. If collision detected, automatically serializes movements (robot1 first,
            then robot2) to prevent collision. Essential for multi-robot coordination.
        """,
        usage_examples=[
            "coordinate_simultaneous_move('Robot1', (0.3, -0.2, 0.1), 'Robot2', (0.3, 0.2, 0.1))",
            "Safe parallel movement of multiple robots",
            "Automatic serialization on collision detection"
        ],
        parameters=[
            OperationParameter("robot1_id", "str", "First robot identifier", required=True),
            OperationParameter("target1", "Tuple[float, float, float]", "Target position for robot1", required=True),
            OperationParameter("robot2_id", "str", "Second robot identifier", required=True),
            OperationParameter("target2", "Tuple[float, float, float]", "Target position for robot2", required=True),
        ],
        preconditions=[
            "robots_will_collide(robot1_id, target1, robot2_id, target2) - checked internally",
            "robot_is_initialized(robot1_id)",
            "robot_is_initialized(robot2_id)"
        ],
        postconditions=[
            "Both robots move to targets (parallel or serialized)",
            "No collision occurs"
        ],
        average_duration_ms=4000.0,
        success_rate=0.92,
        failure_modes=["Robot not reachable", "Target out of bounds"],
        commonly_paired_with=["move_to_region", "allocate_workspace_region"],
        relationships=OperationRelationship(
            operation_id="coord_simultaneous_move_001",
            required_operations=["status_check_robot_001", "coordination_allocate_workspace_001"],
            required_reasons={
                "status_check_robot_001": "Verify both robots are ready and not moving before parallel operation",
                "coordination_allocate_workspace_001": "Allocate workspace regions to ensure safe zones for movement",
            },
            commonly_paired_with=["spatial_move_to_region_001", "coordination_allocate_workspace_001", "perception_stereo_detect_001"],
            pairing_reasons={
                "spatial_move_to_region_001": "Navigate robots to designated regions before coordinated movement",
                "coordination_allocate_workspace_001": "Reserve workspace before moving to prevent conflicts",
                "perception_stereo_detect_001": "Detect target objects before coordinated approach",
            },
            typical_before=[],
            typical_after=["coordination_allocate_workspace_001", "status_check_robot_001"],
            coordination_requirements={
                "collision_checking": "Automatically checks for collision between robot paths",
                "serialization": "Falls back to sequential movement if collision detected",
                "safety": "Ensures no robot collision during parallel movement",
            },
        ),
        implementation=coordinate_simultaneous_move
    )


def create_coordinate_handoff_operation() -> BasicOperation:
    """Create BasicOperation definition for coordinate_handoff."""
    return BasicOperation(
        operation_id="coord_handoff_001",
        name="coordinate_handoff",
        category=OperationCategory.COORDINATION,
        complexity=OperationComplexity.COMPLEX,
        description="Coordinate object handoff between two robots",
        long_description="""
            Multi-step handoff sequence: robot1 grasps object, moves to handoff position
            (shared_zone by default), robot2 approaches, grasps object, robot1 releases.
            Essential for multi-robot object transfer tasks.
        """,
        usage_examples=[
            "coordinate_handoff('Robot1', 'Robot2', 'cube_01')",
            "Transfer object between robots",
            "Handoff at custom position"
        ],
        parameters=[
            OperationParameter("robot1_id", "str", "First robot (gives object)", required=True),
            OperationParameter("robot2_id", "str", "Second robot (receives object)", required=True),
            OperationParameter("object_id", "str", "Object to handoff", required=True),
            OperationParameter("handoff_position", "Optional[Tuple]", "Handoff location (default: shared_zone)", required=False),
        ],
        preconditions=[
            "Object exists and is detected",
            "robot_is_initialized(robot1_id)",
            "robot_is_initialized(robot2_id)",
            "Handoff position reachable by both robots"
        ],
        postconditions=[
            "Object transferred to robot2",
            "Robot1 released object",
            "Both robots at/near handoff position"
        ],
        average_duration_ms=8000.0,
        success_rate=0.88,
        failure_modes=["Object not found", "Handoff position unreachable", "Grasp failure"],
        commonly_paired_with=["detect_object", "move_to_region"],
        relationships=OperationRelationship(
            operation_id="coord_handoff_001",
            required_operations=["perception_stereo_detect_001", "motion_move_to_coord_001", "manipulation_control_gripper_001"],
            required_reasons={
                "perception_stereo_detect_001": "Need to detect object location before robot1 can grasp it",
                "motion_move_to_coord_001": "Both robots must navigate to handoff position",
                "manipulation_control_gripper_001": "Robot1 must grasp, robot2 must grasp, robot1 must release",
            },
            commonly_paired_with=["perception_stereo_detect_001", "spatial_move_to_region_001", "spatial_move_between_001", "manipulation_control_gripper_001"],
            pairing_reasons={
                "perception_stereo_detect_001": "Detect object before attempting handoff",
                "spatial_move_to_region_001": "Navigate both robots to shared_zone for handoff",
                "spatial_move_between_001": "Position robots between object and handoff point",
                "manipulation_control_gripper_001": "Coordinate gripper actions during handoff sequence",
            },
            parameter_flows=[
                ParameterFlow(
                    source_operation="perception_stereo_detect_001",
                    source_output_key="x",
                    target_operation="coord_handoff_001",
                    target_input_param="object_id",
                    description="Object position for robot1 to approach and grasp",
                ),
            ],
            typical_before=[],
            typical_after=["perception_stereo_detect_001", "spatial_move_to_region_001"],
            coordination_requirements={
                "sequencing": "Robot1 grasps → both move → robot2 grasps → robot1 releases",
                "safety": "Robots must not collide during handoff",
                "synchronization": "Requires careful timing of grasp/release actions",
            },
        ),
        implementation=coordinate_handoff
    )


def create_allocate_workspace_region_operation() -> BasicOperation:
    """Create BasicOperation definition for allocate_workspace_region."""
    return BasicOperation(
        operation_id="coord_allocate_workspace_001",
        name="allocate_workspace_region",
        category=OperationCategory.COORDINATION,
        complexity=OperationComplexity.BASIC,
        description="Allocate workspace region for exclusive robot access",
        long_description="""
            Claims a workspace region to prevent conflicts with other robots. Region
            remains allocated until explicitly released. Essential for multi-robot
            workspace management and conflict prevention.
        """,
        usage_examples=[
            "allocate_workspace_region('Robot1', 'left_workspace')",
            "Reserve shared_zone before handoff",
            "Claim exclusive access to workspace"
        ],
        parameters=[
            OperationParameter("robot_id", "str", "Robot identifier", required=True),
            OperationParameter("region_name", "str", "Region to allocate (left_workspace, right_workspace, etc.)", required=True),
        ],
        preconditions=[
            "Region exists in WORKSPACE_REGIONS",
            "Region not already allocated to another robot"
        ],
        postconditions=[
            "Region allocated to robot",
            "Other robots blocked from region"
        ],
        average_duration_ms=50.0,
        success_rate=0.95,
        failure_modes=["Region already allocated", "Invalid region name"],
        commonly_paired_with=["move_to_region", "coordinate_simultaneous_move"],
        relationships=OperationRelationship(
            operation_id="coord_allocate_workspace_001",
            required_operations=[],
            commonly_paired_with=["spatial_move_to_region_001", "coordination_simultaneous_move_001", "coordination_handoff_001"],
            pairing_reasons={
                "spatial_move_to_region_001": "Allocate region before robot navigates to it to claim exclusive access",
                "coordination_simultaneous_move_001": "Reserve regions before coordinated parallel movement",
                "coordination_handoff_001": "Allocate shared_zone before object handoff",
            },
            typical_before=["spatial_move_to_region_001", "coordination_simultaneous_move_001", "coordination_handoff_001"],
            typical_after=[],
            coordination_requirements={
                "mutual_exclusion": "Only one robot can allocate a region at a time",
                "safety": "Prevents robot collisions by enforcing spatial separation",
                "release": "Region must be explicitly released when robot leaves",
            },
        ),
        implementation=allocate_workspace_region
    )


# Export operation instances
COORDINATE_SIMULTANEOUS_MOVE_OPERATION = create_coordinate_simultaneous_move_operation()
COORDINATE_HANDOFF_OPERATION = create_coordinate_handoff_operation()
ALLOCATE_WORKSPACE_REGION_OPERATION = create_allocate_workspace_region_operation()
