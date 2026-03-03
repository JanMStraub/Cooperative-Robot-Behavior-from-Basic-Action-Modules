"""
Spatial Reasoning Operations
=============================

This module implements spatial reasoning operations for robot navigation
based on relationships to objects and workspace regions.

Operations:
- move_relative_to_object: Move to a position relative to an object (left_of, right_of, above, etc.)
- move_between_objects: Move to a position between two objects
- move_to_region: Move to a specific position within a workspace region
"""

import time
from typing import Tuple, Union, Optional

# Import from centralized lazy import system (prevents circular dependencies)
try:
    from ..core.Imports import get_world_state, get_robot_config
except ImportError:
    from core.Imports import get_world_state, get_robot_config

from .MoveOperations import move_to_coordinate
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
from core.LoggingSetup import get_logger
logger = get_logger(__name__)


# ============================================================================
# Implementation: Move Relative to Object
# ============================================================================


def move_relative_to_object(
    robot_id: str,
    object_ref: Union[str, Tuple[float, float, float]],
    relation: str,
    offset: float = 0.1,
    z_override: Optional[float] = None,
    request_id: int = 0,
    use_ros: Optional[bool] = None,
) -> OperationResult:
    """
    Move robot to a position relative to an object or position.

    Calculates target position based on spatial relation (left_of, right_of, above, etc.)
    and applies offset distance. Useful for approach movements or positioning relative
    to detected objects.

    Args:
        robot_id: Robot identifier (e.g., "Robot1")
        object_ref: Either object_id (str) or position tuple (x, y, z)
        relation: Spatial relation - one of: "left_of", "right_of", "above", "below",
                 "in_front_of", "behind"
        offset: Distance from object in meters (default: 0.1m)
        z_override: Optional Z coordinate override (useful for maintaining height)
        request_id: Request tracking ID
        use_ros: Whether to use ROS for motion planning (None = auto-detect from config)

    Returns:
        OperationResult with success status and target position

    Example:
        >>> # Move to the left of detected object
        >>> move_relative_to_object("Robot1", "cube_01", "left_of", offset=0.15)

        >>> # Move above a specific position
        >>> move_relative_to_object("Robot1", (0.3, 0.2, 0.1), "above", offset=0.1)
    """
    try:
        # Get object position
        if isinstance(object_ref, str):
            # Object ID - query from world state
            world_state = get_world_state()
            position = world_state.get_object_position(object_ref)
            if position is None:
                return OperationResult.error_result(
                    "OBJECT_NOT_FOUND",
                    f"Object '{object_ref}' not found in world state",
                    [
                        "Run object detection first to locate objects",
                        "Verify object ID is correct",
                        "Check that object is in camera view",
                    ],
                )
        else:
            # Direct position
            position = object_ref

        # Validate relation
        valid_relations = [
            "left_of",
            "right_of",
            "above",
            "below",
            "in_front_of",
            "behind",
        ]
        if relation not in valid_relations:
            return OperationResult.error_result(
                "INVALID_RELATION",
                f"Invalid relation '{relation}'. Must be one of: {', '.join(valid_relations)}",
                [f"Use one of the valid relations: {', '.join(valid_relations)}"],
            )

        # Validate offset
        if not (0.0 <= offset <= 0.5):
            return OperationResult.error_result(
                "INVALID_OFFSET",
                f"Offset {offset} out of range [0.0, 0.5]",
                ["Use offset between 0.0 and 0.5 meters"],
            )

        # Calculate target position based on relation
        x, y, z = position
        target_x, target_y, target_z = x, y, z

        if relation == "left_of":
            target_x = x - offset
        elif relation == "right_of":
            target_x = x + offset
        elif relation == "above":
            target_z = z + offset
        elif relation == "below":
            target_z = z - offset
        elif relation == "in_front_of":
            target_y = y + offset
        elif relation == "behind":
            target_y = y - offset

        # Apply z_override if provided
        if z_override is not None:
            target_z = z_override

        # Execute movement using move_to_coordinate
        logger.info(
            f"Moving {robot_id} {relation} object at ({x:.3f}, {y:.3f}, {z:.3f}) "
            f"-> target: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})"
        )

        move_result = move_to_coordinate(
            robot_id=robot_id,
            x=target_x,
            y=target_y,
            z=target_z,
            request_id=request_id,
            use_ros=use_ros,
        )

        if not move_result.success:
            return move_result

        # Return success with target info
        return OperationResult.success_result(
            {
                "robot_id": robot_id,
                "relation": relation,
                "object_position": position,
                "target_position": (target_x, target_y, target_z),
                "offset": offset,
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.error(f"Error in move_relative_to_object: {e}", exc_info=True)
        return OperationResult.error_result(
            "EXECUTION_ERROR",
            f"Unexpected error: {str(e)}",
            ["Check logs for details", "Verify parameters are correct"],
        )


# ============================================================================
# Implementation: Move Between Objects
# ============================================================================


def move_between_objects(
    robot_id: str,
    object1: Union[str, Tuple[float, float, float]],
    object2: Union[str, Tuple[float, float, float]],
    bias: float = 0.5,
    z_offset: float = 0.0,
    request_id: int = 0,
    use_ros: Optional[bool] = None,
) -> OperationResult:
    """
    Move robot to a position between two objects.

    Calculates interpolated position between two objects with optional bias
    towards one of them. Useful for handoff operations or positioning between
    multiple objects.

    Args:
        robot_id: Robot identifier
        object1: First object ID or position (x, y, z)
        object2: Second object ID or position (x, y, z)
        bias: Interpolation bias (0.0 = object1, 0.5 = midpoint, 1.0 = object2)
        z_offset: Additional Z offset to apply (positive = higher)
        request_id: Request tracking ID

    Returns:
        OperationResult with success status and target position

    Example:
        >>> # Move to midpoint between two objects
        >>> move_between_objects("Robot1", "cube_01", "cube_02", bias=0.5)

        >>> # Move closer to first object
        >>> move_between_objects("Robot1", "cube_01", "cube_02", bias=0.3)
    """
    try:
        world_state = get_world_state()

        # Get object1 position
        if isinstance(object1, str):
            pos1 = world_state.get_object_position(object1)
            if pos1 is None:
                return OperationResult.error_result(
                    "OBJECT1_NOT_FOUND",
                    f"Object '{object1}' not found in world state",
                    ["Run object detection to locate objects"],
                )
        else:
            pos1 = object1

        # Get object2 position
        if isinstance(object2, str):
            pos2 = world_state.get_object_position(object2)
            if pos2 is None:
                return OperationResult.error_result(
                    "OBJECT2_NOT_FOUND",
                    f"Object '{object2}' not found in world state",
                    ["Run object detection to locate objects"],
                )
        else:
            pos2 = object2

        # Validate bias
        if not (0.0 <= bias <= 1.0):
            return OperationResult.error_result(
                "INVALID_BIAS",
                f"Bias {bias} out of range [0.0, 1.0]",
                ["Use bias between 0.0 and 1.0"],
            )

        # Calculate interpolated position
        target_x = pos1[0] + bias * (pos2[0] - pos1[0])
        target_y = pos1[1] + bias * (pos2[1] - pos1[1])
        target_z = pos1[2] + bias * (pos2[2] - pos1[2]) + z_offset

        # Execute movement
        logger.info(
            f"Moving {robot_id} between objects: "
            f"({pos1[0]:.3f}, {pos1[1]:.3f}, {pos1[2]:.3f}) and "
            f"({pos2[0]:.3f}, {pos2[1]:.3f}, {pos2[2]:.3f}) "
            f"with bias {bias} -> target: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})"
        )

        move_result = move_to_coordinate(
            robot_id=robot_id,
            x=target_x,
            y=target_y,
            z=target_z,
            request_id=request_id,
            use_ros=use_ros,
        )

        if not move_result.success:
            return move_result

        # Return success
        return OperationResult.success_result(
            {
                "robot_id": robot_id,
                "object1_position": pos1,
                "object2_position": pos2,
                "target_position": (target_x, target_y, target_z),
                "bias": bias,
                "z_offset": z_offset,
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.error(f"Error in move_between_objects: {e}", exc_info=True)
        return OperationResult.error_result(
            "EXECUTION_ERROR",
            f"Unexpected error: {str(e)}",
            ["Check logs for details", "Verify parameters are correct"],
        )


# ============================================================================
# Implementation: Move to Region
# ============================================================================


def move_to_region(
    robot_id: str,
    region_name: str,
    position_in_region: str = "center",
    z_height: Optional[float] = None,
    request_id: int = 0,
    use_ros: Optional[bool] = None,
) -> OperationResult:
    """
    Move robot to a specific position within a workspace region.

    Navigates to predefined workspace regions with automatic position calculation.
    Useful for multi-robot coordination and workspace management.

    Args:
        robot_id: Robot identifier
        region_name: Region name - one of: "left_workspace", "right_workspace",
                    "shared_zone", "center"
        position_in_region: Position within region - "center", "near", or "far"
                           (near = closer to robot base, far = farther from base)
        z_height: Optional Z height override (if None, uses region default)
        request_id: Request tracking ID

    Returns:
        OperationResult with success status and target position

    Example:
        >>> # Move to center of shared zone
        >>> move_to_region("Robot1", "shared_zone", "center")

        >>> # Move to left workspace near position
        >>> move_to_region("Robot1", "left_workspace", "near", z_height=0.2)
    """
    try:
        # Validate region name
        robot_config = get_robot_config()
        if region_name not in robot_config.WORKSPACE_REGIONS:
            valid_regions = list(robot_config.WORKSPACE_REGIONS.keys())
            return OperationResult.error_result(
                "INVALID_REGION",
                f"Unknown region '{region_name}'. Valid regions: {', '.join(valid_regions)}",
                [f"Use one of: {', '.join(valid_regions)}"],
            )

        # Validate position_in_region
        valid_positions = ["center", "near", "far"]
        if position_in_region not in valid_positions:
            return OperationResult.error_result(
                "INVALID_POSITION",
                f"Invalid position '{position_in_region}'. Must be one of: {', '.join(valid_positions)}",
                [f"Use one of: {', '.join(valid_positions)}"],
            )

        # Get region bounds
        region = robot_config.WORKSPACE_REGIONS[region_name]

        # Calculate target position based on position_in_region
        x_min, x_max = region["x_min"], region["x_max"]
        y_min, y_max = region["y_min"], region["y_max"]
        z_min, z_max = region["z_min"], region["z_max"]

        # Get robot base position for near/far calculation
        robot_base = robot_config.ROBOT_BASE_POSITIONS.get(robot_id)
        if robot_base is None:
            return OperationResult.error_result(
                "UNKNOWN_ROBOT",
                f"Robot '{robot_id}' not in ROBOT_BASE_POSITIONS",
                ["Check robot ID is correct", "Add robot to LLMConfig"],
            )

        # Calculate X position
        if position_in_region == "center":
            target_x = (x_min + x_max) / 2.0
        elif position_in_region == "near":
            # Closer to robot base
            if robot_base[0] < 0:  # Robot on left
                target_x = x_max - 0.1  # Near the right edge of region
            else:  # Robot on right
                target_x = x_min + 0.1  # Near the left edge of region
        else:  # far
            # Farther from robot base
            if robot_base[0] < 0:  # Robot on left
                target_x = x_min + 0.1  # Far left of region
            else:  # Robot on right
                target_x = x_max - 0.1  # Far right of region

        # Y position (center of region)
        target_y = (y_min + y_max) / 2.0

        # Z position
        if z_height is not None:
            target_z = z_height
        else:
            target_z = (z_min + z_max) / 2.0

        # Execute movement
        logger.info(
            f"Moving {robot_id} to {region_name} ({position_in_region}) -> "
            f"target: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})"
        )

        move_result = move_to_coordinate(
            robot_id=robot_id,
            x=target_x,
            y=target_y,
            z=target_z,
            request_id=request_id,
            use_ros=use_ros,
        )

        if not move_result.success:
            return move_result

        # Return success
        return OperationResult.success_result(
            {
                "robot_id": robot_id,
                "region_name": region_name,
                "position_in_region": position_in_region,
                "target_position": (target_x, target_y, target_z),
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.error(f"Error in move_to_region: {e}", exc_info=True)
        return OperationResult.error_result(
            "EXECUTION_ERROR",
            f"Unexpected error: {str(e)}",
            ["Check logs for details", "Verify parameters are correct"],
        )


# ============================================================================
# BasicOperation Definitions
# ============================================================================


def create_move_relative_to_object_operation() -> BasicOperation:
    """Create BasicOperation definition for move_relative_to_object."""
    return BasicOperation(
        operation_id="spatial_move_relative_001",
        name="move_relative_to_object",
        category=OperationCategory.NAVIGATION,
        complexity=OperationComplexity.INTERMEDIATE,
        description="Move robot to a position relative to an object (left, right, above, etc.)",
        long_description="""
            Calculates and moves to a position relative to a detected object or specified location.
            Supports spatial relations: left_of, right_of, above, below, in_front_of, behind.
            Useful for approach movements, positioning near objects, or spatial task execution.
        """,
        usage_examples=[
            "move_relative_to_object('Robot1', 'cube_01', 'above', offset=0.1)",
            "Position robot to the left of detected object",
            "Approach object from above for grasping",
        ],
        parameters=[
            OperationParameter("robot_id", "str", "Robot identifier", required=True),
            OperationParameter(
                "object_ref",
                "Union[str, Tuple]",
                "Object ID or position (x,y,z)",
                required=True,
            ),
            OperationParameter(
                "relation",
                "str",
                "Spatial relation (left_of, right_of, above, etc.)",
                required=True,
            ),
            OperationParameter(
                "offset",
                "float",
                "Distance from object in meters",
                required=False,
                default=0.1,
                valid_range=(0.0, 0.5),
            ),
            OperationParameter(
                "z_override", "float", "Override Z coordinate", required=False
            ),
        ],
        preconditions=[
            "robot_is_initialized(robot_id)",
        ],
        postconditions=[],
        average_duration_ms=2000.0,
        success_rate=0.95,
        failure_modes=["Object not found", "Target out of reach", "Invalid relation"],
        relationships=OperationRelationship(
            operation_id="spatial_move_relative_001",
            required_operations=["perception_stereo_detect_001"],
            required_reasons={
                "perception_stereo_detect_001": "Need object position to calculate relative target coordinates",
            },
            commonly_paired_with=[
                "perception_stereo_detect_001",
                "manipulation_control_gripper_001",
                "motion_move_to_coord_001",
            ],
            pairing_reasons={
                "perception_stereo_detect_001": "Detect object to get reference position for spatial relation",
                "manipulation_control_gripper_001": "Position relative to object before grasping (e.g., above for pick)",
                "motion_move_to_coord_001": "Alternative direct movement, this adds spatial relation capability",
            },
            parameter_flows=[
                ParameterFlow(
                    source_operation="perception_stereo_detect_001",
                    source_output_key="x",
                    target_operation="spatial_move_relative_001",
                    target_input_param="object_ref",
                    description="Detected object position as reference for spatial relation",
                ),
            ],
            typical_before=["manipulation_control_gripper_001"],
            typical_after=["perception_stereo_detect_001"],
        ),
        implementation=move_relative_to_object,
    )


def create_move_between_objects_operation() -> BasicOperation:
    """Create BasicOperation definition for move_between_objects."""
    return BasicOperation(
        operation_id="spatial_move_between_001",
        name="move_between_objects",
        category=OperationCategory.NAVIGATION,
        complexity=OperationComplexity.INTERMEDIATE,
        description="Move robot to interpolated position between two objects",
        long_description="""
            Calculates and moves to a position between two objects with configurable bias.
            Bias parameter controls position: 0.0 = first object, 0.5 = midpoint, 1.0 = second object.
            Useful for handoff operations, positioning between objects, or multi-object tasks.
        """,
        usage_examples=[
            "move_between_objects('Robot1', 'cube_01', 'cube_02', bias=0.5)",
            "Position between two detected objects for handoff",
            "Move closer to first object with bias=0.3",
        ],
        parameters=[
            OperationParameter("robot_id", "str", "Robot identifier", required=True),
            OperationParameter(
                "object1",
                "Union[str, Tuple]",
                "First object ID or position",
                required=True,
            ),
            OperationParameter(
                "object2",
                "Union[str, Tuple]",
                "Second object ID or position",
                required=True,
            ),
            OperationParameter(
                "bias",
                "float",
                "Interpolation bias (0.0-1.0)",
                required=False,
                default=0.5,
                valid_range=(0.0, 1.0),
            ),
            OperationParameter(
                "z_offset", "float", "Additional Z offset", required=False, default=0.0
            ),
        ],
        preconditions=[
            "robot_is_initialized(robot_id)",
        ],
        postconditions=[],
        average_duration_ms=2000.0,
        success_rate=0.95,
        failure_modes=["Objects not found", "Target out of reach"],
        relationships=OperationRelationship(
            operation_id="spatial_move_between_001",
            required_operations=["perception_stereo_detect_001"],
            required_reasons={
                "perception_stereo_detect_001": "Need positions of both objects to calculate midpoint",
            },
            commonly_paired_with=[
                "perception_stereo_detect_001",
                "coordination_handoff_001",
                "manipulation_control_gripper_001",
            ],
            pairing_reasons={
                "perception_stereo_detect_001": "Detect both objects to get reference positions",
                "coordination_handoff_001": "Position between robots for object handoff",
                "manipulation_control_gripper_001": "Position before receiving/releasing object",
            },
            typical_before=[
                "manipulation_control_gripper_001",
                "coordination_handoff_001",
            ],
            typical_after=["perception_stereo_detect_001"],
        ),
        implementation=move_between_objects,
    )


def create_move_to_region_operation() -> BasicOperation:
    """Create BasicOperation definition for move_to_region."""
    return BasicOperation(
        operation_id="spatial_move_to_region_001",
        name="move_to_region",
        category=OperationCategory.NAVIGATION,
        complexity=OperationComplexity.BASIC,
        description="Move robot to a specific position within a workspace region",
        long_description="""
            Navigates to predefined workspace regions (left_workspace, right_workspace, shared_zone, center).
            Automatically calculates position based on region bounds and position preference (near, center, far).
            Essential for multi-robot coordination and workspace management.
        """,
        usage_examples=[
            "move_to_region('Robot1', 'shared_zone', 'center')",
            "Move to left workspace near position",
            "Navigate to shared zone for handoff",
        ],
        parameters=[
            OperationParameter("robot_id", "str", "Robot identifier", required=True),
            OperationParameter(
                "region_name",
                "str",
                "Region name (left_workspace, right_workspace, etc.)",
                required=True,
            ),
            OperationParameter(
                "position_in_region",
                "str",
                "Position (center, near, far)",
                required=False,
                default="center",
            ),
            OperationParameter(
                "z_height", "float", "Optional Z height override", required=False
            ),
        ],
        preconditions=[
            "robot_is_initialized(robot_id)",
        ],
        postconditions=[],
        average_duration_ms=2000.0,
        success_rate=0.98,
        failure_modes=["Invalid region name", "Target out of reach"],
        relationships=OperationRelationship(
            operation_id="spatial_move_to_region_001",
            required_operations=["coordination_allocate_workspace_001"],
            required_reasons={
                "coordination_allocate_workspace_001": "Must allocate workspace region before moving to it for multi-robot coordination",
            },
            commonly_paired_with=[
                "coordination_allocate_workspace_001",
                "coordination_simultaneous_move_001",
                "coordination_handoff_001",
            ],
            pairing_reasons={
                "coordination_allocate_workspace_001": "Allocate region before navigating to ensure exclusive access",
                "coordination_simultaneous_move_001": "Position robots in separate regions for coordinated parallel movement",
                "coordination_handoff_001": "Navigate to shared region for object handoff between robots",
            },
            typical_before=["coordination_handoff_001"],
            typical_after=["coordination_allocate_workspace_001"],
            coordination_requirements={
                "workspace_allocation": "Requires workspace region to be allocated before movement",
                "safety": "Ensures robots stay in designated regions to avoid collisions",
            },
        ),
        implementation=move_to_region,
    )


# Export operation instances
MOVE_RELATIVE_TO_OBJECT_OPERATION = create_move_relative_to_object_operation()
MOVE_BETWEEN_OBJECTS_OPERATION = create_move_between_objects_operation()
MOVE_TO_REGION_OPERATION = create_move_to_region_operation()
