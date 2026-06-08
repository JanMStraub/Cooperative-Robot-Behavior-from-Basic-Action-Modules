#!/usr/bin/env python3
"""Spatial reasoning operations: move relative to objects/regions (left_of, above, etc.)."""

import time
from typing import Tuple, Union, Optional

try:
    from ..core.Imports import get_world_state
except ImportError:
    from core.Imports import get_world_state

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

from core.LoggingSetup import get_logger

logger = get_logger(__name__)


def move_relative_to_object(
    robot_id: str,
    object_ref: Union[str, Tuple[float, float, float]],
    relation: str,
    offset: float = 0.1,
    z_override: Optional[float] = None,
    request_id: int = 0,
    use_ros: Optional[bool] = None,
) -> OperationResult:
    """Move robot to a position relative to an object (left_of, right_of, above, below, in_front_of, behind)."""
    try:
        if isinstance(object_ref, str):
            world_state = get_world_state()
            position = world_state.get_object_position(object_ref)
            if position is None:
                return OperationResult.error_result(
                    "OBJECT_NOT_FOUND",
                    f"Object '{object_ref}' not found in world state",
                )
        else:
            position = object_ref

        _aliases = {"right": "right_of", "left": "left_of", "front": "in_front_of"}
        relation = _aliases.get(relation, relation)

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
            )

        if not (0.0 <= offset <= 0.5):
            return OperationResult.error_result(
                "INVALID_OFFSET",
                f"Offset {offset} out of range [0.0, 0.5]",
            )

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

        if z_override is not None:
            target_z = z_override

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

        result_data = {
            "robot_id": robot_id,
            "relation": relation,
            "object_position": position,
            "target_position": (target_x, target_y, target_z),
            "offset": offset,
            "timestamp": time.time(),
        }
        # Propagate ROS/VGN execution status so SequenceExecutor skips Unity
        # completion wait (which would otherwise time out after 60 s).
        if move_result.result and "status" in move_result.result:
            result_data["status"] = move_result.result["status"]
        return OperationResult.success_result(result_data)

    except Exception as e:
        logger.error(f"Error in move_relative_to_object: {e}", exc_info=True)
        return OperationResult.error_result("EXECUTION_ERROR", str(e))


def create_move_relative_to_object_operation() -> BasicOperation:
    return BasicOperation(
        operation_id="spatial_move_relative_001",
        name="move_relative_to_object",
        category=OperationCategory.NAVIGATION,
        complexity=OperationComplexity.INTERMEDIATE,
        description="Move robot to a position relative to an object (left, right, above, etc.)",
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


MOVE_RELATIVE_TO_OBJECT_OPERATION = create_move_relative_to_object_operation()
