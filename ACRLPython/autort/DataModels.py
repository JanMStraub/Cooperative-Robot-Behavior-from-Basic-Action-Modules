#!/usr/bin/env python3
"""Pydantic models for scene descriptions, task proposals, and safety verdicts."""

from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import List, Tuple, Optional, Dict, Any


class GroundedObject(BaseModel):
    """3D-grounded detected object."""

    model_config = ConfigDict(extra="forbid")

    object_id: str
    color: str  # Detection color: "red", "blue", etc.
    position: Tuple[float, float, float]  # 3D position in world frame (from stereo)
    confidence: float = Field(ge=0.0, le=1.0)
    graspable: bool = True


class ExecutedTaskContext(BaseModel):
    """Record of the last completed task, used to prompt for continuity."""

    task_id: str
    description: str
    operation_types: List[str]
    success: bool
    result_summary: str = ""


class SceneDescription(BaseModel):
    """Scene state assembled from perception ops and WorldState."""

    timestamp: float
    objects: List[GroundedObject]
    scene_summary: str = ""  # Optional VLM reasoning
    robot_states: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    last_task_context: Optional["ExecutedTaskContext"] = None


class Operation(BaseModel):
    """Single operation in a task sequence."""

    type: str  # e.g. "move_to_coordinate", "control_gripper"
    robot_id: str
    parameters: Dict[str, Any] = Field(
        default_factory=dict
    )  # Optional, defaults to empty dict


class ProposedTask(BaseModel):
    """LLM-proposed task with Pydantic validation."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    description: str
    operations: List[Operation] = Field(min_length=1)
    required_robots: List[str] = Field(min_length=1)
    estimated_complexity: int = Field(ge=1, le=10)
    reasoning: Optional[str] = ""  # Optional - null from LLM is coerced to empty string

    @model_validator(mode="after")
    def validate_robot_ids_consistent(self):
        required = set(self.required_robots)
        for op in self.operations:
            if op.robot_id not in required:
                raise ValueError(
                    f"Operation uses robot '{op.robot_id}' "
                    f"not in required_robots {required}"
                )
        return self


class TaskVerdict(BaseModel):
    """Safety constitution evaluation result."""

    approved: bool
    violations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    modified_task: Optional[ProposedTask] = None
    rejection_reason: Optional[str] = None
