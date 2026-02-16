"""
AutoRT Data Models

Pydantic models for scene descriptions, task proposals, and safety verdicts.
"""

from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import List, Tuple, Optional, Dict, Any


class GroundedObject(BaseModel):
    """Object detected and grounded to 3D space via existing detect_object_stereo"""
    model_config = ConfigDict(extra="forbid")

    object_id: str
    color: str                                 # Detection color: "red", "blue", etc.
    position: Tuple[float, float, float]       # 3D position in world frame (from stereo)
    confidence: float = Field(ge=0.0, le=1.0)
    graspable: bool = True


class SceneDescription(BaseModel):
    """Scene state assembled from existing operations + WorldState"""
    timestamp: float
    objects: List[GroundedObject]
    scene_summary: str = ""                    # Optional VLM reasoning
    robot_states: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class Operation(BaseModel):
    """Single operation in a task sequence.

    Uses plain string type validated against OperationRegistry
    to stay in sync with all 30+ registered operations.
    """
    type: str                                  # e.g. "move_to_coordinate", "control_gripper"
    robot_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)  # Optional, defaults to empty dict


class ProposedTask(BaseModel):
    """Task proposal from LLM with strict validation"""
    model_config = ConfigDict(extra="forbid")

    task_id: str
    description: str
    operations: List[Operation] = Field(min_length=1)
    required_robots: List[str] = Field(min_length=1)
    estimated_complexity: int = Field(ge=1, le=10)
    reasoning: str = ""  # Optional - defaults to empty string if not provided

    @model_validator(mode='after')
    def validate_robot_ids_consistent(self):
        """Ensure all operation robot_ids appear in required_robots"""
        required = set(self.required_robots)
        for op in self.operations:
            if op.robot_id not in required:
                raise ValueError(
                    f"Operation uses robot '{op.robot_id}' "
                    f"not in required_robots {required}"
                )
        return self


class TaskVerdict(BaseModel):
    """Constitution evaluation result"""
    approved: bool
    violations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    modified_task: Optional[ProposedTask] = None
    rejection_reason: Optional[str] = None
