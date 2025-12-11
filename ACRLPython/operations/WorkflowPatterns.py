"""
Workflow Pattern Library
========================

This module defines common operation sequences (workflows) that can be used
by the RAG system to help LLMs understand typical task patterns.

Workflows are reusable templates for common robot tasks like pick-and-place,
detection-and-approach, or multi-robot coordination scenarios.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class WorkflowCategory(Enum):
    """Categories for workflow patterns"""

    SINGLE_ROBOT = "single_robot"  # Single robot tasks
    MULTI_ROBOT = "multi_robot"  # Multi-robot coordination
    PERCEPTION = "perception"  # Vision-based workflows
    MANIPULATION = "manipulation"  # Grasping and object handling


@dataclass
class WorkflowStep:
    """
    A single step in a workflow pattern.

    Attributes:
        operation_id: The operation to execute
        parameter_bindings: Variable bindings from previous steps
        conditional: Optional condition for executing this step
        description: Human-readable description of the step
    """

    operation_id: str
    parameter_bindings: Dict[str, str] = field(default_factory=dict)
    conditional: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "operation_id": self.operation_id,
            "parameter_bindings": self.parameter_bindings,
            "conditional": self.conditional,
            "description": self.description,
        }


@dataclass
class WorkflowPattern:
    """
    A reusable workflow pattern (sequence of operations).

    Attributes:
        pattern_id: Unique identifier for the pattern
        name: Human-readable name
        category: Workflow category
        description: What this workflow accomplishes
        steps: Ordered sequence of workflow steps
        variable_bindings: Parameter flow between steps
        success_criteria: How to determine workflow success
        failure_recovery: What to do if a step fails
        usage_examples: Example invocations
    """

    pattern_id: str
    name: str
    category: WorkflowCategory
    description: str
    steps: List[WorkflowStep]
    variable_bindings: Dict[str, str] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    failure_recovery: Optional[str] = None
    usage_examples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
            "variable_bindings": self.variable_bindings,
            "success_criteria": self.success_criteria,
            "failure_recovery": self.failure_recovery,
            "usage_examples": self.usage_examples,
        }

    def to_rag_document(self) -> str:
        """
        Convert workflow pattern to RAG-optimized text document.

        Returns searchable natural language description for LLM retrieval.
        """
        doc = f"""
        WORKFLOW PATTERN: {self.name} (ID: {self.pattern_id})
        Category: {self.category.value}

        DESCRIPTION:
        {self.description}

        WORKFLOW STEPS:
        """

        for i, step in enumerate(self.steps, 1):
            doc += f"\n        {i}. {step.operation_id}"
            if step.description:
                doc += f": {step.description}"
            if step.parameter_bindings:
                bindings_str = ", ".join(f"{k}=${v}" for k, v in step.parameter_bindings.items())
                doc += f" [{bindings_str}]"
            if step.conditional:
                doc += f" (if {step.conditional})"

        if self.variable_bindings:
            doc += "\n\n        PARAMETER FLOWS:"
            for var, source in self.variable_bindings.items():
                doc += f"\n        - ${var} from {source}"

        if self.success_criteria:
            doc += "\n\n        SUCCESS CRITERIA:"
            for criterion in self.success_criteria:
                doc += f"\n        - {criterion}"

        if self.failure_recovery:
            doc += f"\n\n        FAILURE RECOVERY:\n        {self.failure_recovery}"

        if self.usage_examples:
            doc += "\n\n        USAGE EXAMPLES:"
            for example in self.usage_examples:
                doc += f"\n        - {example}"

        return doc


# ============================================================================
# Pattern Definitions: Single Robot Workflows
# ============================================================================


DETECT_AND_APPROACH_PATTERN = WorkflowPattern(
    pattern_id="workflow_detect_approach_001",
    name="detect_and_approach",
    category=WorkflowCategory.SINGLE_ROBOT,
    description="Detect an object using vision and move robot to its location",
    steps=[
        WorkflowStep(
            operation_id="perception_stereo_detect_001",
            parameter_bindings={},
            description="Detect object with stereo vision to get 3D coordinates"
        ),
        WorkflowStep(
            operation_id="motion_move_to_coord_001",
            parameter_bindings={
                "x": "detect_result.x",
                "y": "detect_result.y",
                "z": "detect_result.z",
            },
            description="Move robot to detected object position"
        ),
    ],
    variable_bindings={
        "detect_result": "perception_stereo_detect_001.result",
    },
    success_criteria=[
        "Object detected with confidence > 0.5",
        "Robot reached target within 2mm tolerance",
    ],
    failure_recovery="If detection fails, try analyze_scene to understand why",
    usage_examples=[
        "Find and approach blue cube: detect_and_approach(color='blue', robot_id='Robot1')",
        "Approach closest object: detect_and_approach(selection='closest', robot_id='Robot1')",
    ],
)


PICK_AND_PLACE_PATTERN = WorkflowPattern(
    pattern_id="workflow_pick_place_001",
    name="pick_and_place",
    category=WorkflowCategory.MANIPULATION,
    description="Detect object, move to it, grasp it, move to target, and release",
    steps=[
        WorkflowStep(
            operation_id="perception_stereo_detect_001",
            parameter_bindings={"color": "source_color"},
            description="Detect source object to pick up"
        ),
        WorkflowStep(
            operation_id="motion_move_to_coord_001",
            parameter_bindings={
                "x": "source.x",
                "y": "source.y",
                "z": "source.z",
                "approach_offset": "0.05",
            },
            description="Approach source object with offset"
        ),
        WorkflowStep(
            operation_id="manipulation_control_gripper_001",
            parameter_bindings={"open_gripper": "False"},
            description="Close gripper to grasp object"
        ),
        WorkflowStep(
            operation_id="motion_move_to_coord_001",
            parameter_bindings={
                "x": "target_x",
                "y": "target_y",
                "z": "target_z",
            },
            description="Move to target placement location"
        ),
        WorkflowStep(
            operation_id="manipulation_control_gripper_001",
            parameter_bindings={"open_gripper": "True"},
            description="Open gripper to release object"
        ),
    ],
    variable_bindings={
        "source": "perception_stereo_detect_001.result",
    },
    success_criteria=[
        "Source object detected",
        "Gripper successfully grasped object",
        "Object transported to target location",
        "Object released at target",
    ],
    failure_recovery="If grasp fails, retry with adjusted gripper position",
    usage_examples=[
        "Pick blue cube and place at (0.2, 0.0, 0.15)",
        "Rearrange objects by color",
    ],
)


VERIFY_AND_ACT_PATTERN = WorkflowPattern(
    pattern_id="workflow_verify_act_001",
    name="verify_and_act",
    category=WorkflowCategory.PERCEPTION,
    description="Analyze scene with LLM vision, then perform appropriate action based on understanding",
    steps=[
        WorkflowStep(
            operation_id="perception_analyze_scene_001",
            parameter_bindings={"prompt": "analysis_prompt"},
            description="Use LLM vision to understand current scene state"
        ),
        WorkflowStep(
            operation_id="perception_stereo_detect_001",
            conditional="scene_analysis indicates object is present",
            description="Detect object if LLM confirms it's visible"
        ),
        WorkflowStep(
            operation_id="motion_move_to_coord_001",
            parameter_bindings={
                "x": "detection.x",
                "y": "detection.y",
                "z": "detection.z",
            },
            conditional="object detected successfully",
            description="Move to detected object"
        ),
    ],
    variable_bindings={
        "scene_analysis": "perception_analyze_scene_001.result",
        "detection": "perception_stereo_detect_001.result",
    },
    success_criteria=[
        "Scene understood by LLM",
        "Appropriate action taken based on scene state",
    ],
    failure_recovery="If scene unclear, request human clarification",
    usage_examples=[
        "Verify workspace is clear before moving",
        "Check if object is graspable orientation",
    ],
)


# ============================================================================
# Pattern Definitions: Multi-Robot Coordination
# ============================================================================


SIMULTANEOUS_MOVE_PATTERN = WorkflowPattern(
    pattern_id="workflow_simultaneous_move_001",
    name="simultaneous_move",
    category=WorkflowCategory.MULTI_ROBOT,
    description="Move two robots simultaneously to different targets with collision checking",
    steps=[
        WorkflowStep(
            operation_id="status_check_robot_001",
            parameter_bindings={"robot_id": "robot1_id"},
            description="Check robot 1 status before moving"
        ),
        WorkflowStep(
            operation_id="status_check_robot_001",
            parameter_bindings={"robot_id": "robot2_id"},
            description="Check robot 2 status before moving"
        ),
        WorkflowStep(
            operation_id="coordination_simultaneous_move_001",
            parameter_bindings={
                "robot1_id": "robot1_id",
                "target1": "target1_coords",
                "robot2_id": "robot2_id",
                "target2": "target2_coords",
            },
            description="Execute collision-checked simultaneous movement"
        ),
    ],
    variable_bindings={
        "robot1_status": "status_check_robot_001[0].result",
        "robot2_status": "status_check_robot_001[1].result",
    },
    success_criteria=[
        "Both robots ready to move",
        "No collision predicted during movement",
        "Both robots reached targets successfully",
    ],
    failure_recovery="If collision predicted, execute sequential movements instead",
    usage_examples=[
        "Move Robot1 to (0.3, 0.1, 0.2) and Robot2 to (0.3, -0.1, 0.2) simultaneously",
        "Coordinate dual-arm assembly task",
    ],
)


HANDOFF_PATTERN = WorkflowPattern(
    pattern_id="workflow_handoff_001",
    name="handoff",
    category=WorkflowCategory.MULTI_ROBOT,
    description="Transfer object from one robot to another",
    steps=[
        WorkflowStep(
            operation_id="perception_stereo_detect_001",
            parameter_bindings={"color": "object_color"},
            description="Detect object to handoff"
        ),
        WorkflowStep(
            operation_id="motion_move_to_coord_001",
            parameter_bindings={
                "robot_id": "source_robot",
                "x": "object.x",
                "y": "object.y",
                "z": "object.z",
            },
            description="Source robot approaches object"
        ),
        WorkflowStep(
            operation_id="manipulation_control_gripper_001",
            parameter_bindings={"robot_id": "source_robot", "open_gripper": "False"},
            description="Source robot grasps object"
        ),
        WorkflowStep(
            operation_id="coordination_handoff_001",
            parameter_bindings={
                "source_robot": "source_robot",
                "target_robot": "target_robot",
                "handoff_position": "handoff_coords",
            },
            description="Coordinate handoff at designated position"
        ),
    ],
    variable_bindings={
        "object": "perception_stereo_detect_001.result",
    },
    success_criteria=[
        "Source robot successfully grasped object",
        "Both robots positioned at handoff location",
        "Target robot successfully received object",
        "Source robot released object",
    ],
    failure_recovery="If handoff fails, source robot returns object to original location",
    usage_examples=[
        "Robot1 hands blue cube to Robot2 at (0.0, 0.0, 0.3)",
        "Collaborative assembly with object transfer",
    ],
)


# ============================================================================
# Pattern Registry
# ============================================================================


class WorkflowPatternRegistry:
    """
    Central registry of all workflow patterns.

    Provides pattern lookup, searching, and RAG document generation.
    """

    def __init__(self):
        """Initialize registry with all defined patterns"""
        self.patterns: Dict[str, WorkflowPattern] = {}
        self._register_patterns()

    def _register_patterns(self):
        """Register all workflow patterns"""
        patterns = [
            DETECT_AND_APPROACH_PATTERN,
            PICK_AND_PLACE_PATTERN,
            VERIFY_AND_ACT_PATTERN,
            SIMULTANEOUS_MOVE_PATTERN,
            HANDOFF_PATTERN,
        ]

        for pattern in patterns:
            self.patterns[pattern.pattern_id] = pattern

    def get_pattern(self, pattern_id: str) -> Optional[WorkflowPattern]:
        """Retrieve pattern by ID"""
        return self.patterns.get(pattern_id)

    def get_all_patterns(self) -> List[WorkflowPattern]:
        """Get all registered patterns"""
        return list(self.patterns.values())

    def get_patterns_by_category(self, category: WorkflowCategory) -> List[WorkflowPattern]:
        """Get all patterns in a specific category"""
        return [p for p in self.patterns.values() if p.category == category]

    def search_patterns(self, query: str) -> List[WorkflowPattern]:
        """Simple text search over pattern names and descriptions"""
        query_lower = query.lower()
        matches = []

        for pattern in self.patterns.values():
            if (query_lower in pattern.name.lower() or
                query_lower in pattern.description.lower() or
                any(query_lower in step.description.lower() for step in pattern.steps)):
                matches.append(pattern)

        return matches


# Singleton instance for global access
_global_workflow_registry = None


def get_global_workflow_registry() -> WorkflowPatternRegistry:
    """Get or create the global workflow pattern registry"""
    global _global_workflow_registry
    if _global_workflow_registry is None:
        _global_workflow_registry = WorkflowPatternRegistry()
    return _global_workflow_registry
