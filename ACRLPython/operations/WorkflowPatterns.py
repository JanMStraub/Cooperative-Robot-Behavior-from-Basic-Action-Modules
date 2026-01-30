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
                bindings_str = ", ".join(
                    f"{k}=${v}" for k, v in step.parameter_bindings.items()
                )
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
            description="Detect object with stereo vision to get 3D coordinates",
        ),
        WorkflowStep(
            operation_id="motion_move_to_coord_001",
            parameter_bindings={
                "x": "detect_result.x",
                "y": "detect_result.y",
                "z": "detect_result.z",
            },
            description="Move robot to detected object position",
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
            description="Detect source object to pick up",
        ),
        WorkflowStep(
            operation_id="motion_move_to_coord_001",
            parameter_bindings={
                "x": "source.x",
                "y": "source.y",
                "z": "source.z",
                "approach_offset": "0.05",
            },
            description="Approach source object with offset",
        ),
        WorkflowStep(
            operation_id="manipulation_control_gripper_001",
            parameter_bindings={"open_gripper": "False"},
            description="Close gripper to grasp object",
        ),
        WorkflowStep(
            operation_id="motion_move_to_coord_001",
            parameter_bindings={
                "x": "target_x",
                "y": "target_y",
                "z": "target_z",
            },
            description="Move to target placement location",
        ),
        WorkflowStep(
            operation_id="manipulation_control_gripper_001",
            parameter_bindings={"open_gripper": "True"},
            description="Open gripper to release object",
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
            description="Use LLM vision to understand current scene state",
        ),
        WorkflowStep(
            operation_id="perception_stereo_detect_001",
            conditional="scene_analysis indicates object is present",
            description="Detect object if LLM confirms it's visible",
        ),
        WorkflowStep(
            operation_id="motion_move_to_coord_001",
            parameter_bindings={
                "x": "detection.x",
                "y": "detection.y",
                "z": "detection.z",
            },
            conditional="object detected successfully",
            description="Move to detected object",
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
    description="Move two robots simultaneously to different targets using atomic operations and sync primitives (LLM-driven coordination)",
    steps=[
        WorkflowStep(
            operation_id="status_check_robot_001",
            parameter_bindings={"robot_id": "robot1_id"},
            description="Check robot 1 status before moving",
        ),
        WorkflowStep(
            operation_id="status_check_robot_001",
            parameter_bindings={"robot_id": "robot2_id"},
            description="Check robot 2 status before moving",
        ),
        WorkflowStep(
            operation_id="motion_move_to_coord_001",
            parameter_bindings={
                "robot_id": "robot1_id",
                "x": "target1_x",
                "y": "target1_y",
                "z": "target1_z",
            },
            description="Robot1 moves to target (parallel with Robot2)",
        ),
        WorkflowStep(
            operation_id="motion_move_to_coord_001",
            parameter_bindings={
                "robot_id": "robot2_id",
                "x": "target2_x",
                "y": "target2_y",
                "z": "target2_z",
            },
            description="Robot2 moves to target (parallel with Robot1)",
        ),
        WorkflowStep(
            operation_id="sync_signal_001",
            parameter_bindings={"event_name": "both_robots_positioned"},
            description="Signal that both robots reached targets",
        ),
    ],
    variable_bindings={
        "robot1_status": "status_check_robot_001[0].result",
        "robot2_status": "status_check_robot_001[1].result",
    },
    success_criteria=[
        "Both robots ready to move",
        "Both robots reached targets successfully",
        "Unity's CollaborativeStrategy handled collision checking automatically",
    ],
    failure_recovery="If robot movement fails, retry with sequential execution using wait_for_signal",
    usage_examples=[
        "Move Robot1 to (0.3, 0.1, 0.2) and Robot2 to (0.3, -0.1, 0.2) simultaneously",
        "Coordinate dual-arm assembly task with parallel movements",
        "Use parallel_group in SequenceExecutor for true simultaneous execution",
    ],
)


HANDOFF_PATTERN = WorkflowPattern(
    pattern_id="workflow_handoff_001",
    name="handoff",
    category=WorkflowCategory.MULTI_ROBOT,
    description="Transfer object from one robot to another using atomic operations and synchronization primitives (LLM-driven coordination)",
    steps=[
        WorkflowStep(
            operation_id="perception_stereo_detect_001",
            parameter_bindings={"color": "object_color", "robot_id": "source_robot"},
            description="Detect object to handoff",
        ),
        WorkflowStep(
            operation_id="motion_move_to_coord_001",
            parameter_bindings={
                "robot_id": "source_robot",
                "x": "object.x",
                "y": "object.y",
                "z": "object.z",
            },
            description="Source robot approaches object",
        ),
        WorkflowStep(
            operation_id="manipulation_control_gripper_001",
            parameter_bindings={"robot_id": "source_robot", "open_gripper": "False"},
            description="Source robot grasps object",
        ),
        WorkflowStep(
            operation_id="sync_signal_001",
            parameter_bindings={"event_name": "object_gripped"},
            description="Signal that object has been gripped",
        ),
        WorkflowStep(
            operation_id="sync_wait_for_signal_001",
            parameter_bindings={"event_name": "object_gripped", "timeout_ms": "10000"},
            description="Target robot waits for grip confirmation (in parallel)",
        ),
        WorkflowStep(
            operation_id="motion_move_to_coord_001",
            parameter_bindings={
                "robot_id": "source_robot",
                "x": "handoff_x",
                "y": "handoff_y",
                "z": "handoff_z",
            },
            description="Source robot moves to handoff position",
        ),
        WorkflowStep(
            operation_id="motion_move_to_coord_001",
            parameter_bindings={
                "robot_id": "target_robot",
                "x": "handoff_x",
                "y": "handoff_y",
                "z": "handoff_z",
            },
            description="Target robot moves to handoff position (parallel)",
        ),
        WorkflowStep(
            operation_id="sync_signal_001",
            parameter_bindings={"event_name": "both_at_handoff"},
            description="Signal that both robots at handoff position",
        ),
        WorkflowStep(
            operation_id="manipulation_control_gripper_001",
            parameter_bindings={"robot_id": "target_robot", "open_gripper": "False"},
            description="Target robot closes gripper to receive object",
        ),
        WorkflowStep(
            operation_id="sync_wait_001",
            parameter_bindings={"duration_ms": "500"},
            description="Wait for gripper to fully close",
        ),
        WorkflowStep(
            operation_id="manipulation_control_gripper_001",
            parameter_bindings={"robot_id": "source_robot", "open_gripper": "True"},
            description="Source robot releases object",
        ),
        WorkflowStep(
            operation_id="sync_signal_001",
            parameter_bindings={"event_name": "handoff_complete"},
            description="Signal handoff completion",
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
        "Handoff completed without dropping object",
    ],
    failure_recovery="If handoff fails, source robot returns object to original location using reverse sequence",
    usage_examples=[
        "Robot1 hands blue cube to Robot2 at (0.0, 0.0, 0.3) using signal/wait coordination",
        "Collaborative assembly with synchronized object transfer",
        "Use parallel_group in SequenceExecutor for Robot1 and Robot2 movements",
    ],
)


LLM_DRIVEN_COORDINATION_PATTERN = WorkflowPattern(
    pattern_id="workflow_llm_coordination_001",
    name="llm_driven_coordination",
    category=WorkflowCategory.MULTI_ROBOT,
    description="Example of LLM-planned multi-robot coordination using only atomic operations and sync primitives - demonstrates the power of emergent coordination without hardcoded patterns",
    steps=[
        WorkflowStep(
            operation_id="perception_stereo_detect_001",
            parameter_bindings={"color": "red", "robot_id": "Robot1"},
            description="Robot1: Detect red cube",
        ),
        WorkflowStep(
            operation_id="motion_move_to_coord_001",
            parameter_bindings={
                "robot_id": "Robot1",
                "x": "cube.x",
                "y": "cube.y",
                "z": "cube.z",
            },
            description="Robot1: Move to cube",
        ),
        WorkflowStep(
            operation_id="manipulation_control_gripper_001",
            parameter_bindings={"robot_id": "Robot1", "open_gripper": "False"},
            description="Robot1: Grip cube",
        ),
        WorkflowStep(
            operation_id="sync_signal_001",
            parameter_bindings={"event_name": "cube_gripped"},
            description="Robot1: Signal that cube is gripped",
        ),
        WorkflowStep(
            operation_id="sync_wait_for_signal_001",
            parameter_bindings={"event_name": "cube_gripped", "timeout_ms": "5000"},
            description="Robot2: Wait for Robot1 to grip cube (parallel execution)",
        ),
        WorkflowStep(
            operation_id="motion_move_to_coord_001",
            parameter_bindings={
                "robot_id": "Robot1",
                "x": "0.0",
                "y": "0.0",
                "z": "0.3",
            },
            description="Robot1: Move to handoff position",
        ),
        WorkflowStep(
            operation_id="motion_move_to_coord_001",
            parameter_bindings={
                "robot_id": "Robot2",
                "x": "0.0",
                "y": "0.0",
                "z": "0.3",
            },
            description="Robot2: Move to handoff position (parallel with Robot1)",
        ),
        WorkflowStep(
            operation_id="sync_signal_001",
            parameter_bindings={"event_name": "robot1_ready"},
            description="Robot1: Signal arrival at handoff",
        ),
        WorkflowStep(
            operation_id="sync_signal_001",
            parameter_bindings={"event_name": "robot2_ready"},
            description="Robot2: Signal arrival at handoff",
        ),
        WorkflowStep(
            operation_id="sync_wait_for_signal_001",
            parameter_bindings={"event_name": "robot2_ready", "timeout_ms": "5000"},
            description="Robot1: Wait for Robot2 to arrive",
        ),
        WorkflowStep(
            operation_id="sync_wait_for_signal_001",
            parameter_bindings={"event_name": "robot1_ready", "timeout_ms": "5000"},
            description="Robot2: Wait for Robot1 to arrive",
        ),
        WorkflowStep(
            operation_id="manipulation_control_gripper_001",
            parameter_bindings={"robot_id": "Robot2", "open_gripper": "False"},
            description="Robot2: Close gripper to receive cube",
        ),
        WorkflowStep(
            operation_id="sync_wait_001",
            parameter_bindings={"duration_ms": "500"},
            description="Wait for gripper to stabilize",
        ),
        WorkflowStep(
            operation_id="manipulation_control_gripper_001",
            parameter_bindings={"robot_id": "Robot1", "open_gripper": "True"},
            description="Robot1: Release cube",
        ),
    ],
    variable_bindings={
        "cube": "perception_stereo_detect_001.result",
    },
    success_criteria=[
        "Cube detected by Robot1",
        "Robot1 successfully gripped cube",
        "Both robots synchronized at handoff position",
        "Cube transferred to Robot2",
        "Robot1 released cube safely",
    ],
    failure_recovery="Use signal/wait error codes to detect coordination failures and retry individual steps",
    usage_examples=[
        "LLM plans: 'Locate the red cube and give it to the other robot'",
        "LLM generates this exact sequence from atomic operations knowledge",
        "No hardcoded coordination - LLM decides sync points and parallel groups",
        "Use parallel_group=[0,0,1,2,2,3,3,4,4,5,5,6,7,8] for optimal execution",
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
            LLM_DRIVEN_COORDINATION_PATTERN,
        ]

        for pattern in patterns:
            self.patterns[pattern.pattern_id] = pattern

    def get_pattern(self, pattern_id: str) -> Optional[WorkflowPattern]:
        """Retrieve pattern by ID"""
        return self.patterns.get(pattern_id)

    def get_pattern_by_name(self, name: str) -> Optional[WorkflowPattern]:
        """Retrieve pattern by name (e.g., 'handoff', 'pick_and_place')"""
        for pattern in self.patterns.values():
            if pattern.name == name:
                return pattern
        return None

    def get_all_patterns(self) -> List[WorkflowPattern]:
        """Get all registered patterns"""
        return list(self.patterns.values())

    def get_patterns_by_category(
        self, category: WorkflowCategory
    ) -> List[WorkflowPattern]:
        """Get all patterns in a specific category"""
        return [p for p in self.patterns.values() if p.category == category]

    def search_patterns(self, query: str) -> List[WorkflowPattern]:
        """Simple text search over pattern names and descriptions"""
        query_lower = query.lower()
        matches = []

        for pattern in self.patterns.values():
            if (
                query_lower in pattern.name.lower()
                or query_lower in pattern.description.lower()
                or any(
                    query_lower in step.description.lower() for step in pattern.steps
                )
            ):
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


# ============================================================================
# TEXT-BASED PATTERNS (for RAG indexing and LLM guidance)
# ============================================================================
# These are detailed textual descriptions removed from non-atomic operations.
# They guide the LLM on how to chain atomic operations for complex workflows.
# ============================================================================

HANDOFF_TEXT_PATTERN = """
REMOVED OPERATION: hand_over_object_to_another_robot
=====================================================

This operation was REMOVED because it is NON-ATOMIC (combines 5 steps).

To perform an object handoff, the LLM should chain these ATOMIC operations:

**Step-by-Step Atomic Operation Sequence:**

1. Source robot moves to handoff region
   → move_to_coordinate(robot_from, handoff_position)

2. Source robot signals ready
   → signal(robot_from, "ready_for_handoff")

3. Target robot waits for signal and moves
   → wait_for_signal(robot_to, "ready_for_handoff")
   → move_to_coordinate(robot_to, handoff_position)

4. Target robot grips object with object_id attachment
   → control_gripper(robot_to, open=False, object_id=object)

5. Source robot releases
   → control_gripper(robot_from, open=True)
   OR: release_object(robot_from)

6. Robots signal completion and move away
   → signal(robot_to, "handoff_complete")
   → move_to_coordinate(robot_from, safe_position)

**Why Removed:**
The original operation hid coordination complexity from the LLM.
By exposing atomic operations, the LLM can:
- See exactly what happens at each step
- Debug failures more easily
- Adapt the sequence to specific contexts
- Learn from successful coordination patterns

**Example LLM Usage:**
"Robot1, hand the red cube to Robot2"

LLM generates:
```
move_to_coordinate("Robot1", x=0.0, y=0.0, z=0.15)
signal("Robot1", "ready_for_handoff")
wait_for_signal("Robot2", "ready_for_handoff")
move_to_coordinate("Robot2", x=0.0, y=0.0, z=0.15)
control_gripper("Robot2", open_gripper=False, object_id="RedCube")
release_object("Robot1")
signal("Robot2", "handoff_complete")
```
"""

STABILIZE_MANIPULATE_TEXT_PATTERN = """
REMOVED OPERATION: stabilize_and_manipulate_collaboratively
=============================================================

This operation was REMOVED because it is NON-ATOMIC (combines grasp + hold + manipulate).

To perform collaborative manipulation, the LLM should chain these ATOMIC operations:

**Step-by-Step Atomic Operation Sequence:**

**Robot1 (Stabilizer):**
1. Move to object
   → move_to_coordinate(robot1, object_position)

2. Grasp object
   → grasp_object(robot1, object_coords)

3. Hold stable with force control
   → stabilize_object(robot1, object_id, duration_ms=10000)

4. Signal stabilization active
   → signal(robot1, "stabilization_active")

5. Wait for manipulation complete
   → wait_for_signal(robot1, "manipulation_complete")

6. Release object
   → control_gripper(robot1, open=True)

**Robot2 (Manipulator) - runs in parallel:**
1. Wait for stabilization active
   → wait_for_signal(robot2, "stabilization_active")

2. Move to manipulation position
   → move_to_coordinate(robot2, manipulation_position)

3. Perform manipulation operation
   → [insert/assemble/etc operation]

4. Signal completion
   → signal(robot2, "manipulation_complete")

**Why Removed:**
The original operation combined multiple robot behaviors into one.
By exposing atomic operations, the LLM can:
- Coordinate parallel robot behaviors explicitly
- Choose different manipulation strategies
- Handle edge cases and failures at each step
- See the full coordination protocol

**Example LLM Usage:**
"Robot1, hold the board stable while Robot2 inserts a peg"

LLM generates for Robot1:
```
move_to_coordinate("Robot1", x=0.2, y=0.0, z=0.05)
grasp_object("Robot1", object_coords={"x": 0.2, "y": 0.0, "z": 0.05})
stabilize_object("Robot1", object_id="PegBoard", duration_ms=10000)
signal("Robot1", "stabilization_active")
wait_for_signal("Robot1", "manipulation_complete")
control_gripper("Robot1", open_gripper=True)
```

LLM generates for Robot2 (parallel):
```
wait_for_signal("Robot2", "stabilization_active")
grasp_object("Robot2", object_coords={"x": 0.1, "y": 0.1, "z": 0.05})
move_to_coordinate("Robot2", x=0.2, y=0.0, z=0.08)
move_to_coordinate("Robot2", x=0.2, y=0.0, z=0.05)  # Insert
control_gripper("Robot2", open_gripper=True)
signal("Robot2", "manipulation_complete")
```
"""
