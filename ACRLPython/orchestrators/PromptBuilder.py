#!/usr/bin/env python3
"""LLM prompt assembly for CommandParser."""

from typing import List

try:
    from ..rag import RAGSystem
    from ..config.Robot import HANDOFF_PRESENTATION_POSITION
    from ..operations.WorkflowPatterns import WorkflowPatternRegistry, WorkflowPattern
except ImportError:
    from rag import RAGSystem
    from config.Robot import HANDOFF_PRESENTATION_POSITION
    from operations.WorkflowPatterns import WorkflowPatternRegistry, WorkflowPattern

import logging

logger = logging.getLogger(__name__)

_HPP = HANDOFF_PRESENTATION_POSITION
_HANDOFF_X, _HANDOFF_Y, _HANDOFF_Z = _HPP[0], _HPP[1], _HPP[2]


class PromptBuilder:
    """Assembles LLM parsing prompts; separated so prompt logic can be unit-tested without a live LLM."""

    def __init__(self, registry, workflow_registry, rag):
        self.registry = registry
        self.workflow_registry = workflow_registry
        self.rag = rag

    def build(
        self,
        command_text: str,
        robot_id: str,
        anti_pattern_section: str = "",
        spatial_section: str = "",
        hint: str = "",
    ) -> str:
        available_ops = self.get_available_operations_summary(command_text)
        anti_pattern_block = (
            f"\n        {anti_pattern_section}\n" if anti_pattern_section else ""
        )
        spatial_block = (
            f"\n        {spatial_section}\n"
            "        NOTE: Spatial context is reference only. Use ONLY operations from Available Operations above.\n"
            if spatial_section
            else ""
        )
        reflection_block = (
            f"\n        === REFLECTION ===\n        {hint}\n" if hint else ""
        )

        # Suppress the default-robot hint for multi-robot tasks — the task text
        # names both robots explicitly, so a default would bias the LLM to one robot.
        import re as _re
        _named_robots = {rid for rid in ["Robot1", "Robot2"] if _re.search(r"\b" + rid + r"\b", command_text)}
        _robot_id_line = (
            f'Default robot_id: "{robot_id}"'
            if len(_named_robots) <= 1
            else f'Robots in task: {", ".join(sorted(_named_robots))} — assign robot_id per-op as named in the task'
        )

        return f"""
        Available Operations: {available_ops}

        Command to parse: "{command_text}"
        {_robot_id_line}

        === ROBOT WORKSPACE BOUNDARIES ===

        Robot1 (left, x=-0.475): reachable x < 0.165. Robot2 (right, x=+0.475): reachable x > -0.165.
        x > 0 -> Robot2's side. x < 0 -> Robot1's side. x = 0 -> shared.
        Wrong-side task -> use HANDOFF sequence.

        === MULTI-ROBOT COORDINATION ===

        Multi-robot tasks use "plan" format (with "reasoning" + per-op "parallel_group"). Single-robot -> "commands" format.
        Same parallel_group = concurrent. Later group waits for all prior groups.
        VARIABLE DEPENDENCY LAW: if B reads $var captured by A, B must have strictly higher parallel_group than A. Never same group.

        === HANDOFF RULE ===

        Exact steps with required parallel_group numbers (no composite ops, no deviations):
        group=1: Robot1: detect_object_stereo (capture_var="target")
        group=2: Robot1: grasp_object(object_id="$target.color"): MUST be group=2 (after detect)
        group=3: Robot1: return_to_start_position: MUST be group=3 (after grasp, never same group as grasp)
        group=4: Robot1: move_to_coordinate({_HANDOFF_X:.2f}, {_HANDOFF_Y:.2f}, {_HANDOFF_Z:.2f}): NO approach_offset; own group
        group=5: Robot1: adjust_end_effector_orientation(pitch=0, yaw=0, roll=0)
        group=6: Robot1: signal("r1_at_handoff") + Robot2: wait_for_signal("r1_at_handoff"): SAME group
        group=7: Robot2: detect_object_stereo(color=<same as step 1>, capture_var="handoff_target"): object moved with Robot1
        group=8: Robot2: receive_handoff(object_id="$handoff_target.color", source_robot_id="Robot1")
        group=9: Robot1: release_object

        Receiving robot: always receive_handoff, never grasp_object. Handoff coord is always exactly ({_HANDOFF_X:.2f}, {_HANDOFF_Y:.2f}, {_HANDOFF_Z:.2f}). Step 7 color must match step 1 (never null). signal+wait_for_signal always same group.

        === SYNCHRONIZATION PRIMITIVES ===

        signal(event_name): emit event. wait_for_signal(event_name, timeout_ms=30000): wait for event. wait(duration_ms): time pause.
        mirror_movement_of_other_robot(duration_ms): duration_ms in [1000, 60000].

        === NAVIGATION RULE ===

        Move/navigate/approach WITHOUT pick/grab/grasp language -> move_to_coordinate only (no gripper ops).
        Always set approach_offset=0.10 when moving to a detected object (lifts gripper above table; range 0.0-0.10).
        receive_handoff is not navigation: never replace with move_to_coordinate.

        "detect blue cube and move to it":
        {{"operation": "detect_object_stereo", "params": {{"robot_id": "Robot1", "color": "blue"}}, "capture_var": "target"}}
        {{"operation": "move_to_coordinate", "params": {{"robot_id": "Robot1", "x": "$target.x", "y": "$target.y", "z": "$target.z", "approach_offset": 0.10}}}}

        === GRASP RULE ===

        Pick/grab/grasp -> grasp_object (handles approach+descent+grip; no separate move_to_coordinate before it).
        object_id always uses ".color" from detection ($target.color: never .id or .name).
        Never use grasp_object with a $field var (detect_field has no .color). Never on place/deposit tasks. Never for receiving robot in handoff (use receive_handoff).

        === PLACE RULE ===

        Place/drop/deposit -> place_object(x, y, z): hover, descend, open, ascend. Not release_object, not control_gripper.
        release_object: only for immediate drop at current position (emergency/handoff transfer).
        Typical sequence: detect_field -> place_object($field.x/y/z).

        === BETWEEN PLACEMENT ===

        When the task says "place between X and Y", "put it midway between", or "place in the middle of X and Y":
        PREFER place_between_objects: it resolves both objects from WorldState and computes the midpoint internally.

        Example: "place the held object between the blue and red cube":
        {{"operation": "detect_object_stereo", "params": {{"robot_id": "Robot1", "color": "blue"}}, "parallel_group": 1}}
        {{"operation": "detect_object_stereo", "params": {{"robot_id": "Robot1", "color": "red"}}, "parallel_group": 1}}
        {{"operation": "place_between_objects", "params": {{"robot_id": "Robot1", "object_id_1": "blue", "object_id_2": "red"}}, "parallel_group": 2}}

        The two detect calls CAN share the same parallel_group (they are independent of each other).
        place_between_objects MUST be in a strictly higher parallel_group than both detects.

        Fallback: if objects are already in WorldState (no detection needed):
        {{"operation": "place_between_objects", "params": {{"robot_id": "Robot1", "object_id_1": "blue_cube", "object_id_2": "red_cube"}}}}

        Multi-variable arithmetic in params is also supported when you need a custom midpoint:
        {{"operation": "place_object", "params": {{"robot_id": "Robot1", "x": "($blue_obj.x + $red_obj.x) / 2", "y": "($blue_obj.y + $red_obj.y) / 2", "z": "($blue_obj.z + $red_obj.z) / 2"}}}}

        === INDEPENDENT PARALLEL RULE ===

        When two robots have fully independent tasks (no shared objects, no handoff, no sync point), assign MATCHING parallel_group numbers so both chains execute simultaneously.
        Step N for Robot1 and step N for Robot2 belong in the same group — do NOT assign monotonically increasing groups across both robots.
        Use distinct capture_var names per robot (e.g. "r1_target" / "r2_target") to avoid collisions.

        === SINGLE-ROBOT RULES ===

        Each action = separate op. Include robot_id in every op. Preserve order.
        "close gripper/grip" (not a pick) -> control_gripper(open_gripper=false). "open gripper/release" -> open_gripper=true.
        Never add return_to_start, signal, or adjust_end_effector_orientation after a grasp unless explicitly requested.

        "grab the blue cube": detect_object_stereo(color="blue", capture_var="target") -> grasp_object(object_id="$target.color")

        === VARIABLE PASSING ===

        capture_var defines a variable; $var references it in later ops (never before capture).
        detect_object_stereo fields: x, y, z, color, confidence. For grasp: $target.color (never .id/.name).
        detect_field fields: x, y, z directly (use $field.x not $field.center.x).

        Pick-and-place: detect_object_stereo(capture_var="target") -> grasp_object($target.color) -> detect_field(field_label="G", capture_var="field") -> place_object($field.x, $field.y, $field.z)

        === DETECT_FIELD RULE (CRITICAL) ===
        detect_field ALWAYS requires field_label (a single letter A-I). NEVER omit it.
        WRONG: {{"operation": "detect_field", "params": {{"robot_id": "Robot1"}}}}
        RIGHT: {{"operation": "detect_field", "params": {{"robot_id": "Robot1", "field_label": "A"}}}}
        If the task does not specify a field letter, infer it from context or ask: do NOT emit detect_field without field_label.

        {spatial_block}{anti_pattern_block}{reflection_block} Output only valid JSON, no explanation, no comments."""

    def get_available_operations_summary(self, command_text: str = "") -> str:
        """Return ops relevant to command_text (via RAG if available, else full list)."""
        if self.rag and command_text:
            try:
                rag_results = self.rag.search(command_text, top_k=8)
                relevant_ops = set()
                workflow_results = []
                operation_results = []
                summary_lines = []

                if rag_results:
                    for result in rag_results:
                        result_type = result.get("metadata", {}).get(
                            "type", "operation"
                        )
                        if result_type == "workflow":
                            workflow_results.append(result)
                        else:
                            operation_results.append(result)

                if workflow_results:
                    summary_lines.append("=== RELEVANT WORKFLOW PATTERNS ===")
                    for result in workflow_results[:3]:
                        pattern_name = result.get("name", "")
                        pattern = self.workflow_registry.get_pattern_by_name(
                            pattern_name
                        )
                        if pattern:
                            summary_lines.append(self.format_workflow_pattern(pattern))
                        else:
                            pattern_id = result.get("metadata", {}).get(
                                "pattern_id", ""
                            )
                            if pattern_id:
                                pattern = self.workflow_registry.get_pattern(pattern_id)
                                if pattern:
                                    summary_lines.append(
                                        self.format_workflow_pattern(pattern)
                                    )
                    summary_lines.append("\n=== MOST RELEVANT OPERATIONS ===")

                ops_added = 0
                if operation_results:
                    if not workflow_results:
                        summary_lines.append(
                            "Most relevant operations for this command:"
                        )
                    for result in operation_results[:5]:
                        op = self.registry.get_operation_by_name(result.get("name", ""))
                        if op:
                            relevant_ops.add(op.name)
                            params = self.format_parameters(op.parameters)
                            score = result.get("similarity_score", 0)
                            summary_lines.append(
                                f"- {op.name}({params}): {op.description} [relevance: {score:.2f}]"
                            )
                            ops_added += 1

                if ops_added == 0:
                    # No RAG op hits or all lookups failed — list all ops as fallback
                    for op in self.registry.get_all_operations():
                        params = self.format_parameters(op.parameters)
                        summary_lines.append(f"- {op.name}({params}): {op.description}")
                return "\n".join(summary_lines)

            except Exception as e:
                logger.warning("RAG search failed, using registry: %s", e)

        ops = self.registry.get_all_operations()
        summary_lines = []
        for op in ops:
            params = self.format_parameters(op.parameters)
            summary_lines.append(f"- {op.name}({params}): {op.description}")
        return "\n".join(summary_lines)

    def format_parameters(self, parameters: List) -> str:
        param_strs = []
        for p in parameters:
            param_str = f"{p.name}: {p.type}"
            if hasattr(p, "valid_values") and p.valid_values:
                valid_vals = ", ".join(
                    ["null" if v is None else f"'{v}'" for v in p.valid_values]
                )
                param_str += f" (valid: {valid_vals})"
            param_strs.append(param_str)
        return ", ".join(param_strs)

    def format_workflow_pattern(self, pattern: WorkflowPattern) -> str:
        steps_text = "\n".join(
            f"    {i}. {step.operation_id}: {step.description}"
            for i, step in enumerate(pattern.steps, 1)
        )
        examples = "\n".join(f"  - {ex}" for ex in pattern.usage_examples[:2])
        return f"""
Pattern: {pattern.name}
Description: {pattern.description}
Steps:
{steps_text}
Examples:
{examples}
"""
