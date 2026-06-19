#!/usr/bin/env python3
"""LLM prompt assembly for CommandParser."""

import re
from typing import FrozenSet, List

try:
    from ..config.Robot import HANDOFF_PRESENTATION_POSITION
    from ..operations.WorkflowPatterns import WorkflowPattern
except ImportError:
    from config.Robot import HANDOFF_PRESENTATION_POSITION
    from operations.WorkflowPatterns import WorkflowPattern

import logging

logger = logging.getLogger(__name__)

_HPP = HANDOFF_PRESENTATION_POSITION
_HANDOFF_X, _HANDOFF_Y, _HANDOFF_Z = _HPP[0], _HPP[1], _HPP[2]

# ── Rule section constants ────────────────────────────────────────────────────
# Extracted at module level so they are built once and reused across calls.

_SECTION_WORKSPACE = """\
=== ROBOT WORKSPACE BOUNDARIES ===

Robot1 (left, x=-0.475): reachable x < 0.165. Robot2 (right, x=+0.475): reachable x > -0.165.
x > 0 -> Robot2's side. x < 0 -> Robot1's side. x = 0 -> shared.
Wrong-side task -> use HANDOFF sequence.
SINGULARITY RULE: Never command a robot to a position directly above its own base.
  Robot1 base column: x=-0.475, z=0 -> keep targets at x > -0.36 or x < -0.59 to avoid.
  Robot2 base column: x=+0.475, z=0 -> keep targets at x < +0.36 or x > +0.59 to avoid."""

_SECTION_ROBOT_ASSIGNMENT = """\
=== ROBOT ASSIGNMENT RULE ===
Assign robot_id based ONLY on which robot can physically reach the object.
"The other robot", "second robot", or "Robot2" are NOT valid reasons by themselves.
Cross-check every grasp/pick against the 'Object reachability by robot' section below.
If an object appears only under Robot1, always use robot_id='Robot1', regardless of task wording."""

_SECTION_MULTI_ROBOT = """\
=== MULTI-ROBOT COORDINATION ===

Multi-robot tasks use "plan" format (with "reasoning" + per-op "parallel_group"). Single-robot -> "commands" format.
Same parallel_group = concurrent. Later group waits for all prior groups.
VARIABLE DEPENDENCY LAW: if B reads $var captured by A, B must have strictly higher parallel_group than A. Never same group."""

_SECTION_HANDOFF = f"""\
=== HANDOFF RULE ===

Handoff = Robot1 picks an object and transfers it to Robot2 at the handoff position ({_HANDOFF_X:.2f}, {_HANDOFF_Y:.2f}, {_HANDOFF_Z:.2f}).

Ordering constraints (derive parallel_group numbers yourself):
- Detect must complete before grasp (needs object location).
- return_to_start must follow grasp before moving to handoff position (clear workspace path).
- move_to_coordinate to handoff position must complete before adjust_end_effector_orientation starts (arm must be stationary; IK conflicts with active trajectory -> timeout).
- adjust_end_effector_orientation(pitch=0, yaw=0, roll=90) must complete before signal (Robot2 must not approach until the object is in the correct orientation).
- Robot1 signals when in position AND orientation is done; Robot2 must wait_for_signal before detecting: signal and wait_for_signal are concurrent (same parallel_group).
- Robot2 detects the object at Robot1's hand (same color as originally detected, capture_var="handoff_target"), then calls receive_handoff(robot_id="Robot2", object_id="$handoff_target.color", source_robot_id="Robot1").
- Robot1 releases only after receive_handoff completes.
- Robot1 returns to start after releasing.

Hard constraints:
- Receiving robot always uses receive_handoff - never grasp_object.
- Handoff coordinate is exactly ({_HANDOFF_X:.2f}, {_HANDOFF_Y:.2f}, {_HANDOFF_Z:.2f}); no approach_offset on that move.
- Robot2's detect color must match Robot1's detect color (never null).
- signal + wait_for_signal must share the same parallel_group."""

_SECTION_SYNC = """\
=== SYNCHRONIZATION PRIMITIVES ===

signal(event_name): emit event. wait_for_signal(event_name, timeout_ms=30000): wait for event. wait(duration_ms): time pause.
mirror_movement_of_other_robot(duration_ms): duration_ms in [1000, 60000]."""

_SECTION_NAVIGATION = """\
=== NAVIGATION RULE ===

Move/navigate/approach WITHOUT pick/grab/grasp language -> move_to_coordinate only (no gripper ops).
Always set approach_offset=0.10 when moving to a detected object (lifts gripper above table; range 0.0-0.10).
receive_handoff is not navigation: never replace with move_to_coordinate.
"return to start" / "go home" / "home position" -> return_to_start_position, NEVER move_to_coordinate with hardcoded coordinates.

"find the yellow cube and approach it":
{"operation": "detect_object_stereo", "params": {"robot_id": "Robot1", "color": "yellow"}, "capture_var": "target"}
{"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "x": "$target.x", "y": "$target.y", "z": "$target.z", "approach_offset": 0.10}}

"pick up the red cube and place it on field a, then return to start":
{"operation": "detect_object_stereo", "params": {"robot_id": "Robot1", "color": "red"}, "capture_var": "target"}
{"operation": "grasp_object", "params": {"robot_id": "Robot1", "object_id": "$target.color"}}
{"operation": "detect_field", "params": {"robot_id": "Robot1", "field_label": "a"}, "capture_var": "field"}
{"operation": "place_object", "params": {"robot_id": "Robot1", "x": "$field.x", "y": "$field.y", "z": "$field.z"}}
{"operation": "return_to_start_position", "params": {"robot_id": "Robot1"}}"""

_SECTION_GRASP = """\
=== GRASP RULE ===

Pick/grab/grasp -> grasp_object (handles approach+descent+grip; no separate move_to_coordinate before it).
object_id always uses ".color" from detection ($target.color: never .id or .name).
Never use grasp_object with a $field var (detect_field has no .color). Never on place/deposit tasks. Never for receiving robot in handoff (use receive_handoff).
If the command explicitly requests a move AFTER grasping (e.g. "lift to y=0.2", "raise to height H"), add move_to_coordinate AFTER grasp_object using $target.x/$target.z for the horizontal position.

"pick up the green cube and raise it to y=0.15":
{"operation": "detect_object_stereo", "params": {"robot_id": "Robot1", "color": "green"}, "capture_var": "target"}
{"operation": "grasp_object", "params": {"robot_id": "Robot1", "object_id": "$target.color"}}
{"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "x": "$target.x", "y": 0.15, "z": "$target.z"}}"""

_SECTION_PLACE = """\
=== PLACE RULE ===

Place/drop/deposit -> place_object(x, y, z): hover, descend, open, ascend. Not release_object, not control_gripper.
release_object: only for immediate drop at current position (emergency/handoff transfer).
Typical sequence: detect_field -> place_object($field.x, $field.y, $field.z).
on_top_of: OPTIONAL string object ID (e.g. "blue_cube"), this auto-computes Y so object stacks on top. NEVER pass true/false; omit entirely when placing in a field.

=== STACKING RULE ===

When the task says "on top of [object]", "stack on [object]", or "place on [object]":
- Use on_top_of="<object_id>" (e.g. "red_cube"), as this computes the correct Y automatically.
- x and z MUST come from that same object (its WorldState position or a prior detect_object_stereo), NOT from detect_field.
- placed_object_height: height of the held object in metres (e.g. 0.02 for a small cube). Omit or set 0.0 if unknown.

Example: "pick up the cyan cube and stack it on top of the orange cube":
{"operation": "detect_object_stereo", "params": {"robot_id": "Robot1", "color": "cyan"}, "capture_var": "cyan_obj"}
{"operation": "grasp_object", "params": {"robot_id": "Robot1", "object_id": "$cyan_obj.color"}}
{"operation": "detect_object_stereo", "params": {"robot_id": "Robot1", "color": "orange"}, "capture_var": "orange_obj"}
{"operation": "place_object", "params": {"robot_id": "Robot1", "x": "$orange_obj.x", "y": "$orange_obj.y", "z": "$orange_obj.z", "on_top_of": "orange_cube", "placed_object_height": 0.02}}

If the target object is already in WorldState (no detect needed), skip the detect step and use on_top_of directly."""

_SECTION_BETWEEN = """\
=== BETWEEN PLACEMENT ===

When the task says "place between X and Y", "put it midway between", or "place in the middle of X and Y":
PREFER place_between_objects: it resolves both objects from WorldState and computes the midpoint internally.

Example: "place the held object between the cyan and orange cube":
{"operation": "detect_object_stereo", "params": {"robot_id": "Robot1", "color": "cyan"}, "parallel_group": 1}
{"operation": "detect_object_stereo", "params": {"robot_id": "Robot1", "color": "orange"}, "parallel_group": 1}
{"operation": "place_between_objects", "params": {"robot_id": "Robot1", "object_id_1": "cyan", "object_id_2": "orange"}, "parallel_group": 2}

The two detect calls are independent, so assign them the same parallel_group so they run concurrently.
place_between_objects depends on both detects, so it must have a strictly higher parallel_group than both.

Fallback: if objects are already in WorldState (no detection needed):
{"operation": "place_between_objects", "params": {"robot_id": "Robot1", "object_id_1": "cyan_cube", "object_id_2": "orange_cube"}}

Multi-variable arithmetic in params is also supported when you need a custom midpoint:
{"operation": "place_object", "params": {"robot_id": "Robot1", "x": "($blue_obj.x + $red_obj.x) / 2", "y": "($blue_obj.y + $red_obj.y) / 2", "z": "($blue_obj.z + $red_obj.z) / 2"}}"""

_SECTION_INDEPENDENT_PARALLEL = """\
=== INDEPENDENT PARALLEL RULE ===

When two robots have fully independent tasks (no shared objects, no handoff, no sync point), assign MATCHING parallel_group numbers so both chains execute simultaneously.
Step N for Robot1 and step N for Robot2 belong in the same group, do NOT assign monotonically increasing groups across both robots.
Use distinct capture_var names per robot (e.g. "r1_target" / "r2_target") to avoid collisions.

Example: "Robot1 picks the red cube and places it in field C; Robot2 picks the yellow cube and places it in field D":
{"operation": "detect_object_stereo", "params": {"robot_id": "Robot1", "color": "red"}, "capture_var": "r1_target", "parallel_group": 1}
{"operation": "detect_object_stereo", "params": {"robot_id": "Robot2", "color": "yellow"}, "capture_var": "r2_target", "parallel_group": 1}
{"operation": "grasp_object", "params": {"robot_id": "Robot1", "object_id": "$r1_target.color"}, "parallel_group": 2}
{"operation": "grasp_object", "params": {"robot_id": "Robot2", "object_id": "$r2_target.color"}, "parallel_group": 2}
{"operation": "detect_field", "params": {"robot_id": "Robot1", "field_label": "c"}, "capture_var": "r1_field", "parallel_group": 3}
{"operation": "detect_field", "params": {"robot_id": "Robot2", "field_label": "d"}, "capture_var": "r2_field", "parallel_group": 3}
{"operation": "place_object", "params": {"robot_id": "Robot1", "x": "$r1_field.x", "y": "$r1_field.y", "z": "$r1_field.z"}, "parallel_group": 4}
{"operation": "place_object", "params": {"robot_id": "Robot2", "x": "$r2_field.x", "y": "$r2_field.y", "z": "$r2_field.z"}, "parallel_group": 4}

grasp is group 2 (not 1) because it reads $r1_target/$r2_target captured in group 1. Variable dependency forces a new group, but both robots' grasps still share group 2. Same logic for place after detect_field."""

_SECTION_COOPERATIVE = """\
=== COOPERATIVE POSITIONING RULE ===

When both robots work on one shared object but only ONE grips and the other braces/supports/steadies (NOT a handoff):
- Gripping robot: grasp_object with preferred_approach="left_side" or "right_side".
- Support/brace robot: move_to_coordinate to opposite side (x offset +-0.08m, y=0.08 above table and NOT $target.y) -> NEVER grasp_object.
- Grasp and brace can happen concurrently (both depend only on the prior detect, not on each other).
- Lift: both robots move to target height concurrently (neither depends on the other's lift).

Ordering rule: detect -> [grasp, brace] -> [lift, lift]. Derive parallel_group numbers from these dependencies.

Trigger phrases: "brace", "support", "steadies", "one grasps ... other supports"."""

_SECTION_DUAL_GRASP = """\
=== DUAL GRASP RULE ===

When BOTH robots physically grip the same object from opposite sides (both "grasp", "grip", or "pick"):
- Both robots use grasp_object with complementary preferred_approach ("left_side" / "right_side").
- Grasps must be sequential, not concurrent, so the stabilising robot grasps first to prevent object drift, then the second robot grasps. Each grasp waits for the previous to complete.
- Lift: both robots move_to_coordinate to target height concurrently (neither lift depends on the other).

Ordering rule: detect -> grasp_A -> grasp_B -> [lift_A, lift_B]. Derive parallel_group numbers from these dependencies.

Trigger phrases: "both grasp", "both grip", "both pick", "Robot1 grasps ... Robot2 grasps", "cooperatively grasp", "cooperatively handle" (when both sides explicitly grip)."""

_SECTION_SINGLE_ROBOT = """\
=== SINGLE-ROBOT RULES ===

Each action = separate op. Include robot_id in every op. Preserve order.
"close gripper/grip" (not a pick) -> control_gripper(open_gripper=false). "open gripper/release" -> open_gripper=true.
Never add return_to_start, signal, or adjust_end_effector_orientation after a grasp unless explicitly requested.

"grip the purple cube": detect_object_stereo(color="purple", capture_var="target") -> grasp_object(object_id="$target.color")"""

_SECTION_VARIABLE_PASSING = """\
=== VARIABLE PASSING ===

capture_var defines a variable; $var references it in later ops (never before capture).
detect_object_stereo fields: x, y, z, color, confidence. For grasp: $target.color (never .id/.name).
detect_field fields: x, y, z directly (use $field.x not $field.center.x).

Pick-and-place: detect_object_stereo(color="purple", capture_var="target") -> grasp_object($target.color) -> detect_field(field_label="C", capture_var="field") -> place_object($field.x, $field.y, $field.z)"""

_SECTION_DETECT_FIELD = """\
=== DETECT_FIELD RULE (CRITICAL) ===
detect_field ALWAYS requires field_label (a single letter A-I). NEVER omit it.
WRONG: {"operation": "detect_field", "params": {"robot_id": "Robot1"}}
RIGHT: {"operation": "detect_field", "params": {"robot_id": "Robot1", "field_label": "A"}}
If the task does not specify a field letter, infer it from context or ask: do NOT emit detect_field without field_label."""

# ── Intent classifiers ────────────────────────────────────────────────────────

_RE_HANDOFF = re.compile(r"\b(hand\w*|pass|give|transfer|handoff)\b", re.IGNORECASE)
_RE_BETWEEN = re.compile(
    r"\b(between|midway|middle\s+of|place\s+in\s+the\s+middle)\b", re.IGNORECASE
)
_RE_COOPERATIVE = re.compile(
    r"\b(brace|support|steadies|one\s+grasps|other\s+supports)\b", re.IGNORECASE
)
_RE_DUAL_GRASP = re.compile(
    r"\b(both\s+(?:robots?\s+)?grasp|both\s+grip|both\s+pick|cooperatively\s+grasp|cooperatively\s+handle)\b",
    re.IGNORECASE,
)
_RE_SYNC = re.compile(
    r"\b(signal|wait\s+for\s+signal|mirror|synchronized)\b", re.IGNORECASE
)
_RE_DETECT_FIELD = re.compile(
    r"\b(detect\s+field|field[_\s]label|field\s+[A-I])\b", re.IGNORECASE
)
_RE_GRASP = re.compile(
    r"\b(pick\w*|grab\w*|grasp\w*|grip(?!per)\w*|hold\w*)\b", re.IGNORECASE
)
_RE_PLACE = re.compile(r"\b(place|deposit|put\s+down|drop)\b", re.IGNORECASE)
_RE_ROBOT1 = re.compile(r"\bRobot1\b")
_RE_ROBOT2 = re.compile(r"\bRobot2\b")


class PromptBuilder:
    """Assembles LLM parsing prompts; separated so prompt logic can be unit-tested without a live LLM."""

    def __init__(self, registry, workflow_registry, rag):
        self.registry = registry
        self.workflow_registry = workflow_registry
        self.rag = rag

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        command_text: str,
        robot_id: str,
        anti_pattern_section: str = "",
        spatial_section: str = "",
        hint: str = "",
    ) -> str:
        tags = self._classify_command(command_text)
        available_ops = self.get_available_operations_summary(command_text)

        _named_robots = {
            rid
            for rid in ["Robot1", "Robot2"]
            if re.search(r"\b" + rid + r"\b", command_text)
        }
        _robot_id_line = (
            f'Default robot_id: "{robot_id}"'
            if len(_named_robots) <= 1
            else f'Robots in task: {", ".join(sorted(_named_robots))} - assign robot_id per-op as named in the task'
        )

        sections: List[str] = [
            f"Available Operations: {available_ops}",
            f'Command to parse: "{command_text}"\n        {_robot_id_line}',
        ]

        if spatial_section:
            sections.append(
                f"{spatial_section}\n        IMPORTANT: The object IDs listed above are the EXACT identifiers you MUST use in operation params (e.g. object_id, color). Do NOT invent, shorten, or paraphrase object names. Use ONLY operations from Available Operations above."
            )

        sections += [
            _SECTION_WORKSPACE,
            _SECTION_ROBOT_ASSIGNMENT,
        ]

        if "multi_robot" in tags:
            sections.append(_SECTION_MULTI_ROBOT)

        if "handoff" in tags:
            sections.append(_SECTION_HANDOFF)

        if tags & {"sync", "handoff", "multi_robot"}:
            sections.append(_SECTION_SYNC)

        sections.append(_SECTION_NAVIGATION)

        if "grasp" in tags:
            sections.append(_SECTION_GRASP)

        if "place" in tags:
            sections.append(_SECTION_PLACE)

        if "between" in tags:
            sections.append(_SECTION_BETWEEN)

        if "multi_robot" in tags and "handoff" not in tags:
            sections.append(_SECTION_INDEPENDENT_PARALLEL)

        if "cooperative" in tags:
            sections.append(_SECTION_COOPERATIVE)

        if "dual_grasp" in tags:
            sections.append(_SECTION_DUAL_GRASP)

        sections.append(_SECTION_SINGLE_ROBOT)
        sections.append(_SECTION_VARIABLE_PASSING)

        if "detect_field" in tags or "place" in tags:
            sections.append(_SECTION_DETECT_FIELD)

        if anti_pattern_section:
            sections.append(anti_pattern_section)
        if hint:
            sections.append(f"=== REFLECTION ===\n        {hint}")

        if hint:
            sections.append(
                "Output exactly ONE valid JSON command object (not an array). No explanation."
            )
        else:
            sections.append("Output only valid JSON, no explanation, no comments.")

        return "\n\n        ".join(sections)

    # ── Intent classification ─────────────────────────────────────────────────

    def _classify_command(self, command_text: str) -> FrozenSet[str]:
        """Return a frozenset of intent tags detected in command_text."""
        tags = set()
        if _RE_HANDOFF.search(command_text):
            tags.add("handoff")
        if _RE_BETWEEN.search(command_text):
            tags.add("between")
        if _RE_COOPERATIVE.search(command_text):
            tags.add("cooperative")
        if _RE_DUAL_GRASP.search(command_text):
            tags.add("dual_grasp")
        if _RE_SYNC.search(command_text):
            tags.add("sync")
        if _RE_DETECT_FIELD.search(command_text):
            tags.add("detect_field")
        if _RE_GRASP.search(command_text):
            tags.add("grasp")
        if _RE_PLACE.search(command_text):
            tags.add("place")
        if _RE_ROBOT1.search(command_text) and _RE_ROBOT2.search(command_text):
            tags.add("multi_robot")
        return frozenset(tags)

    # ── Operations summary ────────────────────────────────────────────────────

    def get_available_operations_summary(self, command_text: str = "") -> str:
        """Return ops relevant to command_text (via RAG if available, else full list)."""
        if self.rag and command_text:
            try:
                rag_results = self.rag.search(command_text, top_k=10)
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
                        pattern_name = result.get("metadata", {}).get("name", "")
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
                        op = self.registry.get_operation_by_name(
                            result.get("metadata", {}).get("name", "")
                        )
                        if op:
                            relevant_ops.add(op.name)
                            params = self.format_parameters(op.parameters)
                            score = result.get("score", 0)
                            summary_lines.append(
                                f"- {op.name}({params}): {op.description} [relevance: {score:.2f}]"
                            )
                            ops_added += 1

                if ops_added == 0:
                    # No RAG op hits or all lookups failed - list all ops as fallback
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
