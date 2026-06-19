#!/usr/bin/env python3
"""LLM-based task generation with Pydantic validation and retry loop."""

import json
import logging
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from pydantic import ValidationError
from openai import OpenAI

from autort.DataModels import ProposedTask, SceneDescription, ExecutedTaskContext
from operations.Registry import get_global_registry
from config.Servers import LLM_THINKING_BUDGET, LLM_THINKING_ENABLED, SYSTEM_PROMPT_BASE
from config.Negotiation import USE_STRUCTURED_OUTPUT
from config.Vision import DEFAULT_CAMERA_ID

logger = logging.getLogger(__name__)


class TaskGenerator:
    """Generates and validates task proposals via LLM."""

    def __init__(self, config):
        self.config = config
        self.llm_client = OpenAI(base_url=config.LM_STUDIO_URL, api_key="not-needed")
        self.model = config.TASK_GENERATION_MODEL
        self.max_retries = config.MAX_JSON_RETRIES
        self.temperature = getattr(config, "TASK_GENERATION_TEMPERATURE", 0.7)
        self.registry = get_global_registry()

        # Cache operations summary (build once, reuse)
        self._operations_summary_cache = None

    def generate_tasks(
        self,
        scene: SceneDescription,
        robot_ids: List[str] = ["Robot1", "Robot2"],
        num_tasks: int = 5,
        include_collaborative: Optional[bool] = None,
    ) -> List[ProposedTask]:
        """Fire num_tasks LLM requests in parallel; deduplicate by task_id."""
        if include_collaborative is None:
            try:
                from config.AutoRT import ENABLE_COLLABORATIVE_TASKS
            except ImportError:
                from ..config.AutoRT import ENABLE_COLLABORATIVE_TASKS
            include_collaborative = ENABLE_COLLABORATIVE_TASKS

        prompt = self._build_task_prompt(scene, robot_ids, 1, include_collaborative)

        logger.info(f"Generating {num_tasks} tasks in parallel...")

        validated_tasks: List[ProposedTask] = []
        seen_ids: set = set()

        with ThreadPoolExecutor(max_workers=num_tasks) as executor:
            futures = {
                executor.submit(self._generate_single_task, prompt, idx): idx
                for idx in range(num_tasks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    task = future.result()
                    if task is not None:
                        # Deduplicate by task_id - drop tasks with the same ID
                        if task.task_id not in seen_ids:
                            seen_ids.add(task.task_id)
                            validated_tasks.append(task)
                except Exception as e:
                    logger.warning(f"Task slot {idx} raised unexpected error: {e}")

        logger.info(
            f"Parallel generation complete: {len(validated_tasks)}/{num_tasks} tasks succeeded"
        )
        return validated_tasks

    def _generate_single_task(
        self, prompt: str, slot_index: int
    ) -> Optional[ProposedTask]:
        last_error: Optional[str] = None
        current_prompt = prompt

        for attempt in range(self.max_retries):
            try:
                if attempt > 0 and last_error:
                    current_prompt = prompt + f"""

PREVIOUS ATTEMPT HAD ERRORS (attempt {attempt}):
{last_error}

Please fix these issues and generate a valid task following the parameter schemas exactly.
"""

                raw_response = self._query_llm(current_prompt)
                tasks = self._parse_llm_response(raw_response)

                # _parse_llm_response may return a list; take the first valid task
                for task in tasks:
                    is_valid, error_msg = self._validate_operations_with_feedback(task)
                    if is_valid:
                        logger.debug(
                            f"Task slot {slot_index}: generated '{task.task_id}'"
                        )
                        return task
                    last_error = f"Parameter validation failed: {error_msg}"
                    logger.warning(
                        f"[AutoRT slot {slot_index} attempt {attempt + 1}] REJECTED - {error_msg}"
                    )

                if not tasks:
                    last_error = "No tasks generated"
                    logger.warning(
                        f"[AutoRT slot {slot_index} attempt {attempt + 1}] EMPTY - LLM returned no parseable tasks"
                    )

            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                last_error = f"JSON/Schema error: {e}"
                logger.warning(
                    f"[AutoRT slot {slot_index} attempt {attempt + 1}] PARSE ERROR - {e}"
                )
                if hasattr(e, "doc"):
                    # JSONDecodeError - show chars around the failure point
                    pos = getattr(e, "pos", 0)
                    snippet = e.doc[max(0, pos - 80) : pos + 80]  # type: ignore[union-attr]
                    logger.info(f"  ↳ context around char {pos}: ...{snippet!r}...")

            if attempt < self.max_retries - 1:
                time.sleep(1)

        logger.error(
            f"Task slot {slot_index} failed after {self.max_retries} retries. Last error: {last_error}"
        )
        return None

    def _query_llm(self, prompt: str) -> str:
        try:
            # Type ignore needed for LM Studio compat - OpenAI SDK expects TypedDict but accepts plain dicts
            messages = [
                {
                    "role": "system",
                    "content": (
                        SYSTEM_PROMPT_BASE
                        + " You are an autonomous task planner. Generate grounded, executable "
                        "task plans using ONLY the objects and operations listed in the user message."
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            create_kwargs: dict = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                # max_tokens covers thinking + output as a shared pool in LM Studio.
                # 4096 for JSON output + full thinking budget headroom.
                "max_tokens": 4096
                + (LLM_THINKING_BUDGET if LLM_THINKING_ENABLED else 0),
            }
            # Structured output forces valid JSON at the inference layer.
            # Set USE_STRUCTURED_OUTPUT=false for models that don't support response_format.
            if USE_STRUCTURED_OUTPUT:
                create_kwargs["response_format"] = {"type": "json_object"}
            if LLM_THINKING_ENABLED:
                create_kwargs["extra_body"] = {
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": LLM_THINKING_BUDGET,
                    }
                }
            response = self.llm_client.chat.completions.create(**create_kwargs)  # type: ignore[arg-type]
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("LLM returned empty response")
            return content
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            logger.error(f"LM Studio URL: {self.config.LM_STUDIO_URL}")
            logger.error(f"Model: {self.model}")
            raise ValueError(f"Failed to query LLM: {e}") from e

    def _build_task_prompt(
        self,
        scene: SceneDescription,
        robot_ids: List[str],
        num_tasks: int,
        include_collaborative: bool,
    ) -> str:
        from config.Robot import ROBOT_BASE_POSITIONS, MAX_ROBOT_REACH

        objects_lines = []
        # Map object_id → set of robot_ids that can reach it (for validation later)
        self._reachability_map: dict[str, set[str]] = {}
        self._object_color_map: dict[str, str] = {
            obj.object_id: obj.color for obj in scene.objects
        }

        for obj in scene.objects:
            reachable_robots = []
            for rid in robot_ids:
                base = ROBOT_BASE_POSITIONS.get(rid, (0.0, 0.0, 0.0))
                dist = math.sqrt(
                    (obj.position[0] - base[0]) ** 2
                    + (obj.position[1] - base[1]) ** 2
                    + (obj.position[2] - base[2]) ** 2
                )
                if dist <= MAX_ROBOT_REACH:
                    reachable_robots.append(rid)

            self._reachability_map[obj.object_id] = set(reachable_robots)

            if len(reachable_robots) == 0:
                reach_hint = " [UNREACHABLE by any robot]"
            elif len(reachable_robots) == len(robot_ids):
                reach_hint = " [reachable by all robots]"
            else:
                reach_hint = f" [REACHABLE: {'/'.join(reachable_robots)} only]"

            objects_lines.append(
                f"- {obj.color} object (id={obj.object_id}) at "
                f"({obj.position[0]:.3f}, {obj.position[1]:.3f}, {obj.position[2]:.3f}) "
                f"(graspable={obj.graspable}, confidence={obj.confidence:.2f}){reach_hint}"
            )
        objects_str = "\n".join(objects_lines)

        operations_str = self._get_operations_summary()

        robot_layout = self._build_robot_layout_description(robot_ids)

        collaborative_hint = ""
        if include_collaborative and len(robot_ids) > 1:
            collaborative_hint = f"""
MULTI-ROBOT COORDINATION: You have {len(robot_ids)} robots: {robot_ids}
{robot_layout}

Collaborative patterns:
1. Handoff: Robot1 picks object, moves to handoff zone, Robot2 receives
2. Parallel: Both robots pick different objects simultaneously
3. Sequential: Robot1 places object, Robot2 stacks on top

Use 'signal' and 'wait_for_signal' for coordination between robots.
"""

        spatial_hints = ""
        if not include_collaborative or len(robot_ids) == 1:
            spatial_hints = f"\n{robot_layout}\n" if robot_layout else ""

        # Strip [THINK]...[/THINK] reasoning traces before injecting into the prompt.
        # The VLM may produce thousands of tokens of reasoning that are useless here
        # and quickly exhaust the 8192-token context window.
        summary = scene.scene_summary or "No VLM analysis available."
        if "[/THINK]" in summary:
            summary = summary.split("[/THINK]", 1)[1].strip()
        elif "[THINK]" in summary:
            # Incomplete reasoning block - drop it entirely, use fallback
            summary = f"Detected {len(scene.objects)} objects in workspace."
        # Hard cap as final safety net (~200 tokens)
        if len(summary) > 800:
            summary = summary[:800] + "..."

        previous_task_section = ""
        if scene.last_task_context is not None:
            previous_task_section = self._build_previous_task_section(
                scene.last_task_context
            )

        return f"""SCENE ANALYSIS:
{summary}

DETECTED OBJECTS:
{objects_str if objects_str else "No objects detected."}

AVAILABLE ROBOTS:
{robot_ids}{spatial_hints}

AVAILABLE OPERATIONS:
{operations_str}

{collaborative_hint}

{previous_task_section}
TASK: Generate {num_tasks} diverse robotic tasks as a JSON array. Each task:
{{
  "task_id": "task_001",
  "description": "one sentence",
  "operations": [{{"type": "op_name", "robot_id": "RobotN", "parameters": {{}}}}],
  "required_robots": ["RobotN"],
  "estimated_complexity": 1-10,
  "reasoning": "one sentence"
}}

Rules:
Only use objects from DETECTED OBJECTS and operations from AVAILABLE OPERATIONS
Every operation needs robot_id and parameters ({{}} if none)
detect_object_stereo: color must be a named color or null; selection must be "left"/"right"/"closest"/"first"/"all"; camera_id="{DEFAULT_CAMERA_ID}"
Assign objects to nearest robot (X<-0.1: Robot1, X>0.1: Robot2, center: either)
Every robot_id in operations must appear in required_robots
GRASP RULE: Always use grasp_object (not pick_object_at_coordinate) when grasping a detected object: grasp_object uses the object's name/ID and handles detection internally
HANDOFF RULE: For transferring an object between robots use handoff (sender) + receive_handoff (receiver): never implement handoffs with raw move/signal/release sequences. Handoffs are ONLY allowed with the red object: never generate a handoff for any other color.
PLACE RULE: For placing a held object use place_object (at a position) or place_between_objects (between two reference objects): never use release_object or move_to_coordinate to place
FIELD RULE: Fields (field_a through field_i) are NOT in WorldState by default. Before using a field as a placement target (on_top_of param) you MUST call detect_field earlier in the same task's operations. Never reference a field ID unless detect_field for that field precedes it in the sequence.
Output compact JSON array only, no markdown"""

    def _parse_llm_response(self, raw_response: str) -> List[ProposedTask]:
        # Strip reasoning tokens from models like Mistral Reasoning
        if "[THINK]" in raw_response and "[/THINK]" in raw_response:
            # Extract content after [/THINK] tag
            parts = raw_response.split("[/THINK]")
            if len(parts) > 1:
                raw_response = parts[1].strip()
                logger.debug("Stripped [THINK] reasoning block from response")
        elif "[THINK]" in raw_response:
            # Model started thinking but didn't finish - response is incomplete
            logger.error(
                "Model output contains incomplete [THINK] block (no closing tag)"
            )
            raise ValueError(
                "Model response incomplete - reasoning block not closed. Response may have been truncated."
            )

        # Strip markdown code blocks if present
        if "```json" in raw_response:
            raw_response = raw_response.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_response:
            raw_response = raw_response.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError:
            # Detect truncated responses (reasoning models hitting token limit mid-JSON)
            stripped = raw_response.strip()
            if stripped and stripped[-1] not in ("}", "]"):
                logger.error(
                    f"JSON truncated - response cut off mid-stream (last char={stripped[-1]!r}). "
                    f"Increase max_tokens or reduce thinking budget. Preview: {raw_response[:200]}"
                )
                raise ValueError(
                    "LLM response truncated before JSON was complete"
                ) from None
            logger.error(f"JSON parsing failed. Response preview: {raw_response[:200]}")
            raise

        # Unwrap {"tasks": [...]} envelope that some LLMs produce
        if isinstance(data, dict) and "tasks" in data:
            logger.debug("Unwrapping {'tasks': [...]} envelope from LLM response")
            data = data["tasks"]

        # Fix missing robot_ids before validation
        if isinstance(data, list):
            fixed_data = [
                self._fix_missing_robot_ids(self._flatten_operations(task))
                for task in data
            ]
            return [ProposedTask(**task) for task in fixed_data]
        elif isinstance(data, dict):
            fixed_task = self._fix_missing_robot_ids(self._flatten_operations(data))
            return [ProposedTask(**fixed_task)]
        else:
            raise ValueError(f"Unexpected response type: {type(data)}")

    def _ensure_unique_id(self, task_dict: dict) -> dict:
        original = task_dict.get("task_id", "task")
        suffix = uuid.uuid4().hex[:6]
        task_dict = dict(task_dict)
        task_dict["task_id"] = f"{original}_{suffix}"
        return task_dict

    def _flatten_operations(self, task_dict: dict) -> dict:
        """Flatten parallel_group wrapper objects some LLMs emit instead of a flat ops list."""
        raw_ops = task_dict.get("operations", [])
        if not raw_ops:
            return task_dict

        flat: list[dict] = []
        for item in raw_ops:
            # Format A: wrapper with nested operations list
            if "operations" in item and "type" not in item:
                nested = item.get("operations", [])
                logger.debug(
                    f"Flattening parallel_group wrapper (group={item.get('parallel_group')}) "
                    f"with {len(nested)} nested ops"
                )
                flat.extend(nested)
            # Format B: type field is "parallel_group" - degenerate, skip
            elif item.get("type") == "parallel_group":
                logger.warning(
                    "Dropping degenerate operation with type='parallel_group'; LLM confused group wrapper with op type"
                )
            else:
                flat.append(item)

        task_dict = dict(task_dict)
        task_dict["operations"] = flat
        return task_dict

    def _fix_missing_robot_ids(self, task_dict: dict) -> dict:
        required_robots = task_dict.get("required_robots", [])
        operations = task_dict.get("operations", [])

        last_robot_id = required_robots[0] if required_robots else "Robot1"

        for op in operations:
            if op.get("robot_id") is None or op.get("robot_id") == "":
                # Infer robot_id from context
                # For coordination ops (signal, wait), use the first required robot
                op_type = op.get("type", "")
                if op_type in ["signal", "wait_for_signal", "wait"]:
                    # Use first available robot for coordination
                    op["robot_id"] = required_robots[0] if required_robots else "Robot1"
                    logger.debug(
                        f"Fixed missing robot_id for {op_type}: {op['robot_id']}"
                    )
                else:
                    # Use last valid robot_id for sequential operations
                    op["robot_id"] = last_robot_id
                    logger.debug(
                        f"Fixed missing robot_id for {op_type}: {op['robot_id']}"
                    )
            else:
                # Update last valid robot_id
                last_robot_id = op["robot_id"]

        return task_dict

    def _validate_operations(self, task: ProposedTask) -> bool:
        is_valid, _ = self._validate_operations_with_feedback(task)
        return is_valid

    def _validate_operations_with_feedback(
        self, task: ProposedTask
    ) -> tuple[bool, str]:
        try:
            for i, op in enumerate(task.operations, 1):
                # Check operation exists
                op_def = self.registry.get_operation_by_name(op.type)
                if op_def is None:
                    return (
                        False,
                        f"Operation #{i} '{op.type}' does not exist in Registry",
                    )

                # Validate parameters against operation definition
                param_errors = self._validate_operation_parameters_with_feedback(
                    op, op_def
                )
                if param_errors:
                    return False, f"Operation #{i} '{op.type}': {param_errors}"

                # Reachability check: if this op targets a detected object, verify
                # the assigned robot can physically reach it.
                reachability_map = getattr(self, "_reachability_map", {})
                object_color_map = getattr(self, "_object_color_map", {})
                object_id = op.parameters.get("object_id")

                if reachability_map and object_id and object_id in reachability_map:
                    allowed = reachability_map[object_id]
                    if op.robot_id not in allowed:
                        allowed_str = "/".join(sorted(allowed)) if allowed else "none"
                        return (
                            False,
                            f"Operation #{i} '{op.type}': {op.robot_id} cannot reach "
                            f"object '{object_id}' - assign to {allowed_str} instead",
                        )

                # Handoff color constraint: handoff/receive_handoff only allowed for red objects.
                if op.type in {"handoff", "receive_handoff"} and object_id:
                    color = object_color_map.get(object_id, "")
                    if color and color.lower() != "red":
                        return (
                            False,
                            f"Operation #{i} '{op.type}': handoffs are only allowed with "
                            f"the red object, but object '{object_id}' is '{color}' - "
                            f"use grasp_object + place_object for non-red objects",
                        )

                # Field dependency check: on_top_of referencing a field requires
                # detect_field to appear earlier in the same task.
                on_top_of = op.parameters.get("on_top_of", "")
                if (
                    on_top_of
                    and isinstance(on_top_of, str)
                    and on_top_of.startswith("field_")
                ):
                    detected_fields = {
                        prev.parameters.get("field_label", "").upper()
                        for prev in task.operations[: i - 1]
                        if prev.type == "detect_field"
                    }
                    field_letter = on_top_of.split("_", 1)[-1].upper()
                    if field_letter not in detected_fields:
                        return (
                            False,
                            f"Operation #{i} 'place_object' references on_top_of='{on_top_of}' "
                            f"but detect_field('{field_letter}') does not precede it - "
                            f"add detect_field for '{field_letter}' before this placement",
                        )

            return True, ""
        except Exception as e:
            return False, f"Validation exception: {str(e)}"

    def _validate_operation_parameters_with_feedback(self, operation, op_def) -> str:
        op_params = operation.parameters if operation.parameters else {}

        for param_def in op_def.parameters:
            param_name = param_def.name

            # Skip robot_id validation - it's a field on Operation model, not a parameter
            # In AutoRT, robot_id is operation.robot_id, not operation.parameters['robot_id']
            if param_name == "robot_id":
                continue

            param_value = op_params.get(param_name)

            # Check required parameters
            if param_def.required and param_value is None:
                return f"Missing required parameter '{param_name}'"

            # Skip validation if parameter not provided (and it's optional)
            if param_value is None:
                continue

            # Validate against valid_values constraint
            if (
                hasattr(param_def, "valid_values")
                and param_def.valid_values is not None
            ):
                if param_value not in param_def.valid_values:
                    valid_str = ", ".join(
                        f"'{v}'" if v is not None else "null"
                        for v in param_def.valid_values
                    )
                    return (
                        f"Parameter '{param_name}' value '{param_value}' not in valid values: {valid_str}. "
                        f"Fix: Use one of these exact values."
                    )

            # Validate against valid_range constraint
            if hasattr(param_def, "valid_range") and param_def.valid_range is not None:
                if not isinstance(param_value, (int, float)):
                    return f"Parameter '{param_name}' must be numeric (got {type(param_value).__name__})"
                min_val, max_val = param_def.valid_range
                # Allow 1mm tolerance for floating-point rounding near boundaries
                # (e.g. -0.001 is physically equivalent to 0.0 at table surface)
                tolerance = 0.001
                if not (min_val - tolerance <= param_value <= max_val + tolerance):
                    return (
                        f"Parameter '{param_name}' value {param_value} outside valid range [{min_val}, {max_val}]. "
                        f"Fix: Use a value between {min_val} and {max_val}."
                    )

        return ""  # No errors

    def _get_operations_summary(self) -> str:
        """Build a token-efficient ops list from the registry; cached after first call."""
        if self._operations_summary_cache is not None:
            return self._operations_summary_cache

        # Ops excluded from AutoRT - superseded by higher-level composite ops.
        # grasp_object > pick_object_at_coordinate (ID-based vs coord-based)
        # handoff/receive_handoff > manual move+signal+release sequences
        # place_object/place_between_objects > release_object for placement
        _AUTORT_EXCLUDED_OPS = {
            "pick_object_at_coordinate",  # use grasp_object
            "release_object",  # use place_object or place_between_objects
        }

        operations = self.registry.get_all_operations()
        lines = []
        for op in operations:
            if op.name in _AUTORT_EXCLUDED_OPS:
                continue
            param_specs = []
            for p in op.parameters:
                spec_parts = [p.name, f":{p.type}"]

                # Add valid values if constrained
                if hasattr(p, "valid_values") and p.valid_values:
                    values_str = "|".join(str(v) for v in p.valid_values)
                    spec_parts.append(f"[{values_str}]")
                elif hasattr(p, "valid_range") and p.valid_range:
                    spec_parts.append(f"[{p.valid_range[0]}-{p.valid_range[1]}]")

                # Add default if exists
                if not p.required and hasattr(p, "default") and p.default is not None:
                    spec_parts.append(f"={p.default}")

                # Mark as optional
                if not p.required:
                    param_specs.append(f"[{' '.join(spec_parts)}]")
                else:
                    param_specs.append(" ".join(spec_parts))

            param_str = ", ".join(param_specs) if param_specs else ""
            lines.append(f"- {op.name}({param_str}) - {op.description}")

        summary = "\n".join(lines)
        self._operations_summary_cache = summary
        return summary

    def _build_robot_layout_description(self, robot_ids: List[str]) -> str:
        if not robot_ids:
            return ""

        try:
            from config.AutoRT import ROBOT_SPATIAL_LAYOUT
        except ImportError:
            from ..config.AutoRT import ROBOT_SPATIAL_LAYOUT

        layout_lines = []
        for robot_id in robot_ids:
            if robot_id in ROBOT_SPATIAL_LAYOUT:
                info = ROBOT_SPATIAL_LAYOUT[robot_id]
                position = info.get("position", "workspace")
                x_range = info.get("x_range", "")
                region = info.get("workspace_region", "")

                desc_parts = [f"- {robot_id}: {position}"]
                if x_range:
                    desc_parts.append(f"  Coordinate range: {x_range}")
                if region:
                    desc_parts.append(f"  Workspace region: {region}")

                layout_lines.append("\n".join(desc_parts))
            else:
                # Fallback for unknown robots
                layout_lines.append(f"- {robot_id}: Location in workspace")

        if layout_lines:
            return "\nRobot Physical Layout:\n" + "\n".join(layout_lines)
        return ""

    _SEQUENCING_HINTS: dict = {
        "grasp_object": (
            "holding an object in its gripper",
            "place_object, move_to_coordinate, handoff, release_object, receive_handoff",
        ),
        "pick_object_at_coordinate": (
            "holding an object picked from a coordinate",
            "place_object, move_to_coordinate, handoff, release_object",
        ),
        "detect_object_stereo": (
            "has just completed a detection scan",
            "grasp_object, move_to_coordinate, pick_object_at_coordinate",
        ),
        "analyze_scene": (
            "has just performed a scene analysis",
            "detect_object_stereo, grasp_object, move_to_coordinate",
        ),
        "release_object": (
            "gripper is open and not holding anything",
            "move_to_coordinate, detect_object_stereo, grasp_object",
        ),
        "place_object": (
            "has just placed an object, gripper is open",
            "move_to_coordinate, detect_object_stereo, grasp_object",
        ),
        "move_to_coordinate": (
            "has just finished a move",
            "detect_object_stereo, grasp_object, control_gripper, move_to_coordinate",
        ),
        "control_gripper": (
            "gripper state was just changed",
            "move_to_coordinate, grasp_object, release_object",
        ),
        "handoff": (
            "just completed a handoff transfer",
            "move_to_coordinate, detect_object_stereo, grasp_object",
        ),
        "receive_handoff": (
            "just received an object from a handoff",
            "place_object, move_to_coordinate, release_object",
        ),
    }

    _FAILURE_HINTS: dict = {
        "grasp_object": "Re-detect the object before retrying grasp - position may have drifted.",
        "detect_object_stereo": "Move to a different viewpoint before re-scanning.",
    }

    def _build_previous_task_section(
        self, last_task_context: ExecutedTaskContext
    ) -> str:
        last_op = (
            last_task_context.operation_types[-1]
            if last_task_context.operation_types
            else ""
        )

        if last_task_context.success:
            state_desc, follow_ups = self._SEQUENCING_HINTS.get(
                last_op,
                (
                    "has just completed an operation",
                    "move_to_coordinate, detect_object_stereo",
                ),
            )
            return (
                f"PREVIOUS TASK CONTEXT:\n"
                f'Last task: "{last_task_context.description}" (succeeded)\n'
                f"Result: {last_task_context.result_summary}\n"
                f"Robot state: {state_desc}\n"
                f"Suggested follow-up operations: {follow_ups}\n"
                f"Generate a task that logically continues from this state.\n"
            )
        else:
            corrective = self._FAILURE_HINTS.get(
                last_op,
                "Retry the failed operation or try a different approach.",
            )
            return (
                f"PREVIOUS TASK CONTEXT:\n"
                f'Last task: "{last_task_context.description}" (FAILED)\n'
                f"Error: {last_task_context.result_summary}\n"
                f"Recovery suggestion: {corrective}\n"
                f"Generate a task that recovers from or works around this failure.\n"
            )

    # Collaborative task templates for prompt enrichment
    COLLABORATIVE_TEMPLATES = {
        "handoff": "{robot1} picks {object}, moves to handoff zone, {robot2} receives",
        "parallel_pick": "{robot1} and {robot2} pick {object1} and {object2} simultaneously",
        "sequential_stack": "{robot1} places {object1}, {robot2} stacks {object2} on top",
    }
