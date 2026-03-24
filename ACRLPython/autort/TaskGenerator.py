#!/usr/bin/env python3
"""
AutoRT Task Generator

LLM-based task generation with Pydantic validation and manual retry loop.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from pydantic import ValidationError
from openai import OpenAI

from autort.DataModels import ProposedTask, SceneDescription
from operations.Registry import get_global_registry
from config.Servers import LLM_THINKING_BUDGET, LLM_THINKING_ENABLED, SYSTEM_PROMPT_BASE
from config.Negotiation import USE_STRUCTURED_OUTPUT
from config.Vision import DEFAULT_CAMERA_ID

logger = logging.getLogger(__name__)


class TaskGenerator:
    """Generates task proposals using LLM with robust JSON parsing"""

    def __init__(self, config):
        """
        Initialize TaskGenerator.

        Args:
            config: AutoRT config module with LLM settings
        """
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
        """
        Generate task proposals in parallel — one LLM request per task.

        Sends ``num_tasks`` concurrent requests to the LLM, each asking for a
        single task.  Results are merged and deduplicated by task_id.  Any
        request that fails all retries is silently skipped so that partial
        results are still returned.

        Args:
            scene: Scene description with detected objects
            robot_ids: List of robot IDs to use
            num_tasks: Number of tasks to generate
            include_collaborative: Whether to generate collaborative tasks (None = use config default)
        """
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
                        # Deduplicate by task_id in case the LLM reuses IDs
                        if task.task_id not in seen_ids:
                            seen_ids.add(task.task_id)
                            validated_tasks.append(task)
                        else:
                            # Rename duplicate so it is not silently dropped
                            task.task_id = f"{task.task_id}_{idx}"
                            seen_ids.add(task.task_id)
                            validated_tasks.append(task)
                except Exception as e:
                    logger.warning(f"Task slot {idx} raised unexpected error: {e}")

        logger.info(f"Parallel generation complete: {len(validated_tasks)}/{num_tasks} tasks succeeded")
        return validated_tasks

    def _generate_single_task(self, prompt: str, slot_index: int) -> Optional[ProposedTask]:
        """
        Generate and validate a single task with retries.

        Called concurrently by ``generate_tasks`` — each invocation runs in its
        own thread and retries independently on JSON/validation errors.

        Args:
            prompt: Base task-generation prompt (built for 1 task)
            slot_index: Index used for logging to distinguish parallel workers

        Returns:
            A validated ProposedTask, or None if all retries failed
        """
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
                        logger.debug(f"Task slot {slot_index}: generated '{task.task_id}'")
                        return task
                    last_error = f"Parameter validation failed: {error_msg}"
                    logger.warning(f"Task slot {slot_index} attempt {attempt + 1}: {last_error}")

                if not tasks:
                    last_error = "No tasks generated"
                    logger.warning(f"Task slot {slot_index} attempt {attempt + 1}: {last_error}")

            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                last_error = f"JSON/Schema error: {e}"
                logger.warning(f"Task slot {slot_index} attempt {attempt + 1}: {last_error}")

            if attempt < self.max_retries - 1:
                time.sleep(1)

        logger.error(f"Task slot {slot_index} failed after {self.max_retries} retries. Last error: {last_error}")
        return None

    def _query_llm(self, prompt: str) -> str:
        """Query LM Studio via OpenAI-compatible API"""
        try:
            # Add system message to suppress reasoning for reasoning models
            # Type ignore needed for LM Studio compatibility - OpenAI SDK expects TypedDict but accepts plain dicts
            messages = [
                {
                    "role": "system",
                    "content": (
                        SYSTEM_PROMPT_BASE
                        + " You are an autonomous task planner. Generate grounded, executable "
                        "task plans using ONLY the objects and operations listed in the user "
                        "message. Never invent object IDs, operation names, camera IDs, or "
                        "coordinates outside the specified workspace bounds. "
                        "Return a JSON array directly — no preamble, no [THINK] tags."
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            create_kwargs: dict = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                # Single task JSON — 2048 tokens covers verbose multi-step tasks.
                "max_tokens": 2048,
            }
            # Structured output forces valid JSON at the inference layer.
            # Set USE_STRUCTURED_OUTPUT=false for models that don't support response_format.
            if USE_STRUCTURED_OUTPUT:
                create_kwargs["response_format"] = {"type": "json_object"}
            # `thinking` is a LM Studio extension; pass via extra_body so the openai SDK
            # forwards it as-is without treating it as an unknown named parameter.
            if LLM_THINKING_ENABLED:
                create_kwargs["extra_body"] = {"thinking": {"type": "enabled", "budget_tokens": LLM_THINKING_BUDGET}}
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
        """Build prompt for LLM task generation"""
        # Build object list with spatial hints (which robot is closer)
        objects_lines = []
        for obj in scene.objects:
            x_pos = obj.position[0]
            # Determine proximity hint based on X coordinate
            proximity_hint = ""
            if x_pos < -0.1:
                proximity_hint = " [closer to Robot1/left]"
            elif x_pos > 0.1:
                proximity_hint = " [closer to Robot2/right]"
            else:
                proximity_hint = " [center workspace]"

            objects_lines.append(
                f"- {obj.color} object at ({obj.position[0]:.3f}, {obj.position[1]:.3f}, {obj.position[2]:.3f}) "
                f"(graspable={obj.graspable}, confidence={obj.confidence:.2f}){proximity_hint}"
            )
        objects_str = "\n".join(objects_lines)

        operations_str = self._get_operations_summary()

        # Build robot spatial layout information
        robot_layout = self._build_robot_layout_description(robot_ids)

        collaborative_hint = ""
        if include_collaborative and len(robot_ids) > 1:
            collaborative_hint = f"""
MULTI-ROBOT COORDINATION:
You have {len(robot_ids)} robots: {robot_ids}
{robot_layout}

Collaborative patterns:
1. Handoff: Robot1 picks object, moves to handoff zone, Robot2 receives
2. Parallel: Both robots pick different objects simultaneously
3. Sequential: Robot1 places object, Robot2 stacks on top

Use 'signal' and 'wait_for_signal' for coordination between robots.
"""

        # Add spatial hints for single-robot tasks
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
            # Incomplete reasoning block — drop it entirely, use fallback
            summary = f"Detected {len(scene.objects)} objects in workspace."
        # Hard cap as final safety net (~200 tokens)
        if len(summary) > 800:
            summary = summary[:800] + "..."

        return f"""SCENE ANALYSIS:
{summary}

DETECTED OBJECTS:
{objects_str if objects_str else "No objects detected."}

AVAILABLE ROBOTS:
{robot_ids}{spatial_hints}

WORKSPACE LAYOUT:
- X axis: -0.6 (left) to +0.6 (right)
- Y axis: 0.0 (table surface) to 0.6 (above table)
- Z axis: -0.6 (back) to +0.6 (front)
- Center workspace: X near 0.0
- Left workspace: X negative (Robot1's area)
- Right workspace: X positive (Robot2's area)

AVAILABLE CAMERAS:
- {DEFAULT_CAMERA_ID}: Stereo camera for depth perception and object detection
- MainCamera: Main camera for scene analysis

AVAILABLE OPERATIONS:
{operations_str}

{collaborative_hint}

TASK:
Generate {num_tasks} diverse robotic tasks in JSON format.

OUTPUT FORMAT (strict JSON array):
[
  {{
    "task_id": "task_001",
    "description": "Pick up red cube from workspace",
    "operations": [
      {{"type": "detect_object_stereo", "robot_id": "Robot1", "parameters": {{"color": "red", "selection": "closest", "camera_id": "{DEFAULT_CAMERA_ID}"}}}},
      {{"type": "grasp_object", "robot_id": "Robot1", "parameters": {{"object_id": "red_cube"}}}}
    ],
    "required_robots": ["Robot1"],
    "estimated_complexity": 4,
    "reasoning": "Detect red objects, select closest, then grasp it"
  }},
  {{
    "task_id": "task_002",
    "description": "Move robot to central position",
    "operations": [
      {{"type": "move_to_coordinate", "robot_id": "Robot1", "parameters": {{"x": 0.0, "y": 0.2, "z": 0.0}}}}
    ],
    "required_robots": ["Robot1"],
    "estimated_complexity": 2,
    "reasoning": "Simple navigation to workspace center"
  }},
  {{
    "task_id": "task_003",
    "description": "Detect all objects and analyze scene",
    "operations": [
      {{"type": "detect_object_stereo", "robot_id": "Robot1", "parameters": {{"color": null, "selection": "all", "camera_id": "{DEFAULT_CAMERA_ID}"}}}}
    ],
    "required_robots": ["Robot1"],
    "estimated_complexity": 2,
    "reasoning": "Detection only, no filtering - returns all detected objects"
  }}
]

CRITICAL PARAMETER RULES:
1. detect_object_stereo parameters:
   - color: Must be "red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta", or null (for all colors)
   - selection: MUST be "left", "right", "closest", "first", or "all" (NOT object names!)
   - camera_id: Must be "{DEFAULT_CAMERA_ID}"
   - Example: {{"color": "red", "selection": "closest"}} finds the closest red object
   - Example: {{"color": null, "selection": "all"}} finds all objects regardless of color

2. grasp_object parameters:
   - object_id: Full object name from DETECTED OBJECTS (e.g., "red_cube", "blue_cube", "field_a")
   - approach: Optional, one of "top", "front", "side" (default: "top")

3. move_to_coordinate parameters:
   - x, y, z: Numeric coordinates in workspace bounds
   - NOT "target_position" - use separate x, y, z parameters

4. Camera IDs: ONLY "{DEFAULT_CAMERA_ID}" or "MainCamera" - no other camera IDs exist

GENERAL RULES:
5. ONLY use objects from DETECTED OBJECTS list above
6. ONLY use operations from AVAILABLE OPERATIONS list above
7. ROBOT ASSIGNMENT EFFICIENCY: Assign objects to their nearest robot based on X coordinate
   - Objects at X < -0.1 should be handled by Robot1 (left robot)
   - Objects at X > 0.1 should be handled by Robot2 (right robot)
   - Objects at X near 0.0 can be handled by either robot
8. Complexity: 1 (trivial) to 10 (very complex)
9. Return ONLY valid JSON, no markdown formatting, no reasoning, no [THINK] tags
10. CRITICAL: Every operation MUST have a valid "robot_id" field (never null/None)
11. Every robot_id in operations must appear in required_robots
12. CRITICAL: Each operation MUST have a "parameters" field (use empty dict {{{{}}}} if no parameters needed)
13. Only use parameter names and values shown in AVAILABLE OPERATIONS schemas
14. Pay close attention to valid_values constraints in parameter schemas - violating these will cause operation failures

COORDINATE GUIDELINES (ROS base_link frame — robot-local, Z-up):
- X: forward from robot base, range -0.5 to 0.5
- Y: left from robot base, range -0.5 to 0.5
- Z: height above robot base, range 0.0 to 0.6
- Typical reachable positions: x in [-0.4, 0.4], y in [-0.3, 0.3], z in [0.05, 0.5]

COMMON TASK PATTERNS:
1. Detection + Grasp:
   - detect_object_stereo(color="red", selection="closest") → returns position
   - grasp_object(object_id="red_cube")

2. Navigation + Detection:
   - move_to_coordinate(x=0.2, y=0.0, z=0.3)
   - detect_object_stereo(color=null, selection="all")

3. Multi-step manipulation:
   - detect_object_stereo(color="blue", selection="first")
   - grasp_object(object_id="blue_cube")
   - move_to_coordinate(x=0.0, y=0.2, z=0.3)
   - release_object()

IMPORTANT: Output the JSON array immediately without any preamble, reasoning, or explanatory text.
IMPORTANT: Output compact JSON with no extra whitespace or indentation to minimize token usage.

Generate tasks now:
"""

    def _parse_llm_response(self, raw_response: str) -> List[ProposedTask]:
        """
        Parse LLM JSON response with Pydantic validation.

        Raises:
            json.JSONDecodeError: If response is not valid JSON
            ValidationError: If JSON doesn't match ProposedTask schema
        """
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
            logger.error(f"JSON parsing failed. Response preview: {raw_response[:200]}")
            raise

        # Unwrap {"tasks": [...]} envelope that some LLMs produce
        if isinstance(data, dict) and "tasks" in data:
            logger.debug("Unwrapping {'tasks': [...]} envelope from LLM response")
            data = data["tasks"]

        # Fix missing robot_ids before validation
        if isinstance(data, list):
            fixed_data = [self._fix_missing_robot_ids(task) for task in data]
            return [ProposedTask(**task) for task in fixed_data]
        elif isinstance(data, dict):
            fixed_task = self._fix_missing_robot_ids(data)
            return [ProposedTask(**fixed_task)]
        else:
            raise ValueError(f"Unexpected response type: {type(data)}")

    def _fix_missing_robot_ids(self, task_dict: dict) -> dict:
        """
        Fix operations with missing robot_ids by inferring from context.

        Args:
            task_dict: Raw task dictionary from LLM

        Returns:
            Fixed task dictionary
        """
        required_robots = task_dict.get("required_robots", [])
        operations = task_dict.get("operations", [])

        # Track last valid robot_id to use for inference
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
        """
        Validate that all operations exist in Registry and have valid parameters.

        Returns:
            True if all operations are valid, False otherwise
        """
        is_valid, _ = self._validate_operations_with_feedback(task)
        return is_valid

    def _validate_operations_with_feedback(
        self, task: ProposedTask
    ) -> tuple[bool, str]:
        """
        Validate operations and return detailed feedback for LLM retry.

        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
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

            return True, ""
        except Exception as e:
            return False, f"Validation exception: {str(e)}"

    def _validate_operation_parameters_with_feedback(self, operation, op_def) -> str:
        """
        Validate operation parameters and return detailed error message.

        Args:
            operation: Operation instance from ProposedTask
            op_def: BasicOperation definition from Registry

        Returns:
            Error message string if invalid, empty string if valid
        """
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
        """
        Get token-efficient operation list from the live Registry with parameter schemas.

        Queries the Registry for all operations and formats them with:
        - Parameter names and types
        - Valid values (enums)
        - Default values
        - Required vs optional distinction

        Cached after first call since operations don't change at runtime.
        """
        if self._operations_summary_cache is not None:
            return self._operations_summary_cache

        operations = self.registry.get_all_operations()
        lines = []
        for op in operations:
            # Build detailed parameter specifications
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
        """
        Build description of robot physical layout based on robot IDs.

        Uses ROBOT_SPATIAL_LAYOUT from config for detailed position information.

        Args:
            robot_ids: List of robot IDs (e.g., ["Robot1", "Robot2"])

        Returns:
            Formatted description of robot positions
        """
        if not robot_ids:
            return ""

        # Import spatial layout from config
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

    # Collaborative task templates for prompt enrichment
    COLLABORATIVE_TEMPLATES = {
        "handoff": "{robot1} picks {object}, moves to handoff zone, {robot2} receives",
        "parallel_pick": "{robot1} and {robot2} pick {object1} and {object2} simultaneously",
        "sequential_stack": "{robot1} places {object1}, {robot2} stacks {object2} on top",
    }
