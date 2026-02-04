"""
AutoRT Task Generator

LLM-based task generation with Pydantic validation and manual retry loop.
"""

import json
import logging
import time
from typing import List, Optional
from pydantic import ValidationError
from openai import OpenAI

from autort.DataModels import ProposedTask, SceneDescription, Operation
from operations.Registry import get_global_registry

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
        self.llm_client = OpenAI(
            base_url=config.LM_STUDIO_URL,
            api_key="not-needed"
        )
        self.model = config.TASK_GENERATION_MODEL
        self.max_retries = config.MAX_JSON_RETRIES
        self.registry = get_global_registry()

        # Cache operations summary (build once, reuse)
        self._operations_summary_cache = None

    def generate_tasks(
        self,
        scene: SceneDescription,
        robot_ids: List[str] = ["Robot1", "Robot2"],
        num_tasks: int = 5,
        include_collaborative: bool = True
    ) -> List[ProposedTask]:
        """
        Generate task proposals with validation.

        Steps:
        1. Build prompt with scene + operations from Registry
        2. Query LLM with manual retry loop (error context preserved between retries)
        3. Validate operation types against Registry
        """
        prompt = self._build_task_prompt(scene, robot_ids, num_tasks, include_collaborative)

        # Manual retry loop — preserves error context between attempts
        tasks = []
        last_error = None
        last_response = None

        for attempt in range(self.max_retries):
            try:
                if attempt > 0 and last_error and last_response:
                    # Augment prompt with error context for retry
                    prompt += f"""

PREVIOUS RESPONSE HAD ERROR (attempt {attempt}):
{last_error}

BROKEN JSON:
{last_response[:500]}...

Please fix the JSON to match the required schema exactly.
"""

                raw_response = self._query_llm(prompt)
                last_response = raw_response
                tasks = self._parse_llm_response(raw_response)
                break  # Success

            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Task generation failed after {self.max_retries} retries")
                    return []
                time.sleep(1)  # Brief pause between retries

        # Validate operation types against Registry
        validated_tasks = []
        for task in tasks:
            if self._validate_operations(task):
                validated_tasks.append(task)
            else:
                logger.warning(f"Task '{task.task_id}' has invalid operations, skipped")

        return validated_tasks

    def _query_llm(self, prompt: str) -> str:
        """Query LM Studio via OpenAI-compatible API"""
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned empty response")
        return content

    def _build_task_prompt(
        self,
        scene: SceneDescription,
        robot_ids: List[str],
        num_tasks: int,
        include_collaborative: bool
    ) -> str:
        """Build prompt for LLM task generation"""
        objects_str = "\n".join([
            f"- {obj.color} object at ({obj.position[0]:.3f}, {obj.position[1]:.3f}, {obj.position[2]:.3f}) "
            f"(graspable={obj.graspable}, confidence={obj.confidence:.2f})"
            for obj in scene.objects
        ])

        operations_str = self._get_operations_summary()

        collaborative_hint = ""
        if include_collaborative and len(robot_ids) > 1:
            collaborative_hint = f"""
MULTI-ROBOT COORDINATION:
You have {len(robot_ids)} robots: {robot_ids}

Collaborative patterns:
1. Handoff: Robot1 picks object, moves to handoff zone, Robot2 receives
2. Parallel: Both robots pick different objects simultaneously
3. Sequential: Robot1 places object, Robot2 stacks on top

Use 'signal' and 'wait_for_signal' for coordination between robots.
"""

        return f"""You are an autonomous robot task generator.

SCENE ANALYSIS:
{scene.scene_summary if scene.scene_summary else "No VLM analysis available."}

DETECTED OBJECTS:
{objects_str if objects_str else "No objects detected."}

AVAILABLE ROBOTS:
{robot_ids}

AVAILABLE OPERATIONS:
{operations_str}

{collaborative_hint}

TASK:
Generate {num_tasks} diverse robotic tasks in JSON format.

OUTPUT FORMAT (strict JSON array):
[
  {{
    "task_id": "task_001",
    "description": "Pick up red object and place it at coordinate (0.5, 0.2, 0.1)",
    "operations": [
      {{"type": "detect_object_stereo", "robot_id": "Robot1", "parameters": {{"color": "red"}}}},
      {{"type": "move_to_coordinate", "robot_id": "Robot1", "parameters": {{"target_position": [0.4, 0.2, 0.15]}}}},
      {{"type": "control_gripper", "robot_id": "Robot1", "parameters": {{"action": "close"}}}}
    ],
    "required_robots": ["Robot1"],
    "estimated_complexity": 3,
    "reasoning": "Simple pick and place task to test basic manipulation"
  }}
]

RULES:
1. Only use objects from DETECTED OBJECTS
2. Only use operations from AVAILABLE OPERATIONS
3. Complexity: 1 (trivial) to 10 (very complex)
4. Return ONLY valid JSON, no markdown formatting
5. Every robot_id in operations must appear in required_robots

Generate tasks now:
"""

    def _parse_llm_response(self, raw_response: str) -> List[ProposedTask]:
        """
        Parse LLM JSON response with Pydantic validation.

        Raises:
            json.JSONDecodeError: If response is not valid JSON
            ValidationError: If JSON doesn't match ProposedTask schema
        """
        # Strip markdown code blocks if present
        if "```json" in raw_response:
            raw_response = raw_response.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_response:
            raw_response = raw_response.split("```")[1].split("```")[0].strip()

        data = json.loads(raw_response)

        if isinstance(data, list):
            return [ProposedTask(**task) for task in data]
        elif isinstance(data, dict):
            return [ProposedTask(**data)]
        else:
            raise ValueError(f"Unexpected response type: {type(data)}")

    def _validate_operations(self, task: ProposedTask) -> bool:
        """Validate that all operation types exist in the Registry"""
        try:
            for op in task.operations:
                if self.registry.get_operation_by_name(op.type) is None:
                    logger.warning(f"Unknown operation: {op.type}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Operation validation failed: {e}")
            return False

    def _get_operations_summary(self) -> str:
        """
        Get token-efficient operation list from the live Registry.

        Queries the Registry for all operations and formats them concisely.
        Cached after first call since operations don't change at runtime.
        """
        if self._operations_summary_cache is not None:
            return self._operations_summary_cache

        operations = self.registry.get_all_operations()
        lines = []
        for op in operations:
            # Build concise param list from operation parameters
            params = ", ".join(p.name for p in op.parameters if p.required)
            optional = [p.name for p in op.parameters if not p.required]
            param_str = params
            if optional:
                param_str += f" [optional: {', '.join(optional)}]"
            lines.append(f"- {op.name}({param_str}) - {op.description}")

        summary = "\n".join(lines)
        self._operations_summary_cache = summary
        return summary

    # Collaborative task templates for prompt enrichment
    COLLABORATIVE_TEMPLATES = {
        "handoff": "{robot1} picks {object}, moves to handoff zone, {robot2} receives",
        "parallel_pick": "{robot1} and {robot2} pick {object1} and {object2} simultaneously",
        "sequential_stack": "{robot1} places {object1}, {robot2} stacks {object2} on top",
    }
