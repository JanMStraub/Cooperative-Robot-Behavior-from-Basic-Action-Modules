#!/usr/bin/env python3
"""Parses natural language commands into structured operation sequences via LLM + registry validation."""

from typing import Dict, Any, List, Optional, Tuple
import re
import logging
import requests
import functools

# Handle both direct execution and package import
try:
    from ..rag import RAGSystem
    from ..config.Servers import (
        LMSTUDIO_BASE_URL,
        DEFAULT_LMSTUDIO_MODEL,
        DEFAULT_TEMPERATURE,
        LLM_REQUEST_TIMEOUT,
        LLM_THINKING_BUDGET,
        LLM_THINKING_ENABLED,
        SYSTEM_PROMPT_BASE,
        USE_MOTION_LAYER,
    )
    from ..config.Negotiation import USE_STRUCTURED_OUTPUT
    from ..config.Robot import HANDOFF_PRESENTATION_POSITION
    from ..operations.WorkflowPatterns import WorkflowPatternRegistry, WorkflowPattern
except ImportError:
    from rag import RAGSystem
    from config.Servers import (
        LMSTUDIO_BASE_URL,
        DEFAULT_LMSTUDIO_MODEL,
        DEFAULT_TEMPERATURE,
        LLM_REQUEST_TIMEOUT,
        LLM_THINKING_BUDGET,
        LLM_THINKING_ENABLED,
        SYSTEM_PROMPT_BASE,
        USE_MOTION_LAYER,
    )
    from config.Negotiation import USE_STRUCTURED_OUTPUT
    from config.Robot import HANDOFF_PRESENTATION_POSITION
    from operations.WorkflowPatterns import WorkflowPatternRegistry, WorkflowPattern

from core.LoggingSetup import setup_logging
from core.LLMUtils import extract_json as _extract_json_util

setup_logging(__name__)
logger = logging.getLogger(__name__)

_HPP = HANDOFF_PRESENTATION_POSITION
_HANDOFF_X, _HANDOFF_Y, _HANDOFF_Z = _HPP[0], _HPP[1], _HPP[2]


class _PromptBuilder:
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

        return f"""
        Available Operations: {available_ops}

        Command to parse: "{command_text}"
        Default robot_id: "{robot_id}"

        === ROBOT WORKSPACE BOUNDARIES ===

        Robot1 (left, x=-0.475): reachable x < 0.165. Robot2 (right, x=+0.475): reachable x > -0.165.
        x > 0 → Robot2's side. x < 0 → Robot1's side. x ≈ 0 → shared.
        Wrong-side task → use HANDOFF sequence.

        === MULTI-ROBOT COORDINATION ===

        Multi-robot tasks use "plan" format (with "reasoning" + per-op "parallel_group"). Single-robot → "commands" format.
        Same parallel_group = concurrent. Later group waits for all prior groups.
        VARIABLE DEPENDENCY LAW: if B reads $var captured by A, B must have strictly higher parallel_group than A. Never same group.

        === HANDOFF RULE ===

        Exact steps with required parallel_group numbers (no composite ops, no deviations):
        group=1: Robot1: detect_object_stereo (capture_var="target")
        group=2: Robot1: grasp_object(object_id="$target.color") — MUST be group=2 (after detect)
        group=3: Robot1: return_to_start_position — MUST be group=3 (after grasp, never same group as grasp)
        group=4: Robot1: move_to_coordinate({_HANDOFF_X:.2f}, {_HANDOFF_Y:.2f}, {_HANDOFF_Z:.2f}) — NO approach_offset; own group
        group=5: Robot1: adjust_end_effector_orientation(pitch=0, yaw=0, roll=0)
        group=6: Robot1: signal("r1_at_handoff") + Robot2: wait_for_signal("r1_at_handoff") — SAME group
        group=7: Robot2: detect_object_stereo(color=<same as step 1>, capture_var="handoff_target") — object moved with Robot1
        group=8: Robot2: receive_handoff(object_id="$handoff_target.color", source_robot_id="Robot1")
        group=9: Robot1: release_object

        Receiving robot: always receive_handoff, never grasp_object. Handoff coord is always exactly ({_HANDOFF_X:.2f}, {_HANDOFF_Y:.2f}, {_HANDOFF_Z:.2f}). Step 7 color must match step 1 (never null). signal+wait_for_signal always same group.

        === SYNCHRONIZATION PRIMITIVES ===

        signal(event_name): emit event. wait_for_signal(event_name, timeout_ms=30000): wait for event. wait(duration_ms): time pause.
        mirror_movement_of_other_robot(duration_ms): duration_ms in [1000, 60000].

        === NAVIGATION RULE ===

        Move/navigate/approach WITHOUT pick/grab/grasp language → move_to_coordinate only (no gripper ops).
        Always set approach_offset=0.10 when moving to a detected object (lifts gripper above table; range 0.0–0.10).
        receive_handoff is not navigation — never replace with move_to_coordinate.

        "detect blue cube and move to it":
        {{"operation": "detect_object_stereo", "params": {{"robot_id": "Robot1", "color": "blue"}}, "capture_var": "target"}}
        {{"operation": "move_to_coordinate", "params": {{"robot_id": "Robot1", "x": "$target.x", "y": "$target.y", "z": "$target.z", "approach_offset": 0.10}}}}

        === GRASP RULE ===

        Pick/grab/grasp → grasp_object (handles approach+descent+grip; no separate move_to_coordinate before it).
        object_id always uses ".color" from detection ($target.color — never .id or .name).
        Never use grasp_object with a $field var (detect_field has no .color). Never on place/deposit tasks. Never for receiving robot in handoff (use receive_handoff).

        === PLACE RULE ===

        Place/drop/deposit → place_object(x, y, z) — hover, descend, open, ascend. Not release_object, not control_gripper.
        release_object: only for immediate drop at current position (emergency/handoff transfer).
        Typical sequence: detect_field → place_object($field.x/y/z).

        === BETWEEN PLACEMENT ===

        When the task says "place between X and Y", "put it midway between", or "place in the middle of X and Y":
        PREFER place_between_objects — it resolves both objects from WorldState and computes the midpoint internally.

        Example — "place the held object between the blue and red cube":
        {{"operation": "detect_object_stereo", "params": {{"robot_id": "Robot1", "color": "blue"}}, "parallel_group": 1}}
        {{"operation": "detect_object_stereo", "params": {{"robot_id": "Robot1", "color": "red"}}, "parallel_group": 1}}
        {{"operation": "place_between_objects", "params": {{"robot_id": "Robot1", "object_id_1": "blue", "object_id_2": "red"}}, "parallel_group": 2}}

        The two detect calls CAN share the same parallel_group (they are independent of each other).
        place_between_objects MUST be in a strictly higher parallel_group than both detects.

        Fallback — if objects are already in WorldState (no detection needed):
        {{"operation": "place_between_objects", "params": {{"robot_id": "Robot1", "object_id_1": "blue_cube", "object_id_2": "red_cube"}}}}

        Multi-variable arithmetic in params is also supported when you need a custom midpoint:
        {{"operation": "place_object", "params": {{"robot_id": "Robot1", "x": "($blue_obj.x + $red_obj.x) / 2", "y": "($blue_obj.y + $red_obj.y) / 2", "z": "($blue_obj.z + $red_obj.z) / 2"}}}}

        === SINGLE-ROBOT RULES ===

        Each action = separate op. Include robot_id in every op. Preserve order.
        "close gripper/grip" (not a pick) → control_gripper(open_gripper=false). "open gripper/release" → open_gripper=true.
        Never add return_to_start, signal, or adjust_end_effector_orientation after a grasp unless explicitly requested.

        "grab the blue cube": detect_object_stereo(color="blue", capture_var="target") → grasp_object(object_id="$target.color")

        === VARIABLE PASSING ===

        capture_var defines a variable; $var references it in later ops (never before capture).
        detect_object_stereo fields: x, y, z, color, confidence. For grasp: $target.color (never .id/.name).
        detect_field fields: x, y, z directly (use $field.x not $field.center.x).

        Pick-and-place: detect_object_stereo(capture_var="target") → grasp_object($target.color) → detect_field(field_label="G", capture_var="field") → place_object($field.x, $field.y, $field.z)

        === DETECT_FIELD RULE (CRITICAL) ===
        detect_field ALWAYS requires field_label (a single letter A-I). NEVER omit it.
        WRONG: {{"operation": "detect_field", "params": {{"robot_id": "Robot1"}}}}
        RIGHT: {{"operation": "detect_field", "params": {{"robot_id": "Robot1", "field_label": "A"}}}}
        If the task does not specify a field letter, infer it from context or ask — do NOT emit detect_field without field_label.
        
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


class CommandParser:
    """Parses compound natural language commands into operation sequences; falls back to regex."""

    def __init__(
        self,
        lm_studio_url: Optional[str] = None,
        model: Optional[str] = None,
        use_rag: bool = True,
        use_cache: bool = True,
    ):
        from core.Imports import get_global_registry

        self.lm_studio_url = lm_studio_url or LMSTUDIO_BASE_URL
        self.model = model or DEFAULT_LMSTUDIO_MODEL
        self.registry = get_global_registry()
        self.workflow_registry = WorkflowPatternRegistry()
        self.use_cache = use_cache

        # Connection pooling for LLM requests
        from requests.adapters import HTTPAdapter

        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=1)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        if self.use_cache:
            self._parse_cache = functools.lru_cache(maxsize=128)(self._do_llm_request)
        else:
            self._parse_cache = self._do_llm_request

        self.rag = None
        if use_rag:
            try:
                self.rag = RAGSystem()
                # Provide control over index rebuilding to speed up startups
                self.rag.index_operations(rebuild=False)
                logger.info("RAG system initialized for command parsing")
            except Exception as e:
                logger.warning(f"Failed to initialize RAG: {e}. Using registry only.")

        # Prompt builder - separated for testability
        self._prompt_builder = _PromptBuilder(
            self.registry, self.workflow_registry, self.rag
        )

    def parse(
        self,
        command_text: str,
        robot_id: str = "Robot1",
        use_llm: bool = True,
        use_motion_layer: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Parse a natural language command into a list of validated operations."""
        if not command_text or not command_text.strip():
            return {"success": False, "commands": [], "error": "Empty command text"}

        motion_layer = (
            USE_MOTION_LAYER if use_motion_layer is None else use_motion_layer
        )

        # Perception-only commands have no physical motions - motion layer Stage 1
        # would hallucinate a move plan and corrupt Stage 2 intent. Bypass it.
        if motion_layer and self._is_perception_only_command(command_text):
            motion_layer = False

        # Try LLM parsing first
        if use_llm:
            if motion_layer:
                result = self._parse_with_motion_layer(command_text, robot_id)
            else:
                result = self._parse_with_llm(command_text, robot_id)
            if result["success"]:
                return result
            logger.warning(
                f"LLM parsing failed: {result.get('error')}. Falling back to regex."
            )

        # Fallback to regex parsing
        regex_result = self._parse_with_regex(command_text, robot_id)
        if regex_result["success"]:
            return regex_result

        return regex_result

    def parse_with_hint(
        self,
        command_text: str,
        robot_id: str = "Robot1",
        hint: str = "",
        use_motion_layer: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Re-parse with a Reflexion error hint so the LLM can fix its parameter choices."""
        motion_layer = (
            USE_MOTION_LAYER if use_motion_layer is None else use_motion_layer
        )

        # Stage 1: decompose to motions if enabled (same as main parse() path)
        effective_command = command_text
        if motion_layer:
            motions = self._decompose_to_motions(command_text, robot_id)
            if motions:
                motion_context = "\n".join(
                    f"  {i + 1}. {m}" for i, m in enumerate(motions)
                )
                effective_command = (
                    f"{command_text}\n\n"
                    f"Motion plan (use as chain-of-thought guidance):\n{motion_context}"
                )
                logger.info(
                    f"Reflexion retry: motion layer Stage 1 produced {len(motions)} steps"
                )

        kg_section = self._get_spatial_context(robot_id, command_text=command_text)
        peer_section = self._get_peer_context(robot_id)
        spatial_section = "\n        ".join(s for s in [kg_section, peer_section] if s)
        prompt = self._prompt_builder.build(
            effective_command,
            robot_id,
            spatial_section=spatial_section,
            hint=hint,
        )
        try:
            result = self._do_llm_request(prompt, command_text)
            if not result.get("success"):
                return {"success": False, "commands": [], "error": result.get("error")}
            parsed = result["parsed"]
            if "plan" in parsed and "commands" not in parsed:
                flat = []
                for item in parsed["plan"]:
                    if "operations" in item:
                        pg = item.get("parallel_group")
                        for op in item["operations"]:
                            op = dict(op)
                            if pg is not None:
                                op["parallel_group"] = pg
                            flat.append(op)
                    else:
                        flat.append(item)
                parsed["commands"] = flat
            commands = parsed.get("commands", [])
            validated = self._validate_commands(commands, robot_id)
            if not validated:
                return {
                    "success": False,
                    "commands": [],
                    "error": "Reflexion retry produced no valid commands",
                }
            return {"success": True, "commands": validated, "error": None}
        except Exception as e:
            return {
                "success": False,
                "commands": [],
                "error": f"Reflexion LLM error: {e}",
            }

    _PERCEPTION_ONLY_PATTERNS = re.compile(
        r"\b(analyze\s+(the\s+)?scene|describe\s+(the\s+)?(scene|workspace|environment)|"
        r"what\s+(do\s+you\s+see|objects?\s+are|can\s+you\s+see|'?s\s+on\s+the\s+table)|"
        r"(scan|inspect|observe|survey|look\s+at)\s+(the\s+)?(scene|workspace)|"
        r"what\s+is\s+in\s+(the\s+)?scene|"
        r"detect\s+(object|field|all\s+fields?)|"
        r"generate\s+point\s+cloud|"
        r"check\s+(robot\s+)?status|"
        r"wait\s+\d|"
        r"wait\s+for\s+(signal|event))\b",
        re.IGNORECASE,
    )

    def _is_perception_only_command(self, command_text: str) -> bool:
        if re.match(r"\s*signal\s+\S", command_text, re.IGNORECASE):
            return True
        return bool(self._PERCEPTION_ONLY_PATTERNS.search(command_text))

    def _decompose_to_motions(self, command_text: str, robot_id: str) -> List[str]:
        """Stage 1: decompose a high-level command into concrete motion strings for chain-of-thought anchoring."""
        prompt = (
            f'High-level command: "{command_text}"\n'
            f"Default robot: {robot_id}\n\n"
            "Decompose this command into an ordered list of short, concrete physical motion descriptions. Each entry should describe one distinct robot motion (e.g. 'move end-effector to target position', 'lower arm to 0.3m height').\n"
            "IMPORTANT: Do NOT add gripper open/close or grasp motions unless the command EXPLICITLY says to pick up, grab, grasp, or grip an object. Pure navigation ('move to it', 'navigate to X', 'go to position') must NOT include any gripper step.\n Navigation example: 'detect the blue cube and move to it': "
            '["detect blue cube position", "move end-effector to detected position"]\n'
            "Pick example: 'grasp the red cube': "
            '["grasp red cube (full approach and grip)"]\n'
            "IMPORTANT: grasp/pick/grab is always a SINGLE step (do NOT split into approach + descend + close).\n"
            "Output a JSON array of strings only. No markdown."
        )
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_BASE},
                    {"role": "user", "content": prompt},
                ],
                "temperature": DEFAULT_TEMPERATURE,
                "max_tokens": 512,
            }
            response = self._session.post(
                f"{self.lm_studio_url}/chat/completions",
                json=payload,
                timeout=LLM_REQUEST_TIMEOUT,
            )
            if response.status_code != 200:
                logger.warning(
                    f"Motion decomposition LLM returned {response.status_code}"
                )
                return []
            content = response.json()["choices"][0]["message"]["content"]
            motions = _extract_json_util(content)
            if isinstance(motions, list) and all(isinstance(m, str) for m in motions):
                logger.info(f"Motion decomposition: {motions}")
                return motions
            logger.warning(
                f"Motion decomposition returned unexpected format: {content[:200]}"
            )
            return []
        except Exception as e:
            logger.warning(f"Motion decomposition failed: {e}")
            return []

    def _parse_with_motion_layer(
        self, command_text: str, robot_id: str
    ) -> Dict[str, Any]:
        """Two-stage RT-H parse: decompose to motions, then map motions to ops."""
        motions = self._decompose_to_motions(command_text, robot_id)
        if not motions:
            logger.info(
                "Motion decomposition empty, falling back to standard LLM parse"
            )
            return self._parse_with_llm(command_text, robot_id)

        motion_context = "\n".join(f"  {i + 1}. {m}" for i, m in enumerate(motions))
        augmented_command = (
            f"{command_text}\n\n"
            f"Motion plan (use as chain-of-thought guidance):\n{motion_context}"
        )
        logger.info(f"Motion layer Stage 2 with {len(motions)} motion steps")
        return self._parse_with_llm(augmented_command, robot_id)

    def _parse_with_llm(self, command_text: str, robot_id: str) -> Dict[str, Any]:
        kg_section = self._get_spatial_context(robot_id, command_text=command_text)
        peer_section = self._get_peer_context(robot_id)
        spatial_section = "\n        ".join(s for s in [kg_section, peer_section] if s)
        prompt = self._prompt_builder.build(
            command_text,
            robot_id,
            spatial_section=spatial_section,
        )

        try:
            # Use cached or direct request depending on initialization
            result = self._parse_cache(prompt, command_text)

            if not result.get("success"):
                return result

            parsed = result["parsed"]

            # Normalize multi-robot "plan" format to "commands" format.
            # LLM may emit plan in two shapes:
            #   A) flat list: [{parallel_group, robot, operation, params, ...}, ...]
            #   B) grouped:   [{parallel_group, operations:[{robot, operation, ...}]}, ...]
            # Shape A has "operation" directly on each item; shape B has "operations" sub-list.
            if "plan" in parsed and "commands" not in parsed:
                logger.info(
                    f"Multi-robot plan detected with reasoning: {parsed.get('reasoning', 'N/A')}"
                )
                flat: List[Dict] = []
                for item in parsed["plan"]:
                    if "operations" in item:
                        # Shape B: grouped format
                        pg = item.get("parallel_group")
                        for op in item["operations"]:
                            op = dict(op)
                            if pg is not None:
                                op["parallel_group"] = pg
                            flat.append(op)
                    else:
                        # Shape A: flat format — item is already an operation
                        flat.append(item)
                parsed["commands"] = flat

            logger.info(f"Parsed {len(parsed.get('commands', []))} commands from LLM")

            # Validate operations
            commands = parsed.get("commands", [])
            validated_commands = self._validate_commands(commands, robot_id)
            logger.info(f"Validated {len(validated_commands)} commands")

            if not validated_commands:
                return {
                    "success": False,
                    "commands": [],
                    "error": "LLM produced no valid commands - will try regex fallback",
                }

            return {
                "success": True,
                "commands": validated_commands,
                "error": None,
                "reasoning": parsed.get("reasoning"),  # Preserve reasoning if present
            }

        except requests.exceptions.Timeout:
            return {"success": False, "commands": [], "error": "LLM request timed out"}
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "commands": [],
                "error": f"Cannot connect to LM Studio at {self.lm_studio_url}",
            }
        except Exception as e:
            return {
                "success": False,
                "commands": [],
                "error": f"LLM parsing error: {str(e)}",
            }

    def _do_llm_request(self, prompt: str, _command_text: str) -> Dict[str, Any]:
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            SYSTEM_PROMPT_BASE
                            + " Map natural language variations to canonical parameter "
                            "values (e.g., 'leftmost' -> 'left', 'rightmost' -> 'right', "
                            "'nearest' -> 'closest', 'grab' -> grasp_object). "
                            "Preserve the exact operation names from the registry - never "
                            "invent new operation names."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": DEFAULT_TEMPERATURE,  # Low temperature for deterministic parsing
                "max_tokens": 8192,  # Must cover thinking budget + actual JSON response
                **(
                    {
                        "thinking": {
                            "type": "enabled",
                            "budget_tokens": LLM_THINKING_BUDGET,
                        }
                    }
                    if LLM_THINKING_ENABLED
                    else {}
                ),
            }
            # Structured output forces the model to emit valid JSON at the inference layer.
            # Set USE_STRUCTURED_OUTPUT=false for models that don't support response_format.
            if USE_STRUCTURED_OUTPUT:
                payload["response_format"] = {"type": "json_object"}
            response = self._session.post(
                f"{self.lm_studio_url}/chat/completions",
                json=payload,
                timeout=LLM_REQUEST_TIMEOUT,
            )

            if response.status_code != 200:
                return {
                    "success": False,
                    "parsed": None,
                    "content": None,
                    "error": f"LLM request failed with status {response.status_code}",
                }

            # Extract content from response
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            logger.info(f"LLM response: {content}")

            # Parse JSON from response
            parsed = self._extract_json_from_response(content)
            logger.debug("Parsed LLM JSON: %s", parsed)

            # Catch array responses
            if isinstance(parsed, list):
                logger.info(f"LLM returned an array directly, wrapping in dict")
                parsed = {"commands": parsed}

            # Catch single-operation responses (no "commands"/"plan" wrapper)
            if (
                isinstance(parsed, dict)
                and "operation" in parsed
                and "commands" not in parsed
                and "plan" not in parsed
            ):
                logger.info(
                    "LLM returned single operation dict, wrapping in commands list"
                )
                parsed = {"commands": [parsed]}

            if not parsed:
                return {
                    "success": False,
                    "parsed": None,
                    "content": content,
                    "error": f"Failed to extract JSON from LLM response: {content[:200]}",
                }

            return {
                "success": True,
                "parsed": parsed,
                "content": content,
                "error": None,
            }

        except requests.exceptions.Timeout:
            raise
        except requests.exceptions.ConnectionError:
            raise
        except Exception:
            raise

    def _get_spatial_context(
        self, robot_id: str, target: Optional[tuple] = None, command_text: str = ""
    ) -> str:
        """Pull KG spatial context for the robot; returns "" if KG is disabled or unavailable."""
        try:
            from config.KnowledgeGraph import KNOWLEDGE_GRAPH_ENABLED

            if not KNOWLEDGE_GRAPH_ENABLED:
                return ""

            from core.Imports import get_graph_query_engine

            qe = get_graph_query_engine()
            if qe is None:
                return ""

            lines = ["=== SPATIAL CONTEXT (Knowledge Graph) ==="]

            # Reachable objects (top 5 by distance)
            reachable = qe.get_objects_in_reach(robot_id)[:5]
            if reachable:
                lines.append("Reachable objects:")
                for obj in reachable:
                    dist_str = (
                        f"{obj['distance']:.2f}m"
                        if obj["distance"] is not None
                        else "?"
                    )
                    held_str = (
                        f" [held by {obj['grasped_by']}]"
                        if obj.get("grasped_by")
                        else ""
                    )
                    lines.append(
                        f"  - {obj['object_id']} ({obj['color']}, {dist_str}){held_str}"
                    )

            # Nearby robots
            nearby = qe.find_robots_near(robot_id)
            if nearby:
                lines.append("Nearby robots:")
                for r in nearby:
                    lines.append(f"  - {r['robot_id']} ({r['distance']:.2f}m)")

            _handoff_keywords = ("hand", "pass", "give", "transfer", "handoff")
            mentions_handoff = any(
                kw in command_text.lower() for kw in _handoff_keywords
            )
            # Handoff candidates — only inject when command mentions handoff intent
            # to avoid polluting single-robot prompts with multi-robot context noise
            if reachable and mentions_handoff:
                all_robots = qe._graph.get_all_nodes(node_type="robot")
                other_robots = [r for r in all_robots if r != robot_id]
                for other in other_robots:
                    for obj in reachable:
                        candidates = qe.get_handoff_candidates(
                            robot_id, other, obj["object_id"]
                        )
                        if candidates:
                            c = candidates[0]
                            pos = c["position"]
                            lines.append(
                                f"Handoff {obj['object_id']} with {other}: "
                                f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
                                f"r1={c['r1_distance']:.2f}m r2={c['r2_distance']:.2f}m"
                            )

            # Path blocking check when a target coordinate is known
            if target is not None:
                blocked = qe.is_path_blocked(robot_id, target)
                lines.append(f"Path to target: {'BLOCKED' if blocked else 'clear'}")

            if len(lines) == 1:
                return ""  # Only header, no data

            return "\n        ".join(lines)

        except Exception as e:
            logger.debug(f"KG spatial context unavailable: {e}")
            return ""

    def _get_peer_context(self, robot_id: str) -> str:
        """Return peer robot state and collision-intent warnings; always runs regardless of KG."""
        try:
            from core.Imports import get_world_state

            ws = get_world_state()
            if ws is None:
                return ""

            peer_lines = []

            all_robots = ws.get_all_robots() if hasattr(ws, "get_all_robots") else []
            for r_state in all_robots:
                if r_state.robot_id == robot_id:
                    continue
                peer_ctx = ws.get_world_context_string(r_state.robot_id)
                if peer_ctx:
                    peer_lines.append(f"  {r_state.robot_id}: {peer_ctx}")

            intents = ws.get_robot_intents() if hasattr(ws, "get_robot_intents") else {}
            for other_id, target_obj in intents.items():
                if other_id != robot_id and target_obj:
                    peer_lines.append(
                        f"  WARNING: {other_id} is already moving toward {target_obj}"
                    )

            if not peer_lines:
                return ""

            lines = ["=== PEER ROBOT STATE ==="] + peer_lines
            return "\n        ".join(lines)
        except Exception as e:
            logger.debug(f"Peer context unavailable: {e}")
            return ""

    def _get_available_operations_summary(self, command_text: str = "") -> str:
        return self._prompt_builder.get_available_operations_summary(command_text)

    def _format_workflow_pattern(self, pattern: WorkflowPattern) -> str:
        return self._prompt_builder.format_workflow_pattern(pattern)

    def _extract_json_from_response(self, content: str) -> Optional[Dict]:
        return _extract_json_util(content)

    def _validate_commands(
        self, commands: List[Dict], default_robot_id: str
    ) -> List[Dict]:
        validated = []
        for cmd in commands:
            operation = cmd.get("operation", "")
            if isinstance(operation, list):
                operation = operation[0] if operation else ""
            params = cmd.get("params", {})

            # Ensure robot_id is present (use "robot" field if specified in multi-robot plan)
            if "robot_id" not in params:
                params["robot_id"] = cmd.get("robot", default_robot_id)

            # Verify operation exists
            op = self.registry.get_operation_by_name(operation)
            if op is None:
                logger.warning(
                    "Unknown operation '%s' (params=%s), skipping",
                    operation,
                    list(params.keys()),
                )
                continue

            validated_cmd = {"operation": operation, "params": params}

            # Preserve capture_var if present
            if "capture_var" in cmd:
                validated_cmd["capture_var"] = cmd["capture_var"]

            # Preserve parallel_group if present (for multi-robot coordination)
            if "parallel_group" in cmd:
                validated_cmd["parallel_group"] = cmd["parallel_group"]

            validated.append(validated_cmd)

        # Fix intra-group variable dependencies: if operation B uses $var captured by
        # operation A and both are in the same parallel_group, move B to a later group.
        validated = self._fix_intra_group_dependencies(validated)

        # Validate multi-robot plans (signal/wait pairs and variable usage)
        if len(validated) > 1:
            is_valid, errors = self._validate_multi_robot_plan(validated)
            if not is_valid:
                for error in errors:
                    logger.warning(f"Multi-robot plan validation warning: {error}")

        return validated

    def _fix_intra_group_dependencies(self, commands: List[Dict]) -> List[Dict]:
        """Push commands that read a $var to a later group than the one capturing it."""
        # Only applies when parallel groups are present
        if not any("parallel_group" in cmd for cmd in commands):
            return commands

        capture_group: Dict[str, int] = {}
        for cmd in commands:
            if "capture_var" in cmd and "parallel_group" in cmd:
                capture_group[cmd["capture_var"]] = cmd["parallel_group"]

        if not capture_group:
            return commands

        changed = True
        while changed:
            changed = False
            for cmd in commands:
                if "parallel_group" not in cmd:
                    continue
                # Find all $varname references in params
                required_groups = []
                for val in cmd.get("params", {}).values():
                    if isinstance(val, str):
                        for match in re.finditer(r"\$([a-zA-Z0-9_]+)", val):
                            var = match.group(1)
                            if var in capture_group:
                                required_groups.append(capture_group[var])
                if not required_groups:
                    continue
                min_required = max(required_groups) + 1  # must be AFTER all captures
                if cmd["parallel_group"] < min_required:
                    logger.warning(
                        f"[fix_intra_group] Moving '{cmd['operation']}' from group "
                        f"{cmd['parallel_group']} to {min_required} (variable dependency)"
                    )
                    cmd["parallel_group"] = min_required
                    changed = True

        # Renumber groups to be contiguous (1, 2, 3, …) preserving relative order
        groups_sorted = sorted(
            set(cmd["parallel_group"] for cmd in commands if "parallel_group" in cmd)
        )
        remap = {old: new for new, old in enumerate(groups_sorted, start=1)}
        for cmd in commands:
            if "parallel_group" in cmd:
                cmd["parallel_group"] = remap[cmd["parallel_group"]]

        return commands

    def _validate_multi_robot_plan(
        self, commands: List[Dict]
    ) -> Tuple[bool, List[str]]:
        errors = []
        defined_signals = set()
        expected_signals = set()
        defined_vars = set()

        for cmd in commands:
            operation = cmd.get("operation", "")
            params = cmd.get("params", {})

            # Track signal definitions
            if operation == "signal":
                event_name = params.get("event_name")
                if event_name:
                    defined_signals.add(event_name)

            # Track wait_for_signal expectations
            elif operation == "wait_for_signal":
                event_name = params.get("event_name")
                if event_name:
                    expected_signals.add(event_name)

            # Track variable definitions
            if "capture_var" in cmd:
                var_name = cmd["capture_var"]
                defined_vars.add(var_name)

            # Check variable usage in parameters
            for key, val in params.items():
                if isinstance(val, str) and "$" in val:
                    # Find all variables in the string, which might be expressions like "$target.x + 0.5"
                    # Capture groups will only pick up the variable name [a-zA-Z0-9_]+
                    matches = re.finditer(r"\$([a-zA-Z0-9_]+)", val)
                    for match in matches:
                        var_name = match.group(1)
                        if var_name not in defined_vars:
                            errors.append(
                                f"Variable ${var_name} used in {operation}.{key} before definition"
                            )

        # Check all waited signals are defined
        missing = expected_signals - defined_signals
        if missing:
            errors.append(
                f"wait_for_signal without matching signal: {', '.join(missing)}"
            )

        return len(errors) == 0, errors

    def _parse_with_regex(self, command_text: str, robot_id: str) -> Dict[str, Any]:
        """Regex fallback for common command patterns when LLM is unavailable."""
        commands = []
        text = command_text.lower()

        # Split by common conjunctions
        parts = re.split(r"\s+(?:and|then|after that|,)\s+", text)

        # Track if we detected something (for "move to it" pattern)
        last_detection_var = None

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Parse detect colored object (unified stereo detection)
            detect_color_match = re.search(
                r"detect\s+(?:the\s+)?(\w+)\s+(?:cube|object|block)",
                part,
            )
            if detect_color_match:
                color = detect_color_match.group(1).lower()
                if color in [
                    "red",
                    "green",
                    "blue",
                    "yellow",
                    "purple",
                    "orange",
                    "cyan",
                    "magenta",
                ]:
                    last_detection_var = "target"
                    commands.append(
                        {
                            "operation": "detect_object_stereo",
                            "params": {"robot_id": robot_id, "color": color},
                            "capture_var": last_detection_var,
                        }
                    )
                    continue

            # Parse "move to it" / "move to the coordinates" (uses last detection)
            if re.search(r"move\s+to\s+(?:it|the\s+coordinates?|there|that)", part):
                if last_detection_var:
                    commands.append(
                        {
                            "operation": "move_to_coordinate",
                            "params": {
                                "robot_id": robot_id,
                                "position": f"${last_detection_var}",
                            },
                        }
                    )
                    continue

            # Parse move commands with explicit coordinates
            move_match = re.search(
                r"move\s+(?:\w+\s+)?(?:to\s+)?(?:"
                r"\(?\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)?|"
                r"x\s*=?\s*(-?[\d.]+).*?y\s*=?\s*(-?[\d.]+).*?z\s*=?\s*(-?[\d.]+)|"
                r"(?:to\s+)?coordinates?\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+))",
                part,
            )
            if move_match:
                groups = move_match.groups()
                if groups[0] is not None:
                    x, y, z = float(groups[0]), float(groups[1]), float(groups[2])
                elif groups[3] is not None:
                    x, y, z = float(groups[3]), float(groups[4]), float(groups[5])
                else:
                    x, y, z = float(groups[6]), float(groups[7]), float(groups[8])

                commands.append(
                    {
                        "operation": "move_to_coordinate",
                        "params": {"robot_id": robot_id, "x": x, "y": y, "z": z},
                    }
                )
                continue

            # Parse gripper commands - check open first to avoid "grip" matching "gripper"
            if re.search(r"open\s+(?:the\s+)?gripper|release|drop", part):
                commands.append(
                    {
                        "operation": "control_gripper",
                        "params": {"robot_id": robot_id, "open_gripper": True},
                    }
                )
                continue

            if re.search(r"close\s+(?:the\s+)?gripper|grasp\b|grip\b|grab\b", part):
                commands.append(
                    {
                        "operation": "control_gripper",
                        "params": {"robot_id": robot_id, "open_gripper": False},
                    }
                )
                continue

            # Parse status check
            if re.search(r"check\s+(?:robot\s+)?status|get\s+status", part):
                commands.append(
                    {
                        "operation": "check_robot_status",
                        "params": {"robot_id": robot_id},
                    }
                )
                continue

            # Parse return to start position (handles "return Robot1 to start" and "return to start")
            if re.search(
                r"return\s+(?:\w+\s+)?(?:to\s+)?(?:start|home|default|initial)\s*(?:position)?|go\s+(?:to\s+)?home|home\s+position",
                part,
            ):
                commands.append(
                    {
                        "operation": "return_to_start_position",
                        "params": {"robot_id": robot_id},
                    }
                )
                continue

            # Parse stereo detection with depth (3D positions) - unified operation
            if re.search(
                r"detect.*(?:depth|3d|position|stereo)|find.*(?:3d|position)|calculate\s+(?:object\s+)?coordinates|locate\s+(?:objects?|cubes?)\s+in\s+3d",
                part,
            ):
                commands.append(
                    {
                        "operation": "detect_object_stereo",
                        "params": {"robot_id": robot_id, "color": None},
                    }
                )
                continue

            # Parse simple object detection - route to stereo detection (3D)
            if re.search(
                r"detect\s+(?:objects?|cubes?)|find\s+(?:objects?|cubes?)|look\s+for|scan\s+for|locate\s+(?:objects?|cubes?)",
                part,
            ):
                commands.append(
                    {
                        "operation": "detect_object_stereo",
                        "params": {"robot_id": robot_id},
                    }
                )
                continue

            # Parse analyze scene
            if re.search(r"analyze\s+(?:the\s+)?scene|scene\s+analysis", part):
                commands.append(
                    {
                        "operation": "analyze_scene",
                        "params": {
                            "robot_id": robot_id,
                            "prompt": "Describe what you see in the scene.",
                        },
                    }
                )
                continue

            # Parse generate point cloud
            if re.search(r"generate\s+(?:a\s+)?point\s+cloud|point\s+cloud", part):
                commands.append(
                    {
                        "operation": "generate_point_cloud",
                        "params": {"robot_id": robot_id},
                    }
                )
                continue

            # Parse wait (duration)
            wait_dur_match = re.search(
                r"wait\s+(?:for\s+)?(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s\b)",
                part,
            )
            if wait_dur_match:
                commands.append(
                    {
                        "operation": "wait",
                        "params": {
                            "robot_id": robot_id,
                            "duration_ms": int(float(wait_dur_match.group(1)) * 1000),
                        },
                    }
                )
                continue

            # Parse wait for signal
            wait_sig_match = re.search(
                r"wait\s+(?:for\s+)?(?:signal|event)\s+(\S+)", part
            )
            if wait_sig_match:
                commands.append(
                    {
                        "operation": "wait_for_signal",
                        "params": {
                            "robot_id": robot_id,
                            "event_name": wait_sig_match.group(1),
                        },
                    }
                )
                continue

            # Parse signal (fire event)
            signal_match = re.search(r"^signal\s+(\S+)", part)
            if signal_match:
                commands.append(
                    {
                        "operation": "signal",
                        "params": {
                            "robot_id": robot_id,
                            "event_name": signal_match.group(1),
                        },
                    }
                )
                continue

        if commands:
            return {"success": True, "commands": commands, "error": None}
        else:
            return {
                "success": False,
                "commands": [],
                "error": f"Could not parse command: {command_text}",
            }

    def get_supported_patterns(self) -> List[str]:
        return [
            "move to (x, y, z) - Move robot to coordinates",
            "move to x=0.3, y=0.2, z=0.1 - Move robot to coordinates",
            "close gripper / grasp / grip / grab - Close the gripper",
            "open gripper / release / drop - Open the gripper",
            "check status / get status - Get robot status",
            "return to start / go home / home position - Return to start position",
            "detect with depth / find 3d positions / detect stereo - Detect objects with 3D positions",
            "Commands can be chained with 'and', 'then', 'after that', or commas",
        ]


# Singleton instance
_parser_instance: Optional[CommandParser] = None


def get_command_parser() -> CommandParser:
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = CommandParser()
    return _parser_instance
