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
        LLM_MAX_TOKENS,
        SYSTEM_PROMPT_BASE,
        USE_MOTION_LAYER,
    )
    from ..config.Negotiation import USE_STRUCTURED_OUTPUT
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
        LLM_MAX_TOKENS,
        SYSTEM_PROMPT_BASE,
        USE_MOTION_LAYER,
    )
    from config.Negotiation import USE_STRUCTURED_OUTPUT
    from operations.WorkflowPatterns import WorkflowPatternRegistry, WorkflowPattern

from core.LoggingSetup import setup_logging
from core.LLMUtils import extract_json as _extract_json_util

setup_logging(__name__)
logger = logging.getLogger(__name__)

try:
    from .PromptBuilder import PromptBuilder as _PromptBuilder  # type: ignore[import]
except ImportError:
    from orchestrators.PromptBuilder import PromptBuilder as _PromptBuilder  # type: ignore[no-redef]


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

        # Perception-only, place, and atomic gripper commands have no motion
        # decomposition benefit — Stage 1 would hallucinate approach coordinates
        # and corrupt Stage 2 intent. Bypass the motion layer for all three.
        if motion_layer and (
            self._is_perception_only_command(command_text)
            or self._is_place_command(command_text)
            or self._is_atomic_gripper_command(command_text)
        ):
            motion_layer = False

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

        # Same bypasses as parse() — atomic gripper/place/perception commands
        # must not go through Stage 1 decomposition on retries either.
        if motion_layer and (
            self._is_perception_only_command(command_text)
            or self._is_place_command(command_text)
            or self._is_atomic_gripper_command(command_text)
        ):
            motion_layer = False

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
                parsed["commands"] = self._flatten_plan(parsed["plan"])
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

    def generate_reflection(
        self,
        command_text: str,
        operation: str,
        error: str,
        params: Dict[str, Any],
        robot_id: str = "Robot1",
    ) -> str:
        """LLM self-reflection on a failed operation (Shinn et al. 2023).

        Returns 1-3 sentences of verbal analysis: what went wrong and what
        should change. Falls back to empty string on any LLM failure so the
        caller still proceeds with the procedural hint.
        """
        import json as _json_mod

        system = (
            "You are a robot command analyzer. Return only JSON with keys "
            '"analysis" (why the operation failed) and "suggestion" '
            "(what specific change — coordinates, operation choice, object ID — "
            "would fix it). Be concise: 1-2 sentences per key."
        )
        user = (
            f'Original task: "{command_text}"\n'
            f"Robot: {robot_id}\n"
            f"Failed operation: {operation}\n"
            f"Parameters tried: {_json_mod.dumps(params)}\n"
            f"Error: {error}"
        )
        payload: Dict[str, Any] = {
            "model": DEFAULT_LMSTUDIO_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": 512,
        }
        if USE_STRUCTURED_OUTPUT:
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = self._session.post(
                f"{LMSTUDIO_BASE_URL}/chat/completions",
                json=payload,
                timeout=min(LLM_REQUEST_TIMEOUT, 30.0),
            )
            if resp.status_code != 200:
                logger.debug("generate_reflection: LLM returned %d", resp.status_code)
                return ""
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = _extract_json_util(content)
            if isinstance(parsed, dict):
                parts = [
                    s
                    for s in [parsed.get("analysis", ""), parsed.get("suggestion", "")]
                    if s
                ]
                return " ".join(parts)
            return content[:400]
        except Exception as exc:
            logger.debug("generate_reflection: fallback due to %s", exc)
            return ""

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

    _PLACE_PATTERNS = re.compile(
        r"\b(place|deposit|put\s+down|put\s+it)\b", re.IGNORECASE
    )

    # Pure gripper state commands — no motion decomposition needed; Stage 1
    # hallucinate approach coordinates and Stage 2 then picks move_to_coordinate,
    # which triggers the navigation collision check erroneously.
    _ATOMIC_GRIPPER_PATTERNS = re.compile(
        r"\b(open\s+(the\s+)?gripper|close\s+(the\s+)?gripper|release|drop(\s+the\s+object)?)\b",
        re.IGNORECASE,
    )

    def _is_place_command(self, command_text: str) -> bool:
        """Return True if command intends to place/deposit an object.

        The motion layer rewrites 'place object at X Y Z' as 'move end-effector
        to target position X Y Z', stripping the place semantic and causing the
        LLM to pick move_to_coordinate instead of place_object. Bypass it.
        """
        return bool(self._PLACE_PATTERNS.search(command_text))

    def _is_atomic_gripper_command(self, command_text: str) -> bool:
        """Return True for bare open/close gripper commands with no grasp target."""
        return bool(self._ATOMIC_GRIPPER_PATTERNS.search(command_text))

    def _is_perception_only_command(self, command_text: str) -> bool:
        if re.match(r"\s*signal\s+\S", command_text, re.IGNORECASE):
            return True
        return bool(self._PERCEPTION_ONLY_PATTERNS.search(command_text))

    def _decompose_to_motions(self, command_text: str, robot_id: str) -> List[str]:
        """Stage 1: decompose a high-level command into concrete motion strings for chain-of-thought anchoring."""
        prompt = f"""
            High-level command: "{command_text}"

            Default robot: {robot_id}

            Decompose this command into an ordered list of short, concrete physical motion descriptions. Each entry should describe one distinct robot motion (e.g. 'move end-effector to target position', 'lower arm to 0.3m height').

            RULES:
            1. Navigation only (no grasp/pick/grab/grip in command): detect position, then move. Do NOT add any gripper or grasp step.
               Example: 'detect the orange cube and approach it': ["detect orange cube position", "move end-effector to detected position"]

            2. Grasp command (grasp/pick/pick up/grab/grip appears in command): detect position, then grasp as a SINGLE step. Never write 'move end-effector' for a grasp, always write 'grasp <object> (full approach and grip)'.
               Example: 'grasp the purple cube': ["detect purple cube position", "grasp purple cube (full approach and grip)"]
               Example: 'Robot1: grasp the yellow cube, and lift it to y=0.15': ["detect yellow cube position", "grasp yellow cube (full approach and grip)", "lift end-effector to y=0.15"]

            3. Place command (place/deposit/put down in command): always a SINGLE step written as 'place object at <target>'. Never write 'move end-effector to position' for a place step. Place always follows a grasp; do NOT merge them.
               Example: 'grasp the cyan cube and place it in field C': ["detect cyan cube position", "grasp cyan cube (full approach and grip)", "place cyan cube at field C"]

            4. Multi-robot tasks: include motions for ALL robots in sequence order.
               Example: 'Robot1 grasps the green cube and hands it to Robot2': ["Robot1: detect green cube", "Robot1: grasp green cube (full approach and grip)", "Robot1: move to handoff position", "Robot1: orient end-effector", "Robot1: signal ready + Robot2: wait for signal", "Robot2: detect green cube at Robot1's hand", "Robot2: receive handoff from Robot1", "Robot1: release object"]

            5. ROBOT ASSIGNMENT RULE for multi-robot tasks:
               Assign robots based on object workspace, NOT linguistic cues like "the other robot" or "Robot2":
               - Objects at x < 0 (left side) -> Robot1 only.
               - Objects at x > 0 (right side) -> Robot2 only.
               - Objects at x = 0 (shared zone, |x| < 0.1) -> either robot.
               If the task says "the other robot picks up the blue cube" and blue_cube is on the left (x < 0), assign Robot1 and NOT Robot2.
               When object positions are unknown, use color/name context: blue/yellow cubes are typically on the left (Robot1), green/magenta/red cubes may vary.

            Output a JSON array of strings only. No markdown.
        """
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
            if not content or not content.strip():
                logger.warning("Motion decomposition returned empty response")
                return []
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
            f"Motion plan ({len(motions)} steps — map each step to exactly one operation; "
            f"steps are strictly sequential unless the step text says '+ ' or 'parallel with', "
            f"in which case those steps share the same parallel_group):\n{motion_context}"
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
            result = self._parse_cache(prompt, command_text)

            if not result.get("success"):
                return result

            parsed = result["parsed"]

            # LLM may emit plan in two shapes:
            #   A) flat list: [{parallel_group, operation, ...}, ...]
            #   B) grouped:   [{parallel_group, operations:[...]}]
            # Also accept "operations" as a top-level alias for "commands".
            if (
                "operations" in parsed
                and "commands" not in parsed
                and "plan" not in parsed
            ):
                parsed["commands"] = parsed["operations"]

            if "plan" in parsed and "commands" not in parsed:
                logger.info(
                    f"Multi-robot plan detected with reasoning: {parsed.get('reasoning', 'N/A')}"
                )
                parsed["commands"] = self._flatten_plan(parsed["plan"])

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
                "max_tokens": LLM_MAX_TOKENS,
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
                try:
                    err_body = response.json()
                except Exception:
                    err_body = response.text[:400]
                logger.error("LLM returned %d: %s", response.status_code, err_body)
                # If the request included a thinking block and the model rejected it
                # (some LM Studio models don't support the OpenAI-style thinking
                # extension), retry once without it.
                if "thinking" in payload and response.status_code == 400:
                    payload_no_thinking = {
                        k: v for k, v in payload.items() if k != "thinking"
                    }
                    logger.info("Retrying LLM request without thinking block")
                    response = self._session.post(
                        f"{self.lm_studio_url}/chat/completions",
                        json=payload_no_thinking,
                        timeout=LLM_REQUEST_TIMEOUT,
                    )
                    if response.status_code != 200:
                        return {
                            "success": False,
                            "parsed": None,
                            "content": None,
                            "error": f"LLM request failed with status {response.status_code}",
                        }
                else:
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

            # Cross-robot reachability map — tells the LLM which robot can reach each object
            try:
                all_robot_nodes = qe._graph.get_all_nodes(node_type="robot")
                other_robots = [r for r in all_robot_nodes if r != robot_id]
                if other_robots:
                    lines.append(
                        "Object reachability by robot (use correct robot for each object):"
                    )
                    for rid in [robot_id] + other_robots:
                        robot_objs = qe.get_objects_in_reach(rid)[:8]
                        obj_names = [o["object_id"] for o in robot_objs]
                        lines.append(
                            f"  {rid}: {', '.join(obj_names) if obj_names else 'none in reach'}"
                        )
            except Exception:
                pass

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

    def _expand_array_operations(self, commands: List[Dict]) -> List[Dict]:
        """Expand {operation: [...], params: [...]} LLM responses into separate command dicts.

        LLMs occasionally pack multiple ops into a single entry in four forms:
          Form A: {"operation": ["signal", "wait_for_signal"], "params": [{...}, {...}], "parallel_group": 6}
          Form B: {"operation": [{"operation": "signal", "params": {...}}, ...], "parallel_group": 6}
          Form C: {"parallel_group": 6, "operations": [{"operation": "signal", "params": {...}}, ...]}
          Form D: {"operation": ["signal", ...], "parallel_group": 6, "ops": [{...}, ...]}
        All are expanded so the rest of the pipeline sees normal single-op commands.
        """
        _SUB_LIST_KEYS = ("operations", "ops", "commands", "steps")

        expanded = []
        for cmd in commands:
            op = cmd.get("operation", "")
            params = cmd.get("params", {})
            shared = {
                k: v
                for k, v in cmd.items()
                if k not in ("operation", "params", *_SUB_LIST_KEYS)
            }

            # Form C/D: sub-op list under "operations", "ops", "commands", or "steps"
            sub_list_key = next(
                (k for k in _SUB_LIST_KEYS if isinstance(cmd.get(k), list)), None
            )
            if sub_list_key and isinstance(cmd.get(sub_list_key), list):
                for sub in cmd[sub_list_key]:
                    new_cmd = {**shared}
                    new_cmd["operation"] = sub.get("operation", "")
                    new_cmd["params"] = sub.get("params", {})
                    expanded.append(new_cmd)
            elif isinstance(op, list) and len(op) > 0:
                if isinstance(params, list) and len(params) == len(op):
                    # Form A: parallel string names + matching params list
                    for o, p in zip(op, params):
                        # guard: if p is itself a sub-op dict, unwrap its params
                        if isinstance(p, dict) and "params" in p:
                            p = p["params"]
                        expanded.append({**shared, "operation": o, "params": p})
                elif all(isinstance(o, dict) for o in op):
                    # Form B: list of sub-operation dicts, each with own "operation"/"params"
                    for sub in op:
                        new_cmd = {**shared}
                        new_cmd["operation"] = sub.get("operation", "")
                        new_cmd["params"] = sub.get("params", {})
                        expanded.append(new_cmd)
                else:
                    expanded.append(cmd)
            else:
                expanded.append(cmd)
        return expanded

    def _flatten_plan(self, plan) -> List[Dict]:
        """Normalize LLM plan shapes into a flat list of command dicts.

        Handles:
          A) flat list of op dicts (with optional parallel_group)
          B) grouped: [{parallel_group, operations:[...]}, ...]
          C) nested dict: {commands:[...], parallel_groups:[{group, commands:[...]}, ...]}
        """
        flat: List[Dict] = []
        parallel_groups_items: List[Dict] = []

        if isinstance(plan, dict):
            parallel_groups_items = plan.get("parallel_groups") or []
            plan = plan.get("commands") or plan.get("operations") or []

        for item in plan:
            sub_ops = item.get("operations") or item.get("commands")
            if sub_ops:
                pg = item.get("parallel_group")
                for op in sub_ops:
                    op = dict(op)
                    if pg is not None:
                        op["parallel_group"] = pg
                    flat.append(op)
            else:
                flat.append(item)

        for group_item in parallel_groups_items:
            pg = group_item.get("group") or group_item.get("parallel_group")
            for op in group_item.get("commands") or group_item.get("operations") or []:
                op = dict(op)
                if pg is not None:
                    op["parallel_group"] = pg
                flat.append(op)

        return flat

    def _validate_commands(
        self, commands: List[Dict], default_robot_id: str
    ) -> List[Dict]:
        commands = self._expand_array_operations(commands)
        validated = []
        for cmd in commands:
            operation = cmd.get("operation", "")
            if isinstance(operation, list):
                operation = operation[0] if operation else ""
            params = cmd.get("params", {})

            # Ensure robot_id is present (use "robot" field if specified in multi-robot plan)
            if "robot_id" not in params:
                params["robot_id"] = cmd.get("robot", default_robot_id)

            # Reject unknown robot IDs
            from config.Robot import ROBOT_BASE_POSITIONS as _KNOWN_ROBOTS

            robot_id_in_cmd = params.get("robot_id", default_robot_id)
            if robot_id_in_cmd not in _KNOWN_ROBOTS:
                logger.warning(
                    "Unknown robot_id '%s', skipping command '%s'",
                    robot_id_in_cmd,
                    operation,
                )
                continue

            # Verify operation exists
            op = self.registry.get_operation_by_name(operation)
            if op is None:
                logger.warning(
                    "Unknown operation '%s' (params=%s), full cmd=%s, skipping",
                    operation,
                    list(params.keys()),
                    cmd,
                )
                continue

            validated_cmd = {"operation": operation, "params": params}

            # Promote capture_var to top level (LLM sometimes puts it inside params)
            if "capture_var" in cmd:
                validated_cmd["capture_var"] = cmd["capture_var"]
            elif "capture_var" in params:
                validated_cmd["capture_var"] = params.pop("capture_var")

            # Preserve parallel_group if present (for multi-robot coordination)
            if "parallel_group" in cmd:
                validated_cmd["parallel_group"] = cmd["parallel_group"]

            validated.append(validated_cmd)

        # Fix intra-group variable dependencies: if operation B uses $var captured by
        # operation A and both are in the same parallel_group, move B to a later group.
        validated = self._fix_intra_group_dependencies(validated)

        # Validate multi-robot plans (signal/wait pairs and variable usage).
        # Run the same capture_var inference the executor will apply so that
        # missing capture_var on perception ops doesn't generate false-positive warnings.
        if len(validated) > 1:
            validated = self._infer_capture_vars_for_validation(validated)
            is_valid, errors = self._validate_multi_robot_plan(validated)
            if not is_valid:
                for error in errors:
                    logger.warning(f"Multi-robot plan validation warning: {error}")

        return validated

    def _infer_capture_vars_for_validation(self, commands: List[Dict]) -> List[Dict]:
        """Backfill missing capture_var on perception ops so the validator sees correct definitions.

        Mirrors SequenceExecutor._infer_capture_vars — keep in sync if the logic changes.
        """
        from operations.Base import OperationCategory

        commands = [dict(cmd) for cmd in commands]
        for i, cmd in enumerate(commands):
            if cmd.get("capture_var"):
                continue
            op_name = cmd.get("operation", "")
            op_def = self.registry.get_operation_by_name(op_name)
            if op_def is None or op_def.category != OperationCategory.PERCEPTION:
                continue
            for later_cmd in commands[i + 1 :]:
                for v in later_cmd.get("params", {}).values():
                    if isinstance(v, str) and v.startswith("$"):
                        var_name = v.lstrip("$").split(".")[0]
                        cmd["capture_var"] = var_name
                        break
                if cmd.get("capture_var"):
                    break
        return commands

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

        # Renumber groups to be contiguous, starting AFTER sequential (ungrouped)
        # commands. The executor assigns those groups 0, 1, 2, … in order, so
        # renumbering from 1 causes parallel groups to collide with them.
        n_sequential = sum(1 for cmd in commands if "parallel_group" not in cmd)
        groups_sorted = sorted(
            set(cmd["parallel_group"] for cmd in commands if "parallel_group" in cmd)
        )
        remap = {old: new for new, old in enumerate(groups_sorted, start=n_sequential)}
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
