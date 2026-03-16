#!/usr/bin/env python3
"""
Command Parser for Multi-Command Sequences
==========================================

This module parses natural language compound commands into structured operation sequences using an LLM for intelligent parsing with operation registry validation.

Example:
    >>> parser = CommandParser()
    >>> result = parser.parse("move to (0.3, 0.2, 0.1) and close the gripper", robot_id="Robot1")
    >>> print(result)
    {
        "success": True,
        "commands": [
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "x": 0.3, "y": 0.2, "z": 0.1}},
            {"operation": "control_gripper", "params": {"robot_id": "Robot1", "open_gripper": False}}
        ]
    }
"""

from typing import Dict, Any, List, Optional, Tuple
import json
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
    )
    from ..operations.WorkflowPatterns import WorkflowPatternRegistry, WorkflowPattern
except ImportError:
    from rag import RAGSystem
    from config.Servers import (
        LMSTUDIO_BASE_URL,
        DEFAULT_LMSTUDIO_MODEL,
        DEFAULT_TEMPERATURE,
        LLM_REQUEST_TIMEOUT,
    )
    from operations.WorkflowPatterns import WorkflowPatternRegistry, WorkflowPattern

# Configure logging
from core.LoggingSetup import setup_logging

setup_logging(__name__)
logger = logging.getLogger(__name__)


class _PromptBuilder:
    """
    Assembles LLM parsing prompts.

    Separated from CommandParser to allow unit testing of prompt construction
    without needing a full CommandParser instance (LLM URL, RAG system, etc.).

    Args:
        registry: OperationRegistry with all available operations.
        workflow_registry: WorkflowPatternRegistry with workflow patterns.
        rag: Optional RAGSystem for semantic operation retrieval.
    """

    def __init__(self, registry, workflow_registry, rag):
        """Initialise the builder with registry references."""
        self.registry = registry
        self.workflow_registry = workflow_registry
        self.rag = rag

    def build(
        self,
        command_text: str,
        robot_id: str,
        anti_pattern_section: str = "",
        spatial_section: str = "",
    ) -> str:
        """
        Build the full LLM parsing prompt.

        Args:
            command_text: Natural language command to parse.
            robot_id: Default robot ID for the command.
            anti_pattern_section: Formatted anti-pattern warning block (may be empty).
            spatial_section: Formatted knowledge-graph spatial context (may be empty).

        Returns:
            Complete prompt string ready to send to the LLM.
        """
        available_ops = self.get_available_operations_summary(command_text)
        anti_pattern_block = (
            f"\n        {anti_pattern_section}\n" if anti_pattern_section else ""
        )
        spatial_block = f"\n        {spatial_section}\n" if spatial_section else ""

        return f"""You are a robot coordinator planning tasks for multiple robots.

        Available Robots:
        - Robot1 (left workspace, near x=-0.4)
        - Robot2 (right workspace, near x=0.4)

        Available Operations:
        {available_ops}

        Command to parse: "{command_text}"
        Default robot_id: "{robot_id}"

        === MULTI-ROBOT COORDINATION ===

        When the task involves multiple robots:
        1. Decide which robot is best positioned for each subtask
        2. Use "parallel_group" to mark operations that can run concurrently
        3. Use synchronization primitives (signal, wait_for_signal) for robot-to-robot coordination
        4. Operations with the SAME parallel_group number execute in parallel
        5. Operations in LATER parallel_groups wait for ALL operations in previous groups to complete

        Example for multi-robot handoff:
        {{
        "reasoning": "Robot1 detects red cube, moves to it, and grips. Robot2 waits for signal, then both move to handoff position. Robot2 grips, then Robot1 releases.",
        "plan": [
            {{"parallel_group": 1, "robot": "Robot1", "operation": "detect_object_stereo", "params": {{"robot_id": "Robot1", "color": "red"}}, "capture_var": "target"}},
            {{"parallel_group": 2, "robot": "Robot1", "operation": "move_to_coordinate", "params": {{"robot_id": "Robot1", "position": "$target"}}}},
            {{"parallel_group": 3, "robot": "Robot1", "operation": "control_gripper", "params": {{"robot_id": "Robot1", "open_gripper": false}}}},
            {{"parallel_group": 3, "robot": "Robot1", "operation": "signal", "params": {{"event_name": "r1_gripped"}}}},
            {{"parallel_group": 3, "robot": "Robot2", "operation": "wait_for_signal", "params": {{"event_name": "r1_gripped"}}}},
            {{"parallel_group": 4, "robot": "Robot1", "operation": "move_to_coordinate", "params": {{"robot_id": "Robot1", "x": 0.0, "y": 0.3, "z": 0.15}}}},
            {{"parallel_group": 4, "robot": "Robot2", "operation": "move_to_coordinate", "params": {{"robot_id": "Robot2", "x": 0.0, "y": 0.3, "z": 0.15}}}},
            {{"parallel_group": 5, "robot": "Robot2", "operation": "control_gripper", "params": {{"robot_id": "Robot2", "open_gripper": false}}}},
            {{"parallel_group": 6, "robot": "Robot1", "operation": "control_gripper", "params": {{"robot_id": "Robot1", "open_gripper": true}}}}
        ]
        }}

        === SYNCHRONIZATION PRIMITIVES ===

        - signal(event_name): Emit named event for other robots to wait on
        * Example: {{"operation": "signal", "params": {{"event_name": "cube_gripped"}}}}

        - wait_for_signal(event_name, timeout_ms): Wait for event (default timeout: 30000ms)
        * Example: {{"operation": "wait_for_signal", "params": {{"event_name": "cube_gripped"}}}}

        - wait(duration_ms): Simple time-based pause
        * Example: {{"operation": "wait", "params": {{"duration_ms": 500}}}}

        === SINGLE-ROBOT RULES ===

        1. Extract each distinct action as a separate operation
        2. Parse coordinates from text like "(0.3, 0.2, 0.1)" or "x=0.3, y=0.2, z=0.1"
        3. "close gripper" or "grasp" means control_gripper with open_gripper=false
        4. "open gripper" or "release" means control_gripper with open_gripper=true
        5. Include robot_id in every operation's params
        6. Preserve the order of operations as specified in the command

        === VARIABLE PASSING ===

        CRITICAL: Variables must be DEFINED before they are USED!
        - Use "capture_var": "target" on detect_object_stereo to store the result
        - Use "$target" in LATER operations to reference the stored result
        - NEVER use a $variable before it has been captured by a previous operation
{spatial_block}{anti_pattern_block}Output only valid JSON, no explanation, no comments."""

    def get_available_operations_summary(self, command_text: str = "") -> str:
        """
        Get a summary of available operations for the LLM prompt.

        If RAG is available and command_text is provided, uses semantic search
        to prioritize the most relevant operations and workflow patterns.

        Args:
            command_text: The command being parsed (for RAG context)

        Returns:
            Formatted string of operations and workflow patterns for LLM prompt
        """
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
                    summary_lines.append("\n=== OTHER AVAILABLE OPERATIONS ===")

                for op in self.registry.get_all_operations():
                    if op.name not in relevant_ops:
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
        """
        Format operation parameters for LLM prompt, including valid values.

        Args:
            parameters: List of OperationParameter objects

        Returns:
            Formatted parameter string
        """
        param_strs = []
        for p in parameters:
            param_str = f"{p.name}: {p.type}"
            if hasattr(p, "valid_values") and p.valid_values:
                valid_vals = ", ".join([f"'{v}'" for v in p.valid_values])
                param_str += f" (valid: {valid_vals})"
            param_strs.append(param_str)
        return ", ".join(param_strs)

    def format_workflow_pattern(self, pattern: WorkflowPattern) -> str:
        """
        Format a workflow pattern for inclusion in LLM prompt.

        Args:
            pattern: WorkflowPattern to format

        Returns:
            Formatted pattern description for LLM
        """
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
    """
    Parses compound natural language commands into structured operation sequences.

    Uses LLM for intelligent parsing with operation registry validation.
    Falls back to regex patterns when LLM is unavailable.
    """

    def __init__(
        self,
        lm_studio_url: Optional[str] = None,
        model: Optional[str] = None,
        use_rag: bool = True,
        use_cache: bool = True,
    ):
        """
        Initialize the CommandParser.

        Args:
            lm_studio_url: LM Studio base URL (default from config)
            model: Model name for parsing (default from config)
            use_rag: Whether to use RAG for semantic operation context
        """
        # Import from centralized lazy import system (prevents circular dependencies)
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

        # Initialize LRU Cache for parsed commands
        if self.use_cache:
            self._parse_cache = functools.lru_cache(maxsize=128)(self._do_llm_request)
        else:
            self._parse_cache = self._do_llm_request

        # Optional FeedbackCollector for self-improvement anti-pattern warnings.
        # Set to a FeedbackCollector instance to enable; None disables the feature.
        self.feedback_collector = None

        # Initialize RAG system for semantic operation search
        self.rag = None
        if use_rag:
            try:
                self.rag = RAGSystem()
                # Provide control over index rebuilding to speed up startups
                self.rag.index_operations(rebuild=False)
                logger.info(
                    "RAG system initialized for command parsing (using existing index if present)"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize RAG: {e}. Using registry only.")

        # Prompt builder — separated for testability
        self._prompt_builder = _PromptBuilder(
            self.registry, self.workflow_registry, self.rag
        )

    def parse(
        self, command_text: str, robot_id: str = "Robot1", use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Parse a compound command into a sequence of operations.

        Args:
            command_text: Natural language command (e.g., "move to (0.3, 0.2, 0.1) and close gripper")
            robot_id: Default robot ID to use for operations
            use_llm: Whether to use LLM parsing (falls back to regex if False or LLM unavailable)

        Returns:
            Dict with structure:
            {
                "success": bool,
                "commands": [
                    {"operation": str, "params": dict},
                    ...
                ],
                "error": str or None
            }
        """
        if not command_text or not command_text.strip():
            return {"success": False, "commands": [], "error": "Empty command text"}

        # Try LLM parsing first
        if use_llm:
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

        # If both LLM and regex failed, try generating a new operation
        generated = self._try_generate_operation(command_text, robot_id)
        if generated:
            return generated

        return regex_result

    def _parse_with_llm(self, command_text: str, robot_id: str) -> Dict[str, Any]:
        """
        Parse command using LLM for intelligent understanding.

        Args:
            command_text: Natural language command
            robot_id: Default robot ID

        Returns:
            Parsed command structure
        """
        # Build prompt for LLM using _PromptBuilder
        anti_pattern_section = self._get_anti_pattern_warnings(command_text)
        spatial_section = self._get_spatial_context(robot_id)
        prompt = self._prompt_builder.build(
            command_text,
            robot_id,
            anti_pattern_section=anti_pattern_section,
            spatial_section=spatial_section,
        )

        try:
            # Use cached or direct request depending on initialization
            result = self._parse_cache(prompt, command_text)

            if not result.get("success"):
                return result

            parsed = result["parsed"]
            content = result["content"]

            # Normalize multi-robot "plan" format to "commands" format
            if "plan" in parsed and "commands" not in parsed:
                logger.info(
                    f"Multi-robot plan detected with reasoning: {parsed.get('reasoning', 'N/A')}"
                )
                parsed["commands"] = parsed["plan"]

            logger.info(f"Parsed {len(parsed.get('commands', []))} commands from LLM")

            # Validate operations
            commands = parsed.get("commands", [])
            validated_commands = self._validate_commands(commands, robot_id)
            logger.info(f"Validated {len(validated_commands)} commands")

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

    def _do_llm_request(self, prompt: str, command_text: str) -> Dict[str, Any]:
        """Actual HTTP request to LLM, separated for caching purposes."""
        try:
            response = self._session.post(
                f"{self.lm_studio_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a robot command parser. Map natural language variations to valid parameter values (e.g., 'leftmost' → 'left', 'rightmost' → 'right', 'nearest' → 'closest'). Output only valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": DEFAULT_TEMPERATURE,  # Low temperature for deterministic parsing
                    "max_tokens": 5000,  # Increased for multi-robot coordination (was 1000)
                },
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

            # Catch array responses
            if isinstance(parsed, list):
                logger.info(f"LLM returned an array directly, wrapping in dict")
                parsed = {"commands": parsed}

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
        except Exception as e:
            raise

    def _get_anti_pattern_warnings(self, command_text: str) -> str:
        """
        Retrieve formatted anti-pattern warnings from FeedbackCollector.

        Returns an empty string when FeedbackCollector is unavailable or no
        high-failure patterns exist so the prompt remains clean by default.

        Args:
            command_text: The command being parsed (passed through for future
                context-sensitive filtering).

        Returns:
            Formatted warning block string, or empty string if no warnings.
        """
        try:
            from config.Memory import MEMORY_ENABLED

            if not MEMORY_ENABLED:
                return ""
            from agents.FeedbackCollector import get_feedback_collector

            return get_feedback_collector().get_anti_pattern_warnings(command_text)
        except Exception as e:
            logger.debug(f"FeedbackCollector unavailable: {e}")
            return ""

    def _get_spatial_context(self, robot_id: str) -> str:
        """
        Retrieve formatted spatial context from the Knowledge Graph for the given robot.

        Queries reachable objects and nearby robots to enrich the LLM prompt with live spatial awareness. Returns an empty string if the KG is disabled, unavailable, or raises any exception (graceful degrade).

        Args:
            robot_id: The robot whose spatial context to retrieve.

        Returns:
            A formatted spatial context block string, or empty string if unavailable.
        """
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

            if len(lines) == 1:
                return ""  # Only header, no data

            return "\n        ".join(lines)

        except Exception as e:
            logger.debug(f"KG spatial context unavailable: {e}")
            return ""

    def _build_parsing_prompt(
        self, command_text: str, robot_id: str, available_ops: str
    ) -> str:
        """Build the prompt for LLM command parsing. Delegates to _PromptBuilder."""
        anti_pattern_section = self._get_anti_pattern_warnings(command_text)
        spatial_section = self._get_spatial_context(robot_id)
        return self._prompt_builder.build(
            command_text,
            robot_id,
            anti_pattern_section=anti_pattern_section,
            spatial_section=spatial_section,
        )

    def _get_available_operations_summary(self, command_text: str = "") -> str:
        """Get operations summary for LLM prompt. Delegates to _PromptBuilder."""
        return self._prompt_builder.get_available_operations_summary(command_text)

    def _format_parameters(self, parameters: List) -> str:
        """Format parameters for LLM prompt. Delegates to _PromptBuilder."""
        return self._prompt_builder.format_parameters(parameters)

    def _format_workflow_pattern(self, pattern: WorkflowPattern) -> str:
        """Format workflow pattern for LLM prompt. Delegates to _PromptBuilder."""
        return self._prompt_builder.format_workflow_pattern(pattern)

    def _extract_json_from_response(self, content: str) -> Optional[Dict]:
        """Extract JSON object from LLM response text."""
        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parse failed: {e}")

        # Try to find JSON in markdown code block
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Markdown JSON parse failed: {e}. Content length: {len(json_str)}, preview: {json_str[:200]}"
                )
            # Retry after stripping JS-style // comments (LLMs often emit these)
            json_str_clean = re.sub(r"//[^\n]*", "", json_str)
            try:
                return json.loads(json_str_clean)
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in text
        json_match = re.search(r"\{.*?\}", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Regex JSON parse failed: {e}. Content length: {len(json_str)}"
                )
            # Retry after stripping JS-style // comments
            json_str_clean = re.sub(r"//[^\n]*", "", json_str)
            try:
                return json.loads(json_str_clean)
            except json.JSONDecodeError:
                pass

        logger.error(
            f"All JSON extraction methods failed. Response length: {len(content)}, preview: {content[:500]}"
        )
        return None

    def _validate_commands(
        self, commands: List[Dict], default_robot_id: str
    ) -> List[Dict]:
        """
        Validate and normalize parsed commands.

        Args:
            commands: List of parsed commands
            default_robot_id: Default robot ID to use

        Returns:
            Validated and normalized commands
        """
        validated = []
        for cmd in commands:
            operation = cmd.get("operation", "")
            params = cmd.get("params", {})

            # Ensure robot_id is present (use "robot" field if specified in multi-robot plan)
            if "robot_id" not in params:
                params["robot_id"] = cmd.get("robot", default_robot_id)

            # Verify operation exists
            op = self.registry.get_operation_by_name(operation)
            if op is None:
                logger.warning(f"Unknown operation: {operation}, skipping")
                continue

            validated_cmd = {"operation": operation, "params": params}

            # Preserve capture_var if present
            if "capture_var" in cmd:
                validated_cmd["capture_var"] = cmd["capture_var"]

            # Preserve parallel_group if present (for multi-robot coordination)
            if "parallel_group" in cmd:
                validated_cmd["parallel_group"] = cmd["parallel_group"]

            validated.append(validated_cmd)

        # Validate multi-robot plans (signal/wait pairs and variable usage)
        if len(validated) > 1:
            is_valid, errors = self._validate_multi_robot_plan(validated)
            if not is_valid:
                for error in errors:
                    logger.warning(f"Multi-robot plan validation warning: {error}")

        return validated

    def _validate_multi_robot_plan(
        self, commands: List[Dict]
    ) -> Tuple[bool, List[str]]:
        """
        Validate signal/wait pairs and variable definitions in multi-robot plans.

        Args:
            commands: List of validated commands

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
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

    def _check_generation_needed(self, rag_results: List[Dict]) -> Tuple[bool, str]:
        """
        Check if RAG scores are too low, indicating no good operation match exists.

        Args:
            rag_results: Results from RAG search

        Returns:
            Tuple of (should_generate, reason)
        """
        try:
            from config.DynamicOperations import (
                ENABLE_DYNAMIC_OPERATIONS,
                GENERATION_TRIGGER_THRESHOLD,
            )
        except ImportError:
            return False, "Dynamic operations config not available"

        if not ENABLE_DYNAMIC_OPERATIONS:
            return False, "Dynamic operations disabled"

        if not rag_results:
            return True, "No RAG results found"

        # Check best score
        best_score = max(
            r.get("similarity_score", r.get("score", 0)) for r in rag_results
        )

        if best_score < GENERATION_TRIGGER_THRESHOLD:
            return (
                True,
                f"Best RAG score {best_score:.2f} below threshold {GENERATION_TRIGGER_THRESHOLD}",
            )

        return False, ""

    def _try_generate_operation(
        self, command_text: str, robot_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt to generate a new operation when no existing operation matches.

        Args:
            command_text: The command that could not be parsed
            robot_id: Default robot ID

        Returns:
            Parsed command structure if generation succeeded and retry worked, None otherwise
        """
        # Check if RAG scores warrant generation
        if self.rag:
            try:
                rag_results = self.rag.search(command_text, top_k=3)
                should_generate, reason = self._check_generation_needed(rag_results)
                if not should_generate:
                    return None
                logger.info(f"Dynamic operation generation triggered: {reason}")
            except Exception as e:
                logger.warning(f"RAG check for generation failed: {e}")
                return None
        else:
            return None

        # Generate the operation
        try:
            from operations.Generator import OperationGenerator

            generator = OperationGenerator()
            success, message, file_path = generator.generate_operation(
                command_text,
                context={"robot_id": robot_id},
            )

            if not success:
                logger.warning(f"Operation generation failed: {message}")
                return None

            logger.info(f"Operation generated: {message}")

            # When review is required the operation is PENDING — it is not registered
            # yet, so retrying the LLM parse would fail or match the wrong operation.
            # Return a structured pending-review payload instead.
            try:
                from config.DynamicOperations import REQUIRE_USER_REVIEW
            except ImportError:
                REQUIRE_USER_REVIEW = True

            if REQUIRE_USER_REVIEW:
                logger.info(
                    f"Operation pending human review before activation: {file_path}"
                )
                return {
                    "success": False,
                    "pending_review": True,
                    "file_path": file_path,
                    "message": (
                        "A new operation was generated but requires human review "
                        "before it can be used. Approve it with: "
                        "python -m tools.ReviewOperations approve <id>"
                    ),
                }

            # Retry parsing with the new operation now registered and active
            result = self._parse_with_llm(command_text, robot_id)
            if result["success"]:
                result["generated_operation"] = file_path
                return result

            return None

        except Exception as e:
            logger.warning(f"Dynamic operation generation error: {e}")
            return None

    def _parse_with_regex(self, command_text: str, robot_id: str) -> Dict[str, Any]:
        """
        Fallback regex-based parsing for common command patterns.

        Args:
            command_text: Natural language command
            robot_id: Default robot ID

        Returns:
            Parsed command structure
        """
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
                if color in ["red", "green", "blue"]:
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

            # Parse simple object detection (2D pixel coordinates)
            if re.search(
                r"detect\s+(?:objects?|cubes?)|find\s+(?:objects?|cubes?)|look\s+for|scan\s+for|locate\s+(?:objects?|cubes?)",
                part,
            ):
                commands.append(
                    {
                        "operation": "detect_objects",
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

            # Parse move_from_a_to_b (e.g. "move Robot1 from X Y Z to X Y Z")
            move_ab_match = re.search(
                r"move\s+(?:\w+\s+)?from\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+to\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)",
                part,
            )
            if move_ab_match:
                g = move_ab_match.groups()
                commands.append(
                    {
                        "operation": "move_from_a_to_b",
                        "params": {
                            "robot_id": robot_id,
                            "point_a": {
                                "x": float(g[0]),
                                "y": float(g[1]),
                                "z": float(g[2]),
                            },
                            "point_b": {
                                "x": float(g[3]),
                                "y": float(g[4]),
                                "z": float(g[5]),
                            },
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
        """Get list of supported regex patterns for documentation."""
        return [
            "move to (x, y, z) - Move robot to coordinates",
            "move to x=0.3, y=0.2, z=0.1 - Move robot to coordinates",
            "close gripper / grasp / grip / grab - Close the gripper",
            "open gripper / release / drop - Open the gripper",
            "check status / get status - Get robot status",
            "return to start / go home / home position - Return to start position",
            "detect objects / find cubes / scan for - Detect objects (2D)",
            "detect with depth / find 3d positions / detect stereo - Detect objects with 3D positions",
            "Commands can be chained with 'and', 'then', 'after that', or commas",
        ]


# Singleton instance
_parser_instance: Optional[CommandParser] = None


def get_command_parser() -> CommandParser:
    """
    Get the global CommandParser singleton.

    Returns:
        The global CommandParser instance
    """
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = CommandParser()
    return _parser_instance
