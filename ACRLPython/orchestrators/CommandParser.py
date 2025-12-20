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

from typing import Dict, Any, List, Optional
import json
import re
import logging
import requests

# Handle both direct execution and package import
try:
    from ..rag import RAGSystem
    # Lazy import to avoid circular dependency
    # from ..operations.Registry import get_global_registry
    from .. import LLMConfig as cfg
except ImportError:
    from rag import RAGSystem
    # from operations.Registry import get_global_registry
    import LLMConfig as cfg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    ):
        """
        Initialize the CommandParser.

        Args:
            lm_studio_url: LM Studio base URL (default from config)
            model: Model name for parsing (default from config)
            use_rag: Whether to use RAG for semantic operation context
        """
        # Lazy import to avoid circular dependency
        from operations.Registry import get_global_registry

        self.lm_studio_url = lm_studio_url or cfg.LMSTUDIO_BASE_URL
        self.model = model or cfg.DEFAULT_LMSTUDIO_MODEL
        self.registry = get_global_registry()

        # Initialize RAG system for semantic operation search
        self.rag = None
        if use_rag:
            try:
                self.rag = RAGSystem()
                # Always rebuild index on startup to prevent dimension mismatches
                self.rag.index_operations(rebuild=True)
                logger.info(
                    "RAG system initialized for command parsing (index rebuilt)"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize RAG: {e}. Using registry only.")

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
        return self._parse_with_regex(command_text, robot_id)

    def _parse_with_llm(self, command_text: str, robot_id: str) -> Dict[str, Any]:
        """
        Parse command using LLM for intelligent understanding.

        Args:
            command_text: Natural language command
            robot_id: Default robot ID

        Returns:
            Parsed command structure
        """
        # Get available operations for the prompt (use RAG for semantic context)
        available_ops = self._get_available_operations_summary(command_text)

        # Build prompt for LLM
        prompt = self._build_parsing_prompt(command_text, robot_id, available_ops)

        try:
            # Call LM Studio
            response = requests.post(
                f"{self.lm_studio_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a robot command parser. Output only valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,  # Low temperature for deterministic parsing
                    "max_tokens": 1000,
                },
                timeout=90,
            )

            if response.status_code != 200:
                return {
                    "success": False,
                    "commands": [],
                    "error": f"LLM request failed with status {response.status_code}",
                }

            # Extract content from response
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            logger.info(f"LLM response: {content[:500]}")

            # Parse JSON from response
            parsed = self._extract_json_from_response(content)
            if not parsed:
                return {
                    "success": False,
                    "commands": [],
                    "error": f"Failed to extract JSON from LLM response: {content[:200]}",
                }

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

    def _build_parsing_prompt(
        self, command_text: str, robot_id: str, available_ops: str
    ) -> str:
        """Build the prompt for LLM command parsing with multi-robot support."""
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
        "reasoning": "Robot1 detects and picks up cube. Robot2 waits for signal, then both move to handoff position. Robot2 grips, then Robot1 releases.",
        "plan": [
            {{"parallel_group": 1, "robot": "Robot1", "operation": "detect_object_stereo", "params": {{"robot_id": "Robot1", "color": "red"}}, "capture_var": "cube"}},
            {{"parallel_group": 2, "robot": "Robot1", "operation": "move_to_coordinate", "params": {{"robot_id": "Robot1", "position": "$cube"}}}},
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

        When detect_object is followed by "move to it/there/that":
        - Add "capture_var": "target" to the detect_object command
        - Use "position": "$target" in the move_to_coordinate params (NOT x, y, z)

        Example:
        {{
        "plan": [
            {{"operation": "detect_object_stereo", "params": {{"robot_id": "Robot1", "color": "blue"}}, "capture_var": "target"}},
            {{"operation": "move_to_coordinate", "params": {{"robot_id": "Robot1", "position": "$target"}}}},
            {{"operation": "control_gripper", "params": {{"robot_id": "Robot1", "open_gripper": false}}}}
        ]
        }}

        === DETECTION CONSTRAINTS ===

        - detect_object_stereo "color": ONLY "red", "green", "blue", or null
        - detect_object_stereo "selection": ONLY "leftmost", "closest", "first", or "all"

        === OUTPUT FORMAT ===

        For single-robot tasks:
        {{
        "commands": [
            {{"operation": "operation_name", "params": {{"robot_id": "Robot1", "param1": value1}}}}
        ]
        }}

        For multi-robot tasks:
        {{
        "reasoning": "Brief explanation of the multi-robot coordination strategy",
        "plan": [
            {{"parallel_group": 1, "robot": "Robot1", "operation": "...", "params": {{...}}}},
            {{"parallel_group": 1, "robot": "Robot2", "operation": "...", "params": {{...}}}}
        ]
        }}

        Output only valid JSON, no explanation."""

    def _get_available_operations_summary(self, command_text: str = "") -> str:
        """
        Get a summary of available operations for the LLM prompt.

        If RAG is available and command_text is provided, uses semantic search
        to prioritize the most relevant operations for the given command.

        Args:
            command_text: The command being parsed (for RAG context)

        Returns:
            Formatted string of operations for LLM prompt
        """
        # If RAG is available, get semantically relevant operations first
        if self.rag and command_text:
            try:
                # Search for relevant operations
                rag_results = self.rag.search(command_text, top_k=5)
                relevant_ops = set()

                summary_lines = []

                # Add RAG-matched operations first (most relevant)
                if rag_results:
                    summary_lines.append("Most relevant operations for this command:")
                    for result in rag_results:
                        op = self.registry.get_operation_by_name(result.get("name", ""))
                        if op:
                            relevant_ops.add(op.name)
                            params = self._format_parameters(op.parameters)
                            score = result.get("similarity_score", 0)
                            summary_lines.append(
                                f"- {op.name}({params}): {op.description} [relevance: {score:.2f}]"
                            )

                    summary_lines.append("\nOther available operations:")

                # Add remaining operations
                for op in self.registry.get_all_operations():
                    if op.name not in relevant_ops:
                        params = self._format_parameters(op.parameters)
                        summary_lines.append(f"- {op.name}({params}): {op.description}")

                return "\n".join(summary_lines)

            except Exception as e:
                logger.warning(f"RAG search failed, using registry: {e}")

        # Fallback: return all operations from registry
        ops = self.registry.get_all_operations()
        summary_lines = []
        for op in ops:
            params = self._format_parameters(op.parameters)
            summary_lines.append(f"- {op.name}({params}): {op.description}")
        return "\n".join(summary_lines)

    def _format_parameters(self, parameters: List) -> str:
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
            # Add valid values if specified
            if hasattr(p, "valid_values") and p.valid_values:
                valid_vals = ", ".join([f"'{v}'" for v in p.valid_values])
                param_str += f" (valid: {valid_vals})"
            param_strs.append(param_str)
        return ", ".join(param_strs)

    def _extract_json_from_response(self, content: str) -> Optional[Dict]:
        """Extract JSON object from LLM response text."""
        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code block
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in text
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

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

        return validated

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
                r"move\s+(?:to\s+)?(?:\(?\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)?|"
                r"x\s*=?\s*(-?[\d.]+).*?y\s*=?\s*(-?[\d.]+).*?z\s*=?\s*(-?[\d.]+))",
                part,
            )
            if move_match:
                groups = move_match.groups()
                if groups[0] is not None:
                    x, y, z = float(groups[0]), float(groups[1]), float(groups[2])
                else:
                    x, y, z = float(groups[3]), float(groups[4]), float(groups[5])

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

            # Parse return to start position
            if re.search(
                r"return\s+(?:to\s+)?(?:start|home|default|initial)\s*(?:position)?|go\s+(?:to\s+)?home|home\s+position",
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
