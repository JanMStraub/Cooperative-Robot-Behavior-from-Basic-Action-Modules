"""
Command Parser for Multi-Command Sequences
==========================================

This module parses natural language compound commands into structured operation sequences
using an LLM for intelligent parsing and RAG for operation validation.

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

from rag import RAGSystem
from operations.Registry import get_global_registry
import LLMConfig as cfg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CommandParser:
    """
    Parses compound natural language commands into structured operation sequences.

    Uses LLM for intelligent parsing and RAG for operation validation.
    Falls back to regex patterns when LLM is unavailable.
    """

    def __init__(
        self,
        lm_studio_url: Optional[str] = None,
        model: Optional[str] = None,
        use_rag_validation: bool = True
    ):
        """
        Initialize the CommandParser.

        Args:
            lm_studio_url: LM Studio base URL (default from config)
            model: Model name for parsing (default from config)
            use_rag_validation: Validate operations against RAG system
        """
        self.lm_studio_url = lm_studio_url or cfg.LMSTUDIO_BASE_URL
        self.model = model or cfg.DEFAULT_LMSTUDIO_MODEL
        self.use_rag_validation = use_rag_validation

        # Initialize RAG for operation validation
        if use_rag_validation:
            try:
                self.rag = RAGSystem()
                self.registry = get_global_registry()
                logger.info("✓ CommandParser initialized with RAG validation")
            except Exception as e:
                logger.warning(f"Failed to initialize RAG: {e}. Validation disabled.")
                self.rag = None
                self.registry = get_global_registry()
        else:
            self.rag = None
            self.registry = get_global_registry()

    def parse(
        self,
        command_text: str,
        robot_id: str = "Robot1",
        use_llm: bool = True
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
            return {
                "success": False,
                "commands": [],
                "error": "Empty command text"
            }

        # Try LLM parsing first
        if use_llm:
            result = self._parse_with_llm(command_text, robot_id)
            if result["success"]:
                return result
            logger.warning(f"LLM parsing failed: {result.get('error')}. Falling back to regex.")

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
        # Get available operations for the prompt
        available_ops = self._get_available_operations_summary()

        # Build prompt for LLM
        prompt = self._build_parsing_prompt(command_text, robot_id, available_ops)

        try:
            # Call LM Studio
            response = requests.post(
                f"{self.lm_studio_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a robot command parser. Output only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,  # Low temperature for deterministic parsing
                    "max_tokens": 1000
                },
                timeout=30
            )

            if response.status_code != 200:
                return {
                    "success": False,
                    "commands": [],
                    "error": f"LLM request failed with status {response.status_code}"
                }

            # Extract content from response
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Parse JSON from response
            parsed = self._extract_json_from_response(content)
            if not parsed:
                return {
                    "success": False,
                    "commands": [],
                    "error": f"Failed to extract JSON from LLM response: {content[:200]}"
                }

            # Validate operations
            commands = parsed.get("commands", [])
            validated_commands = self._validate_commands(commands, robot_id)

            return {
                "success": True,
                "commands": validated_commands,
                "error": None
            }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "commands": [],
                "error": "LLM request timed out"
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "commands": [],
                "error": f"Cannot connect to LM Studio at {self.lm_studio_url}"
            }
        except Exception as e:
            return {
                "success": False,
                "commands": [],
                "error": f"LLM parsing error: {str(e)}"
            }

    def _build_parsing_prompt(self, command_text: str, robot_id: str, available_ops: str) -> str:
        """Build the prompt for LLM command parsing."""
        return f"""Parse the following robot command into a sequence of operations.

Available operations:
{available_ops}

Command to parse: "{command_text}"
Default robot_id: "{robot_id}"

Rules:
1. Extract each distinct action as a separate operation
2. Parse coordinates from text like "(0.3, 0.2, 0.1)" or "x=0.3, y=0.2, z=0.1"
3. "close gripper" or "grasp" means control_gripper with open_gripper=false
4. "open gripper" or "release" means control_gripper with open_gripper=true
5. Include robot_id in every operation's params
6. Preserve the order of operations as specified in the command

Output JSON format:
{{
    "commands": [
        {{"operation": "operation_name", "params": {{"param1": value1, ...}}}},
        ...
    ]
}}

Output only the JSON, no explanation."""

    def _get_available_operations_summary(self) -> str:
        """Get a summary of available operations for the LLM prompt."""
        ops = self.registry.get_all_operations()
        summary_lines = []
        for op in ops:
            params = ", ".join([f"{p.name}: {p.type}" for p in op.parameters])
            summary_lines.append(f"- {op.name}({params}): {op.description}")
        return "\n".join(summary_lines)

    def _extract_json_from_response(self, content: str) -> Optional[Dict]:
        """Extract JSON object from LLM response text."""
        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code block
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in text
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _validate_commands(self, commands: List[Dict], default_robot_id: str) -> List[Dict]:
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

            # Ensure robot_id is present
            if "robot_id" not in params:
                params["robot_id"] = default_robot_id

            # Verify operation exists
            op = self.registry.get_operation_by_name(operation)
            if op is None:
                logger.warning(f"Unknown operation: {operation}, skipping")
                continue

            validated.append({
                "operation": operation,
                "params": params
            })

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
        parts = re.split(r'\s+(?:and|then|after that|,)\s+', text)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Parse move commands
            move_match = re.search(
                r'move\s+(?:to\s+)?(?:\(?\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)?|'
                r'x\s*=?\s*(-?[\d.]+).*?y\s*=?\s*(-?[\d.]+).*?z\s*=?\s*(-?[\d.]+))',
                part
            )
            if move_match:
                groups = move_match.groups()
                if groups[0] is not None:
                    x, y, z = float(groups[0]), float(groups[1]), float(groups[2])
                else:
                    x, y, z = float(groups[3]), float(groups[4]), float(groups[5])

                commands.append({
                    "operation": "move_to_coordinate",
                    "params": {
                        "robot_id": robot_id,
                        "x": x,
                        "y": y,
                        "z": z
                    }
                })
                continue

            # Parse gripper commands
            if re.search(r'close\s+(?:the\s+)?gripper|grasp|grip|grab', part):
                commands.append({
                    "operation": "control_gripper",
                    "params": {
                        "robot_id": robot_id,
                        "open_gripper": False
                    }
                })
                continue

            if re.search(r'open\s+(?:the\s+)?gripper|release|drop', part):
                commands.append({
                    "operation": "control_gripper",
                    "params": {
                        "robot_id": robot_id,
                        "open_gripper": True
                    }
                })
                continue

            # Parse status check
            if re.search(r'check\s+(?:robot\s+)?status|get\s+status', part):
                commands.append({
                    "operation": "check_robot_status",
                    "params": {
                        "robot_id": robot_id
                    }
                })
                continue

        if commands:
            return {
                "success": True,
                "commands": commands,
                "error": None
            }
        else:
            return {
                "success": False,
                "commands": [],
                "error": f"Could not parse command: {command_text}"
            }

    def get_supported_patterns(self) -> List[str]:
        """Get list of supported regex patterns for documentation."""
        return [
            "move to (x, y, z) - Move robot to coordinates",
            "move to x=0.3, y=0.2, z=0.1 - Move robot to coordinates",
            "close gripper / grasp / grip / grab - Close the gripper",
            "open gripper / release / drop - Open the gripper",
            "check status / get status - Get robot status",
            "Commands can be chained with 'and', 'then', 'after that', or commas"
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
