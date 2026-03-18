#!/usr/bin/env python3
"""
Operation Generator
=====================

Generates new operations via LLM when no fitting operation is found in the
RAG system. Generated operations are validated, stored, and indexed for
future use.

This module is only active when ENABLE_DYNAMIC_OPERATIONS is True.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import requests

from core.LoggingSetup import setup_logging

setup_logging(__name__)
logger = logging.getLogger(__name__)

try:
    from config.DynamicOperations import (
        ENABLE_DYNAMIC_OPERATIONS,
        MAX_GENERATED_OPERATIONS,
        REQUIRE_USER_REVIEW,
        AUTO_REBUILD_INDEX,
        GENERATED_OPERATIONS_DIR,
        OPERATION_GENERATION_TEMPERATURE,
    )
    from config.Servers import LMSTUDIO_BASE_URL, DEFAULT_LMSTUDIO_MODEL, SYSTEM_PROMPT_BASE
except ImportError:
    from ..config.DynamicOperations import (
        ENABLE_DYNAMIC_OPERATIONS,
        MAX_GENERATED_OPERATIONS,
        REQUIRE_USER_REVIEW,
        AUTO_REBUILD_INDEX,
        GENERATED_OPERATIONS_DIR,
        OPERATION_GENERATION_TEMPERATURE,
    )
    from ..config.Servers import LMSTUDIO_BASE_URL, DEFAULT_LMSTUDIO_MODEL, SYSTEM_PROMPT_BASE


class OperationGenerator:
    """
    Generates new robot operations using an LLM.

    When the RAG system returns low-confidence results for a command,
    this generator creates a new operation definition by:
    1. Building a prompt with existing operation examples
    2. Calling the LLM to generate code
    3. Validating syntax, structure, and sandbox execution
    4. Saving to operations/generated/ directory
    5. Optionally registering and rebuilding RAG index
    """

    def __init__(
        self,
        lm_studio_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the OperationGenerator.

        Args:
            lm_studio_url: LM Studio base URL (default from config)
            model: Model name for code generation (default from config)
        """
        self.lm_studio_url = lm_studio_url or LMSTUDIO_BASE_URL
        self.model = model or DEFAULT_LMSTUDIO_MODEL
        self.temperature = OPERATION_GENERATION_TEMPERATURE

    def generate_operation(
        self, command_text: str, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Generate a new operation from a natural language command.

        Args:
            command_text: The command that no existing operation matched
            context: Optional context dict with robot_id, recent operations, etc.

        Returns:
            Tuple of (success, message, file_path).
            file_path is None on failure.
        """
        if not ENABLE_DYNAMIC_OPERATIONS:
            return False, "Dynamic operations are disabled", None

        # Check generation limit
        from operations.generated import get_generated_operations_count

        current_count = get_generated_operations_count()
        if current_count >= MAX_GENERATED_OPERATIONS:
            return (
                False,
                f"Maximum generated operations limit reached ({MAX_GENERATED_OPERATIONS})",
                None,
            )

        logger.info(f"Generating new operation for command: '{command_text}'")

        # Build the LLM prompt
        prompt = self._build_generation_prompt(command_text, context)

        # Call the LLM
        code = self._call_llm(prompt)
        if not code:
            return False, "LLM failed to generate operation code", None

        # Validate the generated code
        is_valid, validation_msg = self._validate_generated_code(code)
        if not is_valid:
            logger.warning(f"Generated code failed validation: {validation_msg}")
            return False, f"Validation failed: {validation_msg}", None

        # Save and register
        file_path = self._save_and_register(code, command_text)
        if not file_path:
            return False, "Failed to save generated operation", None

        # Rebuild RAG index only if the operation is immediately active.
        # When REQUIRE_USER_REVIEW=True the file is written as PENDING and not
        # registered — indexing it now would expose the unapproved operation to
        # the CommandParser's RAG search before human approval.
        if AUTO_REBUILD_INDEX and not REQUIRE_USER_REVIEW:
            self._rebuild_rag_index()

        status = "PENDING review" if REQUIRE_USER_REVIEW else "active"
        return (
            True,
            f"Generated new operation ({status}): {file_path}",
            file_path,
        )

    def _build_generation_prompt(
        self, command_text: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build the LLM prompt for operation generation.

        Includes existing operation examples to guide the LLM.

        Args:
            command_text: The command to create an operation for
            context: Optional context information

        Returns:
            Complete prompt string for the LLM
        """
        # Load example operations as templates
        examples = self._load_example_operations()

        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, indent=2)}"

        return f"""You are a robot operation code generator. Generate a Python module that implements a new robot operation.

The operation should handle this command: "{command_text}"
{context_str}

=== REQUIRED STRUCTURE ===

Every operation module MUST have:
1. An implementation function that takes robot_id and operation-specific parameters, returns OperationResult
2. A create_*_operation() factory function that returns a BasicOperation instance
3. A *_OPERATION module-level constant (e.g., ROTATE_GRIPPER_OPERATION = create_rotate_gripper_operation())

=== AVAILABLE IMPORTS ===

You can ONLY use these imports:
- import time
- import logging
- import json
- from typing import Any, Dict, Optional, List
- from operations.Base import BasicOperation, OperationCategory, OperationComplexity, OperationParameter, OperationResult, OperationRelationship

For sending commands to Unity:
- from core.Imports import get_command_broadcaster as _get_command_broadcaster
- Then call: _get_command_broadcaster().send_command(command_dict, request_id)

DO NOT import: os, subprocess, sys, shutil, pathlib, socket
DO NOT use: eval(), exec(), open(), __import__()

=== COMMAND FORMAT FOR UNITY ===

Commands sent to Unity have this structure:
{{
    "command_type": "operation_name",
    "robot_id": robot_id,
    "parameters": {{}},
    "timestamp": time.time(),
    "request_id": request_id
}}

=== EXAMPLE OPERATIONS ===

{examples}

=== RULES ===

1. The implementation function MUST return OperationResult (use .success_result() or .error_result())
2. Validate all parameters before use
3. Send commands to Unity via _get_command_broadcaster().send_command()
4. Include docstrings on every function
5. Use descriptive operation_id with format: "category_name_NNN" (e.g., "manipulation_rotate_gripper_001")
6. The operation name should be snake_case
7. Include realistic usage_examples, preconditions, postconditions, and failure_modes

=== OUTPUT ===

Output ONLY the Python code. No markdown, no explanations, no code fences.
Start directly with the module docstring."""

    def _load_example_operations(self) -> str:
        """
        Load existing operation files as examples for the LLM.

        Returns:
            Formatted string with example operation code
        """
        examples = []
        operations_dir = Path(__file__).parent

        # Load GripperOperations as a concise example
        gripper_file = operations_dir / "GripperOperations.py"
        if gripper_file.exists():
            try:
                content = gripper_file.read_text()
                # Extract just the control_gripper function and its BasicOperation
                # to keep the prompt concise
                examples.append(
                    f"--- Example: GripperOperations.py (abbreviated) ---\n{content[:2500]}\n---"
                )
            except Exception:
                pass

        if not examples:
            examples.append(
                "No example operations available. Follow the REQUIRED STRUCTURE described above."
            )

        return "\n\n".join(examples)

    def _call_llm(self, prompt: str) -> Optional[str]:
        """
        Call the LLM to generate operation code.

        Args:
            prompt: The generation prompt

        Returns:
            Generated Python code string, or None on failure
        """
        try:
            response = requests.post(
                f"{self.lm_studio_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                SYSTEM_PROMPT_BASE
                                + " You are a Python code generator for this robotics framework. "
                                "Output only raw Python source code — no markdown fences, no "
                                "explanations. The generated code will be sandboxed and validated "
                                "before execution. Only use the imports explicitly listed in the "
                                "user message."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": 4000,
                },
                timeout=120,
            )

            if response.status_code != 200:
                logger.error(f"LLM request failed with status {response.status_code}")
                return None

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Strip markdown code fences if present
            code = self._strip_code_fences(content)

            return code

        except requests.exceptions.Timeout:
            logger.error("LLM request timed out")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to LM Studio at {self.lm_studio_url}")
            return None
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def _strip_code_fences(self, content: str) -> str:
        """
        Remove markdown code fences from LLM output.

        Args:
            content: Raw LLM output

        Returns:
            Clean Python code
        """
        # Remove ```python ... ``` or ``` ... ```
        match = re.search(r"```(?:python)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content.strip()

    def _validate_generated_code(self, code: str) -> Tuple[bool, str]:
        """
        Run all validators on generated code.

        Args:
            code: Python source code string

        Returns:
            Tuple of (is_valid, error_message)
        """
        from validation.SyntaxValidator import validate_syntax
        from validation.StructureValidator import validate_structure
        from validation.SandboxExecutor import validate_in_sandbox

        # Step 1: Syntax validation
        is_valid, error = validate_syntax(code)
        if not is_valid:
            return False, f"Syntax: {error}"

        # Step 2: Structure validation
        is_valid, errors = validate_structure(code)
        if not is_valid:
            return False, f"Structure: {'; '.join(errors)}"

        # Step 3: Sandbox execution
        is_valid, error = validate_in_sandbox(code)
        if not is_valid:
            return False, f"Sandbox: {error}"

        return True, ""

    def _save_and_register(self, code: str, command_text: str) -> Optional[str]:
        """
        Save generated code to file and register in the registry.

        Args:
            code: Validated Python code
            command_text: Original command that triggered generation

        Returns:
            File path of saved operation, or None on failure
        """
        generated_dir = Path(GENERATED_OPERATIONS_DIR)
        generated_dir.mkdir(parents=True, exist_ok=True)

        # Extract operation name from the code
        op_name = self._extract_operation_name(code)
        if not op_name:
            op_name = "unknown"

        # Generate filename with timestamp
        timestamp = int(time.time())
        filename = f"{op_name}_{timestamp}.py"
        file_path = generated_dir / filename

        # Add metadata header
        review_status = "PENDING" if REQUIRE_USER_REVIEW else "APPROVED"
        header = (
            f"# REVIEW_STATUS: {review_status}\n"
            f"# GENERATED_AT: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# ORIGINAL_COMMAND: {command_text}\n"
            f"# AUTO_GENERATED: True\n\n"
        )

        try:
            file_path.write_text(header + code)
            logger.info(f"Saved generated operation to {file_path}")

            # Auto-generate a pytest skeleton alongside the operation file
            self._generate_test_file(file_path, op_name, command_text)

            # Register if auto-approved
            if not REQUIRE_USER_REVIEW:
                self._register_operation(file_path)

            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to save generated operation: {e}")
            return None

    def _generate_test_file(
        self, operation_file: Path, op_name: str, command_text: str
    ) -> None:
        """
        Generate a pytest skeleton file alongside a newly created operation.

        The test file is saved next to the operation file with a ``Test_`` prefix
        and ``REVIEW_STATUS: PENDING`` header so the human reviewer can see it in
        the same review workflow as the operation itself.

        The generated skeleton covers three baseline scenarios:
        - Happy path: valid parameters should return a successful OperationResult.
        - Missing required parameter: should return an error OperationResult.
        - Invalid parameter type: should return an error OperationResult.

        Args:
            operation_file: Path to the already-written operation ``.py`` file.
            op_name: Snake-case name extracted from the generated code (e.g. ``rotate_gripper``).
            command_text: The original natural-language command that triggered generation,
                included as a comment so reviewers have context.
        """
        test_path = (
            operation_file.parent
            / f"Test_{op_name}_{operation_file.stem.split('_')[-1]}.py"
        )
        module_stem = operation_file.stem  # e.g. rotate_gripper_1700000000
        fn_name = op_name  # e.g. rotate_gripper
        class_name = "".join(part.capitalize() for part in op_name.split("_"))

        test_code = f'''"""
Auto-generated pytest skeleton for operation: {op_name}
Original command: {command_text}

IMPORTANT: This file was generated automatically. Review and expand tests before approving.
"""
# REVIEW_STATUS: PENDING
# GENERATED_AT: {time.strftime('%Y-%m-%d %H:%M:%S')}
# AUTO_GENERATED: True

import pytest
from unittest.mock import Mock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_broadcaster():
    """Provide a mock CommandBroadcaster that returns a success response."""
    broadcaster = Mock()
    broadcaster.send_command = Mock(return_value=True)
    broadcaster.wait_for_result = Mock(return_value={{
        "success": True,
        "result": {{"status": "completed"}},
        "error": None,
    }})
    return broadcaster


@pytest.fixture
def operation_module(mock_broadcaster):
    """Import the generated operation module with broadcaster patched."""
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "{module_stem}",
        Path(__file__).parent / "{operation_file.name}",
    )
    module = importlib.util.module_from_spec(spec)

    with patch("core.Imports.get_command_broadcaster", return_value=mock_broadcaster):
        spec.loader.exec_module(module)

    return module


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class Test{class_name}:
    """Tests for the auto-generated {op_name} operation."""

    def test_happy_path_returns_success(self, operation_module, mock_broadcaster):
        """Valid parameters should produce a successful OperationResult."""
        # TODO: Replace with actual valid parameters for this operation
        result = operation_module.{fn_name}(robot_id="Robot1")

        assert result is not None
        assert result.success is True, f"Expected success but got error: {{result.error}}"

    def test_missing_robot_id_returns_error(self, operation_module):
        """Calling without robot_id should return an error OperationResult."""
        result = operation_module.{fn_name}(robot_id=None)

        assert result is not None
        assert result.success is False, "Expected failure for missing robot_id"

    def test_returns_operation_result_type(self, operation_module, mock_broadcaster):
        """Return type must always be OperationResult, never raise."""
        from operations.Base import OperationResult

        result = operation_module.{fn_name}(robot_id="Robot1")
        assert isinstance(result, OperationResult), (
            f"Expected OperationResult, got {{type(result).__name__}}"
        )
'''

        try:
            test_path.write_text(test_code)
            logger.info(f"Generated test skeleton at {test_path}")
        except Exception as e:
            # Test file generation is best-effort; don't fail the whole operation save
            logger.warning(f"Could not write test skeleton: {e}")

    def _extract_operation_name(self, code: str) -> Optional[str]:
        """
        Extract the operation name from generated code.

        Looks for the create_*_operation() function name pattern.

        Args:
            code: Python source code

        Returns:
            Operation name (e.g., "rotate_gripper") or None
        """
        import ast

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith("create_") and node.name.endswith(
                        "_operation"
                    ):
                        return node.name.removeprefix("create_").removesuffix(
                            "_operation"
                        )
        except Exception:
            pass

        return None

    def _register_operation(self, file_path: Path):
        """
        Register a generated operation in the global registry.

        Args:
            file_path: Path to the generated operation file
        """
        try:
            from operations.generated import _load_operation_from_file
            from core.Imports import get_global_registry

            operation = _load_operation_from_file(file_path)
            if operation:
                registry = get_global_registry()
                registry.operations[operation.operation_id] = operation
                logger.info(f"Registered generated operation: {operation.name}")

        except Exception as e:
            logger.warning(f"Failed to register generated operation: {e}")

    def _rebuild_rag_index(self):
        """Trigger a RAG index rebuild to include new operations."""
        try:
            from rag import RAGSystem

            rag = RAGSystem()
            rag.index_operations(rebuild=True)
            logger.info("RAG index rebuilt after operation generation")

        except Exception as e:
            logger.warning(f"Failed to rebuild RAG index: {e}")
