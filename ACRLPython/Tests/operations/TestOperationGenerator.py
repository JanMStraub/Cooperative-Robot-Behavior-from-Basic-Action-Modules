#!/usr/bin/env python3
"""
Unit tests for the Dynamic Operation Generation system.

Tests:
- Syntax validation catches errors
- Structure validation rejects restricted imports
- Structure validation requires BasicOperation pattern
- Sandbox execution timeout works
- Sandbox execution rejects dangerous code
- Generator integration (with mocked LLM)
- Generated operations loader
- Review status management
"""

import pytest
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from validation.SyntaxValidator import validate_syntax
from validation.StructureValidator import validate_structure
from validation.SandboxExecutor import validate_in_sandbox


# ============================================================================
# Sample Code Fixtures
# ============================================================================

VALID_OPERATION_CODE = '''
"""Sample generated operation."""

import time
import logging
from typing import Any, Dict, Optional
from operations.Base import (
    BasicOperation,
    OperationCategory,
    OperationComplexity,
    OperationParameter,
    OperationResult,
    OperationRelationship,
)

logger = logging.getLogger(__name__)

from core.Imports import get_command_broadcaster as _get_command_broadcaster


def rotate_end_effector(
    robot_id: str, angle: float = 45.0, request_id: int = 0
) -> OperationResult:
    """Rotate the end effector by a given angle."""
    if not robot_id or not isinstance(robot_id, str):
        return OperationResult.error_result(
            "INVALID_ROBOT_ID",
            "Robot ID must be a non-empty string",
            ["Provide a valid robot ID"],
        )

    command = {
        "command_type": "rotate_end_effector",
        "robot_id": robot_id,
        "parameters": {"angle": angle},
        "timestamp": time.time(),
        "request_id": request_id,
    }

    success = _get_command_broadcaster().send_command(command, request_id)

    if not success:
        return OperationResult.error_result(
            "COMMUNICATION_FAILED",
            "Failed to send command to Unity",
            ["Ensure Unity is running"],
        )

    return OperationResult.success_result({
        "robot_id": robot_id,
        "angle": angle,
        "status": "command_sent",
        "timestamp": time.time(),
    })


def create_rotate_end_effector_operation() -> BasicOperation:
    """Create the BasicOperation definition for rotate_end_effector."""
    return BasicOperation(
        operation_id="manipulation_rotate_end_effector_001",
        name="rotate_end_effector",
        category=OperationCategory.MANIPULATION,
        complexity=OperationComplexity.ATOMIC,
        description="Rotate the robot end effector by a given angle.",
        long_description="Rotates the end effector around its axis by the specified angle in degrees.",
        usage_examples=["rotate_end_effector('Robot1', angle=90.0)"],
        parameters=[
            OperationParameter(name="robot_id", type="str", description="Robot ID", required=True),
            OperationParameter(name="angle", type="float", description="Rotation angle in degrees", required=False, default=45.0),
        ],
        preconditions=["Robot is initialized"],
        postconditions=["End effector rotated"],
        average_duration_ms=300.0,
        success_rate=0.95,
        failure_modes=["Communication failed"],
        implementation=rotate_end_effector,
    )


ROTATE_END_EFFECTOR_OPERATION = create_rotate_end_effector_operation()
'''

INVALID_SYNTAX_CODE = '''
def broken_function(
    return "missing closing paren"
'''

RESTRICTED_IMPORT_CODE = '''
import os
import subprocess
from operations.Base import OperationResult

def do_something():
    """Bad operation."""
    os.system("rm -rf /")
    return OperationResult.success_result({})

def create_bad_operation():
    """Factory."""
    pass

BAD_OPERATION = create_bad_operation()
'''

MISSING_FACTORY_CODE = '''
from operations.Base import OperationResult

def do_something():
    """Implementation without factory."""
    return OperationResult.success_result({})

SOME_OPERATION = "not a factory"
'''

EVAL_CODE = '''
from operations.Base import OperationResult

def do_something():
    """Uses eval."""
    result = eval("1 + 1")
    return OperationResult.success_result({})

def create_eval_operation():
    """Factory."""
    pass

EVAL_OPERATION = create_eval_operation()
'''


# ============================================================================
# Test Class: Syntax Validation
# ============================================================================

class TestSyntaxValidation:
    """Test Python syntax validation."""

    def test_valid_code_passes(self):
        """Test that valid Python code passes syntax validation."""
        is_valid, error = validate_syntax(VALID_OPERATION_CODE)
        assert is_valid is True
        assert error == ""

    def test_invalid_syntax_fails(self):
        """Test that syntax errors are caught."""
        is_valid, error = validate_syntax(INVALID_SYNTAX_CODE)
        assert is_valid is False
        assert "Syntax error" in error

    def test_empty_code_fails(self):
        """Test that empty code is rejected."""
        is_valid, error = validate_syntax("")
        assert is_valid is False
        assert "Empty" in error

    def test_whitespace_only_fails(self):
        """Test that whitespace-only code is rejected."""
        is_valid, error = validate_syntax("   \n\n  ")
        assert is_valid is False

    def test_simple_function_passes(self):
        """Test that a simple function passes."""
        code = "def hello():\n    return 'world'"
        is_valid, error = validate_syntax(code)
        assert is_valid is True


# ============================================================================
# Test Class: Structure Validation
# ============================================================================

class TestStructureValidation:
    """Test operation structure validation."""

    def test_valid_operation_passes(self):
        """Test that a properly structured operation passes."""
        is_valid, errors = validate_structure(VALID_OPERATION_CODE)
        assert is_valid is True
        assert len(errors) == 0

    def test_restricted_imports_rejected(self):
        """Test that restricted imports are caught."""
        is_valid, errors = validate_structure(RESTRICTED_IMPORT_CODE)
        assert is_valid is False
        assert any("os" in e for e in errors)
        assert any("subprocess" in e for e in errors)

    def test_missing_factory_caught(self):
        """Test that missing create_*_operation() is caught."""
        is_valid, errors = validate_structure(MISSING_FACTORY_CODE)
        assert is_valid is False
        assert any("factory" in e.lower() for e in errors)

    def test_eval_rejected(self):
        """Test that eval() usage is caught."""
        is_valid, errors = validate_structure(EVAL_CODE)
        assert is_valid is False
        assert any("eval" in e for e in errors)

    def test_no_functions_rejected(self):
        """Test that code without any functions is rejected."""
        code = "x = 1\ny = 2\nSOME_OPERATION = x + y"
        is_valid, errors = validate_structure(code)
        assert is_valid is False
        assert any("function" in e.lower() for e in errors)

    def test_missing_operation_constant(self):
        """Test that missing *_OPERATION constant is caught."""
        code = """
def something():
    pass

def create_something_operation():
    pass

SOME_VALUE = create_something_operation()
"""
        is_valid, errors = validate_structure(code)
        assert is_valid is False
        assert any("OPERATION" in e for e in errors)


# ============================================================================
# Test Class: Sandbox Execution
# ============================================================================

class TestSandboxExecution:
    """Test sandbox execution of generated operations."""

    def test_valid_operation_passes_sandbox(self):
        """Test that valid operation code passes sandbox execution."""
        is_valid, error = validate_in_sandbox(VALID_OPERATION_CODE, timeout=10)
        assert is_valid is True
        assert error == ""

    def test_timeout_enforced(self):
        """Test that sandbox timeout is enforced."""
        code = """
import time

def infinite_loop():
    while True:
        time.sleep(0.01)

def create_loop_operation():
    pass

infinite_loop()
LOOP_OPERATION = None
"""
        is_valid, error = validate_in_sandbox(code, timeout=1)
        assert is_valid is False
        assert "timed out" in error.lower()

    def test_missing_operation_constant_fails(self):
        """Test that sandbox detects missing *_OPERATION constant."""
        code = """
def simple_func():
    return 42

def create_simple_operation():
    return simple_func()

result = create_simple_operation()
"""
        is_valid, error = validate_in_sandbox(code, timeout=5)
        assert is_valid is False
        assert "OPERATION" in error


# ============================================================================
# Test Class: Generated Operations Loader
# ============================================================================

class TestGeneratedOperationsLoader:
    """Test loading of generated operations from files."""

    def test_get_review_status_pending(self, tmp_path):
        """Test extracting PENDING review status."""
        from operations.generated import _get_review_status

        op_file = tmp_path / "test_op.py"
        op_file.write_text("# REVIEW_STATUS: PENDING\n# GENERATED_AT: 2025-01-01\n\ndef foo(): pass\n")

        status = _get_review_status(op_file)
        assert status == "PENDING"

    def test_get_review_status_approved(self, tmp_path):
        """Test extracting APPROVED review status."""
        from operations.generated import _get_review_status

        op_file = tmp_path / "test_op.py"
        op_file.write_text("# REVIEW_STATUS: APPROVED\n\ndef foo(): pass\n")

        status = _get_review_status(op_file)
        assert status == "APPROVED"

    def test_get_review_status_unknown(self, tmp_path):
        """Test extracting status from file without header."""
        from operations.generated import _get_review_status

        op_file = tmp_path / "test_op.py"
        op_file.write_text("def foo(): pass\n")

        status = _get_review_status(op_file)
        assert status == "UNKNOWN"

    def test_set_review_status(self, tmp_path):
        """Test updating review status."""
        from operations.generated import set_review_status, _get_review_status

        op_file = tmp_path / "test_op.py"
        op_file.write_text("# REVIEW_STATUS: PENDING\n\ndef foo(): pass\n")

        result = set_review_status(str(op_file), "APPROVED")
        assert result is True

        new_status = _get_review_status(op_file)
        assert new_status == "APPROVED"

    def test_generated_operations_count(self, tmp_path):
        """Test counting generated operations."""
        from operations.generated import get_generated_operations_count

        with patch("config.DynamicOperations.GENERATED_OPERATIONS_DIR", str(tmp_path)):
            # Create some files
            (tmp_path / "__init__.py").write_text("")
            (tmp_path / "op1.py").write_text("# test")
            (tmp_path / "op2.py").write_text("# test")

            count = get_generated_operations_count()
            assert count == 2


# ============================================================================
# Test Class: Operation Generator
# ============================================================================

class TestOperationGenerator:
    """Test the OperationGenerator class."""

    def test_generator_disabled_returns_false(self):
        """Test that generator returns False when disabled."""
        from operations.Generator import OperationGenerator

        with patch("operations.Generator.ENABLE_DYNAMIC_OPERATIONS", False):
            generator = OperationGenerator()
            success, msg, path = generator.generate_operation("test command")

            assert success is False
            assert "disabled" in msg.lower()
            assert path is None

    def test_generator_respects_max_limit(self):
        """Test that generator respects MAX_GENERATED_OPERATIONS."""
        from operations.Generator import OperationGenerator

        with patch("operations.Generator.ENABLE_DYNAMIC_OPERATIONS", True), \
             patch("operations.Generator.MAX_GENERATED_OPERATIONS", 5), \
             patch("operations.generated.get_generated_operations_count", return_value=5):

            generator = OperationGenerator()
            success, msg, path = generator.generate_operation("test command")

            assert success is False
            assert "limit" in msg.lower()

    def test_extract_operation_name(self):
        """Test operation name extraction from code."""
        from operations.Generator import OperationGenerator

        generator = OperationGenerator()
        name = generator._extract_operation_name(VALID_OPERATION_CODE)
        assert name == "rotate_end_effector"

    def test_extract_operation_name_none_for_bad_code(self):
        """Test that name extraction returns None for code without factory."""
        from operations.Generator import OperationGenerator

        generator = OperationGenerator()
        name = generator._extract_operation_name("def foo(): pass")
        assert name is None

    def test_strip_code_fences(self):
        """Test stripping markdown code fences."""
        from operations.Generator import OperationGenerator

        generator = OperationGenerator()

        # With python fence
        assert generator._strip_code_fences("```python\ncode here\n```") == "code here"

        # With generic fence
        assert generator._strip_code_fences("```\ncode here\n```") == "code here"

        # Without fence
        assert generator._strip_code_fences("code here") == "code here"

    def test_validate_generated_code_valid(self):
        """Test validation passes for valid code."""
        from operations.Generator import OperationGenerator

        generator = OperationGenerator()
        is_valid, error = generator._validate_generated_code(VALID_OPERATION_CODE)

        assert is_valid is True
        assert error == ""

    def test_validate_generated_code_syntax_error(self):
        """Test validation catches syntax errors."""
        from operations.Generator import OperationGenerator

        generator = OperationGenerator()
        is_valid, error = generator._validate_generated_code(INVALID_SYNTAX_CODE)

        assert is_valid is False
        assert "Syntax" in error

    def test_validate_generated_code_restricted_import(self):
        """Test validation catches restricted imports."""
        from operations.Generator import OperationGenerator

        generator = OperationGenerator()
        is_valid, error = generator._validate_generated_code(RESTRICTED_IMPORT_CODE)

        assert is_valid is False
        assert "Structure" in error

    def test_save_and_register(self, tmp_path):
        """Test saving generated code to file."""
        from operations.Generator import OperationGenerator

        with patch("operations.Generator.GENERATED_OPERATIONS_DIR", str(tmp_path)), \
             patch("operations.Generator.REQUIRE_USER_REVIEW", True):

            generator = OperationGenerator()
            file_path = generator._save_and_register(VALID_OPERATION_CODE, "rotate gripper 45 degrees")

            assert file_path is not None
            assert Path(file_path).exists()

            content = Path(file_path).read_text()
            assert "REVIEW_STATUS: PENDING" in content
            assert "rotate gripper 45 degrees" in content


# ============================================================================
# Test Class: CommandParser Integration
# ============================================================================

class TestCommandParserGeneration:
    """Test CommandParser integration with dynamic generation."""

    def test_check_generation_needed_low_scores(self):
        """Test that low RAG scores trigger generation."""
        from orchestrators.CommandParser import CommandParser

        with patch("orchestrators.CommandParser.RAGSystem"), \
             patch("config.DynamicOperations.ENABLE_DYNAMIC_OPERATIONS", True), \
             patch("config.DynamicOperations.GENERATION_TRIGGER_THRESHOLD", 0.4):

            parser = CommandParser(use_rag=False)
            rag_results = [
                {"name": "move_to_coordinate", "similarity_score": 0.2},
                {"name": "control_gripper", "similarity_score": 0.15},
            ]

            should_generate, reason = parser._check_generation_needed(rag_results)
            assert should_generate is True
            assert "0.20" in reason or "below" in reason.lower()

    def test_check_generation_needed_high_scores(self):
        """Test that high RAG scores don't trigger generation."""
        from orchestrators.CommandParser import CommandParser

        with patch("orchestrators.CommandParser.RAGSystem"), \
             patch("config.DynamicOperations.ENABLE_DYNAMIC_OPERATIONS", True), \
             patch("config.DynamicOperations.GENERATION_TRIGGER_THRESHOLD", 0.4):

            parser = CommandParser(use_rag=False)
            rag_results = [
                {"name": "move_to_coordinate", "similarity_score": 0.85},
                {"name": "control_gripper", "similarity_score": 0.6},
            ]

            should_generate, reason = parser._check_generation_needed(rag_results)
            assert should_generate is False

    def test_check_generation_needed_disabled(self):
        """Test that generation check respects feature flag."""
        from orchestrators.CommandParser import CommandParser

        with patch("orchestrators.CommandParser.RAGSystem"), \
             patch("config.DynamicOperations.ENABLE_DYNAMIC_OPERATIONS", False):

            parser = CommandParser(use_rag=False)
            should_generate, reason = parser._check_generation_needed([])
            assert should_generate is False

    def test_check_generation_needed_empty_results(self):
        """Test that empty RAG results trigger generation."""
        from orchestrators.CommandParser import CommandParser

        with patch("orchestrators.CommandParser.RAGSystem"), \
             patch("config.DynamicOperations.ENABLE_DYNAMIC_OPERATIONS", True):

            parser = CommandParser(use_rag=False)
            should_generate, reason = parser._check_generation_needed([])
            assert should_generate is True


# ============================================================================
# Test Class: _generate_test_file
# ============================================================================

class TestGenerateTestFile:
    """Tests for the auto-generated pytest skeleton writer."""

    def _make_generator(self):
        """Return an OperationGenerator with default config."""
        from operations.Generator import OperationGenerator
        return OperationGenerator()

    def test_creates_test_file_alongside_operation(self, tmp_path):
        """_generate_test_file writes a Test_*.py next to the operation file."""
        generator = self._make_generator()
        op_file = tmp_path / "rotate_gripper_1700000000.py"
        op_file.write_text("# placeholder operation")

        generator._generate_test_file(op_file, "rotate_gripper", "rotate gripper 45 degrees")

        # File should exist adjacent to the operation file
        test_files = list(tmp_path.glob("Test_rotate_gripper_*.py"))
        assert len(test_files) == 1

    def test_generated_file_contains_pending_header(self, tmp_path):
        """Test file must include the REVIEW_STATUS: PENDING header."""
        generator = self._make_generator()
        op_file = tmp_path / "rotate_gripper_1700000000.py"
        op_file.write_text("# placeholder")

        generator._generate_test_file(op_file, "rotate_gripper", "rotate 45 degrees")

        test_file = next(tmp_path.glob("Test_rotate_gripper_*.py"))
        content = test_file.read_text()
        assert "REVIEW_STATUS: PENDING" in content

    def test_generated_file_contains_original_command(self, tmp_path):
        """The original command text is embedded as a comment for reviewers."""
        generator = self._make_generator()
        op_file = tmp_path / "rotate_gripper_1700000000.py"
        op_file.write_text("# placeholder")

        generator._generate_test_file(op_file, "rotate_gripper", "rotate gripper 45 degrees")

        test_file = next(tmp_path.glob("Test_rotate_gripper_*.py"))
        content = test_file.read_text()
        assert "rotate gripper 45 degrees" in content

    def test_generated_file_contains_three_test_methods(self, tmp_path):
        """Skeleton must include the three baseline test scenarios."""
        generator = self._make_generator()
        op_file = tmp_path / "rotate_gripper_1700000000.py"
        op_file.write_text("# placeholder")

        generator._generate_test_file(op_file, "rotate_gripper", "rotate gripper")

        test_file = next(tmp_path.glob("Test_rotate_gripper_*.py"))
        content = test_file.read_text()
        assert "test_happy_path_returns_success" in content
        assert "test_missing_robot_id_returns_error" in content
        assert "test_returns_operation_result_type" in content

    def test_generated_file_contains_class_name_derived_from_op(self, tmp_path):
        """Test class name is CamelCase form of the operation name."""
        generator = self._make_generator()
        op_file = tmp_path / "rotate_gripper_1700000000.py"
        op_file.write_text("# placeholder")

        generator._generate_test_file(op_file, "rotate_gripper", "rotate")

        test_file = next(tmp_path.glob("Test_rotate_gripper_*.py"))
        content = test_file.read_text()
        # rotate_gripper → RotateGripper
        assert "class TestRotateGripper" in content

    def test_generate_test_file_does_not_raise_on_write_error(self, tmp_path):
        """If the test file cannot be written, _generate_test_file silently continues."""
        generator = self._make_generator()
        # Point op_file at a non-existent subdirectory so the write will fail
        op_file = tmp_path / "nonexistent_dir" / "rotate_gripper_1700000000.py"

        # Must not raise — failures are logged as warnings
        generator._generate_test_file(op_file, "rotate_gripper", "rotate gripper")

    def test_save_and_register_creates_test_file_too(self, tmp_path):
        """_save_and_register calls _generate_test_file, so a Test_*.py appears on disk."""
        with patch("operations.Generator.GENERATED_OPERATIONS_DIR", str(tmp_path)), \
             patch("operations.Generator.REQUIRE_USER_REVIEW", True):

            generator = self._make_generator()
            file_path = generator._save_and_register(
                VALID_OPERATION_CODE, "rotate end effector 45 degrees"
            )

        assert file_path is not None
        # At least one Test_*.py sibling should have been created
        test_files = list(tmp_path.glob("Test_*.py"))
        assert len(test_files) >= 1
