"""
Structure Validator for Generated Operations
==============================================

Validates that generated operations follow the BasicOperation pattern:
- Has an implementation function returning OperationResult
- Has a create_*_operation() factory function
- Has a *_OPERATION constant
- No restricted module imports
"""

import ast
import logging
from typing import Tuple, List

from core.LoggingSetup import setup_logging

setup_logging(__name__)
logger = logging.getLogger(__name__)

try:
    from config.DynamicOperations import RESTRICTED_MODULES, RESTRICTED_BUILTINS
except ImportError:
    from ..config.DynamicOperations import RESTRICTED_MODULES, RESTRICTED_BUILTINS


def validate_structure(code: str) -> Tuple[bool, List[str]]:
    """
    Validate that generated code follows the BasicOperation pattern.

    Checks:
    1. No restricted module imports (os, subprocess, sys, etc.)
    2. No restricted builtin usage (eval, exec, open, etc.)
    3. Has at least one function definition (the implementation)
    4. Has a create_*_operation() factory function
    5. Has a *_OPERATION module-level constant assignment

    Args:
        code: Python source code string

    Returns:
        Tuple of (is_valid, list_of_errors).
        Empty error list on success.
    """
    errors = []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"Cannot parse code: {e}"]

    # Check for restricted imports
    _check_restricted_imports(tree, errors)

    # Check for restricted builtins
    _check_restricted_builtins(tree, errors)

    # Check for implementation function
    _check_implementation_function(tree, errors)

    # Check for create_*_operation() factory
    _check_factory_function(tree, errors)

    # Check for *_OPERATION constant
    _check_operation_constant(tree, errors)

    if errors:
        logger.warning(f"Structure validation failed with {len(errors)} errors")
    else:
        logger.debug("Structure validation passed")

    return len(errors) == 0, errors


def _check_restricted_imports(tree: ast.AST, errors: List[str]):
    """Check for restricted module imports in the AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".")[0]
                if module_name in RESTRICTED_MODULES:
                    errors.append(
                        f"Restricted import: '{alias.name}' "
                        f"(module '{module_name}' is not allowed)"
                    )

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module.split(".")[0]
                if module_name in RESTRICTED_MODULES:
                    errors.append(
                        f"Restricted import: 'from {node.module}' "
                        f"(module '{module_name}' is not allowed)"
                    )


def _check_restricted_builtins(tree: ast.AST, errors: List[str]):
    """Check for restricted builtin function calls in the AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check for direct calls like eval(), exec()
            if isinstance(node.func, ast.Name):
                if node.func.id in RESTRICTED_BUILTINS:
                    errors.append(
                        f"Restricted builtin: '{node.func.id}()' is not allowed"
                    )


def _check_implementation_function(tree: ast.AST, errors: List[str]):
    """Check that there is at least one function definition (the implementation)."""
    functions = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]
    if not functions:
        errors.append("No function definitions found - need at least an implementation function")


def _check_factory_function(tree: ast.AST, errors: List[str]):
    """Check for a create_*_operation() factory function."""
    factory_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("create_") and node.name.endswith("_operation"):
                factory_found = True
                break

    if not factory_found:
        errors.append(
            "Missing factory function: expected a 'create_*_operation()' function "
            "(e.g., create_rotate_gripper_operation)"
        )


def _check_operation_constant(tree: ast.AST, errors: List[str]):
    """Check for a *_OPERATION module-level constant assignment."""
    constant_found = False
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.endswith("_OPERATION"):
                    constant_found = True
                    break

    if not constant_found:
        errors.append(
            "Missing operation constant: expected a '*_OPERATION' module-level assignment "
            "(e.g., ROTATE_GRIPPER_OPERATION = create_rotate_gripper_operation())"
        )
