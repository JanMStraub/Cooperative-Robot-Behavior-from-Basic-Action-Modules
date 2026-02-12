"""
Syntax Validator for Generated Operations
===========================================

Validates that generated Python code is syntactically correct using ast.parse().
"""

import ast
import logging
from typing import Tuple

from core.LoggingSetup import setup_logging

setup_logging(__name__)
logger = logging.getLogger(__name__)


def validate_syntax(code: str) -> Tuple[bool, str]:
    """
    Validate Python syntax of generated operation code.

    Uses ast.parse() to check that the code is valid Python.

    Args:
        code: Python source code string

    Returns:
        Tuple of (is_valid, error_message).
        error_message is empty string on success.
    """
    if not code or not code.strip():
        return False, "Empty code provided"

    try:
        ast.parse(code)
        logger.debug("Syntax validation passed")
        return True, ""
    except SyntaxError as e:
        error_msg = f"Syntax error at line {e.lineno}, col {e.offset}: {e.msg}"
        logger.warning(f"Syntax validation failed: {error_msg}")
        return False, error_msg
