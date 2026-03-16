#!/usr/bin/env python3
"""
Generated Operations Loader
===============================

Loads dynamically generated operations from the generated/ directory
and registers them with the OperationRegistry.

Each generated file is expected to contain:
- A *_OPERATION constant (BasicOperation instance)
- A review status header comment (# REVIEW_STATUS: PENDING or APPROVED)
"""

import importlib.util
import logging
from pathlib import Path
from typing import List, Tuple

from core.LoggingSetup import setup_logging

setup_logging(__name__)
logger = logging.getLogger(__name__)


def load_generated_operations() -> List[Tuple[str, str]]:
    """
    Load all approved generated operations and register them.

    Scans the generated/ directory for Python files, checks their
    review status, and loads approved operations into the registry.

    Returns:
        List of (operation_name, file_path) tuples for loaded operations.
    """
    try:
        from config.DynamicOperations import (
            ENABLE_DYNAMIC_OPERATIONS,
            REQUIRE_USER_REVIEW,
            GENERATED_OPERATIONS_DIR,
        )
    except ImportError:
        logger.debug(
            "DynamicOperations config not available, skipping generated operations"
        )
        return []

    if not ENABLE_DYNAMIC_OPERATIONS:
        logger.debug("Dynamic operations disabled, skipping generated operations")
        return []

    generated_dir = Path(GENERATED_OPERATIONS_DIR)
    if not generated_dir.exists():
        logger.debug(f"Generated operations directory does not exist: {generated_dir}")
        return []

    loaded = []

    for py_file in sorted(generated_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue

        try:
            # Check review status
            status = _get_review_status(py_file)

            if REQUIRE_USER_REVIEW and status != "APPROVED":
                logger.debug(
                    f"Skipping unapproved operation: {py_file.name} (status: {status})"
                )
                continue

            # Load the module
            operation = _load_operation_from_file(py_file)
            if operation:
                loaded.append((operation.name, str(py_file)))
                logger.info(
                    f"Loaded generated operation: {operation.name} from {py_file.name}"
                )

        except Exception as e:
            logger.warning(f"Failed to load generated operation {py_file.name}: {e}")

    if loaded:
        logger.info(f"Loaded {len(loaded)} generated operations")

    return loaded


def _get_review_status(file_path: Path) -> str:
    """
    Extract the review status from a generated operation file.

    Looks for a comment line: # REVIEW_STATUS: PENDING or APPROVED

    Args:
        file_path: Path to the generated operation file

    Returns:
        Review status string ("PENDING", "APPROVED", or "UNKNOWN")
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("# REVIEW_STATUS:"):
                    return line.split(":", 1)[1].strip()
                # Stop searching after first non-comment, non-empty line
                if line and not line.startswith("#") and not line.startswith('"""'):
                    break
    except Exception:
        pass

    return "UNKNOWN"


def _load_operation_from_file(file_path: Path):
    """
    Load a BasicOperation from a generated Python file.

    Dynamically imports the module and looks for a *_OPERATION constant.

    Args:
        file_path: Path to the Python file

    Returns:
        BasicOperation instance, or None if not found
    """
    from operations.Base import BasicOperation

    module_name = f"operations.generated.{file_path.stem}"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        logger.warning(f"Cannot create module spec for {file_path}")
        return None

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        logger.warning(f"Failed to execute module {file_path.name}: {e}")
        return None

    # Find *_OPERATION constant
    for attr_name in dir(module):
        if attr_name.endswith("_OPERATION") and not attr_name.startswith("_"):
            value = getattr(module, attr_name)
            if isinstance(value, BasicOperation):
                return value

    logger.warning(f"No BasicOperation constant found in {file_path.name}")
    return None


def get_generated_operations_count() -> int:
    """
    Count the number of generated operation files.

    Returns:
        Number of .py files in the generated/ directory (excluding __init__.py)
    """
    try:
        from config.DynamicOperations import GENERATED_OPERATIONS_DIR
    except ImportError:
        return 0

    generated_dir = Path(GENERATED_OPERATIONS_DIR)
    if not generated_dir.exists():
        return 0

    return sum(1 for f in generated_dir.glob("*.py") if f.name != "__init__.py")


def set_review_status(file_path: str, status: str) -> bool:
    """
    Update the review status of a generated operation file.

    Args:
        file_path: Path to the generated operation file
        status: New status ("APPROVED" or "PENDING")

    Returns:
        True if status was updated successfully
    """
    path = Path(file_path)
    if not path.exists():
        return False

    try:
        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")

        # Find and update the REVIEW_STATUS line
        updated = False
        for i, line in enumerate(lines):
            if line.strip().startswith("# REVIEW_STATUS:"):
                lines[i] = f"# REVIEW_STATUS: {status}"
                updated = True
                break

        if not updated:
            # Insert status after the first comment block
            lines.insert(0, f"# REVIEW_STATUS: {status}")

        path.write_text("\n".join(lines), encoding="utf-8")
        return True

    except Exception as e:
        logger.error(f"Failed to update review status: {e}")
        return False
