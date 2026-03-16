#!/usr/bin/env python3
"""
Sandbox Executor for Generated Operations
============================================

Executes generated operations in a restricted environment with:
- Mocked dependencies (CommandBroadcaster, UnifiedImageStorage)
- Timeout enforcement
- Restricted builtins
"""

import logging
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock

from core.LoggingSetup import setup_logging

setup_logging(__name__)
logger = logging.getLogger(__name__)

try:
    from config.DynamicOperations import SANDBOX_TIMEOUT, RESTRICTED_BUILTINS
except ImportError:
    from ..config.DynamicOperations import SANDBOX_TIMEOUT, RESTRICTED_BUILTINS


def _sandbox_worker(code: str, result_queue: "mp.Queue"):
    """
    Worker function executed in the child process.

    Rebuilds its own sandbox namespace (cannot receive Mock objects from parent
    across process boundary) and executes the generated code.

    Args:
        code: Python source code string to execute
        result_queue: Queue to send (success, error_message) back to parent
    """
    try:
        safe_builtins = _build_safe_builtins()
        sandbox_globals = _build_sandbox_namespace(safe_builtins)
        exec(compile(code, "<generated_operation>", "exec"), sandbox_globals)
        error = _verify_sandbox_artifacts(sandbox_globals)
        if error:
            result_queue.put((False, error))
        else:
            result_queue.put((True, ""))
    except Exception as e:
        result_queue.put((False, f"{type(e).__name__}: {str(e)}"))


def validate_in_sandbox(code: str, timeout: Optional[float] = None) -> Tuple[bool, str]:
    """
    Execute generated code in a sandboxed environment.

    The sandbox provides:
    - Mocked get_command_broadcaster() and get_unified_image_storage()
    - Restricted builtins (no open, eval, exec, etc.)
    - Timeout enforcement via multiprocessing.Process (process can be killed on timeout)
    - Isolated namespace

    Uses fork start context so Mock objects in _build_sandbox_namespace do not need
    to be pickled across the process boundary. The child rebuilds its own namespace.

    Args:
        code: Python source code string to execute
        timeout: Execution timeout in seconds (default from config)

    Returns:
        Tuple of (is_valid, error_message).
        error_message is empty string on success.
    """
    timeout = timeout or SANDBOX_TIMEOUT

    # Use "spawn" context: avoids deadlocks caused by forking a multi-threaded
    # process (Python 3.12+ DeprecationWarning on fork). All args passed to
    # _sandbox_worker (code str + Queue) are picklable; Mock objects are built
    # inside the worker itself so no pickling issues arise.
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_sandbox_worker, args=(code, result_queue), daemon=True
    )
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join()
        error_msg = f"Sandbox execution timed out after {timeout}s"
        logger.warning(error_msg)
        return False, error_msg

    if result_queue.empty():
        error_msg = "No result returned from sandbox process"
        logger.warning(error_msg)
        return False, error_msg

    success, error = result_queue.get()
    if not success:
        logger.warning(f"Sandbox execution failed: {error}")
        return False, error

    logger.debug("Sandbox validation passed")
    return True, ""


def _build_safe_builtins() -> dict:
    """Build a restricted builtins dict excluding dangerous functions."""
    import builtins

    safe = {}
    for name in dir(builtins):
        if name not in RESTRICTED_BUILTINS and not name.startswith("_"):
            safe[name] = getattr(builtins, name)

    # Keep __build_class__ for class definitions
    safe["__build_class__"] = builtins.__build_class__
    # Keep __name__ for module-level code
    safe["__name__"] = "__generated__"

    # Provide a restricted __import__ that only allows safe modules
    safe["__import__"] = _make_restricted_import()

    return safe


# Modules that generated operations are allowed to import
_ALLOWED_MODULES = {
    "time",
    "logging",
    "json",
    "math",
    "re",
    "copy",
    "functools",
    "typing",
    "dataclasses",
    "enum",
    "collections",
    "operations",
    "operations.Base",
    "core",
    "core.Imports",
    "core.LoggingSetup",
}


def _make_restricted_import():
    """Create a restricted __import__ function that only allows safe modules."""
    real_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )

    def restricted_import(name, *args, **kwargs):
        """Import only from allowed modules."""
        from config.DynamicOperations import RESTRICTED_MODULES

        # Check top-level module
        top_module = name.split(".")[0]
        if top_module in RESTRICTED_MODULES:
            raise ImportError(
                f"Import of '{name}' is restricted in generated operations"
            )

        return real_import(name, *args, **kwargs)

    return restricted_import


def _build_sandbox_namespace(safe_builtins: dict) -> dict:
    """Build the sandbox namespace with mocked dependencies."""
    # Mock CommandBroadcaster
    mock_broadcaster = Mock()
    mock_broadcaster.send_command = Mock(return_value=True)

    # Mock UnifiedImageStorage
    mock_storage = Mock()
    mock_storage.get_single_image = Mock(return_value=None)
    mock_storage.get_stereo_pair = Mock(return_value=(None, None))

    namespace: Dict[str, Any] = {
        "__builtins__": safe_builtins,
    }

    # Pre-import allowed modules into the namespace
    # so the generated code can use them without import restrictions
    import time
    import logging as _logging
    import json
    from operations.Base import (
        BasicOperation,
        OperationCategory,
        OperationComplexity,
        OperationParameter,
        OperationResult,
        OperationRelationship,
    )

    namespace.update(
        {
            "time": time,
            "logging": _logging,
            "json": json,
            "Any": Any,
            "Dict": Dict,
            "Optional": Optional,
            "List": List,
            "BasicOperation": BasicOperation,
            "OperationCategory": OperationCategory,
            "OperationComplexity": OperationComplexity,
            "OperationParameter": OperationParameter,
            "OperationResult": OperationResult,
            "OperationRelationship": OperationRelationship,
            "_get_command_broadcaster": lambda: mock_broadcaster,
        }
    )

    return namespace


def _verify_sandbox_artifacts(namespace: dict) -> str:
    """
    Verify that the sandbox produced expected artifacts.

    Returns:
        Error message if verification fails, empty string on success.
    """
    # Check for *_OPERATION constant
    operation_constants = [
        key
        for key in namespace
        if key.endswith("_OPERATION") and not key.startswith("_")
    ]

    if not operation_constants:
        return "Sandbox execution did not produce a *_OPERATION constant"

    # Verify the constant is a BasicOperation instance
    from operations.Base import BasicOperation

    for const_name in operation_constants:
        value = namespace[const_name]
        if isinstance(value, BasicOperation):
            return ""

    return (
        f"*_OPERATION constant(s) found ({', '.join(operation_constants)}) "
        f"but none are BasicOperation instances"
    )
