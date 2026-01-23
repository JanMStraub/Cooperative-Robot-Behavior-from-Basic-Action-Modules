"""
Centralized Lazy Import System
================================

This module provides centralized lazy import functions to break circular dependencies across the Python codebase.

Circular Dependency Chain (without lazy imports):
    operations/DetectionOperations → servers/ImageServer →
    servers/__init__ → servers/SequenceServer →
    orchestrators/SequenceExecutor → operations/Registry →
    operations/DetectionOperations (CIRCULAR!)

Usage:
    from operations.imports import get_command_broadcaster, get_world_state

    # Instead of:
    # from servers.CommandServer import get_command_broadcaster  # Circular!

    # Use:
    broadcaster = get_command_broadcaster()

Design Notes:
    - All imports are lazy (deferred until function call)
    - Thread-safe singleton caching where appropriate
    - Clear error messages if imports fail
    - Minimal overhead for repeated calls
"""

import threading

# ============================================================================
# Cached Instances (Thread-Safe Singletons)
# ============================================================================

_cached_instances = {}
_cache_lock = threading.RLock()


def _get_cached(key: str, factory_fn):
    """
    Get or create a cached singleton instance.

    Args:
        key: Cache key for the instance
        factory_fn: Function to create the instance if not cached

    Returns:
        Cached or newly created instance
    """
    if key not in _cached_instances:
        with _cache_lock:
            if key not in _cached_instances:
                _cached_instances[key] = factory_fn()
    return _cached_instances[key]


# ============================================================================
# Server Components
# ============================================================================


def get_command_broadcaster():
    """
    Get the CommandBroadcaster singleton instance.

    Returns:
        CommandBroadcaster instance for sending commands to Unity

    Raises:
        ImportError: If CommandServer module cannot be imported
        RuntimeError: If CommandBroadcaster not initialized

    Used by:
        - operations/MoveOperations.py
        - operations/GripperOperations.py
        - orchestrators/SequenceExecutor.py
    """
    try:
        from servers.CommandServer import get_command_broadcaster as _get_cb

        return _get_cb()
    except ImportError as e:
        raise ImportError(
            f"Failed to import CommandBroadcaster. Ensure CommandServer is properly initialized. Error: {e}"
        )


def get_unified_image_storage():
    """
    Get the UnifiedImageStorage singleton instance.

    Returns:
        UnifiedImageStorage instance for accessing camera images

    Raises:
        ImportError: If ImageStorageCore module cannot be imported

    Used by:
        - operations/DetectionOperations.py
        - operations/VisionOperations.py
    """
    try:
        from servers.ImageStorageCore import UnifiedImageStorage

        return UnifiedImageStorage()
    except ImportError as e:
        raise ImportError(
            f"Failed to import UnifiedImageStorage. Error: {e}"
        )


# ============================================================================
# Operation Components
# ============================================================================


def get_global_registry():
    """
    Get the global OperationRegistry singleton instance.

    Returns:
        OperationRegistry with all registered operations

    Raises:
        ImportError: If Registry module cannot be imported

    Used by:
        - orchestrators/CommandParser.py
        - orchestrators/SequenceExecutor.py
        - servers/SequenceServer.py
    """
    try:
        from operations.Registry import get_global_registry as _get_registry

        return _get_registry()
    except ImportError as e:
        raise ImportError(f"Failed to import OperationRegistry. Error: {e}")


def get_world_state():
    """
    Get the WorldState singleton instance.

    Returns:
        WorldState instance for robot/object tracking

    Raises:
        ImportError: If WorldState module cannot be imported

    Used by:
        - operations/MoveOperations.py
        - operations/SpatialOperations.py
        - orchestrators/SequenceExecutor.py
    """
    try:
        from operations.WorldState import get_world_state as _get_ws

        return _get_ws()
    except ImportError as e:
        raise ImportError(f"Failed to import WorldState. Error: {e}")


def get_robot_config():
    """
    Get robot configuration module with workspace regions and base positions.

    Returns:
        Module with WORKSPACE_REGIONS, ROBOT_BASE_POSITIONS, and other constants

    Raises:
        ImportError: If config.Robot module cannot be imported

    Used by:
        - operations/SpatialOperations.py
        - operations/CoordinationOperations.py
    """
    try:
        from config import Robot as robot_config

        return robot_config
    except ImportError as e:
        raise ImportError(f"Failed to import robot config. Error: {e}")


# ============================================================================
# Orchestrator Components
# ============================================================================


def get_command_parser(**kwargs):
    """
    Create a CommandParser instance.

    Args:
        **kwargs: Arguments to pass to CommandParser constructor

    Returns:
        CommandParser instance

    Raises:
        ImportError: If CommandParser module cannot be imported

    Used by:
        - servers/SequenceServer.py

    Note: Not cached - each call creates a new instance
    """
    try:
        from orchestrators.CommandParser import CommandParser

        return CommandParser(**kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import CommandParser. Error: {e}")


def get_sequence_executor(**kwargs):
    """
    Create a SequenceExecutor instance.

    Args:
        **kwargs: Arguments to pass to SequenceExecutor constructor

    Returns:
        SequenceExecutor instance

    Raises:
        ImportError: If SequenceExecutor module cannot be imported

    Used by:
        - servers/SequenceServer.py

    Note: Not cached - each call creates a new instance
    """
    try:
        from orchestrators.SequenceExecutor import SequenceExecutor

        return SequenceExecutor(**kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import SequenceExecutor. Error: {e}")


# ============================================================================
# Utility Functions
# ============================================================================


def clear_import_cache():
    """
    Clear all cached singleton instances.

    Useful for testing or when reinitializing the system.
    """
    with _cache_lock:
        _cached_instances.clear()


def get_cached_instances():
    """
    Get all currently cached instances (for debugging).

    Returns:
        Dict of cached instances
    """
    with _cache_lock:
        return dict(_cached_instances)
