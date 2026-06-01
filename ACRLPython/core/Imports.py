#!/usr/bin/env python3
"""
Centralized lazy import functions to break circular dependencies.

Circular chain avoided:
    operations/DetectionOperations → servers/ImageServer →
    servers/__init__ → servers/SequenceServer →
    orchestrators/SequenceExecutor → operations/Registry →
    operations/DetectionOperations (CIRCULAR!)
"""

import threading

# Session-scoped abort event. Set when a sequence is cancelled/timed out so that
# long-running background operations (e.g. follow_target in grasp) stop between
# steps instead of continuing after reset_simulation fires.
_sequence_abort_event = threading.Event()


def signal_sequence_abort() -> None:
    """Signal all background operations to stop at their next checkpoint."""
    _sequence_abort_event.set()


def clear_sequence_abort() -> None:
    """Clear the abort signal at the start of a new sequence."""
    _sequence_abort_event.clear()


def is_sequence_aborted() -> bool:
    """Return True if the current sequence has been cancelled."""
    return _sequence_abort_event.is_set()


def get_command_broadcaster():
    """Get the CommandBroadcaster singleton instance."""
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

    UnifiedImageStorage uses __new__-based singleton enforcement, so calling
    UnifiedImageStorage() always returns the same underlying instance.
    """
    try:
        from servers.ImageStorageCore import UnifiedImageStorage

        return UnifiedImageStorage()
    except ImportError as e:
        raise ImportError(f"Failed to import UnifiedImageStorage. Error: {e}")


def get_global_registry():
    """Get the global OperationRegistry singleton instance."""
    try:
        from operations.Registry import get_global_registry as _get_registry

        return _get_registry()
    except ImportError as e:
        raise ImportError(f"Failed to import OperationRegistry. Error: {e}")


def get_world_state():
    """Get the WorldState singleton instance."""
    try:
        from operations.WorldState import get_world_state as _get_ws

        return _get_ws()
    except ImportError as e:
        raise ImportError(f"Failed to import WorldState. Error: {e}")


def get_robot_config():
    """Get robot configuration module with workspace regions and base positions."""
    try:
        from config import Robot as robot_config

        return robot_config
    except ImportError as e:
        raise ImportError(f"Failed to import robot config. Error: {e}")


def get_command_parser(**kwargs):
    """Create a CommandParser instance. Not cached — each call creates a new instance."""
    try:
        from orchestrators.CommandParser import CommandParser

        return CommandParser(**kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import CommandParser. Error: {e}")


def get_sequence_executor(**kwargs):
    """Create a SequenceExecutor instance. Not cached — each call creates a new instance."""
    try:
        from orchestrators.SequenceExecutor import SequenceExecutor

        return SequenceExecutor(**kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import SequenceExecutor. Error: {e}")


def get_negotiation_hub():
    """Get NegotiationHub singleton, or None if negotiation is disabled."""
    try:
        import config.Negotiation as neg_cfg

        if not neg_cfg.NEGOTIATION_ENABLED:
            return None
        from servers.NegotiationHub import NegotiationHub

        return NegotiationHub()
    except ImportError as e:
        raise ImportError(f"Failed to import NegotiationHub. Error: {e}")


def get_graph_query_engine():
    """
    Get the GraphQueryEngine singleton instance (if knowledge graph is enabled).

    KnowledgeGraph and GraphQueryEngine are created lazily on first access and
    live for the entire process lifetime via module-level singletons in
    knowledge_graph._singleton.

    Returns None if KG is disabled.
    """
    try:
        from knowledge_graph._singleton import get_query_engine

        return get_query_engine()
    except ImportError as e:
        raise ImportError(f"Failed to import GraphQueryEngine. Error: {e}")


def get_hardware_interface(env: str = "sim"):
    """
    Get the RobotHardwareInterface singleton for the active execution environment.

    On the first call the env argument determines which adapter is instantiated.
    Subsequent calls ignore env and return the cached singleton.
    """
    try:
        from hardware import get_hardware_interface as _get_hw

        return _get_hw(env=env)
    except ImportError as e:
        raise ImportError(f"Failed to import hardware interface. Error: {e}")


def get_camera_provider(env: str = "sim"):
    """Get the CameraProvider singleton for the active execution environment."""
    try:
        from camera import get_camera_provider as _get_cam

        return _get_cam(env=env)
    except ImportError as e:
        raise ImportError(f"Failed to import camera provider. Error: {e}")


def get_perception_refresh_daemon():
    """Get the active PerceptionRefreshLoop instance if one has been started.

    Returns:
        PerceptionRefreshLoop instance or None if not running.
    """
    try:
        from operations.PerceptionRefresh import _active_refresh_loop

        return _active_refresh_loop
    except (ImportError, AttributeError):
        return None
