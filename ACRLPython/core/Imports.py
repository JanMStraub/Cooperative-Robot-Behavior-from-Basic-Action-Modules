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
    - Singleton management is delegated to each underlying module
    - Clear error messages if imports fail
    - Minimal overhead for repeated calls
"""

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

    UnifiedImageStorage uses __new__-based singleton enforcement, so calling
    UnifiedImageStorage() always returns the same underlying instance.

    Returns:
        UnifiedImageStorage singleton for accessing camera images

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
        raise ImportError(f"Failed to import UnifiedImageStorage. Error: {e}")


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
# Negotiation Components
# ============================================================================


def get_negotiation_hub():
    """
    Get NegotiationHub singleton, or None if negotiation is disabled.

    Returns:
        NegotiationHub instance, or None if NEGOTIATION_ENABLED is False

    Raises:
        ImportError: If NegotiationHub module cannot be imported

    Used by:
        - orchestrators/SequenceExecutor.py
    """
    try:
        import config.Negotiation as neg_cfg

        if not neg_cfg.NEGOTIATION_ENABLED:
            return None
        from servers.NegotiationHub import NegotiationHub

        return NegotiationHub()
    except ImportError as e:
        raise ImportError(f"Failed to import NegotiationHub. Error: {e}")


# ============================================================================
# Knowledge Graph Components
# ============================================================================


def get_graph_query_engine():
    """
    Get the GraphQueryEngine singleton instance (if knowledge graph is enabled).

    The underlying KnowledgeGraph and GraphQueryEngine are created lazily on
    first access and live for the entire process lifetime via module-level
    singletons in knowledge_graph._singleton.

    Returns:
        GraphQueryEngine instance, or None if KG is disabled

    Raises:
        ImportError: If the knowledge_graph package cannot be imported

    Used by:
        - Any operation or orchestrator needing spatial graph queries
        - orchestrators/RunRobotController.py (wiring only)
    """
    try:
        from knowledge_graph._singleton import get_query_engine

        return get_query_engine()
    except ImportError as e:
        raise ImportError(f"Failed to import GraphQueryEngine. Error: {e}")


# ============================================================================
# Hardware Interface
# ============================================================================


def get_hardware_interface(env: str = "sim"):
    """
    Get the RobotHardwareInterface singleton for the active execution environment.

    On the first call (typically from RunRobotController) the env argument
    determines which adapter is instantiated.  Subsequent calls ignore env
    and return the cached singleton.

    Args:
        env: "sim" for Unity (default), "real" for ROS/MoveIt.

    Returns:
        Concrete RobotHardwareInterface adapter (UnityHardwareInterface or
        ROSHardwareInterface).

    Raises:
        ImportError: If the hardware package cannot be imported.

    Used by:
        - orchestrators/RunRobotController.py (initialisation)
        - Any operation that needs hardware-agnostic motion commands
    """
    try:
        from hardware import get_hardware_interface as _get_hw

        return _get_hw(env=env)
    except ImportError as e:
        raise ImportError(f"Failed to import hardware interface. Error: {e}")


# ============================================================================
# Camera Provider
# ============================================================================


def get_camera_provider(env: str = "sim"):
    """
    Get the CameraProvider singleton for the active execution environment.

    Args:
        env: "sim" for Unity image storage (default), "real" for local USB/RealSense.

    Returns:
        Concrete CameraProvider adapter.

    Raises:
        ImportError: If the camera package cannot be imported.

    Used by:
        - Any operation or vision component needing environment-agnostic camera access
    """
    try:
        from camera import get_camera_provider as _get_cam

        return _get_cam(env=env)
    except ImportError as e:
        raise ImportError(f"Failed to import camera provider. Error: {e}")
