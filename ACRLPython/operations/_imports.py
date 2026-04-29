#!/usr/bin/env python3
"""
Centralised lazy imports for the operations package.
=====================================================

All ``try: from ..X import Y / except ImportError: from X import Y`` blocks
that were duplicated across 8+ operation files are consolidated here.

Usage in operation files::

    from ._imports import get_command_broadcaster
    from ._imports import get_unified_image_storage, get_world_state
    from ._imports import (
        ENABLE_VISION_STREAMING,
        VISION_OPERATION_TIMEOUT,
        DEFAULT_CAMERA_ID,
        DEFAULT_LMSTUDIO_MODEL,
    )

Design notes:
- Only ``core.Imports`` lazy singletons and ``config.*`` constants belong here.
- ``from .Base import`` shims are NOT needed — the package-relative form always
  works when operations are loaded through the package (i.e. always).
- This module must NOT import from any other operations module to stay
  importable by all of them without causing circular dependencies.
"""

# ---------------------------------------------------------------------------
# core.Imports lazy singletons
# ---------------------------------------------------------------------------
try:
    from ..core.Imports import get_command_broadcaster
    from ..core.Imports import get_unified_image_storage
    from ..core.Imports import get_world_state
    from ..core.Imports import get_global_registry
    from ..core.Imports import get_negotiation_hub
except ImportError:
    from core.Imports import get_command_broadcaster  # type: ignore[no-redef]
    from core.Imports import get_unified_image_storage  # type: ignore[no-redef]
    from core.Imports import get_world_state  # type: ignore[no-redef]
    from core.Imports import get_global_registry  # type: ignore[no-redef]
    from core.Imports import get_negotiation_hub  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# config.Robot constants
# ---------------------------------------------------------------------------
try:
    from ..config.Robot import (
        FOLLOW_TARGET_DRIFT_THRESHOLD,
        FOLLOW_TARGET_ENABLED,
        FOLLOW_TARGET_MAX_CORRECTIONS,
        GRASP_DESCENT_ACCELERATION_SCALING,
        GRASP_DESCENT_VELOCITY_SCALING,
        GRASP_TCP_OFFSET,
        DEFAULT_HANDOFF_OBJECT_DIMENSIONS,
        HANDOFF_GRIPPER_CLEARANCE,
        PRE_GRASP_CLEARANCE_Y,
        PRE_GRASP_HOVER_OFFSET,
        PREGRASP_ACCELERATION_SCALING,
        PREGRASP_VELOCITY_SCALING,
        WORKSPACE_REGIONS,
        ROBOT_STATUS_CACHE_TTL,
        WORKSPACE_ALLOCATION_TIMEOUT,
        CONFIDENCE_DECAY_PER_FRAME,
        STALE_CONFIDENCE_THRESHOLD,
        OBJECT_TTL_SECONDS,
        JOINT_MOVEMENT_THRESHOLD,
        ROBOT_BASE_POSITIONS,
    )
except ImportError:
    from config.Robot import (  # type: ignore[no-redef]
        FOLLOW_TARGET_DRIFT_THRESHOLD,
        FOLLOW_TARGET_ENABLED,
        FOLLOW_TARGET_MAX_CORRECTIONS,
        GRASP_DESCENT_ACCELERATION_SCALING,
        GRASP_DESCENT_VELOCITY_SCALING,
        GRASP_TCP_OFFSET,
        DEFAULT_HANDOFF_OBJECT_DIMENSIONS,
        HANDOFF_GRIPPER_CLEARANCE,
        PRE_GRASP_CLEARANCE_Y,
        PRE_GRASP_HOVER_OFFSET,
        PREGRASP_ACCELERATION_SCALING,
        PREGRASP_VELOCITY_SCALING,
        WORKSPACE_REGIONS,
        ROBOT_STATUS_CACHE_TTL,
        WORKSPACE_ALLOCATION_TIMEOUT,
        CONFIDENCE_DECAY_PER_FRAME,
        STALE_CONFIDENCE_THRESHOLD,
        OBJECT_TTL_SECONDS,
        JOINT_MOVEMENT_THRESHOLD,
        ROBOT_BASE_POSITIONS,
    )

# ---------------------------------------------------------------------------
# config.Vision + config.Servers constants
# ---------------------------------------------------------------------------
try:
    from ..config.Vision import (
        ENABLE_VISION_STREAMING,
        VISION_OPERATION_TIMEOUT,
        DEFAULT_CAMERA_ID,
    )
    from ..config.Servers import DEFAULT_LMSTUDIO_MODEL
except ImportError:
    from config.Vision import (  # type: ignore[no-redef]
        ENABLE_VISION_STREAMING,
        VISION_OPERATION_TIMEOUT,
        DEFAULT_CAMERA_ID,
    )
    from config.Servers import DEFAULT_LMSTUDIO_MODEL  # type: ignore[no-redef]
