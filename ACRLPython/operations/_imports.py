#!/usr/bin/env python3
"""
Centralised lazy imports for the operations package.

All ``try: from ..X / except ImportError: from X`` blocks duplicated across 8+ operation files
consolidated here. Must NOT import from any other operations module (circular dependency risk).
"""

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
        PLACE_HOVER_OFFSET,
        PLACE_MIN_Y,
        PLACE_TCP_OFFSET,
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
        PLACE_HOVER_OFFSET,
        PLACE_MIN_Y,
        PLACE_TCP_OFFSET,
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
