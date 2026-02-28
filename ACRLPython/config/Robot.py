"""
Multi-Robot Coordination Configuration
========================================

Workspace regions, robot assignments, and coordination safety parameters.
"""

import os

# ============================================================================
# Workspace Region Definitions (meters, world coordinates)
# ============================================================================

WORKSPACE_REGIONS = {
    "left_workspace": {
        "x_min": -0.5,
        "x_max": -0.15,
        "y_min": 0.0,
        "y_max": 0.6,
        "z_min": -0.45,
        "z_max": 0.45,
    },
    "right_workspace": {
        "x_min": 0.15,
        "x_max": 0.5,
        "y_min": 0.0,
        "y_max": 0.6,
        "z_min": -0.45,
        "z_max": 0.45,
    },
    "shared_zone": {
        "x_min": -0.15,
        "x_max": 0.15,
        "y_min": 0.0,
        "y_max": 0.6,
        "z_min": -0.45,
        "z_max": 0.45,
    },
    "center": {
        "x_min": -0.15,
        "x_max": 0.15,
        "y_min": 0.0,
        "y_max": 0.5,
        "z_min": -0.1,
        "z_max": 0.1,
    },
}

# ============================================================================
# Robot Workspace Assignments (default allocation)
# ============================================================================

# TODO: use ROBOT_WORKSPACE_ASSIGNMENTS in WorkspaceManager default allocation
ROBOT_WORKSPACE_ASSIGNMENTS = {
    "Robot1": "left_workspace",
    "Robot2": "right_workspace",
}

# ============================================================================
# Robot Base Positions (world coordinates, meters)
# ============================================================================

# TODO: use ROBOT_BASE_POSITIONS in WorkspaceManager / collision avoidance
ROBOT_BASE_POSITIONS = {
    "Robot1": (-0.475, 0.0, 0.0),
    "Robot2": (0.475, 0.0, 0.0),
}

# ============================================================================
# Multi-Robot Coordination Safety Parameters
# ============================================================================

# TODO: use COLLISION_SAFETY_MARGIN in collision avoidance planner
COLLISION_SAFETY_MARGIN = float(os.environ.get("COLLISION_SAFETY_MARGIN", "0.01"))  # meters
MIN_ROBOT_SEPARATION = float(os.environ.get("MIN_ROBOT_SEPARATION", "0.2"))  # meters
MAX_ROBOT_REACH = float(os.environ.get("MAX_ROBOT_REACH", "0.8"))  # meters

# ============================================================================
# State Caching Configuration
# ============================================================================

ROBOT_STATUS_CACHE_TTL = float(os.environ.get("ROBOT_STATUS_CACHE_TTL", "0.5"))  # seconds
# TODO: use WORLD_STATE_UPDATE_INTERVAL in WorldStatePublisher polling rate
WORLD_STATE_UPDATE_INTERVAL = float(os.environ.get("WORLD_STATE_UPDATE_INTERVAL", "0.1"))  # seconds
WORKSPACE_ALLOCATION_TIMEOUT = float(os.environ.get("WORKSPACE_ALLOCATION_TIMEOUT", "60.0"))  # seconds

# ============================================================================
# Object Liveness Tracking Configuration
# ============================================================================

CONFIDENCE_DECAY_PER_FRAME = float(os.environ.get("CONFIDENCE_DECAY_PER_FRAME", "0.1"))  # Confidence drops 0.1 per missed detection
STALE_CONFIDENCE_THRESHOLD = float(os.environ.get("STALE_CONFIDENCE_THRESHOLD", "0.3"))  # Mark stale when confidence < 0.3
OBJECT_TTL_SECONDS = float(os.environ.get("OBJECT_TTL_SECONDS", "2.0"))  # Delete object if not seen for 2s
