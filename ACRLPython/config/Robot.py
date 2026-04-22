#!/usr/bin/env python3
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

# Used in operations/CoordinationVerifier.py for workspace allocation queries
ROBOT_WORKSPACE_ASSIGNMENTS = {
    "Robot1": "left_workspace",
    "Robot2": "right_workspace",
}

# ============================================================================
# Robot Base Positions (world coordinates, meters)
# ============================================================================

# Used in ros2/ROSMotionClient.py, servers/NegotiationHub.py, operations/SpatialOperations.py
ROBOT_BASE_POSITIONS = {
    "Robot1": (-0.475, 0.0, 0.0),
    "Robot2": (0.475, 0.0, 0.0),
}

# ============================================================================
# Handoff Geometry Parameters
# ============================================================================

# Default object dimensions (m) used when actual dims are unavailable.
# Matches the scene's standard 2 cm cube prefabs.
DEFAULT_HANDOFF_OBJECT_DIMENSIONS = (0.02, 0.02, 0.02)

# Extra clearance added to the half-extent offset so grippers don't overlap.
HANDOFF_GRIPPER_CLEARANCE = float(
    os.environ.get("HANDOFF_GRIPPER_CLEARANCE", "0.02")
)  # meters

# Fixed world-space position where Robot A presents the object for handoff.
# Chosen to be reachable by both AR4 arms and clear of the table surface.
# Robot B approaches this same point from the side (pitch=90°) while Robot A
# holds still top-down — the different gripper planes prevent finger collision.
HANDOFF_PRESENTATION_POSITION = (
    float(os.environ.get("HANDOFF_PRESENTATION_X", "0.0")),
    float(os.environ.get("HANDOFF_PRESENTATION_Y", "0.35")),
    float(os.environ.get("HANDOFF_PRESENTATION_Z", "0.0")),
)

# ============================================================================
# Follow-Target Configuration
# ============================================================================

# When True the arm re-plans to the live object position after each trajectory
# if the cube has drifted (e.g. pushed by the other robot's open fingers).
FOLLOW_TARGET_ENABLED = (
    os.environ.get("FOLLOW_TARGET_ENABLED", "true").lower() == "true"
)

# Maximum number of corrective moves before closing the gripper regardless.
FOLLOW_TARGET_MAX_CORRECTIONS = int(
    os.environ.get("FOLLOW_TARGET_MAX_CORRECTIONS", "3")
)

# Minimum object drift (meters) that triggers a corrective plan_and_execute.
FOLLOW_TARGET_DRIFT_THRESHOLD = float(
    os.environ.get("FOLLOW_TARGET_DRIFT_THRESHOLD", "0.015")
)

# ============================================================================
# Grasp Geometry & Approach Parameters
# ============================================================================

# Physical offset from ee_link to gripper fingertip plane (meters).
# AR4: ee_link → fingertip = 0.05m, finger length = 0.02m.
# ee_link stops this far above the object centre so the fingertips land at
# the object centre and wrap around it.
GRASP_TCP_OFFSET = float(os.environ.get("GRASP_TCP_OFFSET", "0.05"))  # meters

# Hover height above object centre for the pre-grasp approach waypoint.
# Must be high enough that the arm clears the object when it swings in.
PRE_GRASP_HOVER_OFFSET = float(
    os.environ.get("PRE_GRASP_HOVER_OFFSET", "0.15")
)  # meters

# Absolute world-Y safety height used as an intermediate waypoint before descent.
# The arm moves here first (joint-space) so it cannot sweep through table-height
# objects on its way to the pre-grasp position. Should be above the tallest object.
PRE_GRASP_CLEARANCE_Y = float(os.environ.get("PRE_GRASP_CLEARANCE_Y", "0.35"))  # meters

# Velocity/acceleration scaling for the joint-space pre-grasp approach move.
# Full speed (1.0) causes residual vibration on arrival and risks overshoot into
# the object; lower values give the arm time to decelerate cleanly.
PREGRASP_VELOCITY_SCALING = float(os.environ.get("PREGRASP_VELOCITY_SCALING", "1.0"))
PREGRASP_ACCELERATION_SCALING = float(
    os.environ.get("PREGRASP_ACCELERATION_SCALING", "1.0")
)

# Velocity/acceleration scaling for the final Cartesian descent to grasp position.
# Must be slow enough that the gripper doesn't slam into the object on arrival.
GRASP_DESCENT_VELOCITY_SCALING = float(
    os.environ.get("GRASP_DESCENT_VELOCITY_SCALING", "0.7")
)
GRASP_DESCENT_ACCELERATION_SCALING = float(
    os.environ.get("GRASP_DESCENT_ACCELERATION_SCALING", "0.5")
)

# ============================================================================
# Multi-Robot Coordination Safety Parameters
# ============================================================================

# Used in operations/SpatialPredicates.py for collision avoidance safety margins
COLLISION_SAFETY_MARGIN = float(
    os.environ.get("COLLISION_SAFETY_MARGIN", "0.01")
)  # meters
MIN_ROBOT_SEPARATION = float(os.environ.get("MIN_ROBOT_SEPARATION", "0.2"))  # meters
MAX_ROBOT_REACH = float(
    os.environ.get("MAX_ROBOT_REACH", "0.64")
)  # meters (AR4 kinematic limit)

# ============================================================================
# State Caching Configuration
# ============================================================================

ROBOT_STATUS_CACHE_TTL = float(
    os.environ.get("ROBOT_STATUS_CACHE_TTL", "0.5")
)  # seconds
# WorldStateServer is push-based (Unity initiates); this interval is available for future use
WORLD_STATE_UPDATE_INTERVAL = float(
    os.environ.get("WORLD_STATE_UPDATE_INTERVAL", "0.1")
)  # seconds
WORKSPACE_ALLOCATION_TIMEOUT = float(
    os.environ.get("WORKSPACE_ALLOCATION_TIMEOUT", "60.0")
)  # seconds

# ============================================================================
# Object Liveness Tracking Configuration
# ============================================================================

CONFIDENCE_DECAY_PER_FRAME = float(
    os.environ.get("CONFIDENCE_DECAY_PER_FRAME", "0.1")
)  # Confidence drops 0.1 per missed detection
STALE_CONFIDENCE_THRESHOLD = float(
    os.environ.get("STALE_CONFIDENCE_THRESHOLD", "0.3")
)  # Mark stale when confidence < 0.3
OBJECT_TTL_SECONDS = float(
    os.environ.get("OBJECT_TTL_SECONDS", "30.0")
)  # Delete object if not seen for 2s

# ============================================================================
# FK-based Movement Detection
# ============================================================================

# Minimum total joint angle delta (radians) to mark a robot as is_moving.
# Sum of |Δθᵢ| across all 6 joints must exceed this threshold.
JOINT_MOVEMENT_THRESHOLD = float(os.environ.get("JOINT_MOVEMENT_THRESHOLD", "0.001"))
