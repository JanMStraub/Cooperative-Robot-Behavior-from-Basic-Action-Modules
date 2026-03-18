#!/usr/bin/env python3
"""
ROS Integration Configuration
==============================

Central configuration for ROS 2 / MoveIt integration.
Controls whether the Python backend routes motion commands through ROS or
the existing TCP path to Unity.
"""

# Master switch: enable/disable ROS integration
ROS_ENABLED = True

# ROS bridge connection settings (ROSMotionClient in Docker)
ROS_BRIDGE_HOST = "127.0.0.1"
ROS_BRIDGE_PORT = 5020

# MoveIt planning settings — used in ros2/ROSMotionClient.py planning requests
MOVEIT_PLANNING_TIME = 5.0  # Max planning time in seconds
MOVEIT_PLANNING_ATTEMPTS = 10  # Number of planning attempts
MOVEIT_GOAL_TOLERANCE = 0.01  # Position goal tolerance in meters

# Default control mode: "ros", "unity", or "hybrid"
# - "ros": All movement routed through MoveIt (requires Docker running)
# - "unity": All movement via existing TCP to Unity IK (default, existing behavior)
# - "hybrid": Try ROS first, fall back to Unity on failure
DEFAULT_CONTROL_MODE = "ros"

# Auto-connect to ROS bridge on startup (wired into RunRobotController._auto_connect_ros)
AUTO_CONNECT_ROS = True

# Timeout for waiting on ROS motion execution — used in ros2/ROSBridge.py
ROS_EXECUTION_TIMEOUT = 30.0

# Grasp validation timeout: base + per-candidate increment
ROS_TIMEOUT_BASE = 5.0
ROS_TIMEOUT_PER_CANDIDATE = 0.5

# URDF joint limits for the 6-DOF AR4 arm (radians).
# Used by ROSMotionClient to clamp start states before submitting to MoveIt and to
# build path constraints. Values match the URDF joint limit tags in ar4.urdf.
ARM_JOINT_LIMITS = {
    "joint_1": (-2.9670597283903604, 2.9670597283903604),
    "joint_2": (-0.7330382858376184, 1.5707963267948966),
    "joint_3": (-1.5533430342749532, 0.9075712110370514),
    "joint_4": (-3.141592653589793, 3.141592653589793),
    "joint_5": (-1.8325957145940461, 1.8325957145940461),
    "joint_6": (-3.141592653589793, 3.141592653589793),
}
