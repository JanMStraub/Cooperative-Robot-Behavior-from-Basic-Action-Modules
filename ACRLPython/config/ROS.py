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

# MoveIt planning settings
MOVEIT_PLANNING_TIME = 5.0          # Max planning time in seconds
MOVEIT_PLANNING_ATTEMPTS = 10       # Number of planning attempts
MOVEIT_GOAL_TOLERANCE = 0.01        # Position goal tolerance in meters

# Default control mode: "ros", "unity", or "hybrid"
# - "ros": All movement routed through MoveIt (requires Docker running)
# - "unity": All movement via existing TCP to Unity IK (default, existing behavior)
# - "hybrid": Try ROS first, fall back to Unity on failure
DEFAULT_CONTROL_MODE = "ros"

# Auto-connect to ROS bridge on startup
AUTO_CONNECT_ROS = True

# Timeout for waiting on ROS motion execution (seconds)
ROS_EXECUTION_TIMEOUT = 30.0
