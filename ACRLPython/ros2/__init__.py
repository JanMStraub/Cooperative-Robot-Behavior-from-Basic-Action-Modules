#!/usr/bin/env python3
"""
ROS 2 Integration Module
=========================

Provides a bridge between the Python backend and ROS 2 MoveIt for motion planning.

Architecture (Option B - TCP bridge):
- ROSMotionClient: ROS 2 node running in Docker, exposes TCP API on port 5020
- ROSBridge: Python client that connects to ROSMotionClient from the backend

This avoids requiring rclpy in the Python backend environment.
"""
