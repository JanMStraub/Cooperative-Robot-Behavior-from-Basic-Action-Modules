#!/usr/bin/env python3
"""
Interface.py - Abstract Hardware Interface for Robot Motion Execution

Defines the stable contract (port) that all hardware backends must implement.
Concrete adapters (UnityInterface, ROSInterface) plug into this ABC so that
operation code never needs to know which backend is active.
"""

from abc import ABC, abstractmethod


class RobotHardwareInterface(ABC):
    """Abstract motion execution backend (Unity, ROS/MoveIt, physical drivers)."""

    @abstractmethod
    def move_to(self, robot_id: str, x: float, y: float, z: float, **kwargs) -> bool:
        """Move end-effector to Cartesian position; returns True on success."""

    @abstractmethod
    def set_gripper(self, robot_id: str, open: bool) -> bool:
        """Open (True) or close (False) the gripper; returns True if accepted."""

    @abstractmethod
    def get_joint_states(self, robot_id: str) -> list[float]:
        """Return current joint angles [j1..j6] in radians."""

    @abstractmethod
    def emergency_stop(self) -> bool:
        """Send emergency stop to all robots; returns True if dispatched."""
