#!/usr/bin/env python3
"""
Interface.py - Abstract Hardware Interface for Robot Motion Execution

Defines the stable contract (port) that all hardware backends must implement.
Concrete adapters (UnityInterface, ROSInterface) plug into this ABC so that
operation code never needs to know which backend is active.
"""

from abc import ABC, abstractmethod


class RobotHardwareInterface(ABC):
    """
    Abstract motion execution backend.

    All concrete backends (Unity, ROS/MoveIt, physical drivers) must implement
    every method below.  Operations should call these methods exclusively —
    never import a concrete implementation directly.
    """

    @abstractmethod
    def move_to(self, robot_id: str, x: float, y: float, z: float, **kwargs) -> bool:
        """
        Move the specified robot's end-effector to a Cartesian position.

        Args:
            robot_id: Identifier of the target robot (e.g. "Robot1")
            x: Target X coordinate in world space (metres)
            y: Target Y coordinate in world space (metres)
            z: Target Z coordinate in world space (metres)
            **kwargs: Backend-specific options (e.g. speed, frame)

        Returns:
            True if the motion completed successfully, False otherwise
        """

    @abstractmethod
    def set_gripper(self, robot_id: str, open: bool) -> bool:
        """
        Open or close the gripper for the specified robot.

        Args:
            robot_id: Identifier of the target robot
            open: True to open the gripper, False to close

        Returns:
            True if the gripper command was accepted, False otherwise
        """

    @abstractmethod
    def get_joint_states(self, robot_id: str) -> list[float]:
        """
        Return current joint angles in radians for the specified robot.

        Args:
            robot_id: Identifier of the target robot

        Returns:
            List of joint angles [j1, j2, j3, j4, j5, j6] in radians
        """

    @abstractmethod
    def emergency_stop(self) -> bool:
        """
        Send an emergency stop command to all robots.

        Returns:
            True if the stop command was dispatched, False otherwise
        """
