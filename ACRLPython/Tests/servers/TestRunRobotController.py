#!/usr/bin/env python3
"""
Unit tests for RobotController ROS auto-connect behaviour.
"""

import threading
import pytest
from unittest.mock import patch, MagicMock


class TestAutoConnectROS:
    """Tests for AUTO_CONNECT_ROS wiring in RobotController."""

    @patch("orchestrators.RunRobotController.ROSBridge")
    @patch("orchestrators.RunRobotController.ROS_ENABLED", True)
    @patch("orchestrators.RunRobotController.AUTO_CONNECT_ROS", True)
    def test_auto_connect_called_when_both_flags_true(self, mock_ros_bridge_cls):
        """ROS bridge connect() is called when ROS_ENABLED=True and AUTO_CONNECT_ROS=True."""
        done = threading.Event()
        mock_instance = MagicMock()
        mock_ros_bridge_cls.get_instance.return_value = mock_instance
        mock_instance.connect.side_effect = lambda: done.set() or True

        from orchestrators.RunRobotController import RobotController

        controller = RobotController.__new__(RobotController)
        controller._auto_connect_ros()

        assert done.wait(timeout=2.0), "ROS bridge connect() was never called"
        mock_ros_bridge_cls.get_instance.assert_called_once()
        mock_instance.connect.assert_called_once()

    @patch("orchestrators.RunRobotController.ROSBridge")
    @patch("orchestrators.RunRobotController.ROS_ENABLED", False)
    @patch("orchestrators.RunRobotController.AUTO_CONNECT_ROS", True)
    def test_no_connect_when_ros_disabled(self, mock_ros_bridge_cls):
        """ROS bridge connect() is NOT called when ROS_ENABLED=False."""
        from orchestrators.RunRobotController import RobotController

        controller = RobotController.__new__(RobotController)
        controller._auto_connect_ros()

        # No thread is spawned, so assert immediately
        mock_ros_bridge_cls.get_instance.assert_not_called()

    @patch("orchestrators.RunRobotController.ROSBridge")
    @patch("orchestrators.RunRobotController.ROS_ENABLED", True)
    @patch("orchestrators.RunRobotController.AUTO_CONNECT_ROS", False)
    def test_no_connect_when_auto_connect_disabled(self, mock_ros_bridge_cls):
        """ROS bridge connect() is NOT called when AUTO_CONNECT_ROS=False."""
        from orchestrators.RunRobotController import RobotController

        controller = RobotController.__new__(RobotController)
        controller._auto_connect_ros()

        # No thread is spawned, so assert immediately
        mock_ros_bridge_cls.get_instance.assert_not_called()

    @patch("orchestrators.RunRobotController.ROSBridge")
    @patch("orchestrators.RunRobotController.ROS_ENABLED", True)
    @patch("orchestrators.RunRobotController.AUTO_CONNECT_ROS", True)
    def test_connect_failure_does_not_raise(self, mock_ros_bridge_cls):
        """A failed connect() logs a warning but does not propagate an exception."""
        done = threading.Event()
        mock_instance = MagicMock()
        mock_ros_bridge_cls.get_instance.return_value = mock_instance
        mock_instance.connect.side_effect = lambda: done.set() or False

        from orchestrators.RunRobotController import RobotController

        controller = RobotController.__new__(RobotController)
        controller._auto_connect_ros()

        assert done.wait(timeout=2.0), "connect() was never called"
        # Test passes if no exception was raised
