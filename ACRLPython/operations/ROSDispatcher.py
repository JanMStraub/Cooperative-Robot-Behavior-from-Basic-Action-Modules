#!/usr/bin/env python3
"""
ROSDispatcher - Centralised ROS-vs-TCP routing for robot operations.

Eliminates the ~30-line ROS routing block that was duplicated across 13+
operation functions. Each operation delegates to ``execute_with_ros_fallback``
and passes two callables: a ROS path and a TCP path.

Usage::

    from operations.ROSDispatcher import execute_with_ros_fallback

    return execute_with_ros_fallback(
        ros_func=lambda: _move_via_ros(robot_id, x, y, z),
        tcp_func=lambda: _move_via_tcp(robot_id, x, y, z),
        use_ros=use_ros,
    )

The function handles:
- Reading ``ROS_ENABLED`` / ``DEFAULT_CONTROL_MODE`` from config (once per call).
- Connecting the ROSBridge if not yet connected.
- Falling back to ``tcp_func`` in hybrid mode when ROS is unavailable.
- Returning a structured ``OperationResult`` error in strict ROS mode.
"""

import logging
from typing import Callable, Optional

from .Base import OperationResult

logger = logging.getLogger(__name__)


def _is_ros_enabled(use_ros: Optional[bool]) -> bool:
    """
    Resolve whether ROS should be used for this call.

    Args:
        use_ros: Caller-provided override; None means auto-detect from config.

    Returns:
        True if ROS routing should be attempted.
    """
    if use_ros is not None:
        return bool(use_ros)
    try:
        from config.ROS import ROS_ENABLED, DEFAULT_CONTROL_MODE

        return bool(ROS_ENABLED and DEFAULT_CONTROL_MODE in ("ros", "hybrid"))
    except ImportError:
        return False


def _get_control_mode() -> str:
    """
    Return DEFAULT_CONTROL_MODE from config.

    Returns:
        The configured control mode string, or 'ros' if config is unavailable.
    """
    try:
        from config.ROS import DEFAULT_CONTROL_MODE

        return DEFAULT_CONTROL_MODE
    except ImportError:
        return "ros"


def execute_with_ros_fallback(
    ros_func: Callable[[], Optional[OperationResult]],
    tcp_func: Callable[[], OperationResult],
    use_ros: Optional[bool] = None,
) -> OperationResult:
    """
    Execute via ROS if enabled; fall back to TCP in hybrid mode.

    Tries the ROS path first when ``use_ros`` is True (or auto-detected as
    True from config). In hybrid mode a failed ROS attempt falls back to the
    TCP path transparently. In strict ROS mode a failed attempt returns an
    error OperationResult immediately.

    Args:
        ros_func: Callable that executes the operation via ROS/MoveIt and
                  returns an OperationResult, or None to signal failure to
                  the dispatcher (triggers fallback in hybrid mode).
        tcp_func: Callable that executes the operation via TCP (Unity) and
                  returns an OperationResult.
        use_ros:  Explicit ROS override. None = auto-detect from config.

    Returns:
        OperationResult from whichever path was taken.
    """
    if not _is_ros_enabled(use_ros):
        return tcp_func()

    try:
        from ros2.ROSBridge import ROSBridge

        bridge = ROSBridge.get_instance()

        # Attempt connection if not already connected
        if not bridge.is_connected:
            if not bridge.connect():
                if _get_control_mode() == "hybrid":
                    logger.warning("ROS bridge unavailable, falling back to TCP")
                    return tcp_func()
                return OperationResult.error_result(
                    "ROS_CONNECTION_FAILED",
                    "Failed to connect to ROS bridge (port 5020)",
                    [
                        "Ensure Docker ROS services are running: "
                        "cd rosUnityIntegration && ./start_ros_endpoint.sh",
                        "Set DEFAULT_CONTROL_MODE='hybrid' in config/ROS.py "
                        "for automatic fallback",
                    ],
                )

        # Execute via ROS
        result = ros_func()

        # On failure, check whether to fall back
        if result is None or not result.success:
            error_msg = (
                (result.error or {}).get("message", "Unknown error")
                if result
                else "No response from ROS bridge"
            )
            if _get_control_mode() == "hybrid":
                logger.warning("ROS path failed (%s), falling back to TCP", error_msg)
                return tcp_func()
            return OperationResult.error_result(
                "ROS_PLANNING_FAILED",
                f"MoveIt planning/execution failed: {error_msg}",
                ["Check MoveIt logs", "Verify target is reachable"],
            )

        return result

    except ImportError:
        logger.warning("ros2 module not available, falling back to TCP")
        return tcp_func()
