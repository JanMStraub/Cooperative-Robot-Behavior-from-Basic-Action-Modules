#!/usr/bin/env python3
"""Centralised ROS-vs-TCP routing. Each operation passes ros_func and tcp_func to execute_with_ros_fallback."""

import logging
from typing import Callable, Optional

from .Base import OperationResult

logger = logging.getLogger(__name__)


def _is_ros_enabled(use_ros: Optional[bool]) -> bool:
    if use_ros is not None:
        return bool(use_ros)
    try:
        from config.ROS import ROS_ENABLED, DEFAULT_CONTROL_MODE

        return bool(ROS_ENABLED and DEFAULT_CONTROL_MODE in ("ros", "hybrid"))
    except ImportError:
        return False


def _get_control_mode() -> str:
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
    """Execute via ROS; fall back to TCP in hybrid mode. ros_func returns None to signal failure."""
    if not _is_ros_enabled(use_ros):
        return tcp_func()

    try:
        from ros2.ROSBridge import ROSBridge

        bridge = ROSBridge.get_instance()

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
                        "cd ACRLRosUnityIntegration && ./start_ros_endpoint.sh",
                        "Set DEFAULT_CONTROL_MODE='hybrid' in config/ROS.py "
                        "for automatic fallback",
                    ],
                )

        result = ros_func()

        if result is None or not result.success:
            error_msg = (
                (result.error or {}).get("message", "Unknown error")
                if result
                else "No response from ROS bridge"
            )

            # Wrist-singularity / unreachable-goal: MoveIt returns error code 99999.
            # ROS_ALLOW_TCP_FALLBACK_ON_99999 lets operators allow TCP fallback for
            # this specific error even in strict (non-hybrid) mode.
            if "99999" in error_msg or "UNKNOWN_ERROR_99999" in error_msg:
                try:
                    from config.ROS import ROS_ALLOW_TCP_FALLBACK_ON_99999

                    allow_99999_fallback = ROS_ALLOW_TCP_FALLBACK_ON_99999
                except ImportError:
                    allow_99999_fallback = False
                if allow_99999_fallback:
                    logger.warning(
                        "MoveIt UNKNOWN_ERROR_99999 (singularity/unreachable), "
                        "falling back to TCP (ROS_ALLOW_TCP_FALLBACK_ON_99999=True)"
                    )
                    return tcp_func()

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
