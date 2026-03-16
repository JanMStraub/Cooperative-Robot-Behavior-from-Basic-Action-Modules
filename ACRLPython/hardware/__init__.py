#!/usr/bin/env python3
"""
hardware/__init__.py - Hardware Interface Factory

Provides get_hardware_interface(env) which returns the correct concrete
RobotHardwareInterface adapter for the given execution environment.

Usage (via core.Imports — prefer that over importing directly):
    from core.Imports import get_hardware_interface
    hw = get_hardware_interface()
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hardware.Interface import RobotHardwareInterface

# Module-level singleton — set once by get_hardware_interface().
_instance: "RobotHardwareInterface | None" = None


def get_hardware_interface(env: str = "sim") -> "RobotHardwareInterface":
    """
    Return the RobotHardwareInterface singleton for the given environment.

    On the first call the correct adapter is instantiated and cached.
    Subsequent calls always return the same instance regardless of the env
    argument, so env only matters on the very first call (set by RunRobotController).

    Args:
        env: "sim" (default) for Unity, "real" for ROS/MoveIt.

    Returns:
        Concrete RobotHardwareInterface adapter.
    """
    global _instance
    if _instance is None:
        if env == "real":
            from hardware.ROSInterface import ROSHardwareInterface

            _instance = ROSHardwareInterface()
        else:
            from hardware.UnityInterface import UnityHardwareInterface

            _instance = UnityHardwareInterface()
    return _instance
