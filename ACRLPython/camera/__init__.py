#!/usr/bin/env python3
"""
camera/__init__.py - Camera Provider Factory

Provides get_camera_provider(env) which returns the correct concrete
CameraProvider adapter for the given execution environment.

Usage (via core.Imports — prefer that over importing directly):
    from core.Imports import get_camera_provider
    cam = get_camera_provider()
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from camera.Provider import CameraProvider

# Module-level singleton — set once by get_camera_provider().
_instance: "CameraProvider | None" = None


def get_camera_provider(env: str = "sim") -> "CameraProvider":
    """
    Return the CameraProvider singleton for the given environment.

    On the first call the correct adapter is instantiated and cached.
    Subsequent calls always return the same instance.

    Args:
        env: "sim" (default) for Unity ImageStorage, "real" for local USB/RealSense.

    Returns:
        Concrete CameraProvider adapter.
    """
    global _instance
    if _instance is None:
        if env == "real":
            from camera.LocalProvider import LocalProvider

            _instance = LocalProvider()
        else:
            from camera.UnityProvider import UnityProvider

            _instance = UnityProvider()
    return _instance
