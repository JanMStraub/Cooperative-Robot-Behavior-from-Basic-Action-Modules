#!/usr/bin/env python3
"""
UnityProvider.py - Unity Camera Adapter

Reads images from UnifiedImageStorage, which is populated by the TCP-based
ImageServer that receives frames from the Unity simulation (ports 5005/5006).

This provider is the default for --env sim.
"""

import logging
from typing import Optional, Tuple

import numpy as np

from camera.Provider import CameraProvider

logger = logging.getLogger(__name__)


class UnityProvider(CameraProvider):
    """Camera provider backed by UnifiedImageStorage (populated by ImageServer)."""

    def _storage(self):
        from core.Imports import get_unified_image_storage

        return get_unified_image_storage()

    def get_rgb_frame(self) -> Optional[np.ndarray]:
        """Return most recent RGB frame, preferring stereo-left over single-camera feed."""
        try:
            storage = self._storage()
            stereo = storage.get_latest_stereo()
            if stereo:
                _, img_left, _, _ = stereo
                return img_left
            single = storage.get_latest_single()
            if single:
                _, img, _ = single
                return img
            return None
        except Exception as e:
            logger.error(f"UnityProvider.get_rgb_frame failed: {e}")
            return None

    def get_stereo_pair(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return most recent stereo pair from Unity, or None if not yet received."""
        try:
            stereo = self._storage().get_latest_stereo()
            if stereo:
                _, img_left, img_right, _ = stereo
                return img_left, img_right
            return None
        except Exception as e:
            logger.error(f"UnityProvider.get_stereo_pair failed: {e}")
            return None

    def get_depth_frame(self) -> Optional[np.ndarray]:
        """Not implemented; DepthEstimator reads ImageStorage directly."""
        return None
