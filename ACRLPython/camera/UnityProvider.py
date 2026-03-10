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
    """
    Concrete camera provider backed by UnifiedImageStorage.

    Delegates every call to the singleton image storage that the ImageServer
    continuously fills with frames from Unity.
    """

    def _storage(self):
        """Return the UnifiedImageStorage singleton via the lazy import path."""
        from core.Imports import get_unified_image_storage
        return get_unified_image_storage()

    def get_rgb_frame(self) -> Optional[np.ndarray]:
        """
        Return the most recent single-camera or stereo-left frame from Unity.

        Prefers the stereo left channel when stereo frames are available,
        falls back to the single-camera feed.
        """
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
        """
        Return the most recent stereo image pair (left, right) from Unity.

        Returns None if no stereo frame has been received yet.
        """
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
        """
        Return a depth map derived from the stereo pair (placeholder).

        Full stereo depth reconstruction is handled by DepthEstimator, which
        already reads from ImageStorage directly.  This method returns None
        until a dedicated depth channel is wired in.
        """
        return None
