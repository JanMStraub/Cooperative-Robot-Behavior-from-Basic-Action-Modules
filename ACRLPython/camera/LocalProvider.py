#!/usr/bin/env python3
"""
LocalProvider.py - Local USB/RealSense Camera Adapter

Reads images from physical cameras attached to the host machine via
OpenCV's VideoCapture.  Used for --env real (physical robot operation).

Supports a primary RGB camera and an optional secondary camera for stereo.
A RealSense depth camera can be wired in as a future extension.
"""

import logging
from typing import Optional, Tuple

import numpy as np

from camera.Provider import CameraProvider

logger = logging.getLogger(__name__)


class LocalProvider(CameraProvider):
    """
    Concrete camera provider backed by cv2.VideoCapture.

    Captures frames from locally attached USB or RealSense cameras.
    The primary device_id provides the RGB (and left stereo) feed;
    an optional secondary_device_id provides the right stereo channel.
    """

    def __init__(self, device_id: int = 0, secondary_device_id: Optional[int] = None):
        """
        Initialise the local camera provider.

        Args:
            device_id: OpenCV VideoCapture device index for the primary camera.
            secondary_device_id: Optional second camera index for stereo capture.
        """
        self._device_id = device_id
        self._secondary_device_id = secondary_device_id
        self._cap_primary = None
        self._cap_secondary = None
        self._open_cameras()

    def _open_cameras(self):
        """Open VideoCapture handles for all configured cameras."""
        try:
            import cv2

            self._cap_primary = cv2.VideoCapture(self._device_id)
            if not self._cap_primary.isOpened():
                logger.warning(
                    f"LocalProvider: Could not open primary camera {self._device_id}"
                )
                self._cap_primary = None

            if self._secondary_device_id is not None:
                self._cap_secondary = cv2.VideoCapture(self._secondary_device_id)
                if not self._cap_secondary.isOpened():
                    logger.warning(
                        f"LocalProvider: Could not open secondary camera {self._secondary_device_id}"
                    )
                    self._cap_secondary = None

        except ImportError:
            logger.error(
                "LocalProvider: OpenCV (cv2) is not installed. Cannot open cameras."
            )

    def _read_frame(self, cap) -> Optional[np.ndarray]:
        """Read a single frame from a VideoCapture handle."""
        if cap is None:
            return None
        ret, frame = cap.read()
        if not ret:
            return None
        return frame

    def get_rgb_frame(self) -> Optional[np.ndarray]:
        """Return the most recent frame from the primary camera."""
        return self._read_frame(self._cap_primary)

    def get_stereo_pair(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Return a stereo pair (left, right) by reading both cameras simultaneously.

        Returns None if either camera is unavailable.
        """
        left = self._read_frame(self._cap_primary)
        right = self._read_frame(self._cap_secondary)
        if left is not None and right is not None:
            return left, right
        return None

    def get_depth_frame(self) -> Optional[np.ndarray]:
        """
        Return a depth map.

        Not yet implemented for generic USB cameras.
        Wire in pyrealsense2 or another depth SDK here for Phase 4.
        """
        return None

    def __del__(self):
        """Release VideoCapture handles on garbage collection."""
        if self._cap_primary:
            self._cap_primary.release()
        if self._cap_secondary:
            self._cap_secondary.release()
