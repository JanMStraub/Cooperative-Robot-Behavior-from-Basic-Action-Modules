#!/usr/bin/env python3
"""
Provider.py - Abstract Camera Provider

Defines the CameraProvider contract (port) that all camera backends must implement.
Concrete adapters (UnityProvider for simulation, LocalProvider for real hardware)
implement this interface so that vision operations remain environment-agnostic.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class CameraProvider(ABC):
    """
    Abstract base class for camera backends.

    All concrete providers (Unity image storage, USB camera, RealSense) must
    implement every method below.  Vision operations call these methods and
    never import a concrete implementation directly.
    """

    @abstractmethod
    def get_rgb_frame(self) -> Optional[np.ndarray]:
        """
        Return the most recent RGB frame as a NumPy array (H, W, 3, uint8).

        Returns:
            BGR image array, or None if no frame is available yet.
        """

    @abstractmethod
    def get_stereo_pair(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Return the most recent stereo image pair (left, right).

        Returns:
            Tuple of (left_bgr, right_bgr) arrays, or None if unavailable.
        """

    @abstractmethod
    def get_depth_frame(self) -> Optional[np.ndarray]:
        """
        Return the most recent depth map as a float32 NumPy array (H, W).

        Depth values are in metres.

        Returns:
            Depth array, or None if depth is not supported / not available.
        """
