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
    """Abstract base class for camera backends (Unity, USB, RealSense)."""

    @abstractmethod
    def get_rgb_frame(self) -> Optional[np.ndarray]:
        """Return most recent RGB frame (H, W, 3, uint8 BGR), or None if unavailable."""

    @abstractmethod
    def get_stereo_pair(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return most recent stereo pair (left_bgr, right_bgr), or None if unavailable."""

    @abstractmethod
    def get_depth_frame(self) -> Optional[np.ndarray]:
        """Return most recent depth map (H, W float32, metres), or None if unsupported."""
