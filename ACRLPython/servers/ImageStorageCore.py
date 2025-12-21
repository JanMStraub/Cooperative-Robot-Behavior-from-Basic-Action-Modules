#!/usr/bin/env python3
"""
Unified Image Storage Core
==========================

Thread-safe singleton for storing and retrieving camera images.

Architecture Decision:
    This module is intentionally separated from ImageServer to avoid circular
    import dependencies in the module graph:

    OLD (Circular):
        operations/DetectionOperations → servers/ImageServer →
        servers/__init__ → servers/SequenceServer →
        orchestrators/SequenceExecutor → operations/Registry →
        operations/DetectionOperations (CIRCULAR!)

    NEW (Clean):
        operations/DetectionOperations → servers/ImageStorageCore (✓)
        servers/ImageServer → servers/ImageStorageCore (✓)

    Core Principle:
        - Core modules (servers/ImageStorageCore) have NO high-level dependencies
        - High-level modules (servers/ImageServer) CAN depend on core modules
        - Operations modules CAN depend on core modules but NOT server modules

Thread Safety:
    - Uses double-checked locking for singleton initialization
    - Separate locks for single and stereo image storage
    - All methods are thread-safe

Consolidation:
    Replaces legacy StreamingServer ImageStorage and StereoDetectionServer StereoImageStorage.
"""

import threading
import time
import logging
from typing import Optional, Tuple, List, Dict
import numpy as np

# Import config
try:
    import LLMConfig as cfg
except ImportError:
    from .. import LLMConfig as cfg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedImageStorage:
    """
    Thread-safe storage for all image types (single and stereo).

    Provides unified access for detection and analysis operations.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_storage()
        return cls._instance

    def _init_storage(self):
        """Initialize storage structures."""
        self._single_images: Dict[str, Tuple[np.ndarray, float, str]] = {}
        # Stereo: (imgL, imgR, prompt, timestamp, metadata)
        self._stereo_images: Dict[
            str, Tuple[np.ndarray, np.ndarray, str, float, dict]
        ] = {}
        self._data_lock = threading.Lock()

    # Single camera methods
    def store_single_image(self, camera_id: str, image: np.ndarray, prompt: str = ""):
        """Store a single camera image."""
        with self._data_lock:
            self._single_images[camera_id] = (image, time.time(), prompt)

    def get_single_image(self, camera_id: str) -> Optional[np.ndarray]:
        """Get the latest single camera image."""
        with self._data_lock:
            if camera_id in self._single_images:
                return self._single_images[camera_id][0].copy()
            return None

    def get_single_prompt(self, camera_id: str) -> Optional[str]:
        """Get the prompt for a single camera image."""
        with self._data_lock:
            if camera_id in self._single_images:
                return self._single_images[camera_id][2]
            return None

    def get_single_age(self, camera_id: str) -> Optional[float]:
        """Get age of single camera image in seconds."""
        with self._data_lock:
            if camera_id in self._single_images:
                return time.time() - self._single_images[camera_id][1]
            return None

    def get_latest_single(self) -> Optional[Tuple[str, np.ndarray, str]]:
        """Get the most recently stored single image."""
        with self._data_lock:
            if not self._single_images:
                return None
            latest_id = max(
                self._single_images.keys(), key=lambda k: self._single_images[k][1]
            )
            img, _, prompt = self._single_images[latest_id]
            return latest_id, img.copy(), prompt

    # Stereo camera methods
    def store_stereo_pair(
        self,
        camera_pair_id: str,
        imgL: np.ndarray,
        imgR: np.ndarray,
        prompt: str = "",
        metadata: Optional[dict] = None,
    ):
        """Store a stereo image pair with optional metadata."""
        with self._data_lock:
            self._stereo_images[camera_pair_id] = (
                imgL,
                imgR,
                prompt,
                time.time(),
                metadata or {},
            )

            if not cfg.ENABLE_VISION_STREAMING:
                logger.info(
                    f"Stored stereo pair '{camera_pair_id}' "
                    f"(L: {imgL.shape}, R: {imgR.shape})"
                )

    def get_stereo_pair(
        self, camera_pair_id: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
        """Get a stereo image pair."""
        with self._data_lock:
            if camera_pair_id in self._stereo_images:
                imgL, imgR, prompt, _, _ = self._stereo_images[camera_pair_id]
                return imgL.copy(), imgR.copy(), prompt
            return None

    def get_stereo_metadata(self, camera_pair_id: str) -> Optional[dict]:
        """Get metadata for a stereo pair (baseline, fov, camera_position, camera_rotation)."""
        with self._data_lock:
            if camera_pair_id in self._stereo_images:
                return self._stereo_images[camera_pair_id][4]
            return None

    def get_stereo_age(self, camera_pair_id: str) -> Optional[float]:
        """Get age of stereo pair in seconds."""
        with self._data_lock:
            if camera_pair_id in self._stereo_images:
                return time.time() - self._stereo_images[camera_pair_id][3]
            return None

    def get_stereo_timestamp(self, camera_pair_id: str) -> Optional[float]:
        """Get the timestamp when stereo pair was received."""
        with self._data_lock:
            if camera_pair_id in self._stereo_images:
                return self._stereo_images[camera_pair_id][3]
            return None

    def get_latest_stereo(self) -> Optional[Tuple[str, np.ndarray, np.ndarray, str]]:
        """Get the most recently stored stereo pair."""
        with self._data_lock:
            if not self._stereo_images:
                return None
            latest_id = max(
                self._stereo_images.keys(), key=lambda k: self._stereo_images[k][3]
            )
            imgL, imgR, prompt, _, _ = self._stereo_images[latest_id]
            return latest_id, imgL.copy(), imgR.copy(), prompt

    def get_latest_stereo_image(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, str, float, dict]]:
        """
        Get the most recently stored stereo pair with full metadata.

        Returns:
            Tuple of (imgL, imgR, prompt, timestamp, metadata) or None if no stereo images
        """
        with self._data_lock:
            if not self._stereo_images:
                return None
            latest_id = max(
                self._stereo_images.keys(), key=lambda k: self._stereo_images[k][3]
            )
            imgL, imgR, prompt, timestamp, metadata = self._stereo_images[latest_id]
            return imgL.copy(), imgR.copy(), prompt, timestamp, metadata

    def get_all_stereo_ids(self) -> List[str]:
        """Get all stereo camera pair IDs."""
        with self._data_lock:
            return list(self._stereo_images.keys())

    # General methods
    def get_all_camera_ids(self) -> List[str]:
        """Get all camera IDs (single and stereo)."""
        with self._data_lock:
            single = list(self._single_images.keys())
            stereo = [f"{k} (stereo)" for k in self._stereo_images.keys()]
            return single + stereo

    def cleanup_old_images(self, max_age_seconds: float = 300.0):
        """Remove images older than max_age_seconds."""
        with self._data_lock:
            current = time.time()

            # Clean single images
            to_remove = [
                k
                for k, v in self._single_images.items()
                if current - v[1] > max_age_seconds
            ]
            for k in to_remove:
                del self._single_images[k]

            # Clean stereo images
            to_remove = [
                k
                for k, v in self._stereo_images.items()
                if current - v[3] > max_age_seconds
            ]
            for k in to_remove:
                del self._stereo_images[k]
