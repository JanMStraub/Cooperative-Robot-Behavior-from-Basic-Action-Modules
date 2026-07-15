#!/usr/bin/env python3
"""Background daemon that polls a CameraProvider and forwards stereo pairs into UnifiedImageStorage."""

import threading
from typing import Optional

from camera.Provider import CameraProvider
from config.Vision import DEFAULT_CAMERA_ID, LOCAL_CAPTURE_FPS
from core.LoggingSetup import get_logger

logger = get_logger(__name__)

_active_capture_bridge = None


class CameraCaptureBridge:
    def __init__(
        self,
        provider: CameraProvider,
        camera_pair_id: str = DEFAULT_CAMERA_ID,
        capture_interval: Optional[float] = None,
    ):
        self._provider = provider
        self._camera_pair_id = camera_pair_id
        self._capture_interval = capture_interval or (1.0 / LOCAL_CAPTURE_FPS)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        global _active_capture_bridge
        if self._thread and self._thread.is_alive():
            logger.debug("CameraCaptureBridge already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="camera-capture-bridge",
            daemon=True,
        )
        self._thread.start()
        _active_capture_bridge = self
        logger.debug(
            f"CameraCaptureBridge started (camera_pair_id={self._camera_pair_id}, "
            f"interval={self._capture_interval}s)"
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._capture_interval + 2.0)
        logger.info("CameraCaptureBridge stopped")

    def _capture_loop(self) -> None:
        while not self._stop_event.wait(timeout=self._capture_interval):
            try:
                self._capture_once()
            except Exception as exc:
                logger.error(f"CameraCaptureBridge capture error: {exc}", exc_info=True)

    def _capture_once(self) -> None:
        pair = self._provider.get_stereo_pair()
        if pair is None:
            return
        imgL, imgR = pair

        from core.Imports import get_unified_image_storage

        storage = get_unified_image_storage()
        storage.store_stereo_pair(self._camera_pair_id, imgL, imgR)
