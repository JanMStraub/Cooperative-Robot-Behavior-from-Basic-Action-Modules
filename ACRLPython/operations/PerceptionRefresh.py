#!/usr/bin/env python3
"""
Perception Refresh Loop
=======================

Background daemon that periodically re-detects stale objects in WorldState
using stereo+YOLO (``detect_object_stereo``) so that object positions stay
fresh without the operator explicitly running detection.

When stereo images are unavailable, falls back to an LLM ``analyze_scene``
call which refreshes *confidence* and *last_seen* but cannot supply depth
(so position is not updated).

Usage
-----
Start alongside all other servers in RunRobotController::

    refresh = PerceptionRefreshLoop(world_state=get_world_state())
    refresh.start()

The loop runs as a daemon thread and is automatically stopped when the process
exits.  Call ``stop()`` for a graceful shutdown.
"""

import threading
import time
import logging
from typing import Optional

from core.LoggingSetup import get_logger

logger = get_logger(__name__)

# Default intervals (seconds)
_DEFAULT_REFRESH_INTERVAL = 2.0
_DEFAULT_STALE_THRESHOLD = 0.4


class PerceptionRefreshLoop:
    """Background daemon that re-detects stale WorldState objects.

    Attributes:
        world_state: WorldState singleton to monitor and update.
        refresh_interval: Seconds between each polling sweep.
        stale_threshold: Objects with confidence below this value are refreshed.
    """

    def __init__(
        self,
        world_state,
        refresh_interval: float = _DEFAULT_REFRESH_INTERVAL,
        stale_threshold: float = _DEFAULT_STALE_THRESHOLD,
    ):
        """Initialise the refresh loop (does not start the thread yet).

        Args:
            world_state: WorldState singleton instance.
            refresh_interval: Seconds between sweeps.
            stale_threshold: Confidence cutoff below which an object is refreshed.
        """
        self._world_state = world_state
        self._refresh_interval = refresh_interval
        self._stale_threshold = stale_threshold
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background refresh thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            logger.debug("PerceptionRefreshLoop already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._refresh_loop,
            name="perception-refresh",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"PerceptionRefreshLoop started "
            f"(interval={self._refresh_interval}s, stale_threshold={self._stale_threshold})"
        )

    def stop(self) -> None:
        """Signal the refresh thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._refresh_interval + 2.0)
        logger.info("PerceptionRefreshLoop stopped")

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _refresh_loop(self) -> None:
        """Main polling loop — runs in daemon thread."""
        while not self._stop_event.wait(timeout=self._refresh_interval):
            try:
                self._sweep()
            except Exception as exc:
                logger.error(f"PerceptionRefreshLoop sweep error: {exc}", exc_info=True)

    def _sweep(self) -> None:
        """One sweep: find stale objects and re-detect each one."""
        stale_colors = self._collect_stale_colors()
        if not stale_colors:
            return

        logger.debug(f"PerceptionRefreshLoop: refreshing {stale_colors}")

        for color in stale_colors:
            if self._stop_event.is_set():
                return
            refreshed = self._refresh_stereo(color)
            if not refreshed:
                self._refresh_llm_fallback(color)

    def _collect_stale_colors(self) -> list:
        """Return list of color labels for objects that need refreshing.

        An object needs refreshing when:
        - Its ``stale`` flag is True, OR
        - Its ``confidence`` is below ``stale_threshold``.

        Returns:
            Deduplicated list of color strings.
        """
        stale_colors = []
        seen = set()
        try:
            all_objects = self._world_state.get_all_objects()
            for obj in all_objects:
                color = getattr(obj, "color", None)
                if color and color != "unknown" and color not in seen:
                    is_stale = getattr(obj, "stale", False)
                    confidence = getattr(obj, "confidence", 1.0)
                    if is_stale or confidence < self._stale_threshold:
                        stale_colors.append(color)
                        seen.add(color)
        except Exception as exc:
            logger.debug(f"_collect_stale_colors error: {exc}")
        return stale_colors

    def _refresh_stereo(self, color: str) -> bool:
        """Attempt to re-detect ``color`` via stereo+YOLO.

        Args:
            color: Object color label to search for.

        Returns:
            True if detection succeeded and WorldState was updated.
        """
        try:
            from operations.VisionOperations import detect_object_stereo

            result = detect_object_stereo(
                color=color,
                camera_id=None,
                selection="closest",
                request_id=0,
            )
            if result and result.success:
                logger.debug(f"PerceptionRefreshLoop: stereo refresh OK for '{color}'")
                return True
        except Exception as exc:
            logger.debug(f"PerceptionRefreshLoop stereo refresh failed for '{color}': {exc}")
        return False

    def _refresh_llm_fallback(self, color: str) -> None:
        """Fall back to LLM analyze_scene when stereo images are unavailable.

        This does NOT update the object's world position (no depth information),
        but resets the confidence and last_seen timestamp so the object is not
        immediately evicted by the TTL logic.

        Args:
            color: Object color label to look for in the scene analysis.
        """
        try:
            from operations.VisionOperations import analyze_scene

            result = analyze_scene(request_id=0)
            if not (result and result.success):
                return

            # Check whether the LLM reports the object as present
            description = ""
            if isinstance(result.result, dict):
                description = result.result.get("description", "")
            elif isinstance(result.result, str):
                description = result.result

            if color.lower() in description.lower():
                # Object is still present but we have no depth — bump confidence
                # and last_seen without touching position.
                try:
                    with self._world_state._lock:
                        # _objects keys are color labels (set by detect_object_stereo)
                        obj = self._world_state._objects.get(color) or self._world_state._objects.get(color.lower())
                        if obj is not None:
                            obj.confidence = max(obj.confidence, self._stale_threshold + 0.1)
                            obj.last_seen = time.time()
                            obj.stale = False
                            logger.debug(
                                f"PerceptionRefreshLoop: LLM fallback refreshed confidence for '{color}'"
                            )
                except Exception as exc:
                    logger.debug(f"LLM fallback confidence update failed for '{color}': {exc}")
        except Exception as exc:
            logger.debug(f"PerceptionRefreshLoop LLM fallback error for '{color}': {exc}")
