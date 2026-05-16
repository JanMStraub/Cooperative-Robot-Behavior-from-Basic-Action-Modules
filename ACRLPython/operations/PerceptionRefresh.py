#!/usr/bin/env python3
"""
Background daemon that re-detects stale WorldState objects via stereo+YOLO.

Falls back to LLM analyze_scene when stereo unavailable (refreshes confidence/last_seen
but cannot update position). Start with PerceptionRefreshLoop(world_state).start().
"""

import threading
import time
from typing import Optional

from config.Servers import PERCEPTION_ONLY_MODE
from core.LoggingSetup import get_logger

logger = get_logger(__name__)

# Default intervals (seconds)
_DEFAULT_REFRESH_INTERVAL = 2.0
_DEFAULT_STALE_THRESHOLD = 0.4

# Module-level singleton reference for get_perception_refresh_daemon()
_active_refresh_loop = None


class PerceptionRefreshLoop:
    """Background daemon that re-detects stale WorldState objects."""

    def __init__(
        self,
        world_state,
        refresh_interval: float = _DEFAULT_REFRESH_INTERVAL,
        stale_threshold: float = _DEFAULT_STALE_THRESHOLD,
    ):
        """Does not start the thread — call start() separately."""
        self._world_state = world_state
        self._refresh_interval = refresh_interval
        self._stale_threshold = stale_threshold
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._anticipatory_queue: list = []
        self._anticipatory_lock = threading.Lock()

    def start(self) -> None:
        """Start the background refresh thread (idempotent)."""
        global _active_refresh_loop
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
        _active_refresh_loop = self
        logger.debug(
            f"PerceptionRefreshLoop started "
            f"(interval={self._refresh_interval}s, stale_threshold={self._stale_threshold})"
        )

    def trigger_anticipatory_refresh(self, object_ids: list) -> None:
        """Queue objects for re-detection on next sweep (call when robot commits to moving toward object)."""
        with self._anticipatory_lock:
            for oid in object_ids:
                if oid not in self._anticipatory_queue:
                    self._anticipatory_queue.append(oid)

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

    def _refresh_object_by_id(self, object_id: str) -> bool:
        """Refresh object by ID, looking up its color from WorldState. Returns True on success."""
        try:
            obj = self._world_state.get_object(object_id)
            if obj is None:
                return False
            return self._refresh_stereo(obj.color)
        except Exception as exc:
            logger.debug(f"_refresh_object_by_id failed for '{object_id}': {exc}")
            return False

    def _sweep(self) -> None:
        """Drain anticipatory queue (high-priority), then find and re-detect stale objects."""
        with self._anticipatory_lock:
            pending = list(self._anticipatory_queue)
            self._anticipatory_queue.clear()
        for object_id in pending:
            if self._stop_event.is_set():
                return
            try:
                self._refresh_object_by_id(object_id)
            except Exception as exc:
                logger.debug(f"Anticipatory refresh failed for '{object_id}': {exc}")

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
        """Return deduplicated color labels for objects with stale=True or confidence < threshold."""
        stale_colors = []
        seen = set()
        try:
            all_objects = self._world_state.get_all_objects()
            for obj in all_objects:
                # Fields are static landmarks — never refresh via cube detector.
                if getattr(obj, "object_type", None) == "field":
                    continue
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
        """Re-detect color via stereo+YOLO. In PERCEPTION_ONLY_MODE uses cached images (no Unity round-trip)."""
        try:
            from operations.VisionOperations import detect_object_stereo

            result = detect_object_stereo(
                color=color,
                selection="closest",
                request_id=0,
                request_fresh_capture=not PERCEPTION_ONLY_MODE,
            )
            if result and result.success:
                logger.debug(f"PerceptionRefreshLoop: stereo refresh OK for '{color}'")
                return True
        except Exception as exc:
            logger.debug(
                f"PerceptionRefreshLoop stereo refresh failed for '{color}': {exc}"
            )
        return False

    def _refresh_llm_fallback(self, color: str) -> None:
        """LLM analyze_scene fallback — resets confidence/last_seen but cannot update position (no depth)."""
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
                        obj = self._world_state._objects.get(
                            color
                        ) or self._world_state._objects.get(color.lower())
                        if obj is not None:
                            obj.confidence = max(
                                obj.confidence, self._stale_threshold + 0.1
                            )
                            obj.last_seen = time.time()
                            obj.stale = False
                            logger.debug(
                                f"PerceptionRefreshLoop: LLM fallback refreshed confidence for '{color}'"
                            )
                except Exception as exc:
                    logger.debug(
                        f"LLM fallback confidence update failed for '{color}': {exc}"
                    )
        except Exception as exc:
            logger.debug(
                f"PerceptionRefreshLoop LLM fallback error for '{color}': {exc}"
            )
