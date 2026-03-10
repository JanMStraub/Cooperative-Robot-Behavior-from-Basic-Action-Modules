"""
GraspNet HTTP Client
====================

Thin HTTP client for the Contact-GraspNet FastAPI inference service running
on the GPU server.  Lives in ``operations/`` (not ``servers/``) so that
GraspOperations can import it without violating the layered architecture rule
that operations must not import from servers.

The client exposes two public methods:

* :meth:`is_available` — lightweight health-check with a TTL-based cache so we
  don't hammer the endpoint before every grasp.
* :meth:`predict_grasps` — send a point cloud, receive ranked grasp poses.

All poses are returned in **camera frame** (right-handed, Z-forward).
The caller (GraspOperations) is responsible for the Unity world-space transform.
"""

import base64
import logging
import time
from typing import List, Optional

import numpy as np

from core.LoggingSetup import setup_logging

setup_logging(__name__)
logger = logging.getLogger(__name__)


class GraspNetClient:
    """HTTP client for the Contact-GraspNet FastAPI inference service.

    The service URL and timeout are read from ``config.Servers`` at
    construction time so environment-variable overrides take effect.

    Attributes:
        _base_url:           Base URL of the FastAPI service.
        _timeout:            HTTP request timeout in seconds.
        _top_k_default:      Default number of grasp poses to request.
        _health_cache_ttl:   Seconds between live health-checks.
        _last_health_check:  Unix timestamp of last check.
        _last_health_result: Cached result of last health-check.
    """

    def __init__(self) -> None:
        """Initialise client, reading configuration from config.Servers."""
        try:
            from config.Servers import (
                GRASPNET_URL,
                GRASPNET_TIMEOUT,
                GRASPNET_TOP_K,
                GRASPNET_HEALTH_CACHE_TTL,
            )
        except ImportError:
            # Sensible defaults if config is not yet present
            GRASPNET_URL = "http://192.168.178.53:8765"
            GRASPNET_TIMEOUT = 10.0
            GRASPNET_TOP_K = 20
            GRASPNET_HEALTH_CACHE_TTL = 30.0

        self._base_url: str = GRASPNET_URL.rstrip("/")
        self._timeout: float = float(GRASPNET_TIMEOUT)
        self._top_k_default: int = int(GRASPNET_TOP_K)
        self._health_cache_ttl: float = float(GRASPNET_HEALTH_CACHE_TTL)

        self._last_health_check: float = 0.0
        self._last_health_result: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the GraspNet service is reachable and healthy.

        The result is cached for ``_health_cache_ttl`` seconds to avoid
        per-grasp HTTP round-trips when the service is consistently available.

        Returns:
            True if the ``/health`` endpoint responds with HTTP 200.
        """
        now = time.time()
        if now - self._last_health_check < self._health_cache_ttl:
            return self._last_health_result

        try:
            import urllib.request

            url = f"{self._base_url}/health"
            with urllib.request.urlopen(url, timeout=3.0) as resp:
                ok = resp.status == 200
        except Exception as exc:
            logger.debug(f"GraspNet health-check failed: {exc}")
            ok = False

        self._last_health_check = now
        self._last_health_result = ok
        if ok:
            logger.debug("GraspNet service is healthy")
        else:
            logger.warning(
                f"GraspNet service at {self._base_url} is not available"
            )
        return ok

    def predict_grasps(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        segmentation_mask: Optional[np.ndarray] = None,
        top_k: Optional[int] = None,
    ) -> Optional[List[dict]]:
        """Request ranked grasp poses from the GraspNet service.

        Sends a point cloud (and optional segmentation mask / colors) to the
        ``POST /predict_grasps`` endpoint and returns the list of grasp dicts
        exactly as returned by the service (camera frame, right-handed).

        Args:
            points:            Float32 array of shape (N, 3) in camera frame.
            colors:            Optional uint8 array of shape (N, 3) RGB.
            segmentation_mask: Optional bool array of shape (N,).  When provided
                only the masked subset of ``points`` / ``colors`` is sent
                (applied client-side before encoding; not transmitted to server).
            top_k:             Number of poses to request (default: ``_top_k_default``).

        Returns:
            List of grasp dicts on success, or None on network / decode error.
            Each dict contains at minimum:
            ``{"position": [x,y,z], "rotation": [qx,qy,qz,qw],
               "score": float, "width": float,
               "approach_direction": [dx,dy,dz]}``.
        """
        import json
        import urllib.request

        if top_k is None:
            top_k = self._top_k_default

        # Apply segmentation mask before sending
        pts = points
        clr = colors
        if segmentation_mask is not None and segmentation_mask.any():
            pts = points[segmentation_mask]
            clr = colors[segmentation_mask] if colors is not None else None

        if pts.shape[0] == 0:
            logger.warning("predict_grasps: empty point cloud after masking, skipping")
            return None

        # Convert from Unity left-handed frame (X-negated) to the right-handed
        # OpenCV camera frame that Contact-GraspNet expects (X-right, Y-down, Z-fwd).
        # StereoReconstruction bakes a sign-flip into its Q-matrix so every point
        # arriving here has X already negated.  Un-negate before sending so the
        # model sees the correct geometry.  GraspFrameTransform applies the
        # inverse flip again after inference when transforming poses to Unity world.
        pts_rh = pts.copy()
        pts_rh[:, 0] *= -1.0

        # The NVlabs server resamples to 20 000 points internally regardless of
        # how many we send.  Pre-downsample client-side to avoid transmitting 3 MB
        # of JSON over the LAN when 1.2 MB carries identical information.
        # 4 decimal places = 0.1 mm precision — far more than grasp planning needs.
        _SERVER_MAX_POINTS = 20_000
        if pts_rh.shape[0] > _SERVER_MAX_POINTS:
            idx = np.random.choice(pts_rh.shape[0], _SERVER_MAX_POINTS, replace=False)
            pts_rh = pts_rh[idx]
            if clr is not None:
                clr = clr[idx]

        pts_bytes = pts_rh.astype(np.float32).tobytes()
        payload: dict = {
            "points": base64.b64encode(pts_bytes).decode("ascii"),
            "points_shape": list(pts_rh.shape),  # [N, 3]
            "top_k": top_k,
        }
        if clr is not None:
            clr_bytes = clr.astype(np.uint8).tobytes()
            payload["colors"] = base64.b64encode(clr_bytes).decode("ascii")
            payload["colors_shape"] = list(clr.shape)  # [N, 3]

        body = json.dumps(payload).encode("utf-8")
        url = f"{self._base_url}/predict_grasps"

        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                elapsed_ms = (time.time() - t0) * 1000
                raw = resp.read()
                data = json.loads(raw)

            if not data.get("success", False):
                logger.warning(f"GraspNet returned success=false: {data}")
                return None

            grasps: List[dict] = data.get("grasps", [])
            logger.info(
                f"GraspNet returned {len(grasps)} grasps "
                f"(inference {data.get('inference_time_ms', elapsed_ms):.0f}ms, "
                f"request {elapsed_ms:.0f}ms)"
            )
            # Invalidate health cache — service is clearly alive
            self._last_health_check = time.time()
            self._last_health_result = True
            return grasps

        except Exception as exc:
            logger.warning(f"GraspNet predict_grasps failed: {exc}")
            # Mark service as unavailable so next is_available() re-checks
            self._last_health_result = False
            self._last_health_check = 0.0
            return None
