#!/usr/bin/env python3
"""
VGN Client — Local Mac Inference
=================================

Implements a fully local YOLO → VLM → VGN grasp-pose pipeline that runs on
Apple Silicon without a GPU server.

Pipeline:
    1. VLM (LM Studio) refines the YOLO bounding box to the best grasp region.
    2. The point cloud is masked to that region.
    3. A TSDF voxel grid is built from the masked cloud.
    4. VGN (vgn_conv.pth) predicts 6-DOF grasp poses on the TSDF.
    5. Poses are returned in right-handed camera frame — ready for
       ``GraspFrameTransform.transform_grasp_poses_to_unity()``.

All heavy imports (torch, scipy, VGN) are deferred to the first
``predict_grasps`` call so startup latency is unaffected.
"""

import logging
import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np

from core.LoggingSetup import setup_logging

setup_logging(__name__)
logger = logging.getLogger(__name__)

# VGN vendor source is vendored under ACRLPython/vendor/vgn/src/
_VGN_SRC = os.path.join(os.path.dirname(__file__), "..", "vendor", "vgn", "src")


def _ensure_vgn_on_path() -> bool:
    """Add VGN source directory to sys.path if not already present.

    Returns:
        True if the VGN source directory exists and was added (or already present).
    """
    vgn_abs = os.path.abspath(_VGN_SRC)
    if not os.path.isdir(vgn_abs):
        logger.debug(f"VGN source not found at {vgn_abs}")
        return False
    if vgn_abs not in sys.path:
        sys.path.insert(0, vgn_abs)
    return True


def _parse_bbox_from_vlm_response(
    text: str,
    fallback: Tuple[int, int, int, int],
    image_width: int = 99999,
    image_height: int = 99999,
) -> Tuple[int, int, int, int]:
    """Extract and validate a bounding-box JSON object from VLM response text.

    The VLM may surround the JSON with prose.  This function searches for the
    first ``{...}`` block containing all four required keys and validates that
    the values lie within the image bounds.

    Args:
        text:         Raw text response from the VLM.
        fallback:     (x, y, w, h) to return when parsing fails.
        image_width:  Image width in pixels used for clamping.
        image_height: Image height in pixels used for clamping.

    Returns:
        Parsed and clamped (x, y, w, h) bounding box, or ``fallback`` on error.
    """
    try:
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if not match:
            logger.debug("VLM bbox: no JSON object found in response, using fallback")
            return fallback

        raw = match.group(0)
        import json

        data = json.loads(raw)
        for key in ("x", "y", "w", "h"):
            if key not in data:
                logger.debug(f"VLM bbox: JSON missing key '{key}', using fallback")
                return fallback

        x = max(0, min(int(data["x"]), image_width - 1))
        y = max(0, min(int(data["y"]), image_height - 1))
        w = max(1, min(int(data["w"]), image_width - x))
        h = max(1, min(int(data["h"]), image_height - y))
        return (x, y, w, h)

    except Exception as exc:
        logger.debug(f"VLM bbox parse failed ({exc}), using fallback")
        return fallback


def _points_to_tsdf_grid(
    points: "np.ndarray",
    size: float = 0.3,
    resolution: int = 40,
) -> "np.ndarray":
    """Convert a centred point cloud to an approximate TSDF voxel grid.

    Avoids Open3D by direct voxelisation.  Each point votes for its nearest
    voxel; the SDF is approximated by euclidean distance transform on the
    occupancy volume.

    Args:
        points:     (N, 3) float32 array, already centred at the origin.
        size:       Workspace cube side length in metres.
        resolution: Number of voxels per side (grid is resolution³).

    Returns:
        Float32 ndarray of shape (1, resolution, resolution, resolution).
        Values are truncated signed distances in the range [-1, 1].
    """
    from scipy.ndimage import distance_transform_edt  # type: ignore

    half = size / 2.0
    vox_size = size / resolution

    # Map continuous coords to voxel indices
    idx = np.floor((points + half) / vox_size).astype(np.int32)
    valid = (
        (idx[:, 0] >= 0)
        & (idx[:, 0] < resolution)
        & (idx[:, 1] >= 0)
        & (idx[:, 1] < resolution)
        & (idx[:, 2] >= 0)
        & (idx[:, 2] < resolution)
    )
    idx = idx[valid]

    occupancy = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    if idx.shape[0] > 0:
        occupancy[idx[:, 0], idx[:, 1], idx[:, 2]] = 1

    # SDF: distance in voxel units, then convert to metres and truncate
    dist_outside = (
        np.asarray(distance_transform_edt(1 - occupancy), dtype=np.float32) * vox_size
    )
    dist_inside = (
        np.asarray(distance_transform_edt(occupancy), dtype=np.float32) * vox_size
    )
    sdf = dist_outside - dist_inside

    # Truncate to [-size/2, size/2] and normalise to [-1, 1]
    trunc = size / 2.0
    sdf = np.clip(sdf, -trunc, trunc) / trunc

    return sdf[np.newaxis, ...]  # (1, R, R, R)


class VGNClient:
    """Local grasp-pose predictor: YOLO → VLM → VGN.

    The model is loaded lazily on the first ``predict_grasps`` call and cached
    at class level so subsequent calls reuse the loaded weights.

    Attributes:
        _net:    Class-level cached VGN network (loaded once).
        _device: Class-level cached torch device.
    """

    _net = None
    _device = None

    def __init__(self) -> None:
        """Read configuration from ``config.Servers``."""
        try:
            from config.Servers import VGN_MODEL_PATH, VGN_TOP_K
        except ImportError:
            VGN_MODEL_PATH = "checkpoints/vgn_conv.pth"
            VGN_TOP_K = 20

        # Resolve relative path from ACRLPython/ root
        if not os.path.isabs(VGN_MODEL_PATH):
            _root = os.path.join(os.path.dirname(__file__), "..")
            VGN_MODEL_PATH = os.path.abspath(os.path.join(_root, VGN_MODEL_PATH))

        self._model_path: str = VGN_MODEL_PATH
        self._top_k_default: int = int(VGN_TOP_K)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the VGN model file exists and torch is importable.

        Returns:
            True only when both the model checkpoint and torch library are
            present; False otherwise.
        """
        if not os.path.isfile(self._model_path):
            logger.debug(f"VGN model not found at {self._model_path}")
            return False
        try:
            import torch  # noqa: F401
        except ImportError:
            logger.debug("torch not available — VGN disabled")
            return False
        return True

    def predict_grasps(
        self,
        points: "np.ndarray",
        colors: "Optional[np.ndarray]",
        image: "np.ndarray",
        yolo_bbox: Tuple[int, int, int, int],
        object_label: str,
        image_width: int,
        image_height: int,
        fov: float,
        top_k: int = 0,
    ) -> "Optional[List[dict]]":
        """Run the full VLM-guided VGN pipeline on a stereo point cloud.

        Steps:
            1. Query LM Studio VLM to refine ``yolo_bbox`` to the best grasp
               sub-region.
            2. Mask the point cloud to the refined bbox.
            3. Build a TSDF voxel grid centred on the masked cloud.
            4. Run VGN inference to get quality/rotation/width volumes.
            5. Convert VGN output to the standard grasp dict format.

        Args:
            points:       (N, 3) float32 in Unity left-handed camera frame
                          (X-negated, Unity left-handed convention).
            colors:       Optional (N, 3) uint8 RGB array.
            image:        Full left stereo image as BGR numpy array (H×W×3).
            yolo_bbox:    (x, y, w, h) pixel bbox from YOLO ``detect_objects``.
            object_label: Object class name (e.g. ``"red_cube"``) for the VLM prompt.
            image_width:  Image width in pixels.
            image_height: Image height in pixels.
            fov:          Horizontal camera field-of-view in degrees.
            top_k:        Max poses to return (0 → use ``_top_k_default``).

        Returns:
            List of grasp dicts in right-handed camera frame on success, or
            ``None`` if too few masked points / inference failed.
            Each dict: ``{"position":[x,y,z], "rotation":[qx,qy,qz,qw],
            "score":float, "width":float, "approach_direction":[dx,dy,dz]}``.
        """
        import numpy as np

        if top_k <= 0:
            top_k = self._top_k_default

        bx, by, bw, bh = yolo_bbox

        # ----------------------------------------------------------------
        # Step 1 — VLM bbox refinement (opt-in, off by default)
        # ----------------------------------------------------------------
        refined_bbox = yolo_bbox  # default: use YOLO bbox unchanged
        try:
            from config.Servers import VGN_USE_VLM_REFINEMENT
        except ImportError:
            VGN_USE_VLM_REFINEMENT = False

        if VGN_USE_VLM_REFINEMENT:
            try:
                from vision.AnalyzeImage import LMStudioVisionProcessor

                vlm = LMStudioVisionProcessor()
                prompt = (
                    f"You see a {object_label}. Return ONLY a JSON object with the "
                    f"pixel bounding box of the best region to grasp it, within the "
                    f"region x={bx} y={by} w={bw} h={bh}. "
                    f'Format: {{"x": int, "y": int, "w": int, "h": int}}'
                )
                vlm_result = vlm.send_images([image], ["left"], prompt)
                refined_bbox = _parse_bbox_from_vlm_response(
                    vlm_result.get("response", ""),
                    fallback=yolo_bbox,
                    image_width=image_width,
                    image_height=image_height,
                )
                logger.info(f"[VGN] VLM refined bbox: {yolo_bbox} → {refined_bbox}")
            except Exception as exc:
                logger.warning(
                    f"[VGN] VLM bbox refinement failed ({exc}), "
                    "using YOLO bbox as fallback"
                )
        else:
            logger.debug("[VGN] VLM refinement skipped (VGN_USE_VLM_REFINEMENT=false)")

        # ----------------------------------------------------------------
        # Step 2 — Point cloud masking
        # ----------------------------------------------------------------
        # Flip X back to right-handed frame (undo Unity LH bake)
        pts_rh = points.copy()
        pts_rh[:, 0] *= -1.0

        # Build segmentation mask using refined bbox.
        # Import from GraspUtils (shared module) to avoid circular import with
        # GraspOperations which imports VGNClient.
        from operations.GraspUtils import _build_segmentation_mask

        mask = _build_segmentation_mask(
            pts_rh,
            refined_bbox,
            image_width,
            image_height,
            fov,
            preferred_approach="auto",
        )

        masked_points = pts_rh[mask]
        _MIN_POINTS = 50
        if masked_points.shape[0] < _MIN_POINTS:
            logger.warning(
                f"[VGN] Only {masked_points.shape[0]} points after masking "
                f"(need ≥ {_MIN_POINTS}) — aborting"
            )
            return None

        logger.info(
            f"[VGN] Masked point cloud: {masked_points.shape[0]} / {pts_rh.shape[0]} points"
        )

        # ----------------------------------------------------------------
        # Step 3 — TSDF construction
        # ----------------------------------------------------------------
        centroid = masked_points.mean(axis=0)
        centred = masked_points - centroid

        _TSDF_SIZE = 0.3  # 30 cm workspace
        _TSDF_RES = 40

        grid = _points_to_tsdf_grid(centred, size=_TSDF_SIZE, resolution=_TSDF_RES)
        # grid shape: (1, 40, 40, 40)

        # ----------------------------------------------------------------
        # Step 4 — VGN inference
        # ----------------------------------------------------------------
        net = self._load_model()
        if net is None:
            logger.warning("[VGN] Model failed to load — aborting")
            return None

        import torch

        device = VGNClient._device
        with torch.no_grad():
            tensor = torch.from_numpy(grid).unsqueeze(0).to(device)
            # tensor shape: (1, 1, 40, 40, 40)
            qual_vol, rot_vol, width_vol = net(tensor)

        try:
            if _ensure_vgn_on_path():
                from vgn.detection import process, select  # type: ignore

                # process() returns cleaned volumes on CPU numpy
                qual_np, rot_np, width_np = process(None, qual_vol, rot_vol, width_vol)
                grasps, scores = select(qual_np, rot_np, width_np)
            else:
                logger.warning(
                    "[VGN] VGN source not on path; cannot call process/select"
                )
                return None
        except Exception as exc:
            logger.warning(f"[VGN] process/select failed: {exc}")
            return None

        if not grasps:
            logger.info("[VGN] VGN returned no grasps")
            return None

        # ----------------------------------------------------------------
        # Step 5 — Output conversion
        # ----------------------------------------------------------------
        from scipy.spatial.transform import Rotation  # type: ignore

        results = []
        for grasp, score in zip(grasps, scores):
            try:
                # Undo centring to get back to camera frame
                pos = grasp.pose.translation + centroid
                rot_matrix = grasp.pose.rotation.as_matrix()
                quat = grasp.pose.rotation.as_quat()  # [qx, qy, qz, qw]
                approach = rot_matrix[:, 2].tolist()  # Z-axis = approach

                results.append(
                    {
                        "position": pos.tolist(),
                        "rotation": quat.tolist(),
                        "score": float(score),
                        "width": float(grasp.width),
                        "approach_direction": approach,
                    }
                )
            except Exception as exc:
                logger.debug(f"[VGN] Skipping malformed grasp: {exc}")
                continue

        if not results:
            return None

        results.sort(key=lambda g: g["score"], reverse=True)
        logger.info(
            f"[VGN] Returning {min(top_k, len(results))} / {len(results)} grasps"
        )
        return results[:top_k]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @classmethod
    def _load_model(cls):
        """Load and cache the VGN network (lazy singleton).

        Tries MPS (Apple Silicon) first, falls back to CPU.

        Returns:
            Loaded nn.Module in eval mode, or None on failure.
        """
        if cls._net is not None:
            return cls._net

        if not _ensure_vgn_on_path():
            logger.warning("[VGN] Cannot load model: VGN source directory missing")
            return None

        try:
            import torch
            from vgn.networks import get_network  # type: ignore

            # Device selection: MPS → CPU
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("[VGN] Using MPS (Apple Silicon)")
            else:
                device = torch.device("cpu")
                logger.info("[VGN] Using CPU")

            # Instantiate 'conv' architecture
            net = get_network("conv")
            # Load checkpoint (weights_only=True for security)
            try:
                from config.Servers import VGN_MODEL_PATH
            except ImportError:
                VGN_MODEL_PATH = "checkpoints/vgn_conv.pth"
            if not os.path.isabs(VGN_MODEL_PATH):
                _root = os.path.join(os.path.dirname(__file__), "..")
                VGN_MODEL_PATH = os.path.abspath(os.path.join(_root, VGN_MODEL_PATH))

            state = torch.load(VGN_MODEL_PATH, map_location=device, weights_only=True)
            net.load_state_dict(state)
            net.to(device)
            net.eval()

            cls._net = net
            cls._device = device
            logger.info(f"[VGN] Model loaded from {VGN_MODEL_PATH} on {device}")
            return cls._net

        except Exception as exc:
            logger.warning(f"[VGN] Model load failed: {exc}")
            return None

    @classmethod
    def reset_cache(cls) -> None:
        """Clear the class-level model cache.

        Intended for test teardown: forces the next ``predict_grasps`` call to
        reload the model rather than reusing a cached instance that may have
        been set up under different configuration (e.g. a different model path
        set via environment variable or mock).
        """
        cls._net = None
        cls._device = None
        logger.debug("[VGN] Model cache cleared")
