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

# VGN inference source (trimmed from ethz-asl/vgn) under ACRLPython/vgn/
_VGN_SRC = os.path.join(os.path.dirname(__file__), "..")


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

    # Normalise to [-1, 1] matching Open3D ScalableTSDFVolume with sdf_trunc=4*vox_size.
    # VGN's process() thresholds (> 0.5 = outside, < 0 = inside) assume this convention.
    # Truncation at 4 voxels: distances > 4 vox → ±1; surface voxels → 0.
    _TRUNC_VOXELS = 4.0
    dist_outside = np.asarray(distance_transform_edt(1 - occupancy), dtype=np.float32)
    dist_inside  = np.asarray(distance_transform_edt(occupancy),     dtype=np.float32)
    sdf = dist_outside - dist_inside
    sdf = np.clip(sdf, -_TRUNC_VOXELS, _TRUNC_VOXELS) / _TRUNC_VOXELS

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
        cam_pos: "Optional[List[float]]" = None,
        cam_rot: "Optional[List[float]]" = None,
    ) -> "Optional[List[dict]]":
        """Wrapper to evaluate VGN grasps and broadcast telemetry even if it ends early."""
        debug_info = {}
        try:
            return self._predict_grasps_internal(
                points, colors, image, yolo_bbox, object_label,
                image_width, image_height, fov, top_k, cam_pos, cam_rot, debug_info
            )
        finally:
            try:
                from servers.WebUIServer import broadcast_vgn_debug
                import base64

                _pts_scene = debug_info.get("pts_full_scene")
                pts = _pts_scene if _pts_scene is not None else debug_info.get("pts")
                centroid = debug_info.get("centroid")
                grid = debug_info.get("grid")
                grasps = debug_info.get("grasps", [])
                in_wf = debug_info.get("_in_world_frame", False)

                if pts is not None:
                    # Full-scene cloud (pts_full_scene) is in camera frame — always use
                    # the camera→world fallback path regardless of in_wf.
                    if debug_info.get("pts_full_scene") is not None:
                        pts_full = pts
                        in_wf = False  # force camera-frame path for world transform
                    elif centroid is not None:
                        pts_full = pts + np.asarray(centroid)
                    else:
                        centroid = pts.mean(axis=0)
                        pts_full = pts

                    if in_wf:
                        # Points are in RH world frame → convert to Unity LH (negate X)
                        pts_unity = pts_full.copy()
                        pts_unity[:, 0] *= -1.0
                    else:
                        # Fallback: points in RH camera frame → apply camera transform
                        # Uses the exact same math as GraspFrameTransform:
                        #   1. Negate X (RH→LH flip: undo the X-negate done in Step 2)
                        #   2. Rotate by Unity camera quaternion
                        #   3. Add camera position
                        cam_pos_raw = debug_info.get("cam_pos")
                        cam_rot_raw = debug_info.get("cam_rot")
                        has_cam = cam_pos_raw is not None and cam_rot_raw is not None
                        if has_cam:
                            # For dashboard display only: convert Q-matrix output
                            # (X-right, Y-up, Z-negative) to Unity LH world frame.
                            # Only Z needs negating to reach Unity camera frame,
                            # then rotate by camera quaternion and add position.
                            pts_de = pts_full.astype(np.float64).copy()
                            pts_de[:, 2] *= -1.0  # Z-negative → Z-forward
                            cam_pos_raw_arr = np.array(cam_pos_raw, dtype=np.float64)
                            cam_rot_raw_arr = np.array(cam_rot_raw, dtype=np.float64)
                            cam_rot_raw_arr /= np.linalg.norm(cam_rot_raw_arr) + 1e-12
                            _qvec = cam_rot_raw_arr[:3]
                            _w = cam_rot_raw_arr[3]
                            _t = 2.0 * np.cross(_qvec, pts_de)
                            pts_unity = pts_de + _w * _t + np.cross(_qvec, _t) + cam_pos_raw_arr
                        else:
                            pts_unity = pts_full

                    # Workspace bounding box filter (Unity LH world).
                    # Z is negative-forward in this scene (camera behind origin),
                    # so allow a wide Z range to avoid culling valid points.
                    _WS_MIN = np.array([-2.0, -0.2, -5.0])
                    _WS_MAX = np.array([ 2.0,  2.0,  2.0])
                    ws_mask = np.all(
                        (pts_unity >= _WS_MIN) & (pts_unity <= _WS_MAX), axis=1
                    )
                    n_before = pts_unity.shape[0]
                    pts_filtered = pts_unity[ws_mask]
                    logger.info(
                        f"[VGN] Dashboard WS filter: {pts_filtered.shape[0]}/{n_before} pts "
                        f"(world_frame={in_wf}) "
                        f"pre-filter X=[{float(pts_unity[:,0].min()):.2f},{float(pts_unity[:,0].max()):.2f}] "
                        f"Y=[{float(pts_unity[:,1].min()):.2f},{float(pts_unity[:,1].max()):.2f}] "
                        f"Z=[{float(pts_unity[:,2].min()):.2f},{float(pts_unity[:,2].max()):.2f}]"
                    )
                    # If filter removes everything, send unfiltered
                    if pts_filtered.shape[0] > 0:
                        pts_unity = pts_filtered

                    # Subsample
                    if pts_unity.shape[0] > 20000:
                        pts_unity = pts_unity[::(pts_unity.shape[0] // 20000 + 1)]

                    pts_b64 = base64.b64encode(
                        pts_unity.astype(np.float32).tobytes()
                    ).decode('utf-8')
                    tsdf_b64 = None
                    if grid is not None:
                        tsdf_b64 = base64.b64encode(
                            grid.astype(np.float32).tobytes()
                        ).decode('utf-8')

                    # Centroid for TSDF un-centring in JS.
                    # Grasps are always converted to Unity LH world when _in_world_frame is True
                    # (X negated from RH world).  Send centroid in the same frame so JS can
                    # un-centre TSDF voxels consistently with where grasps are placed.
                    _internal_wf = debug_info.get("_in_world_frame", False)
                    if _internal_wf and centroid is not None:
                        c_unity = [-centroid[0], centroid[1], centroid[2]]
                    elif centroid is not None:
                        c_unity = centroid.tolist()
                    else:
                        c_unity = [0, 0, 0]

                    payload = {
                        "pointcloud_b64": pts_b64,
                        "tsdf_b64": tsdf_b64,
                        "tsdf_size": 0.3,
                        "tsdf_res": 40,
                        "grasps": grasps,
                        "centroid": c_unity,
                        "world_frame": True,
                    }
                    broadcast_vgn_debug(payload)
                    logger.info(
                        f"[VGN] Broadcasted VGN debug: {pts_unity.shape[0]} pts "
                        f"X=[{float(pts_unity[:,0].min()):.3f}, {float(pts_unity[:,0].max()):.3f}] "
                        f"Y=[{float(pts_unity[:,1].min()):.3f}, {float(pts_unity[:,1].max()):.3f}] "
                        f"Z=[{float(pts_unity[:,2].min()):.3f}, {float(pts_unity[:,2].max()):.3f}]"
                    )
            except Exception as exc:
                logger.warning(f"[VGN] Could not broadcast VGN debug info: {exc}")

    def _predict_grasps_internal(
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
        cam_pos: "Optional[List[float]]" = None,
        cam_rot: "Optional[List[float]]" = None,
        debug_info: dict = {},
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
        debug_info["cam_pos"] = cam_pos
        debug_info["cam_rot"] = cam_rot
        
        # Q-matrix output is already (X-right, Y-up, Z-negative).
        # No axis flip needed here; keep frame as-is for projection math.
        pts_rh = points.copy()
        
        debug_info["pts"] = pts_rh
        debug_info["pts_full_scene"] = pts_rh  # full cloud for dashboard; pts is overwritten later

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
        _MIN_POINTS = 20
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
        # Step 2b — Transform to world frame for VGN
        # ----------------------------------------------------------------
        # VGN was trained on axis-aligned table-top scenes (Y-up).
        # Transform from Unity camera frame to world frame so the table
        # surface is flat.
        #   1. Negate Z to get Unity LH camera frame (X-right, Y-up, Z-forward)
        #   2. Rotate by Unity camera quaternion → Unity LH world
        #   3. Add camera position
        #   4. Negate X → RH world (for scipy/VGN compatibility)
        _in_world_frame = False
        if cam_pos is not None and cam_rot is not None:
            cam_p = np.array(cam_pos, dtype=np.float64)
            cam_q = np.array(cam_rot, dtype=np.float64)  # Unity [x,y,z,w]
            cam_q = cam_q / (np.linalg.norm(cam_q) + 1e-12)

            # Step 1: negate Z to reach Unity LH camera frame
            pts_de = masked_points.astype(np.float64).copy()
            pts_de[:, 2] *= -1.0
            # Step 2: vectorised quaternion rotation
            qvec = cam_q[:3]
            w = cam_q[3]
            t = 2.0 * np.cross(qvec, pts_de)
            pts_lh_world = pts_de + w * t + np.cross(qvec, t) + cam_p
            # Step 4: Unity LH → RH world (negate X)
            pts_lh_world[:, 0] *= -1.0
            masked_points = pts_lh_world
            _in_world_frame = True
            logger.info(
                f"[VGN] Transformed {masked_points.shape[0]} points to RH world frame "
                f"(Y range: [{masked_points[:, 1].min():.3f}, {masked_points[:, 1].max():.3f}])"
            )

            # Workspace bounding box filter (RH world = Unity LH world with X negated).
            # With corrected transform: camera at Z≈-0.75 looks toward +Z, so objects
            # appear at world Z ≈ 0.0–1.5.  Y ≈ -0.1 (below table) to 1.0 (above).
            # X bounds cover ±0.8m from robot centre line.
            _WS_MIN = np.array([-0.8, -0.2, -1.0])
            _WS_MAX = np.array([ 0.8,  1.2,  2.0])
            ws_mask = np.all(
                (masked_points >= _WS_MIN) & (masked_points <= _WS_MAX), axis=1
            )
            n_before = masked_points.shape[0]
            filtered = masked_points[ws_mask]
            logger.info(
                f"[VGN] Workspace filter: {filtered.shape[0]}/{n_before} points retained"
            )
            if filtered.shape[0] >= _MIN_POINTS:
                masked_points = filtered
            else:
                logger.warning(
                    f"[VGN] Workspace filter left {filtered.shape[0]} pts (< {_MIN_POINTS}); "
                    f"skipping filter and using all {n_before} bbox-masked points"
                )

        # ----------------------------------------------------------------
        # Step 3 — TSDF construction
        # ----------------------------------------------------------------
        centroid = masked_points.mean(axis=0)

        debug_info["centroid"] = centroid
        debug_info["pts"] = masked_points - centroid
        debug_info["_in_world_frame"] = _in_world_frame

        _TSDF_SIZE = 0.3  # 30 cm workspace
        _TSDF_RES = 40

        # VGN was trained with Z-up (table normal = +Z, gripper approaches from above = -Z).
        # Our RH world frame has Y-up and Z-forward (camera depth).
        # Remap: VGN_X = world_X,  VGN_Y = -world_Z,  VGN_Z = world_Y
        # so the table surface faces +VGN_Z and the gripper descends in -VGN_Z.
        #
        # We use the raw (un-centred) masked_points so we can apply per-axis centring
        # in the VGN frame, matching the training distribution:
        #   - VGN_X, VGN_Y: centred on the object's centroid in that axis.
        #   - VGN_Z (= world_Y, the height axis): shifted so the object bottom sits
        #     one voxel above the grid floor, matching the "object on table" distribution.
        pts_vgn = masked_points[:, [0, 2, 1]].copy()
        pts_vgn[:, 1] *= -1.0  # world_Z → -world_Z for VGN_Y

        # Centre X and Y in the VGN frame; record offsets for inverse mapping.
        _vgn_centroid_x = pts_vgn[:, 0].mean()
        _vgn_centroid_y = pts_vgn[:, 1].mean()
        pts_vgn[:, 0] -= _vgn_centroid_x
        pts_vgn[:, 1] -= _vgn_centroid_y

        _vox_size = _TSDF_SIZE / _TSDF_RES
        _half = _TSDF_SIZE / 2.0

        # Scale the cloud so the object fills ~75% of the 30cm grid.
        # VGN was trained on objects that occupy a significant fraction of the workspace;
        # tiny objects (e.g. 5cm cube in a 30cm grid) cause VGN to hallucinate grasps
        # at the grid ceiling rather than on the object surface.
        # Scale is based on the max of X and Z (VGN_X = world_X, VGN_Z = world_Y = height),
        # NOT VGN_Y (= -world_Z = camera depth) because depth varies with perspective.
        _TARGET_FILL = 0.75
        extents = pts_vgn.max(axis=0) - pts_vgn.min(axis=0)  # [ex, ey, ez]
        xz_max_extent = max(extents[0], extents[2])           # ignore depth axis
        if xz_max_extent > 1e-4:
            _vgn_scale = (_TARGET_FILL * _TSDF_SIZE) / xz_max_extent
        else:
            _vgn_scale = 1.0
        pts_vgn *= _vgn_scale

        # Shift VGN_Z so the object bottom sits at grid floor + 1 voxel margin.
        z_min_vgn = pts_vgn[:, 2].min()
        _vgn_z_shift = -z_min_vgn - _half + _vox_size
        pts_vgn[:, 2] += _vgn_z_shift

        logger.debug(
            f"[VGN] TSDF prep: scale={_vgn_scale:.3f} extents(X,Y_depth,Z_height)={[round(v,4) for v in extents.tolist()]} "
            f"xz_max={xz_max_extent:.4f} z_shift={_vgn_z_shift:.4f}"
        )

        grid = _points_to_tsdf_grid(pts_vgn, size=_TSDF_SIZE, resolution=_TSDF_RES)
        # grid shape: (1, 40, 40, 40)
        debug_info["grid"] = grid

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

                # Move tensors to CPU numpy (process/select expect numpy arrays)
                qual_np = qual_vol.cpu().squeeze().numpy()
                rot_np = rot_vol.cpu().squeeze().numpy()
                width_np = width_vol.cpu().squeeze().numpy()

                # process() needs the TSDF grid to mask surface voxels
                qual_np, rot_np, width_np = process(grid, qual_np, rot_np, width_np)
                grasps, scores = select(qual_np, rot_np, width_np)

                # select() returns grasps with translation in voxel indices (0–40).
                # Convert to metres by multiplying by voxel_size.
                voxel_size = _TSDF_SIZE / _TSDF_RES
                from vgn.grasp import from_voxel_coordinates  # type: ignore
                grasps = [from_voxel_coordinates(g, voxel_size) for g in grasps]
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

        frame_label = "RH world" if _in_world_frame else "camera"
        logger.info(
            f"[VGN] centroid ({frame_label} frame): {centroid.tolist()} | "
            f"VGN offsets: x={_vgn_centroid_x:.4f} y={_vgn_centroid_y:.4f} "
            f"z_shift={_vgn_z_shift:.4f} scale={_vgn_scale:.3f}"
        )
        logger.info(f"[VGN] TSDF size={_TSDF_SIZE}m res={_TSDF_RES} voxel_size={_TSDF_SIZE/_TSDF_RES:.4f}m")
        if grasps:
            logger.info(f"[VGN] first grasp raw translation (VGN frame, from corner): {grasps[0].pose.translation.tolist()}")
        results = []
        for grasp, score in zip(grasps, scores):
            try:
                # grasp.pose.translation is voxel_index * voxel_size, range [0, TSDF_SIZE).
                # The forward pipeline applied, in order:
                #   1. Axis remap (world→VGN)
                #   2. Subtract X/Y centroids
                #   3. Uniform scale (* _vgn_scale)
                #   4. Z floor shift (+ _vgn_z_shift) on VGN_Z
                #   5. Grid: centred coords + _half → voxel index → * voxel_size
                # Undo in reverse:
                t = grasp.pose.translation.copy() - _half  # step 5 inv: back to centred+scaled
                t[2] -= _vgn_z_shift                       # step 4 inv: undo Z shift
                t /= _vgn_scale                            # step 3 inv: undo scale
                t[0] += _vgn_centroid_x                    # step 2 inv: undo X/Y centering
                t[1] += _vgn_centroid_y
                # Undo axis remap: VGN_X=world_X, VGN_Y=-world_Z, VGN_Z=world_Y
                # Inverse: world_X=VGN_X, world_Y=VGN_Z, world_Z=-VGN_Y
                t_world = np.array([t[0], t[2], -t[1]])
                pos = t_world  # already in RH world frame (not centred, using raw world coords)
                logger.info(
                    f"[VGN] grasp vgn={[round(v,4) for v in t.tolist()]} "
                    f"→pos_rh={[round(v,4) for v in pos.tolist()]}"
                )

                # Remap the full rotation matrix: R_world = P^T @ R_vgn @ P
                # where P maps world→VGN: col0=(1,0,0), col1=(0,0,-1), col2=(0,1,0)
                # i.e. P[:,0]=world_X, P[:,1]=-world_Z, P[:,2]=world_Y
                P = np.array([
                    [1.0,  0.0,  0.0],
                    [0.0,  0.0,  1.0],
                    [0.0, -1.0,  0.0],
                ], dtype=np.float64)
                rot_matrix = grasp.pose.rotation.as_matrix()
                rot_world = P.T @ rot_matrix @ P

                from scipy.spatial.transform import Rotation as _R  # type: ignore
                quat = _R.from_matrix(rot_world).as_quat()  # [qx, qy, qz, qw]
                approach = rot_world[:, 2]  # Z-axis = approach in world frame

                if _in_world_frame:
                    # Convert from RH world → Unity LH world: reflect across YZ plane (X → -X).
                    pos_out = (pos * np.array([-1.0, 1.0, 1.0])).tolist()
                    approach_out = (approach * np.array([-1.0, 1.0, 1.0])).tolist()
                    logger.info(f"[VGN] pos_out(UnityLH)={[round(v,4) for v in pos_out]}")
                    # Rotation RH→LH: R_lh = M @ R_world @ M, where M = diag(-1,1,1).
                    # This correctly handles all rotation components (not just qx flip).
                    M = np.diag([-1.0, 1.0, 1.0])
                    rot_lh = M @ rot_world @ M
                    quat_lh = _R.from_matrix(rot_lh).as_quat()
                    quat_out = [float(quat_lh[0]), float(quat_lh[1]),
                                float(quat_lh[2]), float(quat_lh[3])]
                else:
                    pos_out = pos.tolist()
                    approach_out = approach.tolist()
                    quat_out = [float(quat[0]), float(quat[1]),
                                float(quat[2]), float(quat[3])]

                results.append(
                    {
                        "position": pos_out,
                        "rotation": quat_out,
                        "score": float(score),
                        "width": float(grasp.width) / _vgn_scale,  # undo scale
                        "approach_direction": approach_out,
                        "_world_frame": _in_world_frame,
                    }
                )
            except Exception as exc:
                logger.debug(f"[VGN] Skipping malformed grasp: {exc}")
                continue

        if not results:
            return None

        results.sort(key=lambda g: g["score"], reverse=True)
        top_results = results[:top_k]
        debug_info["grasps"] = top_results

        logger.info(
            f"[VGN] Returning {len(top_results)} / {len(results)} grasps "
            f"(frame: {'Unity LH world' if _in_world_frame else 'RH camera'})"
        )
        return top_results

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
