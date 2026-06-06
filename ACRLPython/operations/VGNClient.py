#!/usr/bin/env python3
"""Local YOLO→VLM→VGN grasp-pose pipeline for Apple Silicon. Heavy imports (torch, scipy) deferred to first call."""

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
    """Extract and clamp bounding-box JSON from VLM prose response; returns fallback on parse error."""
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
    """Voxelise a centred point cloud to an approximate TSDF grid (no Open3D). Returns (1, R, R, R) float32."""
    from scipy.ndimage import distance_transform_edt  # type: ignore

    half = size / 2.0
    vox_size = size / resolution

    # continuous coords → voxel indices
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

    # match Open3D sdf_trunc=4*vox_size convention; VGN process() expects > 0.5 = outside
    _TRUNC_VOXELS = 4.0
    dist_outside = np.asarray(distance_transform_edt(1 - occupancy), dtype=np.float32)
    dist_inside = np.asarray(distance_transform_edt(occupancy), dtype=np.float32)
    sdf = dist_outside - dist_inside
    sdf = np.clip(sdf, -_TRUNC_VOXELS, _TRUNC_VOXELS) / _TRUNC_VOXELS

    return sdf[np.newaxis, ...]  # (1, R, R, R)


class VGNClient:
    """Local YOLO→VLM→VGN grasp predictor. Model cached at class level, loaded on first call."""

    _net = None
    _device = None

    def __init__(self) -> None:
        try:
            from config.Servers import VGN_MODEL_PATH, VGN_TOP_K
        except ImportError:
            VGN_MODEL_PATH = "checkpoints/vgn_conv.pth"
            VGN_TOP_K = 20

        # absolute path from ACRLPython/ root
        if not os.path.isabs(VGN_MODEL_PATH):
            _root = os.path.join(os.path.dirname(__file__), "..")
            VGN_MODEL_PATH = os.path.abspath(os.path.join(_root, VGN_MODEL_PATH))

        self._model_path: str = VGN_MODEL_PATH
        self._top_k_default: int = int(VGN_TOP_K)

    def is_available(self) -> bool:
        """True when model checkpoint exists and torch is importable."""
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
        object_world_pos: "Optional[List[float]]" = None,
        detection_depth_m: "Optional[float]" = None,
        object_dimensions: "Optional[Tuple[float, float, float]]" = None,
    ) -> "Optional[List[dict]]":
        """Run VGN pipeline and broadcast debug telemetry via WebUI even on early exit.

        detection_depth_m preferred over object_world_pos for depth hint (stereo disparity > geometry).
        object_dimensions enables box surface synthesis so VGN sees all 6 faces, not just camera-visible top.
        """
        debug_info = {}
        try:
            return self._predict_grasps_internal(
                points,
                colors,
                image,
                yolo_bbox,
                object_label,
                image_width,
                image_height,
                fov,
                top_k,
                cam_pos,
                cam_rot,
                debug_info,
                object_world_pos,
                detection_depth_m,
                object_dimensions,
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
                    # pts_full_scene is in camera frame — always take camera->world path
                    if debug_info.get("pts_full_scene") is not None:
                        pts_full = pts
                        in_wf = False
                    elif centroid is not None:
                        pts_full = pts + np.asarray(centroid)
                    else:
                        centroid = pts.mean(axis=0)
                        pts_full = pts

                    if in_wf:
                        # RH world → Unity LH: negate X
                        pts_unity = pts_full.copy()
                        pts_unity[:, 0] *= -1.0
                    else:
                        # camera frame → Unity LH world (same math as GraspFrameTransform)
                        cam_pos_raw = debug_info.get("cam_pos")
                        cam_rot_raw = debug_info.get("cam_rot")
                        has_cam = cam_pos_raw is not None and cam_rot_raw is not None
                        if has_cam:
                            # dashboard display: Q-matrix output (X-right, Y-up, Z-negative) → Unity LH
                            pts_de = pts_full.astype(np.float64).copy()
                            pts_de[:, 2] *= -1.0  # Z-negative → Z-forward
                            cam_pos_raw_arr = np.array(cam_pos_raw, dtype=np.float64)
                            cam_rot_raw_arr = np.array(cam_rot_raw, dtype=np.float64)
                            cam_rot_raw_arr /= np.linalg.norm(cam_rot_raw_arr) + 1e-12
                            _qvec = cam_rot_raw_arr[:3]
                            _w = cam_rot_raw_arr[3]
                            _t = 2.0 * np.cross(_qvec, pts_de)
                            pts_unity = (
                                pts_de + _w * _t + np.cross(_qvec, _t) + cam_pos_raw_arr
                            )
                        else:
                            pts_unity = pts_full

                    # workspace bbox filter — wide Z range because camera is behind origin
                    _WS_MIN = np.array([-2.0, -0.2, -5.0])
                    _WS_MAX = np.array([2.0, 2.0, 2.0])
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
                    if (
                        pts_filtered.shape[0] > 0
                    ):  # if filter removes everything, send unfiltered
                        pts_unity = pts_filtered

                    if pts_unity.shape[0] > 20000:  # subsample for dashboard bandwidth
                        pts_unity = pts_unity[:: (pts_unity.shape[0] // 20000 + 1)]

                    pts_b64 = base64.b64encode(
                        pts_unity.astype(np.float32).tobytes()
                    ).decode("utf-8")
                    tsdf_b64 = None
                    if grid is not None:
                        tsdf_b64 = base64.b64encode(
                            grid.astype(np.float32).tobytes()
                        ).decode("utf-8")

                    # centroid for JS TSDF un-centring — must match frame of grasp positions
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
                        "tsdf_size": debug_info.get("tsdf_size", 0.12),
                        "tsdf_res": debug_info.get("tsdf_res", 40),
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
        _colors: "Optional[np.ndarray]",
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
        object_world_pos: "Optional[List[float]]" = None,
        detection_depth_m: "Optional[float]" = None,
        object_dimensions: "Optional[Tuple[float, float, float]]" = None,
    ) -> "Optional[List[dict]]":
        """Steps: VLM bbox refine → mask cloud → TSDF → VGN inference → convert poses to Unity LH world."""
        import numpy as np

        if top_k <= 0:
            top_k = self._top_k_default

        bx, by, bw, bh = yolo_bbox

        # Step 1 — VLM bbox refinement (opt-in, disabled by default)
        refined_bbox = yolo_bbox
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

        # Step 2 — Point cloud masking
        debug_info["cam_pos"] = cam_pos
        debug_info["cam_rot"] = cam_rot

        # Q-matrix output: (X-right, Y-up, Z-negative) — keep as-is for projection
        pts_rh = points.copy()

        debug_info["pts"] = pts_rh
        debug_info["pts_full_scene"] = (
            pts_rh  # full cloud for dashboard; pts is overwritten later
        )

        # depth hint: stereo disparity > WorldState geometry (geometry can be stale)
        depth_hint: "Optional[float]" = None
        _depth_hint_from_stereo = False
        if detection_depth_m is not None and detection_depth_m > 0.05:
            depth_hint = detection_depth_m
            _depth_hint_from_stereo = True
            logger.debug(f"[VGN] Depth hint from stereo detection: {depth_hint:.3f} m")
        elif (
            object_world_pos is not None and cam_pos is not None and cam_rot is not None
        ):
            try:
                _obj_w = np.array(object_world_pos, dtype=np.float64)
                _cam_p = np.array(cam_pos, dtype=np.float64)
                _cam_q = np.array(cam_rot, dtype=np.float64)
                _cam_q /= np.linalg.norm(_cam_q) + 1e-12
                # conjugate of unit quaternion = inverse rotation
                _cam_q_inv = _cam_q * np.array([-1.0, -1.0, -1.0, 1.0])
                _delta = _obj_w - _cam_p
                _qvec = _cam_q_inv[:3]
                _w = _cam_q_inv[3]
                _t = 2.0 * np.cross(_qvec, _delta)
                _delta_cam = _delta + _w * _t + np.cross(_qvec, _t)
                if (
                    _delta_cam[2] > 0.05
                ):  # Z-forward in Unity LH cam frame → positive = in front
                    depth_hint = float(_delta_cam[2])
                    logger.debug(
                        f"[VGN] Depth hint from WorldState geometry: {depth_hint:.3f} m"
                    )
            except Exception as _exc:
                logger.debug(f"[VGN] depth_hint computation failed (non-fatal): {_exc}")

        # stereo hint → tight ±0.07 m; WorldState can be stale → wider ±0.25 m
        _depth_margin = 0.07 if _depth_hint_from_stereo else 0.25

        from operations.GraspUtils import _build_segmentation_mask

        mask = _build_segmentation_mask(
            pts_rh,
            refined_bbox,
            image_width,
            image_height,
            fov,
            preferred_approach="auto",
            depth_hint=depth_hint,
            depth_margin=_depth_margin,
        )

        masked_points = pts_rh[mask]
        _MIN_POINTS = 15
        if masked_points.shape[0] < _MIN_POINTS and depth_hint is not None:
            # sparse stereo on small objects — retry with wider margin
            _retry_margin = _depth_margin * 2
            logger.warning(
                f"[VGN] Only {masked_points.shape[0]} points after depth-filtered mask "
                f"(need ≥ {_MIN_POINTS}) — retrying with wider depth margin ({_retry_margin:.2f} m)"
            )
            mask = _build_segmentation_mask(
                pts_rh,
                refined_bbox,
                image_width,
                image_height,
                fov,
                preferred_approach="auto",
                depth_hint=depth_hint,
                depth_margin=_retry_margin,
            )
            masked_points = pts_rh[mask]
        if masked_points.shape[0] < _MIN_POINTS:
            if object_dimensions is None:
                logger.warning(
                    f"[VGN] Only {masked_points.shape[0]} points after masking "
                    f"(need ≥ {_MIN_POINTS}) — aborting (no dims for box synthesis)"
                )
                return None
            logger.warning(
                f"[VGN] Only {masked_points.shape[0]} real points after masking "
                f"(need ≥ {_MIN_POINTS}) — proceeding to box synthesis"
            )

        logger.info(
            f"[VGN] Masked point cloud: {masked_points.shape[0]} / {pts_rh.shape[0]} points"
        )

        # Step 2b — camera frame → world frame so VGN sees axis-aligned table
        # 1. negate Z → Unity LH cam  2. rotate by cam quat → Unity LH world
        # 3. add cam pos  4. negate X → RH world (scipy/VGN convention)
        _in_world_frame = False
        if cam_pos is not None and cam_rot is not None:
            cam_p = np.array(cam_pos, dtype=np.float64)
            cam_q = np.array(cam_rot, dtype=np.float64)  # Unity [x,y,z,w]
            cam_q = cam_q / (np.linalg.norm(cam_q) + 1e-12)

            pts_de = masked_points.astype(np.float64).copy()
            pts_de[:, 2] *= -1.0  # step 1: negate Z → Unity LH cam frame
            # step 2: vectorised quaternion rotation
            qvec = cam_q[:3]
            w = cam_q[3]
            t = 2.0 * np.cross(qvec, pts_de)
            pts_lh_world = pts_de + w * t + np.cross(qvec, t) + cam_p
            pts_lh_world[:, 0] *= -1.0  # step 4: Unity LH → RH world
            masked_points = pts_lh_world
            _in_world_frame = True
            logger.info(
                f"[VGN] Transformed {masked_points.shape[0]} points to RH world frame "
                f"(X=[{masked_points[:,0].min():.3f},{masked_points[:,0].max():.3f}] "
                f"Y=[{masked_points[:,1].min():.3f},{masked_points[:,1].max():.3f}] "
                f"Z=[{masked_points[:,2].min():.3f},{masked_points[:,2].max():.3f}])"
            )

            # workspace bbox: camera at Z≈-0.75 looks +Z → objects at Z≈0–1.5
            _WS_MIN = np.array([-0.8, -0.2, -1.0])
            _WS_MAX = np.array([0.8, 1.2, 2.0])
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

        # Step 2c — surface completion: depth cam sees only top face → flat TSDF → bad grasps
        # synthesise all 6 box faces when dims available; jitter fallback otherwise
        _real_centroid = masked_points.mean(axis=0)
        rng = np.random.default_rng(seed=42)
        _n_orig = masked_points.shape[0]

        if object_dimensions is not None:
            _hw = float(object_dimensions[0]) / 2.0  # X half (width)
            _hh = float(object_dimensions[1]) / 2.0  # Y half (height)
            _hd = float(object_dimensions[2]) / 2.0  # Z half (depth)
            _PTS_PER_FACE = 200
            _faces = []
            for u_range, v_range, fixed_val, axis in [
                # (u_range, v_range, fixed_coord_val, fixed_axis_idx)
                ((-_hw, _hw), (-_hd, _hd), _hh, 1),  # top face    Y=+hh
                ((-_hw, _hw), (-_hd, _hd), -_hh, 1),  # bottom face Y=-hh
                ((-_hh, _hh), (-_hd, _hd), _hw, 0),  # right face  X=+hw
                ((-_hh, _hh), (-_hd, _hd), -_hw, 0),  # left face   X=-hw
                ((-_hw, _hw), (-_hh, _hh), _hd, 2),  # front face  Z=+hd
                ((-_hw, _hw), (-_hh, _hh), -_hd, 2),  # back face   Z=-hd
            ]:
                us = rng.uniform(u_range[0], u_range[1], _PTS_PER_FACE)
                vs = rng.uniform(v_range[0], v_range[1], _PTS_PER_FACE)
                face = np.zeros((_PTS_PER_FACE, 3), dtype=np.float32)
                axes = [i for i in range(3) if i != axis]
                face[:, axes[0]] = us
                face[:, axes[1]] = vs
                face[:, axis] = fixed_val
                _faces.append(face)
            _box_pts = np.concatenate(_faces, axis=0) + _real_centroid.astype(
                np.float32
            )
            masked_points = np.concatenate([masked_points, _box_pts], axis=0)
            logger.info(
                f"[VGN] Box synthesis: {_n_orig} real + {_box_pts.shape[0]} synthetic pts "
                f"(dims={[round(float(d), 4) for d in object_dimensions]}) "
                f"→ {masked_points.shape[0]} total"
            )
        else:
            # no dim data — jitter densification
            _DENSIFY_TARGET = 800
            _n_copies = max(1, _DENSIFY_TARGET // max(1, _n_orig))
            if _n_copies > 1:
                _sigma = 0.005
                noise = rng.normal(scale=_sigma, size=(_n_copies - 1, _n_orig, 3))
                copies = masked_points[np.newaxis] + noise
                masked_points = np.concatenate(
                    [masked_points] + [copies[i] for i in range(_n_copies - 1)],
                    axis=0,
                ).astype(masked_points.dtype)
                logger.info(
                    f"[VGN] Jitter densification (no dims): {_n_orig} → {masked_points.shape[0]} pts "
                    f"({_n_copies}x, σ={_sigma} m)"
                )

        # Step 3 — TSDF
        centroid = masked_points.mean(axis=0)

        debug_info["centroid"] = centroid
        debug_info["pts"] = masked_points - centroid
        debug_info["_in_world_frame"] = _in_world_frame

        _TSDF_SIZE = 0.12  # 12 cm fits a 5 cm cube at ~1.5× fill
        _TSDF_RES = 40

        # VGN: Z-up (table = +Z, gripper approaches = -Z). Our RH world: Y-up, Z-forward.
        # Remap: VGN_X=world_X, VGN_Y=-world_Z, VGN_Z=world_Y → table faces +VGN_Z
        pts_vgn = masked_points[:, [0, 2, 1]].copy()
        pts_vgn[:, 1] *= -1.0  # world_Z → -world_Z for VGN_Y

        # clip stereo depth noise in VGN_Y: 1m working dist + 5cm baseline spreads cube
        # ~14cm in depth → dominates scale. clip to [median ± half_obj] from X/Z extents.
        _vgn_y_median = float(np.median(pts_vgn[:, 1]))
        _x_ext_pre = float(pts_vgn[:, 0].max() - pts_vgn[:, 0].min())
        _z_ext_pre = float(pts_vgn[:, 2].max() - pts_vgn[:, 2].min())
        _half_obj = max(_x_ext_pre, _z_ext_pre, 0.03) / 2.0  # at least 3 cm radius
        pts_vgn[:, 1] = np.clip(
            pts_vgn[:, 1],
            _vgn_y_median - _half_obj,
            _vgn_y_median + _half_obj,
        )
        logger.debug(
            f"[VGN] VGN_Y clip: median={_vgn_y_median:.4f} half_obj={_half_obj:.4f} "
            f"new_Y_range=[{pts_vgn[:,1].min():.4f},{pts_vgn[:,1].max():.4f}]"
        )

        _vgn_centroid_x = pts_vgn[:, 0].mean()
        _vgn_centroid_y = pts_vgn[:, 1].mean()
        pts_vgn[:, 0] -= _vgn_centroid_x
        pts_vgn[:, 1] -= _vgn_centroid_y

        _vox_size = _TSDF_SIZE / _TSDF_RES
        _half = _TSDF_SIZE / 2.0

        # scale cloud to ~75% grid fill; drive on VGN_Z (height) not max(X,Y,Z) —
        # max() under-scales height → flat disc → VGN prefers horizontal side grasps
        _TARGET_FILL = 0.75
        _MAX_SCALE = 5.0  # 5cm cube × 5 = 25cm in 12cm grid is intentional overflow
        extents = pts_vgn.max(axis=0) - pts_vgn.min(axis=0)  # [ex, ey, ez]
        xz_max_extent = max(extents[0], extents[2])  # for logging
        _z_extent = float(extents[2])  # VGN_Z = height axis
        if _z_extent > 1e-4:
            _vgn_scale = min((_TARGET_FILL * _TSDF_SIZE) / _z_extent, _MAX_SCALE)
        elif xz_max_extent > 1e-4:
            _vgn_scale = min((_TARGET_FILL * _TSDF_SIZE) / xz_max_extent, _MAX_SCALE)
        else:
            _vgn_scale = 1.0
        pts_vgn *= _vgn_scale

        # shift so object bottom sits at grid floor + 1 voxel (matches training distribution)
        z_min_vgn = pts_vgn[:, 2].min()
        _vgn_z_shift = -z_min_vgn - _half + _vox_size
        pts_vgn[:, 2] += _vgn_z_shift

        logger.debug(
            f"[VGN] TSDF prep: scale={_vgn_scale:.3f} extents(X,Y_depth,Z_height)={[round(v,4) for v in extents.tolist()]} "
            f"xz_max={xz_max_extent:.4f} z_shift={_vgn_z_shift:.4f}"
        )

        grid = _points_to_tsdf_grid(pts_vgn, size=_TSDF_SIZE, resolution=_TSDF_RES)
        debug_info["grid"] = grid
        debug_info["tsdf_size"] = _TSDF_SIZE
        debug_info["tsdf_res"] = _TSDF_RES

        from config.Vision import (
            VGN_EXPORT_TSDF,
            VGN_EXPORT_TSDF_PATH,
            VGN_EXPORT_TSDF_OBJ,
        )

        if VGN_EXPORT_TSDF:
            np.savez(
                VGN_EXPORT_TSDF_PATH,
                grid=grid,
                size=np.float32(_TSDF_SIZE),
                res=np.int32(_TSDF_RES),
            )
            logger.info(f"[VGN] TSDF grid exported to {VGN_EXPORT_TSDF_PATH}")

            if VGN_EXPORT_TSDF_OBJ:
                try:
                    from skimage import measure
                    import os as _os

                    verts, faces, _, _ = measure.marching_cubes(
                        grid.squeeze(), level=0.0
                    )
                    verts = verts / grid.shape[0] * _TSDF_SIZE
                    obj_path = _os.path.splitext(VGN_EXPORT_TSDF_PATH)[0] + ".obj"
                    with open(obj_path, "w") as _f:
                        for v in verts:
                            _f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                        for face in faces:
                            _f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                    logger.info(f"[VGN] TSDF mesh exported to {obj_path}")
                except ImportError:
                    logger.warning(
                        "[VGN] skimage not installed — skipping OBJ export (pip install scikit-image)"
                    )

        # Step 4 — inference
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

                qual_np = qual_vol.cpu().squeeze().numpy()
                rot_np = rot_vol.cpu().squeeze().numpy()
                width_np = width_vol.cpu().squeeze().numpy()

                qual_np, rot_np, width_np = process(grid, qual_np, rot_np, width_np)
                grasps, scores = select(qual_np, rot_np, width_np)

                # select() returns voxel indices — convert to metres
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

        # Step 5 — convert VGN poses to output format
        frame_label = "RH world" if _in_world_frame else "camera"
        logger.info(
            f"[VGN] centroid ({frame_label} frame): {centroid.tolist()} | "
            f"VGN offsets: x={_vgn_centroid_x:.4f} y={_vgn_centroid_y:.4f} "
            f"z_shift={_vgn_z_shift:.4f} scale={_vgn_scale:.3f}"
        )
        logger.info(
            f"[VGN] TSDF size={_TSDF_SIZE}m res={_TSDF_RES} voxel_size={_TSDF_SIZE/_TSDF_RES:.4f}m"
        )
        if grasps:
            logger.info(
                f"[VGN] first grasp raw translation (VGN frame, from corner): {grasps[0].pose.translation.tolist()}"
            )
        results = []
        for grasp, score in zip(grasps, scores):
            try:
                # undo forward pipeline in reverse: remap→centroids→scale→Z-shift→grid
                t = grasp.pose.translation.copy() - _half  # step 5 inv
                t[2] -= _vgn_z_shift
                t /= _vgn_scale
                t[0] += _vgn_centroid_x
                t[1] += _vgn_centroid_y
                # inverse remap: VGN_X→world_X, VGN_Z→world_Y, -VGN_Y→world_Z
                t_world = np.array([t[0], t[2], -t[1]])
                pos = t_world  # RH world frame, raw coords
                logger.debug(
                    f"[VGN] grasp vgn={[round(v,4) for v in t.tolist()]} "
                    f"→pos_rh={[round(v,4) for v in pos.tolist()]}"
                )

                # R_world = P^T @ R_vgn @ P  where P: world→VGN basis
                P = np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, -1.0, 0.0],
                    ],
                    dtype=np.float64,
                )
                rot_matrix = grasp.pose.rotation.as_matrix()
                rot_world = P.T @ rot_matrix @ P

                from scipy.spatial.transform import Rotation as _R  # type: ignore

                quat = _R.from_matrix(rot_world).as_quat()  # [qx, qy, qz, qw]
                # VGN Z-axis points into surface; negate → approach vector from object to gripper
                approach = -rot_world[:, 2]

                if _in_world_frame:
                    # RH world → Unity LH: reflect X
                    pos_out = (pos * np.array([-1.0, 1.0, 1.0])).tolist()
                    approach_out = (approach * np.array([-1.0, 1.0, 1.0])).tolist()
                    logger.debug(
                        f"[VGN] pos_out(UnityLH)={[round(v,4) for v in pos_out]}"
                    )
                    # R_lh = M @ R_world @ M, M=diag(-1,1,1) — handles all components, not just qx
                    M = np.diag([-1.0, 1.0, 1.0])
                    rot_lh = M @ rot_world @ M
                    quat_lh = _R.from_matrix(rot_lh).as_quat()
                    quat_out = [
                        float(quat_lh[0]),
                        float(quat_lh[1]),
                        float(quat_lh[2]),
                        float(quat_lh[3]),
                    ]
                else:
                    pos_out = pos.tolist()
                    approach_out = approach.tolist()
                    quat_out = [
                        float(quat[0]),
                        float(quat[1]),
                        float(quat[2]),
                        float(quat[3]),
                    ]

                results.append(
                    {
                        "position": pos_out,
                        "rotation": quat_out,
                        "score": float(score),
                        "width": float(grasp.width) / _vgn_scale,
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

    @classmethod
    def _load_model(cls):
        """Lazy-load VGN network; tries MPS first, falls back to CPU. Returns nn.Module or None."""
        if cls._net is not None:
            return cls._net

        if not _ensure_vgn_on_path():
            logger.warning("[VGN] Cannot load model: VGN source directory missing")
            return None

        try:
            import torch
            from vgn.networks import get_network  # type: ignore

            if torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("[VGN] Using MPS (Apple Silicon)")
            else:
                device = torch.device("cpu")
                logger.info("[VGN] Using CPU")

            net = get_network("conv")
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
        """Clear cached model — use in test teardown to force reload on next call."""
        cls._net = None
        cls._device = None
        logger.debug("[VGN] Model cache cleared")
