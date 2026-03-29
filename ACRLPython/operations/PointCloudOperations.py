#!/usr/bin/env python3
"""
Point Cloud Operations
======================

Generates dense 3D point clouds from the robot's stereo camera pair using
semi-global block matching (SGBM) disparity estimation. The resulting cloud
is expressed in Unity camera frame with X negated (left-handed convention).
Downstream consumers (VGNClient) apply their own camera→world transform using
the camera_position and camera_rotation fields returned alongside the points.
"""

import logging
import time

import numpy as np

from .Base import (
    BasicOperation,
    OperationCategory,
    OperationComplexity,
    OperationParameter,
    OperationRelationship,
    OperationResult,
)
from core.LoggingSetup import setup_logging

setup_logging(__name__)
logger = logging.getLogger(__name__)

# Module-level lazy imports — imported here so tests can monkeypatch them
# while still respecting the layered architecture (no server→operation circular dep).
try:
    from core.Imports import get_unified_image_storage
except ImportError:
    from ..core.Imports import get_unified_image_storage

try:
    from vision.StereoReconstruction import stereo_reconstruct_stream
except ImportError:
    from ..vision.StereoReconstruction import stereo_reconstruct_stream

try:
    from config.Vision import (
        SAVE_DEBUG_POINT_CLOUDS,
        DEBUG_POINT_CLOUD_DIR,
        POINT_CLOUD_CLEANING_ENABLED,
        POINT_CLOUD_BG_COLORS_ENABLED,
        POINT_CLOUD_BG_COLORS,
        POINT_CLOUD_MIN_DEPTH,
        POINT_CLOUD_MAX_DEPTH,
        POINT_CLOUD_OUTLIER_NB_NEIGHBORS,
        POINT_CLOUD_OUTLIER_STD_RATIO,
    )
except ImportError:
    from ..config.Vision import (
        SAVE_DEBUG_POINT_CLOUDS,
        DEBUG_POINT_CLOUD_DIR,
        POINT_CLOUD_CLEANING_ENABLED,
        POINT_CLOUD_BG_COLORS_ENABLED,
        POINT_CLOUD_BG_COLORS,
        POINT_CLOUD_MIN_DEPTH,
        POINT_CLOUD_MAX_DEPTH,
        POINT_CLOUD_OUTLIER_NB_NEIGHBORS,
        POINT_CLOUD_OUTLIER_STD_RATIO,
    )


# ============================================================================
# Module-Level Constants
# ============================================================================

# Maximum age (seconds) of cached stereo images before rejecting.
# If the images are older than this the scene may have changed significantly.
# 30s allows for LLM parse latency (typically 3-5s) while still catching truly stale images.
_DEFAULT_MAX_AGE_SECONDS: float = 30.0

# Uniform random downsample target: limits point cloud size for downstream inference.
_DEFAULT_MAX_POINTS: int = 50_000

# Default camera pair identifier stored in UnifiedImageStorage.
_DEFAULT_CAMERA_PAIR_ID: str = "stereo"


# ============================================================================
# Debug Helpers
# ============================================================================


def _save_debug_point_cloud(
    robot_id: str,
    points: np.ndarray,
    colors: np.ndarray,
    camera_position: list,
    camera_rotation: list,
    timestamp: float,
) -> None:
    """Save a point cloud to disk as a binary PLY file for inspection in Blender.

    PLY (Polygon File Format) is natively supported by Blender's built-in
    "Import PLY" operator (File → Import → Stanford PLY).  Per-vertex RGB
    colors are stored as uchar properties and displayed immediately when the
    viewport shading is set to "Vertex Color".

    The file is written to DEBUG_POINT_CLOUD_DIR and named using the robot ID
    and a human-readable UTC timestamp so successive captures do not overwrite
    each other.  Errors are logged but never propagated — debug saves must
    never break the main operation flow.

    Args:
        robot_id: Robot whose camera produced this cloud.
        points: Float32 array of shape (N, 3) in camera frame.
        colors: Uint8 array of shape (N, 3) RGB colours.
        camera_position: [x, y, z] world-space camera origin.
        camera_rotation: [x, y, z, w] or Euler camera rotation.
        timestamp: Unix epoch of the stereo capture.
    """
    import os
    import struct
    from datetime import datetime, timezone

    try:
        os.makedirs(DEBUG_POINT_CLOUD_DIR, exist_ok=True)
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        fname = f"pc_{robot_id}_{dt.strftime('%Y%m%d_%H%M%S_%f')}.ply"
        path = os.path.join(DEBUG_POINT_CLOUD_DIR, fname)

        n = points.shape[0]
        pts = points.astype(np.float32)
        clrs = colors.astype(np.uint8)

        # Build ASCII PLY header
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"comment robot_id {robot_id}\n"
            f"comment camera_position {camera_position[0]:.6f} {camera_position[1]:.6f} {camera_position[2]:.6f}\n"
            f"comment camera_rotation {' '.join(f'{v:.6f}' for v in camera_rotation)}\n"
            f"comment timestamp {timestamp:.6f}\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )

        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            # Interleave xyz (3×float32 = 12 bytes) + rgb (3×uint8 = 3 bytes) per vertex
            for i in range(n):
                f.write(struct.pack("<fff", pts[i, 0], pts[i, 1], pts[i, 2]))
                f.write(struct.pack("BBB", clrs[i, 0], clrs[i, 1], clrs[i, 2]))

        logger.debug(
            f"[{robot_id}] Debug point cloud saved to {path} ({n} points, PLY)"
        )
    except Exception as exc:
        logger.warning(f"[{robot_id}] Failed to save debug point cloud: {exc}")


# ============================================================================
# Internal helpers
# ============================================================================


def _apply_camera_rotation(points: np.ndarray, quaternion: list) -> np.ndarray:
    """Rotate points from Unity camera frame to Unity world frame.

    Expects points already in Unity camera frame (X-right, Y-up, Z-forward).
    The Q-matrix output is (X-right, Y-up, Z-negative); callers must negate Z
    before calling this function.  Applies the camera's world-space rotation
    quaternion to go from Unity camera space to Unity world space.

    Args:
        points: Float32 (N, 3) in Unity camera frame (Q-matrix output).
        quaternion: Camera world rotation ``[x, y, z, w]`` from Unity metadata.

    Returns:
        Float32 (N, 3) in Unity world frame.
    """
    x, y, z, w = quaternion
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return points @ R.T


def _clean_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
) -> tuple:
    """Remove background points and statistical outliers from a point cloud.

    Three-stage pipeline (each stage independently skippable via config):

    1. **Depth clip** — discard points with Z (camera-forward) outside
       [POINT_CLOUD_MIN_DEPTH, POINT_CLOUD_MAX_DEPTH].  Fast numpy operation;
       removes far-field sky/floor noise before the more expensive steps.

    2. **Color-based background removal** — removes points whose RGB color
       closely matches known Unity background/floor colors (configured in
       POINT_CLOUD_BG_COLORS).  Adapted from RoboScan's ``remove_colors()``.
       Uses per-channel absolute tolerance so near-matches are also caught.

    3. **Statistical outlier removal** — removes points whose mean distance
       to their k nearest neighbors exceeds mean + std_ratio * σ.  Uses
       Open3D if available; silently skipped otherwise.  Adapted from
       RoboScan's ``filter_repeating_points()`` final pass.

    Args:
        points: Float32 (N, 3) point positions in world frame.
        colors: Uint8 (N, 3) RGB colors.

    Returns:
        Tuple of (filtered_points, filtered_colors) as (float32, uint8) arrays.
    """
    if len(points) == 0:
        return points, colors

    # --- Stage 1: depth clip (Z axis = camera forward = scene depth) ---
    z = points[:, 2]
    depth_mask = (z >= POINT_CLOUD_MIN_DEPTH) & (z <= POINT_CLOUD_MAX_DEPTH)
    points = points[depth_mask]
    colors = colors[depth_mask]
    logger.debug(
        f"Depth clip [{POINT_CLOUD_MIN_DEPTH}m–{POINT_CLOUD_MAX_DEPTH}m]: "
        f"{depth_mask.sum()} / {len(depth_mask)} points kept"
    )

    if len(points) == 0:
        return points, colors

    # --- Stage 2: color-based background removal ---
    if POINT_CLOUD_BG_COLORS_ENABLED and len(POINT_CLOUD_BG_COLORS) > 0:
        # Build a boolean mask: True = keep (not background)
        keep = np.ones(len(points), dtype=bool)
        colors_f = colors.astype(np.float32)
        for rgb_tuple, tol in POINT_CLOUD_BG_COLORS:
            ref = np.array(rgb_tuple, dtype=np.float32)
            # Point matches background if ALL three channels are within tolerance
            match = np.all(np.abs(colors_f - ref) <= tol, axis=1)
            keep &= ~match
        before = len(points)
        points = points[keep]
        colors = colors[keep]
        logger.debug(
            f"Color background removal: {keep.sum()} / {before} points kept"
        )

    if len(points) == 0:
        return points, colors

    # --- Stage 3: statistical outlier removal (Open3D) ---
    try:
        import open3d as o3d  # type: ignore[import]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

        before = len(points)
        pcd_clean, ind = pcd.remove_statistical_outlier(
            nb_neighbors=POINT_CLOUD_OUTLIER_NB_NEIGHBORS,
            std_ratio=POINT_CLOUD_OUTLIER_STD_RATIO,
        )
        points = np.asarray(pcd_clean.points, dtype=np.float32)
        colors = (np.asarray(pcd_clean.colors) * 255.0).astype(np.uint8)
        logger.debug(
            f"Statistical outlier removal (nb={POINT_CLOUD_OUTLIER_NB_NEIGHBORS}, "
            f"std={POINT_CLOUD_OUTLIER_STD_RATIO}): {len(points)} / {before} points kept"
        )
    except ImportError:
        logger.debug(
            "open3d not installed — skipping statistical outlier removal. "
            "Install with: pip install open3d"
        )

    return points, colors


# ============================================================================
# Implementation
# ============================================================================


def generate_point_cloud(
    robot_id: str,
    camera_pair_id: str = _DEFAULT_CAMERA_PAIR_ID,
    max_points: int = _DEFAULT_MAX_POINTS,
    max_age_seconds: float = _DEFAULT_MAX_AGE_SECONDS,
    request_id: int = 0,
) -> OperationResult:
    """Generate a dense 3D point cloud from the latest stereo image pair.

    Uses the stereo images stored in UnifiedImageStorage (captured from
    Unity's stereo camera rig) to reconstruct a 3D point cloud via
    semi-global block-matching disparity estimation.

    The output points are expressed in Unity world frame: OpenCV camera-space
    points are Y-flipped (OpenCV→Unity camera convention), rotated by the
    camera's world-space quaternion, and translated by the camera's world
    position.  Camera extrinsics are also returned for reference.

    Args:
        robot_id: Robot whose camera pair is queried (e.g., "Robot1").
        camera_pair_id: Identifier of the stereo pair in UnifiedImageStorage.
            Usually "stereo" (default).
        max_points: Maximum number of points to return.  A uniform random
            subsample is applied when the raw cloud exceeds this limit.
        max_age_seconds: Reject images older than this many seconds (default: 30.0).
        request_id: Protocol V2 request identifier (pass-through).

    Returns:
        OperationResult.  On success the ``result`` dict contains:

        .. code-block:: python

            {
                "points":          [[x, y, z], ...],   # Unity camera frame, X-negated (LH)
                "colors":          [[r, g, b], ...],   # 0-255 per channel
                "point_count":     int,
                "camera_position": [x, y, z],          # Unity world
                "camera_rotation": [x, y, z, w],       # Unity quaternion
                "fov":             float,               # horizontal FOV degrees
                "baseline":        float,               # camera separation metres
                "timestamp":       float,               # Unix epoch of capture
            }

        On failure ``error`` contains a code, human-readable message, and
        recovery suggestions (standard OperationResult convention).
    """
    try:
        # --- Retrieve latest stereo pair ---
        storage = get_unified_image_storage()
        stereo_data = storage.get_latest_stereo_image()

        if stereo_data is None:
            return OperationResult.error_result(
                "NO_STEREO_IMAGES",
                "No stereo images available in UnifiedImageStorage",
                [
                    "Capture stereo images first using 'capture_stereo_images' command",
                    "Ensure Unity stereo camera is active and transmitting on port 5006",
                ],
            )

        img_left, img_right, _prompt, timestamp, metadata = stereo_data

        if img_left is None or img_right is None:
            return OperationResult.error_result(
                "INCOMPLETE_STEREO_PAIR",
                "One or both stereo images are missing",
                [
                    "Re-capture stereo images",
                    "Verify StereoCameraController is configured in Unity",
                ],
            )

        # --- Staleness check ---
        # Guard against callers (e.g. LLM) passing None, 0 or negative values.
        effective_max_age = (
            max_age_seconds
            if (max_age_seconds is not None and max_age_seconds > 0)
            else _DEFAULT_MAX_AGE_SECONDS
        )
        age = time.time() - timestamp
        if age > effective_max_age:
            return OperationResult.error_result(
                "STALE_IMAGE",
                f"Stereo images are {age:.1f}s old (max {effective_max_age}s). "
                "Capture fresh images before generating a point cloud.",
                [
                    "Call 'capture_stereo_images' or wait for live streaming to update",
                    "Increase max_age_seconds if images are intentionally pre-cached",
                ],
            )

        # --- Extract camera intrinsics / extrinsics from metadata ---
        try:
            from .StereoUtils import camera_config_from_metadata
        except ImportError:
            from operations.StereoUtils import camera_config_from_metadata

        stereo_params = camera_config_from_metadata(metadata)
        fov = stereo_params.camera_config.fov
        baseline = stereo_params.camera_config.baseline
        camera_position = [float(v) for v in stereo_params.camera_position]
        camera_rotation = [float(v) for v in stereo_params.camera_rotation]

        # --- Compute max_disp to cover the robotic workspace ---
        # SGBM's numDisparities must be >= f_px*baseline/min_depth.
        # The default of 128 is too small: for fov=45°, w=1920, baseline=0.05m,
        # an object at 0.8m needs ~145px disparity, exceeding the 128 cap.
        # SGBM then finds the best wrong in-range match (~79px), reporting ~1.47m
        # instead of 0.8m — a systematic ~1.8x overestimate of depth.
        # Cap at 512 for performance; covers objects down to f_px*b/512 (~0.23m).
        import math as _math
        # fov from Unity is vertical; convert to horizontal for f_px calculation
        _aspect = img_left.shape[1] / img_left.shape[0]  # width / height
        _horiz_fov = 2.0 * _math.degrees(_math.atan(_math.tan(_math.radians(fov / 2.0)) * _aspect))
        _f_norm = 1.0 / (2.0 * _math.tan(_math.radians(_horiz_fov / 2.0)))
        _f_px = _f_norm * img_left.shape[1]  # pixel focal length
        _min_depth = 0.25  # metres — closest expected object in robot workspace
        _max_disp_needed = int(_math.ceil(_f_px * baseline / _min_depth))
        _max_disp_needed = min(((_max_disp_needed + 15) // 16) * 16, 512)
        logger.info(
            f"[{robot_id}] Reconstructing point cloud from stereo pair "
            f"(fov={fov}°, baseline={baseline}m, max_disp={_max_disp_needed})"
        )
        t0 = time.time()
        point_cloud = stereo_reconstruct_stream(
            img_left,
            img_right,
            fov=fov,
            cam_dist=baseline,
            max_disp=_max_disp_needed,
        )
        elapsed_ms = (time.time() - t0) * 1000
        logger.info(
            f"[{robot_id}] Stereo reconstruction completed in {elapsed_ms:.0f}ms"
        )

        raw_points: np.ndarray = np.asarray(point_cloud["points"], dtype=np.float32)
        raw_colors: np.ndarray = np.asarray(point_cloud["colors"], dtype=np.uint8)

        # Remove NaN / Inf points that SGBM sometimes produces at depth discontinuities
        valid_mask = np.isfinite(raw_points).all(axis=1)
        raw_points = raw_points[valid_mask]
        raw_colors = raw_colors[valid_mask]

        if raw_points.shape[0] == 0:
            return OperationResult.error_result(
                "EMPTY_POINT_CLOUD",
                "Stereo reconstruction produced no valid 3D points",
                [
                    "Ensure there is sufficient texture in the scene for disparity matching",
                    "Check that the stereo baseline and FOV are correctly configured",
                ],
            )

        # --- Background removal + statistical outlier cleaning ---
        if POINT_CLOUD_CLEANING_ENABLED:
            n_before = raw_points.shape[0]
            raw_points, raw_colors = _clean_point_cloud(raw_points, raw_colors)
            logger.info(
                f"[{robot_id}] Point cloud cleaned: {raw_points.shape[0]} / {n_before} points retained"
            )
            if raw_points.shape[0] == 0:
                return OperationResult.error_result(
                    "EMPTY_POINT_CLOUD_AFTER_CLEANING",
                    "All points were removed during background/outlier cleaning",
                    [
                        "Check POINT_CLOUD_BG_COLORS in config/Vision.py — tolerance may be too broad",
                        "Disable cleaning with POINT_CLOUD_CLEANING_ENABLED=false to inspect raw cloud",
                        "Check POINT_CLOUD_MIN_DEPTH / POINT_CLOUD_MAX_DEPTH range",
                    ],
                )

        # --- Uniform random downsample ---
        # Guard against callers (e.g. LLM) passing None or non-positive values.
        effective_max_points = (
            max_points
            if (max_points is not None and max_points > 0)
            else _DEFAULT_MAX_POINTS
        )
        n_raw = raw_points.shape[0]
        if n_raw > effective_max_points:
            idx = np.random.choice(n_raw, size=effective_max_points, replace=False)
            raw_points = raw_points[idx]
            raw_colors = raw_colors[idx]
            logger.debug(
                f"[{robot_id}] Downsampled point cloud from {n_raw} to {max_points} points"
            )

        point_count = raw_points.shape[0]

        # Convert to Python lists for JSON serialisation
        points_list = raw_points.tolist()
        colors_list = raw_colors.tolist()

        logger.info(
            f"[{robot_id}] Point cloud ready: {point_count} points "
            f"(camera_position={camera_position})"
        )

        # --- Optional debug save ---
        # For Blender inspection apply camera→world transform so the PLY is
        # axis-aligned (table horizontal, robot upright).  The live pipeline
        # (VGNClient) does its own transform from the raw camera-frame points.
        if SAVE_DEBUG_POINT_CLOUDS:
            # Convert Q-matrix output to Unity camera frame before rotating.
            # Q output is (X-right, Y-up, Z-negative) — only Z needs negating to
            # reach Unity camera frame (X-right, Y-up, Z-forward).
            debug_pts = raw_points.copy().astype(np.float32)
            debug_pts[:, 2] *= -1.0  # Z-negative → Z-forward (positive)
            if camera_rotation and len(camera_rotation) == 4:
                debug_pts = _apply_camera_rotation(debug_pts, camera_rotation)
            if camera_position and len(camera_position) == 3:
                debug_pts = debug_pts + np.array(camera_position, dtype=np.float32)
            _save_debug_point_cloud(
                robot_id,
                debug_pts,
                raw_colors,
                camera_position,
                camera_rotation,
                timestamp,
            )

        return OperationResult.success_result(
            {
                "points": points_list,
                "colors": colors_list,
                "point_count": point_count,
                "camera_position": camera_position,
                "camera_rotation": camera_rotation,
                "fov": fov,
                "baseline": baseline,
                "timestamp": timestamp,
                "image_width": img_left.shape[1],
                "image_height": img_left.shape[0],
            }
        )

    except Exception as exc:
        logger.exception(f"Exception in generate_point_cloud: {exc}")
        return OperationResult.error_result(
            "EXCEPTION",
            f"Unexpected error during point cloud generation: {exc}",
            [
                "Check stack trace in logs",
                "Verify vision dependencies (numpy, opencv) are installed",
            ],
        )


# ============================================================================
# Operation Definition for Registry
# ============================================================================

GENERATE_POINT_CLOUD_OPERATION = BasicOperation(
    operation_id="perception_generate_point_cloud_001",
    name="generate_point_cloud",
    category=OperationCategory.PERCEPTION,
    complexity=OperationComplexity.BASIC,
    description=(
        "Generate a dense 3D point cloud from the robot's stereo camera pair "
        "using SGBM disparity estimation."
    ),
    long_description=(
        "Captures the latest stereo image pair from UnifiedImageStorage and "
        "reconstructs a 3D point cloud via semi-global block matching. "
        "The cloud is expressed in Unity camera space (X-negated, left-handed) "
        "and includes camera extrinsics so downstream operations can transform "
        "poses into world space. Useful as input for neural grasp planners such "
        "as VGN."
    ),
    usage_examples=[
        "generate_point_cloud('Robot1')",
        "generate_point_cloud('Robot1', max_points=30000, max_age_seconds=30.0)",
    ],
    parameters=[
        OperationParameter(
            name="robot_id",
            type="str",
            description="Robot ID whose stereo pair to use (e.g., 'Robot1')",
            required=True,
        ),
        OperationParameter(
            name="camera_pair_id",
            type="str",
            description="Stereo pair ID in UnifiedImageStorage (default: 'stereo')",
            required=False,
            default="stereo",
        ),
        OperationParameter(
            name="max_points",
            type="int",
            description="Uniform random downsample target (default: 50000)",
            required=False,
            default=50000,
        ),
        OperationParameter(
            name="max_age_seconds",
            type="float",
            description="Reject images older than this many seconds (default: 30.0)",
            required=False,
            default=30.0,
        ),
    ],
    preconditions=[
        "stereo_images_available(max_age_seconds)",
    ],
    postconditions=[],
    average_duration_ms=300.0,
    success_rate=0.85,
    failure_modes=[
        "No stereo images in storage (NO_STEREO_IMAGES)",
        "Images too old (STALE_IMAGE)",
        "Insufficient texture for disparity matching (EMPTY_POINT_CLOUD)",
    ],
    relationships=OperationRelationship(
        operation_id="perception_generate_point_cloud_001",
        required_operations=[],
        commonly_paired_with=["detect_objects", "grasp_object"],
        pairing_reasons={
            "detect_objects": "Detect objects in the scene before generating a point cloud for grasp planning",
            "grasp_object": "Point cloud feeds directly into VGN grasp planning pipeline",
        },
        typical_before=["grasp_object"],
    ),
    implementation=generate_point_cloud,
)
