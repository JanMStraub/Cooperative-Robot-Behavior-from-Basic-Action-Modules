#!/usr/bin/env python3
"""
Point Cloud Operations
======================

Generates dense 3D point clouds from the robot's stereo camera pair using
semi-global block matching (SGBM) disparity estimation. The resulting cloud
is expressed in the Unity camera frame with the X-axis already negated (LH
convention) so it can be consumed directly by downstream grasp operations.
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
    from config.Vision import SAVE_DEBUG_POINT_CLOUDS, DEBUG_POINT_CLOUD_DIR
except ImportError:
    from ..config.Vision import SAVE_DEBUG_POINT_CLOUDS, DEBUG_POINT_CLOUD_DIR


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

    The output points are expressed in the Unity camera frame with the
    X-axis negated (left-handed convention), ready for VGN
    inference.  Camera extrinsics (position + rotation in Unity world
    frame) are included so downstream operations can transform poses back
    into world space.

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
                "points":          [[x, y, z], ...],   # camera frame, X-negated
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
        _f_norm = 1.0 / (2.0 * _math.tan(_math.radians(fov / 2.0)))
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
        if SAVE_DEBUG_POINT_CLOUDS:
            _save_debug_point_cloud(
                robot_id,
                raw_points,
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
