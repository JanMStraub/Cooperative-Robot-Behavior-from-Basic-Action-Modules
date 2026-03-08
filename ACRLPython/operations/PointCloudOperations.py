"""
Point Cloud Operations
======================

Generates dense 3D point clouds from the robot's stereo camera pair using
semi-global block matching (SGBM) disparity estimation. The resulting cloud
is expressed in the Unity camera frame with the X-axis already negated (LH
convention) so it can be consumed directly by downstream operations such as
Contact-GraspNet inference.
"""

import logging
import time
from typing import Optional

import numpy as np

from .Base import (
    BasicOperation,
    OperationCategory,
    OperationComplexity,
    OperationParameter,
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


# ============================================================================
# Module-Level Constants
# ============================================================================

# Maximum age (seconds) of cached stereo images before rejecting.
# If the images are older than this the scene may have changed significantly.
_DEFAULT_MAX_AGE_SECONDS: float = 2.0

# Uniform random downsample target: avoids sending huge payloads to GraspNet.
_DEFAULT_MAX_POINTS: int = 50_000

# Default camera pair identifier stored in UnifiedImageStorage.
_DEFAULT_CAMERA_PAIR_ID: str = "stereo"


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
    X-axis negated (left-handed convention), ready for Contact-GraspNet
    inference.  Camera extrinsics (position + rotation in Unity world
    frame) are included so downstream operations can transform poses back
    into world space.

    Args:
        robot_id: Robot whose camera pair is queried (e.g., "Robot1").
        camera_pair_id: Identifier of the stereo pair in UnifiedImageStorage.
            Usually "stereo" (default).
        max_points: Maximum number of points to return.  A uniform random
            subsample is applied when the raw cloud exceeds this limit.
        max_age_seconds: Reject images older than this many seconds.
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
        age = time.time() - timestamp
        if age > max_age_seconds:
            return OperationResult.error_result(
                "STALE_IMAGE",
                f"Stereo images are {age:.1f}s old (max {max_age_seconds}s). "
                "Capture fresh images before generating a point cloud.",
                [
                    "Call 'capture_stereo_images' or wait for live streaming to update",
                    "Increase max_age_seconds if images are intentionally pre-cached",
                ],
            )

        # --- Extract camera intrinsics / extrinsics from metadata ---
        camera_params = metadata.get("camera_params", metadata)
        fov = float(camera_params.get("fov", 60.0))
        baseline = float(camera_params.get("baseline", camera_params.get("cam_dist", 0.1)))

        cam_pos_raw = camera_params.get("camera_position", [0.0, 0.0, 0.0])
        cam_rot_raw = camera_params.get("camera_rotation", [0.0, 0.0, 0.0, 1.0])

        # Normalise to plain lists for JSON serialisation safety
        camera_position = [float(v) for v in cam_pos_raw]
        camera_rotation = [float(v) for v in cam_rot_raw]

        # --- Stereo reconstruction ---
        logger.info(
            f"[{robot_id}] Reconstructing point cloud from stereo pair "
            f"(fov={fov}°, baseline={baseline}m)"
        )
        t0 = time.time()
        point_cloud = stereo_reconstruct_stream(
            img_left,
            img_right,
            fov=fov,
            cam_dist=baseline,
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
        n_raw = raw_points.shape[0]
        if n_raw > max_points:
            idx = np.random.choice(n_raw, size=max_points, replace=False)
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
        "as Contact-GraspNet."
    ),
    usage_examples=[
        "generate_point_cloud('Robot1')",
        "generate_point_cloud('Robot1', max_points=30000, max_age_seconds=5.0)",
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
            description="Reject images older than this many seconds (default: 2.0)",
            required=False,
            default=2.0,
        ),
    ],
    preconditions=[
        "Stereo images must have been captured and stored in UnifiedImageStorage",
        "Stereo camera rig must be active in Unity (port 5006)",
    ],
    postconditions=[
        "Returns 3D point cloud in camera frame ready for neural grasp planning",
    ],
    average_duration_ms=300.0,
    success_rate=0.85,
    failure_modes=[
        "No stereo images in storage (NO_STEREO_IMAGES)",
        "Images too old (STALE_IMAGE)",
        "Insufficient texture for disparity matching (EMPTY_POINT_CLOUD)",
    ],
    required_operations=[],
    commonly_paired_with=["detect_objects", "grasp_object"],
    mutually_exclusive_with=[],
    implementation=generate_point_cloud,
)
