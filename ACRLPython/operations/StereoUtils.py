#!/usr/bin/env python3
"""
Stereo Camera Utilities
=======================

Shared helpers for extracting camera calibration and pose from the stereo
metadata dict that Unity sends alongside each image pair.

All vision operations (VisionOperations, FieldOperations, PointCloudOperations)
receive the same flat metadata structure from StereoCameraController::StereoMetadata:

    {"baseline": float, "fov": float,
     "camera_position": [x, y, z], "camera_rotation": [x, y, z, w]}

Use ``camera_config_from_metadata`` to turn that dict into a typed
``CameraConfig`` and extract the extrinsics in one place.
"""

from dataclasses import dataclass
from typing import List, Optional

try:
    from config.Vision import (
        DEFAULT_STEREO_BASELINE,
        DEFAULT_STEREO_FOV,
        DEFAULT_STEREO_CAMERA_POSITION,
        DEFAULT_STEREO_CAMERA_ROTATION,
    )
except ImportError:
    from ..config.Vision import (
        DEFAULT_STEREO_BASELINE,
        DEFAULT_STEREO_FOV,
        DEFAULT_STEREO_CAMERA_POSITION,
        DEFAULT_STEREO_CAMERA_ROTATION,
    )

try:
    from vision.StereoConfig import CameraConfig
except ImportError:
    from ..vision.StereoConfig import CameraConfig


@dataclass
class StereoParams:
    """Typed camera parameters extracted from Unity stereo metadata."""

    camera_config: CameraConfig
    camera_position: Optional[List[float]]
    camera_rotation: Optional[List[float]]


def camera_config_from_metadata(
    metadata: Optional[dict],
    baseline: Optional[float] = None,
    fov: Optional[float] = None,
    camera_position: Optional[List[float]] = None,
    camera_rotation: Optional[List[float]] = None,
) -> StereoParams:
    """
    Extract a typed ``CameraConfig`` and camera pose from a Unity stereo metadata dict.

    Unity's ``StereoCameraController`` sends a flat JSON object alongside each
    stereo image pair containing ``baseline``, ``fov``, ``camera_position``, and
    ``camera_rotation``.  This function converts that dict into the typed objects
    expected by ``YOLODetector.detect_objects_stereo`` and
    ``estimate_object_world_position_from_disparity``, falling back to config
    defaults when values are absent (e.g. legacy clients).

    Args:
        metadata: Raw metadata dict from ``UnifiedImageStorage.get_latest_stereo_image``,
                  or ``None`` if unavailable.

    Returns:
        ``StereoParams`` with ``camera_config``, ``camera_position``, ``camera_rotation``.

    Args:
        metadata: Raw metadata dict from ``UnifiedImageStorage.get_latest_stereo_image``,
                  or ``None`` if unavailable.
        baseline: Caller-supplied baseline override (used when metadata is absent).
        fov: Caller-supplied FOV override (used when metadata is absent).
        camera_position: Caller-supplied position override (used when metadata is absent).
        camera_rotation: Caller-supplied rotation override (used when metadata is absent).

    Example:
        stereo_params = camera_config_from_metadata(stereo_metadata)
        detections = detector.detect_objects_stereo(
            imgL, imgR,
            camera_config=stereo_params.camera_config,
            camera_position=stereo_params.camera_position,
            camera_rotation=stereo_params.camera_rotation,
        )
    """
    meta = metadata or {}

    # Priority: Unity metadata > caller-supplied value > config default
    baseline = (
        float(meta["baseline"]) if meta.get("baseline") is not None
        else (baseline if baseline is not None else DEFAULT_STEREO_BASELINE)
    )
    fov = (
        float(meta["fov"]) if meta.get("fov") is not None
        else (fov if fov is not None else DEFAULT_STEREO_FOV)
    )
    camera_position = (
        meta.get("camera_position")
        or camera_position
        or DEFAULT_STEREO_CAMERA_POSITION
    )
    camera_rotation = (
        meta.get("camera_rotation")
        or camera_rotation
        or DEFAULT_STEREO_CAMERA_ROTATION
    )

    return StereoParams(
        camera_config=CameraConfig(baseline=baseline, fov=fov),
        camera_position=camera_position,
        camera_rotation=camera_rotation,
    )
