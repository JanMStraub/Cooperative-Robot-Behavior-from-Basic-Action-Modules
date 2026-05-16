#!/usr/bin/env python3
"""Parse Unity stereo metadata dicts into typed CameraConfig + extrinsics — shared by all vision operations."""

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
    """Convert Unity stereo metadata dict to typed StereoParams; falls back to config defaults for legacy clients."""
    meta = metadata or {}

    # Unity metadata wins; caller-supplied values override config defaults
    baseline = (
        float(meta["baseline"])
        if meta.get("baseline") is not None
        else (baseline if baseline is not None else DEFAULT_STEREO_BASELINE)
    )
    fov = (
        float(meta["fov"])
        if meta.get("fov") is not None
        else (fov if fov is not None else DEFAULT_STEREO_FOV)
    )
    camera_position = (
        meta.get("camera_position") or camera_position or DEFAULT_STEREO_CAMERA_POSITION
    )
    camera_rotation = (
        meta.get("camera_rotation") or camera_rotation or DEFAULT_STEREO_CAMERA_ROTATION
    )

    return StereoParams(
        camera_config=CameraConfig(baseline=baseline, fov=fov),
        camera_position=camera_position,
        camera_rotation=camera_rotation,
    )
