#!/usr/bin/env python3
"""Stereo image reconstruction configuration."""

from dataclasses import dataclass
from typing import Optional

try:
    from config.Servers import DEFAULT_HOST, STEREO_DETECTION_PORT
except ImportError:
    from ..config.Servers import DEFAULT_HOST, STEREO_DETECTION_PORT


@dataclass
class CameraConfig:
    """Camera calibration parameters."""

    fov: Optional[float] = 60.0
    focal_length: Optional[float] = None
    sensor_width: Optional[float] = None
    baseline: float = 0.05


@dataclass
class ReconstructionConfig:
    """Stereo reconstruction algorithm parameters."""

    window_size: int = 7
    min_disparity: int = 0
    max_disparity: Optional[int] = 256
    uniqueness_ratio: int = 1
    speckle_window_size: int = 200
    speckle_range: int = 2
    disp12_max_diff: int = -1
    p1_multiplier: int = 4
    p2_multiplier: int = 16
    mask_edges: bool = False
    edge_kernel_size: int = 5
    edge_percentile: float = 90.0
    min_depth_threshold: float = 0.0


@dataclass
class FeatureMatchConfig:
    """Feature matching parameters for ORB."""

    n_features: int = 5000
    scale_factor: float = 1.1
    n_levels: int = 2
    edge_threshold: int = 10
    first_level: int = 0
    wta_k: int = 2
    patch_size: int = 63
    fast_threshold: int = 0
    flann_algorithm: int = 6
    flann_table_number: int = 4
    flann_key_size: int = 12
    flann_multi_probe_level: int = 2
    lowe_ratio: float = 0.5
    max_y_diff: float = 0.5
    match_keep_ratio: float = 0.7


@dataclass
class ServerConfig:
    """Streaming server configuration."""

    host: str = DEFAULT_HOST
    port: int = STEREO_DETECTION_PORT
    max_connections: int = 5
    timeout: float = 1.0
    window_width: int = 1000
    window_height: int = 1000
    voxel_downsample_size: float = 0.02
    update_rate: float = 0.1


@dataclass
class OutputConfig:
    """Output file configuration."""

    output_base_dir: str = "./output"
    point_cloud_dir: str = "point_clouds"
    disparity_dir: str = "disparity_maps"
    save_disparity: bool = True
    save_point_cloud: bool = True


@dataclass
class SGBMPreset:
    """SGBM parameter preset for specific depth range."""

    name: str
    min_range: float
    max_range: float
    max_disparity: int
    window_size: int
    uniqueness_ratio: int
    p1_multiplier: int
    p2_multiplier: int
    min_disparity: int = 0
    speckle_window_size: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1


SGBM_CLOSE = SGBMPreset(
    name="close",
    min_range=0.2,
    max_range=1.0,
    max_disparity=256,
    window_size=3,
    uniqueness_ratio=10,
    p1_multiplier=8,
    p2_multiplier=32,
)

SGBM_MEDIUM = SGBMPreset(
    name="medium",
    min_range=0.5,
    max_range=2.0,
    max_disparity=160,
    window_size=5,
    uniqueness_ratio=5,
    p1_multiplier=8,
    p2_multiplier=32,
)

SGBM_FAR = SGBMPreset(
    name="far",
    min_range=2.0,
    max_range=10.0,
    max_disparity=96,
    window_size=7,
    uniqueness_ratio=5,
    p1_multiplier=8,
    p2_multiplier=32,
)

SGBM_PRESETS = {"close": SGBM_CLOSE, "medium": SGBM_MEDIUM, "far": SGBM_FAR}
DEFAULT_CAMERA_CONFIG = CameraConfig()
DEFAULT_RECONSTRUCTION_CONFIG = ReconstructionConfig()
DEFAULT_FEATURE_CONFIG = FeatureMatchConfig()
DEFAULT_SERVER_CONFIG = ServerConfig()
DEFAULT_OUTPUT_CONFIG = OutputConfig()
