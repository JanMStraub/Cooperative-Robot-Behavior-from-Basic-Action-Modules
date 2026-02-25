"""
Configuration module for stereo image reconstruction system.

This module centralizes all configuration parameters for the stereo reconstruction pipeline,
including camera parameters, reconstruction settings, and server configuration.
"""

from dataclasses import dataclass
from typing import Optional
from config.Servers import DEFAULT_HOST, STREAMING_SERVER_PORT


@dataclass
class CameraConfig:
    """Camera calibration and setup parameters"""

    fov: Optional[float] = 60.0  # Field of view in degrees
    focal_length: Optional[float] = None  # Focal length in mm
    sensor_width: Optional[float] = None  # Sensor width in mm
    baseline: float = 0.05  # Distance between stereo cameras in meters


@dataclass
class ReconstructionConfig:
    """Stereo reconstruction algorithm parameters"""

    # SGBM parameters
    window_size: int = 7  # Larger window for high-res images (1920x1080)
    min_disparity: int = 0
    max_disparity: Optional[int] = (
        256  # Increased from 160 - must be > actual disparity (~107px detected)
    )
    uniqueness_ratio: int = 1  # More lenient for textured scenes (was 5)
    speckle_window_size: int = 200  # Larger for high-res images
    speckle_range: int = 2
    disp12_max_diff: int = -1  # Disable check (-1 = no check, was 1)

    # Smoothness parameters - reduced for high-res textured scenes
    p1_multiplier: int = 4  # Reduced from 8 for less smoothing
    p2_multiplier: int = 16  # Reduced from 32 for less smoothing

    # Filtering
    mask_edges: bool = False
    edge_kernel_size: int = 5
    edge_percentile: float = 90.0

    # Point cloud filtering
    min_depth_threshold: float = 0.0


@dataclass
class FeatureMatchConfig:
    """Feature matching parameters for ORB"""

    n_features: int = 5000
    scale_factor: float = 1.1
    n_levels: int = 2
    edge_threshold: int = 10
    first_level: int = 0
    wta_k: int = 2
    patch_size: int = 63
    fast_threshold: int = 0

    # FLANN parameters
    flann_algorithm: int = 6  # LSH for ORB
    flann_table_number: int = 4
    flann_key_size: int = 12
    flann_multi_probe_level: int = 2

    # Matching parameters
    lowe_ratio: float = 0.5  # Lowe's ratio test threshold
    max_y_diff: float = 0.5  # Max vertical disparity
    match_keep_ratio: float = 0.7  # Keep top 70% of matches


@dataclass
class ServerConfig:
    """Streaming server configuration"""

    host: str = DEFAULT_HOST
    port: int = STREAMING_SERVER_PORT
    max_connections: int = 5
    timeout: float = 1.0

    # Visualization
    window_width: int = 1000
    window_height: int = 1000
    voxel_downsample_size: float = 0.02
    update_rate: float = 0.1  # Seconds between visualization updates


@dataclass
class OutputConfig:
    """Output file configuration"""

    output_base_dir: str = "./output"
    point_cloud_dir: str = "point_clouds"
    disparity_dir: str = "disparity_maps"
    save_disparity: bool = True
    save_point_cloud: bool = True


@dataclass
class SGBMPreset:
    """
    SGBM parameter preset optimized for specific depth range.

    Presets are designed for 5cm baseline stereo camera configuration:
    - CLOSE: <1m (high disparity, small window, strict matching)
    - MEDIUM: 0.5-2m (balanced, current default)
    - FAR: >2m (lower disparity, larger window)
    """

    name: str
    min_range: float  # meters
    max_range: float  # meters
    max_disparity: int
    window_size: int
    uniqueness_ratio: int
    p1_multiplier: int
    p2_multiplier: int
    min_disparity: int = 0
    speckle_window_size: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1


# SGBM Presets for different depth ranges
SGBM_CLOSE = SGBMPreset(
    name="close",
    min_range=0.2,
    max_range=1.0,
    max_disparity=256,  # Higher max for close objects (high disparity)
    window_size=3,  # Smaller window for detail preservation
    uniqueness_ratio=10,  # Stricter matching for accuracy
    p1_multiplier=8,
    p2_multiplier=32,
)

SGBM_MEDIUM = SGBMPreset(
    name="medium",
    min_range=0.5,
    max_range=2.0,
    max_disparity=160,  # Current default
    window_size=5,
    uniqueness_ratio=5,
    p1_multiplier=8,
    p2_multiplier=32,
)

SGBM_FAR = SGBMPreset(
    name="far",
    min_range=2.0,
    max_range=10.0,
    max_disparity=96,  # Lower max for distant objects (low disparity)
    window_size=7,  # Larger window for robustness
    uniqueness_ratio=5,
    p1_multiplier=8,
    p2_multiplier=32,
)

# Preset dictionary for easy lookup
SGBM_PRESETS = {"close": SGBM_CLOSE, "medium": SGBM_MEDIUM, "far": SGBM_FAR}


# Default configurations
DEFAULT_CAMERA_CONFIG = CameraConfig()
DEFAULT_RECONSTRUCTION_CONFIG = ReconstructionConfig()
DEFAULT_FEATURE_CONFIG = FeatureMatchConfig()
DEFAULT_SERVER_CONFIG = ServerConfig()
DEFAULT_OUTPUT_CONFIG = OutputConfig()
