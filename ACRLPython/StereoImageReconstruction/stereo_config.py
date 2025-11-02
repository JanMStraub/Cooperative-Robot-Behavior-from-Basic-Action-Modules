"""
Configuration module for stereo image reconstruction system.

This module centralizes all configuration parameters for the stereo reconstruction pipeline,
including camera parameters, reconstruction settings, and server configuration.
"""

from dataclasses import dataclass
from typing import Optional


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
    window_size: int = 5  # Increased from 2 for better matching
    min_disparity: int = 0
    max_disparity: Optional[int] = 160  # Set to 160 for close-range with reduced baseline (5cm baseline optimal)
    uniqueness_ratio: int = 5  # Reduced from 40 for more lenient matching
    speckle_window_size: int = 100  # Increased from 20 for better noise filtering
    speckle_range: int = 2  # Increased from 1
    disp12_max_diff: int = 1  # Reduced from 5 for stricter matching

    # Smoothness parameters
    p1_multiplier: int = 8  # Increased from 4
    p2_multiplier: int = 32

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

    host: str = "127.0.0.1"
    port: int = 5005
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


# Default configurations
DEFAULT_CAMERA_CONFIG = CameraConfig()
DEFAULT_RECONSTRUCTION_CONFIG = ReconstructionConfig()
DEFAULT_FEATURE_CONFIG = FeatureMatchConfig()
DEFAULT_SERVER_CONFIG = ServerConfig()
DEFAULT_OUTPUT_CONFIG = OutputConfig()
