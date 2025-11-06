"""
Stereo Image Reconstruction Package

This package provides tools for reconstructing 3D point clouds from stereoscopic image pairs.

Main modules:
- Reconstruct: Core stereo reconstruction functions
- FeatureMatching: Feature matching for disparity estimation
- StreamingServer: Real-time streaming server with visualization
- stereo_config: Configuration dataclasses
"""

from .StereoConfig import (
    CameraConfig,
    ReconstructionConfig,
    FeatureMatchConfig,
    ServerConfig,
    OutputConfig,
    DEFAULT_CAMERA_CONFIG,
    DEFAULT_RECONSTRUCTION_CONFIG,
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_SERVER_CONFIG,
    DEFAULT_OUTPUT_CONFIG,
)

from .Reconstruct import (
    PLY,
    load_images,
    calc_disparity,
    reconstruct,
    stereo_reconstruct,
    stereo_reconstruct_stream,
    write_ply,
)

from .FeatureMatching import (
    find_matches,
    draw_matches,
    feature_disparity,
    visualize_matches,
)

from .StreamingServer import StereoStreamingServer

__version__ = "1.0.0"
__author__ = "ACRL Project"

__all__ = [
    # Configuration
    "CameraConfig",
    "ReconstructionConfig",
    "FeatureMatchConfig",
    "ServerConfig",
    "OutputConfig",
    "DEFAULT_CAMERA_CONFIG",
    "DEFAULT_RECONSTRUCTION_CONFIG",
    "DEFAULT_FEATURE_CONFIG",
    "DEFAULT_SERVER_CONFIG",
    "DEFAULT_OUTPUT_CONFIG",
    # Core reconstruction
    "PLY",
    "load_images",
    "calc_disparity",
    "reconstruct",
    "stereo_reconstruct",
    "stereo_reconstruct_stream",
    "write_ply",
    # Feature matching
    "find_matches",
    "draw_matches",
    "feature_disparity",
    "visualize_matches",
    # Streaming
    "StereoStreamingServer",
]
