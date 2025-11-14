"""
Computer Vision and AI Processing Modules

This package contains modules for vision-based processing including
LLM analysis, object detection, and depth estimation.

Modules:
- AnalyzeImage: LM Studio LLM vision processing
- ObjectDetector: Color-based cube detection
- DepthEstimator: Stereo depth estimation and 3D coordinate conversion (integrated disparity calculation)
- StereoConfig: Configuration classes for camera calibration and stereo reconstruction
"""

from .AnalyzeImage import LMStudioVisionProcessor, get_images_from_server, save_response
from .ObjectDetector import DetectionObject, DetectionResult, CubeDetector

# DepthEstimator now has integrated disparity calculation (no external dependencies)
from .DepthEstimator import (
    calc_disparity,
    estimate_depth_at_point,
    estimate_depth_from_disparity,
    estimate_object_world_position,
    estimate_object_world_position_from_disparity,
)

__all__ = [
    # AnalyzeImage
    "LMStudioVisionProcessor",
    "get_images_from_server",
    "save_response",
    # ObjectDetector
    "DetectionObject",
    "DetectionResult",
    "CubeDetector",
    # DepthEstimator
    "calc_disparity",
    "estimate_depth_at_point",
    "estimate_depth_from_disparity",
    "estimate_object_world_position",
    "estimate_object_world_position_from_disparity",
]
