"""
Computer Vision and AI Processing Modules

This package contains modules for vision-based processing including
LLM analysis, object detection, and depth estimation.

Modules:
- AnalyzeImage: LM Studio LLM vision processing
- ObjectDetector: Color-based cube detection
- DepthEstimator: Stereo depth estimation and 3D coordinate conversion
"""

from .AnalyzeImage import LMStudioVisionProcessor, get_images_from_server, save_response
from .ObjectDetector import DetectionObject, DetectionResult, CubeDetector

# Note: DepthEstimator not imported here due to complex external dependencies (StereoImageReconstruction)
# Import directly when needed: from vision.DepthEstimator import estimate_depth_at_point

__all__ = [
    # AnalyzeImage
    "LMStudioVisionProcessor",
    "get_images_from_server",
    "save_response",
    # ObjectDetector
    "DetectionObject",
    "DetectionResult",
    "CubeDetector",
]
