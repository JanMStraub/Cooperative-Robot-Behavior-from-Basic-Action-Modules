"""
LLMCommunication - Unity ↔ Python communication package for robot vision and AI

This package provides TCP servers, vision processing modules, and orchestrators
for integrating LM Studio LLM vision analysis and object detection with Unity simulations.

Subpackages:
- core: Base classes and wire protocol (TCPServerBase, UnityProtocol)
- servers: TCP servers for Unity communication (StreamingServer, ResultsServer, DetectionServer, StereoDetectionServer)
- vision: Vision and AI processing (AnalyzeImage with LM Studio, ObjectDetector, DepthEstimator)
- orchestrators: Main entry point scripts (RunAnalyzer, RunDetector, RunStereoDetector)

Configuration:
- llm_config.py: Centralized configuration for all modules

For detailed usage, see RESTRUCTURING_SUMMARY.md
"""

__version__ = "2.0.0"
__author__ = "ACRL Team"

# Core exports
from . import llm_config as config
from .core import TCPServerBase, ServerConfig, UnityProtocol

# Server exports
from .servers import (
    ImageStorage,
    StreamingServer,
    run_streaming_server_background,
    ResultsBroadcaster,
    ResultsServer,
    run_results_server_background,
    DetectionBroadcaster,
    DetectionServer,
    run_detection_server_background,
    StereoImageStorage,
    StereoDetectionServer,
    run_stereo_detection_server_background,
)

# Vision exports (DepthEstimator excluded due to external dependencies)
from .vision import (
    LMStudioVisionProcessor,
    get_images_from_server,
    save_response,
    DetectionObject,
    DetectionResult,
    CubeDetector,
)

__all__ = [
    # Core
    "config",
    "TCPServerBase",
    "ServerConfig",
    "UnityProtocol",
    # Servers
    "ImageStorage",
    "StreamingServer",
    "run_streaming_server_background",
    "ResultsBroadcaster",
    "ResultsServer",
    "run_results_server_background",
    "DetectionBroadcaster",
    "DetectionServer",
    "run_detection_server_background",
    "StereoImageStorage",
    "StereoDetectionServer",
    "run_stereo_detection_server_background",
    # Vision
    "LMStudioVisionProcessor",
    "get_images_from_server",
    "save_response",
    "DetectionObject",
    "DetectionResult",
    "CubeDetector",
]
