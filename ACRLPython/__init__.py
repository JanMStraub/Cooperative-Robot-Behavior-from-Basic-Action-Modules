#!/usr/bin/env python3
"""
ACRLPython - Unity ↔ Python communication package for robot vision and AI

This package provides TCP servers, vision processing modules, and orchestrators
for integrating LLM vision analysis and object detection with Unity simulations.

Subpackages:
- core: Base classes and wire protocol (TCPServerBase, UnityProtocol)
- servers: TCP servers for Unity communication (ImageServer, CommandServer, SequenceServer)
- vision: Vision and AI processing (AnalyzeImage with Ollama, ObjectDetector, DepthEstimator)
- orchestrators: Main entry point (RunRobotController)
- operations: Registered operations for command execution
- rag: Semantic search for operation matching

Configuration:
- config/: Modular configuration (Servers.py, Vision.py, Rag.py, Robot.py)

Architecture (December 2025):
- Unified backend via RunRobotController
- 3 active servers: ImageServer (5005/5006), CommandServer (5010), SequenceServer (5013)
- Operations system with 17 registered operations
- Protocol V2 with request ID correlation
"""

__version__ = "2.0.0"
__author__ = "Jan M. Straub"

# Core exports
from . import config
from .core import TCPServerBase, ServerConfig, UnityProtocol

# Server exports (December 2025 - Unified Architecture)
from .servers import (
    UnifiedImageStorage,
    ImageServer,
    run_image_server_background,
    CommandBroadcaster,
    CommandServer,
    run_command_server_background,
    SequenceQueryHandler,
    SequenceServer,
    run_sequence_server_background,
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
    # Servers (Unified Architecture)
    "UnifiedImageStorage",
    "ImageServer",
    "run_image_server_background",
    "CommandBroadcaster",
    "CommandServer",
    "run_command_server_background",
    "SequenceQueryHandler",
    "SequenceServer",
    "run_sequence_server_background",
    # Vision
    "LMStudioVisionProcessor",
    "get_images_from_server",
    "save_response",
    "DetectionObject",
    "DetectionResult",
    "CubeDetector",
]
