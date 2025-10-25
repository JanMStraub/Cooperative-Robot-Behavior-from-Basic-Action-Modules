"""
TCP Servers for Unity ↔ Python communication

This package contains all TCP server implementations that handle
network communication between Unity and Python.

Servers:
- StreamingServer: Receives single camera images from Unity (port 5005)
- ResultsServer: Sends LLM analysis results to Unity (port 5006)
- DetectionServer: Sends object detection results to Unity (port 5007)
- StereoDetectionServer: Receives stereo image pairs from Unity (port 5009)
"""

from .StreamingServer import ImageStorage, StreamingServer, run_streaming_server_background
from .ResultsServer import ResultsBroadcaster, ResultsServer, run_results_server_background
from .DetectionServer import DetectionBroadcaster, DetectionServer, run_detection_server_background
from .StereoDetectionServer import StereoImageStorage, StereoDetectionServer, run_stereo_detection_server_background

__all__ = [
    # StreamingServer
    "ImageStorage",
    "StreamingServer",
    "run_streaming_server_background",
    # ResultsServer
    "ResultsBroadcaster",
    "ResultsServer",
    "run_results_server_background",
    # DetectionServer
    "DetectionBroadcaster",
    "DetectionServer",
    "run_detection_server_background",
    # StereoDetectionServer
    "StereoImageStorage",
    "StereoDetectionServer",
    "run_stereo_detection_server_background",
]
