"""
TCP Servers for Unity ↔ Python communication

This package contains all TCP server implementations that handle
network communication between Unity and Python.

Active Servers (December 2025 Architecture):
- ImageServer: Unified image receiver for single/stereo cameras (ports 5005, 5006)
- CommandServer: Bidirectional commands and results (port 5010)
- SequenceServer: Multi-command sequence execution (port 5013)

Legacy servers (DetectionServer, StreamingServer, StereoDetectionServer,
ResultsServer, RAGServer) have been consolidated into the above three servers.
"""

from .ImageServer import (
    UnifiedImageStorage,
    ImageServer,
    run_image_server_background,
)
from .CommandServer import (
    CommandBroadcaster,
    CommandServer,
    run_command_server_background,
)
from .SequenceServer import (
    SequenceQueryHandler,
    SequenceServer,
    run_sequence_server_background,
)

__all__ = [
    # ImageServer (replaces StreamingServer + StereoDetectionServer)
    "UnifiedImageStorage",
    "ImageServer",
    "run_image_server_background",
    # CommandServer (replaces ResultsServer + DetectionServer)
    "CommandBroadcaster",
    "CommandServer",
    "run_command_server_background",
    # SequenceServer (integrates RAG functionality)
    "SequenceQueryHandler",
    "SequenceServer",
    "run_sequence_server_background",
]
