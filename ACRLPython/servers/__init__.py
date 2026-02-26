"""
TCP Servers for Unity ↔ Python communication

This package contains all TCP server implementations that handle
network communication between Unity and Python.

Active Servers:
- ImageServer: Unified image receiver for single/stereo cameras (ports 5005, 5006)
- CommandServer: Bidirectional commands and results (port 5010)
- SequenceServer: Multi-command sequence execution (port 5013)
- WorldStateServer: One-way robot/object state stream from Unity (port 5014)

Legacy servers (DetectionServer, StreamingServer, StereoDetectionServer,
ResultsServer, RAGServer) have been consolidated into the above servers.

Module Architecture:
- ImageStorageCore: Core image storage singleton (no server dependencies)
- ImageServer: TCP server that uses ImageStorageCore
"""

# Import storage singleton from core module (avoids circular dependencies)
from .ImageStorageCore import UnifiedImageStorage

# Import server classes
from .ImageServer import (
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
from .WorldStateServer import WorldStateServer

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
    # WorldStateServer (one-way state stream from Unity)
    "WorldStateServer",
]
