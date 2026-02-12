"""
Configuration Module
====================

Modular configuration system for ACRL Python backend.

Structure:
- servers.py: Network, ports, and server configuration
- vision.py: Vision, detection, and YOLO configuration
- rag.py: RAG system configuration
- robot.py: Multi-robot workspace and coordination configuration
- DynamicOperations.py: Dynamic operation generation configuration
- validation.py: Configuration validation utilities

All configs are aggregated in LLMConfig.py for backward compatibility.
"""

from .Servers import *
from .Vision import *
from .Rag import *
from .Robot import *
from .Validation import validate_config

__all__ = [
    # Re-export validation
    "validate_config",
]
