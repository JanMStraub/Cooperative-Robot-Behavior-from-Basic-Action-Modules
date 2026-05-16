#!/usr/bin/env python3
"""Robot control operations with rich metadata for RAG retrieval."""

from .Base import (
    BasicOperation,
    OperationCategory,
    OperationComplexity,
    OperationParameter,
    OperationResult,
)
from .MoveOperations import (
    move_to_coordinate,
    MOVE_TO_COORDINATE_OPERATION,
    create_move_to_coordinate_operation,
)
from .Registry import OperationRegistry, get_global_registry
from . import GraspOperations

# VisionOperations are imported by Registry

__all__ = [
    # Base classes
    "BasicOperation",
    "OperationCategory",
    "OperationComplexity",
    "OperationParameter",
    "OperationResult",
    # Move operations
    "move_to_coordinate",
    "MOVE_TO_COORDINATE_OPERATION",
    "create_move_to_coordinate_operation",
    # Registry
    "OperationRegistry",
    "get_global_registry",
    # Modules (imported for 'from operations import GraspOperations' usage)
    "GraspOperations",
]
