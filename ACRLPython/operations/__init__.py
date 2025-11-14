"""
Operations Module for Robot Control
====================================

This module contains implementations of basic operations that control the robot
through Unity. Each operation sends commands to Unity via the ResultsServer.

Operations are defined with rich metadata for RAG retrieval and structured
as BasicOperation instances for LLM consumption.

Usage:
    >>> from LLMCommunication.operations import get_global_registry
    >>> registry = get_global_registry()
    >>> result = registry.execute_operation_by_name(
    ...     "move_to_coordinate",
    ...     robot_id="Robot1",
    ...     x=0.3, y=0.15, z=0.1
    ... )
"""

from .Base import (
    BasicOperation,
    OperationCategory,
    OperationComplexity,
    OperationParameter,
    OperationResult
)
from .MoveOperations import (
    move_to_coordinate,
    MOVE_TO_COORDINATE_OPERATION,
    create_move_to_coordinate_operation
)
from .Registry import (
    OperationRegistry,
    get_global_registry
)

__all__ = [
    # Base classes
    'BasicOperation',
    'OperationCategory',
    'OperationComplexity',
    'OperationParameter',
    'OperationResult',
    # Move operations
    'move_to_coordinate',
    'MOVE_TO_COORDINATE_OPERATION',
    'create_move_to_coordinate_operation',
    # Registry
    'OperationRegistry',
    'get_global_registry'
]
