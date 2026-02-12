"""
Operation Registry for RAG System
==================================

This module provides a central registry of all available robot operations
that can be queried by a RAG system to enable LLM-driven robot control.
"""

from typing import Dict, List, Optional
import json
import os
import threading
from .Base import BasicOperation, OperationCategory, OperationComplexity

from .MoveOperations import (
    MOVE_TO_COORDINATE_OPERATION,
    MOVE_FROM_A_TO_B_OPERATION,
    ADJUST_END_EFFECTOR_ORIENTATION_OPERATION,
)
from .StatusOperations import CHECK_ROBOT_STATUS_OPERATION
from .GripperOperations import (
    CONTROL_GRIPPER_OPERATION,
    RELEASE_OBJECT_OPERATION,
)
from .GraspOperations import GRASP_OBJECT_OPERATION
from .DefaultPositionOperation import RETURN_TO_START_POSITION_OPERATION
from .DetectionOperations import (
    DETECT_OBJECTS_OPERATION,
    ESTIMATE_DISTANCE_TO_OBJECT_OPERATION,
    ESTIMATE_DISTANCE_BETWEEN_OBJECTS_OPERATION,
)
from .VisionOperations import ANALYZE_SCENE_OPERATION, DETECT_OBJECT_STEREO_OPERATION
from .SpatialOperations import (
    MOVE_RELATIVE_TO_OBJECT_OPERATION,
    MOVE_BETWEEN_OBJECTS_OPERATION,
    MOVE_TO_REGION_OPERATION,
)
from .SyncOperations import (
    SIGNAL_OPERATION,
    WAIT_FOR_SIGNAL_OPERATION,
    WAIT_OPERATION,
)
from .FieldOperations import (
    DETECT_FIELD_OPERATION,
    GET_FIELD_CENTER_OPERATION,
    DETECT_ALL_FIELDS_OPERATION,
)
from .IntermediateOperations import (
    GRIP_OBJECT_OPERATION,
    ALIGN_OBJECT_OPERATION,
    FOLLOW_PATH_OPERATION,
    DRAW_WITH_PEN_OPERATION,
)
from .CoordinationOperations import (
    DETECT_OTHER_ROBOT_OPERATION,
    MIRROR_MOVEMENT_OPERATION,
    # HAND_OVER_OBJECT_OPERATION removed - non-atomic (see WorkflowPatterns.py)
)
from .CollaborativeOperations import (
    STABILIZE_OBJECT_OPERATION,
    # STABILIZE_AND_MANIPULATE_OPERATION removed - non-atomic (see WorkflowPatterns.py)
)

class OperationRegistry:
    """
    Central registry of all operations that can be queried by the RAG system.

    This registry maintains all available operations and provides methods to:
    - Retrieve operations by ID, category, or complexity
    - Execute operations with parameter validation
    - Export operations for RAG ingestion
    - Generate operation documentation
    """

    def __init__(self):
        """Initialize the registry with all available operations"""
        self.operations: Dict[str, BasicOperation] = {}
        self._initialize_operations()

    def _initialize_operations(self):
        """Load all available operations into the registry"""
        operations = [
            # ============================================================================
            # LEVEL 1-2: BASIC OPERATIONS (Atomic actions)
            # ============================================================================

            # Navigation & Motion
            MOVE_TO_COORDINATE_OPERATION,
            MOVE_FROM_A_TO_B_OPERATION,
            ADJUST_END_EFFECTOR_ORIENTATION_OPERATION,
            RETURN_TO_START_POSITION_OPERATION,

            # Gripper Control
            CONTROL_GRIPPER_OPERATION,
            RELEASE_OBJECT_OPERATION,

            # Perception & Detection
            DETECT_OBJECTS_OPERATION,
            DETECT_OBJECT_STEREO_OPERATION,
            ANALYZE_SCENE_OPERATION,
            ESTIMATE_DISTANCE_TO_OBJECT_OPERATION,
            ESTIMATE_DISTANCE_BETWEEN_OBJECTS_OPERATION,

            # Field Detection (YOLO-based)
            DETECT_FIELD_OPERATION,
            GET_FIELD_CENTER_OPERATION,
            DETECT_ALL_FIELDS_OPERATION,

            # Status
            CHECK_ROBOT_STATUS_OPERATION,

            # Synchronization Primitives
            SIGNAL_OPERATION,
            WAIT_FOR_SIGNAL_OPERATION,
            WAIT_OPERATION,

            # ============================================================================
            # LEVEL 3: INTERMEDIATE OPERATIONS (Complex single-robot tasks)
            # ============================================================================

            # Advanced Manipulation
            GRASP_OBJECT_OPERATION,
            GRIP_OBJECT_OPERATION,
            ALIGN_OBJECT_OPERATION,
            DRAW_WITH_PEN_OPERATION,

            # Spatial Reasoning & Navigation
            MOVE_RELATIVE_TO_OBJECT_OPERATION,
            MOVE_BETWEEN_OBJECTS_OPERATION,
            MOVE_TO_REGION_OPERATION,
            FOLLOW_PATH_OPERATION,

            # ============================================================================
            # LEVEL 4: MULTI-ROBOT COORDINATION (Inter-robot operations)
            # ============================================================================

            DETECT_OTHER_ROBOT_OPERATION,
            MIRROR_MOVEMENT_OPERATION,
            # HAND_OVER_OBJECT_OPERATION removed - use WorkflowPatterns.HANDOFF_PATTERN

            # ============================================================================
            # LEVEL 5: COLLABORATIVE MANIPULATION (Advanced coordination)
            # ============================================================================

            STABILIZE_OBJECT_OPERATION,
            # STABILIZE_AND_MANIPULATE_OPERATION removed - use WorkflowPatterns.STABILIZE_MANIPULATE_PATTERN
        ]

        for op in operations:
            self.operations[op.operation_id] = op

        # Backward compatibility aliases - map old operation IDs to unified operation
        # This allows old code to continue working without changes
        self.operations["perception_detect_object_001"] = DETECT_OBJECT_STEREO_OPERATION
        self.operations["perception_detect_depth_001"] = DETECT_OBJECT_STEREO_OPERATION
        self.operations["perception_depth_detect_001"] = DETECT_OBJECT_STEREO_OPERATION

        # Load generated operations (if dynamic operations feature is enabled)
        self._load_generated_operations()

    def _load_generated_operations(self):
        """Load dynamically generated operations into the registry."""
        try:
            from operations.generated import load_generated_operations

            loaded = load_generated_operations()
            for op_name, file_path in loaded:
                # The operation is loaded by the loader; register it here
                from operations.generated import _load_operation_from_file
                from pathlib import Path

                operation = _load_operation_from_file(Path(file_path))
                if operation and operation.operation_id not in self.operations:
                    self.operations[operation.operation_id] = operation
        except Exception:
            # Silently skip if dynamic operations not available
            pass

    def get_operation(self, operation_id: str) -> Optional[BasicOperation]:
        """
        Retrieve specific operation by ID.

        Args:
            operation_id: The unique operation identifier

        Returns:
            BasicOperation if found, None otherwise
        """
        return self.operations.get(operation_id)

    def get_operation_by_name(self, name: str) -> Optional[BasicOperation]:
        """
        Retrieve operation by name (case-insensitive).

        Args:
            name: The operation name (e.g., "move_to_coordinate")

        Returns:
            BasicOperation if found, None otherwise
        """
        for op in self.operations.values():
            if op.name.lower() == name.lower():
                return op
        return None

    def get_all_operations(self) -> List[BasicOperation]:
        """Get all available operations"""
        return list(self.operations.values())

    def get_operations_by_category(self, category: OperationCategory) -> List[BasicOperation]:
        """
        Get operations in a specific category.

        Args:
            category: The operation category (e.g., OperationCategory.NAVIGATION)

        Returns:
            List of operations in that category
        """
        return [op for op in self.operations.values() if op.category == category]

    def get_operations_by_complexity(self, complexity: OperationComplexity) -> List[BasicOperation]:
        """
        Get operations at a specific complexity level.

        Args:
            complexity: The complexity level (e.g., OperationComplexity.BASIC)

        Returns:
            List of operations at that complexity
        """
        return [op for op in self.operations.values() if op.complexity == complexity]

    def execute_operation(self, operation_id: str, **kwargs):
        """
        Execute an operation by ID with given parameters.

        Args:
            operation_id: The operation to execute
            **kwargs: Operation parameters

        Returns:
            OperationResult with success status and data

        Example:
            >>> registry = OperationRegistry()
            >>> result = registry.execute_operation(
            ...     "motion_move_to_coord_001",
            ...     robot_id="Robot1",
            ...     x=0.3, y=0.15, z=0.1
            ... )
            >>> if result.success:
            ...     print("Operation succeeded!")
        """
        operation = self.get_operation(operation_id)
        if operation is None:
            from .Base import OperationResult
            return OperationResult.error_result(
                error_code="OPERATION_NOT_FOUND",
                message=f"Operation '{operation_id}' not found in registry",
                recovery_suggestions=[
                    "Check operation ID spelling",
                    "List available operations with get_all_operations()",
                    "Verify operation has been added to registry"
                ]
            )

        return operation.execute(**kwargs)

    def execute_operation_by_name(self, name: str, **kwargs):
        """
        Execute an operation by name with given parameters.

        Args:
            name: The operation name (e.g., "move_to_coordinate")
            **kwargs: Operation parameters

        Returns:
            OperationResult with success status and data
        """
        operation = self.get_operation_by_name(name)
        if operation is None:
            from .Base import OperationResult
            return OperationResult.error_result(
                error_code="OPERATION_NOT_FOUND",
                message=f"Operation '{name}' not found in registry",
                recovery_suggestions=[
                    "Check operation name spelling",
                    "List available operations with get_all_operations()",
                    "Verify operation has been added to registry"
                ]
            )

        return operation.execute(**kwargs)

    def export_for_rag(self, output_dir: str = "./rag_documents"):
        """
        Export all operations as rich text documents for RAG ingestion.

        This creates individual text files for each operation, optimized for
        semantic search and RAG retrieval. Also creates a master index JSON.

        Args:
            output_dir: Directory to write the documents to

        Example:
            >>> registry = OperationRegistry()
            >>> registry.export_for_rag("./robot_operations")
            Exported 1 operations to ./robot_operations
        """
        os.makedirs(output_dir, exist_ok=True)

        # Export each operation as a text document
        for op in self.operations.values():
            filename = f"{output_dir}/{op.operation_id}.txt"
            with open(filename, 'w') as f:
                f.write(op.to_rag_document())

        # Create a master index for quick reference
        with open(f"{output_dir}/operations_index.json", 'w') as f:
            index = {
                op_id: {
                    "name": op.name,
                    "category": op.category.value,
                    "complexity": op.complexity.value,
                    "description": op.description
                }
                for op_id, op in self.operations.items()
            }
            json.dump(index, f, indent=2)

        print(f"Exported {len(self.operations)} operations to {output_dir}")

    def generate_summary(self) -> str:
        """
        Generate a summary of all available operations.

        Returns:
            String with formatted summary
        """
        summary = []
        summary.append("=" * 70)
        summary.append("ROBOT OPERATIONS REGISTRY")
        summary.append("=" * 70)
        summary.append(f"\nTotal operations: {len(self.operations)}\n")

        # Group by category
        summary.append("Operations by Category:")
        for category in OperationCategory:
            ops = self.get_operations_by_category(category)
            if ops:
                summary.append(f"  {category.value}: {len(ops)} operations")
                for op in ops:
                    summary.append(f"    - {op.name} (ID: {op.operation_id})")

        # Group by complexity
        summary.append("\nOperations by Complexity:")
        for complexity in OperationComplexity:
            ops = self.get_operations_by_complexity(complexity)
            if ops:
                summary.append(f"  {complexity.value}: {len(ops)} operations")

        return "\n".join(summary)


# Global registry instance
_global_registry: Optional[OperationRegistry] = None
_registry_lock = threading.RLock()


def get_global_registry() -> OperationRegistry:
    """
    Get the global operation registry singleton (thread-safe).

    Returns:
        The global OperationRegistry instance

    Example:
        >>> from LLMCommunication.operations.registry import get_global_registry
        >>> registry = get_global_registry()
        >>> result = registry.execute_operation_by_name("move_to_coordinate", ...)
    """
    global _global_registry
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = OperationRegistry()
    return _global_registry
