#!/usr/bin/env python3
"""Central registry of all available robot operations."""

import logging
from typing import Dict, List, Optional
import json
import os
import threading
from .Base import BasicOperation, OperationCategory, OperationComplexity

logger = logging.getLogger(__name__)

from .MoveOperations import (
    MOVE_TO_COORDINATE_OPERATION,
    ADJUST_END_EFFECTOR_ORIENTATION_OPERATION,
)
from .StatusOperations import CHECK_ROBOT_STATUS_OPERATION
from .GripperOperations import (
    CONTROL_GRIPPER_OPERATION,
    RELEASE_OBJECT_OPERATION,
    PLACE_OBJECT_OPERATION,
    PLACE_BETWEEN_OBJECTS_OPERATION,
)
from .GraspOperations import (
    GRASP_OBJECT_OPERATION,
    RECEIVE_HANDOFF_OPERATION,
)
from .DefaultPositionOperation import RETURN_TO_START_POSITION_OPERATION
from .VisionOperations import ANALYZE_SCENE_OPERATION, DETECT_OBJECT_STEREO_OPERATION
from .PointCloudOperations import GENERATE_POINT_CLOUD_OPERATION
from .FieldOperations import (
    DETECT_FIELD_OPERATION,
    DETECT_ALL_FIELDS_OPERATION,
)
from .SyncOperations import (
    SIGNAL_OPERATION,
    WAIT_FOR_SIGNAL_OPERATION,
    WAIT_OPERATION,
    RESET_SIMULATION_OPERATION,
    YIELD_WORKSPACE_OPERATION,
)


class OperationRegistry:
    """Central registry of all operations queryable by RAG system."""

    def __init__(self):
        self.operations: Dict[str, BasicOperation] = {}
        self._initialize_operations()

    def _initialize_operations(self):
        operations = [
            MOVE_TO_COORDINATE_OPERATION,
            ADJUST_END_EFFECTOR_ORIENTATION_OPERATION,
            RETURN_TO_START_POSITION_OPERATION,
            CONTROL_GRIPPER_OPERATION,
            RELEASE_OBJECT_OPERATION,
            PLACE_OBJECT_OPERATION,
            PLACE_BETWEEN_OBJECTS_OPERATION,
            DETECT_OBJECT_STEREO_OPERATION,
            GENERATE_POINT_CLOUD_OPERATION,
            ANALYZE_SCENE_OPERATION,
            DETECT_FIELD_OPERATION,
            DETECT_ALL_FIELDS_OPERATION,
            CHECK_ROBOT_STATUS_OPERATION,
            SIGNAL_OPERATION,
            WAIT_FOR_SIGNAL_OPERATION,
            WAIT_OPERATION,
            RESET_SIMULATION_OPERATION,
            GRASP_OBJECT_OPERATION,
            RECEIVE_HANDOFF_OPERATION,
            YIELD_WORKSPACE_OPERATION,
        ]

        for op in operations:
            self.operations[op.operation_id] = op

    def register_operation(self, operation: BasicOperation) -> None:
        """Register new operation at runtime (thread-safe)."""
        with _registry_lock:
            self.operations[operation.operation_id] = operation

    def get_operation(self, operation_id: str) -> Optional[BasicOperation]:
        """Retrieve operation by ID."""
        return self.operations.get(operation_id)

    def get_operation_by_name(self, name: str) -> Optional[BasicOperation]:
        """Retrieve operation by name (case-insensitive)."""
        for op in self.operations.values():
            if op.name.lower() == name.lower():
                return op
        return None

    def get_all_operations(self) -> List[BasicOperation]:
        """Get all available operations"""
        return list(self.operations.values())

    def get_operations_by_category(
        self, category: OperationCategory
    ) -> List[BasicOperation]:
        """Get operations in a specific category."""
        return [op for op in self.operations.values() if op.category == category]

    def get_operations_by_complexity(
        self, complexity: OperationComplexity
    ) -> List[BasicOperation]:
        """Get operations at a specific complexity level."""
        return [op for op in self.operations.values() if op.complexity == complexity]

    def execute_operation(self, operation_id: str, **kwargs):
        """Execute operation by ID with given parameters."""
        operation = self.get_operation(operation_id)
        if operation is None:
            from .Base import OperationResult

            return OperationResult.error_result(
                error_code="OPERATION_NOT_FOUND",
                message=f"Operation '{operation_id}' not found in registry",
                recovery_suggestions=[
                    "Check operation ID spelling",
                    "List available operations with get_all_operations()",
                    "Verify operation has been added to registry",
                ],
            )

        return operation.execute(**kwargs)

    def execute_operation_by_name(self, name: str, **kwargs):
        """Execute operation by name with given parameters."""
        operation = self.get_operation_by_name(name)
        if operation is None:
            from .Base import OperationResult

            return OperationResult.error_result(
                error_code="OPERATION_NOT_FOUND",
                message=f"Operation '{name}' not found in registry",
                recovery_suggestions=[
                    "Check operation name spelling",
                    "List available operations with get_all_operations()",
                    "Verify operation has been added to registry",
                ],
            )

        return operation.execute(**kwargs)

    def export_for_rag(self, output_dir: str = "./rag_documents"):
        """Export all operations as rich text documents for RAG ingestion."""
        os.makedirs(output_dir, exist_ok=True)

        for op in self.operations.values():
            filename = f"{output_dir}/{op.operation_id}.txt"
            with open(filename, "w") as f:
                f.write(op.to_rag_document())

        with open(f"{output_dir}/operations_index.json", "w") as f:
            index = {
                op_id: {
                    "name": op.name,
                    "category": op.category.value,
                    "complexity": op.complexity.value,
                    "description": op.description,
                }
                for op_id, op in self.operations.items()
            }
            json.dump(index, f, indent=2)

        logger.info("Exported %d operations to %s", len(self.operations), output_dir)

    def generate_summary(self) -> str:
        """Generate summary of all available operations."""
        summary = []
        summary.append("=" * 60)
        summary.append("ROBOT OPERATIONS REGISTRY")
        summary.append("=" * 60)
        summary.append(f"\nTotal operations: {len(self.operations)}\n")

        summary.append("Operations by Category:")
        for category in OperationCategory:
            ops = self.get_operations_by_category(category)
            if ops:
                summary.append(f"  {category.value}: {len(ops)} operations")
                for op in ops:
                    summary.append(f"    - {op.name} (ID: {op.operation_id})")

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
    """Get global operation registry singleton (thread-safe)."""
    global _global_registry
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = OperationRegistry()
    return _global_registry
