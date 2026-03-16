#!/usr/bin/env python3
"""
Unit tests for Registry.py

Tests the operation registry including:
- Operation registration and lookup
- Execute operation by name/ID
- Category and complexity filtering
- Operation listing and metadata
- Concurrent access safety
- Invalid operation handling
- Performance with multiple operations
"""

import pytest
import threading
import time
from unittest.mock import Mock, MagicMock, patch

from operations.Registry import OperationRegistry, get_global_registry
from operations.Base import (
    BasicOperation,
    OperationCategory,
    OperationComplexity,
    OperationParameter,
    OperationResult,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_operation():
    """
    Create a sample BasicOperation for testing.

    Returns:
        BasicOperation instance
    """

    def test_impl(**kwargs):
        return OperationResult.success_result({"executed": True})

    op = BasicOperation(
        operation_id="test_op_001",
        name="test_operation",
        category=OperationCategory.NAVIGATION,
        complexity=OperationComplexity.BASIC,
        description="A test operation for unit tests",
        long_description="Detailed description",
        usage_examples=["test_operation(robot_id='Robot1')"],
        parameters=[
            OperationParameter(
                name="robot_id",
                type="str",
                description="Robot identifier",
                required=True,
            )
        ],
        preconditions=["Robot is initialized"],
        postconditions=["Operation completed"],
        average_duration_ms=100.0,
        success_rate=0.95,
        failure_modes=["Network error"],
        required_operations=[],
        commonly_paired_with=[],
        mutually_exclusive_with=[],
        implementation=test_impl,
    )
    return op


@pytest.fixture
def clean_registry():
    """
    Create a fresh registry for testing.

    Returns:
        OperationRegistry instance
    """
    # Reset global registry
    import operations.Registry as registry_module

    registry_module._global_registry = None

    registry = OperationRegistry()
    return registry


# ============================================================================
# Test Class: Registration & Lookup
# ============================================================================


class TestRegistryRegistration:
    """Test operation registration and lookup."""

    def test_registry_initializes_with_operations(self):
        """Test registry initializes with default operations."""
        registry = OperationRegistry()

        ops = registry.get_all_operations()
        assert len(ops) > 0
        # Should have at least move, gripper, status operations
        op_names = [op.name for op in ops]
        assert "move_to_coordinate" in op_names
        assert "control_gripper" in op_names

    def test_get_operation_by_id(self):
        """Test retrieving operation by ID."""
        registry = OperationRegistry()

        op = registry.get_operation("motion_move_to_coord_001")

        assert op is not None
        assert op.name == "move_to_coordinate"

    def test_get_operation_by_name(self):
        """Test retrieving operation by name."""
        registry = OperationRegistry()

        op = registry.get_operation_by_name("move_to_coordinate")

        assert op is not None
        assert op.operation_id == "motion_move_to_coord_001"

    def test_get_operation_by_name_case_insensitive(self):
        """Test retrieving operation by name is case-insensitive."""
        registry = OperationRegistry()

        op1 = registry.get_operation_by_name("move_to_coordinate")
        op2 = registry.get_operation_by_name("MOVE_TO_COORDINATE")
        op3 = registry.get_operation_by_name("Move_To_Coordinate")

        assert op1 is not None
        assert op1 is op2
        assert op1 is op3

    def test_get_nonexistent_operation_by_id(self):
        """Test retrieving non-existent operation by ID returns None."""
        registry = OperationRegistry()

        op = registry.get_operation("nonexistent_op_999")

        assert op is None

    def test_get_nonexistent_operation_by_name(self):
        """Test retrieving non-existent operation by name returns None."""
        registry = OperationRegistry()

        op = registry.get_operation_by_name("nonexistent_operation")

        assert op is None

    def test_list_all_operations(self):
        """Test listing all operations."""
        registry = OperationRegistry()

        ops = registry.get_all_operations()

        assert isinstance(ops, list)
        assert len(ops) > 0
        assert all(isinstance(op, BasicOperation) for op in ops)


# ============================================================================
# Test Class: Execution
# ============================================================================


class TestRegistryExecution:
    """Test operation execution through registry."""

    def test_execute_operation_by_id_success(self, clean_registry, sample_operation):
        """Test executing operation by ID with valid parameters."""
        clean_registry.operations[sample_operation.operation_id] = sample_operation

        result = clean_registry.execute_operation(
            sample_operation.operation_id, robot_id="Robot1"
        )

        assert result.success is True
        assert result.result["executed"] is True

    def test_execute_operation_by_name_success(self, clean_registry, sample_operation):
        """Test executing operation by name with valid parameters."""
        clean_registry.operations[sample_operation.operation_id] = sample_operation

        result = clean_registry.execute_operation_by_name(
            "test_operation", robot_id="Robot1"
        )

        assert result.success is True

    def test_execute_nonexistent_operation_by_id(self):
        """Test executing non-existent operation by ID returns error."""
        registry = OperationRegistry()

        result = registry.execute_operation("nonexistent_op", robot_id="Robot1")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "OPERATION_NOT_FOUND"

    def test_execute_nonexistent_operation_by_name(self):
        """Test executing non-existent operation by name returns error."""
        registry = OperationRegistry()

        result = registry.execute_operation_by_name("nonexistent", robot_id="Robot1")

        assert result.success is False
        assert result.error is not None
        assert result.error["code"] == "OPERATION_NOT_FOUND"


# ============================================================================
# Test Class: Filtering
# ============================================================================


class TestRegistryFiltering:
    """Test operation filtering by category and complexity."""

    def test_get_operations_by_category(self):
        """Test retrieving operations by category."""
        registry = OperationRegistry()

        nav_ops = registry.get_operations_by_category(OperationCategory.NAVIGATION)

        assert len(nav_ops) > 0
        assert all(op.category == OperationCategory.NAVIGATION for op in nav_ops)

    def test_get_operations_by_complexity(self):
        """Test retrieving operations by complexity."""
        registry = OperationRegistry()

        basic_ops = registry.get_operations_by_complexity(OperationComplexity.BASIC)

        assert len(basic_ops) > 0
        assert all(op.complexity == OperationComplexity.BASIC for op in basic_ops)

    def test_get_operations_by_multiple_filters(self, clean_registry, sample_operation):
        """Test filtering by both category and complexity."""
        clean_registry.operations[sample_operation.operation_id] = sample_operation

        # Get navigation operations
        nav_ops = clean_registry.get_operations_by_category(
            OperationCategory.NAVIGATION
        )

        # Further filter by basic complexity
        basic_nav_ops = [
            op for op in nav_ops if op.complexity == OperationComplexity.BASIC
        ]

        assert len(basic_nav_ops) > 0
        assert all(op.category == OperationCategory.NAVIGATION for op in basic_nav_ops)
        assert all(op.complexity == OperationComplexity.BASIC for op in basic_nav_ops)


# ============================================================================
# Test Class: Concurrency
# ============================================================================


class TestRegistryConcurrency:
    """Test concurrent access to registry."""

    def test_concurrent_operation_execution(self, clean_registry, sample_operation):
        """Test executing operations concurrently from multiple threads."""
        clean_registry.operations[sample_operation.operation_id] = sample_operation

        results = []
        errors = []

        def execute_op():
            try:
                result = clean_registry.execute_operation_by_name(
                    "test_operation", robot_id="Robot1"
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=execute_op) for _ in range(10)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all to complete
        for t in threads:
            t.join(timeout=5.0)

        # Verify all succeeded
        assert len(errors) == 0
        assert len(results) == 10
        assert all(r.success for r in results)

    def test_thread_safe_registry_access(self):
        """Test thread-safe access to registry operations."""
        registry = OperationRegistry()

        operations = []
        errors = []

        def get_ops():
            try:
                ops = registry.get_all_operations()
                operations.append(len(ops))
            except Exception as e:
                errors.append(e)

        # Create multiple reader threads
        threads = [threading.Thread(target=get_ops) for _ in range(20)]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=5.0)

        # Should have no errors
        assert len(errors) == 0
        # All should see same number of operations
        assert len(set(operations)) == 1


# ============================================================================
# Test Class: Performance
# ============================================================================


class TestRegistryPerformance:
    """Test registry performance with many operations."""

    def test_lookup_performance_large_registry(self):
        """Test operation lookup performance with many operations."""
        registry = OperationRegistry()

        # Measure lookup time
        start = time.time()
        for _ in range(1000):
            op = registry.get_operation_by_name("move_to_coordinate")
            assert op is not None
        duration = time.time() - start

        # Should be very fast (< 100ms for 1000 lookups)
        assert duration < 0.1


# ============================================================================
# Test Class: Export and Documentation
# ============================================================================


class TestRegistryExport:
    """Test registry export and documentation generation."""

    def test_generate_summary(self):
        """Test generating registry summary."""
        registry = OperationRegistry()

        summary = registry.generate_summary()

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "ROBOT OPERATIONS REGISTRY" in summary
        assert "Total operations:" in summary

    def test_export_for_rag(self, clean_registry, sample_operation, tmp_path):
        """Test exporting operations for RAG system."""
        clean_registry.operations[sample_operation.operation_id] = sample_operation

        output_dir = tmp_path / "rag_test"

        # Mock to_rag_document method
        sample_operation.to_rag_document = Mock(return_value="Test RAG document")

        clean_registry.export_for_rag(str(output_dir))

        # Verify files were created
        assert output_dir.exists()
        assert (output_dir / "operations_index.json").exists()


# ============================================================================
# Test Class: Global Registry Singleton
# ============================================================================


class TestGlobalRegistry:
    """Test global registry singleton."""

    def test_get_global_registry_singleton(self):
        """Test get_global_registry returns singleton instance."""
        registry1 = get_global_registry()
        registry2 = get_global_registry()

        assert registry1 is registry2

    def test_global_registry_has_operations(self):
        """Test global registry is initialized with operations."""
        registry = get_global_registry()

        ops = registry.get_all_operations()
        assert len(ops) > 0
