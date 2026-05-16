#!/usr/bin/env python3
"""
Test Cases for RAG Vector Store
================================

Tests for the vector storage and similarity search module.
"""

import pytest
import numpy as np
import os
import tempfile
import threading
from rag.VectorStore import VectorStore


class TestVectorStore:
    """Test vector store functionality"""

    def test_initialization(self):
        """Test vector store initialization"""
        store = VectorStore()

        assert len(store) == 0
        assert store.embedding_dimension is None
        assert len(store.operation_ids) == 0
        assert len(store.metadata) == 0

    def test_add_operation(self):
        """Test adding an operation to the store"""
        store = VectorStore()
        embedding = np.array([0.1, 0.2, 0.3])
        metadata = {"name": "test_op", "category": "navigation"}

        store.add_operation("op_001", embedding, metadata)

        assert len(store) == 1
        assert store.embedding_dimension == 3
        assert "op_001" in store.operation_ids
        assert store.metadata[0] == metadata

    def test_add_multiple_operations(self):
        """Test adding multiple operations.

        store.vectors is flushed lazily, so shape is only guaranteed after a
        flush-triggering call (search, get_operation, save). Use len(store)
        for size and trigger a flush via save before checking shape.
        """
        store = VectorStore()

        for i in range(5):
            embedding = np.array([0.1 * i, 0.2 * i, 0.3 * i])
            metadata = {"name": f"op_{i}", "category": "navigation"}
            store.add_operation(f"op_00{i}", embedding, metadata)

        assert len(store) == 5
        # Trigger flush by calling _flush_pending_vectors directly
        with store._lock:
            store._flush_pending_vectors()
        assert store.vectors.shape == (5, 3)

    def test_embedding_dimension_mismatch(self):
        """Test error when adding embeddings with different dimensions"""
        store = VectorStore()

        # Add first operation
        store.add_operation("op_001", np.array([0.1, 0.2, 0.3]), {})

        # Try to add operation with different dimension
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.add_operation("op_002", np.array([0.1, 0.2]), {})

    def test_search_basic(self):
        """Test basic similarity search"""
        store = VectorStore()

        # Add operations
        store.add_operation("op_001", np.array([1.0, 0.0, 0.0]), {"name": "op1"})
        store.add_operation("op_002", np.array([0.0, 1.0, 0.0]), {"name": "op2"})
        store.add_operation("op_003", np.array([0.9, 0.1, 0.0]), {"name": "op3"})

        # Search with query similar to op_001
        query = np.array([1.0, 0.0, 0.0])
        results = store.search(query, top_k=3)

        # Should return 2 results (op_001 and op_003)
        # op_002 is filtered out due to MIN_SIMILARITY_SCORE threshold (0.5)
        assert len(results) == 2
        assert results[0]["operation_id"] == "op_001"  # Most similar
        assert results[0]["score"] > results[1]["score"]  # Scores are sorted
        assert results[1]["operation_id"] == "op_003"  # Second most similar

    def test_search_with_min_score(self):
        """Test search with minimum score threshold"""
        store = VectorStore()

        store.add_operation("op_001", np.array([1.0, 0.0, 0.0]), {"name": "op1"})
        store.add_operation("op_002", np.array([0.0, 1.0, 0.0]), {"name": "op2"})

        query = np.array([1.0, 0.0, 0.0])
        results = store.search(query, top_k=10, min_score=0.9)

        # Only op_001 should match with high score
        assert len(results) == 1
        assert results[0]["operation_id"] == "op_001"

    def test_search_with_category_filter(self):
        """Test search with category filtering"""
        store = VectorStore()

        store.add_operation(
            "op_001", np.array([1.0, 0.0]), {"name": "op1", "category": "navigation"}
        )
        store.add_operation(
            "op_002", np.array([0.9, 0.1]), {"name": "op2", "category": "manipulation"}
        )
        store.add_operation(
            "op_003", np.array([0.8, 0.2]), {"name": "op3", "category": "navigation"}
        )

        query = np.array([1.0, 0.0])
        results = store.search(query, top_k=10, category_filter="navigation")

        assert len(results) == 2
        assert all(r["metadata"]["category"] == "navigation" for r in results)

    def test_search_empty_store(self):
        """Test search on empty store"""
        store = VectorStore()
        query = np.array([1.0, 0.0])
        results = store.search(query)

        assert results == []

    def test_get_operation(self):
        """Test retrieving operation by ID"""
        store = VectorStore()
        embedding = np.array([0.1, 0.2, 0.3])
        metadata = {"name": "test_op"}

        store.add_operation("op_001", embedding, metadata)

        op = store.get_operation("op_001")

        assert op is not None
        assert op["operation_id"] == "op_001"
        assert np.array_equal(op["embedding"], embedding)
        assert op["metadata"] == metadata

    def test_get_operation_not_found(self):
        """Test retrieving non-existent operation"""
        store = VectorStore()
        op = store.get_operation("nonexistent")

        assert op is None

    def test_save_and_load(self):
        """Test saving and loading vector store"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create and populate store
            store = VectorStore()
            store.add_operation("op_001", np.array([0.1, 0.2]), {"name": "op1"})
            store.add_operation("op_002", np.array([0.3, 0.4]), {"name": "op2"})

            # Save
            store.save(tmp_path)

            # Load into new store
            loaded_store = VectorStore.load(tmp_path)

            assert len(loaded_store) == 2
            assert loaded_store.operation_ids == ["op_001", "op_002"]
            assert loaded_store.embedding_dimension == 2
            assert np.array_equal(loaded_store.vectors, store.vectors)

        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file returns empty store"""
        store = VectorStore.load("nonexistent_file.pkl")

        assert len(store) == 0
        assert store.embedding_dimension is None

    def test_get_stats(self):
        """Test getting store statistics"""
        store = VectorStore()

        store.add_operation(
            "op_001",
            np.array([0.1, 0.2]),
            {"name": "op1", "category": "navigation", "complexity": "basic"},
        )
        store.add_operation(
            "op_002",
            np.array([0.3, 0.4]),
            {"name": "op2", "category": "manipulation", "complexity": "basic"},
        )
        store.add_operation(
            "op_003",
            np.array([0.5, 0.6]),
            {"name": "op3", "category": "navigation", "complexity": "intermediate"},
        )

        stats = store.get_stats()

        assert stats["num_operations"] == 3
        assert stats["embedding_dimension"] == 2
        assert stats["categories"]["navigation"] == 2
        assert stats["categories"]["manipulation"] == 1
        assert stats["complexities"]["basic"] == 2

    def test_clear(self):
        """Test clearing the store"""
        store = VectorStore()
        store.add_operation("op_001", np.array([0.1, 0.2]), {"name": "op1"})
        store.add_operation("op_002", np.array([0.3, 0.4]), {"name": "op2"})

        assert len(store) == 2

        store.clear()

        assert len(store) == 0
        assert store.embedding_dimension is None
        assert len(store.operation_ids) == 0

    def test_repr(self):
        """Test string representation"""
        store = VectorStore()
        store.add_operation("op_001", np.array([0.1, 0.2]), {"name": "op1"})

        repr_str = repr(store)

        assert "VectorStore" in repr_str
        assert "operations=1" in repr_str
        assert "dim=2" in repr_str

    def test_thread_safety(self):
        """Test thread safety with concurrent adds and searches"""
        store = VectorStore()
        errors = []

        def add_operations(start_idx, count):
            """Worker function to add operations"""
            try:
                for i in range(count):
                    op_id = f"op_{start_idx + i:03d}"
                    embedding = np.random.rand(10)
                    metadata = {"name": op_id, "category": "test"}
                    store.add_operation(op_id, embedding, metadata)
            except Exception as e:
                errors.append(e)

        def search_operations(count):
            """Worker function to search operations"""
            try:
                for _ in range(count):
                    query = np.random.rand(10)
                    store.search(query, top_k=5)
            except Exception as e:
                errors.append(e)

        # Create threads for concurrent add and search
        threads = []
        for i in range(5):
            t = threading.Thread(target=add_operations, args=(i * 20, 20))
            threads.append(t)
        for i in range(3):
            t = threading.Thread(target=search_operations, args=(10,))
            threads.append(t)

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Check no errors occurred
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        # Verify all operations were added
        assert len(store) == 100  # 5 threads * 20 operations each


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
