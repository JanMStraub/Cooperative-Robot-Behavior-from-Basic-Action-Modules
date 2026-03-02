"""
Tests for VectorStore deferred-flush and metadata-update features
=================================================================

Covers:
- _flush_pending_vectors() consolidates staging list into self.vectors in one vstack
- add_operation() does NOT eagerly update self.vectors
- search() triggers flush before cosine similarity computation
- get_operation() triggers flush before index lookup
- save() triggers flush before pickling
- clear() resets _pending_vectors alongside other state
- update_operation_metadata() merges fields into existing metadata
- update_operation_metadata() returns False for unknown operation_id
- outcome tracking fields (execution_count etc.) survive a save/load round-trip
"""

import os
import tempfile

import numpy as np
import pytest

from rag.VectorStore import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(n: int) -> np.ndarray:
    """Return a unit vector of length n pointing along axis 0."""
    v = np.zeros(n)
    v[0] = 1.0
    return v


def _store_with_ops(n: int, dim: int = 4) -> VectorStore:
    """Return a VectorStore pre-populated with n random operations."""
    store = VectorStore()
    rng = np.random.default_rng(42)
    for i in range(n):
        emb = rng.random(dim)
        emb /= np.linalg.norm(emb)
        store.add_operation(f"op_{i:03d}", emb, {"name": f"op_{i}", "type": "test"})
    return store


# ---------------------------------------------------------------------------
# Deferred flush: internal staging behaviour
# ---------------------------------------------------------------------------

class TestDeferredFlush:
    """add_operation stages embeddings; flush materialises them in one pass."""

    def test_vectors_stay_empty_before_flush(self):
        """store.vectors must remain empty immediately after add_operation calls."""
        store = VectorStore()
        store.add_operation("op_0", _unit(4), {"name": "op_0"})
        store.add_operation("op_1", _unit(4) * 0.5, {"name": "op_1"})

        # Pending list has 2 entries, but self.vectors has not been touched yet
        assert len(store._pending_vectors) == 2
        assert store.vectors.shape == (0,)

    def test_pending_list_clears_after_flush(self):
        """_flush_pending_vectors() drains _pending_vectors."""
        store = VectorStore()
        store.add_operation("op_0", _unit(4), {"name": "op_0"})

        with store._lock:
            store._flush_pending_vectors()

        assert store._pending_vectors == []

    def test_flush_builds_correct_shape(self):
        """After flush, self.vectors has shape (n_ops, dim)."""
        store = _store_with_ops(7, dim=6)

        with store._lock:
            store._flush_pending_vectors()

        assert store.vectors.shape == (7, 6)

    def test_flush_is_idempotent(self):
        """Calling flush twice should not duplicate rows."""
        store = _store_with_ops(3, dim=4)

        with store._lock:
            store._flush_pending_vectors()
            shape_first = store.vectors.shape
            store._flush_pending_vectors()  # second call — nothing pending
            shape_second = store.vectors.shape

        assert shape_first == shape_second == (3, 4)

    def test_incremental_adds_after_initial_flush(self):
        """Adding more ops after a flush should extend vectors correctly."""
        store = _store_with_ops(3, dim=4)
        # Trigger first flush via search
        store.search(_unit(4), top_k=1)
        assert store.vectors.shape == (3, 4)

        # Add two more and trigger another flush
        store.add_operation("extra_0", _unit(4), {"name": "extra_0"})
        store.add_operation("extra_1", _unit(4), {"name": "extra_1"})
        store.search(_unit(4), top_k=1)

        assert store.vectors.shape == (5, 4)

    def test_clear_resets_pending_list(self):
        """clear() must also empty _pending_vectors."""
        store = _store_with_ops(5, dim=4)
        assert len(store._pending_vectors) > 0

        store.clear()

        assert store._pending_vectors == []
        assert len(store) == 0

    def test_flush_single_embedding_not_squeezed(self):
        """Flushing a single embedding must produce shape (1, dim), not (dim,)."""
        store = VectorStore()
        store.add_operation("solo", _unit(8), {"name": "solo"})

        with store._lock:
            store._flush_pending_vectors()

        assert store.vectors.shape == (1, 8)


# ---------------------------------------------------------------------------
# Flush triggered by public API
# ---------------------------------------------------------------------------

class TestFlushTriggeredByPublicAPI:
    """search, get_operation, and save all auto-flush before use."""

    def test_search_flushes_pending(self):
        """search() must flush pending vectors so cosine similarity sees all ops."""
        store = _store_with_ops(5, dim=4)
        # Pending vectors not yet flushed
        assert len(store._pending_vectors) == 5

        results = store.search(_unit(4), top_k=3)

        assert len(store._pending_vectors) == 0  # flushed
        assert len(results) > 0

    def test_get_operation_flushes_pending(self):
        """get_operation() must flush so the returned embedding is correct."""
        store = VectorStore()
        emb = _unit(4)
        store.add_operation("my_op", emb, {"name": "my_op"})

        result = store.get_operation("my_op")

        assert result is not None
        assert np.array_equal(result["embedding"], emb)
        assert len(store._pending_vectors) == 0

    def test_save_flushes_pending(self):
        """save() must flush so the persisted pickle contains all operations."""
        store = _store_with_ops(4, dim=3)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp = f.name
        try:
            store.save(tmp)
            loaded = VectorStore.load(tmp)
            assert len(loaded) == 4
            assert loaded.vectors.shape == (4, 3)
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# update_operation_metadata
# ---------------------------------------------------------------------------

class TestUpdateOperationMetadata:
    """Merge new fields into metadata without re-embedding."""

    def test_adds_new_outcome_fields(self):
        """update_operation_metadata merges execution stats into existing metadata."""
        store = VectorStore()
        store.add_operation("op_0", _unit(4), {"name": "move", "type": "navigation"})

        ok = store.update_operation_metadata("op_0", {
            "execution_count": 10,
            "success_count": 8,
            "failure_count": 2,
            "last_outcome": "success",
        })

        assert ok is True
        meta = store.metadata[0]
        assert meta["execution_count"] == 10
        assert meta["success_count"] == 8
        assert meta["failure_count"] == 2
        assert meta["last_outcome"] == "success"

    def test_preserves_existing_fields(self):
        """Merging new fields must not overwrite unrelated existing metadata."""
        store = VectorStore()
        store.add_operation("op_0", _unit(4), {"name": "grasp", "category": "manipulation"})

        store.update_operation_metadata("op_0", {"execution_count": 5})

        meta = store.metadata[0]
        assert meta["name"] == "grasp"
        assert meta["category"] == "manipulation"
        assert meta["execution_count"] == 5

    def test_update_can_overwrite_existing_field(self):
        """An explicit update to an existing key should overwrite it."""
        store = VectorStore()
        store.add_operation("op_0", _unit(4), {"name": "old_name"})

        store.update_operation_metadata("op_0", {"name": "new_name"})

        assert store.metadata[0]["name"] == "new_name"

    def test_returns_false_for_unknown_id(self):
        """Updating a non-existent operation_id must return False."""
        store = VectorStore()
        store.add_operation("op_0", _unit(4), {"name": "op_0"})

        result = store.update_operation_metadata("does_not_exist", {"x": 1})

        assert result is False

    def test_metadata_survives_save_load_round_trip(self):
        """Outcome metadata written via update_operation_metadata persists through pickle."""
        store = VectorStore()
        store.add_operation("op_0", _unit(4), {"name": "move"})
        store.update_operation_metadata("op_0", {
            "execution_count": 3,
            "last_outcome": "failure",
        })

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp = f.name
        try:
            store.save(tmp)
            loaded = VectorStore.load(tmp)
            meta = loaded.metadata[0]
            assert meta["execution_count"] == 3
            assert meta["last_outcome"] == "failure"
        finally:
            os.unlink(tmp)

    def test_update_works_before_flush(self):
        """update_operation_metadata should work even if vectors are still pending."""
        store = VectorStore()
        store.add_operation("op_0", _unit(4), {"name": "op_0"})
        # pending not yet flushed
        assert len(store._pending_vectors) == 1

        ok = store.update_operation_metadata("op_0", {"my_field": 42})

        assert ok is True
        assert store.metadata[0]["my_field"] == 42
        # vectors should still be pending (update_metadata doesn't flush)
        assert len(store._pending_vectors) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
