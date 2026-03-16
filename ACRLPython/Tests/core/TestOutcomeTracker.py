#!/usr/bin/env python3
"""
Tests for OutcomeTracker
========================

Covers:
- Singleton: two OutcomeTracker() calls return the same object
- record(): appends outcomes to the ring buffer
- get_recent_failures(): returns only failures, most-recent-first; filters by operation_name
- get_failure_rate(): correct computation; returns 0.0 for unknown op; respects window
- get_frequent_failure_patterns(): thresholds, sort order, sample_errors, min_occurrences
- reset(): clears ring buffer
- Ring-buffer cap: evicts oldest entry when maxlen is exceeded
- VectorStore integration: update_operation_metadata called with correct counters
"""

import time
import threading
from unittest.mock import MagicMock, patch
import pytest

from orchestrators.OutcomeTracker import OutcomeTracker, get_outcome_tracker


# ---------------------------------------------------------------------------
# Fixture: reset singleton between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_outcome_tracker():
    """
    Reset the OutcomeTracker singleton and block the lazy RAGSystem import.

    OutcomeTracker._get_vector_store() lazily imports RAGSystem, which triggers
    embedding-model loading. We patch `rag.RAGSystem` to raise ImportError so
    the method silently returns None — the same code path as an unavailable RAG
    system in production, without triggering slow model loading.

    Tests that need a real mock store inject it via tracker._vector_store directly;
    because _get_vector_store() checks `if self._vector_store is not None` first,
    those tests bypass the patched import path entirely.
    """
    OutcomeTracker._instance = None
    # Patch the `rag` module import inside _get_vector_store so RAGSystem is never
    # actually loaded.  We patch at the source module level.
    import sys

    _fake_rag = MagicMock()
    _fake_rag.RAGSystem.return_value.vector_store = None
    _orig = sys.modules.get("rag")
    sys.modules["rag"] = _fake_rag
    yield
    sys.modules.pop("rag", None)
    if _orig is not None:
        sys.modules["rag"] = _orig
    OutcomeTracker._instance = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record_n(tracker, op_name, n_success, n_failure, robot_id="Robot1"):
    """Record n_success successes and n_failure failures for op_name."""
    for _ in range(n_success):
        tracker.record(op_name, robot_id, success=True)
    for _ in range(n_failure):
        tracker.record(op_name, robot_id, success=False, error=f"{op_name}_error")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    """OutcomeTracker follows the project-wide singleton pattern."""

    def test_two_calls_return_same_instance(self):
        """OutcomeTracker() must return the same object on repeated calls."""
        a = OutcomeTracker()
        b = OutcomeTracker()
        assert a is b

    def test_get_outcome_tracker_returns_singleton(self):
        """get_outcome_tracker() convenience function returns the singleton."""
        a = OutcomeTracker()
        b = get_outcome_tracker()
        assert a is b

    def test_reset_fixture_gives_fresh_state(self):
        """After the fixture resets _instance, a new call gives a clean tracker."""
        tracker = OutcomeTracker()
        tracker.record("op_a", "Robot1", success=False)
        assert len(tracker.get_recent_failures()) == 1

        # Simulate what the fixture does
        OutcomeTracker._instance = None

        fresh = OutcomeTracker()
        assert len(fresh.get_recent_failures()) == 0


# ---------------------------------------------------------------------------
# record()
# ---------------------------------------------------------------------------


class TestRecord:
    """record() appends outcomes to the ring buffer with correct fields."""

    def test_record_success_appends_to_buffer(self):
        """Successful record increments buffer length."""
        tracker = OutcomeTracker()
        tracker.record("move_to_coordinate", "Robot1", success=True)
        assert len(tracker._recent_outcomes) == 1

    def test_record_failure_appends_to_buffer(self):
        """Failed record appends with success=False."""
        tracker = OutcomeTracker()
        tracker.record("grasp_object", "Robot1", success=False, error="timeout")
        outcomes = list(tracker._recent_outcomes)
        assert outcomes[0]["success"] is False
        assert outcomes[0]["error"] == "timeout"

    def test_record_stores_all_fields(self):
        """record() stores operation_name, robot_id, success, error, duration, timestamp."""
        tracker = OutcomeTracker()
        before = time.time()
        tracker.record("move_to_coordinate", "Robot2", success=True, duration_ms=120.5)
        after = time.time()

        entry = list(tracker._recent_outcomes)[0]
        assert entry["operation_name"] == "move_to_coordinate"
        assert entry["robot_id"] == "Robot2"
        assert entry["success"] is True
        assert entry["duration_ms"] == 120.5
        assert before <= entry["timestamp"] <= after

    def test_record_multiple_operations(self):
        """Multiple records accumulate in order."""
        tracker = OutcomeTracker()
        tracker.record("op_a", "Robot1", success=True)
        tracker.record("op_b", "Robot1", success=False)
        tracker.record("op_c", "Robot1", success=True)

        names = [o["operation_name"] for o in tracker._recent_outcomes]
        assert names == ["op_a", "op_b", "op_c"]


# ---------------------------------------------------------------------------
# get_recent_failures()
# ---------------------------------------------------------------------------


class TestGetRecentFailures:
    """get_recent_failures() returns only failures, most-recent-first."""

    def test_empty_buffer_returns_empty_list(self):
        """No records → empty list."""
        tracker = OutcomeTracker()
        assert tracker.get_recent_failures() == []

    def test_filters_to_failures_only(self):
        """Successes are excluded from the result."""
        tracker = OutcomeTracker()
        tracker.record("op_a", "Robot1", success=True)
        tracker.record("op_b", "Robot1", success=False, error="err")
        tracker.record("op_c", "Robot1", success=True)

        failures = tracker.get_recent_failures()
        assert len(failures) == 1
        assert failures[0]["operation_name"] == "op_b"

    def test_most_recent_first_ordering(self):
        """Most recent failures come first."""
        tracker = OutcomeTracker()
        tracker.record("op_a", "Robot1", success=False)
        tracker.record("op_b", "Robot1", success=False)
        tracker.record("op_c", "Robot1", success=False)

        failures = tracker.get_recent_failures()
        names = [f["operation_name"] for f in failures]
        assert names == ["op_c", "op_b", "op_a"]

    def test_limit_parameter_is_respected(self):
        """limit= caps the number of returned failures."""
        tracker = OutcomeTracker()
        for i in range(10):
            tracker.record(f"op_{i}", "Robot1", success=False)

        failures = tracker.get_recent_failures(limit=3)
        assert len(failures) == 3

    def test_operation_name_filter(self):
        """operation_name= filters to only that operation."""
        tracker = OutcomeTracker()
        tracker.record("grasp_object", "Robot1", success=False)
        tracker.record("move_to_coordinate", "Robot1", success=False)
        tracker.record("grasp_object", "Robot1", success=False)

        failures = tracker.get_recent_failures(operation_name="grasp_object")
        assert len(failures) == 2
        assert all(f["operation_name"] == "grasp_object" for f in failures)

    def test_operation_name_filter_no_match(self):
        """Filter for an op that never failed returns empty list."""
        tracker = OutcomeTracker()
        tracker.record("op_a", "Robot1", success=False)
        assert tracker.get_recent_failures(operation_name="nonexistent") == []


# ---------------------------------------------------------------------------
# get_failure_rate()
# ---------------------------------------------------------------------------


class TestGetFailureRate:
    """get_failure_rate() computes failure rate over the most recent window."""

    def test_unknown_operation_returns_zero(self):
        """Operations not in the buffer return 0.0."""
        tracker = OutcomeTracker()
        assert tracker.get_failure_rate("nonexistent_op") == 0.0

    def test_all_successes_returns_zero(self):
        """100% success rate → 0.0 failure rate."""
        tracker = OutcomeTracker()
        _record_n(tracker, "op_a", n_success=5, n_failure=0)
        assert tracker.get_failure_rate("op_a") == 0.0

    def test_all_failures_returns_one(self):
        """100% failure rate → 1.0."""
        tracker = OutcomeTracker()
        _record_n(tracker, "op_a", n_success=0, n_failure=4)
        assert tracker.get_failure_rate("op_a") == 1.0

    def test_mixed_rate(self):
        """2 failures out of 10 executions = 0.2."""
        tracker = OutcomeTracker()
        _record_n(tracker, "op_a", n_success=8, n_failure=2)
        rate = tracker.get_failure_rate("op_a")
        assert abs(rate - 0.2) < 1e-9

    def test_window_limits_history(self):
        """window= restricts to the most recent N executions of that operation."""
        tracker = OutcomeTracker()
        # First 10 all succeed, then 5 all fail
        _record_n(tracker, "op_a", n_success=10, n_failure=0)
        _record_n(tracker, "op_a", n_success=0, n_failure=5)

        # Full window of 50 → 5/15 ≈ 0.333
        assert abs(tracker.get_failure_rate("op_a", window=50) - 5 / 15) < 1e-9

        # Window of 5 (only the recent failures) → 1.0
        assert tracker.get_failure_rate("op_a", window=5) == 1.0


# ---------------------------------------------------------------------------
# get_frequent_failure_patterns()
# ---------------------------------------------------------------------------


class TestGetFrequentFailurePatterns:
    """get_frequent_failure_patterns() identifies high-failure-rate operations."""

    def test_empty_buffer_returns_empty_list(self):
        """No records → no patterns."""
        tracker = OutcomeTracker()
        assert tracker.get_frequent_failure_patterns() == []

    def test_low_failure_rate_excluded(self):
        """Operations below min_failure_rate are not included."""
        tracker = OutcomeTracker()
        _record_n(tracker, "op_a", n_success=9, n_failure=1)  # 10% failure

        patterns = tracker.get_frequent_failure_patterns(
            min_failure_rate=0.3, min_occurrences=1
        )
        assert patterns == []

    def test_high_failure_rate_included(self):
        """Operations at or above min_failure_rate are included."""
        tracker = OutcomeTracker()
        _record_n(tracker, "op_a", n_success=2, n_failure=8)  # 80% failure

        patterns = tracker.get_frequent_failure_patterns(
            min_failure_rate=0.3, min_occurrences=1
        )
        assert len(patterns) == 1
        assert patterns[0]["operation_name"] == "op_a"
        assert abs(patterns[0]["failure_rate"] - 0.8) < 1e-9

    def test_min_occurrences_excludes_sparse_ops(self):
        """Operations with fewer executions than min_occurrences are skipped."""
        tracker = OutcomeTracker()
        _record_n(
            tracker, "op_a", n_success=0, n_failure=2
        )  # 100% failure, 2 executions

        # Require 3 min — should be excluded
        patterns = tracker.get_frequent_failure_patterns(
            min_failure_rate=0.3, min_occurrences=3
        )
        assert patterns == []

    def test_sorted_by_failure_rate_descending(self):
        """Patterns are sorted highest failure rate first."""
        tracker = OutcomeTracker()
        _record_n(tracker, "op_low", n_success=7, n_failure=3)  # 30%
        _record_n(tracker, "op_high", n_success=1, n_failure=9)  # 90%
        _record_n(tracker, "op_mid", n_success=4, n_failure=6)  # 60%

        patterns = tracker.get_frequent_failure_patterns(
            min_failure_rate=0.1, min_occurrences=1
        )
        rates = [p["failure_rate"] for p in patterns]
        assert rates == sorted(rates, reverse=True)

    def test_sample_errors_collected(self):
        """sample_errors contains up to 3 distinct non-None error messages."""
        tracker = OutcomeTracker()
        for err in ["err_a", "err_b", "err_c", "err_a", "err_d"]:
            tracker.record("op_a", "Robot1", success=False, error=err)

        patterns = tracker.get_frequent_failure_patterns(
            min_failure_rate=0.0, min_occurrences=1
        )
        assert len(patterns) == 1
        errors = patterns[0]["sample_errors"]
        assert len(errors) <= 3
        assert all(isinstance(e, str) for e in errors)

    def test_none_errors_excluded_from_sample(self):
        """Failures with error=None are not included in sample_errors."""
        tracker = OutcomeTracker()
        tracker.record("op_a", "Robot1", success=False, error=None)
        tracker.record("op_a", "Robot1", success=False, error=None)
        tracker.record("op_a", "Robot1", success=False, error=None)

        patterns = tracker.get_frequent_failure_patterns(
            min_failure_rate=0.0, min_occurrences=1
        )
        assert patterns[0]["sample_errors"] == []

    def test_total_executions_field(self):
        """total_executions counts both successes and failures."""
        tracker = OutcomeTracker()
        _record_n(tracker, "op_a", n_success=4, n_failure=6)

        patterns = tracker.get_frequent_failure_patterns(
            min_failure_rate=0.0, min_occurrences=1
        )
        assert patterns[0]["total_executions"] == 10


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    """reset() clears the in-memory ring buffer."""

    def test_reset_empties_buffer(self):
        """After reset(), buffer is empty."""
        tracker = OutcomeTracker()
        _record_n(tracker, "op_a", n_success=3, n_failure=2)
        assert len(tracker._recent_outcomes) > 0

        tracker.reset()
        assert len(tracker._recent_outcomes) == 0

    def test_reset_does_not_affect_vector_store(self):
        """reset() only touches the ring buffer, not _vector_store."""
        tracker = OutcomeTracker()
        mock_store = MagicMock()
        tracker._vector_store = mock_store

        tracker.reset()

        # Vector store should be untouched
        assert tracker._vector_store is mock_store
        mock_store.update_operation_metadata.assert_not_called()


# ---------------------------------------------------------------------------
# Ring-buffer cap
# ---------------------------------------------------------------------------


class TestRingBufferCap:
    """The ring buffer never grows beyond _MAX_RECENT_OUTCOMES."""

    def test_buffer_does_not_exceed_max(self):
        """Recording more than _MAX_RECENT_OUTCOMES entries evicts the oldest."""
        from orchestrators.OutcomeTracker import _MAX_RECENT_OUTCOMES

        tracker = OutcomeTracker()
        for i in range(_MAX_RECENT_OUTCOMES + 10):
            tracker.record(f"op_{i}", "Robot1", success=True)

        assert len(tracker._recent_outcomes) == _MAX_RECENT_OUTCOMES

    def test_oldest_entry_evicted(self):
        """The first recorded entry is gone after the buffer fills."""
        from orchestrators.OutcomeTracker import _MAX_RECENT_OUTCOMES

        tracker = OutcomeTracker()
        tracker.record("first_op", "Robot1", success=True)

        for i in range(_MAX_RECENT_OUTCOMES):
            tracker.record(f"filler_{i}", "Robot1", success=True)

        names = [o["operation_name"] for o in tracker._recent_outcomes]
        assert "first_op" not in names


# ---------------------------------------------------------------------------
# VectorStore integration
# ---------------------------------------------------------------------------


class TestVectorStoreIntegration:
    """record() calls update_operation_metadata on the VectorStore when available."""

    def _make_mock_store(self, op_name: str):
        """Return a mock VectorStore with one operation already indexed."""
        store = MagicMock()
        store.operation_ids = [f"{op_name}_id"]
        store.metadata = [{"name": op_name}]
        store.get_operation.return_value = {
            "metadata": {"execution_count": 0, "success_count": 0, "failure_count": 0}
        }
        store.update_operation_metadata.return_value = True
        return store

    def test_success_increments_counters(self):
        """Successful record increments execution_count and success_count."""
        tracker = OutcomeTracker()
        mock_store = self._make_mock_store("move_to_coordinate")
        tracker._vector_store = mock_store

        tracker.record("move_to_coordinate", "Robot1", success=True)

        mock_store.update_operation_metadata.assert_called_once()
        _, update_kwargs = mock_store.update_operation_metadata.call_args
        if not update_kwargs:
            _, meta_dict = mock_store.update_operation_metadata.call_args[0]
        else:
            meta_dict = (
                update_kwargs.get("metadata_update")
                or mock_store.update_operation_metadata.call_args[0][1]
            )

        assert meta_dict["execution_count"] == 1
        assert meta_dict["success_count"] == 1
        assert meta_dict["failure_count"] == 0
        assert meta_dict["last_outcome"] == "success"

    def test_failure_increments_failure_count(self):
        """Failed record increments execution_count and failure_count."""
        tracker = OutcomeTracker()
        mock_store = self._make_mock_store("grasp_object")
        tracker._vector_store = mock_store

        tracker.record("grasp_object", "Robot1", success=False)

        mock_store.update_operation_metadata.assert_called_once()
        meta_dict = mock_store.update_operation_metadata.call_args[0][1]
        assert meta_dict["failure_count"] == 1
        assert meta_dict["success_count"] == 0
        assert meta_dict["last_outcome"] == "failure"

    def test_no_vector_store_does_not_raise(self):
        """When VectorStore is unavailable, record() silently skips the update."""
        tracker = OutcomeTracker()
        tracker._vector_store = None  # Ensure lazy fetch also fails

        with patch.object(tracker, "_get_vector_store", return_value=None):
            tracker.record("op_a", "Robot1", success=True)  # Must not raise

    def test_unknown_operation_skips_update(self):
        """When the operation_id is not in the store, update is silently skipped."""
        tracker = OutcomeTracker()
        store = MagicMock()
        store.operation_ids = []
        store.metadata = []
        tracker._vector_store = store

        tracker.record("unknown_op", "Robot1", success=True)

        store.update_operation_metadata.assert_not_called()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Concurrent records and reads must not corrupt the ring buffer."""

    def test_concurrent_records_are_safe(self):
        """Multiple threads recording simultaneously must not raise or corrupt state."""
        tracker = OutcomeTracker()
        errors = []

        def worker(robot_id, n):
            try:
                for i in range(n):
                    tracker.record(f"op_{i}", robot_id, success=(i % 2 == 0))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(f"Robot{i}", 50)) for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
