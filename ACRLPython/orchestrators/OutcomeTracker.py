#!/usr/bin/env python3
"""
Outcome Tracker
================

Records robot operation outcomes (success/failure) and persists them into
VectorStore metadata so the self-improvement loop can learn which operation
sequences fail and adjust RAG examples accordingly.

Also maintains an in-memory ring buffer of recent outcomes that
FeedbackCollector queries to build anti-pattern warnings for CommandParser.

Architecture:
    SequenceExecutor  -->  OutcomeTracker.record()
                               |
                               +-- VectorStore.update_operation_metadata()
                               +-- _recent_outcomes deque  <-- FeedbackCollector
"""

import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

from config.Memory import OUTCOME_BUFFER_SIZE
from core.LoggingSetup import setup_logging

setup_logging(__name__)
logger = logging.getLogger(__name__)

# Module-level alias for tests that need to validate ring-buffer capacity.
_MAX_RECENT_OUTCOMES: int = OUTCOME_BUFFER_SIZE


class OutcomeTracker:
    """
    Singleton tracker for robot operation outcomes.

    Records each executed operation's success/failure into:
    1. The active VectorStore (persisted metadata, updated in-place).
    2. An in-memory ring buffer for FeedbackCollector to read.

    Thread-safe: all public methods acquire _lock before mutating state.
    """

    _instance: Optional["OutcomeTracker"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "OutcomeTracker":
        """Return the singleton instance, creating it on first call."""
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize the tracker (runs once due to singleton guard)."""
        if self._initialized:
            return
        self._initialized = True
        self._recent_outcomes: deque = deque(maxlen=OUTCOME_BUFFER_SIZE)
        self._vector_store = None  # Lazily fetched from RAGSystem
        logger.info("OutcomeTracker initialized")

    # ==========================================================================
    # Public API
    # ==========================================================================

    def record(
        self,
        operation_name: str,
        robot_id: str,
        success: bool,
        error: Optional[str] = None,
        duration_ms: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record the outcome of a single executed operation.

        Appends the outcome to the in-memory ring buffer and, if a
        VectorStore is available, merges updated execution statistics
        into the matching operation's metadata.

        Args:
            operation_name: Registry name of the operation (e.g. "move_to_coordinate").
            robot_id: ID of the robot that ran the operation (e.g. "Robot1").
            success: True if the operation succeeded, False otherwise.
            error: Optional error message on failure.
            duration_ms: Optional execution duration in milliseconds.
            params: Optional parameter dict for context (not persisted to VectorStore).
        """
        outcome: Dict[str, Any] = {
            "operation_name": operation_name,
            "robot_id": robot_id,
            "success": success,
            "error": error,
            "duration_ms": duration_ms,
            "timestamp": time.time(),
        }

        with self._lock:
            self._recent_outcomes.append(outcome)
            self._update_vector_store(operation_name, success)

        log_level = logging.DEBUG if success else logging.WARNING
        logger.log(
            log_level,
            f"[{robot_id}] {operation_name}: {'OK' if success else 'FAIL'}"
            + (f" — {error}" if error else ""),
        )

    def get_recent_failures(
        self, limit: int = 20, operation_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Return the most recent failed outcomes from the ring buffer.

        Args:
            limit: Maximum number of failure records to return.
            operation_name: If provided, filter to only this operation name.

        Returns:
            List of outcome dicts (most recent first), each with keys:
            operation_name, robot_id, success, error, duration_ms, timestamp.
        """
        with self._lock:
            failures = [
                o
                for o in self._recent_outcomes
                if not o["success"]
                and (operation_name is None or o["operation_name"] == operation_name)
            ]
        # Most recent first (deque is oldest-first)
        return list(reversed(failures))[:limit]

    def get_failure_rate(self, operation_name: str, window: int = 50) -> float:
        """
        Compute the failure rate for an operation over the last N executions.

        Args:
            operation_name: Operation name to compute rate for.
            window: Number of most recent executions to consider.

        Returns:
            Float in [0.0, 1.0] representing failure rate, or 0.0 if no
            executions found.
        """
        with self._lock:
            relevant = [
                o
                for o in self._recent_outcomes
                if o["operation_name"] == operation_name
            ]
        # Most recent N
        recent = relevant[-window:]
        if not recent:
            return 0.0
        failures = sum(1 for o in recent if not o["success"])
        return failures / len(recent)

    def get_frequent_failure_patterns(
        self, min_failure_rate: float = 0.3, min_occurrences: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Identify operation names with a failure rate above the threshold.

        Used by FeedbackCollector to build anti-pattern warnings.

        Args:
            min_failure_rate: Minimum failure rate (0-1) to flag an operation.
            min_occurrences: Minimum total executions before an operation
                is considered for flagging.

        Returns:
            List of dicts with keys: operation_name, failure_rate, sample_errors.
        """
        with self._lock:
            # Build per-operation execution lists from the ring buffer
            by_operation: Dict[str, List[Dict]] = {}
            for o in self._recent_outcomes:
                by_operation.setdefault(o["operation_name"], []).append(o)

        patterns = []
        for op_name, outcomes in by_operation.items():
            if len(outcomes) < min_occurrences:
                continue
            failures = [o for o in outcomes if not o["success"]]
            rate = len(failures) / len(outcomes)
            if rate >= min_failure_rate:
                # Collect up to 3 distinct non-None error messages
                sample_errors = list(
                    dict.fromkeys(o["error"] for o in failures if o.get("error"))
                )[:3]
                patterns.append(
                    {
                        "operation_name": op_name,
                        "failure_rate": rate,
                        "total_executions": len(outcomes),
                        "sample_errors": sample_errors,
                    }
                )

        # Sort by failure rate descending
        patterns.sort(key=lambda p: p["failure_rate"], reverse=True)
        return patterns

    def reset(self) -> None:
        """
        Clear the in-memory ring buffer (does not affect VectorStore).

        Primarily used in tests to ensure a clean state between test runs.
        """
        with self._lock:
            self._recent_outcomes.clear()
        logger.debug("OutcomeTracker ring buffer cleared")

    # ==========================================================================
    # Private helpers
    # ==========================================================================

    def _update_vector_store(self, operation_name: str, success: bool) -> None:
        """
        Update execution statistics in the VectorStore for the given operation.

        Lazily fetches the active VectorStore from the RAGSystem singleton.
        Silently skips if RAG is not initialized (e.g. during tests without
        a full stack).

        Must be called with self._lock held.

        Args:
            operation_name: Name of the operation to update.
            success: Whether the execution succeeded.
        """
        store = self._get_vector_store()
        if store is None:
            return

        # Look up the operation_id from the VectorStore by name
        operation_id = self._resolve_operation_id(store, operation_name)
        if operation_id is None:
            return

        # Read current counters from metadata, incrementing them
        try:
            entry = store.get_operation(operation_id)
            if entry is None:
                return
            meta = entry.get("metadata", {})
            exec_count = meta.get("execution_count", 0) + 1
            success_count = meta.get("success_count", 0) + (1 if success else 0)
            failure_count = meta.get("failure_count", 0) + (0 if success else 1)

            store.update_operation_metadata(
                operation_id,
                {
                    "execution_count": exec_count,
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "last_outcome": "success" if success else "failure",
                    "last_outcome_at": time.time(),
                },
            )
        except Exception as e:
            logger.debug(
                f"Could not update VectorStore metadata for {operation_name}: {e}"
            )

    def _get_vector_store(self):
        """
        Lazily fetch the active VectorStore from the RAGSystem singleton.

        Returns:
            VectorStore instance or None if unavailable.
        """
        if self._vector_store is not None:
            return self._vector_store

        try:
            from rag import RAGSystem

            rag = RAGSystem()
            # RAGSystem exposes its store as .vector_store after index_operations()
            if hasattr(rag, "vector_store") and rag.vector_store is not None:
                self._vector_store = rag.vector_store
                return self._vector_store
        except Exception:
            pass
        return None

    def _resolve_operation_id(self, store, operation_name: str) -> Optional[str]:
        """
        Find the VectorStore operation_id that matches the given operation name.

        Checks VectorStore metadata where meta['name'] == operation_name.

        Args:
            store: VectorStore instance to search.
            operation_name: Human-readable operation name (e.g. "move_to_coordinate").

        Returns:
            operation_id string if found, None otherwise.
        """
        for op_id, meta in zip(store.operation_ids, store.metadata):
            if meta.get("name") == operation_name:
                return op_id
        return None


# ==========================================================================---------
# Module-level singleton accessor
# ==========================================================================---------


def get_outcome_tracker() -> OutcomeTracker:
    """
    Return the global OutcomeTracker singleton.

    Returns:
        The global OutcomeTracker instance.
    """
    return OutcomeTracker()
