#!/usr/bin/env python3
"""
Feedback Collector
===================

Bridges OutcomeTracker (raw execution data) and CommandParser (LLM prompt
builder) for the self-improvement loop.

Queries OutcomeTracker for recent failure patterns and converts them into
formatted anti-pattern warning strings that CommandParser can inject into the
LLM prompt. This allows the LLM to avoid sequences known to fail in the
current environment without requiring a full model retraining cycle.

Usage (from CommandParser):

    from agents.FeedbackCollector import get_feedback_collector
    collector = get_feedback_collector()
    warnings = collector.get_anti_pattern_warnings(command_text)
    if warnings:
        prompt += f"\\n{warnings}"
"""

import logging
import threading
from typing import List, Optional

from config.Memory import (
    FEEDBACK_MAX_WARNINGS,
    FEEDBACK_MIN_FAILURE_RATE,
    FEEDBACK_MIN_OCCURRENCES,
)
from core.LoggingSetup import setup_logging

setup_logging(__name__)
logger = logging.getLogger(__name__)

# Module-level alias for tests that need to verify the warning cap.
_MAX_WARNINGS: int = FEEDBACK_MAX_WARNINGS


class FeedbackCollector:
    """
    Collects execution feedback and formats it as LLM prompt warnings.

    Reads from the OutcomeTracker ring buffer and produces human-readable
    warning text that CommandParser can inject into the planning prompt so
    the LLM avoids known-failing operation sequences.

    Thread-safe singleton.
    """

    _instance: Optional["FeedbackCollector"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "FeedbackCollector":
        """Return the singleton instance, creating it on first call."""
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize the collector (runs once due to singleton guard)."""
        if self._initialized:
            return
        self._initialized = True
        logger.debug("FeedbackCollector initialized")

    # ==========================================================================
    # Public API
    # ==========================================================================

    def get_anti_pattern_warnings(
        self,
        command_text: str = "",
        min_failure_rate: float = FEEDBACK_MIN_FAILURE_RATE,
        min_occurrences: int = FEEDBACK_MIN_OCCURRENCES,
    ) -> str:
        """
        Return a formatted anti-pattern warning block for the LLM prompt.

        Queries OutcomeTracker for operations with high recent failure rates
        and formats them as a warning section suitable for injection into the
        CommandParser prompt.

        Args:
            command_text: The command being parsed. Used to filter warnings
                to only those relevant to the command (future use; currently
                all patterns above threshold are returned).
            min_failure_rate: Minimum failure rate (0-1) to include a warning.
            min_occurrences: Minimum executions before an operation is flagged.

        Returns:
            Formatted warning string (may be empty string if no patterns found).

        Example:
            >>> collector = FeedbackCollector()
            >>> warnings = collector.get_anti_pattern_warnings("grasp the red cube")
            >>> if warnings:
            ...     prompt += "\\n" + warnings
        """
        patterns = self._get_failure_patterns(min_failure_rate, min_occurrences)
        if not patterns:
            return ""

        return self._format_warnings(patterns[:FEEDBACK_MAX_WARNINGS])

    def get_recent_error_context(
        self, operation_name: str, limit: int = 3
    ) -> List[str]:
        """
        Return recent error messages for a specific operation.

        Useful for including concrete failure context in a prompt when the
        LLM is asked to retry a failed sequence.

        Args:
            operation_name: Operation name to fetch errors for.
            limit: Maximum number of error strings to return.

        Returns:
            List of recent error message strings (may be empty).
        """
        try:
            from orchestrators.OutcomeTracker import get_outcome_tracker

            tracker = get_outcome_tracker()
            failures = tracker.get_recent_failures(
                limit=limit * 2, operation_name=operation_name
            )
            errors = [f["error"] for f in failures if f.get("error")]
            return list(dict.fromkeys(errors))[:limit]  # deduplicate, preserve order
        except Exception as e:
            logger.debug(f"Could not fetch recent errors for {operation_name}: {e}")
            return []

    def summarize_session_outcomes(self) -> str:
        """
        Return a brief human-readable summary of execution outcomes this session.

        Intended for logging and the AutoRT loop's status display, not for
        LLM prompt injection (too verbose).

        Returns:
            Multi-line summary string.
        """
        try:
            from orchestrators.OutcomeTracker import get_outcome_tracker

            tracker = get_outcome_tracker()
            patterns = tracker.get_frequent_failure_patterns(
                min_failure_rate=0.0, min_occurrences=1
            )

            if not patterns:
                return "No operations executed yet this session."

            lines = ["=== Session Outcome Summary ==="]
            for p in patterns:
                rate_pct = int(p["failure_rate"] * 100)
                lines.append(
                    f"  {p['operation_name']}: "
                    f"{p['total_executions']} executions, "
                    f"{rate_pct}% failure rate"
                )
            return "\n".join(lines)
        except Exception as e:
            logger.debug(f"Could not build session summary: {e}")
            return ""

    # ==========================================================================
    # Private helpers
    # ==========================================================================

    def _get_failure_patterns(
        self, min_failure_rate: float, min_occurrences: int
    ) -> list:
        """
        Fetch failure patterns from OutcomeTracker.

        Returns an empty list if OutcomeTracker is unavailable or has no data.

        Args:
            min_failure_rate: Minimum failure rate threshold.
            min_occurrences: Minimum executions threshold.

        Returns:
            List of pattern dicts from OutcomeTracker.get_frequent_failure_patterns().
        """
        try:
            from orchestrators.OutcomeTracker import get_outcome_tracker

            tracker = get_outcome_tracker()
            return tracker.get_frequent_failure_patterns(
                min_failure_rate=min_failure_rate,
                min_occurrences=min_occurrences,
            )
        except Exception as e:
            logger.debug(f"Could not fetch failure patterns: {e}")
            return []

    def _format_warnings(self, patterns: list) -> str:
        """
        Convert a list of failure pattern dicts into a prompt-ready warning block.

        Args:
            patterns: List of pattern dicts with keys: operation_name,
                failure_rate, total_executions, sample_errors.

        Returns:
            Formatted multi-line string starting with a section header.
        """
        lines = ["=== ANTI-PATTERN WARNINGS (based on recent execution history) ==="]
        lines.append(
            "The following operations have been failing recently. "
            "Avoid these sequences or use alternatives where possible:"
        )
        lines.append("")

        for p in patterns:
            rate_pct = int(p["failure_rate"] * 100)
            op_name = p["operation_name"]
            n = p["total_executions"]
            lines.append(
                f"- {op_name}: {rate_pct}% failure rate ({n} recent executions)"
            )

            for err in p.get("sample_errors", [])[:2]:
                lines.append(f'    Recent error: "{err}"')

        lines.append("")
        lines.append(
            "If you must use a flagged operation, add error recovery steps "
            "(e.g. check_robot_status after the call, retry logic)."
        )

        return "\n".join(lines)


# ==========================================================================---------
# Module-level singleton accessor
# ==========================================================================---------


def get_feedback_collector() -> FeedbackCollector:
    """
    Return the global FeedbackCollector singleton.

    Returns:
        The global FeedbackCollector instance.
    """
    return FeedbackCollector()
