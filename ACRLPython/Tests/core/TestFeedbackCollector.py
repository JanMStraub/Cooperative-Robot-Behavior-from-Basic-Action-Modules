#!/usr/bin/env python3
"""
Tests for FeedbackCollector
============================

Covers:
- Singleton: two FeedbackCollector() calls return the same object
- get_anti_pattern_warnings(): empty string when no patterns
- get_anti_pattern_warnings(): formatted block with correct content when patterns exist
- get_anti_pattern_warnings(): respects min_failure_rate / min_occurrences args
- get_anti_pattern_warnings(): caps output at _MAX_WARNINGS=5
- get_recent_error_context(): returns deduplicated errors for named op
- get_recent_error_context(): gracefully handles unavailable OutcomeTracker
- summarize_session_outcomes(): "No operations" message when empty tracker
- summarize_session_outcomes(): formatted summary when patterns exist
- summarize_session_outcomes(): gracefully handles unavailable OutcomeTracker

All OutcomeTracker calls are mocked so FeedbackCollector tests are fully isolated.
"""

import pytest
from unittest.mock import MagicMock, patch

from agents.FeedbackCollector import FeedbackCollector, get_feedback_collector


# ---------------------------------------------------------------------------
# Fixture: reset singleton between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_feedback_collector():
    """Null out the FeedbackCollector singleton before and after every test."""
    FeedbackCollector._instance = None
    yield
    FeedbackCollector._instance = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PATCH_TARGET = "orchestrators.OutcomeTracker.get_outcome_tracker"


def _mock_tracker_with_patterns(patterns):
    """Return a mock OutcomeTracker whose get_frequent_failure_patterns returns patterns."""
    tracker = MagicMock()
    tracker.get_frequent_failure_patterns.return_value = patterns
    tracker.get_recent_failures.return_value = []
    return tracker


def _make_pattern(op_name, failure_rate, total_executions, sample_errors=None):
    """Build a pattern dict matching OutcomeTracker's output format."""
    return {
        "operation_name": op_name,
        "failure_rate": failure_rate,
        "total_executions": total_executions,
        "sample_errors": sample_errors or [],
    }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    """FeedbackCollector follows the project-wide singleton pattern."""

    def test_two_calls_return_same_instance(self):
        """FeedbackCollector() must return the same object on repeated calls."""
        a = FeedbackCollector()
        b = FeedbackCollector()
        assert a is b

    def test_get_feedback_collector_returns_singleton(self):
        """get_feedback_collector() convenience function returns the singleton."""
        a = FeedbackCollector()
        b = get_feedback_collector()
        assert a is b


# ---------------------------------------------------------------------------
# get_anti_pattern_warnings() — empty cases
# ---------------------------------------------------------------------------


class TestGetAntiPatternWarningsEmpty:
    """get_anti_pattern_warnings() returns empty string when nothing to warn about."""

    def test_returns_empty_string_when_no_patterns(self):
        """No failure patterns → empty string returned."""
        collector = FeedbackCollector()
        tracker = _mock_tracker_with_patterns([])

        with patch(PATCH_TARGET, return_value=tracker):
            result = collector.get_anti_pattern_warnings()

        assert result == ""

    def test_returns_empty_string_when_tracker_unavailable(self):
        """If OutcomeTracker import raises, the method returns empty string."""
        collector = FeedbackCollector()

        with patch(PATCH_TARGET, side_effect=Exception("unavailable")):
            result = collector.get_anti_pattern_warnings()

        assert result == ""


# ---------------------------------------------------------------------------
# get_anti_pattern_warnings() — content
# ---------------------------------------------------------------------------


class TestGetAntiPatternWarningsContent:
    """Returned warning block contains expected operation names and header."""

    def test_contains_header_line(self):
        """Warning block must start with the ANTI-PATTERN WARNINGS header."""
        collector = FeedbackCollector()
        pattern = _make_pattern("grasp_object", 0.8, 10, ["timeout"])
        tracker = _mock_tracker_with_patterns([pattern])

        with patch(PATCH_TARGET, return_value=tracker):
            result = collector.get_anti_pattern_warnings()

        assert "ANTI-PATTERN WARNINGS" in result

    def test_contains_operation_name(self):
        """Operation name appears in the warning output."""
        collector = FeedbackCollector()
        pattern = _make_pattern("grasp_object", 0.8, 10)
        tracker = _mock_tracker_with_patterns([pattern])

        with patch(PATCH_TARGET, return_value=tracker):
            result = collector.get_anti_pattern_warnings()

        assert "grasp_object" in result

    def test_contains_failure_rate_percentage(self):
        """Failure rate is formatted as an integer percentage."""
        collector = FeedbackCollector()
        pattern = _make_pattern("op_a", 0.75, 8)
        tracker = _mock_tracker_with_patterns([pattern])

        with patch(PATCH_TARGET, return_value=tracker):
            result = collector.get_anti_pattern_warnings()

        assert "75%" in result

    def test_sample_errors_included_in_output(self):
        """Up to 2 sample error messages appear in the warning block."""
        collector = FeedbackCollector()
        pattern = _make_pattern("op_a", 0.9, 10, ["IK timeout", "gripper stuck"])
        tracker = _mock_tracker_with_patterns([pattern])

        with patch(PATCH_TARGET, return_value=tracker):
            result = collector.get_anti_pattern_warnings()

        assert "IK timeout" in result
        assert "gripper stuck" in result

    def test_multiple_operations_all_appear(self):
        """All flagged operations are listed in the output."""
        collector = FeedbackCollector()
        patterns = [
            _make_pattern("op_a", 0.9, 10),
            _make_pattern("op_b", 0.6, 5),
        ]
        tracker = _mock_tracker_with_patterns(patterns)

        with patch(PATCH_TARGET, return_value=tracker):
            result = collector.get_anti_pattern_warnings()

        assert "op_a" in result
        assert "op_b" in result


# ---------------------------------------------------------------------------
# get_anti_pattern_warnings() — thresholds and cap
# ---------------------------------------------------------------------------


class TestGetAntiPatternWarningsThresholds:
    """min_failure_rate, min_occurrences, and _MAX_WARNINGS are passed/respected."""

    def test_min_failure_rate_forwarded_to_tracker(self):
        """The min_failure_rate argument is passed to OutcomeTracker."""
        collector = FeedbackCollector()
        tracker = _mock_tracker_with_patterns([])

        with patch(PATCH_TARGET, return_value=tracker):
            collector.get_anti_pattern_warnings(min_failure_rate=0.5)

        call_kwargs = tracker.get_frequent_failure_patterns.call_args
        # Accept both positional and keyword style
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        args = call_kwargs.args if call_kwargs.args else ()
        # min_failure_rate is the first positional arg or a keyword
        passed_rate = kwargs.get("min_failure_rate", args[0] if args else None)
        assert passed_rate == 0.5

    def test_min_occurrences_forwarded_to_tracker(self):
        """The min_occurrences argument is passed to OutcomeTracker."""
        collector = FeedbackCollector()
        tracker = _mock_tracker_with_patterns([])

        with patch(PATCH_TARGET, return_value=tracker):
            collector.get_anti_pattern_warnings(min_occurrences=5)

        call_kwargs = tracker.get_frequent_failure_patterns.call_args
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        args = call_kwargs.args if call_kwargs.args else ()
        passed_occ = kwargs.get("min_occurrences", args[1] if len(args) > 1 else None)
        assert passed_occ == 5

    def test_capped_at_max_warnings(self):
        """More than _MAX_WARNINGS patterns are truncated to exactly 5 operations."""
        from agents.FeedbackCollector import _MAX_WARNINGS

        collector = FeedbackCollector()
        # Create more patterns than the cap
        patterns = [_make_pattern(f"op_{i}", 0.9, 10) for i in range(_MAX_WARNINGS + 3)]
        tracker = _mock_tracker_with_patterns(patterns)

        with patch(PATCH_TARGET, return_value=tracker):
            result = collector.get_anti_pattern_warnings()

        # Count how many operation names appear (each should be unique)
        op_count = sum(1 for i in range(_MAX_WARNINGS + 3) if f"op_{i}" in result)
        assert op_count <= _MAX_WARNINGS


# ---------------------------------------------------------------------------
# get_recent_error_context()
# ---------------------------------------------------------------------------


class TestGetRecentErrorContext:
    """get_recent_error_context() fetches and deduplicates errors for an operation."""

    def test_returns_error_messages_for_operation(self):
        """Error strings from recent failures are returned."""
        collector = FeedbackCollector()
        tracker = MagicMock()
        tracker.get_recent_failures.return_value = [
            {"error": "IK timeout", "success": False},
            {"error": "gripper stuck", "success": False},
        ]

        with patch(PATCH_TARGET, return_value=tracker):
            errors = collector.get_recent_error_context("grasp_object")

        assert "IK timeout" in errors
        assert "gripper stuck" in errors

    def test_deduplicates_repeated_errors(self):
        """Duplicate error messages appear only once."""
        collector = FeedbackCollector()
        tracker = MagicMock()
        tracker.get_recent_failures.return_value = [
            {"error": "timeout", "success": False},
            {"error": "timeout", "success": False},
            {"error": "timeout", "success": False},
        ]

        with patch(PATCH_TARGET, return_value=tracker):
            errors = collector.get_recent_error_context("op_a", limit=5)

        assert errors.count("timeout") == 1

    def test_limit_parameter_caps_output(self):
        """limit= parameter restricts the number of returned errors."""
        collector = FeedbackCollector()
        tracker = MagicMock()
        tracker.get_recent_failures.return_value = [
            {"error": f"err_{i}", "success": False} for i in range(10)
        ]

        with patch(PATCH_TARGET, return_value=tracker):
            errors = collector.get_recent_error_context("op_a", limit=2)

        assert len(errors) <= 2

    def test_returns_empty_list_when_tracker_unavailable(self):
        """If OutcomeTracker raises, get_recent_error_context returns []."""
        collector = FeedbackCollector()

        with patch(PATCH_TARGET, side_effect=Exception("unavailable")):
            errors = collector.get_recent_error_context("op_a")

        assert errors == []

    def test_none_errors_excluded(self):
        """Failures with error=None are not included in the result."""
        collector = FeedbackCollector()
        tracker = MagicMock()
        tracker.get_recent_failures.return_value = [
            {"error": None, "success": False},
            {"error": "real error", "success": False},
        ]

        with patch(PATCH_TARGET, return_value=tracker):
            errors = collector.get_recent_error_context("op_a")

        assert None not in errors
        assert "real error" in errors


# ---------------------------------------------------------------------------
# summarize_session_outcomes()
# ---------------------------------------------------------------------------


class TestSummarizeSessionOutcomes:
    """summarize_session_outcomes() returns a human-readable session summary."""

    def test_no_operations_returns_no_operations_message(self):
        """Empty tracker returns the 'No operations executed yet' message."""
        collector = FeedbackCollector()
        tracker = MagicMock()
        tracker.get_frequent_failure_patterns.return_value = []

        with patch(PATCH_TARGET, return_value=tracker):
            summary = collector.summarize_session_outcomes()

        assert "No operations" in summary

    def test_summary_contains_operation_name(self):
        """Operation names appear in the summary."""
        collector = FeedbackCollector()
        pattern = _make_pattern("grasp_object", 0.6, 15)
        tracker = MagicMock()
        tracker.get_frequent_failure_patterns.return_value = [pattern]

        with patch(PATCH_TARGET, return_value=tracker):
            summary = collector.summarize_session_outcomes()

        assert "grasp_object" in summary

    def test_summary_contains_execution_count(self):
        """Total execution count appears in the summary."""
        collector = FeedbackCollector()
        pattern = _make_pattern("op_a", 0.4, 25)
        tracker = MagicMock()
        tracker.get_frequent_failure_patterns.return_value = [pattern]

        with patch(PATCH_TARGET, return_value=tracker):
            summary = collector.summarize_session_outcomes()

        assert "25" in summary

    def test_summary_contains_failure_percentage(self):
        """Failure rate is expressed as an integer percentage in the summary."""
        collector = FeedbackCollector()
        pattern = _make_pattern("op_a", 0.4, 10)
        tracker = MagicMock()
        tracker.get_frequent_failure_patterns.return_value = [pattern]

        with patch(PATCH_TARGET, return_value=tracker):
            summary = collector.summarize_session_outcomes()

        assert "40%" in summary

    def test_returns_empty_string_when_tracker_unavailable(self):
        """If OutcomeTracker raises, summarize_session_outcomes returns empty string."""
        collector = FeedbackCollector()

        with patch(PATCH_TARGET, side_effect=Exception("unavailable")):
            summary = collector.summarize_session_outcomes()

        assert summary == ""


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
