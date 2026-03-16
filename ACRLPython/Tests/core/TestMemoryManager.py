#!/usr/bin/env python3
"""
Tests for MemoryManager
========================

Covers:
- Singleton: two MemoryManager() calls return the same object
- read_memory(): returns "" when no file exists
- write_memory(): produces correct Markdown sections (Lessons, Errors, Stats)
- write_memory(): atomic write — no partial file visible (temp+rename pattern)
- clear_memory(): removes the file
- Lessons section reflects OutcomeTracker failure rates
- Recent Errors section filters to the correct robot_id
- Session Stats reflect per-robot outcome counts
- MEMORY_MAX_LESSONS / MEMORY_MAX_ERRORS limits respected
"""

import os
import threading
import time
import pytest

from core.MemoryManager import MemoryManager, get_memory_manager
from orchestrators.OutcomeTracker import OutcomeTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_memory_manager(tmp_path):
    """
    Reset the MemoryManager singleton before each test and redirect
    the memory directory to a temporary path to avoid touching the filesystem.
    """
    # Reset singleton
    with MemoryManager._lock:
        MemoryManager._instance = None

    mgr = MemoryManager()
    mgr._memory_dir = str(tmp_path / "memory")
    yield mgr

    # Cleanup: reset again so subsequent tests start fresh
    with MemoryManager._lock:
        MemoryManager._instance = None


@pytest.fixture()
def fresh_tracker():
    """Return a clean OutcomeTracker singleton with an empty ring buffer."""
    # Reset the singleton
    with OutcomeTracker._lock:
        OutcomeTracker._instance = None
    tracker = OutcomeTracker()
    yield tracker
    # Cleanup
    tracker.reset()
    with OutcomeTracker._lock:
        OutcomeTracker._instance = None


# ---------------------------------------------------------------------------
# Singleton behaviour
# ---------------------------------------------------------------------------


def test_singleton_returns_same_instance(reset_memory_manager):
    """Two MemoryManager() calls must return the same object."""
    a = MemoryManager()
    b = MemoryManager()
    assert a is b


def test_get_memory_manager_returns_singleton(reset_memory_manager):
    """get_memory_manager() must return the same singleton as MemoryManager()."""
    assert get_memory_manager() is MemoryManager()


# ---------------------------------------------------------------------------
# read_memory — missing file
# ---------------------------------------------------------------------------


def test_read_memory_returns_empty_string_when_no_file(reset_memory_manager):
    """read_memory() must return '' when the file does not exist."""
    result = reset_memory_manager.read_memory("Robot1")
    assert result == ""


# ---------------------------------------------------------------------------
# write_memory / read_memory round-trip
# ---------------------------------------------------------------------------


def test_write_creates_file(reset_memory_manager, fresh_tracker):
    """write_memory() must create the memory file on disk."""
    reset_memory_manager.write_memory("Robot1", fresh_tracker)
    path = reset_memory_manager._memory_path("Robot1")
    assert os.path.exists(path)


def test_read_returns_written_content(reset_memory_manager, fresh_tracker):
    """read_memory() must return the content written by write_memory()."""
    fresh_tracker.record("move_to_coordinate", "Robot1", success=True)
    reset_memory_manager.write_memory("Robot1", fresh_tracker)
    content = reset_memory_manager.read_memory("Robot1")
    assert "## Lessons Learned" in content
    assert "## Recent Errors" in content
    assert "## Session Stats" in content


# ---------------------------------------------------------------------------
# Lessons section
# ---------------------------------------------------------------------------


def test_lessons_reflect_failure_rate(reset_memory_manager, fresh_tracker):
    """
    Operations with failures must appear in Lessons with a non-zero failure rate.
    """
    for _ in range(6):
        fresh_tracker.record("grasp_object", "Robot1", success=False, error="IK failed")
    for _ in range(4):
        fresh_tracker.record("grasp_object", "Robot1", success=True)

    reset_memory_manager.write_memory("Robot1", fresh_tracker)
    content = reset_memory_manager.read_memory("Robot1")

    assert "grasp_object" in content
    assert "60% failure rate" in content
    assert "IK failed" in content


def test_lessons_shows_zero_failure_for_successful_op(
    reset_memory_manager, fresh_tracker
):
    """Operations with no failures must show 0% failure rate."""
    fresh_tracker.record("move_to_coordinate", "Robot1", success=True)
    reset_memory_manager.write_memory("Robot1", fresh_tracker)
    content = reset_memory_manager.read_memory("Robot1")

    assert "move_to_coordinate" in content
    assert "0% failure rate" in content


def test_lessons_no_operations_shows_placeholder(reset_memory_manager, fresh_tracker):
    """Empty tracker must show 'No operations recorded yet.' in Lessons."""
    reset_memory_manager.write_memory("Robot1", fresh_tracker)
    content = reset_memory_manager.read_memory("Robot1")
    assert "No operations recorded yet." in content


def test_lessons_max_lessons_limit(reset_memory_manager, fresh_tracker):
    """
    Lessons section must not exceed MEMORY_MAX_LESSONS entries.
    """
    from config.Memory import MEMORY_MAX_LESSONS

    # Record more operations than the limit
    for i in range(MEMORY_MAX_LESSONS + 5):
        fresh_tracker.record(f"op_{i}", "Robot1", success=True)

    reset_memory_manager.write_memory("Robot1", fresh_tracker)
    content = reset_memory_manager.read_memory("Robot1")

    # Count lesson lines (lines starting with "- op_")
    lesson_lines = [l for l in content.splitlines() if l.startswith("- op_")]
    assert len(lesson_lines) <= MEMORY_MAX_LESSONS


# ---------------------------------------------------------------------------
# Recent Errors section
# ---------------------------------------------------------------------------


def test_recent_errors_filters_to_robot(reset_memory_manager, fresh_tracker):
    """
    Recent Errors section for Robot1 must not include errors from Robot2.

    Note: Lessons Learned is cross-robot (global operation stats), so R2 error
    may appear there. We assert only on the Recent Errors section specifically.
    """
    fresh_tracker.record(
        "grasp_object", "Robot1", success=False, error="R1 unique error"
    )
    fresh_tracker.record(
        "move_to_coordinate", "Robot2", success=False, error="R2 unique error"
    )

    reset_memory_manager.write_memory("Robot1", fresh_tracker)
    content = reset_memory_manager.read_memory("Robot1")

    # Extract only the Recent Errors section
    errors_start = content.find("## Recent Errors")
    stats_start = content.find("## Session Stats")
    errors_section = (
        content[errors_start:stats_start]
        if errors_start != -1 and stats_start != -1
        else ""
    )

    assert "R1 unique error" in errors_section
    assert "R2 unique error" not in errors_section


def test_recent_errors_no_failures_shows_placeholder(
    reset_memory_manager, fresh_tracker
):
    """No failures for the robot must show 'No failures recorded.' in Errors."""
    fresh_tracker.record("move_to_coordinate", "Robot1", success=True)
    reset_memory_manager.write_memory("Robot1", fresh_tracker)
    content = reset_memory_manager.read_memory("Robot1")
    assert "No failures recorded." in content


def test_recent_errors_max_errors_limit(reset_memory_manager, fresh_tracker):
    """Recent Errors section must not exceed MEMORY_MAX_ERRORS entries."""
    from config.Memory import MEMORY_MAX_ERRORS

    for i in range(MEMORY_MAX_ERRORS + 5):
        fresh_tracker.record("grasp_object", "Robot1", success=False, error=f"err_{i}")

    reset_memory_manager.write_memory("Robot1", fresh_tracker)
    content = reset_memory_manager.read_memory("Robot1")

    # Count error lines (lines starting with "- [")
    error_lines = [l for l in content.splitlines() if l.startswith("- [")]
    assert len(error_lines) <= MEMORY_MAX_ERRORS


# ---------------------------------------------------------------------------
# Session Stats
# ---------------------------------------------------------------------------


def test_session_stats_correct_counts(reset_memory_manager, fresh_tracker):
    """Session Stats must reflect per-robot success rate accurately."""
    fresh_tracker.record("op_a", "Robot1", success=True)
    fresh_tracker.record("op_b", "Robot1", success=True)
    fresh_tracker.record("op_c", "Robot1", success=False)

    reset_memory_manager.write_memory("Robot1", fresh_tracker)
    content = reset_memory_manager.read_memory("Robot1")

    assert "Operations: 3" in content
    assert "Success rate: 66%" in content


def test_session_stats_filters_to_robot(reset_memory_manager, fresh_tracker):
    """Session Stats must count only the target robot's outcomes."""
    fresh_tracker.record("op_a", "Robot1", success=True)
    fresh_tracker.record("op_b", "Robot2", success=True)
    fresh_tracker.record("op_c", "Robot2", success=True)

    reset_memory_manager.write_memory("Robot1", fresh_tracker)
    content = reset_memory_manager.read_memory("Robot1")

    # Only Robot1 outcomes should count
    assert "Operations: 1" in content
    assert "Success rate: 100%" in content


# ---------------------------------------------------------------------------
# clear_memory
# ---------------------------------------------------------------------------


def test_clear_memory_removes_file(reset_memory_manager, fresh_tracker):
    """clear_memory() must delete the file if it exists."""
    reset_memory_manager.write_memory("Robot1", fresh_tracker)
    assert os.path.exists(reset_memory_manager._memory_path("Robot1"))

    reset_memory_manager.clear_memory("Robot1")
    assert not os.path.exists(reset_memory_manager._memory_path("Robot1"))


def test_clear_memory_no_error_when_file_missing(reset_memory_manager):
    """clear_memory() must not raise if the file does not exist."""
    reset_memory_manager.clear_memory("NonExistentRobot")  # should not raise


# ---------------------------------------------------------------------------
# Atomic write (no partial reads)
# ---------------------------------------------------------------------------


def test_atomic_write_no_partial_file(reset_memory_manager, fresh_tracker, tmp_path):
    """
    Simulate a concurrent read during write: the reader must either see the
    old complete file or the new complete file — never a partial one.

    We verify this by checking the file at the path never exists as a .tmp
    file from the caller's perspective (os.replace is atomic on POSIX).
    """
    memory_dir = str(tmp_path / "memory")
    reset_memory_manager._memory_dir = memory_dir
    os.makedirs(memory_dir, exist_ok=True)

    # Pre-populate so we can read during the write
    fresh_tracker.record("op_a", "Robot1", success=True)
    reset_memory_manager.write_memory("Robot1", fresh_tracker)

    errors = []

    def reader():
        for _ in range(50):
            try:
                content = reset_memory_manager.read_memory("Robot1")
                # Content must either be empty or contain valid sections
                if content:
                    assert "## Session Stats" in content, f"Partial file: {content!r}"
            except Exception as e:
                errors.append(e)
            time.sleep(0.001)

    def writer():
        for i in range(20):
            fresh_tracker.record(f"op_{i}", "Robot1", success=(i % 2 == 0))
            reset_memory_manager.write_memory("Robot1", fresh_tracker)
            time.sleep(0.002)

    r = threading.Thread(target=reader)
    w = threading.Thread(target=writer)
    r.start()
    w.start()
    r.join()
    w.join()

    assert not errors, f"Concurrent read/write errors: {errors}"


# ---------------------------------------------------------------------------
# Per-robot file isolation
# ---------------------------------------------------------------------------


def test_separate_files_per_robot(reset_memory_manager, fresh_tracker):
    """
    write_memory() must create separate files for different robots.

    Recent Errors sections must be robot-specific; Lessons Learned is
    cross-robot (global stats) so we only check the errors section.
    """
    fresh_tracker.record(
        "grasp_object", "Robot1", success=False, error="R1-specific-error"
    )
    fresh_tracker.record(
        "move_to_coordinate", "Robot2", success=False, error="R2-specific-error"
    )

    reset_memory_manager.write_memory("Robot1", fresh_tracker)
    reset_memory_manager.write_memory("Robot2", fresh_tracker)

    def errors_section(content: str) -> str:
        start = content.find("## Recent Errors")
        end = content.find("## Session Stats")
        return content[start:end] if start != -1 and end != -1 else ""

    errors1 = errors_section(reset_memory_manager.read_memory("Robot1"))
    errors2 = errors_section(reset_memory_manager.read_memory("Robot2"))

    # Robot1 recent errors must only have Robot1 errors
    assert "R1-specific-error" in errors1
    assert "R2-specific-error" not in errors1

    # Robot2 recent errors must only have Robot2 errors
    assert "R2-specific-error" in errors2
    assert "R1-specific-error" not in errors2
