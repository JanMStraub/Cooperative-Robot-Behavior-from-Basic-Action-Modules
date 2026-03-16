#!/usr/bin/env python3
"""
Memory Manager
==============

Reads and writes structured per-robot Markdown memory files that persist
operation outcomes across sessions.

**Design principle**: Python writes the memory deterministically from
OutcomeTracker data — the LLM only reads it as prompt context.
This avoids the unreliability of LLM-written memory (hallucination,
malformed JSON, irrelevant content).

File layout (one file per robot):
    {MEMORY_DIR}/memory_{robot_id}.md

Example content:
    ## Lessons Learned
    - grasp_object: 45% failure rate (12 executions). Common error: "IK solver failed"
    - move_to_coordinate: 0% failure rate (8 executions)

    ## Recent Errors (last session)
    - [grasp_object] IK solver failed — target out of reach

    ## Session Stats
    Last updated: 2026-03-10 14:23:00 | Operations: 20 | Success rate: 75%

Usage:
    from core.MemoryManager import get_memory_manager
    mgr = get_memory_manager()
    mgr.write_memory("Robot1", outcome_tracker_instance)
    text = mgr.read_memory("Robot1")   # returns formatted string for prompt
"""

import logging
import os
import tempfile
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from config.Memory import MEMORY_DIR, MEMORY_MAX_ERRORS, MEMORY_MAX_LESSONS

if TYPE_CHECKING:
    from orchestrators.OutcomeTracker import OutcomeTracker

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Singleton manager for per-robot Markdown memory files.

    Writes structured memory from OutcomeTracker data and exposes it
    as formatted text for injection into LLM prompts.

    Thread-safe: file writes use atomic rename via tempfile.
    """

    _instance: Optional["MemoryManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "MemoryManager":
        """Return the singleton instance, creating it on first call."""
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize the manager (runs once due to singleton guard)."""
        if self._initialized:
            return
        self._initialized = True
        self._memory_dir = MEMORY_DIR
        logger.info(f"MemoryManager initialized (dir={self._memory_dir})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_memory(self, robot_id: str, outcome_tracker: "OutcomeTracker") -> None:
        """
        Write or overwrite the memory file for a robot.

        Queries OutcomeTracker for failure patterns and recent errors,
        then formats them as a Markdown file. Writes atomically via
        tempfile + rename to prevent partial reads.

        Args:
            robot_id: Robot identifier (e.g. "Robot1").
            outcome_tracker: OutcomeTracker instance to query for stats.
        """
        try:
            os.makedirs(self._memory_dir, exist_ok=True)
            content = self._format_memory(robot_id, outcome_tracker)
            path = self._memory_path(robot_id)
            self._atomic_write(path, content)
            logger.debug(f"[{robot_id}] Memory written to {path}")
        except Exception as e:
            logger.warning(f"[{robot_id}] Failed to write memory: {e}")

    def read_memory(self, robot_id: str) -> str:
        """
        Read the memory file for a robot and return it as a string.

        Returns an empty string if no memory file exists yet (first session).

        Args:
            robot_id: Robot identifier (e.g. "Robot1").

        Returns:
            Memory file contents as a string, or "" if not found.
        """
        path = self._memory_path(robot_id)
        if not os.path.exists(path):
            logger.debug(f"[{robot_id}] No memory file found at {path}")
            return ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"[{robot_id}] Failed to read memory: {e}")
            return ""

    def clear_memory(self, robot_id: str) -> None:
        """
        Delete the memory file for a robot.

        Primarily used in tests to ensure a clean state between runs.

        Args:
            robot_id: Robot identifier (e.g. "Robot1").
        """
        path = self._memory_path(robot_id)
        if os.path.exists(path):
            try:
                os.remove(path)
                logger.debug(f"[{robot_id}] Memory file deleted: {path}")
            except Exception as e:
                logger.warning(f"[{robot_id}] Failed to clear memory: {e}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _memory_path(self, robot_id: str) -> str:
        """
        Return the absolute path for a robot's memory file.

        Args:
            robot_id: Robot identifier.

        Returns:
            Absolute file path string.
        """
        return os.path.join(self._memory_dir, f"memory_{robot_id}.md")

    def _format_memory(self, robot_id: str, outcome_tracker: "OutcomeTracker") -> str:
        """
        Build the Markdown memory string from OutcomeTracker data.

        Sections:
        - Lessons Learned: per-operation failure rates (sorted by rate)
        - Recent Errors: last N failure messages for this robot
        - Session Stats: total ops and success rate

        Args:
            robot_id: Robot identifier (used to filter per-robot errors).
            outcome_tracker: Source of operation statistics.

        Returns:
            Formatted Markdown string.
        """
        lines = []

        # --- Lessons Learned ---
        patterns = outcome_tracker.get_frequent_failure_patterns(
            min_failure_rate=0.0,  # Include all operations, not just high-failure ones
            min_occurrences=1,
        )
        lines.append("## Lessons Learned")
        if patterns:
            for p in patterns[:MEMORY_MAX_LESSONS]:
                rate_pct = int(p["failure_rate"] * 100)
                count = p["total_executions"]
                errors = p.get("sample_errors", [])
                line = f"- {p['operation_name']}: {rate_pct}% failure rate ({count} executions)"
                if errors:
                    line += f'. Common error: "{errors[0]}"'
                lines.append(line)
        else:
            lines.append("- No operations recorded yet.")

        lines.append("")

        # --- Recent Errors (filtered to this robot) ---
        all_failures = outcome_tracker.get_recent_failures(limit=MEMORY_MAX_ERRORS * 3)
        robot_failures = [f for f in all_failures if f.get("robot_id") == robot_id][
            :MEMORY_MAX_ERRORS
        ]

        lines.append("## Recent Errors (last session)")
        if robot_failures:
            for f in robot_failures:
                op = f.get("operation_name", "unknown")
                err = f.get("error") or "no details"
                lines.append(f"- [{op}] {err}")
        else:
            lines.append("- No failures recorded.")

        lines.append("")

        # --- Session Stats ---
        recent = outcome_tracker._recent_outcomes  # Access ring buffer directly
        # Filter to this robot's outcomes
        robot_outcomes = [o for o in recent if o.get("robot_id") == robot_id]
        total = len(robot_outcomes)
        successes = sum(1 for o in robot_outcomes if o.get("success"))
        rate = int((successes / total) * 100) if total > 0 else 0
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines.append("## Session Stats")
        lines.append(
            f"Last updated: {now} | Operations: {total} | Success rate: {rate}%"
        )

        return "\n".join(lines) + "\n"

    def _atomic_write(self, path: str, content: str) -> None:
        """
        Write content to path atomically using a temporary file + rename.

        Prevents readers from seeing partial file contents during writes.

        Args:
            path: Target file path.
            content: Text content to write.
        """
        dir_name = os.path.dirname(path)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=dir_name,
            delete=False,
            suffix=".tmp",
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------


def get_memory_manager() -> MemoryManager:
    """
    Return the global MemoryManager singleton.

    Returns:
        The global MemoryManager instance.
    """
    return MemoryManager()
