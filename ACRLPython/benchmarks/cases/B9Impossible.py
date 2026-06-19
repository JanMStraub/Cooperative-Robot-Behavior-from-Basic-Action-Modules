#!/usr/bin/env python3
"""B9: Solve impossible task - verify parser rejects unexecutable requests."""

from __future__ import annotations

# Exposed for test assertions
INVALID_ROBOT = "Robot3"
INVALID_OBJECT = "tree"


def get_task() -> str:
    """
    Return natural language task description for B9.

    Robot3 does not exist in the system; 'tree' is not a graspable object.
    Both failure modes should cause the parser to reject the task.
    """
    return "Robot3: Move the tree to the center."
