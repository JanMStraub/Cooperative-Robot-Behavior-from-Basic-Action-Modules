#!/usr/bin/env python3
"""B6: Dual-Robot Handoff — Robot A grasps red cube, hands it off to Robot B."""

from __future__ import annotations

def get_task() -> str:
    """
    Return natural language task description for B6.

    Returns:
        Task string sent to the LLM via SequenceServer.
    """
    return f"Robot1 and Robot2 perform a handoff of the red cube"
