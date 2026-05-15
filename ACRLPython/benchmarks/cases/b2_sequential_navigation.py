#!/usr/bin/env python3
"""B2: Sequential Navigation — detect and navigate to three cubes by color."""

from __future__ import annotations

def get_task() -> str:
    """
    Return natural language task description for B2.

    Returns:
        Task string sent to the LLM via SequenceServer.
    """
    return f"Robot1: detect the red cube and move to it, then detect the yellow cube and move to it, then detect the purple cube and move to it."
