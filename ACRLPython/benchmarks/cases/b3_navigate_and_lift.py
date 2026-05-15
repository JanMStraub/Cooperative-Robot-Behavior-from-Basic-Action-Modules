#!/usr/bin/env python3
"""B3: Navigate and Lift — detect red object, approach from above, grasp it."""

from __future__ import annotations

def get_task() -> str:
    """
    Return natural language task description for B3.

    Returns:
        Task string sent to the LLM via SequenceServer.
    """
    return f"Robot1: Grasp blue cube and lift it to y=0.2."
