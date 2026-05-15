#!/usr/bin/env python3
"""B4: Pick and Place — detect blue object, pick it up, place it at field A."""

from __future__ import annotations

def get_task() -> str:
    """
    Return natural language task description for B4.

    Returns:
        Task string sent to the LLM via SequenceServer.
    """
    return f"Robot1: Grasp blue cube, and place it in field A."
