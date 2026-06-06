#!/usr/bin/env python3
"""B1: Navigate to Object — detect blue cube and move to it."""

from __future__ import annotations

EXPECTED_OP_CHAIN: list[str] = ["detect_object_stereo", "move_to_coordinate"]


def get_task() -> str:
    """
    Return natural language task description for B1.
    """
    return "Robot1: Detect the blue cube and move to it."
