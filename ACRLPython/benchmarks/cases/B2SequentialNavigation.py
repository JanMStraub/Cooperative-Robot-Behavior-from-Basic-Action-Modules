#!/usr/bin/env python3
"""B2: Sequential Navigation - detect and navigate to three cubes by color."""

from __future__ import annotations

EXPECTED_OP_CHAIN: list[str] = [
    "detect_object_stereo",
    "move_to_coordinate",
    "detect_object_stereo",
    "move_to_coordinate",
]


def get_task() -> str:
    """
    Return natural language task description for B2.
    """
    return "Robot1: Detect the red cube and move to it, then detect the yellow cube and move to it."
