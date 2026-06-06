#!/usr/bin/env python3
"""B3: Navigate and Lift — detect red object, approach from above, grasp it."""

from __future__ import annotations

EXPECTED_OP_CHAIN: list[str] = [
    "detect_object_stereo",
    "grasp_object",
    "move_to_coordinate",
]


def get_task() -> str:
    """
    Return natural language task description for B3.
    """
    return "Robot1: Grasp the blue cube, and lift it to y=0.2."
