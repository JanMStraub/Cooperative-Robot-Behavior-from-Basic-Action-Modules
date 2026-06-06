#!/usr/bin/env python3
"""B4: Pick and Place — detect blue object, pick it up, place it at field A."""

from __future__ import annotations

EXPECTED_OP_CHAIN: list[str] = [
    "detect_object_stereo",
    "grasp_object",
    "detect_field",
    "place_object",
]


def get_task() -> str:
    """
    Return natural language task description for B4.
    """
    return "Robot2: Grasp magenta cube, and place it in field H."
