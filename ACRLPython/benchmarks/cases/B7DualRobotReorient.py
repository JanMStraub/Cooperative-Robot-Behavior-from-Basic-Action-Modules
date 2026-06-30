#!/usr/bin/env python3
"""B7: Dual-Robot Reorient - B6 plus second sync barrier and parallel arm swings."""

from __future__ import annotations

EXPECTED_OP_CHAIN: list[str] = [
    "detect_object_stereo",
    "grasp_object",
    "signal",
    "wait_for_signal",
    "grasp_object",
    "signal",
    "wait_for_signal",
    "move_to_coordinate",
    "move_to_coordinate",
]


def get_task() -> str:
    """
    Return natural language task description for B7.
    """
    return "Robot1 and Robot2 cooperatively handle the red cube. First, Robot1 grasps the right side of the cube. After Robot1 is in position, Robot2 grasps the cube from the left side. Once Robot2 has secured the cube, both robots simultaneously lift it to y=0.15."
