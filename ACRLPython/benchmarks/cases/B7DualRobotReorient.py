#!/usr/bin/env python3
"""B7: Dual-Robot Reorient — B6 plus second sync barrier and parallel arm swings."""

from __future__ import annotations


def get_task() -> str:
    """
    Return natural language task description for B7.

    """
    return (
        "Robot1 and Robot2 cooperatively handle the red cube. First, Robot2 grasps the right side of the cube. After Robot2 is in position, Robot1 grasps the cube from the left side Once Robot1 has secured the cube, both robots simultaneously lift it to y=0.15."
    )
