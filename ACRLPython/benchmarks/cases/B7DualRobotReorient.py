#!/usr/bin/env python3
"""B7: Dual-Robot Reorient — B6 plus second sync barrier and parallel arm swings."""

from __future__ import annotations


def get_task() -> str:
    """
    Return natural language task description for B7.

    """
    return f"Robot1 and Robot2 work together to grasp the red cube. Robot1 should grasp on one side and Robot2  on the other side and they should lift the cube together to y=0.1."
