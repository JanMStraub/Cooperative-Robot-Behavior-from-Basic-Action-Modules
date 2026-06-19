#!/usr/bin/env python3
"""B5: Pose-Aware Grasp - detect red object, orient gripper downward, grasp it."""

from __future__ import annotations

EXPECTED_OP_CHAIN: list[str] = ["detect_object_stereo", "grasp_object"]


def get_task() -> str:
    """
    Return natural language task description for B5.
    """
    return "Robot1: Grasp the yellow cube."
