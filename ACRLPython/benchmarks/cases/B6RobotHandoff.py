#!/usr/bin/env python3
"""B6: Dual-Robot Handoff — Robot A grasps red cube, hands it off to Robot B."""

from __future__ import annotations

EXPECTED_OP_CHAIN: list[str] = [
    "detect_object_stereo",
    "grasp_object",
    "move_to_coordinate",
    "adjust_end_effector_orientation",
    "signal",
    "wait_for_signal",
    "detect_object_stereo",
    "receive_handoff",
]

OPTIONAL_OPS: list[str] = [
    "release_object",
    "return_to_start_position",
    "control_gripper",
]

OPTIONAL_SUFFIX_OPS: list[str] = [
    "signal",
]


def get_task() -> str:
    """
    Return natural language task description for B6.
    """
    return "Robot1 grasps the red cube and hands it to Robot2."
