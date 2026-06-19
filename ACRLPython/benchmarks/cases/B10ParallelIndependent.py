#!/usr/bin/env python3
"""B10: Parallel Independent Tasks - dual-robot, zero shared objects, no sync point."""

from __future__ import annotations

EXPECTED_OP_CHAIN: list[str] = [
    "detect_object_stereo",
    "detect_object_stereo",
    "grasp_object",
    "grasp_object",
    "detect_field",
    "detect_field",
    "place_object",
    "place_object",
]


def get_task() -> str:
    """
    Return natural language task description for B16.

    Both robots perform a full pick-and-place on disjoint objects in disjoint
    fields.  No handoff, no shared grasp, no signal/wait pair.  The correct
    LLM response assigns Robot1 ops and Robot2 ops to the same parallel_group
    so both chains execute concurrently.
    """
    return (
        "Robot1 and Robot2 work independently. "
        "Robot1: detect the blue cube, grasp it, and place it in field A. "
        "Robot2: detect the green cube, grasp it, and place it in field B."
    )
