#!/usr/bin/env python3
"""B11: RAG Ablation - operation selection accuracy with and without RAG retrieval.

Each task is ambiguous: multiple operations could plausibly apply, but only one
is correct given the system's semantics. RAG provides workflow context and richer
descriptions that should guide the LLM toward the ground-truth operation.
"""

from __future__ import annotations

from typing import List, Tuple

# (task_string, ground_truth_operation)
# Ground truth is the primary (first) operation the parser should select.
TASKS: List[Tuple[str, str]] = [
    (
        "request access to Robot2's workspace region and wait until Robot2 has cleared it",
        "yield_workspace",
    ),
    (
        "the red cube is sitting on the table pick it up",
        "grasp_object",
    ),
    (
        "put the block down exactly halfway between the blue block and the green block",
        "place_between_objects",
    ),
    (
        "take the object that Robot1 is offering",
        "receive_handoff",
    ),
    (
        "before grasping, first check that you yourself aren't already holding something",
        "check_robot_status",
    ),
    (
        "map out every zone on the table before deciding where to place the object",
        "detect_all_fields",
    ),
    (
        "without moving from your current position, angle your gripper straight down over the object",
        "adjust_end_effector_orientation",
    ),
    (
        "do not move until you receive the go signal from Robot1",
        "wait_for_signal",
    ),
]


def get_tasks(config=None) -> List[str]:
    return [task for task, _ in TASKS]


def get_ground_truth(config=None) -> List[str]:
    return [gt for _, gt in TASKS]
