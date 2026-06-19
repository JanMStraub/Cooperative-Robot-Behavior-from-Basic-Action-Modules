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
        "you are currently holding the box - keep it stable while Robot2 manipulates it",
        "stabilize_object",
    ),
    (
        "you are currently holding the cube - deposit it at the shared handoff zone for Robot2 to pick up",
        "place_for_partner",
    ),
    (
        "take the object that Robot1 is offering",
        "receive_handoff",
    ),
    (
        "check whether your partner robot is ready before starting the joint task",
        "check_partner_status",
    ),
    (
        "perform a bimanual grasp of the beam with Robot2 from opposite sides",
        "synchronized_grasp",
    ),
    (
        "cooperatively carry the already grasped object with Robot2 to the drop zone",
        "joint_transport",
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
