#!/usr/bin/env python3
"""B8: Heterogeneous Chain - rotating cube colors, field targets, and three phase types per cycle."""

from __future__ import annotations

from typing import List, Tuple

from ..Config import BenchmarkConfig

EXPECTED_OP_CHAIN_PHASE_ABC: list[str] = [
    "detect_object_stereo",
    "grasp_object",
    "detect_field",
    "place_object",
    "return_to_start_position",
]
EXPECTED_OP_CHAIN_PHASE_D: list[str] = ["detect_all_fields", "detect_object_stereo"]


def _phase_a() -> str:
    return "Robot1: Grasp the blue cube and place it on field h, then return to start position."


def _phase_b() -> str:
    return "Robot2: Grasp the blue cube and place it on field b, then return to start position."


def _phase_c() -> str:
    return "Robot1: Grasp the blue cube and place it on field e, then return to start position."


def _phase_d() -> str:
    return (
        "Robot1 and Robot2: Survey the scene. "
        "Robot1 should detect all visible fields. "
        "Robot2 should detect the yellow cube and report its current position."
    )


def get_sub_tasks(_cfg: BenchmarkConfig, task_count: int) -> List[Tuple[str, str, str]]:
    """
    Return task_count full cycles; each cycle has 4 sub-tasks (phases A, B, C, D).
    Total sub-tasks returned = task_count * 4.
    """
    result: List[Tuple[str, str, str]] = []
    for i in range(task_count):
        result.append(("Robot1", f"cycle_{i}_phase_a", _phase_a()))
        result.append(("Robot2", f"cycle_{i}_phase_b", _phase_b()))
        result.append(("Robot1", f"cycle_{i}_phase_c", _phase_c()))
        result.append(("Robot1", f"cycle_{i}_phase_d", _phase_d()))
    return result
