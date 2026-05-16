#!/usr/bin/env python3
"""B8: Heterogeneous Chain — alternating dual-robot narrative sequence."""

from __future__ import annotations

from typing import List, Tuple

from ..config import BenchmarkConfig

# Each robot runs: navigate → grasp+lift → place, targeting its own field.
_ROBOT_CHAINS: dict[str, List[Tuple[str, str]]] = {
    "Robot1": [
        ("b1_navigate", "Detect the blue cube and move to it."),
        ("b3_lift", "The blue cube is in front of you. Grasp it and lift it to y=0.2."),
        ("b4_place", "You are holding the blue cube at y=0.2. Place it in field B."),
    ],
    "Robot2": [
        ("b1_navigate", "Detect the blue cube and move to it."),
        ("b3_lift", "The blue cube is in front of you. Grasp it and lift it to y=0.2."),
        ("b4_place", "You are holding the blue cube at y=0.2. Place it in field H."),
    ],
}

_ROBOT_ORDER = ["Robot1", "Robot2"]


def get_sub_tasks(cfg: BenchmarkConfig, task_count: int) -> List[Tuple[str, str]]:
    """
    Return an alternating dual-robot narrative chain where each step assumes the prior succeeded.

    Robot1 and Robot2 alternate full chains (navigate → grasp+lift → place).
    Robot1 targets field B; Robot2 targets field H.
    Cycles indefinitely up to task_count entries.


    """
    result: List[Tuple[str, str]] = []
    chain_len = len(next(iter(_ROBOT_CHAINS.values())))
    for i in range(task_count):
        robot = _ROBOT_ORDER[(i // chain_len) % len(_ROBOT_ORDER)]
        step = i % chain_len
        name, prompt = _ROBOT_CHAINS[robot][step]
        result.append((f"{robot}_{name}_{i}", f"{robot}: {prompt}"))
    return result
