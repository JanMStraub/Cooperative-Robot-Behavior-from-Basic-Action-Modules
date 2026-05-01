#!/usr/bin/env python3
"""B8: Heterogeneous Chain — cycles through B1/B3/B4 natural language sub-tasks."""

from __future__ import annotations

from itertools import cycle
from typing import List, Tuple

from ..config import BenchmarkConfig
from . import b1_navigate_to_object, b3_navigate_and_lift, b4_pick_and_place

_TASK_BUILDERS = [
    ("b1_navigate", b1_navigate_to_object.get_task),
    ("b3_lift", b3_navigate_and_lift.get_task),
    ("b4_pick_place", b4_pick_and_place.get_task),
]


def get_sub_tasks(
    cfg: BenchmarkConfig, task_count: int
) -> List[Tuple[str, str]]:
    """
    Generate a heterogeneous list of (name, task_string) tuples.

    Cycles through B1/B3/B4 patterns up to task_count entries.

    Args:
        cfg: Benchmark configuration passed to each sub-task builder.
        task_count: Total number of sub-tasks to generate.

    Returns:
        List of (task_name, task_string) tuples for BenchmarkRunner._run_b8_chain.
    """
    tasks: List[Tuple[str, str]] = []
    for i, (name, builder) in enumerate(cycle(_TASK_BUILDERS)):
        if i >= task_count:
            break
        tasks.append((f"{name}_{i}", builder(cfg)))
    return tasks
