#!/usr/bin/env python3
"""B10: Reflexion Ablation — measure recovery rate with/without reflexion."""

from __future__ import annotations

from typing import List

from ..config import BenchmarkConfig


def get_tasks(cfg: BenchmarkConfig) -> List[str]:
    """
    Return NL navigation tasks for reflexion ablation.

    Tasks chosen because they exercise NAVIGATION category ops (move_to_coordinate,
    pick_object_at_coordinate) — the only category where reflexion applies.

    Args:
        cfg: Benchmark configuration.

    Returns:
        List of natural language task strings.
    """
    r = cfg.robot_id
    return [
        f"Robot {r}: Move to position (-0.1, 0.1, 0.1).",
        f"Robot {r}: Grasp the red cube and lift it to y=0.25.",
        f"Robot {r}: Grasp the green cube and place it at (-0.2, 0.0, -0.25).",
        f"Robot {r}: Move to position (-0.15, 0.15, 0.2), then move to (-0.1, 0.2, -0.2).",
        f"Robot {r}: Navigate to the blue cube and return to start.",
    ]
