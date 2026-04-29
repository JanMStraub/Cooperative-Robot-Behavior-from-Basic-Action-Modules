#!/usr/bin/env python3
"""B2: Sequential Navigation — detect and navigate to three objects by color."""

from __future__ import annotations

from typing import Any, Dict, List

from ..config import BenchmarkConfig

_COLORS = ["red", "green", "blue"]


def build_sequence(cfg: BenchmarkConfig) -> List[Dict[str, Any]]:
    """
    Build 2×N-step sequence: for each color, detect then move.

    Args:
        cfg: Benchmark configuration.

    Returns:
        List of command dicts for SequenceExecutor (6 steps for default 3 colors).
    """
    steps: List[Dict[str, Any]] = []
    for color in _COLORS:
        var = f"target_{color}"
        steps.append(
            {
                "operation": "detect_object_stereo",
                "params": {"robot_id": cfg.robot_id, "color": color},
                "capture_var": var,
            }
        )
        steps.append(
            {
                "operation": "move_to_coordinate",
                "params": {
                    "robot_id": cfg.robot_id,
                    "x": f"${var}.x",
                    "y": f"${var}.y",
                    "z": f"${var}.z",
                },
            }
        )
    return steps
