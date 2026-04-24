#!/usr/bin/env python3
"""B1: Navigate to Object — detect red object, move end-effector to it."""

from __future__ import annotations

from typing import Any, Dict, List

from ..config import BenchmarkConfig


def build_sequence(cfg: BenchmarkConfig) -> List[Dict[str, Any]]:
    """
    Build 2-step sequence: detect object then navigate to it.

    Args:
        cfg: Benchmark configuration.

    Returns:
        List of command dicts for SequenceExecutor.
    """
    return [
        {
            "operation": "detect_object_stereo",
            "params": {"robot_id": cfg.robot_id, "color": "red"},
            "capture_var": "target",
        },
        {
            "operation": "move_to_coordinate",
            "params": {
                "robot_id": cfg.robot_id,
                "x": "$target.x",
                "y": "$target.y",
                "z": "$target.z",
            },
        },
    ]
