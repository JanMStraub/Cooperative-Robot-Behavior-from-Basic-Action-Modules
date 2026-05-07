#!/usr/bin/env python3
"""B13: VGN Ablation — neural grasp prediction vs geometric fallback."""

from __future__ import annotations

from typing import List

from ..config import BenchmarkConfig


def get_tasks(cfg: BenchmarkConfig) -> List[str]:
    """
    Return NL grasp tasks for VGN vs geometric-fallback ablation.

    With VGN enabled, grasp_object calls VGNClient which runs TSDF+VGN inference
    and returns 6-DOF poses.  With VGN disabled, grasp_object uses the geometric
    top-down fallback.  Live mode reveals success-rate and timing differences;
    offline mode verifies the command-parse path works for both conditions.

    Args:
        cfg: Benchmark configuration.

    Returns:
        List of natural language task strings.
    """
    return [
        f"Robot {cfg.robot_id}: Grasp red cube.",
        f"Robot {cfg.robot_id}: Grasp the blue cube and lift it to y=0.3.",
        f"Robot {cfg.robot_id}: Grasp the yellow cube and place it on field H.",
        f"Robot {cfg.robot_id}: Detect the nearest object and grasp it.",
    ]
