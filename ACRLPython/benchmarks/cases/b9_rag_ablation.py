#!/usr/bin/env python3
"""B9: RAG Ablation — parse-only comparison with and without RAG retrieval."""

from __future__ import annotations

from typing import List

from ..config import BenchmarkConfig


def get_tasks(cfg: BenchmarkConfig) -> List[str]:
    """
    Return NL task strings reused from B1–B5 for parse-only RAG ablation.

    Args:
        cfg: Benchmark configuration (robot_id used to format tasks).

    Returns:
        List of natural language task strings.
    """
    r = cfg.robot_id
    return [
        f"Robot {cfg.robot_id}: Detect the blue cube and move to it.",
        f"Robot {cfg.robot_id}: Move to position (0.0, 0.15, 0.2), then move to (-0.1, 0.2, -0.2).",
        f"Robot {cfg.robot_id}: Grasp the red cube and lift it to y=0.3.",
        f"Robot {cfg.robot_id}: Grasp the yellow cube and place it on field C.",
    ]
