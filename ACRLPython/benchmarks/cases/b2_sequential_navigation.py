#!/usr/bin/env python3
"""B2: Sequential Navigation — detect and navigate to three objects by color."""

from __future__ import annotations

from ..config import BenchmarkConfig


def get_task(cfg: BenchmarkConfig) -> str:
    """
    Return natural language task description for B2.

    Args:
        cfg: Benchmark configuration.

    Returns:
        Task string sent to the LLM via SequenceServer.
    """
    return (
        f"Robot {cfg.robot_id}: detect the red object and move to it, "
        f"then detect the yellow object and move to it, "
        f"then detect the purple object and move to it."
    )
