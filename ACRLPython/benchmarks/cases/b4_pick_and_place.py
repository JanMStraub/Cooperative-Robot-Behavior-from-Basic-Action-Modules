#!/usr/bin/env python3
"""B4: Pick and Place — detect red object, pick it up, place it at (0.2, 0.15, 0.1)."""

from __future__ import annotations

from ..config import BenchmarkConfig


def get_task(cfg: BenchmarkConfig) -> str:
    """
    Return natural language task description for B4.

    Args:
        cfg: Benchmark configuration.

    Returns:
        Task string sent to the LLM via SequenceServer.
    """
    return (
        f"Robot {cfg.robot_id}: detect the red object, pick it up, "
        f"then place it at coordinates x=0.2, y=0.15, z=0.1."
    )
