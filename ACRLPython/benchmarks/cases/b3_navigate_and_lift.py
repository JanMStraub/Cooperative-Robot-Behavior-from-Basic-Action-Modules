#!/usr/bin/env python3
"""B3: Navigate and Lift — detect red object, approach from above, grasp it."""

from __future__ import annotations

from ..config import BenchmarkConfig


def get_task(cfg: BenchmarkConfig) -> str:
    """
    Return natural language task description for B3.

    Args:
        cfg: Benchmark configuration.

    Returns:
        Task string sent to the LLM via SequenceServer.
    """
    return (
        f"Robot {cfg.robot_id}: Grasp blue cube and lift it to y=0.1."
    )
