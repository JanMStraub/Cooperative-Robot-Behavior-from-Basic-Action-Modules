#!/usr/bin/env python3
"""B1: Navigate to Object — detect blue object and grasp it."""

from __future__ import annotations

from ..config import BenchmarkConfig


def get_task(cfg: BenchmarkConfig) -> str:
    """
    Return natural language task description for B1.

    Args:
        cfg: Benchmark configuration.

    Returns:
        Task string sent to the LLM via SequenceServer.
    """
    return f"Robot {cfg.robot_id}: detect the blue object and grasp it."
