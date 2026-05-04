#!/usr/bin/env python3
"""B1: Navigate to Object — detect blue object and move to it."""

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
    return f"Robot {cfg.robot_id}: Detect the blue cube and move to it."
