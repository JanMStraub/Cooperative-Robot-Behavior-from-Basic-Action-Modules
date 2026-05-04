#!/usr/bin/env python3
"""B4: Pick and Place — detect blue object, pick it up, place it at field A."""

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
        f"Robot {cfg.robot_id}: Grasp blue cube and place it on field A."
    )
