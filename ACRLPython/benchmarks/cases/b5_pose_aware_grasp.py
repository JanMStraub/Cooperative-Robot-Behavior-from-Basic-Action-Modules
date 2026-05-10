#!/usr/bin/env python3
"""B5: Pose-Aware Grasp — detect red object, orient gripper downward, grasp it."""

from __future__ import annotations

from ..config import BenchmarkConfig


def get_task(cfg: BenchmarkConfig) -> str:
    """
    Return natural language task description for B5.

    Args:
        cfg: Benchmark configuration.

    Returns:
        Task string sent to the LLM via SequenceServer.
    """
    return (
        f"Robot {cfg.robot_id}: Grasp yellow cube."
    )
