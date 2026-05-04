#!/usr/bin/env python3
"""B6: Dual-Robot Handoff — Robot A grasps red cube, hands it off to Robot B."""

from __future__ import annotations

from ..config import DualRobotConfig


def get_task(cfg: DualRobotConfig) -> str:
    """
    Return natural language task description for B6.

    Args:
        cfg: DualRobotConfig with robot_id_a and robot_id_b.

    Returns:
        Task string sent to the LLM via SequenceServer.
    """
    return (
        f"{cfg.robot_id_a} and {cfg.robot_id_b} perform a handoff of the red cube"
    )
