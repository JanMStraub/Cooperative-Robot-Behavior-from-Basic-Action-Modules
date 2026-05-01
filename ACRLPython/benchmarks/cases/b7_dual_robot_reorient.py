#!/usr/bin/env python3
"""B7: Dual-Robot Reorient — B6 plus second sync barrier and parallel arm swings."""

from __future__ import annotations

from ..config import DualRobotConfig


def get_task(cfg: DualRobotConfig) -> str:
    """
    Return natural language task description for B7.

    Args:
        cfg: DualRobotConfig with robot_id_a and robot_id_b.

    Returns:
        Task string sent to the LLM via SequenceServer.
    """
    return (
        f"Robot {cfg.robot_id_a} and Robot {cfg.robot_id_b} work together: "
        f"{cfg.robot_id_a} detects the red object while {cfg.robot_id_b} moves to the handoff position at (0.0, 0.2, 0.0). "
        f"Then {cfg.robot_id_a} moves to the detected object and grasps it. "
        f"Once {cfg.robot_id_a} has grasped the object, {cfg.robot_id_b} closes its gripper to assist. "
        f"Then both robots simultaneously reorient: {cfg.robot_id_a} swings to (-0.1, 0.35, 0.1) "
        f"and {cfg.robot_id_b} swings to (0.1, 0.35, 0.1)."
    )
