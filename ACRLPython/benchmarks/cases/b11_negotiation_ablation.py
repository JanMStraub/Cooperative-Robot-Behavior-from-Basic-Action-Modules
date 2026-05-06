#!/usr/bin/env python3
"""B11: Negotiation Ablation — LLM negotiation vs direct single-LLM plan."""

from __future__ import annotations

from typing import List

from ..config import DualRobotConfig


def get_tasks(cfg: DualRobotConfig) -> List[str]:
    """
    Return dual-robot coordination tasks for negotiation ablation.

    Tasks require both robots to coordinate — making negotiation's contribution
    visible: with negotiation, robots explicitly agree on roles and sequencing;
    without it, the single LLM generates a plan without inter-robot dialogue.

    Args:
        cfg: Dual robot configuration.

    Returns:
        List of natural language task strings.
    """

    return [
        (
            f"{cfg.robot_id_a} and {cfg.robot_id_b} perform a handoff of the red cube and {cfg.robot_id_b} places it at (0.2, 0.0, 0.3)."
        ),
        (
            f"Collaboratively reorient the red cube, {cfg.robot_id_a} holds one end while {cfg.robot_id_b} rotates the other 90 degrees."
        ),
        (
            f"{cfg.robot_id_a} picks up the red cube and passes it to {cfg.robot_id_b} and {cfg.robot_id_b} stacks it on the green block."
        ),
    ]
