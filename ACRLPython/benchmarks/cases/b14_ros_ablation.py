#!/usr/bin/env python3
"""B14: ROS vs Unity Movement Ablation — MoveIt planning path vs direct TCP path."""

from __future__ import annotations

from typing import List

from ..config import BenchmarkConfig


def get_tasks(cfg: BenchmarkConfig) -> List[str]:
    """
    Return NL movement tasks for ROS/MoveIt vs Unity-TCP ablation.

    With ROS enabled, move_to_coordinate routes through ROSDispatcher →
    ROSBridge (TCP:5020) → MoveIt → /arm_controller/joint_trajectory → Unity.
    With ROS disabled (unity mode), commands go direct to CommandServer (TCP:5007).
    Live mode exposes success-rate and timing differences (MoveIt planning overhead);
    offline mode verifies the parse and dispatch path for both conditions.

    Args:
        cfg: Benchmark configuration.

    Returns:
        List of natural language task strings.
    """
    return [
        f"Robot {cfg.robot_id}: Move to position (0.2, 0.15, 0.1).",
        f"Robot {cfg.robot_id}: Move to position (0.0, 0.15, 0.2), then move to (-0.1, 0.2, -0.2).",
        f"Robot {cfg.robot_id}: Detect the blue cube and move to it.",
        f"Robot {cfg.robot_id}: Move to position (0.0, 0.15, 0.2), then return to start position.",
    ]
