#!/usr/bin/env python3
"""B14: ROS vs Unity Movement Ablation — MoveIt planning path vs direct TCP path."""

from __future__ import annotations

from typing import List


def get_tasks() -> List[str]:
    """
    Return NL movement tasks for ROS/MoveIt vs Unity-TCP ablation.

    With ROS enabled, move_to_coordinate routes through ROSDispatcher →
    ROSBridge (TCP:5020) → MoveIt → /arm_controller/joint_trajectory → Unity.
    With ROS disabled (unity mode), commands go direct to CommandServer (TCP:5007).
    Live mode exposes success-rate and timing differences (MoveIt planning overhead);
    offline mode verifies the parse and dispatch path for both conditions.

    """
    return [
        "Robot2: Move to position (0.2, 0.15, 0.1).",
        "Robot1: Move to position (0.0, 0.15, 0.2), then move to (-0.1, 0.2, -0.2).",
        "Robot1: Detect the blue cube and move to it.",
        "Robot1: Move to position (0.0, 0.15, 0.2), then return to start position.",
        "Robot1: Move to position (-0.05, 0.1, 0.05), then move to (-0.3, 0.2, 0.3).",
    ]
