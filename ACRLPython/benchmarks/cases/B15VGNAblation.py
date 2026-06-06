#!/usr/bin/env python3
"""B15: VGN Ablation — neural grasp prediction vs geometric fallback."""

from __future__ import annotations

from typing import List


def get_tasks(config=None) -> List[str]:
    """
    Return grasp tasks for VGN vs geometric-fallback ablation.

    With VGN enabled, grasp_object calls VGNClient (TSDF+VGN inference,
    6-DOF poses). With VGN disabled, grasp_object uses the geometric
    top-down fallback. Tasks mirror B3–B6 scenarios but use cfg.robot_id
    so the ablation is repeatable with either robot.
    """

    return [
        f"Robot1: Grasp the blue cube, and lift it to y=0.2.",
        f"Robot1: Grasp magenta cube, and place it in field H.",
        f"Robot1: Grasp the yellow cube.",
        f"Robot1 grasps the red cube and hands it to Robot2.",
    ]
