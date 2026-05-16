#!/usr/bin/env python3
"""B13: VGN Ablation — neural grasp prediction vs geometric fallback."""

from __future__ import annotations

from typing import List


def get_tasks() -> List[str]:
    """
    Return NL grasp tasks for VGN vs geometric-fallback ablation.

    With VGN enabled, grasp_object calls VGNClient which runs TSDF+VGN inference
    and returns 6-DOF poses.  With VGN disabled, grasp_object uses the geometric
    top-down fallback.  Live mode reveals success-rate and timing differences;
    offline mode verifies the command-parse path works for both conditions.

    """
    return [
        f"Robot2: Grasp red cube.",
        f"Robot1: Grasp the blue cube and lift it to y=0.3.",
        f"Robot1: Grasp the yellow cube and place it on field H.",
        f"Robot2: Detect the nearest object and grasp it.",
    ]
