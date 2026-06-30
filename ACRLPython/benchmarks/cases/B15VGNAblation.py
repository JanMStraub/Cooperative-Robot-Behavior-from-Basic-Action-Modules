#!/usr/bin/env python3
"""B15: VGN Ablation - neural grasp prediction vs geometric fallback."""

from __future__ import annotations

from typing import List

LIFT_TARGET_Y: float = 0.2
OFF_TABLE_MIN_Y: float = 0.1


def _grasp_probe(robot_id: str, color: str) -> list[dict]:
    return [
        {
            "operation": "detect_object_stereo",
            "params": {"robot_id": robot_id, "color": color},
            "capture_var": "last_detected",
        },
        {
            "operation": "grasp_object",
            "params": {"robot_id": robot_id, "object_id": "$last_detected"},
        },
        {
            "operation": "move_to_coordinate",
            "params": {
                "robot_id": robot_id,
                "x": "$last_detected.x",
                "y": LIFT_TARGET_Y,
                "z": "$last_detected.z",
            },
        },
        {
            "operation": "detect_object_stereo",
            "params": {"robot_id": robot_id, "color": color},
        },
    ]


FIXED_OP_CHAINS: list[list[dict]] = [
    _grasp_probe("Robot1", "blue"),
    _grasp_probe("Robot2", "magenta"),  # magenta spawns in right_workspace (Robot2)
    _grasp_probe("Robot1", "yellow"),
    _grasp_probe("Robot1", "red"),
]


def get_tasks(config=None) -> List[str]:
    return [
        f"Robot1: Grasp the blue cube, and lift it to y=0.2.",
        f"Robot2: Grasp magenta cube, and place it in field H.",
        f"Robot1: Grasp the yellow cube.",
        f"Robot1 grasps the red cube and hands it to Robot2.",
    ]
