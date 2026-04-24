#!/usr/bin/env python3
"""B3: Navigate and Lift — detect object, hover above, move to grasp position, close gripper."""

from __future__ import annotations

from typing import Any, Dict, List

from ..config import BenchmarkConfig

_HOVER_OFFSET = 0.05  # metres above object for approach


def build_sequence(cfg: BenchmarkConfig) -> List[Dict[str, Any]]:
    """
    Build 4-step sequence: detect → hover → grasp position → gripper close.

    Args:
        cfg: Benchmark configuration.

    Returns:
        List of command dicts for SequenceExecutor.
    """
    return [
        {
            "operation": "detect_object_stereo",
            "params": {"robot_id": cfg.robot_id, "color": "red"},
            "capture_var": "target",
        },
        {
            "operation": "move_to_coordinate",
            "params": {
                "robot_id": cfg.robot_id,
                "x": "$target.x",
                "y": f"$target.y + {_HOVER_OFFSET}",
                "z": "$target.z",
            },
        },
        {
            "operation": "move_to_coordinate",
            "params": {
                "robot_id": cfg.robot_id,
                "x": "$target.x",
                "y": "$target.y",
                "z": "$target.z",
            },
        },
        {
            "operation": "control_gripper",
            "params": {"robot_id": cfg.robot_id, "open_gripper": False},
        },
    ]
