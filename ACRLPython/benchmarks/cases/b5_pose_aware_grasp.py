#!/usr/bin/env python3
"""B5: Pose-Aware Grasp — detect, orient gripper, then execute grasp operation."""

from __future__ import annotations

from typing import Any, Dict, List

from ..config import BenchmarkConfig

_APPROACH_OFFSET = 0.1  # metres above object for approach


def build_sequence(cfg: BenchmarkConfig) -> List[Dict[str, Any]]:
    """
    Build 5-step sequence: detect → approach → orient → grasp → close.

    Uses grasp_object operation (Level 3) with explicit gripper orientation,
    exercising the full grasp planning pipeline.

    Args:
        cfg: Benchmark configuration.

    Returns:
        List of command dicts for SequenceExecutor.
    """
    return [
        {
            "operation": "detect_object_stereo",
            "params": {"robot_id": cfg.robot_id, "color": "blue"},
            "capture_var": "target",
        },
        {
            "operation": "move_to_coordinate",
            "params": {
                "robot_id": cfg.robot_id,
                "x": "$target.x",
                "y": f"$target.y + {_APPROACH_OFFSET}",
                "z": "$target.z",
            },
        },
        {
            "operation": "adjust_end_effector_orientation",
            "params": {
                "robot_id": cfg.robot_id,
                "roll": 0.0,
                "pitch": -90.0,
                "yaw": 0.0,
            },
        },
        {
            "operation": "grasp_object",
            "params": {"robot_id": cfg.robot_id, "object_id": "$target.color"},
        },
        {
            "operation": "control_gripper",
            "params": {"robot_id": cfg.robot_id, "open_gripper": False},
        },
    ]
