#!/usr/bin/env python3
"""B4: Pick and Place — full pick cycle with lift and place at fixed coordinates."""

from __future__ import annotations

from typing import Any, Dict, List

from ..config import BenchmarkConfig

_HOVER_OFFSET = 0.05  # approach height above object
_LIFT_OFFSET = 0.15  # lift height after grasp
_PLACE_COORDS = {"x": 0.2, "y": 0.15, "z": 0.1}


def build_sequence(cfg: BenchmarkConfig) -> List[Dict[str, Any]]:
    """
    Build 7-step sequence: detect → hover → grasp → close → lift → place → open.

    Args:
        cfg: Benchmark configuration.

    Returns:
        List of command dicts for SequenceExecutor.
    """
    return [
        # Steps 0–3: same as B3
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
        # Step 4: lift
        {
            "operation": "move_to_coordinate",
            "params": {
                "robot_id": cfg.robot_id,
                "x": "$target.x",
                "y": f"$target.y + {_LIFT_OFFSET}",
                "z": "$target.z",
            },
        },
        # Step 5: move to place position
        {
            "operation": "move_to_coordinate",
            "params": {"robot_id": cfg.robot_id, **_PLACE_COORDS},
        },
        # Step 6: open gripper
        {
            "operation": "control_gripper",
            "params": {"robot_id": cfg.robot_id, "open_gripper": True},
        },
    ]
