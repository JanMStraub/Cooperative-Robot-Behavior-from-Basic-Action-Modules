#!/usr/bin/env python3
"""B7: Dual-Robot Reorient — B6 plus second sync barrier and parallel arm swings."""

from __future__ import annotations

from typing import Any, Dict, List

from ..config import DualRobotConfig

_HANDOFF_POS = {"x": 0.0, "y": 0.2, "z": 0.0}
_LIFT_POS = {"x": 0.0, "y": 0.35, "z": 0.0}
_SWING_A = {"x": -0.1, "y": 0.35, "z": 0.1}
_SWING_B = {"x": 0.1, "y": 0.35, "z": 0.1}


def build_sequence(cfg: DualRobotConfig) -> List[Dict[str, Any]]:
    """
    Build 11-step dual-robot sequence with two sync barriers and parallel swings.

    Extends B6 with:
      Group 5: second sync barrier (A signals ready, B waits)
      Group 6: simultaneous incremental arm swings for both robots

    Args:
        cfg: DualRobotConfig with robot_id_a, robot_id_b, sync_timeout_ms.

    Returns:
        List of command dicts for SequenceExecutor.
    """
    A = cfg.robot_id_a
    B = cfg.robot_id_b
    ev1 = f"bench7_{A}_grasped"
    ev2 = f"bench7_{A}_ready"

    return [
        # Groups 1–4: same as B6 (with bench7 event names)
        {
            "operation": "detect_object_stereo",
            "params": {"robot_id": A, "color": "red"},
            "capture_var": "target",
            "parallel_group": 1,
        },
        {
            "operation": "move_to_coordinate",
            "params": {"robot_id": B, **_HANDOFF_POS},
            "parallel_group": 1,
        },
        {
            "operation": "move_to_coordinate",
            "params": {
                "robot_id": A,
                "x": "$target.x",
                "y": "$target.y",
                "z": "$target.z",
            },
            "parallel_group": 2,
        },
        {
            "operation": "control_gripper",
            "params": {"robot_id": A, "open_gripper": False},
            "parallel_group": 3,
        },
        {
            "operation": "signal",
            "params": {"event_name": ev1, "robot_id": A},
            "parallel_group": 3,
        },
        {
            "operation": "wait_for_signal",
            "params": {
                "event_name": ev1,
                "robot_id": B,
                "timeout_ms": cfg.sync_timeout_ms,
            },
            "parallel_group": 3,
        },
        {
            "operation": "move_to_coordinate",
            "params": {"robot_id": A, **_LIFT_POS},
            "parallel_group": 4,
        },
        {
            "operation": "control_gripper",
            "params": {"robot_id": B, "open_gripper": False},
            "parallel_group": 4,
        },
        # Group 5: second sync barrier
        {
            "operation": "signal",
            "params": {"event_name": ev2, "robot_id": A},
            "parallel_group": 5,
        },
        {
            "operation": "wait_for_signal",
            "params": {
                "event_name": ev2,
                "robot_id": B,
                "timeout_ms": cfg.sync_timeout_ms,
            },
            "parallel_group": 5,
        },
        # Group 6: parallel incremental arm swings
        {
            "operation": "move_to_coordinate",
            "params": {"robot_id": A, **_SWING_A},
            "parallel_group": 6,
        },
        {
            "operation": "move_to_coordinate",
            "params": {"robot_id": B, **_SWING_B},
            "parallel_group": 6,
        },
    ]
