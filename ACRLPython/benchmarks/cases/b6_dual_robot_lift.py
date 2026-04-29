#!/usr/bin/env python3
"""B6: Dual-Robot Lift — parallel detection/positioning then synchronized grasp."""

from __future__ import annotations

from typing import Any, Dict, List

from ..config import DualRobotConfig

_HANDOFF_POS = {"x": 0.0, "y": 0.2, "z": 0.0}
_LIFT_POS = {"x": 0.0, "y": 0.35, "z": 0.0}


def build_sequence(cfg: DualRobotConfig) -> List[Dict[str, Any]]:
    """
    Build 8-step dual-robot sequence with one sync barrier.

    Uses parallel_group field so SequenceExecutor dispatches concurrent steps
    via ThreadPoolExecutor. The sync barrier (group 3) places signal and
    wait_for_signal in the same group — both dispatch concurrently, so signal
    fires immediately and wait_for_signal returns without blocking.

    Groups:
      1: A detects object (parallel) + B moves to handoff position (parallel)
      2: A moves to detected object
      3: A closes gripper + A signals + B waits (sync barrier)
      4: A lifts + B closes gripper

    Args:
        cfg: DualRobotConfig with robot_id_a, robot_id_b, sync_timeout_ms.

    Returns:
        List of command dicts for SequenceExecutor.
    """
    A = cfg.robot_id_a
    B = cfg.robot_id_b
    ev = f"bench6_{A}_grasped"

    return [
        # Group 1: A detects, B moves to handoff position (parallel)
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
        # Group 2: A moves to detected object
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
        # Group 3: sync barrier — A grasps and signals, B waits
        {
            "operation": "control_gripper",
            "params": {"robot_id": A, "open_gripper": False},
            "parallel_group": 3,
        },
        {
            "operation": "signal",
            "params": {"event_name": ev, "robot_id": A},
            "parallel_group": 3,
        },
        {
            "operation": "wait_for_signal",
            "params": {
                "event_name": ev,
                "robot_id": B,
                "timeout_ms": cfg.sync_timeout_ms,
            },
            "parallel_group": 3,
        },
        # Group 4: A lifts, B closes gripper
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
    ]
