#!/usr/bin/env python3
"""B15: VGN Ablation - neural grasp prediction vs geometric fallback."""

from __future__ import annotations

from typing import List

# Fixed operation chains for live-mode execution - bypasses LLM so both VGN and
# geometric conditions execute identical sequences, isolating grasp-pose quality only.
FIXED_OP_CHAINS: list[list[dict]] = [
    # "Grasp blue cube, lift to y=0.2"
    [
        {
            "operation": "detect_object_stereo",
            "params": {"robot_id": "Robot1", "color": "blue"},
        },
        {
            "operation": "grasp_object",
            "params": {"robot_id": "Robot1", "object_id": "$last_detected"},
        },
        {
            "operation": "move_to_coordinate",
            "params": {"robot_id": "Robot1", "x": 0.0, "y": 0.2, "z": 0.0},
        },
    ],
    # "Grasp magenta cube, place in field H" - magenta spawns in right_workspace (Robot2)
    [
        {
            "operation": "detect_object_stereo",
            "params": {"robot_id": "Robot2", "color": "magenta"},
        },
        {
            "operation": "grasp_object",
            "params": {"robot_id": "Robot2", "object_id": "$last_detected"},
        },
        {
            "operation": "detect_field",
            "params": {"robot_id": "Robot2", "field_label": "H"},
            "capture_var": "field_pos",
        },
        {
            "operation": "place_object",
            "params": {
                "robot_id": "Robot2",
                "x": "$field_pos.x",
                "y": "$field_pos.y",
                "z": "$field_pos.z",
            },
        },
    ],
    # "Grasp yellow cube"
    [
        {
            "operation": "detect_object_stereo",
            "params": {"robot_id": "Robot1", "color": "yellow"},
        },
        {
            "operation": "grasp_object",
            "params": {"robot_id": "Robot1", "object_id": "$last_detected"},
        },
    ],
    # "Robot1 grasp red cube and hand it to Robot2" - full B6 handoff protocol
    [
        {
            "operation": "detect_object_stereo",
            "params": {"robot_id": "Robot1", "color": "red"},
        },
        {
            "operation": "grasp_object",
            "params": {"robot_id": "Robot1", "object_id": "$last_detected"},
        },
        {
            "operation": "move_to_coordinate",
            "params": {"robot_id": "Robot1", "x": 0.0, "y": 0.35, "z": 0.0},
        },
        {
            "operation": "adjust_end_effector_orientation",
            "params": {"robot_id": "Robot1", "pitch": 0.0},
        },
        {"operation": "signal", "params": {"event_name": "handoff_ready"}},
        {
            "operation": "wait_for_signal",
            "params": {"robot_id": "Robot2", "event_name": "handoff_ready"},
        },
        {
            "operation": "detect_object_stereo",
            "params": {"robot_id": "Robot2", "color": "red"},
        },
        {
            "operation": "receive_handoff",
            "params": {
                "robot_id": "Robot2",
                "object_id": "$last_detected",
                "source_robot_id": "Robot1",
            },
        },
        {"operation": "release_object", "params": {"robot_id": "Robot1"}},
    ],
]


def get_tasks(config=None) -> List[str]:
    """
    Return grasp tasks for VGN vs geometric-fallback ablation.

    With VGN enabled, grasp_object calls VGNClient (TSDF+VGN inference,
    6-DOF poses). With VGN disabled, grasp_object uses the geometric
    top-down fallback. Tasks mirror B3-B6 scenarios but use cfg.robot_id
    so the ablation is repeatable with either robot.
    """

    return [
        f"Robot1: Grasp the blue cube, and lift it to y=0.2.",
        f"Robot2: Grasp magenta cube, and place it in field H.",
        f"Robot1: Grasp the yellow cube.",
        f"Robot1 grasps the red cube and hands it to Robot2.",
    ]
