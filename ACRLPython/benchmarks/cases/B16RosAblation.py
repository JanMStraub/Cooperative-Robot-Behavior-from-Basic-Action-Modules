#!/usr/bin/env python3
"""B16: ROS vs Unity Movement Ablation - MoveIt planning path vs direct TCP path."""

from __future__ import annotations

from typing import List

from benchmarks.cases.B1NavigateToObject import get_task as _b1
from benchmarks.cases.B2SequentialNavigation import get_task as _b2
from benchmarks.cases.B3NavigateAndLift import get_task as _b3
from benchmarks.cases.B4PickAndPlace import get_task as _b4
from benchmarks.cases.B5PoseAwareGrasp import get_task as _b5

# Fixed operation chains for live-mode execution - bypasses LLM so both ROS and
# Unity conditions execute identical sequences, isolating routing overhead only.
# Derived from B1-B5 EXPECTED_OP_CHAIN values with explicit params.
FIXED_OP_CHAINS: list[list[dict]] = [
    # B1: detect blue cube → move above it (approach_offset lifts target 10 cm above cube;
    # avoids near-table IK struggles when cubes sit at y≈0.01-0.02 m)
    [
        {
            "operation": "detect_object_stereo",
            "params": {"robot_id": "Robot1", "color": "blue"},
        },
        {
            "operation": "move_to_coordinate",
            "params": {"robot_id": "Robot1", "approach_offset": 0.1},
        },
    ],
    # B2: detect red → move above it, detect yellow → move above it
    [
        {
            "operation": "detect_object_stereo",
            "params": {"robot_id": "Robot1", "color": "red"},
        },
        {
            "operation": "move_to_coordinate",
            "params": {"robot_id": "Robot1", "approach_offset": 0.1},
        },
        {
            "operation": "detect_object_stereo",
            "params": {"robot_id": "Robot1", "color": "yellow"},
        },
        {
            "operation": "move_to_coordinate",
            "params": {"robot_id": "Robot1", "approach_offset": 0.1},
        },
    ],
    # B3: detect blue → grasp → lift to y=0.2
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
    # B4: detect magenta → grasp → detect field H → place at field coords
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
    # B5: detect yellow → grasp
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
]


def get_tasks(_config=None) -> List[str]:
    """
    Return B1-B5 movement tasks for ROS/MoveIt vs Unity-TCP ablation.

    With ROS enabled, move_to_coordinate routes through ROSDispatcher →
    ROSBridge (TCP:5020) → MoveIt → /arm_controller/joint_trajectory → Unity.
    With ROS disabled (unity mode), commands go direct to CommandServer (TCP:5007).
    Using the same tasks as B1/B2 ties the ablation to the main navigation
    benchmarks. Live mode exposes timing differences (MoveIt planning overhead);
    offline mode verifies the parse and dispatch path for both conditions.
    """
    return [_b1(), _b2(), _b3(), _b4(), _b5()]
