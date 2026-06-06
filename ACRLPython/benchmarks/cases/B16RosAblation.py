#!/usr/bin/env python3
"""B16: ROS vs Unity Movement Ablation — MoveIt planning path vs direct TCP path."""

from __future__ import annotations

from typing import List

from benchmarks.cases.B1NavigateToObject import get_task as _b1
from benchmarks.cases.B2SequentialNavigation import get_task as _b2
from benchmarks.cases.B3NavigateAndLift import get_task as _b3
from benchmarks.cases.B4PickAndPlace import get_task as _b4
from benchmarks.cases.B5PoseAwareGrasp import get_task as _b5


def get_tasks(config=None) -> List[str]:
    """
    Return B1–B2 movement tasks for ROS/MoveIt vs Unity-TCP ablation.

    With ROS enabled, move_to_coordinate routes through ROSDispatcher →
    ROSBridge (TCP:5020) → MoveIt → /arm_controller/joint_trajectory → Unity.
    With ROS disabled (unity mode), commands go direct to CommandServer (TCP:5007).
    Using the same tasks as B1/B2 ties the ablation to the main navigation
    benchmarks. Live mode exposes timing differences (MoveIt planning overhead);
    offline mode verifies the parse and dispatch path for both conditions.
    """
    return [_b1(), _b2(), _b3(), _b4(), _b5()]
