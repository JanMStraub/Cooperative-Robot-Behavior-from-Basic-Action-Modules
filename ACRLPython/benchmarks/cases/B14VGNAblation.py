#!/usr/bin/env python3
"""B14: VGN Ablation — neural grasp prediction vs geometric fallback."""

from __future__ import annotations

from typing import List

from benchmarks.cases.B3NavigateAndLift import get_task as _b3
from benchmarks.cases.B4PickAndPlace import get_task as _b4
from benchmarks.cases.B5PoseAwareGrasp import get_task as _b5


def get_tasks(config=None) -> List[str]:
    """
    Return the B3–B5 grasp tasks for VGN vs geometric-fallback ablation.

    With VGN enabled, grasp_object calls VGNClient (TSDF+VGN inference,
    6-DOF poses). With VGN disabled, grasp_object uses the geometric
    top-down fallback. Using the same tasks as B3–B5 ties the ablation
    directly to the main benchmark grasp scenarios.
    """
    return [_b3(), _b4(), _b5()]
