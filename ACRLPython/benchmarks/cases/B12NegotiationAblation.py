#!/usr/bin/env python3
"""B12: Negotiation Ablation — LLM negotiation vs direct single-LLM plan."""

from __future__ import annotations

from typing import List

from benchmarks.cases.B6RobotHandoff import get_task as _b6
from benchmarks.cases.B7DualRobotReorient import get_task as _b7


def get_tasks(config=None) -> List[str]:
    """
    Return dual-robot coordination tasks for negotiation ablation.

    B6 and B7 are included verbatim so the ablation measures negotiation's
    contribution on the same tasks as the main dual-robot benchmarks.
    The third task adds an explicit stacking goal absent from B6/B7.
    """
    return [
        _b6(),
        _b7(),
        "Robot1 picks up the red cube and passes it to Robot2 and Robot2 stacks it on the green block.",
    ]
