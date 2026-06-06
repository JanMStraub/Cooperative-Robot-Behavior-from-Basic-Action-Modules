#!/usr/bin/env python3
"""B13: Negotiation Ablation — LLM negotiation vs direct single-LLM plan."""

from __future__ import annotations

from typing import List


def get_tasks(config=None) -> List[str]:
    """
    Return dual-robot coordination tasks for negotiation ablation.

    All tasks have ambiguous or unspecified role assignment so negotiation
    must add value: resolving who leads, ordering sequential dependencies,
    or arbitrating resource contention. Tasks with explicit per-robot
    instructions (like B6/B7) are excluded — the direct planner can follow
    those without negotiation, making the enabled/disabled delta uninformative.
    """
    return [
        "Both robots need to retrieve their respective objects from the table.",
        "Stack the blue cube on top of the red cube using both robots.",
        "One of the robots should pick up the red cube and pass it to the other robot.",
        "One robot picks up the red cube and places it on the desk. Then the other robot picks up the blue cube and stacks it on top of the red cube.",
    ]
