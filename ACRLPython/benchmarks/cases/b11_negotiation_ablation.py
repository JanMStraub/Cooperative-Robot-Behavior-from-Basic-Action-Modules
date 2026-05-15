#!/usr/bin/env python3
"""B11: Negotiation Ablation — LLM negotiation vs direct single-LLM plan."""

from __future__ import annotations

from typing import List

def get_tasks() -> List[str]:
    """
    Return dual-robot coordination tasks for negotiation ablation.

    Tasks require both robots to coordinate — making negotiation's contribution
    visible: with negotiation, robots explicitly agree on roles and sequencing;
    without it, the single LLM generates a plan without inter-robot dialogue.

    Returns:
        List of natural language task strings.
    """

    return [
            f"Robot1 and Robot2 perform a handoff of the red cube and Robot2 places it at (0.2, 0.0, 0.3).",
            f"Robot1 and Robot2 collaboratively reorient the red cube: Robot1 holds one end while Robot2 rotates the other 90 degrees.",
            f"Robot1 picks up the red cube and passes it to Robot2 and Robot2 stacks it on the green block.",
    ]
