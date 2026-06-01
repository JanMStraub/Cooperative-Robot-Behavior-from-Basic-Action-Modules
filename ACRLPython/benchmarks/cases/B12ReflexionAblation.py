#!/usr/bin/env python3
"""B12: Reflexion Ablation — measure recovery rate with/without reflexion."""

from __future__ import annotations

from typing import List


def get_tasks(config=None) -> List[str]:
    """
    Return failure-prone NL tasks for reflexion ablation.

    Tasks are designed to stress IK reachability and precision so that initial
    attempts are likely to fail, giving reflexion retries a chance to recover.
    Covers NAVIGATION, MANIPULATION, and PERCEPTION eligible categories.

    """
    # Boundary coords stress IK reachability; low y risks table collision;
    # "nearest object" is ambiguous, forcing perception retry on first miss.
    return [
        "Robot1: Move to position (-0.58, 0.05, 0.45).",
        "Robot2: Grasp the red cube and lift it to y=0.03.",
        "Robot2: Grasp the green cube and place it at (0.55, 0.02, -0.48).",
        "Robot1: Move to position (-0.57, 0.08, 0.44), then move to (-0.12, 0.04, -0.47).",
        "Robot1: Navigate to the nearest object and return to start.",
    ]
