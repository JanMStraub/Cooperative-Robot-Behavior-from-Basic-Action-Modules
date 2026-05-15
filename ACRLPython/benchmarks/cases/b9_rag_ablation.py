#!/usr/bin/env python3
"""B9: RAG Ablation — parse-only comparison with and without RAG retrieval."""

from __future__ import annotations

from typing import List

def get_tasks() -> List[str]:
    """
    Return NL task strings reused from B1–B5 for parse-only RAG ablation.

    Returns:
        List of natural language task strings.
    """

    return [
        f"Robot1: Detect the blue cube and move to it.",
        f"Robot2: Move to position (0.0, 0.15, 0.2), then move to (-0.1, 0.2, -0.2).",
        f"Robot1: Grasp the red cube and lift it to y=0.3.",
        f"Robot1: Grasp the yellow cube and place it on field C.",
    ]
