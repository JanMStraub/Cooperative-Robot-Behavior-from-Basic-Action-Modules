#!/usr/bin/env python3
"""B10: RAG Ablation — parse-only comparison with and without RAG retrieval."""

from __future__ import annotations

from typing import List

from benchmarks.cases.B1NavigateToObject import get_task as _b1
from benchmarks.cases.B2SequentialNavigation import get_task as _b2
from benchmarks.cases.B3NavigateAndLift import get_task as _b3
from benchmarks.cases.B4PickAndPlace import get_task as _b4
from benchmarks.cases.B5PoseAwareGrasp import get_task as _b5


def get_tasks(config=None) -> List[str]:
    """
    Return the B1–B5 task strings for parse-only RAG ablation.

    Using the same tasks as the main benchmarks ensures the ablation
    directly measures RAG's effect on the evaluated task distribution.
    """
    return [_b1(), _b2(), _b3(), _b4(), _b5()]
