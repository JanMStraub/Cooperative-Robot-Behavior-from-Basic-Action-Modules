#!/usr/bin/env python3
"""B8: Heterogeneous Chain — AutoRT-generated sub-tasks from live scene."""

from __future__ import annotations

import logging
import time
from itertools import cycle
from typing import List, Tuple

from ..config import BenchmarkConfig
from . import b1_navigate_to_object, b3_navigate_and_lift, b4_pick_and_place

logger = logging.getLogger(__name__)

_FALLBACK_BUILDERS = [
    ("b1_navigate", b1_navigate_to_object.get_task),
    ("b3_lift", b3_navigate_and_lift.get_task),
    ("b4_pick_place", b4_pick_and_place.get_task),
]

_DEFAULT_SCENE_OBJECTS = [
    {"object_id": "red_cube", "color": "red", "position": (-0.2, 0.05, 0.3), "confidence": 0.9},
    {"object_id": "blue_cube", "color": "blue", "position": (0.2, 0.05, 0.3), "confidence": 0.9},
    {"object_id": "green_cube", "color": "green", "position": (0.0, 0.05, 0.15), "confidence": 0.85},
]


def _build_scene():
    """Build SceneDescription from WorldState, or default mock if empty."""
    from autort.DataModels import SceneDescription, GroundedObject
    from operations.WorldState import get_world_state

    grounded = []
    try:
        ws = get_world_state()
        for obj in ws.get_all_objects():
            grounded.append(
                GroundedObject(
                    object_id=obj.object_id,
                    color=obj.color,
                    position=obj.position,
                    confidence=obj.confidence,
                    graspable=obj.is_graspable,
                )
            )
    except Exception as e:
        logger.debug(f"WorldState unavailable: {e}")

    if not grounded:
        for raw in _DEFAULT_SCENE_OBJECTS:
            grounded.append(
                GroundedObject(
                    object_id=raw["object_id"],
                    color=raw["color"],
                    position=raw["position"],
                    confidence=raw["confidence"],
                )
            )

    labels = [o.color for o in grounded]
    return SceneDescription(
        timestamp=time.time(),
        objects=grounded,
        scene_summary=f"Benchmark scene: {len(grounded)} objects ({labels})",
    )


def _generate_autort_tasks(
    cfg: BenchmarkConfig, task_count: int
) -> List[Tuple[str, str]]:
    """Generate sub-tasks via AutoRT TaskGenerator."""
    from autort.TaskGenerator import TaskGenerator
    import config.AutoRT as autort_config

    scene = _build_scene()
    generator = TaskGenerator(autort_config)
    proposed = generator.generate_tasks(
        scene=scene,
        robot_ids=[cfg.robot_id],
        num_tasks=task_count,
        include_collaborative=False,
    )
    return [(t.task_id, t.description) for t in proposed]


def _fallback_tasks(
    cfg: BenchmarkConfig, task_count: int
) -> List[Tuple[str, str]]:
    """Cycle B1/B3/B4 tasks — used in dry-run or on generation failure."""
    tasks: List[Tuple[str, str]] = []
    for i, (name, builder) in enumerate(cycle(_FALLBACK_BUILDERS)):
        if i >= task_count:
            break
        tasks.append((f"{name}_{i}", builder(cfg)))
    return tasks


def get_sub_tasks(
    cfg: BenchmarkConfig, task_count: int
) -> List[Tuple[str, str]]:
    """
    Generate heterogeneous sub-tasks via AutoRT TaskGenerator.

    Falls back to B1/B3/B4 cycle if dry_run or generation fails.

    Args:
        cfg: Benchmark configuration.
        task_count: Number of sub-tasks to generate.

    Returns:
        List of (task_name, task_string) tuples for BenchmarkRunner._run_b8_chain.
    """
    if not cfg.dry_run:
        try:
            tasks = _generate_autort_tasks(cfg, task_count)
            if tasks:
                return tasks
            logger.warning("AutoRT returned no tasks, falling back")
        except Exception as e:
            logger.warning(f"AutoRT generation failed, falling back: {e}")

    return _fallback_tasks(cfg, task_count)
