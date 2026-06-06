#!/usr/bin/env python3
"""Benchmark configuration dataclasses."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class BenchmarkConfig:
    """Configuration for single-robot benchmarks (B1–B5)."""

    robot_id: str = "Robot1"
    timeout_per_step_s: float = 90.0
    max_retries: int = 1
    dry_run: bool = False
    task_count: int = 1  # B8 sub-task count
    reflexion: bool = True
    check_completion: bool = True
    use_rag: bool = True
    reflexion_enabled: bool = True
    use_knowledge_graph: bool = True
    use_vgn: bool = True
    use_ros_movement: bool = True
    execution_mode: str = (
        "offline"  # "offline" = dry-run+mocks, "live" = real SequenceServer
    )


@dataclasses.dataclass
class DualRobotConfig(BenchmarkConfig):
    """Configuration for dual-robot benchmarks (B6–B8)."""

    robot_id_a: str = "Robot1"
    robot_id_b: str = "Robot2"
    sync_timeout_ms: int = 30000
    use_negotiation: bool = True
