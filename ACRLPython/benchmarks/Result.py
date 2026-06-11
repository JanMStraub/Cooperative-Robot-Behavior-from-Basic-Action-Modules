#!/usr/bin/env python3
"""Benchmark result dataclasses."""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional
from datetime import datetime, timezone
from uuid import uuid4


@dataclasses.dataclass
class StepResult:
    """Result for a single benchmark step."""

    index: int
    operation: str
    success: bool
    duration_ms: float
    error_code: Optional[str]
    error_message: Optional[str]
    retry_count: int = 0
    robot_id: Optional[str] = None
    parallel_group_id: Optional[int] = None


@dataclasses.dataclass
class ChainMetrics:
    """Aggregate metrics for B8 heterogeneous chain benchmark."""

    total_tasks: int
    completed_tasks: int
    error_rate: float
    recovery_count: int
    per_error_code: Dict[str, int]
    per_phase_success: Dict[str, float] = dataclasses.field(default_factory=dict)
    # keys: "phase_a", "phase_b", "phase_c" → fraction of cycles that succeeded


@dataclasses.dataclass
class AblationMetrics:
    """Per-condition metrics for a single ablation run (enabled or disabled)."""

    condition: str  # "enabled" or "disabled"
    hallucinated_ops: int
    reflexion_recoveries: int
    negotiation_rounds: int
    success_rate: float
    ops_executed: int
    ops_succeeded: int
    # Improvement over baseline (enabled - disabled); positive = feature helps
    condition_delta: float = 0.0  # success_rate_enabled - success_rate_disabled
    hallucination_delta: int = 0  # hallucinated_ops_disabled - hallucinated_ops_enabled
    # Grasp-specific metrics (populated by B15)
    grasp_sr: float = 0.0
    avg_grasp_duration_ms: float = 0.0


@dataclasses.dataclass
class BenchmarkResult:
    """Complete result for a single benchmark run."""

    benchmark_id: int
    benchmark_name: str
    run_id: str
    config_snapshot: dict
    success: bool
    total_duration_ms: float
    steps: List[StepResult]
    ops_executed: int
    ops_succeeded: int
    success_rate: float
    avg_step_duration_ms: float
    # LLM model under test. Empty for legacy runs (recovered from dir path by readers).
    model: str = ""
    retry_count: int = 0
    first_failure_step: Optional[int] = None
    chain_metrics: Optional[ChainMetrics] = None
    hallucinated_ops: int = 0
    reflexion_recoveries: int = 0
    negotiation_rounds: int = 0
    ablation: Optional[AblationMetrics] = None
    ablation_baseline: Optional[AblationMetrics] = None  # disabled-condition metrics
    feature_flags: dict = dataclasses.field(default_factory=dict)
    parsed_plan: List[str] = dataclasses.field(default_factory=list)
    per_op_stats: dict = dataclasses.field(default_factory=dict)
    execution_mode: str = "offline"
    task_breakdown: List[dict] = dataclasses.field(default_factory=list)


def make_run_id() -> str:
    """Generate unique run identifier."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:6]
