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


@dataclasses.dataclass
class ChainMetrics:
    """Aggregate metrics for B8 heterogeneous chain benchmark."""

    total_tasks: int
    completed_tasks: int
    error_rate: float
    recovery_count: int
    per_error_code: Dict[str, int]


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
    retry_count: int = 0
    first_failure_step: Optional[int] = None
    chain_metrics: Optional[ChainMetrics] = None
    hallucinated_ops: int = 0
    reflexion_recoveries: int = 0
    negotiation_rounds: int = 0
    ablation: Optional[AblationMetrics] = None


def make_run_id() -> str:
    """Generate unique run identifier."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:6]
