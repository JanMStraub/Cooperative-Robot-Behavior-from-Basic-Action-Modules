#!/usr/bin/env python3
"""
BenchmarkRunner — wraps SequenceExecutor for benchmark execution.

Disables Reflexion retries and optionally installs dry-run mocks before
each run, restoring both in a finally block.
"""

from __future__ import annotations

import dataclasses
import importlib
import time
from typing import Any, Dict, List, Optional

import orchestrators.SequenceExecutor as _seq_mod
from orchestrators.SequenceExecutor import SequenceExecutor

from . import mock_registry
from .config import BenchmarkConfig, DualRobotConfig
from .result import BenchmarkResult, ChainMetrics, StepResult, make_run_id

_BENCHMARK_NAMES: Dict[int, str] = {
    1: "Navigate to Object",
    2: "Sequential Navigation",
    3: "Navigate and Lift",
    4: "Pick and Place",
    5: "Pose-Aware Grasp",
    6: "Dual-Robot Lift",
    7: "Dual-Robot Reorient",
    8: "Heterogeneous Chain",
}

_CASE_MODULES: Dict[int, str] = {
    1: "benchmarks.cases.b1_navigate_to_object",
    2: "benchmarks.cases.b2_sequential_navigation",
    3: "benchmarks.cases.b3_navigate_and_lift",
    4: "benchmarks.cases.b4_pick_and_place",
    5: "benchmarks.cases.b5_pose_aware_grasp",
    6: "benchmarks.cases.b6_dual_robot_lift",
    7: "benchmarks.cases.b7_dual_robot_reorient",
    8: "benchmarks.cases.b8_heterogeneous_chain",
}


class BenchmarkRunner:
    """Executes individual benchmarks and returns structured BenchmarkResult objects."""

    def run(self, benchmark_id: int, cfg: BenchmarkConfig) -> BenchmarkResult:
        """
        Run a single benchmark.

        Patches REFLEXION_ENABLED=False in the SequenceExecutor module namespace
        for deterministic timing. Installs dry-run mocks when cfg.dry_run=True.
        Both patches are always restored via finally.

        Args:
            benchmark_id: Integer 1–8 identifying the benchmark.
            cfg: BenchmarkConfig (or DualRobotConfig for B6–B8).

        Returns:
            BenchmarkResult with full metrics and step details.
        """
        # Patch in SequenceExecutor module namespace — not in config.Servers.
        # SequenceExecutor does `from config.Servers import REFLEXION_ENABLED` at
        # load time, binding the bool value into its own module namespace.
        prev_reflexion = _seq_mod.REFLEXION_ENABLED
        _seq_mod.REFLEXION_ENABLED = False
        mock_original = None

        try:
            if cfg.dry_run:
                mock_original = mock_registry.install_mock("always_succeed")

            module = importlib.import_module(_CASE_MODULES[benchmark_id])

            if benchmark_id == 8:
                return self._run_b8_chain(cfg, module)

            executor = SequenceExecutor(
                default_timeout=cfg.timeout_per_step_s,
                check_completion=False,
                enable_verification=False,
            )
            sequence = module.build_sequence(cfg)
            raw = executor.execute_sequence(sequence)
            metrics = executor.get_metrics()
            return self._build_result(benchmark_id, cfg, raw, metrics)

        finally:
            _seq_mod.REFLEXION_ENABLED = prev_reflexion
            if mock_original is not None:
                mock_registry.restore_mock(mock_original)

    def _build_result(
        self,
        benchmark_id: int,
        cfg: BenchmarkConfig,
        raw: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> BenchmarkResult:
        """
        Convert raw SequenceExecutor output into a BenchmarkResult.

        Args:
            benchmark_id: Benchmark identifier.
            cfg: Config used for this run.
            raw: Return value of executor.execute_sequence().
            metrics: Return value of executor.get_metrics().

        Returns:
            Populated BenchmarkResult.
        """
        steps = self._parse_steps(raw.get("results") or [])
        first_fail = next((s.index for s in steps if not s.success), None)

        return BenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_name=_BENCHMARK_NAMES[benchmark_id],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            success=bool(raw.get("success", False)),
            total_duration_ms=float(raw.get("total_duration_ms", 0.0)),
            steps=steps,
            ops_executed=metrics["ops_executed"],
            ops_succeeded=metrics["ops_succeeded"],
            success_rate=metrics["ops_success_rate"],
            avg_step_duration_ms=metrics["avg_duration_ms"],
            first_failure_step=first_fail,
        )

    def _parse_steps(self, results: List[Optional[Dict[str, Any]]]) -> List[StepResult]:
        """
        Convert raw result dicts to StepResult list, skipping None entries.

        Args:
            results: List of per-step result dicts from SequenceExecutor.

        Returns:
            List of StepResult objects.
        """
        steps = []
        for r in results:
            if r is None:
                continue
            error = r.get("error")
            if isinstance(error, dict):
                error_code = error.get("code")
                error_message = error.get("message")
            elif isinstance(error, str):
                error_code = None
                error_message = error
            else:
                error_code = None
                error_message = None
            steps.append(
                StepResult(
                    index=r["index"],
                    operation=r["operation"],
                    success=bool(r.get("success", False)),
                    duration_ms=float(r.get("duration_ms", 0.0)),
                    error_code=error_code,
                    error_message=error_message,
                )
            )
        return steps

    def _run_b8_chain(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B8 heterogeneous chain benchmark.

        Creates a fresh SequenceExecutor per sub-task to prevent variable/metric
        bleed between tasks. Accumulates ChainMetrics across all sub-tasks.

        Args:
            cfg: BenchmarkConfig with task_count controlling chain length.
            module: The b8 cases module (must expose get_sub_tasks).

        Returns:
            BenchmarkResult with chain_metrics populated.
        """
        sub_tasks = module.get_sub_tasks(cfg, cfg.task_count)

        total = len(sub_tasks)
        completed = 0
        error_counts: Dict[str, int] = {}
        recovery_count = 0
        all_steps: List[StepResult] = []
        total_ms = 0.0
        ops_executed = 0
        ops_succeeded = 0

        for task_name, sequence in sub_tasks:
            executor = SequenceExecutor(
                default_timeout=cfg.timeout_per_step_s,
                check_completion=False,
                enable_verification=False,
            )
            raw = executor.execute_sequence(sequence)
            metrics = executor.get_metrics()
            total_ms += float(raw.get("total_duration_ms", 0.0))
            ops_executed += metrics["ops_executed"]
            ops_succeeded += metrics["ops_succeeded"]

            step_offset = len(all_steps)
            task_steps = self._parse_steps(raw.get("results") or [])
            for s in task_steps:
                s.index += step_offset
                if not s.success and s.error_code:
                    error_counts[s.error_code] = error_counts.get(s.error_code, 0) + 1
            all_steps.extend(task_steps)

            if raw.get("success"):
                completed += 1
            else:
                recovery_count += 1

        error_rate = (total - completed) / total if total > 0 else 0.0
        first_fail = next((s.index for s in all_steps if not s.success), None)

        return BenchmarkResult(
            benchmark_id=8,
            benchmark_name=_BENCHMARK_NAMES[8],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            success=(completed == total),
            total_duration_ms=total_ms,
            steps=all_steps,
            ops_executed=ops_executed,
            ops_succeeded=ops_succeeded,
            success_rate=(completed / total) if total > 0 else 0.0,
            avg_step_duration_ms=(total_ms / len(all_steps)) if all_steps else 0.0,
            first_failure_step=first_fail,
            chain_metrics=ChainMetrics(
                total_tasks=total,
                completed_tasks=completed,
                error_rate=error_rate,
                recovery_count=recovery_count,
                per_error_code=error_counts,
            ),
        )
