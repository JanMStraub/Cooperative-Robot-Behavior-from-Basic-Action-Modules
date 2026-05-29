#!/usr/bin/env python3
"""Benchmark result reporter — JSON file output + stdout summary."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from .Result import BenchmarkResult


def write_json(result: BenchmarkResult, output_dir: str = ".") -> str:
    """
    Serialise BenchmarkResult to a JSON file.


    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = f"benchmark{result.benchmark_id}_{result.run_id}.json"
    path = Path(output_dir) / fname
    path.write_text(json.dumps(dataclasses.asdict(result), indent=2))
    return str(path)


def write_csv(result: BenchmarkResult, output_dir: str = ".") -> str:
    """
    Serialise BenchmarkResult to a flat CSV file for offline analysis.

    One row per step; benchmark-level summary columns repeated on every row.


    """
    import csv

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = f"benchmark{result.benchmark_id}_{result.run_id}.csv"
    path = Path(output_dir) / fname

    # Flatten feature_flags into individual columns for easy filtering
    flag_keys = [
        "use_rag",
        "use_vgn",
        "use_knowledge_graph",
        "use_ros_movement",
        "reflexion_enabled",
        "dry_run",
        "use_negotiation",
    ]
    fieldnames = [
        "benchmark_id",
        "benchmark_name",
        "run_id",
        "execution_mode",
        "benchmark_success",
        "total_duration_ms",
        "success_rate",
        "ops_executed",
        "ops_succeeded",
        "avg_step_duration_ms",
        *flag_keys,
        "step_index",
        "operation",
        "robot_id",
        "parallel_group_id",
        "step_success",
        "step_duration_ms",
        "error_code",
        "retry_count",
    ]
    flags = result.feature_flags or {}
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in result.steps:
            row = {
                "benchmark_id": result.benchmark_id,
                "benchmark_name": result.benchmark_name,
                "run_id": result.run_id,
                "execution_mode": result.execution_mode,
                "benchmark_success": result.success,
                "total_duration_ms": result.total_duration_ms,
                "success_rate": result.success_rate,
                "ops_executed": result.ops_executed,
                "ops_succeeded": result.ops_succeeded,
                "avg_step_duration_ms": result.avg_step_duration_ms,
                "step_index": s.index,
                "operation": s.operation,
                "robot_id": s.robot_id or "",
                "parallel_group_id": (
                    "" if s.parallel_group_id is None else s.parallel_group_id
                ),
                "step_success": s.success,
                "step_duration_ms": s.duration_ms,
                "error_code": s.error_code or "",
                "retry_count": s.retry_count,
            }
            for k in flag_keys:
                row[k] = flags.get(k, "")
            writer.writerow(row)
    return str(path)


def print_summary(result: BenchmarkResult) -> None:
    """
    Print a human-readable benchmark summary to stdout.

    """
    status = "PASS" if result.success else "FAIL"
    ok = sum(1 for s in result.steps if s.success)
    total = len(result.steps)

    print(f"\nBenchmark {result.benchmark_id}: {result.benchmark_name}")
    print(f"  Status:    {status}")
    print(f"  Mode:      {result.execution_mode}")
    if result.feature_flags:
        active = [k for k, v in result.feature_flags.items() if v]
        if active:
            print(f"  Flags:     {', '.join(active)}")
    print(f"  Duration:  {result.total_duration_ms:.0f}ms")
    print(f"  Steps:     {ok}/{total} succeeded")
    print(f"  Avg step:  {result.avg_step_duration_ms:.0f}ms")
    print(f"  Retries:   {result.retry_count}")
    if result.per_op_stats:
        failing = {
            op: v
            for op, v in result.per_op_stats.items()
            if isinstance(v, dict) and v.get("fail_count", 0) > 0
        }
        if failing:
            print("  Failing ops:")
            for op, v in failing.items():
                print(
                    f"    {op:<35} {v['fail_count']}/{v['count']} fails  avg {v['avg_duration_ms']:.0f}ms"
                )
    print("  Steps:")
    for s in result.steps:
        tag = "OK  " if s.success else "FAIL"
        err = f"  [{s.error_code}]" if s.error_code else ""
        robot = f"  {s.robot_id}" if s.robot_id else ""
        pg = f"  pg={s.parallel_group_id}" if s.parallel_group_id is not None else ""
        print(
            f"    [{s.index}] {s.operation:<35} {tag} {s.duration_ms:>7.0f}ms{robot}{pg}{err}"
        )

    if result.ablation is not None:
        ab = result.ablation
        print("")
        print("  ABLATION METRICS")
        print(f"  Condition:             {ab.condition}")
        print(f"  Hallucinated ops:      {ab.hallucinated_ops}")
        print(f"  Reflexion recoveries:  {ab.reflexion_recoveries}")
        print(f"  Negotiation rounds:    {ab.negotiation_rounds}")
        print(f"  Ablation success rate: {ab.success_rate:.1%}")

    if result.chain_metrics is not None:
        cm = result.chain_metrics
        print("  Chain Metrics:")
        print(f"    Tasks:      {cm.completed_tasks}/{cm.total_tasks}")
        print(f"    Error rate: {cm.error_rate:.1%}")
        print(f"    Recoveries: {cm.recovery_count}")
        for code, count in cm.per_error_code.items():
            print(f"    {code}: {count}")
