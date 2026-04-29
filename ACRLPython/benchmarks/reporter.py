#!/usr/bin/env python3
"""Benchmark result reporter — JSON file output + stdout summary."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from .result import BenchmarkResult


def write_json(result: BenchmarkResult, output_dir: str = ".") -> str:
    """
    Serialise BenchmarkResult to a JSON file.

    Args:
        result: The benchmark result to write.
        output_dir: Directory to write the file into (created if absent).

    Returns:
        Absolute path to the written file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = f"benchmark{result.benchmark_id}_{result.run_id}.json"
    path = Path(output_dir) / fname
    path.write_text(json.dumps(dataclasses.asdict(result), indent=2))
    return str(path)


def print_summary(result: BenchmarkResult) -> None:
    """
    Print a human-readable benchmark summary to stdout.

    Args:
        result: The benchmark result to summarise.
    """
    status = "PASS" if result.success else "FAIL"
    ok = sum(1 for s in result.steps if s.success)
    total = len(result.steps)

    print(f"\nBenchmark {result.benchmark_id}: {result.benchmark_name}")
    print(f"  Status:    {status}")
    print(f"  Duration:  {result.total_duration_ms:.0f}ms")
    print(f"  Steps:     {ok}/{total} succeeded")
    print(f"  Avg step:  {result.avg_step_duration_ms:.0f}ms")
    print(f"  Retries:   {result.retry_count}")
    print("  Steps:")
    for s in result.steps:
        tag = "OK  " if s.success else "FAIL"
        err = f"  [{s.error_code}]" if s.error_code else ""
        print(f"    [{s.index}] {s.operation:<35} {tag} {s.duration_ms:>7.0f}ms{err}")

    if result.chain_metrics is not None:
        cm = result.chain_metrics
        print("  Chain Metrics:")
        print(f"    Tasks:      {cm.completed_tasks}/{cm.total_tasks}")
        print(f"    Error rate: {cm.error_rate:.1%}")
        print(f"    Recoveries: {cm.recovery_count}")
        for code, count in cm.per_error_code.items():
            print(f"    {code}: {count}")
