#!/usr/bin/env python3
"""
Benchmark CLI entry point.

Usage:
    python -m benchmarks.run --benchmark 3
    python -m benchmarks.run --all
    python -m benchmarks.run --benchmark 8 --task-count 20 --dry-run
    python -m benchmarks.run --all --output-dir ./results/
    python -m benchmarks.run --benchmark 1 --reflexion
    python -m benchmarks.run --benchmark 1 --startup-timeout 180
"""

from __future__ import annotations

import argparse
import sys

from .config import BenchmarkConfig, DualRobotConfig
from .reporter import print_summary, write_json
from .runner import BenchmarkRunner
from .server_manager import ServerManager

_DUAL_ROBOT_BENCHMARKS = {6, 7, 8}


def _make_config(benchmark_id: int, args: argparse.Namespace) -> BenchmarkConfig:
    """
    Instantiate the appropriate config type for a given benchmark.

    Args:
        benchmark_id: Benchmark number 1–8.
        args: Parsed CLI arguments.

    Returns:
        BenchmarkConfig or DualRobotConfig instance.
    """
    live = not args.dry_run
    kwargs = dict(
        dry_run=args.dry_run,
        task_count=args.task_count,
        reflexion=args.reflexion,
        check_completion=live,
    )
    if benchmark_id in _DUAL_ROBOT_BENCHMARKS:
        return DualRobotConfig(**kwargs)
    return BenchmarkConfig(**kwargs)


def _run_benchmarks(
    runner: BenchmarkRunner,
    benchmark_ids: list,
    args: argparse.Namespace,
) -> int:
    """
    Execute the requested benchmarks and write results.

    Args:
        runner: BenchmarkRunner instance.
        benchmark_ids: List of benchmark IDs to run.
        args: Parsed CLI arguments.

    Returns:
        Exit code: 0 if all passed, 1 if any failed.
    """
    exit_code = 0
    for bid in benchmark_ids:
        cfg = _make_config(bid, args)
        result = runner.run(bid, cfg)
        path = write_json(result, args.output_dir)
        print_summary(result)
        print(f"  JSON: {path}")
        if not result.success:
            exit_code = 1
    return exit_code


def main() -> None:
    """Parse CLI arguments and run requested benchmarks."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.run",
        description="ACRL Benchmark Runner",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--benchmark",
        type=int,
        choices=range(1, 9),
        metavar="N",
        help="Run a single benchmark (1–8)",
    )
    group.add_argument("--all", action="store_true", help="Run all 8 benchmarks")
    parser.add_argument(
        "--dry-run", action="store_true", help="Use mock operations (no hardware)"
    )
    parser.add_argument(
        "--reflexion",
        action="store_true",
        help="Enable Reflexion LLM retry on operation failure (live mode only)",
    )
    parser.add_argument(
        "--output-dir",
        default="./benchmark_results",
        help="Directory for JSON result files (default: ./benchmark_results)",
    )
    parser.add_argument(
        "--task-count",
        type=int,
        default=5,
        help="Number of sub-tasks for B8 (default: 5)",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for servers to become ready (default: 120)",
    )
    args = parser.parse_args()

    runner = BenchmarkRunner()
    benchmark_ids = list(range(1, 9)) if args.all else [args.benchmark]

    if args.dry_run:
        exit_code = _run_benchmarks(runner, benchmark_ids, args)
    else:
        with ServerManager(startup_timeout=args.startup_timeout) as _mgr:
            exit_code = _run_benchmarks(runner, benchmark_ids, args)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
