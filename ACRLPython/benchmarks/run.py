#!/usr/bin/env python3
"""
Benchmark CLI entry point.

Usage:
    python -m benchmarks.run --benchmark 3
    python -m benchmarks.run --all
    python -m benchmarks.run --benchmark 8 --task-count 20 --dry-run
    python -m benchmarks.run --all --output-dir ./results/
    python -m benchmarks.run --benchmark 1 --reflexion
"""

from __future__ import annotations

import argparse
import socket
import sys

from .config import BenchmarkConfig, DualRobotConfig
from .reporter import print_summary, write_json
from .runner import BenchmarkRunner

_DUAL_ROBOT_BENCHMARKS = {6, 7, 8, 11}
_PARSE_ONLY_BENCHMARKS = {9, 12}  # no server required
_REQUIRED_PORTS = (5007, 5008)


def _check_servers_running() -> None:
    """Abort with a clear message if required server ports are not open."""
    missing = []
    for port in _REQUIRED_PORTS:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            if sock.connect_ex(("localhost", port)) != 0:
                missing.append(port)
            sock.close()
        except Exception:
            missing.append(port)
    if missing:
        ports = ", ".join(str(p) for p in missing)
        print(
            f"ERROR: Required servers not reachable on port(s) {ports}.\n"
            "Start the backend first:\n"
            "  cd ACRLPython && ./start_servers.sh",
            file=sys.stderr,
        )
        sys.exit(1)


def _make_config(benchmark_id: int, args: argparse.Namespace) -> BenchmarkConfig:
    """
    Instantiate the appropriate config type for a given benchmark.

    Args:
        benchmark_id: Benchmark number 1–12.
        args: Parsed CLI arguments.

    Returns:
        BenchmarkConfig or DualRobotConfig instance.
    """
    live = not args.dry_run and benchmark_id not in _PARSE_ONLY_BENCHMARKS
    kwargs = dict(
        dry_run=args.dry_run,
        task_count=args.task_count,
        reflexion=args.reflexion,
        check_completion=live,
        use_rag=not args.no_rag,
        reflexion_enabled=args.reflexion,
        use_knowledge_graph=not args.no_kg,
        use_vgn=not args.no_vgn,
        use_ros_movement=not args.no_ros,
        execution_mode="live" if getattr(args, "live", False) else "offline",
    )
    if benchmark_id in _DUAL_ROBOT_BENCHMARKS:
        return DualRobotConfig(**kwargs, use_negotiation=not args.no_negotiation)
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
        choices=range(1, 15),
        metavar="N",
        help="Run a single benchmark (1–14); 9–14 are ablation benchmarks",
    )
    group.add_argument("--all", action="store_true", help="Run all benchmarks (1–14)")
    group.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation benchmarks only (9–14, no server required for 9 and 12)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Use mock operations (no hardware)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Run B10/B11 ablations against live SequenceServer instead of dry-run mocks",
    )
    parser.add_argument(
        "--reflexion",
        action="store_true",
        help="Enable Reflexion LLM retry on operation failure (live mode only)",
    )
    parser.add_argument(
        "--no-rag", action="store_true", help="Disable RAG retrieval (B9 ablation: disabled condition)"
    )
    parser.add_argument(
        "--no-kg", action="store_true", help="Disable Knowledge Graph context (B12 ablation: disabled condition)"
    )
    parser.add_argument(
        "--no-negotiation", action="store_true", help="Disable LLM negotiation (B11 ablation: disabled condition)"
    )
    parser.add_argument(
        "--no-vgn", action="store_true", help="Disable VGN neural grasp (B13 ablation: disabled condition)"
    )
    parser.add_argument(
        "--no-ros", action="store_true", help="Disable ROS/MoveIt movement (B14 ablation: disabled condition)"
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
    args = parser.parse_args()

    runner = BenchmarkRunner()
    if args.all:
        benchmark_ids = list(range(1, 15))
    elif args.ablation:
        benchmark_ids = list(range(9, 15))
    else:
        benchmark_ids = [args.benchmark]

    needs_server = any(
        (bid not in _PARSE_ONLY_BENCHMARKS and not args.dry_run)
        or (bid in {10, 11} and getattr(args, "live", False))
        for bid in benchmark_ids
    )
    if needs_server:
        _check_servers_running()

    exit_code = _run_benchmarks(runner, benchmark_ids, args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
