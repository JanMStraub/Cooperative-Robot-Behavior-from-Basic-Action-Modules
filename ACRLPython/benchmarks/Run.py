#!/usr/bin/env python3
"""
Benchmark CLI entry point.

Usage:
    python -m benchmarks.run --benchmark 3
    python -m benchmarks.run --benchmark 1-6 --live
    python -m benchmarks.run --all
    python -m benchmarks.run --benchmark 8 --task-count 20 --dry-run
    python -m benchmarks.run --all --output-dir ./results/
    python -m benchmarks.run --benchmark 1 --no-reflection
"""

from __future__ import annotations

from typing import Any

import argparse
import socket
import sys

from .Config import BenchmarkConfig, DualRobotConfig
from .Reporter import print_summary, write_json
from .Runner import BenchmarkRunner

_DUAL_ROBOT_BENCHMARKS = {6, 7, 8, 10, 13}  # B10=Parallel, B13=Negotiation
_PARSE_ONLY_BENCHMARKS = {
    9,
    11,
    14,
    17,
}  # no server required: B9=Impossible, B11=RAG, B14=KG, B17=AutoRT
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


def _parse_benchmark_arg(value: str) -> list[int]:
    """Parse '3' or '1-6' into a list of benchmark IDs."""
    if "-" in value:
        parts = value.split("-", 1)
        try:
            lo, hi = int(parts[0]), int(parts[1])
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid benchmark range: {value!r}")
        if lo < 1 or hi > 17 or lo > hi:
            raise argparse.ArgumentTypeError(
                f"Range must be within 1-17 and lo ≤ hi, got {value!r}"
            )
        return list(range(lo, hi + 1))
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid benchmark id: {value!r}")
    if n < 1 or n > 17:
        raise argparse.ArgumentTypeError(f"Benchmark id must be 1-17, got {n}")
    return [n]


def _make_config(benchmark_id: int, args: argparse.Namespace) -> BenchmarkConfig:
    """
    Instantiate the appropriate config type for a given benchmark.
    """
    live = not args.dry_run and benchmark_id not in _PARSE_ONLY_BENCHMARKS
    kwargs: dict[str, Any] = dict(
        dry_run=args.dry_run or benchmark_id in _PARSE_ONLY_BENCHMARKS,
        task_count=args.task_count,
        reflection=not args.no_reflection,
        check_completion=live,
        use_rag=not args.no_rag,
        reflection_enabled=not args.no_reflection,
        use_knowledge_graph=not args.no_kg,
        use_vgn=not args.no_vgn,
        use_ros_movement=not args.no_ros,
        execution_mode="live" if getattr(args, "live", False) else "offline",
    )
    if benchmark_id in _DUAL_ROBOT_BENCHMARKS:
        return DualRobotConfig(**kwargs, use_negotiation=not args.no_negotiation)
    return BenchmarkConfig(**kwargs)


def _is_embedding_model(m: dict) -> bool:
    """True if an LM Studio /models entry is an embedding model, not a chat LLM.

    When RAG is enabled the embedding model is loaded alongside the chat model
    and LM Studio lists both; it must be skipped so it isn't recorded as the
    model under test. Checks the explicit ``type`` field (LM Studio's richer
    endpoint) and falls back to the id (the configured RAG embedder, or any id
    containing "embed").
    """
    if (m.get("type") or "").lower() in {"embeddings", "embedding"}:
        return True
    model_id = (m.get("id") or "").lower()
    try:
        from config.Rag import RAG_LM_STUDIO_MODEL

        if model_id == RAG_LM_STUDIO_MODEL.lower():
            return True
    except Exception:
        pass
    return "embed" in model_id


def _detect_model(args: argparse.Namespace) -> str:
    """
    Resolve the LLM model under test for result tagging.

    Priority: explicit --model flag > live LM Studio /models > configured default.
    Recorded in each result so analysis can break results out by model instead of
    relying on the output directory layout.
    """
    if getattr(args, "model", None):
        return args.model
    try:
        from config.Servers import LMSTUDIO_BASE_URL
        import json as _json
        import urllib.request

        url = LMSTUDIO_BASE_URL.rstrip("/")
        req = urllib.request.Request(f"{url}/models", method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = _json.loads(resp.read())
        models = data.get("data") or []
        # First non-embedding model. /models lists the embedding model too when
        # RAG is loaded, and it is often first - taking models[0] blindly tagged
        # every run with the embedder.
        for m in models:
            if m.get("id") and not _is_embedding_model(m):
                return m["id"]
    except Exception:
        pass
    try:
        from config.Servers import DEFAULT_LMSTUDIO_MODEL

        return DEFAULT_LMSTUDIO_MODEL
    except Exception:
        return ""


def _run_benchmarks(
    runner: BenchmarkRunner,
    benchmark_ids: list,
    args: argparse.Namespace,
) -> int:
    """
    Execute the requested benchmarks and write results.
    """
    exit_code = 0
    model = _detect_model(args)
    for bid in benchmark_ids:
        cfg = _make_config(bid, args)
        result = runner.run(bid, cfg)
        result.model = model
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
        type=str,
        metavar="N or N-M",
        help="Run benchmark(s) 1-17; accepts single id (e.g. 3) or range (e.g. 1-6)",
    )
    group.add_argument("--all", action="store_true", help="Run all benchmarks (1-17)")
    group.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation benchmarks only (11-17, no server required for 11, 14 and 17)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Use mock operations (no hardware)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Run B12/B13 ablations against live SequenceServer instead of dry-run mocks",
    )
    parser.add_argument(
        "--no-reflection",
        action="store_true",
        help="Disable Reflection LLM retry on operation failure (B12 ablation: disabled condition)",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG retrieval (B11 ablation: disabled condition)",
    )
    parser.add_argument(
        "--no-kg",
        action="store_true",
        help="Disable Knowledge Graph context (B14 ablation: disabled condition)",
    )
    parser.add_argument(
        "--no-negotiation",
        action="store_true",
        help="Disable LLM negotiation (B13 ablation: disabled condition)",
    )
    parser.add_argument(
        "--no-vgn",
        action="store_true",
        help="Disable VGN neural grasp (B15 ablation: disabled condition)",
    )
    parser.add_argument(
        "--no-ros",
        action="store_true",
        help="Disable ROS/MoveIt movement (B16 ablation: disabled condition)",
    )
    parser.add_argument(
        "--output-dir",
        default="./benchmark_results",
        help="Directory for JSON result files (default: ./benchmark_results)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="LLM model id to tag results with (default: auto-detect from LM Studio)",
    )
    parser.add_argument(
        "--task-count",
        type=int,
        default=5,
        help="Number of sub-tasks for B8 (default: 5)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="Run the selected benchmarks N times in sequence (default: 1)",
    )
    args = parser.parse_args()

    runner = BenchmarkRunner()
    if args.all:
        benchmark_ids = list(range(1, 18))
    elif args.ablation:
        benchmark_ids = list(range(11, 18))
    else:
        try:
            benchmark_ids = _parse_benchmark_arg(args.benchmark)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))

    needs_server = any(
        (bid not in _PARSE_ONLY_BENCHMARKS and not args.dry_run)
        or (bid in {12, 13} and getattr(args, "live", False))
        for bid in benchmark_ids
    )
    if needs_server:
        _check_servers_running()

    exit_code = 0
    for run_index in range(args.repeat):
        if args.repeat > 1:
            print(f"\n=== Run {run_index + 1}/{args.repeat} ===")
        code = _run_benchmarks(runner, benchmark_ids, args)
        if code != 0:
            exit_code = code
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
