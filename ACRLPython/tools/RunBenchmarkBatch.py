#!/usr/bin/env python3
"""Batch-run live benchmarks across models, one server restart per (benchmark, model).

For each job in JOBS, for each model in that job's models: restarts
start_servers.sh with DEFAULT_LMSTUDIO_MODEL set to that model, waits for the
command/sequence servers to come up, runs benchmarks.Run for the
ablation-enabled condition then the ablation-disabled condition (job's
disable_flag), shuts the servers down cleanly, and copies the session's
server log into the model's results folder.

Assumes Unity is already running and connected (start_servers.sh's `open` on
an already-running build just refocuses it) and that LM Studio just-in-time
loads whichever model name is requested.

    cd ACRLPython && python3 tools/RunBenchmarkBatch.py
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPEAT = 15


@dataclass
class BenchmarkJob:
    benchmark: int
    disable_flag: str
    models: list[str] = field(default_factory=list)


JOBS = [
    BenchmarkJob(
        benchmark=12,
        disable_flag="--no-reflection",
        models=[
            "google/gemma-4-e2b",
            "google/gemma-4-e4b",
            "mistralai/ministral-3-14b-reasoning",
            "qwen/qwen3-vl-30b",
            "qwen/qwen3-vl-8b",
        ],  # magistral-small-2509 already has a complete additional_b12 run
    ),
    BenchmarkJob(
        benchmark=13,
        disable_flag="--no-negotiation",
        models=[
            "mistralai/magistral-small-2509",
            "mistralai/ministral-3-14b-reasoning",
        ],  # the other 4 models already have a complete additional_b13 run
    ),
]

ACRLPYTHON_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ACRLPYTHON_DIR / "logs"
REQUIRED_PORTS = (5007, 5008)
SERVER_READY_TIMEOUT = 120
SERVER_SHUTDOWN_TIMEOUT = 60


def model_short_name(model: str) -> str:
    return model.rsplit("/", 1)[-1]


def wait_for_ports(ports: tuple[int, ...], timeout: int) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(_port_open(p) for p in ports):
            return True
        time.sleep(1)
    return all(_port_open(p) for p in ports)


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_servers(model: str, startup_log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["DEFAULT_LMSTUDIO_MODEL"] = model
    startup_log = open(startup_log_path, "w")
    return subprocess.Popen(
        ["./start_servers.sh"],
        cwd=ACRLPYTHON_DIR,
        env=env,
        stdout=startup_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def stop_servers(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=SERVER_SHUTDOWN_TIMEOUT)
    except subprocess.TimeoutExpired:
        print("  WARNING: servers did not exit within timeout, sending SIGKILL")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=SERVER_SHUTDOWN_TIMEOUT)


def run_benchmark_condition(
    benchmark: int, out_dir: Path, extra_args: list[str]
) -> bool:
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.Run",
        "--benchmark",
        str(benchmark),
        "--live",
        "--repeat",
        str(REPEAT),
        "--output-dir",
        str(out_dir),
        *extra_args,
    ]
    result = subprocess.run(cmd, cwd=ACRLPYTHON_DIR)
    return result.returncode == 0


def copy_latest_server_log(out_dir: Path, model_short: str) -> Path | None:
    logs = sorted(LOG_DIR.glob("server_logs_*.txt"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    dest = out_dir / f"server_logs_{model_short}.txt"
    shutil.copy(logs[-1], dest)
    return dest


def run_job_model(job: BenchmarkJob, model: str) -> bool:
    model_short = model_short_name(model)
    out_dir = (
        ACRLPYTHON_DIR
        / "benchmark_results"
        / f"additional_b{job.benchmark}"
        / model_short
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    startup_log_path = LOG_DIR / f"batch_start_b{job.benchmark}_{model_short}.txt"

    print(f"\n=== B{job.benchmark} / {model_short} ===")
    print(f"Starting servers with DEFAULT_LMSTUDIO_MODEL={model} ...")
    proc = start_servers(model, startup_log_path)
    try:
        if not wait_for_ports(REQUIRED_PORTS, SERVER_READY_TIMEOUT):
            print(
                f"  ERROR: servers not ready after {SERVER_READY_TIMEOUT}s; "
                f"see {startup_log_path}"
            )
            tail = startup_log_path.read_text(errors="replace").splitlines()[-30:]
            print("  --- startup log tail ---")
            print("\n".join(f"  {line}" for line in tail))
            return False

        print("  Servers ready. Running enabled condition...")
        ok_on = run_benchmark_condition(job.benchmark, out_dir, [])
        print(f"  Running disabled condition ({job.disable_flag})...")
        ok_off = run_benchmark_condition(job.benchmark, out_dir, [job.disable_flag])
    finally:
        print("  Stopping servers...")
        stop_servers(proc)

    log_dest = copy_latest_server_log(out_dir, model_short)
    if log_dest:
        print(f"  Server log copied to {log_dest}")
    else:
        print("  WARNING: no server_logs_*.txt found to copy")

    success = ok_on and ok_off
    status = "OK" if success else "FAILED (see benchmarks.Run exit code above)"
    print(f"  B{job.benchmark} / {model_short}: {status}")
    return success


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[tuple[int, str], bool] = {}
    for job in JOBS:
        for model in job.models:
            results[(job.benchmark, model)] = run_job_model(job, model)

    print("\n=== Summary ===")
    for (benchmark, model), ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  B{benchmark}  {model}")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
