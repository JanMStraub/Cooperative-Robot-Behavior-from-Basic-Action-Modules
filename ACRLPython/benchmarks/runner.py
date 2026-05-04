#!/usr/bin/env python3
"""
BenchmarkRunner — sends benchmark tasks to the running SequenceServer over TCP.

B1–B5 and B8 send natural language task strings; the LLM parses them into
operations. B6–B7 use explicit op lists with parallel_group fields that the
LLM cannot express, sent via the EXEC: prefix.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import random
import socket
import struct
import time
from typing import Any, Dict, List, Optional, Union

from . import mock_registry
from .config import BenchmarkConfig
from .result import BenchmarkResult, ChainMetrics, StepResult, make_run_id

# Protocol V2 constants (mirrors core/UnityProtocol.py)
_SEQUENCE_QUERY = 0x08
_RESULT_TYPE = 0x02
_HOST = "127.0.0.1"
_PORT = 5008
_DEFAULT_CAMERA = "TableStereoCamera"
_EXEC_PREFIX = "EXEC:"

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
        Run a single benchmark by sending a natural language task to the LLM.

        All benchmarks use get_task() → NL string → SequenceServer → LLM → ops.
        B8 chains multiple sub-tasks. B6/B7 use DualRobotConfig.

        Args:
            benchmark_id: Integer 1–8.
            cfg: BenchmarkConfig (or DualRobotConfig for B6–B8).

        Returns:
            BenchmarkResult with full metrics and step details.
        """
        mock_original = None
        try:
            if cfg.dry_run:
                mock_original = mock_registry.install_mock("always_succeed")

            module = importlib.import_module(_CASE_MODULES[benchmark_id])

            if benchmark_id == 8:
                return self._run_b8_chain(cfg, module)

            task = module.get_task(cfg)
            robot_id = getattr(cfg, "robot_id_a", cfg.robot_id)
            raw = self._send(task, robot_id, cfg)
            return self._build_result(benchmark_id, cfg, raw)

        finally:
            if mock_original is not None:
                mock_registry.restore_mock(mock_original)
            if not cfg.dry_run:
                self._reset(cfg)

    def _send(
        self,
        payload: Union[str, List[Dict[str, Any]]],
        robot_id: str,
        cfg: BenchmarkConfig,
    ) -> Dict[str, Any]:
        """
        Send a task string or op list to SequenceServer and return result dict.

        Natural language strings go as-is (LLM path).
        Op lists are prefixed with EXEC: to bypass the LLM.
        Dry-run mode executes in-process instead.

        Args:
            payload: NL task string → LLM parses; list of op dicts → EXEC: prefix.
            robot_id: Primary robot for this sequence.
            cfg: Benchmark config (timeout, dry_run).

        Returns:
            Result dict with keys: success, results, total_duration_ms.
        """
        if cfg.dry_run:
            ops = payload if isinstance(payload, list) else []
            return self._run_local(ops, cfg)

        if isinstance(payload, list):
            command_text = _EXEC_PREFIX + json.dumps(payload)
        else:
            command_text = payload

        timeout = cfg.timeout_per_step_s * 10  # generous: LLM + execution
        request_id = random.randint(1, 0xFFFFFFFF)

        cmd_b = command_text.encode("utf-8")
        rob_b = robot_id.encode("utf-8")
        cam_b = _DEFAULT_CAMERA.encode("utf-8")
        msg = struct.pack("<BI", _SEQUENCE_QUERY, request_id)
        msg += struct.pack("<I", len(cmd_b)) + cmd_b
        msg += struct.pack("<I", len(rob_b)) + rob_b
        msg += struct.pack("<I", len(cam_b)) + cam_b
        msg += struct.pack("<B", 1)  # auto_execute=True

        start = time.monotonic()
        with socket.create_connection((_HOST, _PORT), timeout=10) as sock:
            sock.sendall(msg)
            raw = self._read_response(sock, timeout)

        elapsed_ms = (time.monotonic() - start) * 1000.0
        raw.setdefault("total_duration_ms", elapsed_ms)
        return raw

    def _read_response(self, sock: socket.socket, timeout: float) -> Dict[str, Any]:
        """
        Read a RESULT response from SequenceServer.

        Args:
            sock: Connected socket.
            timeout: Seconds to wait.

        Returns:
            Parsed JSON response dict.
        """
        sock.settimeout(timeout + 30)
        header = b""
        while len(header) < 9:
            chunk = sock.recv(9 - len(header))
            if not chunk:
                raise ConnectionError("Server closed connection before response")
            header += chunk

        msg_type = header[0]
        resp_len = struct.unpack("<I", header[5:9])[0]

        if msg_type != _RESULT_TYPE:
            raise ValueError(f"Unexpected response type 0x{msg_type:02x}")

        body = b""
        while len(body) < resp_len:
            chunk = sock.recv(resp_len - len(body))
            if not chunk:
                raise ConnectionError("Server closed connection mid-response")
            body += chunk

        return json.loads(body.decode("utf-8"))

    def _reset(self, cfg: BenchmarkConfig) -> None:
        """
        Send reset_simulation to Unity via EXEC: prefix and wait for completion.

        Silently ignores errors — a failed reset should not fail the benchmark result.

        Args:
            cfg: Benchmark config (used for timeout).
        """
        try:
            reset_op = [{"operation": "reset_simulation", "params": {}}]
            self._send(reset_op, "system", cfg)
        except Exception:
            pass

    def _run_local(
        self, ops: List[Dict[str, Any]], cfg: BenchmarkConfig
    ) -> Dict[str, Any]:
        """
        Execute ops in-process via SequenceExecutor (dry-run only).

        Args:
            ops: Operation list.
            cfg: Benchmark config.

        Returns:
            Raw result dict from SequenceExecutor.
        """
        import orchestrators.SequenceExecutor as _seq_mod
        from orchestrators.SequenceExecutor import SequenceExecutor

        prev = _seq_mod.REFLEXION_ENABLED
        _seq_mod.REFLEXION_ENABLED = False
        try:
            executor = SequenceExecutor(
                default_timeout=cfg.timeout_per_step_s,
                check_completion=False,
                enable_verification=False,
            )
            return executor.execute_sequence(ops)
        finally:
            _seq_mod.REFLEXION_ENABLED = prev

    def _build_result(
        self,
        benchmark_id: int,
        cfg: BenchmarkConfig,
        raw: Dict[str, Any],
    ) -> BenchmarkResult:
        """
        Convert raw SequenceServer response into a BenchmarkResult.

        Args:
            benchmark_id: Benchmark identifier.
            cfg: Config used for this run.
            raw: Response dict.

        Returns:
            Populated BenchmarkResult.
        """
        steps = self._parse_steps(raw.get("results") or [])
        first_fail = next((s.index for s in steps if not s.success), None)
        total_ms = float(raw.get("total_duration_ms", 0.0))
        ops_executed = len(steps)
        ops_succeeded = sum(1 for s in steps if s.success)

        return BenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_name=_BENCHMARK_NAMES[benchmark_id],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            success=bool(raw.get("success", False)),
            total_duration_ms=total_ms,
            steps=steps,
            ops_executed=ops_executed,
            ops_succeeded=ops_succeeded,
            success_rate=(ops_succeeded / ops_executed) if ops_executed else 0.0,
            avg_step_duration_ms=(total_ms / ops_executed) if ops_executed else 0.0,
            first_failure_step=first_fail,
        )

    def _parse_steps(self, results: List[Optional[Dict[str, Any]]]) -> List[StepResult]:
        """
        Convert raw result dicts to StepResult list, skipping None entries.

        Args:
            results: Per-step result dicts from SequenceServer.

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

        Each sub-task is a natural language string sent as a separate sequence.

        Args:
            cfg: BenchmarkConfig with task_count controlling chain length.
            module: b8 cases module (exposes get_sub_tasks).

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

        for _, task in sub_tasks:
            raw = self._send(task, cfg.robot_id, cfg)
            total_ms += float(raw.get("total_duration_ms", 0.0))

            step_offset = len(all_steps)
            task_steps = self._parse_steps(raw.get("results") or [])
            for s in task_steps:
                s.index += step_offset
                if not s.success and s.error_code:
                    error_counts[s.error_code] = error_counts.get(s.error_code, 0) + 1
            all_steps.extend(task_steps)

            if not cfg.dry_run:
                self._reset(cfg)

            if raw.get("success"):
                completed += 1
            else:
                recovery_count += 1

        ops_executed = len(all_steps)
        ops_succeeded = sum(1 for s in all_steps if s.success)
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
            avg_step_duration_ms=(total_ms / ops_executed) if ops_executed else 0.0,
            first_failure_step=first_fail,
            chain_metrics=ChainMetrics(
                total_tasks=total,
                completed_tasks=completed,
                error_rate=error_rate,
                recovery_count=recovery_count,
                per_error_code=error_counts,
            ),
        )
