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
import os
import random
import socket
import struct
import time
from typing import Any, Dict, List, Optional, Union

from . import MockRegistry
from .Config import BenchmarkConfig
from .FeatureFlags import BenchmarkFeatureFlags
from .Result import BenchmarkResult, ChainMetrics, StepResult, make_run_id

# Protocol V2 constants (mirrors core/UnityProtocol.py)
_SEQUENCE_QUERY = 0x08
_RESULT_TYPE = 0x02
_HOST = os.environ.get("ACRL_HOST", "127.0.0.1")
_PORT = 5008
_DEFAULT_CAMERA = "TableStereoCamera"
_EXEC_PREFIX = "EXEC:"

_BENCHMARK_NAMES: Dict[int, str] = {
    1: "Navigate to Object",
    2: "Sequential Navigation",
    3: "Navigate and Lift",
    4: "Pick and Place",
    5: "Pose-Aware Grasp",
    6: "Robot Handoff",
    7: "Dual-Robot Reorient",
    8: "Heterogeneous Chain",
    9: "Impossible Task",
    10: "Parallel Independent Tasks",
    11: "RAG Ablation",
    12: "Reflexion Ablation",
    13: "Negotiation Ablation",
    14: "Knowledge Graph Ablation",
    15: "VGN Ablation",
    16: "ROS vs Unity Movement",
}

_CASE_MODULES: Dict[int, str] = {
    1: "benchmarks.cases.B1NavigateToObject",
    2: "benchmarks.cases.B2SequentialNavigation",
    3: "benchmarks.cases.B3NavigateAndLift",
    4: "benchmarks.cases.B4PickAndPlace",
    5: "benchmarks.cases.B5PoseAwareGrasp",
    6: "benchmarks.cases.B6RobotHandoff",
    7: "benchmarks.cases.B7DualRobotReorient",
    8: "benchmarks.cases.B8HeterogeneousChain",
    9: "benchmarks.cases.B9Impossible",
    10: "benchmarks.cases.B10ParallelIndependent",
    11: "benchmarks.cases.B11RagAblation",
    12: "benchmarks.cases.B12ReflexionAblation",
    13: "benchmarks.cases.B13NegotiationAblation",
    14: "benchmarks.cases.B14KgAblation",
    15: "benchmarks.cases.B15VGNAblation",
    16: "benchmarks.cases.B16RosAblation",
}


class BenchmarkRunner:
    """Executes individual benchmarks and returns structured BenchmarkResult objects."""

    def run(self, benchmark_id: int, cfg: BenchmarkConfig) -> BenchmarkResult:
        """
        Run a single benchmark by sending a natural language task to the LLM.

        All benchmarks use get_task() → NL string → SequenceServer → LLM → ops.
        B8 chains multiple sub-tasks. B6/B7 use DualRobotConfig.


        """
        mock_original = None
        try:
            if cfg.dry_run:
                mock_original = MockRegistry.install_mock("always_succeed")

            module = importlib.import_module(_CASE_MODULES[benchmark_id])

            if benchmark_id == 9:
                return self._run_b9_impossible(cfg, module)
            if benchmark_id == 10:
                return self._run_b10_parallel(cfg, module)
            if benchmark_id == 11:
                return self._run_b11_rag(cfg, module)
            if benchmark_id == 12:
                return self._run_b12_reflexion(cfg, module)
            if benchmark_id == 13:
                return self._run_b13_negotiation(cfg, module)
            if benchmark_id == 14:
                return self._run_b14_kg(cfg, module)
            if benchmark_id == 15:
                return self._run_b15_vgn(cfg, module)

            if benchmark_id == 16:
                return self._run_b16_ros(cfg, module)

            if benchmark_id == 8:
                return self._run_b8_chain(cfg, module)

            task = module.get_task()
            robot_id = getattr(cfg, "robot_id_a", cfg.robot_id)
            raw = self._send(task, robot_id, cfg)
            return self._build_result(benchmark_id, cfg, raw)

        finally:
            if mock_original is not None:
                MockRegistry.restore_mock(mock_original)
            if not cfg.dry_run:
                self._reset(cfg)

    def _build_sequence_message(
        self,
        payload: Union[str, List[Dict[str, Any]]],
        robot_id: str,
        flags: BenchmarkFeatureFlags,
    ) -> bytes:
        """
        Build the raw SEQUENCE_QUERY bytes for the given payload and flags.


        """
        if isinstance(payload, list):
            command_text = _EXEC_PREFIX + json.dumps(payload)
        else:
            command_text = payload

        request_id = random.randint(1, 0xFFFFFFFF)
        cmd_b = command_text.encode("utf-8")
        rob_b = robot_id.encode("utf-8")
        cam_b = _DEFAULT_CAMERA.encode("utf-8")
        flags_b = flags.to_json().encode("utf-8")

        msg = struct.pack("<BI", _SEQUENCE_QUERY, request_id)
        msg += struct.pack("<I", len(cmd_b)) + cmd_b
        msg += struct.pack("<I", len(rob_b)) + rob_b
        msg += struct.pack("<I", len(cam_b)) + cam_b
        msg += struct.pack("<B", 1)  # auto_execute=True
        msg += struct.pack("<I", len(flags_b)) + flags_b
        return msg

    def _send(
        self,
        payload: Union[str, List[Dict[str, Any]]],
        robot_id: str,
        cfg: BenchmarkConfig,
        flags: Optional[BenchmarkFeatureFlags] = None,
    ) -> Dict[str, Any]:
        """
        Send a task string or op list to SequenceServer and return result dict.

        Natural language strings go as-is (LLM path).
        Op lists are prefixed with EXEC: to bypass the LLM.
        Dry-run mode executes in-process instead.


        """
        if cfg.dry_run:
            if isinstance(payload, list):
                ops = payload
            else:
                # NL string in dry_run: parse via CommandParser so B1-B8 get real ops
                from orchestrators.CommandParser import CommandParser

                parse_result = CommandParser(use_rag=cfg.use_rag).parse(
                    payload, robot_id=robot_id
                )
                ops = (
                    parse_result.get("commands", [])
                    if parse_result.get("success")
                    else []
                )
            result = self._run_local(ops, cfg)
            # Inject parsed_commands so _build_result can extract robot_id and
            # parallel_group_id per step, matching the live-mode structure.
            result.setdefault("parsed_commands", ops)
            return result

        if flags is None:
            flags = BenchmarkFeatureFlags()

        timeout = cfg.timeout_per_step_s * 10  # generous: LLM + execution
        msg = self._build_sequence_message(payload, robot_id, flags)

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

        """
        try:
            reset_op = [{"operation": "reset_simulation", "params": {}}]
            self._send(reset_op, "system", cfg)
        except Exception:
            pass

    def _extract_feature_flags(self, cfg: BenchmarkConfig) -> dict:
        """
        Extract active feature flags from a BenchmarkConfig as a flat, indexed dict.

        Having these at the top level of BenchmarkResult makes them trivially
        queryable without digging into config_snapshot.


        """
        return {
            "use_rag": getattr(cfg, "use_rag", False),
            "use_vgn": getattr(cfg, "use_vgn", False),
            "use_knowledge_graph": getattr(cfg, "use_knowledge_graph", False),
            "use_ros_movement": getattr(cfg, "use_ros_movement", False),
            "reflexion_enabled": getattr(cfg, "reflexion_enabled", False),
            "dry_run": getattr(cfg, "dry_run", False),
            "use_negotiation": getattr(cfg, "use_negotiation", False),
        }

    def _compute_per_op_stats(self, steps: List[StepResult]) -> dict:
        """
        Aggregate per-operation statistics from a step list.

        Useful for thesis analysis of which operation types fail most and their
        timing characteristics.


        """
        buckets: Dict[str, Dict[str, Any]] = {}
        for s in steps:
            b = buckets.setdefault(
                s.operation, {"count": 0, "fail_count": 0, "total_ms": 0.0}
            )
            b["count"] += 1
            b["total_ms"] += s.duration_ms
            if not s.success:
                b["fail_count"] += 1
        return {
            op: {
                "count": v["count"],
                "fail_count": v["fail_count"],
                "avg_duration_ms": (
                    round(v["total_ms"] / v["count"], 1) if v["count"] else 0.0
                ),
            }
            for op, v in buckets.items()
        }

    def _run_local(
        self, ops: List[Dict[str, Any]], cfg: BenchmarkConfig
    ) -> Dict[str, Any]:
        """
        Execute ops in-process via SequenceExecutor (dry-run only).


        """
        import orchestrators.SequenceExecutor as _seq_mod
        from orchestrators.SequenceExecutor import SequenceExecutor

        prev = _seq_mod.REFLEXION_ENABLED
        _seq_mod.REFLEXION_ENABLED = getattr(cfg, "reflexion_enabled", False)
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


        """
        parsed_cmds = raw.get("parsed_commands") or []
        steps = self._parse_steps(raw.get("results") or [], parsed_cmds)
        first_fail = next((s.index for s in steps if not s.success), None)
        total_ms = float(raw.get("total_duration_ms", 0.0))
        ops_executed = len(steps)
        ops_succeeded = sum(1 for s in steps if s.success)

        total_retries = sum(s.retry_count for s in steps)
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
            retry_count=total_retries,
            first_failure_step=first_fail,
            feature_flags=self._extract_feature_flags(cfg),
            parsed_plan=[c.get("operation", "") for c in parsed_cmds],
            per_op_stats=self._compute_per_op_stats(steps),
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )

    def _parse_steps(
        self,
        results: List[Optional[Dict[str, Any]]],
        parsed_cmds: Optional[List[Dict[str, Any]]] = None,
    ) -> List[StepResult]:
        """
        Convert raw result dicts to StepResult list, skipping None entries.

        When parsed_cmds is provided (from raw["parsed_commands"] or injected
        in dry-run), robot_id and parallel_group_id are populated per step by
        matching on step index.


        """
        # Build a quick lookup from step index → parsed command dict
        cmd_by_index: Dict[int, Dict[str, Any]] = {}
        if parsed_cmds:
            for i, cmd in enumerate(parsed_cmds):
                cmd_by_index[i] = cmd

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

            step_index = r["index"]
            cmd = cmd_by_index.get(step_index, {})
            steps.append(
                StepResult(
                    index=step_index,
                    operation=r["operation"],
                    success=bool(r.get("success", False)),
                    duration_ms=float(r.get("duration_ms", 0.0)),
                    error_code=error_code,
                    error_message=error_message,
                    retry_count=int(r.get("retry_count", 0)),
                    robot_id=cmd.get("params", {}).get("robot_id") or None,
                    parallel_group_id=cmd.get("parallel_group") or None,
                )
            )
        return steps

    def _run_b10_parallel(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B10 Parallel Independent Tasks.

        Sends a single NL prompt covering both robots doing independent work.
        The LLM is expected to assign Robot1 ops and Robot2 ops to the same
        parallel_group so both chains execute concurrently.

        Success = execution success AND parallelism_ratio >= 0.5.
        parallelism_ratio = ops_in_shared_groups / total_ops, where a "shared"
        group contains ops from more than one robot_id.
        """
        task = module.get_task()
        robot_id = getattr(cfg, "robot_id_a", cfg.robot_id)
        raw = self._send(task, robot_id, cfg)
        result = self._build_result(10, cfg, raw)

        parallelism_ratio, ops_in_parallel = self._compute_parallelism_ratio(
            result.steps
        )
        parallelism_success = parallelism_ratio >= 0.5

        return dataclasses.replace(
            result,
            success=result.success and parallelism_success,
            per_op_stats={
                **result.per_op_stats,
                "_parallelism_ratio": parallelism_ratio,
                "_ops_in_parallel": ops_in_parallel,
                "_parallelism_success": parallelism_success,
            },
        )

    def _compute_parallelism_ratio(self, steps: List[StepResult]) -> tuple[float, int]:
        """
        Return (parallelism_ratio, ops_in_shared_groups).

        A parallel group is "shared" if ops from more than one robot_id appear in it.
        parallelism_ratio = ops_in_shared_groups / len(steps).
        Returns (0.0, 0) when no steps have parallel_group_id set (sequential scheduling).
        """
        from collections import defaultdict

        if not steps:
            return 0.0, 0

        group_robots: Dict[int, set] = defaultdict(set)
        for s in steps:
            if s.parallel_group_id is not None:
                group_robots[s.parallel_group_id].add(s.robot_id or "unknown")

        shared = {gid for gid, robots in group_robots.items() if len(robots) > 1}
        ops_in_shared = sum(1 for s in steps if s.parallel_group_id in shared)
        return round(ops_in_shared / len(steps), 4), ops_in_shared

    def _run_b8_chain(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B8 heterogeneous chain benchmark.

        Each sub-task is a natural language string sent as a separate sequence.


        """
        sub_tasks = module.get_sub_tasks(cfg, cfg.task_count)

        total = len(sub_tasks)
        completed = 0
        error_counts: Dict[str, int] = {}
        recovery_count = 0
        all_steps: List[StepResult] = []
        total_ms = 0.0
        phase_results: Dict[str, List[bool]] = {"phase_a": [], "phase_b": [], "phase_c": []}

        for robot_id, task_name, task in sub_tasks:
            phase = next(
                (p for p in ("phase_a", "phase_b", "phase_c") if p in task_name),
                "phase_a",
            )
            raw = self._send(task, robot_id, cfg)
            total_ms += float(raw.get("total_duration_ms", 0.0))

            step_offset = len(all_steps)
            task_steps = self._parse_steps(
                raw.get("results") or [], raw.get("parsed_commands") or []
            )
            for s in task_steps:
                s.index += step_offset
                if not s.success and s.error_code:
                    error_counts[s.error_code] = error_counts.get(s.error_code, 0) + 1
            all_steps.extend(task_steps)

            success = bool(raw.get("success"))
            phase_results[phase].append(success)
            if success:
                completed += 1
            else:
                recovery_count += 1

        if not cfg.dry_run:
            self._reset(cfg)

        per_phase_success = {
            phase: round(sum(vals) / len(vals), 4) if vals else 0.0
            for phase, vals in phase_results.items()
        }

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
                per_phase_success=per_phase_success,
            ),
            feature_flags=self._extract_feature_flags(cfg),
            per_op_stats=self._compute_per_op_stats(all_steps),
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )

    def _run_b9_impossible(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B9 Impossible Task: verify the parser gracefully rejects an unexecutable task.

        A graceful rejection means the parser explicitly returned success=False with
        zero commands — NOT that it raised an exception. Crashes are failures.
        No Unity connection required — parse-only.


        """
        from orchestrators.CommandParser import CommandParser

        task = module.get_task()
        parser = CommandParser(use_rag=cfg.use_rag)

        parse_exception: Optional[Exception] = None
        parse_result: Dict[str, Any] = {}
        try:
            parse_result = parser.parse(task, robot_id=cfg.robot_id)
        except Exception as exc:
            parse_exception = exc

        if parse_exception is not None:
            # Parser threw — this is a bug, not a graceful rejection
            gracefully_rejected = False
        else:
            commands = parse_result.get("commands", [])
            parse_failed = not parse_result.get("success", True)
            gracefully_rejected = parse_failed or len(commands) == 0

        return BenchmarkResult(
            benchmark_id=9,
            benchmark_name=_BENCHMARK_NAMES[9],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            success=gracefully_rejected,
            total_duration_ms=0.0,
            steps=[],
            ops_executed=len(parse_result.get("commands", [])),
            ops_succeeded=0,
            success_rate=1.0 if gracefully_rejected else 0.0,
            avg_step_duration_ms=0.0,
            feature_flags=self._extract_feature_flags(cfg),
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )

    def _run_b11_rag(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B11 RAG ablation: parse all tasks under both conditions in one pass.

        Returns AblationMetrics for enabled and disabled conditions so a single
        invocation produces a self-contained paired comparison.
        No Unity connection required — parse-only.


        """
        from orchestrators.CommandParser import CommandParser
        from core.Imports import get_global_registry
        from .Result import AblationMetrics

        registry = get_global_registry()
        tasks = module.get_tasks(cfg)
        condition_results: Dict[str, AblationMetrics] = {}

        for use_rag in (True, False):
            parser = CommandParser(use_rag=use_rag)
            hallucinated = 0
            succeeded = 0
            total_ops = 0

            for task in tasks:
                parse_result = parser.parse(task, robot_id=cfg.robot_id)
                if not parse_result["success"]:
                    continue
                for cmd in parse_result.get("commands", []):
                    op_name = cmd.get("operation", "")
                    total_ops += 1
                    if registry.get_operation_by_name(op_name) is None:
                        hallucinated += 1
                    else:
                        succeeded += 1

            condition = "enabled" if use_rag else "disabled"
            condition_results[condition] = AblationMetrics(
                condition=condition,
                hallucinated_ops=hallucinated,
                reflexion_recoveries=0,
                negotiation_rounds=0,
                success_rate=(succeeded / total_ops) if total_ops else 0.0,
                ops_executed=total_ops,
                ops_succeeded=succeeded,
            )

        ablation_on = condition_results["enabled"]
        ablation_off = condition_results["disabled"]

        # Delta: positive = RAG reduced hallucinations / improved success
        hallucination_delta = (
            ablation_off.hallucinated_ops - ablation_on.hallucinated_ops
        )
        success_rate_delta = ablation_on.success_rate - ablation_off.success_rate
        ablation_on = dataclasses.replace(
            ablation_on,
            condition_delta=round(success_rate_delta, 4),
            hallucination_delta=hallucination_delta,
        )

        # Success: both conditions executed ops, and RAG must not be worse
        success = (
            ablation_on.ops_executed > 0
            and ablation_off.ops_executed > 0
            and ablation_on.hallucinated_ops <= ablation_off.hallucinated_ops
        )

        return BenchmarkResult(
            benchmark_id=11,
            benchmark_name=_BENCHMARK_NAMES[11],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            success=success,
            total_duration_ms=0.0,
            steps=[],
            ops_executed=ablation_on.ops_executed,
            ops_succeeded=ablation_on.ops_succeeded,
            success_rate=ablation_on.success_rate,
            avg_step_duration_ms=0.0,
            hallucinated_ops=ablation_on.hallucinated_ops,
            ablation=ablation_on,
            ablation_baseline=ablation_off,
            feature_flags=self._extract_feature_flags(cfg),
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )

    def _run_b12_reflexion_live(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Send B11 tasks to live SequenceServer; measure reflexion recoveries from server response.


        """
        from .Result import AblationMetrics

        tasks = module.get_tasks(cfg)
        total_recoveries = 0
        completed = 0
        ops_executed = 0
        ops_succeeded = 0
        all_steps: List[StepResult] = []
        total_ms = 0.0
        _flags = BenchmarkFeatureFlags(use_reflexion=cfg.reflexion_enabled)

        for task in tasks:
            raw = self._send(task, cfg.robot_id, cfg, flags=_flags)
            if raw.get("success"):
                completed += 1
            total_recoveries += raw.get("reflexion_recoveries", 0)
            task_steps = self._parse_steps(
                raw.get("results") or [], raw.get("parsed_commands") or []
            )
            step_offset = len(all_steps)
            for s in task_steps:
                s.index += step_offset
            all_steps.extend(task_steps)
            ops_executed += raw.get("ops_executed", len(task_steps))
            ops_succeeded += raw.get(
                "ops_succeeded", sum(1 for s in task_steps if s.success)
            )
            total_ms += float(raw.get("total_duration_ms", 0.0))

        success_rate = completed / len(tasks) if tasks else 0.0
        ablation = AblationMetrics(
            condition="enabled" if cfg.reflexion_enabled else "disabled",
            hallucinated_ops=0,
            reflexion_recoveries=total_recoveries,
            negotiation_rounds=0,
            success_rate=success_rate,
            ops_executed=ops_executed,
            ops_succeeded=ops_succeeded,
        )
        return BenchmarkResult(
            benchmark_id=12,
            benchmark_name=_BENCHMARK_NAMES[12],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            success=success_rate > 0.5,
            total_duration_ms=total_ms,
            steps=all_steps,
            ops_executed=ops_executed,
            ops_succeeded=ops_succeeded,
            success_rate=success_rate,
            avg_step_duration_ms=(total_ms / ops_executed) if ops_executed else 0.0,
            reflexion_recoveries=total_recoveries,
            ablation=ablation,
            feature_flags=self._extract_feature_flags(cfg),
            per_op_stats=self._compute_per_op_stats(all_steps),
            execution_mode="live",
        )

    def _run_b12_reflexion(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B12 Reflexion ablation.

        In live mode (cfg.execution_mode == "live"), sends NL tasks to SequenceServer and
        reads reflexion_recoveries from the server response.

        In offline mode, uses first_fail_nav mock and dry-run execution.


        """
        if cfg.execution_mode == "live":
            return self._run_b12_reflexion_live(cfg, module)

        from orchestrators.CommandParser import CommandParser
        from .Result import AblationMetrics
        import orchestrators.SequenceExecutor as _seq_mod

        tasks = module.get_tasks(cfg)
        condition_results: Dict[str, AblationMetrics] = {}
        condition_steps: Dict[str, List[StepResult]] = {}

        for reflexion_on in (True, False):
            all_steps: List[StepResult] = []
            total_ms = 0.0
            total_recoveries = 0
            completed_tasks = 0

            prev_reflexion = _seq_mod.REFLEXION_ENABLED
            _seq_mod.REFLEXION_ENABLED = reflexion_on
            mock_original = MockRegistry.install_mock("first_fail_nav")
            try:
                parser = CommandParser(use_rag=cfg.use_rag)
                for task in tasks:
                    MockRegistry.reset_counts()
                    parse_result = parser.parse(task, robot_id=cfg.robot_id)
                    if not parse_result["success"]:
                        continue
                    ops = parse_result["commands"]
                    for op in ops:
                        op["_original_text"] = task
                    cfg_local = dataclasses.replace(cfg, dry_run=True)
                    raw = self._run_local(ops, cfg_local)
                    total_ms += float(raw.get("total_duration_ms", 0.0))
                    total_recoveries += raw.get("reflexion_recoveries", 0)
                    task_steps = self._parse_steps(raw.get("results") or [], ops)
                    step_offset = len(all_steps)
                    for s in task_steps:
                        s.index += step_offset
                    all_steps.extend(task_steps)
                    if raw.get("success"):
                        completed_tasks += 1
            finally:
                MockRegistry.restore_mock(mock_original)
                _seq_mod.REFLEXION_ENABLED = prev_reflexion

            ops_executed = len(all_steps)
            ops_succeeded = sum(1 for s in all_steps if s.success)
            success_rate = (completed_tasks / len(tasks)) if tasks else 0.0
            condition = "enabled" if reflexion_on else "disabled"
            condition_results[condition] = AblationMetrics(
                condition=condition,
                hallucinated_ops=0,
                reflexion_recoveries=total_recoveries,
                negotiation_rounds=0,
                success_rate=success_rate,
                ops_executed=ops_executed,
                ops_succeeded=ops_succeeded,
            )
            condition_steps[condition] = all_steps

        ablation_on = condition_results["enabled"]
        ablation_off = condition_results["disabled"]
        steps_on = condition_steps["enabled"]

        delta = round(ablation_on.success_rate - ablation_off.success_rate, 4)
        ablation_on = dataclasses.replace(ablation_on, condition_delta=delta)

        # Success: enabled condition executed ops and is not worse than disabled
        success = (
            ablation_on.ops_executed > 0
            and ablation_on.success_rate >= ablation_off.success_rate
        )

        return BenchmarkResult(
            benchmark_id=12,
            benchmark_name=_BENCHMARK_NAMES[12],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            success=success,
            total_duration_ms=0.0,
            steps=steps_on,
            ops_executed=ablation_on.ops_executed,
            ops_succeeded=ablation_on.ops_succeeded,
            success_rate=ablation_on.success_rate,
            avg_step_duration_ms=0.0,
            reflexion_recoveries=ablation_on.reflexion_recoveries,
            ablation=ablation_on,
            ablation_baseline=ablation_off,
            feature_flags=self._extract_feature_flags(cfg),
            per_op_stats=self._compute_per_op_stats(steps_on),
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )

    def _run_b13_negotiation_live(
        self, cfg: BenchmarkConfig, module
    ) -> BenchmarkResult:
        """
        Send B13 dual-robot tasks to live SequenceServer; measure negotiation rounds via hub.


        """
        from core.Imports import get_negotiation_hub
        from .Result import AblationMetrics

        tasks = module.get_tasks(cfg)
        total_rounds = 0
        completed = 0
        ops_executed = 0
        ops_succeeded = 0
        all_steps: List[StepResult] = []
        total_ms = 0.0
        hub = get_negotiation_hub()
        robot_id = getattr(cfg, "robot_id_a", cfg.robot_id)
        _flags = BenchmarkFeatureFlags(
            use_negotiation=getattr(cfg, "use_negotiation", True)
        )

        for task in tasks:
            raw = self._send(task, robot_id, cfg, flags=_flags)
            if raw.get("success"):
                completed += 1
            task_steps = self._parse_steps(raw.get("results") or [])
            step_offset = len(all_steps)
            for s in task_steps:
                s.index += step_offset
            all_steps.extend(task_steps)
            ops_executed += raw.get("ops_executed", len(task_steps))
            ops_succeeded += raw.get(
                "ops_succeeded", sum(1 for s in task_steps if s.success)
            )
            total_ms += float(raw.get("total_duration_ms", 0.0))
            if hub is not None:
                total_rounds += hub.get_last_round_count()

        success_rate = completed / len(tasks) if tasks else 0.0
        ablation = AblationMetrics(
            condition=(
                "enabled" if getattr(cfg, "use_negotiation", True) else "disabled"
            ),
            hallucinated_ops=0,
            reflexion_recoveries=0,
            negotiation_rounds=total_rounds,
            success_rate=success_rate,
            ops_executed=ops_executed,
            ops_succeeded=ops_succeeded,
        )
        return BenchmarkResult(
            benchmark_id=13,
            benchmark_name=_BENCHMARK_NAMES[13],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            success=success_rate > 0.5,
            total_duration_ms=total_ms,
            steps=all_steps,
            ops_executed=ops_executed,
            ops_succeeded=ops_succeeded,
            success_rate=success_rate,
            avg_step_duration_ms=(total_ms / ops_executed) if ops_executed else 0.0,
            negotiation_rounds=total_rounds,
            ablation=ablation,
        )

    def _run_b13_negotiation(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B13 Negotiation ablation.

        In live mode (cfg.execution_mode == "live"), sends dual-robot NL tasks to
        SequenceServer and reads negotiation_rounds from the hub.

        In offline mode, patches NEGOTIATION_ENABLED, runs dual-robot tasks in dry-run,
        and reads negotiation_rounds from the hub after each task.


        """
        if cfg.execution_mode == "live":
            return self._run_b13_negotiation_live(cfg, module)

        import config.Negotiation as _neg_cfg
        from orchestrators.CommandParser import CommandParser
        from .Result import AblationMetrics

        tasks = module.get_tasks(cfg)
        robot_id = getattr(cfg, "robot_id_a", cfg.robot_id)
        condition_results: Dict[str, AblationMetrics] = {}

        for negotiation_on in (True, False):
            all_steps: List[StepResult] = []
            total_ms = 0.0
            completed_tasks = 0
            total_negotiation_rounds = 0

            prev_neg = _neg_cfg.NEGOTIATION_ENABLED
            _neg_cfg.NEGOTIATION_ENABLED = negotiation_on
            mock_original = MockRegistry.install_mock("always_succeed")
            try:
                from core.Imports import get_negotiation_hub

                hub = get_negotiation_hub()

                parser = CommandParser(use_rag=cfg.use_rag)
                for task in tasks:
                    parse_result = parser.parse(task, robot_id=robot_id)
                    if not parse_result["success"]:
                        continue
                    ops = parse_result["commands"]
                    cfg_local = dataclasses.replace(cfg, dry_run=True)
                    raw = self._run_local(ops, cfg_local)
                    total_ms += float(raw.get("total_duration_ms", 0.0))
                    # Prefer real hub round count; fall back to signal-op proxy
                    if hub is not None:
                        total_negotiation_rounds += hub.get_last_round_count()
                    else:
                        total_negotiation_rounds += sum(
                            1 for op in ops if op.get("operation") == "signal"
                        )
                    task_steps = self._parse_steps(raw.get("results") or [], ops)
                    step_offset = len(all_steps)
                    for s in task_steps:
                        s.index += step_offset
                    all_steps.extend(task_steps)
                    if raw.get("success"):
                        completed_tasks += 1
            finally:
                MockRegistry.restore_mock(mock_original)
                _neg_cfg.NEGOTIATION_ENABLED = prev_neg

            ops_executed = len(all_steps)
            ops_succeeded = sum(1 for s in all_steps if s.success)
            success_rate = (completed_tasks / len(tasks)) if tasks else 0.0
            condition = "enabled" if negotiation_on else "disabled"
            condition_results[condition] = AblationMetrics(
                condition=condition,
                hallucinated_ops=0,
                reflexion_recoveries=0,
                negotiation_rounds=total_negotiation_rounds,
                success_rate=success_rate,
                ops_executed=ops_executed,
                ops_succeeded=ops_succeeded,
            )

        ablation_on = condition_results["enabled"]
        ablation_off = condition_results["disabled"]

        round_delta = ablation_on.negotiation_rounds - ablation_off.negotiation_rounds
        rate_delta = round(ablation_on.success_rate - ablation_off.success_rate, 4)
        ablation_on = dataclasses.replace(
            ablation_on, condition_delta=rate_delta, hallucination_delta=round_delta
        )

        # Success: enabled condition executed ops and is not worse than disabled
        success = (
            ablation_on.ops_executed > 0
            and ablation_on.success_rate >= ablation_off.success_rate
        )

        return BenchmarkResult(
            benchmark_id=13,
            benchmark_name=_BENCHMARK_NAMES[13],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            success=success,
            total_duration_ms=0.0,
            steps=[],
            ops_executed=ablation_on.ops_executed,
            ops_succeeded=ablation_on.ops_succeeded,
            success_rate=ablation_on.success_rate,
            avg_step_duration_ms=0.0,
            negotiation_rounds=ablation_on.negotiation_rounds,
            ablation=ablation_on,
            ablation_baseline=ablation_off,
            feature_flags=self._extract_feature_flags(cfg),
            per_op_stats={},
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )

    def _run_b15_vgn(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B15 VGN ablation: execute grasp tasks with VGN enabled vs disabled.

        Offline mode patches VGN_ENABLED, mocks operations with always_succeed, and
        verifies the parse+dry-run path works for both conditions.  Live mode sends
        NL grasp tasks to SequenceServer and measures actual grasp success rate.


        """
        import config.Servers as _srv_cfg
        from orchestrators.CommandParser import CommandParser
        from .Result import AblationMetrics

        tasks = module.get_tasks(cfg)

        if cfg.execution_mode == "live":
            completed = 0
            ops_executed = 0
            ops_succeeded = 0
            all_steps: List[StepResult] = []
            total_ms = 0.0
            _flags = BenchmarkFeatureFlags(use_vgn=cfg.use_vgn)
            for task in tasks:
                raw = self._send(task, cfg.robot_id, cfg, flags=_flags)
                if raw.get("success"):
                    completed += 1
                task_steps = self._parse_steps(raw.get("results") or [])
                step_offset = len(all_steps)
                for s in task_steps:
                    s.index += step_offset
                all_steps.extend(task_steps)
                ops_executed += raw.get("ops_executed", len(task_steps))
                ops_succeeded += raw.get(
                    "ops_succeeded", sum(1 for s in task_steps if s.success)
                )
                total_ms += float(raw.get("total_duration_ms", 0.0))
                self._reset(cfg)

            success_rate = completed / len(tasks) if tasks else 0.0
            ablation = AblationMetrics(
                condition="enabled" if cfg.use_vgn else "disabled",
                hallucinated_ops=0,
                reflexion_recoveries=0,
                negotiation_rounds=0,
                success_rate=success_rate,
                ops_executed=ops_executed,
                ops_succeeded=ops_succeeded,
            )
            return BenchmarkResult(
                benchmark_id=15,
                benchmark_name=_BENCHMARK_NAMES[15],
                run_id=make_run_id(),
                config_snapshot=dataclasses.asdict(cfg),
                success=success_rate > 0.5,
                total_duration_ms=total_ms,
                steps=all_steps,
                ops_executed=ops_executed,
                ops_succeeded=ops_succeeded,
                success_rate=success_rate,
                avg_step_duration_ms=(total_ms / ops_executed) if ops_executed else 0.0,
                ablation=ablation,
                feature_flags=self._extract_feature_flags(cfg),
                per_op_stats=self._compute_per_op_stats(all_steps),
                execution_mode="live",
            )

        # Offline: parse + dry-run under both conditions in one pass
        condition_results: Dict[str, AblationMetrics] = {}
        last_steps: List[StepResult] = []

        for vgn_on in (True, False):
            prev_vgn = _srv_cfg.VGN_ENABLED
            _srv_cfg.VGN_ENABLED = vgn_on
            mock_original = MockRegistry.install_mock("always_succeed")
            all_steps: List[StepResult] = []
            total_ms = 0.0
            completed_tasks = 0
            try:
                parser = CommandParser(use_rag=cfg.use_rag)
                for task in tasks:
                    parse_result = parser.parse(task, robot_id=cfg.robot_id)
                    if not parse_result["success"]:
                        continue
                    ops = parse_result["commands"]
                    cfg_local = dataclasses.replace(cfg, dry_run=True)
                    raw = self._run_local(ops, cfg_local)
                    total_ms += float(raw.get("total_duration_ms", 0.0))
                    task_steps = self._parse_steps(raw.get("results") or [], ops)
                    step_offset = len(all_steps)
                    for s in task_steps:
                        s.index += step_offset
                    all_steps.extend(task_steps)
                    if raw.get("success"):
                        completed_tasks += 1
            finally:
                MockRegistry.restore_mock(mock_original)
                _srv_cfg.VGN_ENABLED = prev_vgn

            ops_executed = len(all_steps)
            ops_succeeded = sum(1 for s in all_steps if s.success)
            success_rate = (completed_tasks / len(tasks)) if tasks else 0.0
            condition = "enabled" if vgn_on else "disabled"
            condition_results[condition] = AblationMetrics(
                condition=condition,
                hallucinated_ops=0,
                reflexion_recoveries=0,
                negotiation_rounds=0,
                success_rate=success_rate,
                ops_executed=ops_executed,
                ops_succeeded=ops_succeeded,
            )
            if vgn_on:
                last_steps = all_steps

        ablation_on = condition_results["enabled"]
        ablation_off = condition_results["disabled"]

        delta = round(ablation_on.success_rate - ablation_off.success_rate, 4)
        ablation_on = dataclasses.replace(ablation_on, condition_delta=delta)

        # Offline: success = both conditions parsed ops (parse-path validation only)
        # Actual grasp quality difference only measurable in live mode
        success = ablation_on.ops_executed > 0 and ablation_off.ops_executed > 0

        return BenchmarkResult(
            benchmark_id=15,
            benchmark_name=_BENCHMARK_NAMES[15],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            success=success,
            total_duration_ms=0.0,
            steps=last_steps,
            ops_executed=ablation_on.ops_executed,
            ops_succeeded=ablation_on.ops_succeeded,
            success_rate=ablation_on.success_rate,
            avg_step_duration_ms=0.0,
            ablation=ablation_on,
            ablation_baseline=ablation_off,
            feature_flags=self._extract_feature_flags(cfg),
            per_op_stats=self._compute_per_op_stats(last_steps),
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )

    def _run_b16_ros(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B16 ROS vs Unity movement ablation.

        Patches config.ROS.ROS_ENABLED and DEFAULT_CONTROL_MODE, then executes
        movement tasks.  Offline mode uses always_succeed mock + dry-run to verify
        the parse path.  Live mode sends NL movement tasks to SequenceServer;
        avg_step_duration_ms captures MoveIt planning overhead vs direct Unity TCP.


        """
        import config.ROS as _ros_cfg
        from orchestrators.CommandParser import CommandParser
        from .Result import AblationMetrics

        tasks = module.get_tasks(cfg)

        if cfg.execution_mode == "live":
            completed = 0
            ops_executed = 0
            ops_succeeded = 0
            all_steps: List[StepResult] = []
            total_ms = 0.0
            _flags = BenchmarkFeatureFlags(use_ros=cfg.use_ros_movement)
            for task in tasks:
                raw = self._send(task, cfg.robot_id, cfg, flags=_flags)
                if raw.get("success"):
                    completed += 1
                task_steps = self._parse_steps(raw.get("results") or [])
                step_offset = len(all_steps)
                for s in task_steps:
                    s.index += step_offset
                all_steps.extend(task_steps)
                ops_executed += len(task_steps)
                ops_succeeded += sum(1 for s in task_steps if s.success)
                total_ms += float(raw.get("total_duration_ms", 0.0))
                self._reset(cfg)

            success_rate = completed / len(tasks) if tasks else 0.0
            ablation = AblationMetrics(
                condition="ros" if cfg.use_ros_movement else "unity",
                hallucinated_ops=0,
                reflexion_recoveries=0,
                negotiation_rounds=0,
                success_rate=success_rate,
                ops_executed=ops_executed,
                ops_succeeded=ops_succeeded,
            )
            return BenchmarkResult(
                benchmark_id=16,
                benchmark_name=_BENCHMARK_NAMES[16],
                run_id=make_run_id(),
                config_snapshot=dataclasses.asdict(cfg),
                success=success_rate > 0.5,
                total_duration_ms=total_ms,
                steps=all_steps,
                ops_executed=ops_executed,
                ops_succeeded=ops_succeeded,
                success_rate=success_rate,
                avg_step_duration_ms=(total_ms / ops_executed) if ops_executed else 0.0,
                ablation=ablation,
                feature_flags=self._extract_feature_flags(cfg),
                per_op_stats=self._compute_per_op_stats(all_steps),
                execution_mode="live",
            )

        # Offline: parse + dry-run under both conditions in one pass
        condition_results: Dict[str, AblationMetrics] = {}
        last_steps: List[StepResult] = []

        for ros_on in (True, False):
            prev_enabled = _ros_cfg.ROS_ENABLED
            prev_mode = _ros_cfg.DEFAULT_CONTROL_MODE
            _ros_cfg.ROS_ENABLED = ros_on
            _ros_cfg.DEFAULT_CONTROL_MODE = "ros" if ros_on else "unity"
            mock_original = MockRegistry.install_mock("always_succeed")
            all_steps: List[StepResult] = []
            total_ms = 0.0
            completed_tasks = 0
            try:
                parser = CommandParser(use_rag=cfg.use_rag)
                for task in tasks:
                    parse_result = parser.parse(task, robot_id=cfg.robot_id)
                    if not parse_result["success"]:
                        continue
                    ops = parse_result["commands"]
                    cfg_local = dataclasses.replace(cfg, dry_run=True)
                    raw = self._run_local(ops, cfg_local)
                    total_ms += float(raw.get("total_duration_ms", 0.0))
                    task_steps = self._parse_steps(raw.get("results") or [], ops)
                    step_offset = len(all_steps)
                    for s in task_steps:
                        s.index += step_offset
                    all_steps.extend(task_steps)
                    if raw.get("success"):
                        completed_tasks += 1
            finally:
                MockRegistry.restore_mock(mock_original)
                _ros_cfg.ROS_ENABLED = prev_enabled
                _ros_cfg.DEFAULT_CONTROL_MODE = prev_mode

            ops_executed = len(all_steps)
            ops_succeeded = sum(1 for s in all_steps if s.success)
            success_rate = (completed_tasks / len(tasks)) if tasks else 0.0
            condition = "ros" if ros_on else "unity"
            condition_results[condition] = AblationMetrics(
                condition=condition,
                hallucinated_ops=0,
                reflexion_recoveries=0,
                negotiation_rounds=0,
                success_rate=success_rate,
                ops_executed=ops_executed,
                ops_succeeded=ops_succeeded,
            )
            if ros_on:
                last_steps = all_steps

        ablation_on = condition_results["ros"]
        ablation_off = condition_results["unity"]

        delta = round(ablation_on.success_rate - ablation_off.success_rate, 4)
        ablation_on = dataclasses.replace(ablation_on, condition_delta=delta)

        # Offline: success = both conditions parsed ops (parse-path validation only)
        # Timing difference only measurable in live mode
        success = ablation_on.ops_executed > 0 and ablation_off.ops_executed > 0

        return BenchmarkResult(
            benchmark_id=16,
            benchmark_name=_BENCHMARK_NAMES[16],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            success=success,
            total_duration_ms=0.0,
            steps=last_steps,
            ops_executed=ablation_on.ops_executed,
            ops_succeeded=ablation_on.ops_succeeded,
            success_rate=ablation_on.success_rate,
            avg_step_duration_ms=0.0,
            ablation=ablation_on,
            ablation_baseline=ablation_off,
            feature_flags=self._extract_feature_flags(cfg),
            per_op_stats=self._compute_per_op_stats(last_steps),
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )

    def _run_b14_kg(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B14 KG ablation: parse spatial tasks under both conditions in one pass.

        Populates synthetic KG for the enabled condition; skips it for disabled.
        Measures whether parsed ops reference correct KG object IDs.
        No Unity required — parse-only.


        """
        import config.KnowledgeGraph as _kg_cfg
        from orchestrators.CommandParser import CommandParser
        from core.Imports import get_global_registry
        from .Result import AblationMetrics

        registry = get_global_registry()
        tasks = module.get_tasks(cfg)
        condition_results: Dict[str, AblationMetrics] = {}

        for kg_on in (True, False):
            prev_kg = _kg_cfg.KNOWLEDGE_GRAPH_ENABLED
            _kg_cfg.KNOWLEDGE_GRAPH_ENABLED = kg_on
            try:
                if kg_on:
                    module.populate_synthetic_kg(cfg.robot_id)

                parser = CommandParser(use_rag=cfg.use_rag)
                hallucinated = 0
                wrong_object = 0
                succeeded = 0
                total_ops = 0

                for task in tasks:
                    parse_result = parser.parse(task, robot_id=cfg.robot_id)
                    if not parse_result["success"]:
                        continue
                    for cmd in parse_result.get("commands", []):
                        op_name: str = cmd.get("operation") or ""
                        total_ops += 1
                        if registry.get_operation_by_name(op_name) is None:
                            hallucinated += 1
                        else:
                            succeeded += 1
                            if kg_on:
                                params_str = str(cmd.get("params", {})).lower()
                                uses_variable_ref = "$" in params_str
                                has_kg_ref = uses_variable_ref or any(
                                    obj.lower() in params_str
                                    for obj in module.KG_OBJECTS
                                )
                                if not has_kg_ref and "object" in op_name:
                                    wrong_object += 1
            finally:
                if kg_on:
                    module.clear_synthetic_kg()
                _kg_cfg.KNOWLEDGE_GRAPH_ENABLED = prev_kg

            total_bad = hallucinated + wrong_object
            ablation_succeeded = total_ops - total_bad
            rate = (ablation_succeeded / total_ops) if total_ops else 0.0
            condition = "enabled" if kg_on else "disabled"
            condition_results[condition] = AblationMetrics(
                condition=condition,
                hallucinated_ops=total_bad,
                reflexion_recoveries=0,
                negotiation_rounds=0,
                success_rate=rate,
                ops_executed=total_ops,
                ops_succeeded=ablation_succeeded,
            )

        ablation_on = condition_results["enabled"]
        ablation_off = condition_results["disabled"]

        return BenchmarkResult(
            benchmark_id=14,
            benchmark_name=_BENCHMARK_NAMES[14],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            success=(ablation_on.success_rate >= 0.5 and ablation_off.ops_executed > 0),
            total_duration_ms=0.0,
            steps=[],
            ops_executed=ablation_on.ops_executed,
            ops_succeeded=ablation_on.ops_succeeded,
            success_rate=ablation_on.success_rate,
            avg_step_duration_ms=0.0,
            hallucinated_ops=ablation_on.hallucinated_ops,
            ablation=ablation_on,
            ablation_baseline=ablation_off,
            feature_flags=self._extract_feature_flags(cfg),
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )
