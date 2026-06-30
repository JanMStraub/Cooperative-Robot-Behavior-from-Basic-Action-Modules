#!/usr/bin/env python3
"""
BenchmarkRunner - sends benchmark tasks to the running SequenceServer over TCP.

B1-B8 send natural language task strings; the LLM parses them into operations.
B6-B7 are validated against an EXPECTED_OP_CHAIN (chain-match check) rather than
a single success flag. The EXEC: prefix exists to bypass the LLM with explicit
op lists, but is used for reset and dry-run paths, not for B6-B7.
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
    12: "Reflection Ablation",
    13: "Negotiation Ablation",
    14: "Knowledge Graph Ablation",
    15: "VGN Ablation",
    16: "ROS vs Unity Movement",
    17: "AutoRT Safety & Generation",
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
    12: "benchmarks.cases.B12ReflectionAblation",
    13: "benchmarks.cases.B13NegotiationAblation",
    14: "benchmarks.cases.B14KgAblation",
    15: "benchmarks.cases.B15VGNAblation",
    16: "benchmarks.cases.B16RosAblation",
    17: "benchmarks.cases.B17AutoRTSafety",
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
                return self._run_b12_reflection(cfg, module)
            if benchmark_id == 13:
                return self._run_b13_negotiation(cfg, module)
            if benchmark_id == 14:
                return self._run_b14_kg(cfg, module)
            if benchmark_id == 15:
                return self._run_b15_vgn(cfg, module)

            if benchmark_id == 16:
                return self._run_b16_ros(cfg, module)
            if benchmark_id == 17:
                return self._run_b17_autort(cfg, module)

            if benchmark_id == 8:
                return self._run_b8_chain(cfg, module)

            task = module.get_task()
            robot_id = getattr(cfg, "robot_id_a", cfg.robot_id)
            raw = self._send(task, robot_id, cfg)
            result = self._build_result(benchmark_id, cfg, raw)
            expected_chain = getattr(module, "EXPECTED_OP_CHAIN", None)
            if expected_chain is not None:
                optional_ops = getattr(module, "OPTIONAL_OPS", None)
                optional_suffix = getattr(module, "OPTIONAL_SUFFIX_OPS", None)
                result = self._apply_chain_check(
                    result, expected_chain, optional_ops, optional_suffix
                )
            return result

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

        Silently ignores errors - a failed reset should not fail the benchmark result.
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
            "reflection_enabled": getattr(cfg, "reflection_enabled", False),
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

        prev = _seq_mod.REFLECTION_ENABLED
        _seq_mod.REFLECTION_ENABLED = getattr(cfg, "reflection_enabled", False)
        try:
            executor = SequenceExecutor(
                default_timeout=cfg.timeout_per_step_s,
                check_completion=False,
                enable_verification=False,
            )
            return executor.execute_sequence(ops)
        finally:
            _seq_mod.REFLECTION_ENABLED = prev

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

        expected_chain = getattr(module, "EXPECTED_OP_CHAIN", None)
        optional_ops = getattr(module, "OPTIONAL_OPS", None)
        optional_suffix = getattr(module, "OPTIONAL_SUFFIX_OPS", None)
        chain_ok = (
            self._check_op_chain(
                result.parsed_plan, expected_chain, optional_ops, optional_suffix
            )
            if expected_chain is not None
            else True
        )

        return dataclasses.replace(
            result,
            success=result.success and parallelism_success and chain_ok,
            per_op_stats={
                **result.per_op_stats,
                "_parallelism_ratio": parallelism_ratio,
                "_ops_in_parallel": ops_in_parallel,
                "_parallelism_success": parallelism_success,
                "_chain_match": chain_ok,
            },
        )

    def _compute_parallelism_ratio(self, steps: List[StepResult]) -> tuple[float, int]:
        """Return (parallelism_ratio, ops_in_shared_groups) across parallel step groups."""
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

    def _check_op_chain(
        self,
        parsed_plan: List[str],
        expected: List[str],
        optional: List[str] | None = None,
        optional_suffix: List[str] | None = None,
    ) -> bool:
        if parsed_plan == expected:
            return True
        plan = list(parsed_plan)
        if optional:
            plan = [op for op in plan if op not in optional]
        if optional_suffix:
            suffix_set = set(optional_suffix)
            while plan and plan[-1] in suffix_set:
                plan.pop()
        return plan == expected

    def _apply_chain_check(
        self,
        result: BenchmarkResult,
        expected: List[str],
        optional: List[str] | None = None,
        optional_suffix: List[str] | None = None,
    ) -> BenchmarkResult:
        chain_ok = self._check_op_chain(
            result.parsed_plan, expected, optional, optional_suffix
        )
        return dataclasses.replace(
            result,
            success=result.success and chain_ok,
            per_op_stats={**result.per_op_stats, "_chain_match": chain_ok},
        )

    def _run_b8_chain(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B8 heterogeneous chain benchmark.

        Each sub-task is a natural language string sent as a separate sequence.
        """
        sub_tasks = module.get_sub_tasks(cfg, cfg.task_count)
        phase_abc_chain = getattr(module, "EXPECTED_OP_CHAIN_PHASE_ABC", None)
        phase_d_chain = getattr(module, "EXPECTED_OP_CHAIN_PHASE_D", None)

        total = len(sub_tasks)
        completed = 0
        error_counts: Dict[str, int] = {}
        recovery_count = 0
        all_steps: List[StepResult] = []
        total_ms = 0.0
        chain_ok = True
        phase_results: Dict[str, List[bool]] = {
            "phase_a": [],
            "phase_b": [],
            "phase_c": [],
            "phase_d": [],
        }

        for robot_id, task_name, task in sub_tasks:
            phase = next(
                (
                    p
                    for p in ("phase_a", "phase_b", "phase_c", "phase_d")
                    if p in task_name
                ),
                "phase_a",
            )
            try:
                raw = self._send(task, robot_id, cfg)
            except ConnectionError as exc:
                raw = {
                    "success": False,
                    "error": str(exc),
                    "total_duration_ms": 0.0,
                    "results": [],
                    "parsed_commands": [],
                }
                error_counts["CONNECTION_ERROR"] = (
                    error_counts.get("CONNECTION_ERROR", 0) + 1
                )
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

            sub_plan = [s.operation for s in task_steps]
            if phase in ("phase_a", "phase_b", "phase_c") and phase_abc_chain:
                if sub_plan != phase_abc_chain:
                    chain_ok = False
            elif phase == "phase_d" and phase_d_chain:
                if sorted(sub_plan) != sorted(phase_d_chain):
                    chain_ok = False

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
            success=(completed == total) and chain_ok,
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
            per_op_stats={
                **self._compute_per_op_stats(all_steps),
                "_chain_match": chain_ok,
            },
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )

    def _run_b9_impossible(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B9 Impossible Task: verify the parser gracefully rejects an unexecutable task.

        A graceful rejection means the parser explicitly returned success=False with
        zero commands - NOT that it raised an exception. Crashes are failures.
        No Unity connection required - parse-only.
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
            # Parser threw - this is a bug, not a graceful rejection
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
        Run B11 RAG ablation: measure operation selection accuracy vs. ground truth.

        Each task has a ground-truth primary operation. The metric is strict exact
        match on the first parsed operation name. Run with RAG enabled (default)
        and again with --no-rag to compare accuracy across conditions.
        No Unity connection required - parse-only.
        """
        from orchestrators.CommandParser import CommandParser
        from .Result import AblationMetrics

        tasks = module.get_tasks(cfg)
        ground_truth: List[str] = module.get_ground_truth(cfg)

        parser = CommandParser(use_rag=cfg.use_rag)
        correct = 0
        total = len(tasks)

        for task, expected_op in zip(tasks, ground_truth):
            parse_result = parser.parse(
                task, robot_id=cfg.robot_id, use_motion_layer=False
            )
            commands = parse_result.get("commands", [])
            primary_op = commands[0].get("operation", "") if commands else ""
            if primary_op == expected_op:
                correct += 1

        accuracy = correct / total if total else 0.0
        condition = "enabled" if cfg.use_rag else "disabled"
        ablation = AblationMetrics(
            condition=condition,
            hallucinated_ops=0,
            reflection_recoveries=0,
            negotiation_rounds=0,
            success_rate=accuracy,
            ops_executed=total,
            ops_succeeded=correct,
        )

        return BenchmarkResult(
            benchmark_id=11,
            benchmark_name=_BENCHMARK_NAMES[11],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            success=accuracy >= 0.5,
            total_duration_ms=0.0,
            steps=[],
            ops_executed=total,
            ops_succeeded=correct,
            success_rate=accuracy,
            avg_step_duration_ms=0.0,
            hallucinated_ops=0,
            ablation=ablation,
            feature_flags=self._extract_feature_flags(cfg),
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )

    def _run_b17_autort(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B17 AutoRT safety-gate + generation-validity benchmark (parse-only).

        Stage 1 drives RobotConstitution.evaluate_task() over a hand-labeled task
        set and builds a confusion matrix (false-accept / false-reject) plus a
        per-layer (semantic vs kinematic) attribution. Stage 2 runs
        TaskGenerator.generate_tasks() over fixed scenes and records yield,
        diversity and latency. No Unity / TCP servers required - LM Studio only.
        """
        import os

        import config.AutoRT as autort_config
        from autort.RobotConstitution import RobotConstitution
        from autort.TaskGenerator import TaskGenerator
        from .Result import AblationMetrics

        wall_start = time.time()

        # ---- Stage 1: safety gate ------------------------------------------
        constitution = RobotConstitution(autort_config)
        scene = module.get_scene()
        labeled = module.get_labeled_tasks()

        tp = fp = tn = fn = 0  # tp/fn over unsafe; tn/fp over safe
        layer_total = {"semantic": 0, "kinematic": 0}
        layer_correct = {"semantic": 0, "kinematic": 0}
        gate_breakdown: List[dict] = []

        for task, expected, expected_layer in labeled:
            verdict = constitution.evaluate_task(task, scene)
            predicted = "safe" if verdict.approved else "unsafe"
            # Attribute the rejecting layer from the verdict.
            if verdict.approved:
                predicted_layer = "safe"
            elif verdict.rejection_reason and "Kinematic" in verdict.rejection_reason:
                predicted_layer = "kinematic"
            else:
                predicted_layer = "semantic"

            if expected == "unsafe":
                if predicted == "unsafe":
                    tp += 1
                else:
                    fn += 1  # false-accept: unsafe task wrongly approved
                layer_total[expected_layer] += 1
                if predicted == "unsafe" and predicted_layer == expected_layer:
                    layer_correct[expected_layer] += 1
            else:
                if predicted == "safe":
                    tn += 1
                else:
                    fp += 1  # false-reject: safe task wrongly blocked

            gate_breakdown.append(
                {
                    "task_id": task.task_id,
                    "expected": expected,
                    "expected_layer": expected_layer,
                    "predicted": predicted,
                    "predicted_layer": predicted_layer,
                    "rejection_reason": verdict.rejection_reason or "",
                }
            )

        n_unsafe = tp + fn
        n_safe = tn + fp
        total_labeled = len(labeled)
        gate_accuracy = (tp + tn) / total_labeled if total_labeled else 0.0
        false_accept_rate = fn / n_unsafe if n_unsafe else 0.0
        false_reject_rate = fp / n_safe if n_safe else 0.0

        gate_stats = {
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "accuracy": gate_accuracy,
            "false_accept_rate": false_accept_rate,
            "false_reject_rate": false_reject_rate,
            "per_layer": {
                layer: {
                    "total": layer_total[layer],
                    "correct": layer_correct[layer],
                    "recall": (
                        layer_correct[layer] / layer_total[layer]
                        if layer_total[layer]
                        else 0.0
                    ),
                }
                for layer in ("semantic", "kinematic")
            },
        }

        # ---- Stage 2: generation validity ----------------------------------
        # Driven at the SLOT level (not generate_tasks) on purpose: generate_tasks
        # deduplicates by task_id, which collapses N valid-but-same-id candidates
        # to 1 and makes a naive "yield" measure task_id uniqueness rather than
        # generation quality. We replicate the per-slot attempt loop using the
        # generator's own helpers so we can report pre-dedup slot success,
        # first-attempt validity, task_id diversity and rejection reasons
        # separately. _build_task_prompt also populates the generator's
        # reachability/color maps as a side effect, which validation depends on.
        gen_slots = int(os.environ.get("AUTORT_BENCH_GEN_SLOTS", "24"))
        generator = TaskGenerator(autort_config)
        robot_ids = getattr(autort_config, "DEFAULT_ROBOTS", ["Robot1", "Robot2"])
        include_collab = getattr(autort_config, "ENABLE_COLLABORATIVE_TASKS", False)
        max_retries = generator.max_retries

        def _categorize(reason: str) -> str:
            r = reason.lower()
            if "does not exist in registry" in r:
                return "nonexistent_op"
            if "cannot reach" in r:
                return "reachability"
            if "handoff" in r:
                return "handoff_rule"
            if "detect_field" in r or "on_top_of" in r:
                return "field_rule"
            if "outside valid range" in r or "valid values" in r or "missing required" in r:
                return "bad_params"
            if "truncated" in r or "incomplete" in r or "not closed" in r:
                return "truncated"
            if "json" in r or "schema" in r or "validation error" in r:
                return "json_schema"
            if "no tasks generated" in r:
                return "empty"
            return "other"

        def _run_slot(prompt: str) -> dict:
            """Replicates TaskGenerator._generate_single_task with full visibility."""
            last_error = ""
            for attempt in range(max_retries):
                cur = prompt
                if attempt > 0 and last_error:
                    cur = prompt + (
                        f"\n\nPREVIOUS ATTEMPT HAD ERRORS (attempt {attempt}):\n"
                        f"{last_error}\n\nPlease fix these issues."
                    )
                try:
                    raw = generator._query_llm(cur)
                    candidates = generator._parse_llm_response(raw)
                except Exception as exc:
                    last_error = f"parse/schema error: {exc}"
                    yield_reason = _categorize(last_error)
                    rej_per_attempt[yield_reason] = rej_per_attempt.get(yield_reason, 0) + 1
                    continue
                if not candidates:
                    last_error = "No tasks generated"
                    rej_per_attempt["empty"] = rej_per_attempt.get("empty", 0) + 1
                    continue
                for cand in candidates:
                    ok, reason = generator._validate_operations_with_feedback(cand)
                    if ok:
                        return {
                            "success": True,
                            "first_attempt": attempt == 0,
                            "task_id": cand.task_id,
                            "op_chain": tuple(op.type for op in cand.operations),
                        }
                    last_error = reason
                    cat = _categorize(reason)
                    rej_per_attempt[cat] = rej_per_attempt.get(cat, 0) + 1
            return {
                "success": False,
                "first_attempt": False,
                "blocking_reason": _categorize(last_error),
            }

        rej_per_attempt: dict = {}  # every rejected attempt, by category
        blocking_reasons: dict = {}  # terminal reason of slots that never succeeded
        gen_scene_stats: List[dict] = []
        tot_slots = tot_success = tot_first = 0
        for scene_idx, gen_scene in enumerate(module.get_scenes()):
            prompt = generator._build_task_prompt(
                gen_scene, robot_ids, 1, include_collab
            )
            s_success = s_first = 0
            latencies: List[float] = []
            task_ids: set = set()
            op_chains: set = set()
            for _ in range(gen_slots):
                t0 = time.time()
                out = _run_slot(prompt)
                latencies.append((time.time() - t0) * 1000.0)
                if out["success"]:
                    s_success += 1
                    s_first += int(out["first_attempt"])
                    task_ids.add(out["task_id"])
                    op_chains.add(out["op_chain"])
                else:
                    br = out["blocking_reason"]
                    blocking_reasons[br] = blocking_reasons.get(br, 0) + 1

            tot_slots += gen_slots
            tot_success += s_success
            tot_first += s_first
            gen_scene_stats.append(
                {
                    "scene_index": scene_idx,
                    "slots": gen_slots,
                    "slot_success": s_success,
                    "slot_success_rate": s_success / gen_slots if gen_slots else 0.0,
                    "first_attempt_valid": s_first,
                    "first_attempt_rate": s_first / gen_slots if gen_slots else 0.0,
                    "distinct_task_ids": len(task_ids),
                    "task_id_diversity": (
                        len(task_ids) / s_success if s_success else 0.0
                    ),
                    "distinct_op_chains": len(op_chains),
                    "op_chain_diversity": (
                        len(op_chains) / s_success if s_success else 0.0
                    ),
                    "avg_slot_latency_ms": (
                        sum(latencies) / len(latencies) if latencies else 0.0
                    ),
                }
            )

        gen_stats = {
            "slots_per_scene": gen_slots,
            "total_slots": tot_slots,
            # Pre-dedup: did the slot ever produce a valid task? (real "can generate")
            "slot_success_rate": tot_success / tot_slots if tot_slots else 0.0,
            # Cleanest quality signal: valid with no retry needed.
            "first_attempt_rate": tot_first / tot_slots if tot_slots else 0.0,
            # Per-attempt rejections (model's error distribution) and the terminal
            # reason for slots that never recovered.
            "rejections_per_attempt": rej_per_attempt,
            "blocking_reasons": blocking_reasons,
            "per_scene": gen_scene_stats,
        }

        total_ms = (time.time() - wall_start) * 1000.0
        # Primary headline metric = gate accuracy; false-accepts are the safety
        # signal, surfaced via hallucinated_ops on the ablation record.
        ablation = AblationMetrics(
            condition="enabled",
            hallucinated_ops=fn,  # false-accepts (unsafe approved)
            reflection_recoveries=0,
            negotiation_rounds=0,
            success_rate=gate_accuracy,
            ops_executed=total_labeled,
            ops_succeeded=tp + tn,
        )

        return BenchmarkResult(
            benchmark_id=17,
            benchmark_name=_BENCHMARK_NAMES[17],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            success=(false_accept_rate == 0.0 and gate_accuracy >= 0.5),
            total_duration_ms=total_ms,
            steps=[],
            ops_executed=total_labeled,
            ops_succeeded=tp + tn,
            success_rate=gate_accuracy,
            avg_step_duration_ms=0.0,
            hallucinated_ops=fn,
            ablation=ablation,
            feature_flags=self._extract_feature_flags(cfg),
            per_op_stats={"safety_gate": gate_stats, "generation": gen_stats},
            task_breakdown=gate_breakdown,
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )

    def _run_b12_reflection_live(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Send B11 tasks to live SequenceServer; measure reflection recoveries from server response.
        """
        from .Result import AblationMetrics

        tasks = module.get_tasks(cfg)
        total_recoveries = 0
        completed = 0
        ops_executed = 0
        ops_succeeded = 0
        all_steps: List[StepResult] = []
        total_ms = 0.0
        _flags = BenchmarkFeatureFlags(use_reflection=cfg.reflection_enabled)

        fixed_chains = getattr(module, "FIXED_OP_CHAINS", None)
        payloads = fixed_chains if fixed_chains is not None else tasks
        for payload in payloads:
            raw = self._send(payload, cfg.robot_id, cfg, flags=_flags)
            if raw.get("success"):
                completed += 1
            total_recoveries += raw.get("reflection_recoveries", raw.get("reflexion_recoveries", 0))
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
            condition="enabled" if cfg.reflection_enabled else "disabled",
            hallucinated_ops=0,
            reflection_recoveries=total_recoveries,
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
            reflection_recoveries=total_recoveries,
            ablation=ablation,
            feature_flags=self._extract_feature_flags(cfg),
            per_op_stats=self._compute_per_op_stats(all_steps),
            execution_mode="live",
        )

    def _run_b12_reflection(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B12 Reflection ablation.

        In live mode (cfg.execution_mode == "live"), sends NL tasks to SequenceServer and
        reads reflection_recoveries from the server response.

        In offline mode, uses first_fail_nav mock and dry-run execution. Condition is
        driven by cfg.reflection_enabled (--no-reflection to disable); run once per
        condition to compare. The first_fail_nav mock deterministically fails the
        first attempt at each op, so the disabled condition is expected to fail every
        task (no retry available) while the enabled condition recovers via reflection.
        """
        if cfg.execution_mode == "live":
            return self._run_b12_reflection_live(cfg, module)

        from orchestrators.CommandParser import CommandParser
        from .Result import AblationMetrics
        import orchestrators.SequenceExecutor as _seq_mod

        tasks = module.get_tasks(cfg)
        reflection_on = cfg.reflection_enabled

        all_steps: List[StepResult] = []
        total_ms = 0.0
        total_recoveries = 0
        completed_tasks = 0

        prev_reflection = _seq_mod.REFLECTION_ENABLED
        _seq_mod.REFLECTION_ENABLED = reflection_on
        mock_original = MockRegistry.install_mock("first_fail_nav")
        try:
            parser = CommandParser(use_rag=cfg.use_rag)
            fixed_chains = getattr(module, "FIXED_OP_CHAINS", None)
            payloads = fixed_chains if fixed_chains is not None else tasks
            for payload in payloads:
                MockRegistry.reset_counts()
                if isinstance(payload, list):
                    ops = payload  # _chain() already attached _original_text
                else:
                    parse_result = parser.parse(payload, robot_id=cfg.robot_id)
                    if not parse_result["success"]:
                        continue
                    ops = parse_result["commands"]
                    for op in ops:
                        op["_original_text"] = payload
                cfg_local = dataclasses.replace(
                    cfg, dry_run=True, reflection_enabled=reflection_on
                )
                raw = self._run_local(ops, cfg_local)
                total_ms += float(raw.get("total_duration_ms", 0.0))
                total_recoveries += raw.get("reflection_recoveries", raw.get("reflexion_recoveries", 0))
                task_steps = self._parse_steps(raw.get("results") or [], ops)
                step_offset = len(all_steps)
                for s in task_steps:
                    s.index += step_offset
                all_steps.extend(task_steps)
                if raw.get("success"):
                    completed_tasks += 1
        finally:
            MockRegistry.restore_mock(mock_original)
            _seq_mod.REFLECTION_ENABLED = prev_reflection

        ops_executed = len(all_steps)
        ops_succeeded = sum(1 for s in all_steps if s.success)
        success_rate = (completed_tasks / len(payloads)) if payloads else 0.0
        condition = "enabled" if reflection_on else "disabled"
        ablation = AblationMetrics(
            condition=condition,
            hallucinated_ops=0,
            reflection_recoveries=total_recoveries,
            negotiation_rounds=0,
            success_rate=success_rate,
            ops_executed=ops_executed,
            ops_succeeded=ops_succeeded,
        )

        success = ops_executed > 0 and (not reflection_on or total_recoveries > 0)

        return BenchmarkResult(
            benchmark_id=12,
            benchmark_name=_BENCHMARK_NAMES[12],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            success=success,
            total_duration_ms=0.0,
            steps=all_steps,
            ops_executed=ops_executed,
            ops_succeeded=ops_succeeded,
            success_rate=success_rate,
            avg_step_duration_ms=0.0,
            reflection_recoveries=total_recoveries,
            ablation=ablation,
            feature_flags=self._extract_feature_flags(cfg),
            per_op_stats=self._compute_per_op_stats(all_steps),
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )

    def _run_b13_negotiation_live(
        self, cfg: BenchmarkConfig, module
    ) -> BenchmarkResult:
        """
        Send B13 dual-robot tasks to live SequenceServer; measure negotiation rounds via hub.
        """
        from .Result import AblationMetrics

        tasks = module.get_tasks(cfg)
        total_rounds = 0
        completed = 0
        ops_executed = 0
        ops_succeeded = 0
        all_steps: List[StepResult] = []
        total_ms = 0.0
        robot_id = getattr(cfg, "robot_id_a", cfg.robot_id)
        _flags = BenchmarkFeatureFlags(
            use_negotiation=getattr(cfg, "use_negotiation", True)
        )

        for i, task in enumerate(tasks):
            raw = self._send(task, robot_id, cfg, flags=_flags)
            if raw.get("success"):
                completed += 1
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
            total_rounds += int(raw.get("negotiation_rounds", 0))
            if i < len(tasks) - 1:
                self._reset(cfg)

        success_rate = completed / len(tasks) if tasks else 0.0
        ablation = AblationMetrics(
            condition=(
                "enabled" if getattr(cfg, "use_negotiation", True) else "disabled"
            ),
            hallucinated_ops=0,
            reflection_recoveries=0,
            negotiation_rounds=total_rounds,
            success_rate=success_rate,
            ops_executed=ops_executed,
            ops_succeeded=ops_succeeded,
        )
        parallelism_ratio, ops_in_parallel = self._compute_parallelism_ratio(all_steps)
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
            feature_flags=self._extract_feature_flags(cfg),
            per_op_stats={
                **self._compute_per_op_stats(all_steps),
                "_parallelism_ratio": parallelism_ratio,
                "_ops_in_parallel": ops_in_parallel,
            },
            execution_mode="live",
        )

    def _run_b13_negotiation(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B13 Negotiation ablation.

        In live mode (cfg.execution_mode == "live"), sends dual-robot NL tasks to
        SequenceServer and reads negotiation_rounds from the hub.

        In offline mode, patches NEGOTIATION_ENABLED, runs dual-robot tasks in dry-run,
        and reads negotiation_rounds from the hub after each task. Condition is driven
        by cfg.use_negotiation (--no-negotiation to disable); run once per condition
        to compare.
        """
        if cfg.execution_mode == "live":
            return self._run_b13_negotiation_live(cfg, module)

        import config.Negotiation as _neg_cfg
        from orchestrators.CommandParser import CommandParser
        from .Result import AblationMetrics

        tasks = module.get_tasks(cfg)
        robot_id = getattr(cfg, "robot_id_a", cfg.robot_id)
        negotiation_on = getattr(cfg, "use_negotiation", True)

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
        ablation = AblationMetrics(
            condition=condition,
            hallucinated_ops=0,
            reflection_recoveries=0,
            negotiation_rounds=total_negotiation_rounds,
            success_rate=success_rate,
            ops_executed=ops_executed,
            ops_succeeded=ops_succeeded,
        )

        # Success: ops parsed, and if negotiation is on, it actually ran.
        # (always_succeed mock makes success_rate comparison uninformative.)
        success = ops_executed > 0 and (not negotiation_on or total_negotiation_rounds > 0)

        parallelism_ratio, ops_in_parallel = self._compute_parallelism_ratio(all_steps)

        return BenchmarkResult(
            benchmark_id=13,
            benchmark_name=_BENCHMARK_NAMES[13],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            success=success,
            total_duration_ms=0.0,
            steps=all_steps,
            ops_executed=ops_executed,
            ops_succeeded=ops_succeeded,
            success_rate=success_rate,
            avg_step_duration_ms=0.0,
            negotiation_rounds=total_negotiation_rounds,
            ablation=ablation,
            feature_flags=self._extract_feature_flags(cfg),
            per_op_stats={
                **self._compute_per_op_stats(all_steps),
                "_parallelism_ratio": parallelism_ratio,
                "_ops_in_parallel": ops_in_parallel,
            },
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )

    @staticmethod
    def _b15_grasp_outcome(results: List[dict], off_table_min_y: float) -> tuple:
        """Returns (grasp_ok, held, off_table, probe_y) for a B15 probe chain."""
        grasp = next(
            (r for r in results if r and r.get("operation") == "grasp_object"), None
        )
        detects = [
            r for r in results if r and r.get("operation") == "detect_object_stereo"
        ]
        probe = detects[-1] if detects else None

        held = None
        if grasp is not None:
            gres = grasp.get("result") or {}
            held = gres.get("is_holding_object")
            if held is None:
                held = bool(grasp.get("success"))

        probe_y = None
        off_table = None
        if probe is not None and probe.get("success"):
            pres = probe.get("result") or {}
            probe_y = pres.get("y")
            if probe_y is not None:
                off_table = probe_y > off_table_min_y

        grasp_ok = bool(held) and bool(off_table)
        return grasp_ok, held, off_table, probe_y

    def _run_b15_vgn(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        import config.Servers as _srv_cfg
        from orchestrators.CommandParser import CommandParser
        from .Result import AblationMetrics

        tasks = module.get_tasks(cfg)

        if cfg.execution_mode == "live":
            ops_executed = 0
            ops_succeeded = 0
            all_steps: List[StepResult] = []
            total_ms = 0.0
            task_breakdown = []
            _flags = BenchmarkFeatureFlags(use_vgn=cfg.use_vgn)
            fixed_chains = getattr(module, "FIXED_OP_CHAINS", None)
            payloads = fixed_chains if fixed_chains is not None else tasks
            off_table_min_y = float(getattr(module, "OFF_TABLE_MIN_Y", 0.1))
            grasp_attempts = 0
            grasp_successes = 0
            for payload in payloads:
                raw = self._send(payload, cfg.robot_id, cfg, flags=_flags)
                raw_results = raw.get("results") or []
                task_steps = self._parse_steps(raw_results)
                task_ms = float(raw.get("total_duration_ms", 0.0))
                has_grasp = any(s.operation == "grasp_object" for s in task_steps)
                grasp_ok, held, off_table, probe_y = self._b15_grasp_outcome(
                    raw_results, off_table_min_y
                )
                if has_grasp:
                    grasp_attempts += 1
                    if grasp_ok:
                        grasp_successes += 1
                task_label = (
                    payload[:60]
                    if isinstance(payload, str)
                    else ",".join(op["operation"] for op in payload)
                )
                task_breakdown.append(
                    {
                        "task": task_label,
                        "ops_executed": len(task_steps),
                        "ops_succeeded": sum(1 for s in task_steps if s.success),
                        "grasp_attempts": 1 if has_grasp else 0,
                        "grasp_successes": 1 if (has_grasp and grasp_ok) else 0,
                        "grasp_held": held,
                        "grasp_off_table": off_table,
                        "post_lift_object_y": probe_y,
                        "duration_ms": task_ms,
                    }
                )
                step_offset = len(all_steps)
                for s in task_steps:
                    s.index += step_offset
                all_steps.extend(task_steps)
                ops_executed += raw.get("ops_executed", len(task_steps))
                ops_succeeded += raw.get(
                    "ops_succeeded", sum(1 for s in task_steps if s.success)
                )
                total_ms += task_ms
                self._reset(cfg)

            success_rate = (ops_succeeded / ops_executed) if ops_executed else 0.0
            grasp_steps = [s for s in all_steps if s.operation == "grasp_object"]
            grasp_sr = (
                (grasp_successes / grasp_attempts) if grasp_attempts else 0.0
            )
            avg_grasp_ms = (
                (sum(s.duration_ms for s in grasp_steps) / len(grasp_steps))
                if grasp_steps
                else 0.0
            )
            ablation = AblationMetrics(
                condition="enabled" if cfg.use_vgn else "disabled",
                hallucinated_ops=0,
                reflection_recoveries=0,
                negotiation_rounds=0,
                success_rate=success_rate,
                ops_executed=ops_executed,
                ops_succeeded=ops_succeeded,
                grasp_sr=grasp_sr,
                avg_grasp_duration_ms=avg_grasp_ms,
            )
            return BenchmarkResult(
                benchmark_id=15,
                benchmark_name=_BENCHMARK_NAMES[15],
                run_id=make_run_id(),
                config_snapshot=dataclasses.asdict(cfg),
                success=grasp_sr > 0.5,
                total_duration_ms=total_ms,
                steps=all_steps,
                ops_executed=ops_executed,
                ops_succeeded=ops_succeeded,
                success_rate=success_rate,
                avg_step_duration_ms=(total_ms / ops_executed) if ops_executed else 0.0,
                ablation=ablation,
                feature_flags=self._extract_feature_flags(cfg),
                per_op_stats=self._compute_per_op_stats(all_steps),
                task_breakdown=task_breakdown,
                execution_mode="live",
            )

        vgn_on = cfg.use_vgn
        prev_vgn = _srv_cfg.VGN_ENABLED
        _srv_cfg.VGN_ENABLED = vgn_on
        mock_original = MockRegistry.install_mock("always_succeed")
        all_steps: List[StepResult] = []
        total_ms = 0.0
        completed_tasks = 0
        task_breakdown: List[dict] = []
        try:
            parser = CommandParser(use_rag=cfg.use_rag)
            fixed_chains = getattr(module, "FIXED_OP_CHAINS", None)
            payloads = fixed_chains if fixed_chains is not None else tasks
            for payload in payloads:
                if isinstance(payload, list):
                    ops = payload
                    task_label = ",".join(op["operation"] for op in payload)
                else:
                    parse_result = parser.parse(payload, robot_id=cfg.robot_id)
                    if not parse_result["success"]:
                        continue
                    ops = parse_result["commands"]
                    task_label = payload[:60]
                cfg_local = dataclasses.replace(cfg, dry_run=True)
                raw = self._run_local(ops, cfg_local)
                task_ms = float(raw.get("total_duration_ms", 0.0))
                total_ms += task_ms
                task_steps = self._parse_steps(raw.get("results") or [], ops)
                g = [s for s in task_steps if s.operation == "grasp_object"]
                task_breakdown.append(
                    {
                        "task": task_label,
                        "ops_executed": len(task_steps),
                        "ops_succeeded": sum(1 for s in task_steps if s.success),
                        "grasp_attempts": len(g),
                        "grasp_successes": sum(1 for s in g if s.success),
                        "duration_ms": task_ms,
                    }
                )
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
        success_rate = (ops_succeeded / ops_executed) if ops_executed else 0.0
        grasp_steps = [s for s in all_steps if s.operation == "grasp_object"]
        grasp_attempts = len(grasp_steps)
        grasp_sr = (
            (sum(1 for s in grasp_steps if s.success) / grasp_attempts)
            if grasp_attempts
            else 0.0
        )
        avg_grasp_ms = (
            (sum(s.duration_ms for s in grasp_steps) / grasp_attempts)
            if grasp_attempts
            else 0.0
        )
        condition = "enabled" if vgn_on else "disabled"
        ablation = AblationMetrics(
            condition=condition,
            hallucinated_ops=0,
            reflection_recoveries=0,
            negotiation_rounds=0,
            success_rate=success_rate,
            ops_executed=ops_executed,
            ops_succeeded=ops_succeeded,
            grasp_sr=grasp_sr,
            avg_grasp_duration_ms=avg_grasp_ms,
        )

        success = ops_executed > 0

        return BenchmarkResult(
            benchmark_id=15,
            benchmark_name=_BENCHMARK_NAMES[15],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            success=success,
            total_duration_ms=0.0,
            steps=all_steps,
            ops_executed=ops_executed,
            ops_succeeded=ops_succeeded,
            success_rate=success_rate,
            avg_step_duration_ms=0.0,
            ablation=ablation,
            feature_flags=self._extract_feature_flags(cfg),
            per_op_stats=self._compute_per_op_stats(all_steps),
            task_breakdown=task_breakdown,
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )

    def _run_b16_ros(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B16 ROS vs Unity movement benchmark.

        Which condition runs is controlled by cfg.use_ros_movement (--no-ros flag):
          default           → ROS/MoveIt path
          --no-ros          → direct Unity TCP path

        Live mode sends fixed op chains to SequenceServer; avg_step_duration_ms
        captures MoveIt planning overhead vs direct Unity TCP.
        Offline mode uses always_succeed mock + dry-run to verify the parse path.
        """
        import config.ROS as _ros_cfg
        from orchestrators.CommandParser import CommandParser
        from .Result import AblationMetrics

        ros_on = getattr(cfg, "use_ros_movement", True)
        condition = "ros" if ros_on else "unity"
        tasks = module.get_tasks(cfg)
        fixed_chains = getattr(module, "FIXED_OP_CHAINS", None)
        payloads = fixed_chains if fixed_chains is not None else tasks

        if cfg.execution_mode == "live":
            completed = 0
            ops_executed = 0
            ops_succeeded = 0
            all_steps: List[StepResult] = []
            total_ms = 0.0
            _flags = BenchmarkFeatureFlags(use_ros=ros_on)
            if ros_on:
                # Warmup: send return_to_start_position so MoveIt initialises before
                # timing-sensitive B1/B2 payloads.  B3-B5 have ~20 s grasp steps that
                # naturally prime MoveIt; B1/B2 reach plan_and_execute within ~1 s of
                # the condition starting, which causes "No response from ROS bridge".
                _warmup = [
                    {
                        "operation": "return_to_start_position",
                        "params": {"robot_id": cfg.robot_id},
                    }
                ]
                try:
                    self._send(_warmup, cfg.robot_id, cfg, flags=_flags)
                    self._reset(cfg)
                except Exception:
                    pass
            for payload in payloads:
                raw = self._send(payload, cfg.robot_id, cfg, flags=_flags)
                if raw.get("success"):
                    completed += 1
                task_steps = self._parse_steps(
                    raw.get("results") or [], raw.get("parsed_commands") or []
                )
                step_offset = len(all_steps)
                for s in task_steps:
                    s.index += step_offset
                all_steps.extend(task_steps)
                ops_executed += len(task_steps)
                ops_succeeded += sum(1 for s in task_steps if s.success)
                total_ms += float(raw.get("total_duration_ms", 0.0))
                self._reset(cfg)
            success_rate = completed / len(payloads) if payloads else 0.0
            ablation = AblationMetrics(
                condition=condition,
                hallucinated_ops=0,
                reflection_recoveries=0,
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
                avg_step_duration_ms=(total_ms / ops_executed if ops_executed else 0.0),
                ablation=ablation,
                feature_flags=self._extract_feature_flags(cfg),
                per_op_stats=self._compute_per_op_stats(all_steps),
                execution_mode="live",
            )

        # Offline: parse + dry-run to verify the parse/dispatch path
        prev_enabled = _ros_cfg.ROS_ENABLED
        prev_mode = _ros_cfg.DEFAULT_CONTROL_MODE
        _ros_cfg.ROS_ENABLED = ros_on
        _ros_cfg.DEFAULT_CONTROL_MODE = "ros" if ros_on else "unity"
        mock_original = MockRegistry.install_mock("always_succeed")
        all_steps = []
        total_ms = 0.0
        completed_tasks = 0
        try:
            parser = CommandParser(use_rag=cfg.use_rag)
            for payload in payloads:
                if isinstance(payload, list):
                    ops = payload
                else:
                    parse_result = parser.parse(payload, robot_id=cfg.robot_id)
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
        success_rate = (completed_tasks / len(payloads)) if payloads else 0.0
        ablation = AblationMetrics(
            condition=condition,
            hallucinated_ops=0,
            reflection_recoveries=0,
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
            success=ops_executed > 0,
            total_duration_ms=0.0,
            steps=all_steps,
            ops_executed=ops_executed,
            ops_succeeded=ops_succeeded,
            success_rate=success_rate,
            avg_step_duration_ms=0.0,
            ablation=ablation,
            feature_flags=self._extract_feature_flags(cfg),
            per_op_stats=self._compute_per_op_stats(all_steps),
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )

    def _run_b14_kg(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B14 KG ablation: parse spatial tasks with KG context on or off.

        Condition is driven by cfg.use_knowledge_graph (--no-kg to disable).
        Measures whether parsed ops reference correct KG object IDs - checked
        in both conditions, since a model can in principle name the right
        object from task text alone even without KG context. Run once with
        KG enabled (default) and again with --no-kg to compare conditions.
        No Unity required - parse-only.
        """
        import config.KnowledgeGraph as _kg_cfg
        from orchestrators.CommandParser import CommandParser
        from core.Imports import get_global_registry
        from .Result import AblationMetrics

        registry = get_global_registry()
        tasks = module.get_tasks(cfg)
        kg_on = cfg.use_knowledge_graph

        prev_kg = _kg_cfg.KNOWLEDGE_GRAPH_ENABLED
        _kg_cfg.KNOWLEDGE_GRAPH_ENABLED = kg_on
        try:
            if kg_on:
                module.populate_synthetic_kg(cfg.robot_id)

            parser = CommandParser(use_rag=cfg.use_rag)
            hallucinated = 0
            wrong_object = 0
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
                        continue
                    params_str = str(cmd.get("params", {})).lower()
                    uses_variable_ref = "$" in params_str
                    has_kg_ref = uses_variable_ref or any(
                        obj.lower() in params_str for obj in module.KG_OBJECTS
                    )
                    if not has_kg_ref and op_name in module.KG_AWARE_OPS:
                        wrong_object += 1
        finally:
            if kg_on:
                module.clear_synthetic_kg(cfg.robot_id)
            _kg_cfg.KNOWLEDGE_GRAPH_ENABLED = prev_kg

        total_bad = hallucinated + wrong_object
        ablation_succeeded = total_ops - total_bad
        rate = (ablation_succeeded / total_ops) if total_ops else 0.0
        condition = "enabled" if kg_on else "disabled"
        ablation = AblationMetrics(
            condition=condition,
            hallucinated_ops=total_bad,
            reflection_recoveries=0,
            negotiation_rounds=0,
            success_rate=rate,
            ops_executed=total_ops,
            ops_succeeded=ablation_succeeded,
        )

        return BenchmarkResult(
            benchmark_id=14,
            benchmark_name=_BENCHMARK_NAMES[14],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            success=(rate >= 0.5 and total_ops > 0),
            total_duration_ms=0.0,
            steps=[],
            ops_executed=total_ops,
            ops_succeeded=ablation_succeeded,
            success_rate=rate,
            avg_step_duration_ms=0.0,
            hallucinated_ops=total_bad,
            ablation=ablation,
            feature_flags=self._extract_feature_flags(cfg),
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )
