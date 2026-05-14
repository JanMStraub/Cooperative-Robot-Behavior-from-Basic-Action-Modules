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

from . import mock_registry
from .config import BenchmarkConfig
from .feature_flags import BenchmarkFeatureFlags
from .result import BenchmarkResult, ChainMetrics, StepResult, make_run_id

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
    9: "RAG Ablation",
    10: "Reflexion Ablation",
    11: "Negotiation Ablation",
    12: "Knowledge Graph Ablation",
    13: "VGN Ablation",
    14: "ROS vs Unity Movement",
}

_CASE_MODULES: Dict[int, str] = {
    1: "benchmarks.cases.b1_navigate_to_object",
    2: "benchmarks.cases.b2_sequential_navigation",
    3: "benchmarks.cases.b3_navigate_and_lift",
    4: "benchmarks.cases.b4_pick_and_place",
    5: "benchmarks.cases.b5_pose_aware_grasp",
    6: "benchmarks.cases.b6_robot_handoff",
    7: "benchmarks.cases.b7_dual_robot_reorient",
    8: "benchmarks.cases.b8_heterogeneous_chain",
    9: "benchmarks.cases.b9_rag_ablation",
    10: "benchmarks.cases.b10_reflexion_ablation",
    11: "benchmarks.cases.b11_negotiation_ablation",
    12: "benchmarks.cases.b12_kg_ablation",
    13: "benchmarks.cases.b13_vgn_ablation",
    14: "benchmarks.cases.b14_ros_ablation",
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

            if benchmark_id == 9:
                return self._run_b9_rag(cfg, module)
            if benchmark_id == 10:
                return self._run_b10_reflexion(cfg, module)
            if benchmark_id == 11:
                return self._run_b11_negotiation(cfg, module)
            if benchmark_id == 12:
                return self._run_b12_kg(cfg, module)
            if benchmark_id == 13:
                return self._run_b13_vgn(cfg, module)
            if benchmark_id == 14:
                return self._run_b14_ros(cfg, module)

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

    def _build_sequence_message(
        self,
        payload: Union[str, List[Dict[str, Any]]],
        robot_id: str,
        flags: BenchmarkFeatureFlags,
    ) -> bytes:
        """
        Build the raw SEQUENCE_QUERY bytes for the given payload and flags.

        Args:
            payload: NL string or op list (EXEC: prefix added for lists).
            robot_id: Robot to execute the sequence.
            flags: Feature flag overrides to embed in message.

        Returns:
            Complete message bytes ready to send over TCP.
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

        Args:
            payload: NL task string → LLM parses; list of op dicts → EXEC: prefix.
            robot_id: Primary robot for this sequence.
            cfg: Benchmark config (timeout, dry_run).
            flags: Optional feature flag overrides embedded in message (live mode only).

        Returns:
            Result dict with keys: success, results, total_duration_ms.
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
                ops = parse_result.get("commands", []) if parse_result.get("success") else []
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

    def _extract_feature_flags(self, cfg: BenchmarkConfig) -> dict:
        """
        Extract active feature flags from a BenchmarkConfig as a flat, indexed dict.

        Having these at the top level of BenchmarkResult makes them trivially
        queryable without digging into config_snapshot.

        Args:
            cfg: BenchmarkConfig or DualRobotConfig instance.

        Returns:
            Dict of boolean feature flags.
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

        Args:
            steps: List of StepResult objects.

        Returns:
            Dict keyed by operation name with count, fail_count, avg_duration_ms.
        """
        buckets: Dict[str, Dict[str, Any]] = {}
        for s in steps:
            b = buckets.setdefault(s.operation, {"count": 0, "fail_count": 0, "total_ms": 0.0})
            b["count"] += 1
            b["total_ms"] += s.duration_ms
            if not s.success:
                b["fail_count"] += 1
        return {
            op: {
                "count": v["count"],
                "fail_count": v["fail_count"],
                "avg_duration_ms": round(v["total_ms"] / v["count"], 1) if v["count"] else 0.0,
            }
            for op, v in buckets.items()
        }

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

        Args:
            benchmark_id: Benchmark identifier.
            cfg: Config used for this run.
            raw: Response dict.

        Returns:
            Populated BenchmarkResult.
        """
        parsed_cmds = raw.get("parsed_commands") or []
        steps = self._parse_steps(raw.get("results") or [], parsed_cmds)
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

        Args:
            results: Per-step result dicts from SequenceServer.
            parsed_cmds: Parsed command list; used to populate robot_id and
                parallel_group_id on each StepResult.

        Returns:
            List of StepResult objects.
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
                    robot_id=cmd.get("params", {}).get("robot_id") or None,
                    parallel_group_id=cmd.get("parallel_group") or None,
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
            task_steps = self._parse_steps(raw.get("results") or [], raw.get("parsed_commands") or [])
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
            feature_flags=self._extract_feature_flags(cfg),
            per_op_stats=self._compute_per_op_stats(all_steps),
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )

    def _run_b9_rag(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B9 RAG ablation: parse same tasks with and without RAG, compare hallucinations.

        No Unity connection required — parse-only.

        Args:
            cfg: BenchmarkConfig (use_rag flag controls which condition runs).
            module: b9_rag_ablation module exposing get_tasks().

        Returns:
            BenchmarkResult with hallucinated_ops and ablation fields populated.
        """
        from orchestrators.CommandParser import CommandParser
        from core.Imports import get_global_registry
        from .result import AblationMetrics

        registry = get_global_registry()
        tasks = module.get_tasks(cfg)
        parser = CommandParser(use_rag=cfg.use_rag)

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

        success_rate = (succeeded / total_ops) if total_ops else 0.0
        condition = "enabled" if cfg.use_rag else "disabled"

        ablation = AblationMetrics(
            condition=condition,
            hallucinated_ops=hallucinated,
            reflexion_recoveries=0,
            negotiation_rounds=0,
            success_rate=success_rate,
            ops_executed=total_ops,
            ops_succeeded=succeeded,
        )

        return BenchmarkResult(
            benchmark_id=9,
            benchmark_name=_BENCHMARK_NAMES[9],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            success=(hallucinated == 0),
            total_duration_ms=0.0,
            steps=[],
            ops_executed=total_ops,
            ops_succeeded=succeeded,
            success_rate=success_rate,
            avg_step_duration_ms=0.0,
            hallucinated_ops=hallucinated,
            ablation=ablation,
            feature_flags=self._extract_feature_flags(cfg),
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )

    def _run_b10_reflexion_live(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Send B10 tasks to live SequenceServer; measure reflexion recoveries from server response.

        Args:
            cfg: BenchmarkConfig (reflexion_enabled controls server-side reflexion).
            module: b10_reflexion_ablation module exposing get_tasks().

        Returns:
            BenchmarkResult with reflexion_recoveries populated.
        """
        from .result import AblationMetrics

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
            task_steps = self._parse_steps(raw.get("results") or [], raw.get("parsed_commands") or [])
            step_offset = len(all_steps)
            for s in task_steps:
                s.index += step_offset
            all_steps.extend(task_steps)
            ops_executed += raw.get("ops_executed", len(task_steps))
            ops_succeeded += raw.get("ops_succeeded", sum(1 for s in task_steps if s.success))
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
            benchmark_id=10,
            benchmark_name=_BENCHMARK_NAMES[10],
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

    def _run_b10_reflexion(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B10 Reflexion ablation.

        In live mode (cfg.execution_mode == "live"), sends NL tasks to SequenceServer and
        reads reflexion_recoveries from the server response.

        In offline mode, uses first_fail_nav mock and dry-run execution.

        Args:
            cfg: BenchmarkConfig (reflexion_enabled controls reflexion behaviour).
            module: b10_reflexion_ablation module exposing get_tasks().

        Returns:
            BenchmarkResult with reflexion_recoveries populated.
        """
        if cfg.execution_mode == "live":
            return self._run_b10_reflexion_live(cfg, module)

        from orchestrators.CommandParser import CommandParser
        from .result import AblationMetrics

        tasks = module.get_tasks(cfg)
        all_steps: List[StepResult] = []
        total_ms = 0.0
        total_recoveries = 0
        completed_tasks = 0

        mock_original = mock_registry.install_mock("first_fail_nav")
        try:
            parser = CommandParser(use_rag=cfg.use_rag)
            for task in tasks:
                mock_registry.reset_counts()
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
            mock_registry.restore_mock(mock_original)

        ops_executed = len(all_steps)
        ops_succeeded = sum(1 for s in all_steps if s.success)
        success_rate = (completed_tasks / len(tasks)) if tasks else 0.0
        condition = "enabled" if cfg.reflexion_enabled else "disabled"

        ablation = AblationMetrics(
            condition=condition,
            hallucinated_ops=0,
            reflexion_recoveries=total_recoveries,
            negotiation_rounds=0,
            success_rate=success_rate,
            ops_executed=ops_executed,
            ops_succeeded=ops_succeeded,
        )

        return BenchmarkResult(
            benchmark_id=10,
            benchmark_name=_BENCHMARK_NAMES[10],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            # Offline ablation success = data was collected. Task failures in the
            # disabled condition are expected — reflexion_recoveries tells the story.
            success=True,
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
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )

    def _run_b11_negotiation_live(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Send B11 dual-robot tasks to live SequenceServer; measure negotiation rounds via hub.

        Args:
            cfg: DualRobotConfig (use_negotiation controls NEGOTIATION_ENABLED).
            module: b11_negotiation_ablation module exposing get_tasks().

        Returns:
            BenchmarkResult with negotiation_rounds populated.
        """
        from core.Imports import get_negotiation_hub
        from .result import AblationMetrics

        tasks = module.get_tasks(cfg)
        total_rounds = 0
        completed = 0
        ops_executed = 0
        ops_succeeded = 0
        all_steps: List[StepResult] = []
        total_ms = 0.0
        hub = get_negotiation_hub()
        robot_id = getattr(cfg, "robot_id_a", cfg.robot_id)
        _flags = BenchmarkFeatureFlags(use_negotiation=getattr(cfg, "use_negotiation", True))

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
            ops_succeeded += raw.get("ops_succeeded", sum(1 for s in task_steps if s.success))
            total_ms += float(raw.get("total_duration_ms", 0.0))
            if hub is not None:
                total_rounds += hub.get_last_round_count()

        success_rate = completed / len(tasks) if tasks else 0.0
        ablation = AblationMetrics(
            condition="enabled" if getattr(cfg, "use_negotiation", True) else "disabled",
            hallucinated_ops=0,
            reflexion_recoveries=0,
            negotiation_rounds=total_rounds,
            success_rate=success_rate,
            ops_executed=ops_executed,
            ops_succeeded=ops_succeeded,
        )
        return BenchmarkResult(
            benchmark_id=11,
            benchmark_name=_BENCHMARK_NAMES[11],
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

    def _run_b11_negotiation(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B11 Negotiation ablation.

        In live mode (cfg.execution_mode == "live"), sends dual-robot NL tasks to
        SequenceServer and reads negotiation_rounds from the hub.

        In offline mode, patches NEGOTIATION_ENABLED, runs dual-robot tasks in dry-run,
        and reads negotiation_rounds from the hub after each task.

        Args:
            cfg: DualRobotConfig (use_negotiation controls the config patch).
            module: b11_negotiation_ablation module exposing get_tasks().

        Returns:
            BenchmarkResult with negotiation_rounds and ablation fields populated.
        """
        if cfg.execution_mode == "live":
            return self._run_b11_negotiation_live(cfg, module)

        import config.Negotiation as _neg_cfg
        from .result import AblationMetrics

        tasks = module.get_tasks(cfg)
        all_steps: List[StepResult] = []
        total_ms = 0.0
        completed_tasks = 0
        total_negotiation_rounds = 0

        prev_neg = _neg_cfg.NEGOTIATION_ENABLED
        _neg_cfg.NEGOTIATION_ENABLED = getattr(cfg, "use_negotiation", True)

        mock_original = mock_registry.install_mock("always_succeed")
        try:
            from orchestrators.CommandParser import CommandParser
            parser = CommandParser(use_rag=cfg.use_rag)
            robot_id = getattr(cfg, "robot_id_a", cfg.robot_id)

            for task in tasks:
                parse_result = parser.parse(task, robot_id=robot_id)
                if not parse_result["success"]:
                    continue

                ops = parse_result["commands"]
                # Count signal ops as a proxy for negotiation coordination points:
                # a negotiated plan includes explicit signal/wait_for_signal pairs that
                # a non-negotiated plan omits.
                total_negotiation_rounds += sum(
                    1 for op in ops if op.get("operation") == "signal"
                )

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
            mock_registry.restore_mock(mock_original)
            _neg_cfg.NEGOTIATION_ENABLED = prev_neg

        ops_executed = len(all_steps)
        ops_succeeded = sum(1 for s in all_steps if s.success)
        success_rate = (completed_tasks / len(tasks)) if tasks else 0.0
        condition = "enabled" if getattr(cfg, "use_negotiation", True) else "disabled"

        ablation = AblationMetrics(
            condition=condition,
            hallucinated_ops=0,
            reflexion_recoveries=0,
            negotiation_rounds=total_negotiation_rounds,
            success_rate=success_rate,
            ops_executed=ops_executed,
            ops_succeeded=ops_succeeded,
        )

        return BenchmarkResult(
            benchmark_id=11,
            benchmark_name=_BENCHMARK_NAMES[11],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            success=(completed_tasks == len(tasks)),
            total_duration_ms=total_ms,
            steps=all_steps,
            ops_executed=ops_executed,
            ops_succeeded=ops_succeeded,
            success_rate=success_rate,
            avg_step_duration_ms=(total_ms / ops_executed) if ops_executed else 0.0,
            negotiation_rounds=total_negotiation_rounds,
            ablation=ablation,
            feature_flags=self._extract_feature_flags(cfg),
            per_op_stats=self._compute_per_op_stats(all_steps),
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )

    def _run_b13_vgn(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B13 VGN ablation: execute grasp tasks with VGN enabled vs disabled.

        Offline mode patches VGN_ENABLED, mocks operations with always_succeed, and
        verifies the parse+dry-run path works for both conditions.  Live mode sends
        NL grasp tasks to SequenceServer and measures actual grasp success rate.

        Args:
            cfg: BenchmarkConfig (use_vgn controls VGN_ENABLED patch).
            module: b13_vgn_ablation module exposing get_tasks().

        Returns:
            BenchmarkResult with ablation fields populated.
        """
        import config.Servers as _srv_cfg
        from orchestrators.CommandParser import CommandParser
        from .result import AblationMetrics

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
                ops_succeeded += raw.get("ops_succeeded", sum(1 for s in task_steps if s.success))
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
                ablation=ablation,
                feature_flags=self._extract_feature_flags(cfg),
                per_op_stats=self._compute_per_op_stats(all_steps),
                execution_mode="live",
            )

        # Offline: parse + dry-run with always_succeed mock
        prev_vgn = _srv_cfg.VGN_ENABLED
        _srv_cfg.VGN_ENABLED = cfg.use_vgn
        mock_original = mock_registry.install_mock("always_succeed")
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
            mock_registry.restore_mock(mock_original)
            _srv_cfg.VGN_ENABLED = prev_vgn

        ops_executed = len(all_steps)
        ops_succeeded = sum(1 for s in all_steps if s.success)
        success_rate = (completed_tasks / len(tasks)) if tasks else 0.0
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
            benchmark_id=13,
            benchmark_name=_BENCHMARK_NAMES[13],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            success=(completed_tasks == len(tasks)),
            total_duration_ms=total_ms,
            steps=all_steps,
            ops_executed=ops_executed,
            ops_succeeded=ops_succeeded,
            success_rate=success_rate,
            avg_step_duration_ms=(total_ms / ops_executed) if ops_executed else 0.0,
            ablation=ablation,
            feature_flags=self._extract_feature_flags(cfg),
            per_op_stats=self._compute_per_op_stats(all_steps),
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )

    def _run_b14_ros(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B14 ROS vs Unity movement ablation.

        Patches config.ROS.ROS_ENABLED and DEFAULT_CONTROL_MODE, then executes
        movement tasks.  Offline mode uses always_succeed mock + dry-run to verify
        the parse path.  Live mode sends NL movement tasks to SequenceServer;
        avg_step_duration_ms captures MoveIt planning overhead vs direct Unity TCP.

        Args:
            cfg: BenchmarkConfig (use_ros_movement controls ROS_ENABLED patch).
            module: b14_ros_ablation module exposing get_tasks().

        Returns:
            BenchmarkResult with ablation fields populated.
        """
        import config.ROS as _ros_cfg
        from orchestrators.CommandParser import CommandParser
        from .result import AblationMetrics

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
                benchmark_id=14,
                benchmark_name=_BENCHMARK_NAMES[14],
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

        # Offline: parse + dry-run with always_succeed mock
        prev_enabled = _ros_cfg.ROS_ENABLED
        prev_mode = _ros_cfg.DEFAULT_CONTROL_MODE
        _ros_cfg.ROS_ENABLED = cfg.use_ros_movement
        _ros_cfg.DEFAULT_CONTROL_MODE = "ros" if cfg.use_ros_movement else "unity"
        mock_original = mock_registry.install_mock("always_succeed")
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
            mock_registry.restore_mock(mock_original)
            _ros_cfg.ROS_ENABLED = prev_enabled
            _ros_cfg.DEFAULT_CONTROL_MODE = prev_mode

        ops_executed = len(all_steps)
        ops_succeeded = sum(1 for s in all_steps if s.success)
        success_rate = (completed_tasks / len(tasks)) if tasks else 0.0
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
            benchmark_id=14,
            benchmark_name=_BENCHMARK_NAMES[14],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            success=(completed_tasks == len(tasks)),
            total_duration_ms=total_ms,
            steps=all_steps,
            ops_executed=ops_executed,
            ops_succeeded=ops_succeeded,
            success_rate=success_rate,
            avg_step_duration_ms=(total_ms / ops_executed) if ops_executed else 0.0,
            ablation=ablation,
            feature_flags=self._extract_feature_flags(cfg),
            per_op_stats=self._compute_per_op_stats(all_steps),
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )

    def _run_b12_kg(self, cfg: BenchmarkConfig, module) -> BenchmarkResult:
        """
        Run B12 KG ablation: parse spatial tasks with KG populated vs disabled.

        Populates KG with synthetic state (two objects + one nearby robot), then
        measures whether parsed ops reference the correct KG object IDs.

        No Unity required — parse-only.

        Args:
            cfg: BenchmarkConfig (use_knowledge_graph controls KNOWLEDGE_GRAPH_ENABLED).
            module: b12_kg_ablation module.

        Returns:
            BenchmarkResult with hallucinated_ops and ablation fields.
        """
        import config.KnowledgeGraph as _kg_cfg
        from orchestrators.CommandParser import CommandParser
        from core.Imports import get_global_registry
        from .result import AblationMetrics

        registry = get_global_registry()
        tasks = module.get_tasks(cfg)

        prev_kg = _kg_cfg.KNOWLEDGE_GRAPH_ENABLED
        _kg_cfg.KNOWLEDGE_GRAPH_ENABLED = cfg.use_knowledge_graph

        try:
            if cfg.use_knowledge_graph:
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
                        if cfg.use_knowledge_graph:
                            params_str = str(cmd.get("params", {})).lower()
                            # Variable references ($target.color etc.) are correct
                            # variable-passing — not a failure to use KG object IDs.
                            uses_variable_ref = "$" in params_str
                            has_kg_ref = uses_variable_ref or any(
                                obj.lower() in params_str
                                for obj in module.KG_OBJECTS
                            )
                            if not has_kg_ref and "object" in op_name:
                                wrong_object += 1

        finally:
            if cfg.use_knowledge_graph:
                module.clear_synthetic_kg()
            _kg_cfg.KNOWLEDGE_GRAPH_ENABLED = prev_kg

        total_bad = hallucinated + wrong_object
        ablation_succeeded = total_ops - total_bad
        ablation_success_rate = (ablation_succeeded / total_ops) if total_ops else 0.0
        condition = "enabled" if cfg.use_knowledge_graph else "disabled"

        ablation = AblationMetrics(
            condition=condition,
            hallucinated_ops=total_bad,
            reflexion_recoveries=0,
            negotiation_rounds=0,
            success_rate=ablation_success_rate,
            ops_executed=total_ops,
            ops_succeeded=ablation_succeeded,
        )

        return BenchmarkResult(
            benchmark_id=12,
            benchmark_name=_BENCHMARK_NAMES[12],
            run_id=make_run_id(),
            config_snapshot=dataclasses.asdict(cfg),
            # Threshold-based: LLM is stochastic; require ≥50% clean ops rather
            # than zero tolerance, which would cause flaky results across runs.
            success=ablation_success_rate >= 0.5,
            total_duration_ms=0.0,
            steps=[],
            ops_executed=total_ops,
            ops_succeeded=ablation_succeeded,
            success_rate=ablation_success_rate,
            avg_step_duration_ms=0.0,
            hallucinated_ops=total_bad,
            ablation=ablation,
            feature_flags=self._extract_feature_flags(cfg),
            execution_mode=getattr(cfg, "execution_mode", "offline"),
        )
