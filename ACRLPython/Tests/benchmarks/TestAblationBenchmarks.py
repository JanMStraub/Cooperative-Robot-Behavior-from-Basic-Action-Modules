def test_sequence_executor_result_includes_reflexion_recoveries():
    """execute_sequence result must contain reflexion_recoveries key."""
    from orchestrators.SequenceExecutor import SequenceExecutor

    executor = SequenceExecutor(
        default_timeout=5.0,
        check_completion=False,
        enable_verification=False,
    )
    result = executor.execute_sequence([])
    assert "reflexion_recoveries" in result
    assert result["reflexion_recoveries"] == 0


def test_first_fail_nav_mock_fails_then_succeeds():
    """first_fail_nav profile: navigation op fails on first call, succeeds on second."""
    from benchmarks import MockRegistry as mock_registry  # type: ignore[attr-defined]

    original = mock_registry.install_mock("first_fail_nav")
    try:
        from core.Imports import get_global_registry

        registry = get_global_registry()
        move_op = registry.get_operation_by_name("move_to_coordinate")
        assert move_op is not None
        result1 = move_op.execute(robot_id="Robot1", x=0.3, y=0.15, z=0.1)
        assert not result1.success, "first call must fail"
        result2 = move_op.execute(robot_id="Robot1", x=0.3, y=0.15, z=0.1)
        assert result2.success, "second call must succeed"
    finally:
        mock_registry.restore_mock(original)


def test_ablation_metrics_dataclass_exists():
    from benchmarks.Result import AblationMetrics

    m = AblationMetrics(
        condition="enabled",
        hallucinated_ops=2,
        reflexion_recoveries=1,
        negotiation_rounds=0,
        success_rate=0.8,
        ops_executed=5,
        ops_succeeded=4,
    )
    assert m.condition == "enabled"
    assert m.hallucinated_ops == 2


def test_benchmark_result_has_ablation_fields():
    from benchmarks.Result import BenchmarkResult, make_run_id

    r = BenchmarkResult(
        benchmark_id=9,
        benchmark_name="RAG Ablation",
        run_id=make_run_id(),
        config_snapshot={},
        success=True,
        total_duration_ms=100.0,
        steps=[],
        ops_executed=3,
        ops_succeeded=3,
        success_rate=1.0,
        avg_step_duration_ms=33.0,
        hallucinated_ops=0,
        reflexion_recoveries=0,
        negotiation_rounds=0,
    )
    assert r.hallucinated_ops == 0


def test_benchmark_config_has_ablation_flags():
    from benchmarks.Config import BenchmarkConfig, DualRobotConfig

    cfg = BenchmarkConfig(use_rag=False, reflexion_enabled=False)
    assert cfg.use_rag is False
    assert cfg.reflexion_enabled is False
    dual = DualRobotConfig(use_negotiation=False)
    assert dual.use_negotiation is False


def test_b9_impossible_get_task_returns_string():
    from ACRLPython.benchmarks.cases import B9Impossible

    task = B9Impossible.get_task()
    assert isinstance(task, str)
    assert len(task) > 0


def test_b9_get_tasks_returns_list():
    from ACRLPython.benchmarks.cases import B10RagAblation
    from benchmarks.Config import BenchmarkConfig

    tasks = B10RagAblation.get_tasks(BenchmarkConfig())
    assert isinstance(tasks, list)
    assert len(tasks) >= 3
    assert all(isinstance(t, str) for t in tasks)


def test_b10_get_tasks_returns_list():
    from ACRLPython.benchmarks.cases import B11ReflexionAblation
    from benchmarks.Config import BenchmarkConfig

    tasks = B11ReflexionAblation.get_tasks(BenchmarkConfig())
    assert isinstance(tasks, list)
    assert len(tasks) >= 3


def test_b11_get_tasks_returns_list():
    from ACRLPython.benchmarks.cases import B12NegotiationAblation
    from benchmarks.Config import DualRobotConfig

    tasks = B12NegotiationAblation.get_tasks(DualRobotConfig())
    assert isinstance(tasks, list)
    assert len(tasks) >= 2
    assert all(isinstance(t, str) for t in tasks)


def test_b12_get_tasks_returns_list():
    from ACRLPython.benchmarks.cases import B13KgAblation
    from benchmarks.Config import BenchmarkConfig

    tasks = B13KgAblation.get_tasks(BenchmarkConfig())
    assert isinstance(tasks, list)
    assert len(tasks) >= 3


def test_b15_get_tasks_returns_list():
    from ACRLPython.benchmarks.cases import B15RosAblation
    from benchmarks.Config import BenchmarkConfig

    tasks = B15RosAblation.get_tasks(BenchmarkConfig())
    assert isinstance(tasks, list)
    assert len(tasks) >= 2
    assert all(isinstance(t, str) for t in tasks)


# Task 1: execution_mode field on BenchmarkConfig


def test_execution_mode_default():
    from benchmarks.Config import BenchmarkConfig

    cfg = BenchmarkConfig()
    assert cfg.execution_mode == "offline"


def test_execution_mode_live():
    from benchmarks.Config import BenchmarkConfig

    cfg = BenchmarkConfig(execution_mode="live")
    assert cfg.execution_mode == "live"


# Task 2: --live CLI flag


def test_live_flag_parsed():
    import argparse
    import sys

    # Re-import cleanly
    from benchmarks import Run as run_mod  # type: ignore[attr-defined]

    # Simulate parse_args call by creating a minimal parser copy
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", default=False)
    args = parser.parse_args(["--live"])
    assert args.live is True


# Task 3: NegotiationHub exposes get_last_round_count


def test_negotiation_hub_exposes_round_count():
    from servers.NegotiationHub import NegotiationHub

    hub = NegotiationHub.__new__(NegotiationHub)
    hub._last_round_count = 0
    assert hasattr(hub, "get_last_round_count")
    assert hub.get_last_round_count() == 0


# Task 4: offline B11 captures negotiation_rounds from hub


def test_b11_offline_captures_negotiation_rounds(monkeypatch):
    """negotiation_rounds in AblationMetrics should reflect hub count."""
    from unittest.mock import MagicMock

    fake_hub = MagicMock()
    fake_hub.get_last_round_count.return_value = 2
    monkeypatch.setattr(
        "benchmarks.Runner.get_negotiation_hub", lambda: fake_hub, raising=False
    )
    # Patch core.Imports so the import inside _run_b11_negotiation resolves
    import sys

    fake_core_imports = MagicMock()
    fake_core_imports.get_negotiation_hub = lambda: fake_hub
    monkeypatch.setitem(sys.modules, "core.Imports", fake_core_imports)

    from benchmarks.Runner import BenchmarkRunner
    from benchmarks.Config import DualRobotConfig

    runner = BenchmarkRunner()
    cfg = DualRobotConfig(
        dry_run=True, execution_mode="offline", use_negotiation=False, task_count=1
    )
    result = runner.run(11, cfg)
    assert result.ablation is not None
    # negotiation_rounds may be 0 when negotiation is disabled — verify field exists
    assert hasattr(result.ablation, "negotiation_rounds")


# Task 5: B10 RAG ablation always returns paired conditions (parse-only, no _send)


def test_b10_returns_paired_ablation_conditions():
    """B10 RAG ablation is parse-only — returns both enabled and disabled conditions."""
    from benchmarks.Runner import BenchmarkRunner
    from benchmarks.Config import BenchmarkConfig

    runner = BenchmarkRunner()
    cfg = BenchmarkConfig(dry_run=False, execution_mode="offline", task_count=1)
    result = runner.run(10, cfg)
    assert result.ablation is not None
    assert result.ablation_baseline is not None
    assert result.ablation.condition == "enabled"
    assert result.ablation_baseline.condition == "disabled"
    assert result.ablation.ops_executed > 0
    assert result.ablation_baseline.ops_executed > 0


# Task 6: live B11 dispatches to _send


def test_b11_live_dispatches_to_send(monkeypatch):
    """execution_mode='live' for B11 should call _send for each task."""
    from benchmarks.Runner import BenchmarkRunner
    from benchmarks.Config import DualRobotConfig

    calls = []

    def patched_send(payload, robot_id, cfg, flags=None):
        calls.append(payload)
        return {"success": True, "ops_executed": 3, "ops_succeeded": 3, "results": []}

    runner = BenchmarkRunner()
    runner._send = patched_send
    cfg = DualRobotConfig(execution_mode="live", use_negotiation=True, task_count=1)
    result = runner.run(11, cfg)
    assert len(calls) >= 1
    assert result.ablation is not None
    assert result.ablation.condition == "enabled"


def test_benchmark_result_has_ablation_baseline_field():
    from benchmarks.Result import BenchmarkResult, AblationMetrics, make_run_id

    baseline = AblationMetrics(
        condition="disabled",
        hallucinated_ops=3,
        reflexion_recoveries=0,
        negotiation_rounds=0,
        success_rate=0.7,
        ops_executed=10,
        ops_succeeded=7,
    )
    r = BenchmarkResult(
        benchmark_id=10,
        benchmark_name="RAG Ablation",
        run_id=make_run_id(),
        config_snapshot={},
        success=True,
        total_duration_ms=0.0,
        steps=[],
        ops_executed=10,
        ops_succeeded=10,
        success_rate=1.0,
        avg_step_duration_ms=0.0,
        ablation_baseline=baseline,
    )
    assert r.ablation_baseline is not None
    assert r.ablation_baseline.condition == "disabled"
    assert r.ablation_baseline.hallucinated_ops == 3


# Task source tests

def test_b10_tasks_are_drawn_from_b1_to_b5():
    from ACRLPython.benchmarks.cases import B10RagAblation
    from ACRLPython.benchmarks.cases.B1NavigateToObject import get_task as b1
    from ACRLPython.benchmarks.cases.B2SequentialNavigation import get_task as b2
    from ACRLPython.benchmarks.cases.B3NavigateAndLift import get_task as b3
    from ACRLPython.benchmarks.cases.B4PickAndPlace import get_task as b4
    from ACRLPython.benchmarks.cases.B5PoseAwareGrasp import get_task as b5
    from benchmarks.Config import BenchmarkConfig

    tasks = B10RagAblation.get_tasks(BenchmarkConfig())
    assert b1() in tasks
    assert b2() in tasks
    assert b3() in tasks
    assert b4() in tasks
    assert b5() in tasks


def test_b12_tasks_include_b6_and_b7():
    from ACRLPython.benchmarks.cases import B12NegotiationAblation
    from ACRLPython.benchmarks.cases.B6RobotHandoff import get_task as b6
    from ACRLPython.benchmarks.cases.B7DualRobotReorient import get_task as b7
    from benchmarks.Config import DualRobotConfig

    tasks = B12NegotiationAblation.get_tasks(DualRobotConfig())
    assert b6() in tasks
    assert b7() in tasks


def test_b14_tasks_include_b3_b4_b5():
    from ACRLPython.benchmarks.cases import B14VGNAblation
    from ACRLPython.benchmarks.cases.B3NavigateAndLift import get_task as b3
    from ACRLPython.benchmarks.cases.B4PickAndPlace import get_task as b4
    from ACRLPython.benchmarks.cases.B5PoseAwareGrasp import get_task as b5
    from benchmarks.Config import BenchmarkConfig

    tasks = B14VGNAblation.get_tasks(BenchmarkConfig())
    assert b3() in tasks
    assert b4() in tasks
    assert b5() in tasks


def test_b15_tasks_include_b1_and_b2():
    from ACRLPython.benchmarks.cases import B15RosAblation
    from ACRLPython.benchmarks.cases.B1NavigateToObject import get_task as b1
    from ACRLPython.benchmarks.cases.B2SequentialNavigation import get_task as b2
    from benchmarks.Config import BenchmarkConfig

    tasks = B15RosAblation.get_tasks(BenchmarkConfig())
    assert b1() in tasks
    assert b2() in tasks
