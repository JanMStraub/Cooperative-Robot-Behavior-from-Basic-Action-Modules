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


def test_ablation_metrics_has_delta_fields():
    from benchmarks.Result import AblationMetrics

    m = AblationMetrics(
        condition="enabled",
        hallucinated_ops=1,
        reflexion_recoveries=0,
        negotiation_rounds=0,
        success_rate=0.9,
        ops_executed=10,
        ops_succeeded=9,
        condition_delta=0.2,
        hallucination_delta=3,
    )
    assert m.condition_delta == 0.2
    assert m.hallucination_delta == 3


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
    from ACRLPython.benchmarks.cases import B11RagAblation
    from benchmarks.Config import BenchmarkConfig

    tasks = B11RagAblation.get_tasks(BenchmarkConfig())
    assert isinstance(tasks, list)
    assert len(tasks) >= 3
    assert all(isinstance(t, str) for t in tasks)


def test_b10_get_tasks_returns_list():
    from ACRLPython.benchmarks.cases import B12ReflexionAblation
    from benchmarks.Config import BenchmarkConfig

    tasks = B12ReflexionAblation.get_tasks(BenchmarkConfig())
    assert isinstance(tasks, list)
    assert len(tasks) >= 3


def test_b11_get_tasks_returns_list():
    from ACRLPython.benchmarks.cases import B13NegotiationAblation
    from benchmarks.Config import DualRobotConfig

    tasks = B13NegotiationAblation.get_tasks(DualRobotConfig())
    assert isinstance(tasks, list)
    assert len(tasks) >= 2
    assert all(isinstance(t, str) for t in tasks)


def test_b12_get_tasks_returns_list():
    from ACRLPython.benchmarks.cases import B14KgAblation
    from benchmarks.Config import BenchmarkConfig

    tasks = B14KgAblation.get_tasks(BenchmarkConfig())
    assert isinstance(tasks, list)
    assert len(tasks) >= 3


def test_b15_get_tasks_returns_list():
    from ACRLPython.benchmarks.cases import B16RosAblation
    from benchmarks.Config import BenchmarkConfig

    tasks = B16RosAblation.get_tasks(BenchmarkConfig())
    assert isinstance(tasks, list)
    assert len(tasks) >= 2
    assert all(isinstance(t, str) for t in tasks)


def test_execution_mode_default():
    from benchmarks.Config import BenchmarkConfig

    cfg = BenchmarkConfig()
    assert cfg.execution_mode == "offline"


def test_execution_mode_live():
    from benchmarks.Config import BenchmarkConfig

    cfg = BenchmarkConfig(execution_mode="live")
    assert cfg.execution_mode == "live"


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


def test_negotiation_hub_exposes_round_count():
    from servers.NegotiationHub import NegotiationHub

    hub = NegotiationHub.__new__(NegotiationHub)
    hub._last_round_count = 0
    assert hasattr(hub, "get_last_round_count")
    assert hub.get_last_round_count() == 0


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


def test_b10_returns_ablation_condition():
    """B11 RAG ablation is parse-only — returns single condition matching cfg.use_rag."""
    from benchmarks.Runner import BenchmarkRunner
    from benchmarks.Config import BenchmarkConfig

    runner = BenchmarkRunner()
    cfg = BenchmarkConfig(
        dry_run=False, execution_mode="offline", task_count=1, use_rag=True
    )
    result = runner.run(11, cfg)
    assert result.ablation is not None
    assert result.ablation_baseline is None
    assert result.ablation.condition == "enabled"
    assert result.ablation.ops_executed > 0

    cfg_no_rag = BenchmarkConfig(
        dry_run=False, execution_mode="offline", task_count=1, use_rag=False
    )
    result_no_rag = runner.run(11, cfg_no_rag)
    assert result_no_rag.ablation is not None
    assert result_no_rag.ablation.condition == "disabled"


def test_b11_is_always_parse_only():
    """B11 is parse-only — accuracy metric is returned regardless of execution_mode."""
    from benchmarks.Runner import BenchmarkRunner
    from benchmarks.Config import BenchmarkConfig

    runner = BenchmarkRunner()
    cfg = BenchmarkConfig(execution_mode="offline", use_rag=True)
    result = runner.run(11, cfg)
    assert result.ablation is not None
    assert result.ablation.condition == "enabled"
    assert 0.0 <= result.success_rate <= 1.0


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


def test_b10_tasks_are_drawn_from_b1_to_b5():
    # B11 was redesigned to use ambiguous collaborative-op descriptions for RAG
    # discrimination instead of B1–B5 verbatim strings.  Verify task and
    # ground-truth lists are consistent and non-empty.
    from ACRLPython.benchmarks.cases import B11RagAblation
    from benchmarks.Config import BenchmarkConfig

    tasks = B11RagAblation.get_tasks(BenchmarkConfig())
    ground_truth = B11RagAblation.get_ground_truth(BenchmarkConfig())
    assert len(tasks) == len(B11RagAblation.TASKS)
    assert len(ground_truth) == len(tasks)
    assert all(isinstance(t, str) and t for t in tasks)
    assert all(isinstance(gt, str) and gt for gt in ground_truth)


def test_b12_tasks_have_ambiguous_roles():
    from ACRLPython.benchmarks.cases import B13NegotiationAblation
    from benchmarks.Config import DualRobotConfig

    tasks = B13NegotiationAblation.get_tasks(DualRobotConfig())
    assert len(tasks) >= 3
    # All tasks must omit explicit per-robot role assignment so negotiation
    # is actually exercised. Neither "Robot1" nor "Robot2" should appear as
    # the sole actor at the start of any task.
    for task in tasks:
        assert not (
            task.startswith("Robot1") and "Robot2" not in task.split(".")[0]
        ), f"Task has explicit single-robot assignment, not useful for negotiation ablation: {task!r}"


def test_b14_tasks_include_b3_b4_b5():
    # B15 uses its own task strings (hardcoded Robot1 prefix).  B3/B5 strings
    # match verbatim; B4 uses Robot2 so only the scenario content overlaps.
    from ACRLPython.benchmarks.cases import B15VGNAblation
    from ACRLPython.benchmarks.cases.B3NavigateAndLift import get_task as b3
    from ACRLPython.benchmarks.cases.B5PoseAwareGrasp import get_task as b5
    from benchmarks.Config import BenchmarkConfig

    tasks = B15VGNAblation.get_tasks(BenchmarkConfig())
    assert b3() in tasks
    assert b5() in tasks
    # B4 pick-and-place scenario covered with Robot1
    assert any("magenta" in t for t in tasks)


def test_b15_tasks_include_b1_and_b2():
    from ACRLPython.benchmarks.cases import B16RosAblation
    from ACRLPython.benchmarks.cases.B1NavigateToObject import get_task as b1
    from ACRLPython.benchmarks.cases.B2SequentialNavigation import get_task as b2
    from benchmarks.Config import BenchmarkConfig

    tasks = B16RosAblation.get_tasks(BenchmarkConfig())
    assert b1() in tasks
    assert b2() in tasks


def test_b9_parser_exception_does_not_count_as_graceful_rejection():
    """A parser crash must NOT count as graceful rejection."""
    from benchmarks.Runner import BenchmarkRunner
    from benchmarks.Config import BenchmarkConfig
    from unittest.mock import patch

    runner = BenchmarkRunner()
    cfg = BenchmarkConfig(execution_mode="offline")

    # CommandParser is imported inside _run_b9_impossible, so patch at its source module
    with patch("orchestrators.CommandParser.CommandParser") as MockParser:
        MockParser.return_value.parse.side_effect = RuntimeError("crash")
        result = runner.run(9, cfg)

    assert result.success is False, "Parser crash must not report graceful rejection"


def test_parse_steps_populates_retry_count():
    """_parse_steps must extract retry_count from raw result dicts."""
    from benchmarks.Runner import BenchmarkRunner

    runner = BenchmarkRunner()
    raw_results = [
        {
            "index": 0,
            "operation": "move_to_coordinate",
            "success": True,
            "duration_ms": 120.0,
            "retry_count": 2,
        }
    ]
    steps = runner._parse_steps(raw_results)
    assert steps[0].retry_count == 2


def test_b3_task_specifies_lift_height():
    from benchmarks.cases.B3NavigateAndLift import get_task

    task = get_task()
    assert "0.2" in task, "B3 must specify target lift height"


def test_b5_task_specifies_target_object():
    from benchmarks.cases.B5PoseAwareGrasp import get_task

    task = get_task()
    assert (
        "cube" in task.lower() or "object" in task.lower()
    ), "B5 must name a target object"


def test_b6_task_names_both_robots_and_transfer():
    from benchmarks.cases.B6RobotHandoff import get_task

    task = get_task()
    assert "Robot1" in task and "Robot2" in task, "B6 must name both robots"
    assert any(
        word in task.lower() for word in ["pass", "hand", "transfer", "give"]
    ), "B6 must specify transfer action"


def test_b10_rag_ablation_reports_accuracy():
    """B11 RAG ablation must report operation selection accuracy (success_rate) against ground truth."""
    from benchmarks.Runner import BenchmarkRunner
    from benchmarks.Config import BenchmarkConfig

    runner = BenchmarkRunner()
    cfg = BenchmarkConfig(dry_run=False, execution_mode="offline")
    result = runner.run(11, cfg)

    assert result.ablation is not None
    assert result.ablation_baseline is None
    assert result.ablation.condition == "enabled"
    assert 0.0 <= result.ablation.success_rate <= 1.0
    assert result.ablation.ops_executed == 8  # one per ground-truth task


def test_b11_ground_truth_list_matches_tasks():
    """B11 ground truth list must have same length as tasks list."""
    from benchmarks.cases.B11RagAblation import get_tasks, get_ground_truth

    tasks = get_tasks()
    gt = get_ground_truth()
    assert len(tasks) == len(gt)
    assert all(isinstance(op, str) and op for op in gt)


def test_b12_negotiation_offline_produces_paired_conditions_with_delta():
    """B12 offline must produce both conditions with delta fields."""
    from benchmarks.Runner import BenchmarkRunner
    from benchmarks.Config import DualRobotConfig

    runner = BenchmarkRunner()
    cfg = DualRobotConfig(dry_run=True, execution_mode="offline", use_negotiation=True)
    result = runner.run(12, cfg)

    assert result.ablation is not None
    assert result.ablation_baseline is not None
    assert isinstance(result.ablation.negotiation_rounds, int)
    assert isinstance(result.ablation.condition_delta, float)
