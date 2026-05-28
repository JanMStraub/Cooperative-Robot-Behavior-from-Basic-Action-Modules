#!/usr/bin/env python3
"""Tests for B16 Parallel Independent Tasks benchmark."""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Case file tests
# ---------------------------------------------------------------------------


def test_b16_get_task_returns_string():
    from benchmarks.cases.B16ParallelIndependent import get_task

    task = get_task()
    assert isinstance(task, str)
    assert len(task) > 0


def test_b16_task_names_both_robots():
    from benchmarks.cases.B16ParallelIndependent import get_task

    task = get_task()
    assert "Robot1" in task
    assert "Robot2" in task


def test_b16_task_contains_independence_signal():
    from benchmarks.cases.B16ParallelIndependent import get_task

    task = get_task().lower()
    assert any(word in task for word in ["independently", "simultaneously", "parallel"])


def test_b16_task_uses_disjoint_objects():
    from benchmarks.cases.B16ParallelIndependent import get_task

    task = get_task().lower()
    assert "blue" in task and ("red" in task or "green" in task), "B16 must reference two distinct objects"


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


def test_b16_registered_in_benchmark_names():
    from benchmarks.Runner import _BENCHMARK_NAMES

    assert 16 in _BENCHMARK_NAMES
    assert _BENCHMARK_NAMES[16] == "Parallel Independent Tasks"


def test_b16_registered_in_case_modules():
    from benchmarks.Runner import _CASE_MODULES

    assert 16 in _CASE_MODULES
    assert "B16" in _CASE_MODULES[16]


# ---------------------------------------------------------------------------
# _compute_parallelism_ratio unit tests
# ---------------------------------------------------------------------------


def _make_step(index, robot_id, group_id):
    from benchmarks.Result import StepResult

    return StepResult(
        index=index,
        operation="move_to_coordinate",
        success=True,
        duration_ms=50.0,
        error_code=None,
        error_message=None,
        robot_id=robot_id,
        parallel_group_id=group_id,
    )


def test_parallelism_ratio_all_shared():
    """All ops in same group, two robots → ratio 1.0."""
    from benchmarks.Runner import BenchmarkRunner

    runner = BenchmarkRunner()
    steps = [
        _make_step(0, "Robot1", 1),
        _make_step(1, "Robot1", 1),
        _make_step(2, "Robot2", 1),
        _make_step(3, "Robot2", 1),
    ]
    ratio, count = runner._compute_parallelism_ratio(steps)
    assert ratio == 1.0
    assert count == 4


def test_parallelism_ratio_no_groups():
    """No parallel_group_id set → ratio 0.0."""
    from benchmarks.Runner import BenchmarkRunner

    runner = BenchmarkRunner()
    steps = [
        _make_step(0, "Robot1", None),
        _make_step(1, "Robot2", None),
    ]
    ratio, count = runner._compute_parallelism_ratio(steps)
    assert ratio == 0.0
    assert count == 0


def test_parallelism_ratio_single_robot_groups():
    """Each robot in its own group (no shared groups) → ratio 0.0."""
    from benchmarks.Runner import BenchmarkRunner

    runner = BenchmarkRunner()
    steps = [
        _make_step(0, "Robot1", 1),
        _make_step(1, "Robot2", 2),
    ]
    ratio, count = runner._compute_parallelism_ratio(steps)
    assert ratio == 0.0
    assert count == 0


def test_parallelism_ratio_empty_steps():
    from benchmarks.Runner import BenchmarkRunner

    runner = BenchmarkRunner()
    ratio, count = runner._compute_parallelism_ratio([])
    assert ratio == 0.0
    assert count == 0


def test_parallelism_ratio_partial():
    """Two of four ops in a shared group → ratio 0.5."""
    from benchmarks.Runner import BenchmarkRunner

    runner = BenchmarkRunner()
    steps = [
        _make_step(0, "Robot1", 1),  # shared group
        _make_step(1, "Robot2", 1),  # shared group
        _make_step(2, "Robot1", 2),  # Robot1 only
        _make_step(3, "Robot2", 3),  # Robot2 only
    ]
    ratio, count = runner._compute_parallelism_ratio(steps)
    assert ratio == 0.5
    assert count == 2


# ---------------------------------------------------------------------------
# Runner integration (dry-run, no hardware)
# ---------------------------------------------------------------------------


def test_b16_dry_run_has_parallelism_metrics():
    """Dry-run must return BenchmarkResult with parallelism keys in per_op_stats."""
    from benchmarks.Runner import BenchmarkRunner
    from benchmarks.Config import DualRobotConfig

    runner = BenchmarkRunner()
    cfg = DualRobotConfig(dry_run=True)
    result = runner.run(16, cfg)

    assert result.benchmark_id == 16
    assert result.benchmark_name == "Parallel Independent Tasks"
    assert "_parallelism_ratio" in result.per_op_stats
    assert "_ops_in_parallel" in result.per_op_stats
    assert "_parallelism_success" in result.per_op_stats
    assert isinstance(result.per_op_stats["_parallelism_ratio"], float)
    assert isinstance(result.per_op_stats["_ops_in_parallel"], int)


def test_b16_success_false_when_no_parallel_groups():
    """Overall success must be False when parallelism_ratio == 0 even if all ops succeed."""
    from benchmarks.Runner import BenchmarkRunner
    from benchmarks.Config import DualRobotConfig

    runner = BenchmarkRunner()

    def fake_send(payload, robot_id, cfg, flags=None):
        ops = [
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1"}},
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot2"}},
        ]
        return {
            "success": True,
            "results": [
                {"index": 0, "operation": "move_to_coordinate", "success": True, "duration_ms": 50.0},
                {"index": 1, "operation": "move_to_coordinate", "success": True, "duration_ms": 50.0},
            ],
            "parsed_commands": ops,
            "total_duration_ms": 100.0,
        }

    runner._send = fake_send  # type: ignore[method-assign]
    cfg = DualRobotConfig(dry_run=False)
    result = runner.run(16, cfg)

    assert result.per_op_stats["_parallelism_ratio"] == 0.0
    assert result.success is False


def test_b16_success_true_when_all_parallel():
    """Overall success must be True when all ops are in shared groups and execution succeeds."""
    from benchmarks.Runner import BenchmarkRunner
    from benchmarks.Config import DualRobotConfig

    runner = BenchmarkRunner()

    def fake_send(payload, robot_id, cfg, flags=None):
        ops = [
            {"operation": "grasp_object", "params": {"robot_id": "Robot1"}, "parallel_group": 1},
            {"operation": "grasp_object", "params": {"robot_id": "Robot2"}, "parallel_group": 1},
        ]
        return {
            "success": True,
            "results": [
                {"index": 0, "operation": "grasp_object", "success": True, "duration_ms": 50.0},
                {"index": 1, "operation": "grasp_object", "success": True, "duration_ms": 50.0},
            ],
            "parsed_commands": ops,
            "total_duration_ms": 100.0,
        }

    runner._send = fake_send  # type: ignore[method-assign]
    cfg = DualRobotConfig(dry_run=False)
    result = runner.run(16, cfg)

    assert result.per_op_stats["_parallelism_ratio"] == 1.0
    assert result.per_op_stats["_parallelism_success"] is True
    assert result.success is True
