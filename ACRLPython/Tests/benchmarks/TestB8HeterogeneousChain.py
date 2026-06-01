#!/usr/bin/env python3
"""Tests for B8 Heterogeneous Chain benchmark."""
from __future__ import annotations


def _get_sub_tasks(task_count=2):
    from benchmarks.cases.B8HeterogeneousChain import get_sub_tasks
    from benchmarks.Config import BenchmarkConfig

    return get_sub_tasks(BenchmarkConfig(), task_count)


def test_three_sub_tasks_per_cycle():
    tasks = _get_sub_tasks(task_count=3)
    assert len(tasks) == 9  # 3 cycles × 3 phases


def test_phase_a_uses_robot1():
    tasks = _get_sub_tasks(task_count=1)
    robot_id, name, _ = tasks[0]
    assert robot_id == "Robot1"
    assert "phase_a" in name


def test_phase_b_uses_robot2():
    tasks = _get_sub_tasks(task_count=1)
    robot_id, name, _ = tasks[1]
    assert robot_id == "Robot2"
    assert "phase_b" in name


def test_phase_c_is_perception_only():
    tasks = _get_sub_tasks(task_count=1)
    _, _, text = tasks[2]
    text_lower = text.lower()
    assert "detect" in text_lower or "survey" in text_lower
    assert "grasp" not in text_lower
    assert "place" not in text_lower


def test_cycle_configs_rotate_across_cycles():
    tasks = _get_sub_tasks(task_count=4)
    phase_a_texts = [text for _, name, text in tasks if "phase_a" in name]
    assert len(set(phase_a_texts)) == 4  # unique (color, field) per cycle


def test_cycle_configs_wrap_after_four_cycles():
    tasks = _get_sub_tasks(task_count=5)
    phase_a = [text for _, name, text in tasks if "phase_a" in name]
    assert phase_a[0] == phase_a[4]


def test_colors_all_distinct():
    from benchmarks.cases.B8HeterogeneousChain import _CYCLE_CONFIGS
    colors = [c for c, _, _ in _CYCLE_CONFIGS]
    assert len(set(colors)) == len(_CYCLE_CONFIGS)


def test_phase_a_and_b_use_same_color_within_cycle():
    tasks = _get_sub_tasks(task_count=4)
    for i in range(4):
        _, _, text_a = tasks[i * 3]
        _, _, text_b = tasks[i * 3 + 1]
        color_a = text_a.split("the ")[1].split(" cube")[0]
        color_b = text_b.split("the ")[1].split(" cube")[0]
        assert color_a == color_b, f"Cycle {i}: phase_a='{color_a}' phase_b='{color_b}'"


def test_per_phase_success_in_chain_metrics():
    from benchmarks.Result import ChainMetrics

    m = ChainMetrics(
        total_tasks=6,
        completed_tasks=6,
        error_rate=0.0,
        recovery_count=0,
        per_error_code={},
        per_phase_success={"phase_a": 1.0, "phase_b": 0.5, "phase_c": 1.0},
    )
    assert m.per_phase_success["phase_b"] == 0.5


def test_runner_b8_registered():
    from benchmarks.Runner import _CASE_MODULES, _BENCHMARK_NAMES

    assert 8 in _CASE_MODULES
    assert "B8" in _CASE_MODULES[8]
    assert _BENCHMARK_NAMES[8] == "Heterogeneous Chain"
