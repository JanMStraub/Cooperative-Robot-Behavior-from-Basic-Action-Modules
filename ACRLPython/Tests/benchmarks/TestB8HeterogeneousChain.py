from __future__ import annotations


def _get_sub_tasks(task_count=2):
    from benchmarks.cases.B8HeterogeneousChain import get_sub_tasks
    from benchmarks.Config import BenchmarkConfig

    return get_sub_tasks(BenchmarkConfig(), task_count)


def test_four_sub_tasks_per_cycle():
    tasks = _get_sub_tasks(task_count=3)
    assert len(tasks) == 12  # 3 cycles × 4 phases


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


def test_phase_c_is_grasp():
    tasks = _get_sub_tasks(task_count=1)
    _, _, text = tasks[2]
    text_lower = text.lower()
    assert "grasp" in text_lower
    assert "robot1" in text_lower


def test_phase_d_is_perception_only():
    tasks = _get_sub_tasks(task_count=1)
    _, _, text = tasks[3]
    text_lower = text.lower()
    assert "detect" in text_lower or "survey" in text_lower
    assert "grasp" not in text_lower
    assert "place" not in text_lower


def test_phase_a_text_consistent_across_cycles():
    tasks = _get_sub_tasks(task_count=4)
    phase_a_texts = [text for _, name, text in tasks if "phase_a" in name]
    # All cycles use the same static phase text
    assert len(set(phase_a_texts)) == 1


def test_phase_a_and_b_use_same_color_within_cycle():
    tasks = _get_sub_tasks(task_count=4)
    for i in range(4):
        _, _, text_a = tasks[i * 4]
        _, _, text_b = tasks[i * 4 + 1]
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
