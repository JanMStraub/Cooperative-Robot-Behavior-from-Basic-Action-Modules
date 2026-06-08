import pytest
from autort.TaskSelector import TaskSelector
from autort.DataModels import ProposedTask, Operation


@pytest.fixture
def task_selector():
    return TaskSelector()


@pytest.fixture
def sample_tasks():
    return [
        ProposedTask(
            task_id="task_001",
            description="Pick red cube",
            operations=[
                Operation(type="wait", robot_id="Robot1", parameters={"seconds": 1})
            ],
            required_robots=["Robot1"],
            estimated_complexity=3,
            reasoning="test",
        ),
        ProposedTask(
            task_id="task_002",
            description="Pick blue cube",
            operations=[
                Operation(type="wait", robot_id="Robot1", parameters={"seconds": 1})
            ],
            required_robots=["Robot1"],
            estimated_complexity=3,
            reasoning="test",
        ),
        ProposedTask(
            task_id="task_003",
            description="Move to position",
            operations=[
                Operation(
                    type="move_to_coordinate",
                    robot_id="Robot1",
                    parameters={"x": 0.3, "y": 0.2, "z": 0.1},
                )
            ],
            required_robots=["Robot1"],
            estimated_complexity=2,
            reasoning="test",
        ),
    ]


def test_select_task_random(task_selector, sample_tasks):
    selected = task_selector.select_task(sample_tasks, strategy="random")
    assert selected in sample_tasks


def test_select_task_empty_candidates(task_selector):
    selected = task_selector.select_task([], strategy="balanced")
    assert selected is None


def test_select_task_explore_prefers_new(task_selector, sample_tasks):
    task_selector.update_history(sample_tasks[0], {"success": True})
    task_selector.update_history(sample_tasks[0], {"success": True})

    selected = task_selector.select_task(sample_tasks, strategy="explore")
    assert selected in [sample_tasks[1], sample_tasks[2]]


def test_select_task_exploit_prefers_successful(task_selector, sample_tasks):
    task_selector.update_history(sample_tasks[0], {"success": True})
    task_selector.update_history(sample_tasks[0], {"success": True})
    task_selector.update_history(sample_tasks[1], {"success": False})
    task_selector.update_history(sample_tasks[1], {"success": False})

    selected = task_selector.select_task(sample_tasks, strategy="exploit")
    assert selected == sample_tasks[0]


def test_select_task_balanced_scoring(task_selector, sample_tasks):
    task_selector.update_history(sample_tasks[0], {"success": True})
    task_selector.update_history(sample_tasks[0], {"success": True})
    task_selector.update_history(sample_tasks[0], {"success": True})
    task_selector.update_history(sample_tasks[1], {"success": False})

    selected = task_selector.select_task(sample_tasks, strategy="balanced")
    # Balanced scoring: first = 1.0 * 0.6 + 0.25 * 0.4 = 0.7
    #                   third = 0.5 * 0.6 + 1.0 * 0.4 = 0.7
    assert selected in [sample_tasks[0], sample_tasks[2]]


def test_update_history(task_selector, sample_tasks):
    task = sample_tasks[0]
    task_selector.update_history(task, {"success": True, "duration": 5.2})

    key = task_selector._task_key(task)
    assert len(task_selector.history[key]) == 1
    assert task_selector.history[key][0]["success"] is True


def test_update_history_multiple(task_selector, sample_tasks):
    task = sample_tasks[0]

    task_selector.update_history(task, {"success": True})
    task_selector.update_history(task, {"success": False})
    task_selector.update_history(task, {"success": True})

    key = task_selector._task_key(task)
    assert len(task_selector.history[key]) == 3
    assert sum(1 for h in task_selector.history[key] if h["success"]) == 2


def test_task_key_same_operations(task_selector):
    task1 = ProposedTask(
        task_id="task_001",
        description="test",
        operations=[
            Operation(type="wait", robot_id="Robot1", parameters={"seconds": 1})
        ],
        required_robots=["Robot1"],
        estimated_complexity=1,
        reasoning="test",
    )
    task2 = ProposedTask(
        task_id="task_002",
        description="test",
        operations=[
            Operation(type="wait", robot_id="Robot1", parameters={"seconds": 5})
        ],
        required_robots=["Robot1"],
        estimated_complexity=1,
        reasoning="test",
    )
    assert task_selector._task_key(task1) == task_selector._task_key(task2)


def test_task_key_different_operations(task_selector):
    task1 = ProposedTask(
        task_id="task_001",
        description="test",
        operations=[
            Operation(type="wait", robot_id="Robot1", parameters={"seconds": 1})
        ],
        required_robots=["Robot1"],
        estimated_complexity=1,
        reasoning="test",
    )
    task2 = ProposedTask(
        task_id="task_002",
        description="test",
        operations=[
            Operation(
                type="move_to_coordinate",
                robot_id="Robot1",
                parameters={"x": 0.3, "y": 0.2, "z": 0.1},
            )
        ],
        required_robots=["Robot1"],
        estimated_complexity=1,
        reasoning="test",
    )
    assert task_selector._task_key(task1) != task_selector._task_key(task2)


def test_task_key_operation_sequence_matters(task_selector):
    task1 = ProposedTask(
        task_id="task_001",
        description="test",
        operations=[
            Operation(type="wait", robot_id="Robot1", parameters={"seconds": 1}),
            Operation(
                type="move_to_coordinate",
                robot_id="Robot1",
                parameters={"x": 0.3, "y": 0.2, "z": 0.1},
            ),
        ],
        required_robots=["Robot1"],
        estimated_complexity=2,
        reasoning="test",
    )
    task2 = ProposedTask(
        task_id="task_002",
        description="test",
        operations=[
            Operation(
                type="move_to_coordinate",
                robot_id="Robot1",
                parameters={"x": 0.3, "y": 0.2, "z": 0.1},
            ),
            Operation(type="wait", robot_id="Robot1", parameters={"seconds": 1}),
        ],
        required_robots=["Robot1"],
        estimated_complexity=2,
        reasoning="test",
    )
    assert task_selector._task_key(task1) != task_selector._task_key(task2)


def test_explore_prioritizes_untried(task_selector, sample_tasks):
    for _ in range(5):
        task_selector.update_history(sample_tasks[0], {"success": True})
    task_selector.update_history(sample_tasks[1], {"success": True})

    selected = task_selector.select_task(sample_tasks, strategy="explore")
    assert selected == sample_tasks[2]


def test_exploit_handles_unknown_tasks(task_selector, sample_tasks):
    selected = task_selector.select_task(sample_tasks, strategy="exploit")
    assert selected in sample_tasks


def test_exploit_prefers_consistency(task_selector, sample_tasks):
    for _ in range(3):
        task_selector.update_history(sample_tasks[0], {"success": True})
    task_selector.update_history(sample_tasks[1], {"success": True})
    task_selector.update_history(sample_tasks[1], {"success": False})

    selected = task_selector.select_task(sample_tasks, strategy="exploit")
    assert selected == sample_tasks[0]


def test_balanced_novelty_decay(task_selector, sample_tasks):
    task = sample_tasks[0]

    for i in range(10):
        task_selector.update_history(task, {"success": True})

        key = task_selector._task_key(task)
        outcomes = task_selector.history[key]
        novelty = 1.0 / (1.0 + len(outcomes))
        assert novelty <= 1.0 / (i + 2)


def test_balanced_weights_success_and_novelty(task_selector, sample_tasks):
    # task[0] (wait) and task[2] (move_to_coordinate) have different operation keys
    task_practiced = sample_tasks[0]
    task_novel = sample_tasks[2]

    for _ in range(5):
        task_selector.update_history(task_practiced, {"success": True})

    # task_practiced: success_rate=1.0, novelty=1/(1+5)=0.167 → score = 1.0*0.6 + 0.167*0.4 = 0.667
    # task_novel: neutral_success=0.5, novelty=1.0 → score = 0.5*0.6 + 1.0*0.4 = 0.7

    selected = task_selector.select_task(
        [task_practiced, task_novel], strategy="balanced"
    )
    assert selected == task_novel
