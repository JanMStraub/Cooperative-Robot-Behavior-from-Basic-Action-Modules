"""
Test AutoRT Task Selector

Tests for task selection with exploration/exploitation balance.
"""

import pytest
from autort.TaskSelector import TaskSelector
from autort.DataModels import ProposedTask, Operation


@pytest.fixture
def task_selector():
    """Create TaskSelector instance"""
    return TaskSelector()


@pytest.fixture
def sample_tasks():
    """Sample task candidates"""
    return [
        ProposedTask(
            task_id="task_001",
            description="Pick red cube",
            operations=[
                Operation(type="wait", robot_id="Robot1", parameters={"seconds": 1})
            ],
            required_robots=["Robot1"],
            estimated_complexity=3,
            reasoning="test"
        ),
        ProposedTask(
            task_id="task_002",
            description="Pick blue cube",
            operations=[
                Operation(type="wait", robot_id="Robot1", parameters={"seconds": 1})
            ],
            required_robots=["Robot1"],
            estimated_complexity=3,
            reasoning="test"
        ),
        ProposedTask(
            task_id="task_003",
            description="Move to position",
            operations=[
                Operation(
                    type="move_to_coordinate",
                    robot_id="Robot1",
                    parameters={"x": 0.3, "y": 0.2, "z": 0.1}
                )
            ],
            required_robots=["Robot1"],
            estimated_complexity=2,
            reasoning="test"
        ),
    ]


# ============================================================================
# Selection Strategy Tests
# ============================================================================


def test_select_task_random(task_selector, sample_tasks):
    """Random strategy selects from candidates"""
    selected = task_selector.select_task(sample_tasks, strategy="random")
    assert selected in sample_tasks


def test_select_task_empty_candidates(task_selector):
    """Select from empty list returns None"""
    selected = task_selector.select_task([], strategy="balanced")
    assert selected is None


def test_select_task_explore_prefers_new(task_selector, sample_tasks):
    """Explore strategy prioritizes untried tasks"""
    # Record history for first task
    task_selector.update_history(sample_tasks[0], {"success": True})
    task_selector.update_history(sample_tasks[0], {"success": True})

    # Select with explore strategy
    selected = task_selector.select_task(sample_tasks, strategy="explore")

    # Should prefer tasks[1] or tasks[2] (never tried)
    assert selected in [sample_tasks[1], sample_tasks[2]]


def test_select_task_exploit_prefers_successful(task_selector, sample_tasks):
    """Exploit strategy prioritizes high-success tasks"""
    # Record successful history for first task
    task_selector.update_history(sample_tasks[0], {"success": True})
    task_selector.update_history(sample_tasks[0], {"success": True})

    # Record failed history for second task
    task_selector.update_history(sample_tasks[1], {"success": False})
    task_selector.update_history(sample_tasks[1], {"success": False})

    # Select with exploit strategy
    selected = task_selector.select_task(sample_tasks, strategy="exploit")

    # Should prefer first task (100% success rate)
    assert selected == sample_tasks[0]


def test_select_task_balanced_scoring(task_selector, sample_tasks):
    """Balanced strategy weighs success and novelty"""
    # Record mixed history for first task (high success, low novelty)
    task_selector.update_history(sample_tasks[0], {"success": True})
    task_selector.update_history(sample_tasks[0], {"success": True})
    task_selector.update_history(sample_tasks[0], {"success": True})

    # Record single failure for second task (low success, medium novelty)
    task_selector.update_history(sample_tasks[1], {"success": False})

    # Select with balanced strategy (third task is untried, high novelty)
    selected = task_selector.select_task(sample_tasks, strategy="balanced")

    # Should select either first (high success) or third (high novelty)
    # Balanced scoring: first = 1.0 * 0.6 + 0.25 * 0.4 = 0.7
    #                   third = 0.5 * 0.6 + 1.0 * 0.4 = 0.7
    assert selected in [sample_tasks[0], sample_tasks[2]]


# ============================================================================
# History Tracking Tests
# ============================================================================


def test_update_history(task_selector, sample_tasks):
    """Update history records task outcome"""
    task = sample_tasks[0]
    result = {"success": True, "duration": 5.2}

    task_selector.update_history(task, result)

    # Check history was recorded
    key = task_selector._task_key(task)
    assert len(task_selector.history[key]) == 1
    assert task_selector.history[key][0]["success"] is True


def test_update_history_multiple(task_selector, sample_tasks):
    """Update history accumulates multiple outcomes"""
    task = sample_tasks[0]

    task_selector.update_history(task, {"success": True})
    task_selector.update_history(task, {"success": False})
    task_selector.update_history(task, {"success": True})

    key = task_selector._task_key(task)
    assert len(task_selector.history[key]) == 3
    assert sum(1 for h in task_selector.history[key] if h["success"]) == 2


# ============================================================================
# Task Key Generation Tests
# ============================================================================


def test_task_key_same_operations(task_selector):
    """Tasks with same operation types have same key"""
    task1 = ProposedTask(
        task_id="task_001",
        description="test",
        operations=[
            Operation(type="wait", robot_id="Robot1", parameters={"seconds": 1})
        ],
        required_robots=["Robot1"],
        estimated_complexity=1,
        reasoning="test"
    )

    task2 = ProposedTask(
        task_id="task_002",
        description="test",
        operations=[
            Operation(type="wait", robot_id="Robot1", parameters={"seconds": 5})  # Different params
        ],
        required_robots=["Robot1"],
        estimated_complexity=1,
        reasoning="test"
    )

    # Should have same key (same operation sequence)
    assert task_selector._task_key(task1) == task_selector._task_key(task2)


def test_task_key_different_operations(task_selector):
    """Tasks with different operation types have different keys"""
    task1 = ProposedTask(
        task_id="task_001",
        description="test",
        operations=[
            Operation(type="wait", robot_id="Robot1", parameters={"seconds": 1})
        ],
        required_robots=["Robot1"],
        estimated_complexity=1,
        reasoning="test"
    )

    task2 = ProposedTask(
        task_id="task_002",
        description="test",
        operations=[
            Operation(
                type="move_to_coordinate",
                robot_id="Robot1",
                parameters={"x": 0.3, "y": 0.2, "z": 0.1}
            )
        ],
        required_robots=["Robot1"],
        estimated_complexity=1,
        reasoning="test"
    )

    # Should have different keys (different operations)
    assert task_selector._task_key(task1) != task_selector._task_key(task2)


def test_task_key_operation_sequence_matters(task_selector):
    """Task key depends on operation sequence order"""
    task1 = ProposedTask(
        task_id="task_001",
        description="test",
        operations=[
            Operation(type="wait", robot_id="Robot1", parameters={"seconds": 1}),
            Operation(
                type="move_to_coordinate",
                robot_id="Robot1",
                parameters={"x": 0.3, "y": 0.2, "z": 0.1}
            ),
        ],
        required_robots=["Robot1"],
        estimated_complexity=2,
        reasoning="test"
    )

    task2 = ProposedTask(
        task_id="task_002",
        description="test",
        operations=[
            Operation(
                type="move_to_coordinate",
                robot_id="Robot1",
                parameters={"x": 0.3, "y": 0.2, "z": 0.1}
            ),
            Operation(type="wait", robot_id="Robot1", parameters={"seconds": 1}),
        ],
        required_robots=["Robot1"],
        estimated_complexity=2,
        reasoning="test"
    )

    # Should have different keys (different sequence)
    assert task_selector._task_key(task1) != task_selector._task_key(task2)


# ============================================================================
# Exploration Tests
# ============================================================================


def test_explore_prioritizes_untried(task_selector, sample_tasks):
    """Explore strategy picks task with fewest attempts"""
    # Try first task multiple times
    for _ in range(5):
        task_selector.update_history(sample_tasks[0], {"success": True})

    # Try second task once
    task_selector.update_history(sample_tasks[1], {"success": True})

    # Third task never tried

    selected = task_selector.select_task(sample_tasks, strategy="explore")

    # Should select third task (untried)
    assert selected == sample_tasks[2]


# ============================================================================
# Exploitation Tests
# ============================================================================


def test_exploit_handles_unknown_tasks(task_selector, sample_tasks):
    """Exploit strategy assigns neutral score to unknown tasks"""
    # No history recorded

    selected = task_selector.select_task(sample_tasks, strategy="exploit")

    # Should still select something (all have neutral score 0.5)
    assert selected in sample_tasks


def test_exploit_prefers_consistency(task_selector, sample_tasks):
    """Exploit strategy prefers consistently successful task"""
    # First task: 100% success (3/3)
    for _ in range(3):
        task_selector.update_history(sample_tasks[0], {"success": True})

    # Second task: 50% success (1/2)
    task_selector.update_history(sample_tasks[1], {"success": True})
    task_selector.update_history(sample_tasks[1], {"success": False})

    selected = task_selector.select_task(sample_tasks, strategy="exploit")

    # Should prefer first task (higher success rate)
    assert selected == sample_tasks[0]


# ============================================================================
# Balanced Strategy Tests
# ============================================================================


def test_balanced_novelty_decay(task_selector, sample_tasks):
    """Balanced strategy novelty decays with attempts"""
    task = sample_tasks[0]

    # Record increasing history
    for i in range(10):
        task_selector.update_history(task, {"success": True})

        # Check novelty decreases
        key = task_selector._task_key(task)
        outcomes = task_selector.history[key]
        novelty = 1.0 / (1.0 + len(outcomes))

        # More attempts → lower novelty
        assert novelty <= 1.0 / (i + 2)


def test_balanced_weights_success_and_novelty(task_selector, sample_tasks):
    """Balanced strategy weights success (60%) and novelty (40%)"""
    # First task: high success, many attempts
    for _ in range(5):
        task_selector.update_history(sample_tasks[0], {"success": True})

    # Second task: untried (high novelty)

    # Calculate expected scores
    # task1: success_rate=1.0, novelty=1/(1+5)=0.167
    #        score = 1.0 * 0.6 + 0.167 * 0.4 = 0.667
    # task2: neutral_success=0.5, novelty=1.0
    #        score = 0.5 * 0.6 + 1.0 * 0.4 = 0.7

    selected = task_selector.select_task(sample_tasks[:2], strategy="balanced")

    # Second task should win (higher balanced score)
    assert selected == sample_tasks[1]
