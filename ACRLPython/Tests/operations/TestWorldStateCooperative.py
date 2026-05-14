"""Tests for cooperative WorldState extensions."""
import pytest
from operations.WorldState import WorldState, RobotState


class TestRobotStateIntentFields:
    """RobotState carries intent fields for joint-attention coordination."""

    def test_robot_state_has_moving_toward_object_field(self):
        state = RobotState(robot_id="Robot1")
        assert hasattr(state, "moving_toward_object")
        assert state.moving_toward_object is None

    def test_robot_state_has_workspace_intent_field(self):
        state = RobotState(robot_id="Robot1")
        assert hasattr(state, "workspace_intent")
        assert state.workspace_intent is None

    def test_update_robot_state_sets_moving_toward_object(self):
        ws = WorldState()
        ws.update_robot_state("Robot1", {"moving_toward_object": "red_cube"})
        state = ws.get_robot_state("Robot1")
        assert state.moving_toward_object == "red_cube"

    def test_update_robot_state_clears_intent_on_none(self):
        ws = WorldState()
        ws.update_robot_state("Robot1", {"moving_toward_object": "red_cube"})
        ws.update_robot_state("Robot1", {"moving_toward_object": None})
        state = ws.get_robot_state("Robot1")
        assert state.moving_toward_object is None

    def test_get_robot_intents_returns_all_non_none(self):
        ws = WorldState()
        ws.update_robot_state("Robot1", {"moving_toward_object": "red_cube"})
        ws.update_robot_state("Robot2", {"moving_toward_object": None})
        intents = ws.get_robot_intents()
        assert "Robot1" in intents
        assert intents["Robot1"] == "red_cube"
        assert "Robot2" not in intents


class TestIntentAwareWorkspaceAllocation:
    """Workspace allocation respects urgency and estimated duration."""

    def _get_region(self):
        from config.Robot import WORKSPACE_REGIONS
        ws = WorldState()
        # Release all allocations to ensure clean state
        for region in list(WORKSPACE_REGIONS.keys()):
            ws._workspace_allocations[region] = None
        return next(iter(WORKSPACE_REGIONS))

    def test_allocate_workspace_accepts_urgency_and_duration(self):
        region = self._get_region()
        ws = WorldState()
        ok = ws.allocate_workspace(region, "Robot1", urgency=1, estimated_duration=10.0)
        assert ok is True

    def test_high_urgency_preempts_low_urgency_long_holder(self):
        region = self._get_region()
        ws = WorldState()
        ws.allocate_workspace(region, "Robot1", urgency=1, estimated_duration=60.0)
        ok = ws.allocate_workspace(region, "Robot2", urgency=5, estimated_duration=3.0)
        assert ok is True
        assert ws._workspace_allocations[region].robot_id == "Robot2"

    def test_same_urgency_does_not_preempt(self):
        region = self._get_region()
        ws = WorldState()
        ws.allocate_workspace(region, "Robot1", urgency=3, estimated_duration=30.0)
        ok = ws.allocate_workspace(region, "Robot2", urgency=3, estimated_duration=5.0)
        assert ok is False
        assert ws._workspace_allocations[region].robot_id == "Robot1"

    def test_get_free_workspace_regions(self):
        ws = WorldState()
        free = ws.get_free_workspace_regions()
        assert isinstance(free, list)


class TestTaskOutcomeBroadcasting:
    """Completed sequences publish outcomes to WorldState for peer robots."""

    def test_broadcast_task_outcome_stores_result(self):
        ws = WorldState()
        ws._task_outcomes.clear()
        ws.broadcast_task_outcome(
            robot_id="Robot1",
            task_id="seq_001",
            success=True,
            duration_ms=1200.0,
            final_object_states={"red_cube": {"position": (0.1, 0.05, 0.2)}},
        )
        outcomes = ws.get_task_outcomes()
        assert len(outcomes) >= 1
        latest = outcomes[-1]
        assert latest["robot_id"] == "Robot1"
        assert latest["task_id"] == "seq_001"
        assert latest["success"] is True
        assert latest["duration_ms"] == 1200.0
        assert "red_cube" in latest["final_object_states"]

    def test_get_task_outcomes_returns_last_n(self):
        ws = WorldState()
        ws._task_outcomes.clear()
        for i in range(15):
            ws.broadcast_task_outcome(
                robot_id="Robot1",
                task_id=f"seq_{i:03d}",
                success=True,
                duration_ms=500.0,
                final_object_states={},
            )
        outcomes = ws.get_task_outcomes(last_n=5)
        assert len(outcomes) == 5
        assert outcomes[-1]["task_id"] == "seq_014"

    def test_get_task_outcomes_filtered_by_robot(self):
        ws = WorldState()
        ws._task_outcomes.clear()
        ws.broadcast_task_outcome("Robot1", "seq_r1", True, 400.0, {})
        ws.broadcast_task_outcome("Robot2", "seq_r2", False, 600.0, {})
        r1_outcomes = ws.get_task_outcomes(robot_id="Robot1")
        assert all(o["robot_id"] == "Robot1" for o in r1_outcomes)

    def test_get_all_robots_returns_list(self):
        ws = WorldState()
        ws.update_robot_state("Robot1", {"is_initialized": True})
        robots = ws.get_all_robots()
        assert isinstance(robots, list)
        ids = [r.robot_id for r in robots]
        assert "Robot1" in ids
