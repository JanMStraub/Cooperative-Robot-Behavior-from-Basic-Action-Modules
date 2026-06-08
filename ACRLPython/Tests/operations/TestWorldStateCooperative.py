import pytest
from operations.WorldState import WorldState, RobotState


class TestRobotStateIntentFields:

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
        assert state is not None
        assert state.moving_toward_object == "red_cube"

    def test_update_robot_state_clears_intent_on_none(self):
        ws = WorldState()
        ws.update_robot_state("Robot1", {"moving_toward_object": "red_cube"})
        ws.update_robot_state("Robot1", {"moving_toward_object": None})
        state = ws.get_robot_state("Robot1")
        assert state is not None
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
        alloc2 = ws._workspace_allocations[region]
        assert alloc2 is not None
        assert alloc2.robot_id == "Robot2"

    def test_same_urgency_does_not_preempt(self):
        region = self._get_region()
        ws = WorldState()
        ws.allocate_workspace(region, "Robot1", urgency=3, estimated_duration=30.0)
        ok = ws.allocate_workspace(region, "Robot2", urgency=3, estimated_duration=5.0)
        assert ok is False
        alloc1 = ws._workspace_allocations[region]
        assert alloc1 is not None
        assert alloc1.robot_id == "Robot1"

    def test_get_free_workspace_regions(self):
        ws = WorldState()
        free = ws.get_free_workspace_regions()
        assert isinstance(free, list)


class TestTaskOutcomeBroadcasting:

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


class TestSupplementObjectFromUnity:

    def _get(self, ws, object_id):
        return ws._objects.get(object_id)

    def test_creates_new_object_when_not_in_worldstate(self):
        ws = WorldState()
        ws.supplement_object_from_unity(
            "blue_cube",
            (2.0, 0.0, 0.0),
            color="blue",
            object_type="cube",
            confidence=1.0,
            dimensions=(0.05, 0.05, 0.05),
            rotation=None,
        )
        obj = self._get(ws, "blue_cube")
        assert obj is not None
        assert obj.position == (2.0, 0.0, 0.0)
        assert obj.source == "unity"

    def test_does_not_overwrite_vision_position(self):
        ws = WorldState()
        ws.update_object_position(
            "red_cube", (1.0, 0.0, 0.0), color="red", source="vision"
        )
        ws.supplement_object_from_unity(
            "red_cube",
            (9.0, 9.0, 9.0),
            color="red",
            object_type="cube",
            confidence=1.0,
            dimensions=(0.05, 0.05, 0.05),
            rotation=None,
        )
        obj = self._get(ws, "red_cube")
        assert obj is not None
        assert obj.position == (1.0, 0.0, 0.0)  # type: ignore[union-attr]
        assert obj.source == "vision"  # type: ignore[union-attr]

    def test_fills_missing_dimensions_for_vision_object(self):
        ws = WorldState()
        ws.update_object_position(
            "green_cube", (0.5, 0.0, 0.0), color="green", dimensions=None
        )
        ws.supplement_object_from_unity(
            "green_cube",
            (9.0, 9.0, 9.0),
            color="green",
            object_type="cube",
            confidence=1.0,
            dimensions=(0.05, 0.05, 0.05),
            rotation=None,
        )
        obj = self._get(ws, "green_cube")
        assert obj is not None
        assert obj.dimensions == (0.05, 0.05, 0.05)  # type: ignore[union-attr]
        assert obj.position == (0.5, 0.0, 0.0)  # type: ignore[union-attr]

    def test_does_not_overwrite_existing_dimensions(self):
        ws = WorldState()
        ws.update_object_position(
            "red_cube", (1.0, 0.0, 0.0), dimensions=(0.03, 0.03, 0.03)
        )
        ws.supplement_object_from_unity(
            "red_cube",
            (9.0, 9.0, 9.0),
            dimensions=(0.99, 0.99, 0.99),
        )
        obj = self._get(ws, "red_cube")
        assert obj is not None
        assert obj.dimensions == (0.03, 0.03, 0.03)  # type: ignore[union-attr]

    def test_source_field_on_objectstate(self):
        from operations.WorldState import ObjectState

        obj = ObjectState(object_id="x", position=(0, 0, 0), source="vision")
        assert obj.source == "vision"
        obj2 = ObjectState(object_id="y", position=(0, 0, 0))
        assert obj2.source == "unity"
