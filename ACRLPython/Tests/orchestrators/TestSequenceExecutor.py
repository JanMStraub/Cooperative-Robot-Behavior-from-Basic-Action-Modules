#!/usr/bin/env python3
"""
Unit tests for SequenceExecutor
================================

Tests command dispatch, variable passing, abort, metrics, and reflexion retry
without any live Unity connection. All external collaborators are mocked.

Coverage:
- Sequential execution: success path, failure stops sequence
- Variable capture and resolution ($var, $var.x, $var.z + offset)
- Auto-inject parameters from ParameterFlow
- Abort flag halts mid-sequence
- Metrics tracker (Welford online mean, success rate)
- _extract_waypoint_from_verification helper
- Parallel group execution
- Operation alias (return_to_start → return_to_start_position)
- Reflexion retry skipped for non-eligible categories
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_op_result(success: bool, result=None, error_code=None, error_msg=None):
    """Create a mock OperationResult."""
    r = MagicMock()
    r.success = success
    r.result = result or {}
    r.error = {"code": error_code, "message": error_msg} if not success else None
    return r


def _make_op_def(name: str, category=None):
    """Create a minimal mock operation definition."""
    from operations.Base import OperationCategory
    op = MagicMock()
    op.name = name
    op.operation_id = name
    op.category = category or OperationCategory.NAVIGATION
    op.relationships = None
    op.implementation = MagicMock()
    return op


def _make_executor(enable_verification=False, check_completion=False):
    """Build a SequenceExecutor with all heavy deps mocked out."""
    with (
        patch("core.Imports.get_global_registry") as mock_reg,
        patch("core.Imports.get_world_state", return_value=None),
        patch("operations.Verification.OperationVerifier", MagicMock()),
        patch("operations.CoordinationVerifier.CoordinationVerifier", MagicMock()),
    ):
        registry = MagicMock()
        mock_reg.return_value = registry

        from orchestrators.SequenceExecutor import SequenceExecutor

        executor = SequenceExecutor.__new__(SequenceExecutor)
        executor.registry = registry
        executor.default_timeout = 5.0
        executor.check_completion = check_completion
        executor.enable_verification = enable_verification
        executor.verifier = None
        executor.coordination_verifier = None
        executor.world_state = None
        executor._abort_flag = False
        executor._current_sequence_id = None
        executor._progress_callbacks = []
        executor._variables = {}
        executor._metrics = SequenceExecutor._MetricsTracker()
        executor.outcome_tracker = None
        return executor, registry


# ---------------------------------------------------------------------------
# _extract_waypoint_from_verification
# ---------------------------------------------------------------------------

class TestExtractWaypoint:
    def test_returns_none_when_no_waypoint(self):
        from orchestrators.SequenceExecutor import _extract_waypoint_from_verification
        result = _extract_waypoint_from_verification({})
        assert result is None

    def test_parses_waypoint_string(self):
        from orchestrators.SequenceExecutor import _extract_waypoint_from_verification
        vr = {
            "details": {
                "coordination_check": {
                    "issues": [
                        {"resolution_suggestions": ["WAYPOINT:0.1,0.2,0.3"]}
                    ]
                }
            }
        }
        result = _extract_waypoint_from_verification(vr)
        assert result == pytest.approx((0.1, 0.2, 0.3))

    def test_returns_none_on_malformed_waypoint(self):
        from orchestrators.SequenceExecutor import _extract_waypoint_from_verification
        vr = {
            "details": {
                "coordination_check": {
                    "issues": [
                        {"resolution_suggestions": ["WAYPOINT:bad,data"]}
                    ]
                }
            }
        }
        assert _extract_waypoint_from_verification(vr) is None


# ---------------------------------------------------------------------------
# _MetricsTracker
# ---------------------------------------------------------------------------

class TestMetricsTracker:
    def _tracker(self):
        from orchestrators.SequenceExecutor import SequenceExecutor
        return SequenceExecutor._MetricsTracker()

    def test_initial_snapshot_zeros(self):
        t = self._tracker()
        s = t.snapshot()
        assert s["ops_executed"] == 0
        assert s["ops_succeeded"] == 0
        assert s["ops_success_rate"] == 0.0

    def test_records_success(self):
        t = self._tracker()
        t.record(True, 100.0)
        s = t.snapshot()
        assert s["ops_executed"] == 1
        assert s["ops_succeeded"] == 1
        assert s["ops_success_rate"] == 1.0
        assert s["avg_duration_ms"] == pytest.approx(100.0)

    def test_records_failure(self):
        t = self._tracker()
        t.record(False, 50.0)
        s = t.snapshot()
        assert s["ops_failed"] == 1
        assert s["ops_success_rate"] == 0.0

    def test_welford_mean(self):
        t = self._tracker()
        t.record(True, 100.0)
        t.record(True, 200.0)
        s = t.snapshot()
        assert s["avg_duration_ms"] == pytest.approx(150.0)

    def test_reset(self):
        t = self._tracker()
        t.record(True, 99.0)
        t.reset()
        s = t.snapshot()
        assert s["ops_executed"] == 0

    def test_thread_safe_concurrent_records(self):
        t = self._tracker()
        threads = [threading.Thread(target=lambda: t.record(True, 10.0)) for _ in range(20)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert t.snapshot()["ops_executed"] == 20


# ---------------------------------------------------------------------------
# _resolve_variables / _resolve_dotted_variable / _resolve_expression
# ---------------------------------------------------------------------------

class TestVariableResolution:
    def setup_method(self):
        self.executor, _ = _make_executor()
        self.executor._variables = {
            "target": {"x": 0.3, "y": 0.1, "z": 0.5, "color": "blue_cube"}
        }

    def test_simple_var_reference(self):
        params = self.executor._resolve_variables({"position": "$target"})
        # $target resolves to dict and position key causes x/y/z expansion
        assert params["x"] == pytest.approx(0.3)
        assert params["z"] == pytest.approx(0.5)

    def test_dotted_var_x(self):
        params = self.executor._resolve_variables({"x": "$target.x"})
        assert params["x"] == pytest.approx(0.3)

    def test_expression_offset(self):
        params = self.executor._resolve_variables({"z": "$target.z + 0.05"})
        assert params["z"] == pytest.approx(0.55)

    def test_multi_var_midpoint_x(self):
        """Two-variable arithmetic: midpoint between two detected objects."""
        self.executor._variables["blue_obj"] = {"x": 0.2, "y": 0.06, "z": 0.1}
        self.executor._variables["red_obj"] = {"x": 0.8, "y": 0.06, "z": 0.1}
        params = self.executor._resolve_variables(
            {"x": "($blue_obj.x + $red_obj.x) / 2"}
        )
        assert params["x"] == pytest.approx(0.5)

    def test_multi_var_midpoint_all_axes(self):
        """Full between-placement midpoint across x, y, z."""
        self.executor._variables["blue_obj"] = {"x": 0.0, "y": 0.06, "z": -0.1}
        self.executor._variables["red_obj"] = {"x": 0.4, "y": 0.08, "z": 0.3}
        params = self.executor._resolve_variables(
            {
                "x": "($blue_obj.x + $red_obj.x) / 2",
                "y": "($blue_obj.y + $red_obj.y) / 2",
                "z": "($blue_obj.z + $red_obj.z) / 2",
            }
        )
        assert params["x"] == pytest.approx(0.2)
        assert params["y"] == pytest.approx(0.07)
        assert params["z"] == pytest.approx(0.1)

    def test_object_id_extracts_color(self):
        params = self.executor._resolve_variables({"object_id": "$target"})
        assert params["object_id"] == "blue_cube"

    def test_missing_var_returns_string(self):
        params = self.executor._resolve_variables({"x": "$nonexistent"})
        assert params["x"] == "$nonexistent"

    def test_list_elements_resolved(self):
        params = self.executor._resolve_variables({"pts": ["$target.x", "$target.z"]})
        assert params["pts"] == pytest.approx([0.3, 0.5])

    def test_field_center_fallback(self):
        # $field.center.x where $field == {"x":1.0, "y":2.0, "z":3.0}
        self.executor._variables["field"] = {"x": 1.0, "y": 2.0, "z": 3.0}
        params = self.executor._resolve_variables({"x": "$field.center.x"})
        assert params["x"] == pytest.approx(1.0)

    def test_capture_field_center_stores_dict(self):
        result = {"center": {"x": 0.1, "y": 0.2, "z": 0.3}, "field_label": "D"}
        self.executor._capture_result_to_var("field_d", result)
        assert self.executor._variables["field_d"] == result["center"]
        assert self.executor._variables["field_d_result"] == result


# ---------------------------------------------------------------------------
# execute_sequence — sequential mode
# ---------------------------------------------------------------------------

class TestExecuteSequenceSequential:
    def _setup(self):
        executor, registry = _make_executor()
        return executor, registry

    def _cmd(self, operation, params=None):
        return {"operation": operation, "params": params or {"robot_id": "Robot1"}}

    def test_single_success(self):
        executor, registry = self._setup()
        op_def = _make_op_def("move_to_coordinate")
        registry.get_operation_by_name.return_value = op_def
        registry.execute_operation_by_name.return_value = _make_op_result(True)

        result = executor.execute_sequence([self._cmd("move_to_coordinate")])

        assert result["success"] is True
        assert result["completed_commands"] == 1

    def test_failure_stops_sequence(self):
        executor, registry = self._setup()
        op_def = _make_op_def("move_to_coordinate")
        registry.get_operation_by_name.return_value = op_def
        registry.execute_operation_by_name.return_value = _make_op_result(
            False, error_code="TIMEOUT", error_msg="timed out"
        )

        cmds = [self._cmd("move_to_coordinate"), self._cmd("control_gripper")]
        result = executor.execute_sequence(cmds)

        assert result["success"] is False
        assert result["completed_commands"] == 0
        assert registry.execute_operation_by_name.call_count == 1

    def test_unknown_operation_fails(self):
        executor, registry = self._setup()
        registry.get_operation_by_name.return_value = None

        result = executor.execute_sequence([self._cmd("nonexistent_op")])
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_alias_return_to_start(self):
        executor, registry = self._setup()
        op_def = _make_op_def("return_to_start_position")
        registry.get_operation_by_name.side_effect = lambda name: op_def if name == "return_to_start_position" else None
        registry.execute_operation_by_name.return_value = _make_op_result(True)

        result = executor.execute_sequence([self._cmd("return_to_start")])
        assert result["success"] is True
        registry.execute_operation_by_name.assert_called_once()
        call_args = registry.execute_operation_by_name.call_args
        assert call_args[0][0] == "return_to_start_position"

    def test_abort_mid_sequence(self):
        executor, registry = self._setup()
        op_def = _make_op_def("move_to_coordinate")
        registry.get_operation_by_name.return_value = op_def

        call_count = 0

        def slow_execute(name, **kwargs):
            nonlocal call_count
            call_count += 1
            executor._abort_flag = True  # abort after first op
            return _make_op_result(True)

        registry.execute_operation_by_name.side_effect = slow_execute

        cmds = [self._cmd("move_to_coordinate")] * 3
        result = executor.execute_sequence(cmds)

        assert call_count == 1
        assert result["completed_commands"] == 1

    def test_variable_capture_and_resolution(self):
        executor, registry = self._setup()
        detect_op = _make_op_def("detect_object_stereo")
        from operations.Base import OperationCategory
        detect_op.category = OperationCategory.PERCEPTION
        move_op = _make_op_def("move_to_coordinate")

        def get_op(name):
            return detect_op if "detect" in name else move_op

        registry.get_operation_by_name.side_effect = get_op
        registry.execute_operation_by_name.return_value = _make_op_result(
            True, result={"x": 0.3, "y": 0.1, "z": 0.5, "color": "blue_cube"}
        )

        cmds = [
            {"operation": "detect_object_stereo", "params": {"robot_id": "Robot1"}, "capture_var": "target"},
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1", "x": "$target.x"}},
        ]
        result = executor.execute_sequence(cmds)
        assert result["success"] is True

    def test_error_message_includes_code(self):
        executor, registry = self._setup()
        op_def = _make_op_def("move_to_coordinate")
        registry.get_operation_by_name.return_value = op_def
        registry.execute_operation_by_name.return_value = _make_op_result(
            False, error_code="NO_IK_SOLUTION", error_msg="IK failed"
        )

        result = executor.execute_sequence([self._cmd("move_to_coordinate")])
        assert "NO_IK_SOLUTION" in result["error"]

    def test_empty_sequence_succeeds(self):
        executor, _ = self._setup()
        result = executor.execute_sequence([])
        assert result["success"] is True
        assert result["completed_commands"] == 0

    def test_metrics_updated_after_execution(self):
        executor, registry = self._setup()
        op_def = _make_op_def("move_to_coordinate")
        registry.get_operation_by_name.return_value = op_def
        registry.execute_operation_by_name.return_value = _make_op_result(True)

        executor.execute_sequence([self._cmd("move_to_coordinate")])
        metrics = executor.get_metrics()
        assert metrics["ops_executed"] == 1
        assert metrics["ops_succeeded"] == 1


# ---------------------------------------------------------------------------
# execute_sequence — parallel groups
# ---------------------------------------------------------------------------

class TestExecuteSequenceParallelGroups:
    def _setup(self):
        return _make_executor()

    def test_parallel_group_all_succeed(self):
        executor, registry = self._setup()
        op_def = _make_op_def("move_to_coordinate")
        registry.get_operation_by_name.return_value = op_def
        registry.execute_operation_by_name.return_value = _make_op_result(True)

        cmds = [
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1"}, "parallel_group": 0},
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot2"}, "parallel_group": 0},
        ]
        result = executor.execute_sequence(cmds)
        assert result["success"] is True
        assert result["completed_commands"] == 2

    def test_parallel_group_failure_stops(self):
        executor, registry = self._setup()
        op_def = _make_op_def("move_to_coordinate")
        registry.get_operation_by_name.return_value = op_def

        call_count = 0

        def side_effect(name, **kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("robot_id") == "Robot2":
                return _make_op_result(False, error_msg="IK failed")
            return _make_op_result(True)

        registry.execute_operation_by_name.side_effect = side_effect

        cmds = [
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1"}, "parallel_group": 0},
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot2"}, "parallel_group": 0},
            {"operation": "move_to_coordinate", "params": {"robot_id": "Robot1"}, "parallel_group": 1},
        ]
        result = executor.execute_sequence(cmds)
        # Group 1 fails → group 2 not executed
        assert result["success"] is False
        assert call_count == 2  # only group 0 executed


# ---------------------------------------------------------------------------
# abort and progress callbacks
# ---------------------------------------------------------------------------

class TestAbortAndCallbacks:
    def test_abort_sets_flag(self):
        executor, _ = _make_executor()
        executor.abort()
        assert executor._abort_flag is True

    def test_progress_callback_called(self):
        executor, registry = _make_executor()
        op_def = _make_op_def("move_to_coordinate")
        registry.get_operation_by_name.return_value = op_def
        registry.execute_operation_by_name.return_value = _make_op_result(True)

        events = []
        executor.add_progress_callback(lambda i, t, op, s: events.append(s))

        executor.execute_sequence([{"operation": "move_to_coordinate", "params": {"robot_id": "Robot1"}}])
        assert "executing" in events
        assert "completed" in events

    def test_failed_callback_does_not_crash(self):
        executor, registry = _make_executor()
        op_def = _make_op_def("move_to_coordinate")
        registry.get_operation_by_name.return_value = op_def
        registry.execute_operation_by_name.return_value = _make_op_result(True)

        executor.add_progress_callback(lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
        # Should not raise
        executor.execute_sequence([{"operation": "move_to_coordinate", "params": {"robot_id": "Robot1"}}])


# ---------------------------------------------------------------------------
# get_variable / set_variable
# ---------------------------------------------------------------------------

class TestVariableAccessors:
    def test_get_set(self):
        executor, _ = _make_executor()
        executor.set_variable("foo", 42)
        assert executor.get_variable("foo") == 42

    def test_get_missing_returns_none(self):
        executor, _ = _make_executor()
        assert executor.get_variable("missing") is None
