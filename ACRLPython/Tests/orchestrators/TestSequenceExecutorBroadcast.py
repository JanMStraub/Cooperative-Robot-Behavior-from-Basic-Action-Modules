from unittest.mock import MagicMock, patch
import pytest


def _make_op_result(success: bool):
    r = MagicMock()
    r.success = success
    r.result = {}
    r.error = None if success else {"code": "ERR", "message": "fail"}
    return r


def _make_executor(mock_ws=None):
    """Build a SequenceExecutor with heavy deps mocked, injecting a mock WorldState."""
    ws = mock_ws or MagicMock()
    with (
        patch("core.Imports.get_global_registry") as mock_reg,
        patch("core.Imports.get_world_state", return_value=ws),
        patch("operations.Verification.OperationVerifier", MagicMock()),
        patch("operations.CoordinationVerifier.CoordinationVerifier", MagicMock()),
    ):
        registry = MagicMock()
        mock_reg.return_value = registry

        from orchestrators.SequenceExecutor import SequenceExecutor

        executor = SequenceExecutor.__new__(SequenceExecutor)
        executor.registry = registry
        executor.default_timeout = 5.0
        executor.check_completion = False
        executor.enable_verification = False
        executor.verifier = None
        executor.coordination_verifier = None
        executor.world_state = None
        executor._abort_flag = False
        executor._current_sequence_id = None
        executor._progress_callbacks = []
        executor._variables = {}
        executor._metrics = SequenceExecutor._MetricsTracker()
        executor.outcome_tracker = None
        return executor, registry, ws


class TestSequenceExecutorBroadcastIntegration:
    """SequenceExecutor calls WorldState.broadcast_task_outcome after sequence completes."""

    def test_broadcast_called_on_successful_sequence(self):
        mock_ws = MagicMock()
        executor, registry, _ = _make_executor(mock_ws)

        from operations.Base import OperationCategory

        op_def = MagicMock()
        op_def.name = "wait"
        op_def.category = OperationCategory.NAVIGATION
        op_def.relationships = None
        registry.get_operation_by_name.return_value = op_def
        registry.execute_operation_by_name.return_value = _make_op_result(True)

        with patch(
            "orchestrators.SequenceExecutor.get_world_state", return_value=mock_ws
        ):
            executor.execute_sequence(
                [
                    {
                        "operation": "wait",
                        "params": {"robot_id": "Robot1", "duration_ms": 1},
                    }
                ],
                sequence_id="test_seq_001",
            )

        mock_ws.broadcast_task_outcome.assert_called_once()
        kwargs = mock_ws.broadcast_task_outcome.call_args[1]
        assert kwargs["robot_id"] == "Robot1"
        assert kwargs["task_id"] == "test_seq_001"
        assert kwargs["success"] is True

    def test_broadcast_called_on_failed_sequence(self):
        mock_ws = MagicMock()
        executor, registry, _ = _make_executor(mock_ws)

        from operations.Base import OperationCategory

        op_def = MagicMock()
        op_def.name = "wait"
        op_def.category = OperationCategory.NAVIGATION
        op_def.relationships = None
        registry.get_operation_by_name.return_value = op_def
        registry.execute_operation_by_name.return_value = _make_op_result(False)

        with patch(
            "orchestrators.SequenceExecutor.get_world_state", return_value=mock_ws
        ):
            executor.execute_sequence(
                [
                    {
                        "operation": "wait",
                        "params": {"robot_id": "Robot2", "duration_ms": 1},
                    }
                ],
                sequence_id="test_seq_002",
            )

        mock_ws.broadcast_task_outcome.assert_called_once()
        kwargs = mock_ws.broadcast_task_outcome.call_args[1]
        assert kwargs["success"] is False
        assert kwargs["robot_id"] == "Robot2"

    def test_broadcast_failure_does_not_break_sequence(self):
        """Broadcasting errors must not propagate to the caller."""
        mock_ws = MagicMock()
        mock_ws.broadcast_task_outcome.side_effect = RuntimeError("ws down")
        executor, registry, _ = _make_executor(mock_ws)

        from operations.Base import OperationCategory

        op_def = MagicMock()
        op_def.name = "wait"
        op_def.category = OperationCategory.NAVIGATION
        op_def.relationships = None
        registry.get_operation_by_name.return_value = op_def
        registry.execute_operation_by_name.return_value = _make_op_result(True)

        with patch(
            "orchestrators.SequenceExecutor.get_world_state", return_value=mock_ws
        ):
            result = executor.execute_sequence(
                [
                    {
                        "operation": "wait",
                        "params": {"robot_id": "Robot1", "duration_ms": 1},
                    }
                ],
            )

        assert result["success"] is True
