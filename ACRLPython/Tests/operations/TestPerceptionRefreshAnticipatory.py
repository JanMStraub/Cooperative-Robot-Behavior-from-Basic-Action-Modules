import threading
import pytest
from unittest.mock import MagicMock, patch


def _make_loop():
    """Construct a PerceptionRefreshLoop with mocked world_state."""
    from operations.PerceptionRefresh import PerceptionRefreshLoop

    mock_ws = MagicMock()
    loop = PerceptionRefreshLoop.__new__(PerceptionRefreshLoop)
    loop._world_state = mock_ws
    loop._stop_event = threading.Event()
    loop._refresh_interval = 999
    loop._stale_threshold = 0.4
    loop._anticipatory_queue = []
    loop._anticipatory_lock = threading.Lock()
    loop._thread = None
    return loop, mock_ws


class TestAnticipatoryRefresh:
    """trigger_anticipatory_refresh queues object IDs for immediate re-detection."""

    def test_trigger_queues_object_ids(self):
        loop, _ = _make_loop()
        loop.trigger_anticipatory_refresh(["red_cube", "blue_box"])
        with loop._anticipatory_lock:
            assert "red_cube" in loop._anticipatory_queue
            assert "blue_box" in loop._anticipatory_queue

    def test_trigger_deduplicates(self):
        loop, _ = _make_loop()
        loop.trigger_anticipatory_refresh(["red_cube"])
        loop.trigger_anticipatory_refresh(["red_cube", "blue_box"])
        with loop._anticipatory_lock:
            assert loop._anticipatory_queue.count("red_cube") == 1

    def test_sweep_drains_anticipatory_queue(self):
        loop, mock_ws = _make_loop()
        loop.trigger_anticipatory_refresh(["red_cube"])

        refreshed = []

        def fake_refresh_by_id(object_id):
            refreshed.append(object_id)
            return True

        loop._refresh_object_by_id = fake_refresh_by_id
        loop._collect_stale_colors = MagicMock(return_value=[])
        loop._sweep()

        assert "red_cube" in refreshed
        with loop._anticipatory_lock:
            assert len(loop._anticipatory_queue) == 0

    def test_queue_empty_after_sweep(self):
        loop, _ = _make_loop()
        loop.trigger_anticipatory_refresh(["green_cylinder"])
        loop._refresh_object_by_id = MagicMock(return_value=True)
        loop._collect_stale_colors = MagicMock(return_value=[])
        loop._sweep()
        with loop._anticipatory_lock:
            assert loop._anticipatory_queue == []

    def test_refresh_object_by_id_looks_up_color(self):
        loop, mock_ws = _make_loop()
        obj = MagicMock()
        obj.color = "red"
        mock_ws.get_object.return_value = obj
        loop._refresh_stereo = MagicMock(return_value=True)

        result = loop._refresh_object_by_id("red_cube")

        mock_ws.get_object.assert_called_once_with("red_cube")
        loop._refresh_stereo.assert_called_once_with("red")
        assert result is True

    def test_refresh_object_by_id_returns_false_when_not_found(self):
        loop, mock_ws = _make_loop()
        mock_ws.get_object.return_value = None

        result = loop._refresh_object_by_id("nonexistent")
        assert result is False


class TestIntentTriggersRefresh:
    """Setting moving_toward_object in WorldState causes anticipatory refresh."""

    def test_trigger_from_external_caller(self):
        """trigger_anticipatory_refresh can be called externally with any object list."""
        from operations.PerceptionRefresh import PerceptionRefreshLoop

        loop, _ = _make_loop()
        loop.trigger_anticipatory_refresh(["green_cylinder"])
        with loop._anticipatory_lock:
            assert "green_cylinder" in loop._anticipatory_queue


class TestGraspObjectSetsIntent:
    """grasp_object sets moving_toward_object on WorldState before executing."""

    def test_grasp_sets_moving_toward_object(self):
        mock_ws = MagicMock()
        import core.Imports as _ci

        with (
            patch.object(_ci, "get_world_state", return_value=mock_ws),
            patch.object(_ci, "get_perception_refresh_daemon", return_value=None),
            patch("config.ROS.ROS_ENABLED", False),
            patch("config.Servers.VGN_ENABLED", False),
        ):
            from operations.GraspOperations import grasp_object

            try:
                grasp_object(robot_id="Robot1", object_id="red_cube", request_id=0)
            except Exception:
                pass

        update_calls = mock_ws.update_robot_state.call_args_list
        intent_set = [
            c
            for c in update_calls
            if isinstance(c[0][1], dict)
            and c[0][1].get("moving_toward_object") == "red_cube"
        ]
        assert len(intent_set) >= 1

    def test_grasp_clears_intent_after_completion(self):
        mock_ws = MagicMock()
        import core.Imports as _ci

        with (
            patch.object(_ci, "get_world_state", return_value=mock_ws),
            patch.object(_ci, "get_perception_refresh_daemon", return_value=None),
            patch("config.ROS.ROS_ENABLED", False),
            patch("config.Servers.VGN_ENABLED", False),
        ):
            from operations.GraspOperations import grasp_object

            try:
                grasp_object(robot_id="Robot1", object_id="red_cube", request_id=0)
            except Exception:
                pass

        update_calls = mock_ws.update_robot_state.call_args_list
        intent_cleared = [
            c
            for c in update_calls
            if isinstance(c[0][1], dict) and c[0][1].get("moving_toward_object") is None
        ]
        assert len(intent_cleared) >= 1
