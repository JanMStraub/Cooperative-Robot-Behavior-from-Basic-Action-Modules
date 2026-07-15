from unittest.mock import patch

import pytest

from ros2.ROSBridge import ROSBridge


@pytest.fixture(autouse=True)
def _reset_parallel_instances():
    ROSBridge._parallel_instances = {}
    yield
    ROSBridge._parallel_instances = {}


class TestGetParallelInstance:
    def test_returns_distinct_instances_per_robot(self):
        with patch.object(ROSBridge, "connect", return_value=True):
            robot1_bridge = ROSBridge.get_parallel_instance("Robot1")
            robot2_bridge = ROSBridge.get_parallel_instance("Robot2")

        assert robot1_bridge is not robot2_bridge

    def test_returns_same_instance_for_repeated_calls(self):
        with patch.object(ROSBridge, "connect", return_value=True):
            first = ROSBridge.get_parallel_instance("Robot1")
            second = ROSBridge.get_parallel_instance("Robot1")

        assert first is second

    def test_connects_new_instance_on_creation(self):
        with patch.object(ROSBridge, "connect", return_value=True) as mock_connect:
            ROSBridge.get_parallel_instance("Robot1")

        mock_connect.assert_called_once()


class TestAllowParallelCommandField:
    def test_plan_and_execute_sets_allow_parallel_true(self):
        bridge = ROSBridge()
        with patch.object(
            bridge, "_send_command", return_value={"success": True}
        ) as mock_send:
            bridge.plan_and_execute(
                position={"x": 0.0, "y": 0.0, "z": 0.0}, allow_parallel=True
            )

        cmd = mock_send.call_args[0][0]
        assert cmd["allow_parallel"] is True

    def test_plan_and_execute_omits_allow_parallel_by_default(self):
        bridge = ROSBridge()
        with patch.object(
            bridge, "_send_command", return_value={"success": True}
        ) as mock_send:
            bridge.plan_and_execute(position={"x": 0.0, "y": 0.0, "z": 0.0})

        cmd = mock_send.call_args[0][0]
        assert "allow_parallel" not in cmd

    def test_plan_orientation_change_sets_allow_parallel_true(self):
        bridge = ROSBridge()
        with patch.object(
            bridge, "_send_command", return_value={"success": True}
        ) as mock_send:
            bridge.plan_orientation_change(
                orientation={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                allow_parallel=True,
            )

        cmd = mock_send.call_args[0][0]
        assert cmd["allow_parallel"] is True

    def test_plan_orientation_change_omits_allow_parallel_by_default(self):
        bridge = ROSBridge()
        with patch.object(
            bridge, "_send_command", return_value={"success": True}
        ) as mock_send:
            bridge.plan_orientation_change(
                orientation={"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
            )

        cmd = mock_send.call_args[0][0]
        assert "allow_parallel" not in cmd

    def test_plan_cartesian_descent_sets_allow_parallel_true(self):
        bridge = ROSBridge()
        with patch.object(
            bridge, "_send_command", return_value={"success": True}
        ) as mock_send:
            bridge.plan_cartesian_descent(
                position={"x": 0.0, "y": 0.0, "z": 0.0}, allow_parallel=True
            )

        cmd = mock_send.call_args[0][0]
        assert cmd["allow_parallel"] is True

    def test_plan_cartesian_descent_omits_allow_parallel_by_default(self):
        bridge = ROSBridge()
        with patch.object(
            bridge, "_send_command", return_value={"success": True}
        ) as mock_send:
            bridge.plan_cartesian_descent(position={"x": 0.0, "y": 0.0, "z": 0.0})

        cmd = mock_send.call_args[0][0]
        assert "allow_parallel" not in cmd

    def test_plan_return_to_start_sets_allow_parallel_true(self):
        bridge = ROSBridge()
        with patch.object(
            bridge, "_send_command", return_value={"success": True}
        ) as mock_send:
            bridge.plan_return_to_start(allow_parallel=True)

        cmd = mock_send.call_args[0][0]
        assert cmd["allow_parallel"] is True

    def test_plan_return_to_start_omits_allow_parallel_by_default(self):
        bridge = ROSBridge()
        with patch.object(
            bridge, "_send_command", return_value={"success": True}
        ) as mock_send:
            bridge.plan_return_to_start()

        cmd = mock_send.call_args[0][0]
        assert "allow_parallel" not in cmd
