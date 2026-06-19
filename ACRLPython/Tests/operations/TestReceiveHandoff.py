import pytest
from unittest.mock import patch, MagicMock


def _make_world_state(
    obj_pos=(0.0, 0.15, 0.0),
    obj_dims=(0.04, 0.06, 0.04),
    robot_pos=(0.475, 0.0, 0.0),
):
    ws = MagicMock()
    ws.get_object_position.return_value = obj_pos
    ws.get_object_dimensions.return_value = obj_dims
    robot_state = MagicMock()
    robot_state.position = robot_pos
    ws.get_robot_state.return_value = robot_state
    return ws


@pytest.fixture
def mock_deps():
    ws = _make_world_state()
    move_result = MagicMock()
    move_result.success = True
    move_result.error = None
    gripper_result = MagicMock()
    gripper_result.success = True

    with (
        patch("core.Imports.get_world_state", return_value=ws),
        patch("config.ROS.ROS_ENABLED", False),
        patch("config.ROS.DEFAULT_CONTROL_MODE", "unity"),
        patch(
            "operations.SpatialPredicates.target_within_reach",
            return_value=(True, "ok"),
        ),
        patch(
            "operations.MoveOperations.move_to_coordinate", return_value=move_result
        ) as mock_move,
        patch(
            "operations.GripperOperations.control_gripper", return_value=gripper_result
        ) as mock_grip,
        patch("operations.MoveOperations._tcp_wait_for_not_moving"),
    ):
        yield {"ws": ws, "move": mock_move, "grip": mock_grip}


class TestHandoffGeometry:
    def test_tcp_path_stops_at_near_face(self, mock_deps):
        from operations.grasp._handoff import receive_handoff

        # obj at x=0.0, dims lx=0.04 → near_face = 0.0 + 1.0*0.02 = 0.02
        # robot2 is at x=+0.475 so approach_sign=+1.
        # With roll=90° the jaws are vertical in Y and straddle the object at
        # the near face - inserting further (to center_x) pushed the object
        # away (see _handoff.py approach comments).
        receive_handoff("Robot2", "red_bar", "Robot1")

        move_call = mock_deps["move"].call_args
        actual_x = move_call.kwargs.get("x") or move_call[1].get("x")
        assert actual_x == pytest.approx(0.02, abs=1e-6), (
            f"TCP move should stop at the near face (0.02), got {actual_x}. "
            "No X insertion - jaws straddle the object in Y at the face."
        )

    def test_gripper_closed_after_move(self, mock_deps):
        from operations.grasp._handoff import receive_handoff

        result = receive_handoff("Robot2", "red_bar", "Robot1")

        assert result.success is True
        close_calls = [
            c
            for c in mock_deps["grip"].call_args_list
            if c.kwargs.get("open_gripper", c[1].get("open_gripper")) is False
        ]
        assert close_calls, "Gripper never closed"

    def test_reach_check_still_uses_near_face(self, mock_deps):
        from operations.grasp._handoff import receive_handoff

        with patch(
            "operations.SpatialPredicates.target_within_reach",
            return_value=(True, "ok"),
        ) as mock_reach:
            receive_handoff("Robot2", "red_bar", "Robot1")
            reach_x = mock_reach.call_args[0][1]
            # near_face for robot2 (approach_sign=+1): 0.0 + 0.02 = 0.02
            assert reach_x == pytest.approx(
                0.02, abs=1e-6
            ), f"Reach check should use near_face_x (0.02), got {reach_x}"
