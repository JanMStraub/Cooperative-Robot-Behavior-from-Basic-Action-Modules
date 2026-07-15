from unittest.mock import MagicMock, patch

from grasp_planning.GraspCandidate import GraspCandidate
from operations.grasp._ros import _grasp_via_ros_planned, _grasp_via_ros_position_only


class TestGraspViaRosPositionOnlyAllowParallel:
    def _run(self, allow_parallel_ros):
        bridge = MagicMock()
        bridge.plan_and_execute.return_value = {"success": True}
        bridge.plan_cartesian_descent.return_value = {
            "success": True,
            "planning_time": 1.0,
        }

        with patch(
            "operations.grasp._ros._execute_grasp_with_follow_target",
            return_value=(True, ""),
        ) as mock_follow_target:
            result, fallback = _grasp_via_ros_position_only(
                bridge=bridge,
                robot_id="Robot1",
                object_id="blue_cube",
                object_position=(0.3, 0.05, 0.1),
                request_id=1,
                world_state=MagicMock(),
                grasp_yaw_override=0.0,
                allow_parallel=allow_parallel_ros,
            )

        assert fallback is False
        assert result.success is True
        return bridge, mock_follow_target

    def test_allow_parallel_true_reaches_all_plan_calls(self):
        bridge, mock_follow_target = self._run(True)

        for _, kwargs in bridge.plan_and_execute.call_args_list:
            assert kwargs["allow_parallel"] is True
        for _, kwargs in bridge.plan_cartesian_descent.call_args_list:
            assert kwargs["allow_parallel"] is True
        assert mock_follow_target.call_args[1]["allow_parallel"] is True

    def test_allow_parallel_false_by_default(self):
        bridge, mock_follow_target = self._run(False)

        for _, kwargs in bridge.plan_and_execute.call_args_list:
            assert kwargs["allow_parallel"] is False
        for _, kwargs in bridge.plan_cartesian_descent.call_args_list:
            assert kwargs["allow_parallel"] is False
        assert mock_follow_target.call_args[1]["allow_parallel"] is False


class TestGraspViaRosPlannedAllowParallel:
    def _best_grasp(self):
        return GraspCandidate(
            pre_grasp_position=(0.3, 0.2, 0.1),
            pre_grasp_rotation=(0.0, 0.0, 0.0, 1.0),
            grasp_position=(0.3, 0.05, 0.1),
            grasp_rotation=(0.0, 0.0, 0.0, 1.0),
            approach_type="top",
            total_score=0.9,
        )

    def _run(self, allow_parallel_ros):
        bridge = MagicMock()
        bridge.plan_and_execute.return_value = {"success": True}
        bridge.plan_cartesian_descent.return_value = {
            "success": True,
            "planning_time": 1.0,
        }

        mock_planner = MagicMock()
        mock_planner.plan_grasp.return_value = self._best_grasp()

        robot_state = MagicMock()
        robot_state.position = (0.0, 0.0, 0.0)

        with patch(
            "grasp_planning.GraspPlanner.GraspPlanner", return_value=mock_planner
        ), patch(
            "operations.grasp._ros._execute_grasp_with_follow_target",
            return_value=(True, ""),
        ) as mock_follow_target:
            result, fallback = _grasp_via_ros_planned(
                bridge=bridge,
                robot_id="Robot1",
                object_id="blue_cube",
                object_position=(0.3, 0.05, 0.1),
                object_dimensions=(0.05, 0.05, 0.05),
                robot_state=robot_state,
                preferred_approach="top",
                request_id=1,
                world_state=MagicMock(),
                grasp_yaw_override=0.0,
                allow_parallel=allow_parallel_ros,
            )

        assert fallback is False
        assert result.success is True
        return bridge, mock_follow_target

    def test_allow_parallel_true_reaches_all_plan_calls(self):
        bridge, mock_follow_target = self._run(True)

        for _, kwargs in bridge.plan_and_execute.call_args_list:
            assert kwargs["allow_parallel"] is True
        for _, kwargs in bridge.plan_cartesian_descent.call_args_list:
            assert kwargs["allow_parallel"] is True
        assert mock_follow_target.call_args[1]["allow_parallel"] is True

    def test_allow_parallel_false_by_default(self):
        bridge, mock_follow_target = self._run(False)

        for _, kwargs in bridge.plan_and_execute.call_args_list:
            assert kwargs["allow_parallel"] is False
        for _, kwargs in bridge.plan_cartesian_descent.call_args_list:
            assert kwargs["allow_parallel"] is False
        assert mock_follow_target.call_args[1]["allow_parallel"] is False
