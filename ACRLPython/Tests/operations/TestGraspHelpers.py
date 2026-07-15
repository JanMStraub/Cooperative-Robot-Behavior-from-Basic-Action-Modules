import math
from unittest.mock import MagicMock, patch

from operations.grasp._helpers import (
    _execute_grasp_with_follow_target,
    _side_offset_world_xz,
)


class TestExecuteGraspWithFollowTargetAllowParallel:
    def _run(self, allow_parallel):
        bridge = MagicMock()
        bridge.plan_and_execute.return_value = {"success": True}
        bridge.control_gripper.return_value = {"success": True}

        world_state = MagicMock()
        # Drifted 10cm from planned position - well above FOLLOW_TARGET_DRIFT_THRESHOLD,
        # forcing exactly one retract+hover+correction round (drift resolves to 0 after
        # one correction since approach_offset_xz defaults to (0.0, 0.0)).
        world_state.get_object_position.return_value = (0.4, 0.05, 0.1)

        with patch("operations.grasp._helpers.FOLLOW_TARGET_ENABLED", True), patch(
            "operations.grasp._helpers.FOLLOW_TARGET_MAX_CORRECTIONS", 2
        ), patch("operations.grasp._helpers.FOLLOW_TARGET_DRIFT_THRESHOLD", 0.01):
            success, reason = _execute_grasp_with_follow_target(
                bridge=bridge,
                robot_id="Robot1",
                object_id="blue_cube",
                planned_position={"x": 0.3, "y": 0.05, "z": 0.1},
                orientation={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                world_state=world_state,
                allow_parallel=allow_parallel,
            )

        assert success is True, reason
        return bridge

    def test_allow_parallel_true_reaches_correction_plan_calls(self):
        bridge = self._run(True)

        assert (
            bridge.plan_and_execute.call_args_list
        ), "expected at least one correction call"
        for _, kwargs in bridge.plan_and_execute.call_args_list:
            assert kwargs["allow_parallel"] is True

    def test_allow_parallel_false_by_default(self):
        bridge = self._run(False)

        assert (
            bridge.plan_and_execute.call_args_list
        ), "expected at least one correction call"
        for _, kwargs in bridge.plan_and_execute.call_args_list:
            assert kwargs["allow_parallel"] is False


class TestSideOffsetWorldXZ:
    def test_axis_aligned_offset_is_pure_x(self):
        dx, dz = _side_offset_world_xz(lx=0.13, lz=0.02, yaw_deg=0.0, side_sign=1.0)

        assert dx == 0.6 * 0.065
        assert dz == 0.0

    def test_rotated_offset_stays_on_object_axis(self):
        lx, lz = 0.13, 0.02
        yaw_deg = 70.0
        dx, dz = _side_offset_world_xz(lx, lz, yaw_deg, side_sign=1.0)

        magnitude = math.hypot(dx, dz)
        assert math.isclose(magnitude, 0.6 * (lx / 2.0))

        yaw_rad = math.radians(yaw_deg)
        expected_dir = (math.cos(yaw_rad), -math.sin(yaw_rad))
        actual_dir = (dx / magnitude, dz / magnitude)
        assert math.isclose(actual_dir[0], expected_dir[0])
        assert math.isclose(actual_dir[1], expected_dir[1])

    def test_left_and_right_are_opposite(self):
        right = _side_offset_world_xz(0.13, 0.02, 70.0, side_sign=1.0)
        left = _side_offset_world_xz(0.13, 0.02, 70.0, side_sign=-1.0)

        assert right[0] == -left[0]
        assert right[1] == -left[1]

    def test_uses_longer_local_axis(self):
        dx, dz = _side_offset_world_xz(lx=0.02, lz=0.13, yaw_deg=0.0, side_sign=1.0)

        assert dx == 0.0
        assert dz == 0.6 * 0.065


class TestFollowTargetOffsetAwareDrift:
    def test_stationary_object_with_offset_target_reports_no_drift(self):
        bridge = MagicMock()
        bridge.plan_and_execute.return_value = {"success": True}
        bridge.control_gripper.return_value = {"success": True}

        object_center = (-0.048, 0.028, -0.095)
        offset_xz = (-0.013, 0.037)
        planned_position = {
            "x": object_center[0] + offset_xz[0],
            "y": object_center[1],
            "z": object_center[2] + offset_xz[1],
        }

        world_state = MagicMock()
        world_state.get_object_position.return_value = object_center

        with patch("operations.grasp._helpers.FOLLOW_TARGET_ENABLED", True), patch(
            "operations.grasp._helpers.FOLLOW_TARGET_MAX_CORRECTIONS", 1
        ), patch("operations.grasp._helpers.FOLLOW_TARGET_DRIFT_THRESHOLD", 0.025):
            success, reason = _execute_grasp_with_follow_target(
                bridge=bridge,
                robot_id="Robot1",
                object_id="red_cube",
                planned_position=planned_position,
                orientation={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                world_state=world_state,
                approach_offset_xz=offset_xz,
            )

        assert success is True, reason
        assert (
            bridge.plan_and_execute.call_count == 0
        ), "object never moved - no retract/hover/correction should have been planned"
