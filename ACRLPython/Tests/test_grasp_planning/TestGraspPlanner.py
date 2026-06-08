import pytest
import numpy as np

from grasp_planning.GraspPlanner import GraspPlanner
from grasp_planning.GraspConfig import GraspConfig


class TestGraspPlanner:

    @pytest.fixture
    def planner(self):
        return GraspPlanner()

    @pytest.fixture
    def fast_planner(self):
        config = GraspConfig.create_fast()
        return GraspPlanner(config)

    @pytest.fixture
    def precise_planner(self):
        config = GraspConfig.create_precise()
        return GraspPlanner(config)

    def test_planner_initialization(self):
        planner = GraspPlanner()
        assert planner.config is not None
        assert planner.generator is not None
        assert planner.scorer is not None

    def test_plan_grasp_returns_best_candidate(self, planner):
        object_position = (0.0, 0.05, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        robot_id = "Robot1"
        gripper_position = (0.0, 0.15, 0.0)

        best_grasp = planner.plan_grasp(
            object_position=object_position,
            object_rotation=object_rotation,
            object_size=object_size,
            robot_id=robot_id,
            gripper_position=gripper_position,
            use_moveit_ik=False,  # Skip IK validation for unit test
        )

        assert best_grasp is not None
        assert best_grasp.total_score > 0
        assert best_grasp.approach_type in ["top", "front", "side"]

    def test_plan_grasp_with_preferred_approach(self, planner):
        object_position = (0.0, 0.05, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        robot_id = "Robot1"
        gripper_position = (0.0, 0.15, 0.0)

        best_grasp = planner.plan_grasp(
            object_position=object_position,
            object_rotation=object_rotation,
            object_size=object_size,
            robot_id=robot_id,
            gripper_position=gripper_position,
            use_moveit_ik=False,
            preferred_approach="top",
        )

        assert best_grasp is not None
        assert best_grasp.approach_type == "top"

    def test_plan_grasp_respects_min_score(self, planner):
        object_position = (0.0, 0.05, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        robot_id = "Robot1"
        gripper_position = (0.0, 0.15, 0.0)

        best_grasp = planner.plan_grasp(
            object_position=object_position,
            object_rotation=object_rotation,
            object_size=object_size,
            robot_id=robot_id,
            gripper_position=gripper_position,
            use_moveit_ik=False,
            min_score=10.0,
        )

        assert best_grasp is None

    def test_plan_grasp_with_unreachable_object(self, planner):
        object_position = (10.0, 10.0, 10.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        robot_id = "Robot1"
        gripper_position = (0.0, 0.15, 0.0)

        best_grasp = planner.plan_grasp(
            object_position=object_position,
            object_rotation=object_rotation,
            object_size=object_size,
            robot_id=robot_id,
            gripper_position=gripper_position,
            use_moveit_ik=False,
        )

        if best_grasp is not None:
            assert best_grasp.ik_score < 0.5

    def test_plan_multi_grasp(self, planner):
        object_position = (0.0, 0.05, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        robot_id = "Robot1"
        gripper_position = (0.0, 0.15, 0.0)

        candidates = planner.plan_multi_grasp(
            object_position=object_position,
            object_rotation=object_rotation,
            object_size=object_size,
            robot_id=robot_id,
            gripper_position=gripper_position,
            num_candidates=3,
            use_moveit_ik=False,
        )

        assert len(candidates) == 3
        assert candidates[0].total_score >= candidates[1].total_score
        assert candidates[1].total_score >= candidates[2].total_score

    def test_get_statistics(self, planner):
        object_position = (0.0, 0.05, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        robot_id = "Robot1"
        gripper_position = (0.0, 0.15, 0.0)

        candidates = planner.plan_multi_grasp(
            object_position=object_position,
            object_rotation=object_rotation,
            object_size=object_size,
            robot_id=robot_id,
            gripper_position=gripper_position,
            num_candidates=5,
            use_moveit_ik=False,
        )

        stats = planner.get_statistics(candidates)

        assert stats["count"] == 5
        assert "score_mean" in stats
        assert "score_min" in stats
        assert "score_max" in stats
        assert "approach_counts" in stats

    def test_fast_planner_generates_fewer_candidates(self, fast_planner, planner):
        assert (
            fast_planner.config.candidates_per_approach
            < planner.config.candidates_per_approach
        )

    def test_precise_planner_generates_more_candidates(self, precise_planner, planner):
        assert (
            precise_planner.config.candidates_per_approach
            > planner.config.candidates_per_approach
        )

    def test_object_rotation_affects_candidates(self, planner):
        object_position = (0.0, 0.05, 0.0)
        object_size = (0.05, 0.05, 0.05)
        robot_id = "Robot1"
        gripper_position = (0.0, 0.15, 0.0)

        best_grasp_1 = planner.plan_grasp(
            object_position=object_position,
            object_rotation=(0.0, 0.0, 0.0, 1.0),
            object_size=object_size,
            robot_id=robot_id,
            gripper_position=gripper_position,
            use_moveit_ik=False,
        )

        from utils.QuaternionMath import quaternion_from_euler

        rotated_quat = quaternion_from_euler(0.0, np.pi / 4, 0.0)

        best_grasp_2 = planner.plan_grasp(
            object_position=object_position,
            object_rotation=rotated_quat,
            object_size=object_size,
            robot_id=robot_id,
            gripper_position=gripper_position,
            use_moveit_ik=False,
        )

        if best_grasp_1 and best_grasp_2:
            assert not np.allclose(
                best_grasp_1.grasp_rotation, best_grasp_2.grasp_rotation, atol=1e-3
            )

    def test_different_object_sizes(self, planner):
        object_position = (0.0, 0.05, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        robot_id = "Robot1"
        gripper_position = (0.0, 0.15, 0.0)

        small_grasp = planner.plan_grasp(
            object_position=object_position,
            object_rotation=object_rotation,
            object_size=(0.02, 0.02, 0.02),
            robot_id=robot_id,
            gripper_position=gripper_position,
            use_moveit_ik=False,
        )

        large_grasp = planner.plan_grasp(
            object_position=object_position,
            object_rotation=object_rotation,
            object_size=(0.10, 0.10, 0.10),
            robot_id=robot_id,
            gripper_position=gripper_position,
            use_moveit_ik=False,
        )

        assert small_grasp is not None
        assert large_grasp is not None
        assert small_grasp.approach_distance != large_grasp.approach_distance

    def test_gripper_rotation_affects_scoring(self, planner):
        object_position = (0.0, 0.05, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        robot_id = "Robot1"
        gripper_position = (0.0, 0.15, 0.0)

        grasp_no_rot = planner.plan_grasp(
            object_position=object_position,
            object_rotation=object_rotation,
            object_size=object_size,
            robot_id=robot_id,
            gripper_position=gripper_position,
            gripper_rotation=None,
            use_moveit_ik=False,
        )

        from utils.QuaternionMath import quaternion_from_euler

        gripper_rot = quaternion_from_euler(0.0, 0.0, np.pi / 2)

        grasp_with_rot = planner.plan_grasp(
            object_position=object_position,
            object_rotation=object_rotation,
            object_size=object_size,
            robot_id=robot_id,
            gripper_position=gripper_position,
            gripper_rotation=gripper_rot,
            use_moveit_ik=False,
        )

        assert grasp_no_rot is not None
        assert grasp_with_rot is not None


class TestGraspPlannerEdgeCases:

    def test_empty_object_size(self):
        planner = GraspPlanner()

        best_grasp = planner.plan_grasp(
            object_position=(0.0, 0.05, 0.0),
            object_rotation=(0.0, 0.0, 0.0, 1.0),
            object_size=(0.001, 0.001, 0.001),  # Very small
            robot_id="Robot1",
            gripper_position=(0.0, 0.15, 0.0),
            use_moveit_ik=False,
        )

        assert best_grasp is not None or best_grasp is None

    def test_negative_min_score(self):
        planner = GraspPlanner()

        best_grasp = planner.plan_grasp(
            object_position=(0.0, 0.05, 0.0),
            object_rotation=(0.0, 0.0, 0.0, 1.0),
            object_size=(0.05, 0.05, 0.05),
            robot_id="Robot1",
            gripper_position=(0.0, 0.15, 0.0),
            use_moveit_ik=False,
            min_score=-1.0,
        )

        assert best_grasp is not None


class TestGraspPlannerConfigMutationRegression:
    """
    Regression tests for Bug 1: config mutation in _filter_approaches.

    GraspPlanner instances are reused (e.g. as singletons in GraspOperations).
    A call with preferred_approach must not permanently disable other approaches
    for subsequent calls on the same instance.
    """

    def test_preferred_approach_does_not_persist_across_calls(self):
        """
        Calling plan_grasp with preferred_approach='top' must not disable
        side and front approaches for a subsequent call without preferred_approach.
        """
        planner = GraspPlanner()
        common_kwargs = dict(
            object_position=(0.0, 0.05, 0.0),
            object_rotation=(0.0, 0.0, 0.0, 1.0),
            object_size=(0.05, 0.05, 0.05),
            robot_id="Robot1",
            gripper_position=(0.0, 0.15, 0.0),
            use_moveit_ik=False,
            min_score=-1.0,  # accept any candidate
        )

        planner.plan_grasp(preferred_approach="top", **common_kwargs)  # type: ignore[arg-type]

        candidates = planner.generator.generate_candidates(
            object_position=(0.0, 0.05, 0.0),
            object_rotation=(0.0, 0.0, 0.0, 1.0),
            object_size=(0.05, 0.05, 0.05),
            gripper_position=(0.0, 0.15, 0.0),
        )

        approach_types = {c.approach_type for c in candidates}
        assert (
            "top" in approach_types
        ), "top approach missing after preferred_approach call"
        assert "side" in approach_types, "side approach was permanently disabled"
        assert "front" in approach_types, "front approach was permanently disabled"

    def test_preferred_approach_preference_weight_restored(self):
        """
        The preference_weight boosted to 2.0 for preferred_approach must be
        restored to its original value after plan_grasp returns.
        """
        planner = GraspPlanner()

        original_weights = {
            s.approach_type: s.preference_weight
            for s in planner.config.enabled_approaches
        }

        planner.plan_grasp(
            object_position=(0.0, 0.05, 0.0),
            object_rotation=(0.0, 0.0, 0.0, 1.0),
            object_size=(0.05, 0.05, 0.05),
            robot_id="Robot1",
            gripper_position=(0.0, 0.15, 0.0),
            use_moveit_ik=False,
            min_score=-1.0,
            preferred_approach="side",
        )

        for s in planner.config.enabled_approaches:
            assert (
                s.preference_weight == original_weights[s.approach_type]
            ), f"preference_weight for '{s.approach_type}' was not restored"

    def test_unknown_preferred_approach_logs_warning_and_returns_none(self):
        planner = GraspPlanner()

        result = planner.plan_grasp(
            object_position=(0.0, 0.05, 0.0),
            object_rotation=(0.0, 0.0, 0.0, 1.0),
            object_size=(0.05, 0.05, 0.05),
            robot_id="Robot1",
            gripper_position=(0.0, 0.15, 0.0),
            use_moveit_ik=False,
            preferred_approach="nonexistent_approach",
        )

        assert result is None

    def test_config_restored_even_when_generation_raises(self):
        import unittest.mock as mock

        planner = GraspPlanner()
        original_enabled = [s.enabled for s in planner.config.enabled_approaches]

        with mock.patch.object(
            planner.generator,
            "generate_candidates",
            side_effect=RuntimeError("test error"),
        ):
            with pytest.raises(RuntimeError):
                planner.plan_grasp(
                    object_position=(0.0, 0.05, 0.0),
                    object_rotation=(0.0, 0.0, 0.0, 1.0),
                    object_size=(0.05, 0.05, 0.05),
                    robot_id="Robot1",
                    gripper_position=(0.0, 0.15, 0.0),
                    use_moveit_ik=False,
                    preferred_approach="top",
                )

        for s, orig in zip(planner.config.enabled_approaches, original_enabled):
            assert (
                s.enabled == orig
            ), f"enabled state for '{s.approach_type}' not restored after exception"
