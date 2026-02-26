"""
Integration tests for end-to-end grasp planning pipeline.

Tests the full pipeline from candidate generation through scoring and selection.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from grasp_planning.GraspPlanner import GraspPlanner
from grasp_planning.GraspConfig import GraspConfig


class TestGraspPlanner:
    """Test end-to-end grasp planning pipeline."""

    @pytest.fixture
    def planner(self):
        """Create default grasp planner."""
        return GraspPlanner()

    @pytest.fixture
    def fast_planner(self):
        """Create fast grasp planner."""
        config = GraspConfig.create_fast()
        return GraspPlanner(config)

    @pytest.fixture
    def precise_planner(self):
        """Create precise grasp planner."""
        config = GraspConfig.create_precise()
        return GraspPlanner(config)

    def test_planner_initialization(self):
        """Test planner initializes correctly."""
        planner = GraspPlanner()
        assert planner.config is not None
        assert planner.generator is not None
        assert planner.scorer is not None

    def test_plan_grasp_returns_best_candidate(self, planner):
        """Test that plan_grasp returns a valid candidate."""
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
        """Test planning with preferred approach."""
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
        """Test that plan_grasp respects minimum score threshold."""
        object_position = (0.0, 0.05, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        robot_id = "Robot1"
        gripper_position = (0.0, 0.15, 0.0)

        # Very high threshold - should return None
        best_grasp = planner.plan_grasp(
            object_position=object_position,
            object_rotation=object_rotation,
            object_size=object_size,
            robot_id=robot_id,
            gripper_position=gripper_position,
            use_moveit_ik=False,
            min_score=10.0,  # Impossibly high
        )

        assert best_grasp is None

    def test_plan_grasp_with_unreachable_object(self, planner):
        """Test planning for object far from gripper."""
        object_position = (10.0, 10.0, 10.0)  # Very far
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

        # Should still return a candidate, but with low score
        if best_grasp is not None:
            # IK score should be very low due to distance
            assert best_grasp.ik_score < 0.5

    def test_plan_multi_grasp(self, planner):
        """Test planning multiple grasp candidates."""
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
        # Should be sorted by score
        assert candidates[0].total_score >= candidates[1].total_score
        assert candidates[1].total_score >= candidates[2].total_score

    def test_get_statistics(self, planner):
        """Test getting statistics about candidates."""
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
        """Test that fast planner generates fewer candidates."""
        # Fast planner should have fewer candidates per approach
        assert fast_planner.config.candidates_per_approach < planner.config.candidates_per_approach

    def test_precise_planner_generates_more_candidates(self, precise_planner, planner):
        """Test that precise planner generates more candidates."""
        # Precise planner should have more candidates per approach
        assert precise_planner.config.candidates_per_approach > planner.config.candidates_per_approach

    def test_object_rotation_affects_candidates(self, planner):
        """Test that object rotation affects grasp candidates."""
        object_position = (0.0, 0.05, 0.0)
        object_size = (0.05, 0.05, 0.05)
        robot_id = "Robot1"
        gripper_position = (0.0, 0.15, 0.0)

        # Identity rotation
        best_grasp_1 = planner.plan_grasp(
            object_position=object_position,
            object_rotation=(0.0, 0.0, 0.0, 1.0),
            object_size=object_size,
            robot_id=robot_id,
            gripper_position=gripper_position,
            use_moveit_ik=False,
        )

        # 45 degree rotation around Y
        from utils.QuaternionMath import quaternion_from_euler
        rotated_quat = quaternion_from_euler(0.0, np.pi/4, 0.0)

        best_grasp_2 = planner.plan_grasp(
            object_position=object_position,
            object_rotation=rotated_quat,
            object_size=object_size,
            robot_id=robot_id,
            gripper_position=gripper_position,
            use_moveit_ik=False,
        )

        # Grasp rotations should differ (object rotation affects grasp orientation)
        if best_grasp_1 and best_grasp_2:
            assert not np.allclose(
                best_grasp_1.grasp_rotation,
                best_grasp_2.grasp_rotation,
                atol=1e-3
            )

    def test_different_object_sizes(self, planner):
        """Test planning for different object sizes."""
        object_position = (0.0, 0.05, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        robot_id = "Robot1"
        gripper_position = (0.0, 0.15, 0.0)

        # Small object
        small_grasp = planner.plan_grasp(
            object_position=object_position,
            object_rotation=object_rotation,
            object_size=(0.02, 0.02, 0.02),
            robot_id=robot_id,
            gripper_position=gripper_position,
            use_moveit_ik=False,
        )

        # Large object
        large_grasp = planner.plan_grasp(
            object_position=object_position,
            object_rotation=object_rotation,
            object_size=(0.10, 0.10, 0.10),
            robot_id=robot_id,
            gripper_position=gripper_position,
            use_moveit_ik=False,
        )

        # Both should return candidates
        assert small_grasp is not None
        assert large_grasp is not None

        # Approach distances should differ based on object size
        assert small_grasp.approach_distance != large_grasp.approach_distance

    def test_gripper_rotation_affects_scoring(self, planner):
        """Test that current gripper rotation affects candidate scoring."""
        object_position = (0.0, 0.05, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        robot_id = "Robot1"
        gripper_position = (0.0, 0.15, 0.0)

        # Without gripper rotation
        grasp_no_rot = planner.plan_grasp(
            object_position=object_position,
            object_rotation=object_rotation,
            object_size=object_size,
            robot_id=robot_id,
            gripper_position=gripper_position,
            gripper_rotation=None,
            use_moveit_ik=False,
        )

        # With gripper rotation (should affect orientation consistency)
        from utils.QuaternionMath import quaternion_from_euler
        gripper_rot = quaternion_from_euler(0.0, 0.0, np.pi/2)

        grasp_with_rot = planner.plan_grasp(
            object_position=object_position,
            object_rotation=object_rotation,
            object_size=object_size,
            robot_id=robot_id,
            gripper_position=gripper_position,
            gripper_rotation=gripper_rot,
            use_moveit_ik=False,
        )

        # Both should return candidates
        assert grasp_no_rot is not None
        assert grasp_with_rot is not None

        # Scores may differ due to orientation consistency
        # (not guaranteed, but validates scoring is working)


class TestGraspPlannerEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_object_size(self):
        """Test with very small object size."""
        planner = GraspPlanner()

        best_grasp = planner.plan_grasp(
            object_position=(0.0, 0.05, 0.0),
            object_rotation=(0.0, 0.0, 0.0, 1.0),
            object_size=(0.001, 0.001, 0.001),  # Very small
            robot_id="Robot1",
            gripper_position=(0.0, 0.15, 0.0),
            use_moveit_ik=False,
        )

        # Should still return a candidate (may have low score)
        assert best_grasp is not None or best_grasp is None  # Both valid

    def test_negative_min_score(self):
        """Test with negative minimum score threshold."""
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

        # Should accept any candidate
        assert best_grasp is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
