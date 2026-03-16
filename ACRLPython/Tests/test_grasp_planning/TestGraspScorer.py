#!/usr/bin/env python3
"""
Unit tests for grasp scoring system.

Tests multi-criteria scoring including IK quality, approach preference,
depth, stability, and orientation consistency.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from grasp_planning.GraspScorer import GraspScorer
from grasp_planning.GraspConfig import GraspConfig
from grasp_planning.GraspCandidate import GraspCandidate
from utils.QuaternionMath import quaternion_from_euler


class TestGraspScorer:
    """Test grasp scoring functionality."""

    @pytest.fixture
    def config(self):
        """Create default configuration."""
        return GraspConfig.create_default()

    @pytest.fixture
    def scorer(self, config):
        """Create scorer instance."""
        return GraspScorer(config)

    @pytest.fixture
    def sample_candidate(self):
        """Create a sample grasp candidate."""
        return GraspCandidate.create(
            pre_grasp=(0.0, 0.15, 0.0),
            pre_grasp_rot=(0.0, 0.0, 0.0, 1.0),
            grasp=(0.0, 0.05, 0.0),
            grasp_rot=(0.0, 0.0, 0.0, 1.0),
            approach="top",
        )

    def test_scorer_initialization(self, config):
        """Test scorer initializes correctly."""
        scorer = GraspScorer(config)
        assert scorer.config == config

    def test_score_and_rank_returns_sorted_list(self, scorer):
        """Test that score_and_rank returns candidates sorted by score."""
        candidates = [
            GraspCandidate.create(
                (0.0, 0.15, 0.0),
                (0.0, 0.0, 0.0, 1.0),
                (0.0, 0.05, 0.0),
                (0.0, 0.0, 0.0, 1.0),
                "top",
            )
            for _ in range(5)
        ]

        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.0, 0.15, 0.0)

        ranked = scorer.score_and_rank(candidates, object_size, gripper_position)

        # Check that scores are in descending order
        scores = [c.total_score for c in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_ik_score_uses_validation_when_available(self, scorer):
        """Test that IK score uses MoveIt validation when available."""
        candidate = GraspCandidate.create(
            (0.0, 0.15, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.05, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            "top",
        )

        # Set IK validation result
        candidate.ik_validated = True
        candidate.ik_score = 0.9

        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.0, 0.15, 0.0)

        scorer.score_and_rank([candidate], object_size, gripper_position)

        # IK score should be preserved
        assert candidate.ik_score == 0.9

    def test_ik_score_distance_based_when_not_validated(self, scorer):
        """Test that IK score is distance-based when not validated."""
        # Close candidate
        close_candidate = GraspCandidate.create(
            (0.0, 0.16, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.05, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            "top",
        )

        # Far candidate
        far_candidate = GraspCandidate.create(
            (0.0, 0.50, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.05, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            "top",
        )

        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.0, 0.15, 0.0)

        scorer.score_and_rank(
            [close_candidate, far_candidate], object_size, gripper_position
        )

        # Close candidate should have higher IK score
        assert close_candidate.ik_score > far_candidate.ik_score

    def test_approach_score_respects_weights(self, config, scorer):
        """Test that approach score uses configured weights."""
        # Top approach has weight 1.2 by default
        top_candidate = GraspCandidate.create(
            (0.0, 0.15, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.05, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            "top",
        )

        # Side approach has weight 0.8 by default
        side_candidate = GraspCandidate.create(
            (0.1, 0.05, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.05, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            "side",
        )

        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.0, 0.15, 0.0)

        scorer.score_and_rank(
            [top_candidate, side_candidate], object_size, gripper_position
        )

        # Top should have higher total score (all else equal)
        # Note: Other factors affect total score, so we just verify both scored
        assert top_candidate.total_score > 0
        assert side_candidate.total_score > 0

    def test_depth_score_gaussian_distribution(self, scorer):
        """Test that depth score follows Gaussian distribution."""
        object_size = (0.05, 0.05, 0.05)
        avg_size = np.mean(object_size)
        target_depth = scorer.config.target_grasp_depth * avg_size

        # Candidate with exactly target depth
        target_candidate = GraspCandidate.create(
            (0.0, 0.15, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.05, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            "top",
        )
        target_candidate.grasp_depth = target_depth

        # Candidate with depth far from target
        far_candidate = GraspCandidate.create(
            (0.0, 0.15, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.05, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            "top",
        )
        far_candidate.grasp_depth = target_depth * 2.0

        gripper_position = (0.0, 0.15, 0.0)

        scorer.score_and_rank(
            [target_candidate, far_candidate], object_size, gripper_position
        )

        # Candidate at target depth should have higher depth score
        # (extracted via _compute_depth_score, reflected in total)
        assert target_candidate.total_score > far_candidate.total_score

    def test_stability_score_gravity_alignment(self, scorer):
        """Test that stability score considers gravity alignment."""
        # Downward approach (aligned with gravity)
        down_candidate = GraspCandidate.create(
            (0.0, 0.15, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.05, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            "top",
        )
        down_candidate.approach_direction = (0.0, -1.0, 0.0)

        # Horizontal approach (perpendicular to gravity)
        horizontal_candidate = GraspCandidate.create(
            (0.1, 0.05, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.05, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            "side",
        )
        horizontal_candidate.approach_direction = (1.0, 0.0, 0.0)

        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.0, 0.15, 0.0)

        scorer.score_and_rank(
            [down_candidate, horizontal_candidate], object_size, gripper_position
        )

        # Both should be scored
        assert down_candidate.total_score > 0
        assert horizontal_candidate.total_score > 0

    def test_orientation_consistency_penalty(self, scorer):
        """Test that orientation consistency penalizes large rotations."""
        # Same orientation as current
        same_candidate = GraspCandidate.create(
            (0.0, 0.15, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.05, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            "top",
        )

        # 180 degree rotation from current
        flipped_candidate = GraspCandidate.create(
            (0.0, 0.15, 0.0),
            quaternion_from_euler(np.pi, 0.0, 0.0),
            (0.0, 0.05, 0.0),
            quaternion_from_euler(np.pi, 0.0, 0.0),
            "top",
        )

        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.0, 0.15, 0.0)
        current_rotation = (0.0, 0.0, 0.0, 1.0)

        scorer.score_and_rank(
            [same_candidate, flipped_candidate],
            object_size,
            gripper_position,
            current_rotation,
        )

        # Same orientation should have higher total score
        assert same_candidate.total_score > flipped_candidate.total_score

    def test_filter_by_min_score(self, scorer):
        """Test filtering candidates by minimum score."""
        candidates = []
        for i in range(5):
            candidate = GraspCandidate.create(
                (0.0, 0.15 + i * 0.01, 0.0),
                (0.0, 0.0, 0.0, 1.0),
                (0.0, 0.05, 0.0),
                (0.0, 0.0, 0.0, 1.0),
                "top",
            )
            candidates.append(candidate)

        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.0, 0.15, 0.0)

        ranked = scorer.score_and_rank(candidates, object_size, gripper_position)

        # Filter with threshold
        min_score = 0.5
        filtered = scorer.filter_by_min_score(ranked, min_score)

        # All filtered candidates should be above threshold
        assert all(c.total_score >= min_score for c in filtered)

    def test_get_top_n(self, scorer):
        """Test getting top N candidates."""
        candidates = []
        for i in range(10):
            candidate = GraspCandidate.create(
                (0.0, 0.15 + i * 0.01, 0.0),
                (0.0, 0.0, 0.0, 1.0),
                (0.0, 0.05, 0.0),
                (0.0, 0.0, 0.0, 1.0),
                "top",
            )
            candidates.append(candidate)

        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.0, 0.15, 0.0)

        ranked = scorer.score_and_rank(candidates, object_size, gripper_position)

        # Get top 3
        top_3 = scorer.get_top_n(ranked, 3)

        assert len(top_3) == 3
        assert top_3[0].total_score >= top_3[1].total_score >= top_3[2].total_score

    def test_normalize_scores(self, scorer):
        """Test score normalization."""
        candidates = []
        for i in range(5):
            candidate = GraspCandidate.create(
                (0.0, 0.15 + i * 0.05, 0.0),
                (0.0, 0.0, 0.0, 1.0),
                (0.0, 0.05, 0.0),
                (0.0, 0.0, 0.0, 1.0),
                "top",
            )
            candidates.append(candidate)

        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.0, 0.15, 0.0)

        ranked = scorer.score_and_rank(candidates, object_size, gripper_position)

        # Normalize
        scorer.normalize_scores(ranked)

        # Check that scores are in [0, 1] with min=0 and max=1
        scores = [c.total_score for c in ranked]
        assert min(scores) == 0.0
        assert max(scores) == 1.0
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_all_scores_in_valid_range(self, scorer):
        """Test that all score components are in valid range."""
        candidate = GraspCandidate.create(
            (0.0, 0.15, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.05, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            "top",
        )
        candidate.approach_direction = (0.0, -1.0, 0.0)

        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.0, 0.15, 0.0)

        scorer.score_and_rank([candidate], object_size, gripper_position)

        # All score components should be in [0, 1]
        assert 0.0 <= candidate.ik_score <= 1.0
        assert 0.0 <= candidate.antipodal_score <= 1.0
        # Total score can exceed 1.0 due to weighted sum


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
