"""
Unit tests for grasp candidate generation.

Tests candidate generation algorithm including approach variation,
depth calculation, and antipodal scoring.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from grasp_planning.GraspCandidateGenerator import GraspCandidateGenerator
from grasp_planning.GraspConfig import GraspConfig
from grasp_planning.GraspCandidate import GraspCandidate


class TestGraspCandidateGenerator:
    """Test grasp candidate generation."""

    @pytest.fixture
    def config(self):
        """Create default configuration."""
        return GraspConfig.create_default()

    @pytest.fixture
    def generator(self, config):
        """Create generator with fixed seed for reproducibility."""
        return GraspCandidateGenerator(config, seed=42)

    def test_generator_initialization(self, config):
        """Test generator initializes correctly."""
        generator = GraspCandidateGenerator(config, seed=42)
        assert generator.config == config
        assert generator.random is not None

    def test_generates_correct_number_of_candidates(self, generator):
        """Test that correct number of candidates are generated."""
        object_position = (0.0, 0.0, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.3, 0.15, 0.0)

        candidates = generator.generate_candidates(
            object_position, object_rotation, object_size, gripper_position
        )

        # Should generate 8 candidates per approach (3 approaches enabled)
        expected_count = 8 * 3
        assert len(candidates) == expected_count

    def test_candidates_have_required_fields(self, generator):
        """Test that all candidates have required fields."""
        object_position = (0.0, 0.0, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.3, 0.15, 0.0)

        candidates = generator.generate_candidates(
            object_position, object_rotation, object_size, gripper_position
        )

        for candidate in candidates:
            assert candidate.pre_grasp_position is not None
            assert candidate.grasp_position is not None
            assert candidate.approach_type in ["top", "front", "side"]
            assert candidate.approach_distance > 0
            assert candidate.grasp_depth >= 0
            assert 0.0 <= candidate.antipodal_score <= 1.0

    def test_pre_grasp_distance_from_grasp(self, generator):
        """Test that pre-grasp is farther than grasp along approach direction."""
        object_position = (0.0, 0.0, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.3, 0.15, 0.0)

        candidates = generator.generate_candidates(
            object_position, object_rotation, object_size, gripper_position
        )

        for candidate in candidates:
            # Pre-grasp should be farther from object than grasp point
            pre_grasp_dist = np.linalg.norm(
                np.array(candidate.pre_grasp_position) - np.array(object_position)
            )
            grasp_dist = np.linalg.norm(
                np.array(candidate.grasp_position) - np.array(object_position)
            )
            assert pre_grasp_dist > grasp_dist

    def test_reproducibility_with_seed(self, config):
        """Test that same seed produces same candidates."""
        object_position = (0.0, 0.0, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.3, 0.15, 0.0)

        generator1 = GraspCandidateGenerator(config, seed=42)
        candidates1 = generator1.generate_candidates(
            object_position, object_rotation, object_size, gripper_position
        )

        generator2 = GraspCandidateGenerator(config, seed=42)
        candidates2 = generator2.generate_candidates(
            object_position, object_rotation, object_size, gripper_position
        )

        # Positions should match exactly with same seed
        for c1, c2 in zip(candidates1, candidates2):
            assert np.allclose(c1.grasp_position, c2.grasp_position, atol=1e-6)
            assert np.allclose(c1.pre_grasp_position, c2.pre_grasp_position, atol=1e-6)

    def test_different_seeds_produce_different_candidates(self, config):
        """Test that different seeds produce different candidates."""
        object_position = (0.0, 0.0, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.3, 0.15, 0.0)

        generator1 = GraspCandidateGenerator(config, seed=42)
        candidates1 = generator1.generate_candidates(
            object_position, object_rotation, object_size, gripper_position
        )

        generator2 = GraspCandidateGenerator(config, seed=99)
        candidates2 = generator2.generate_candidates(
            object_position, object_rotation, object_size, gripper_position
        )

        # At least some positions should differ
        differences = 0
        for c1, c2 in zip(candidates1, candidates2):
            if not np.allclose(c1.grasp_position, c2.grasp_position, atol=1e-6):
                differences += 1

        assert differences > 0

    def test_object_size_affects_pre_grasp_distance(self, generator):
        """Test that larger objects have larger pre-grasp distances."""
        object_position = (0.0, 0.0, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        gripper_position = (0.3, 0.15, 0.0)

        # Small object
        small_size = (0.02, 0.02, 0.02)
        small_candidates = generator.generate_candidates(
            object_position, object_rotation, small_size, gripper_position
        )

        # Reset generator with same seed
        generator.random = np.random.RandomState(42)

        # Large object
        large_size = (0.10, 0.10, 0.10)
        large_candidates = generator.generate_candidates(
            object_position, object_rotation, large_size, gripper_position
        )

        # Large object should generally have larger pre-grasp distances
        small_avg_dist = np.mean([c.approach_distance for c in small_candidates])
        large_avg_dist = np.mean([c.approach_distance for c in large_candidates])

        assert large_avg_dist > small_avg_dist

    def test_antipodal_score_range(self, generator):
        """Test that antipodal scores are in valid range."""
        object_position = (0.0, 0.0, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.3, 0.15, 0.0)

        candidates = generator.generate_candidates(
            object_position, object_rotation, object_size, gripper_position
        )

        for candidate in candidates:
            assert 0.0 <= candidate.antipodal_score <= 1.0

    def test_top_approach_candidates(self, config):
        """Test that top approach candidates point downward."""
        # Enable only top approach
        config.enabled_approaches = [
            approach for approach in config.enabled_approaches
            if approach.approach_type == "top"
        ]

        generator = GraspCandidateGenerator(config, seed=42)

        object_position = (0.0, 0.0, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.0, 0.15, 0.0)

        candidates = generator.generate_candidates(
            object_position, object_rotation, object_size, gripper_position
        )

        for candidate in candidates:
            assert candidate.approach_type == "top"
            # Pre-grasp should be above grasp point
            assert candidate.pre_grasp_position[1] > candidate.grasp_position[1]

    def test_approach_direction_normalized(self, generator):
        """Test that approach directions are normalized."""
        object_position = (0.0, 0.0, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.3, 0.15, 0.0)

        candidates = generator.generate_candidates(
            object_position, object_rotation, object_size, gripper_position
        )

        for candidate in candidates:
            if candidate.approach_direction is not None:
                magnitude = np.linalg.norm(candidate.approach_direction)
                assert np.isclose(magnitude, 1.0, atol=1e-5)

    def test_grasp_depth_variation(self, generator):
        """Test that grasp depth varies across candidates."""
        object_position = (0.0, 0.0, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.3, 0.15, 0.0)

        candidates = generator.generate_candidates(
            object_position, object_rotation, object_size, gripper_position
        )

        depths = [c.grasp_depth for c in candidates]

        # Should have variation in depths
        assert len(set(depths)) > 1
        assert max(depths) > min(depths)


class TestGraspConfigIntegration:
    """Test generator behavior with different configurations."""

    def test_fast_config_generates_fewer_candidates(self):
        """Test that fast config generates fewer candidates."""
        fast_config = GraspConfig.create_fast()
        default_config = GraspConfig.create_default()

        assert fast_config.candidates_per_approach < default_config.candidates_per_approach

    def test_precise_config_generates_more_candidates(self):
        """Test that precise config generates more candidates."""
        precise_config = GraspConfig.create_precise()
        default_config = GraspConfig.create_default()

        assert precise_config.candidates_per_approach > default_config.candidates_per_approach

    def test_disabled_approach_not_generated(self):
        """Test that disabled approaches are not generated."""
        config = GraspConfig.create_default()

        # Disable front and side approaches
        for approach in config.enabled_approaches:
            if approach.approach_type in ["front", "side"]:
                approach.enabled = False

        generator = GraspCandidateGenerator(config, seed=42)

        object_position = (0.0, 0.0, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.3, 0.15, 0.0)

        candidates = generator.generate_candidates(
            object_position, object_rotation, object_size, gripper_position
        )

        # Should only have top approach candidates
        approach_types = [c.approach_type for c in candidates]
        assert all(approach == "top" for approach in approach_types)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
