import pytest
import numpy as np

from grasp_planning.GraspCandidateGenerator import GraspCandidateGenerator
from grasp_planning.GraspConfig import GraspConfig


class TestGraspCandidateGenerator:

    @pytest.fixture
    def config(self):
        return GraspConfig.create_default()

    @pytest.fixture
    def generator(self, config):
        return GraspCandidateGenerator(config, seed=42)

    def test_generator_initialization(self, config):
        generator = GraspCandidateGenerator(config, seed=42)
        assert generator.config == config
        assert generator.random is not None

    def test_generates_correct_number_of_candidates(self, generator):
        object_position = (0.0, 0.0, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.3, 0.15, 0.0)

        candidates = generator.generate_candidates(
            object_position, object_rotation, object_size, gripper_position
        )

        expected_count = 8 * 3
        assert len(candidates) == expected_count

    def test_candidates_have_required_fields(self, generator):
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
            pre_grasp_dist = np.linalg.norm(
                np.array(candidate.pre_grasp_position) - np.array(object_position)
            )
            grasp_dist = np.linalg.norm(
                np.array(candidate.grasp_position) - np.array(object_position)
            )
            assert pre_grasp_dist > grasp_dist

    def test_reproducibility_with_seed(self, config):
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

        for c1, c2 in zip(candidates1, candidates2):
            assert np.allclose(c1.grasp_position, c2.grasp_position, atol=1e-6)
            assert np.allclose(c1.pre_grasp_position, c2.pre_grasp_position, atol=1e-6)

    def test_different_seeds_produce_different_candidates(self, config):
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

        small_avg_dist = np.mean([c.approach_distance for c in small_candidates])
        large_avg_dist = np.mean([c.approach_distance for c in large_candidates])

        assert large_avg_dist > small_avg_dist

    def test_antipodal_score_range(self, generator):
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
        # Enable only top approach
        config.enabled_approaches = [
            approach
            for approach in config.enabled_approaches
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
            assert candidate.pre_grasp_position[1] > candidate.grasp_position[1]

    def test_approach_direction_normalized(self, generator):
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
        object_position = (0.0, 0.0, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.3, 0.15, 0.0)

        candidates = generator.generate_candidates(
            object_position, object_rotation, object_size, gripper_position
        )

        depths = [c.grasp_depth for c in candidates]

        assert len(set(depths)) > 1
        assert max(depths) > min(depths)


class TestGraspConfigIntegration:

    def test_fast_config_generates_fewer_candidates(self):
        fast_config = GraspConfig.create_fast()
        default_config = GraspConfig.create_default()

        assert (
            fast_config.candidates_per_approach < default_config.candidates_per_approach
        )

    def test_precise_config_generates_more_candidates(self):
        precise_config = GraspConfig.create_precise()
        default_config = GraspConfig.create_default()

        assert (
            precise_config.candidates_per_approach
            > default_config.candidates_per_approach
        )

    def test_disabled_approach_not_generated(self):
        config = GraspConfig.create_default()

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

        approach_types = [c.approach_type for c in candidates]
        assert all(approach == "top" for approach in approach_types)


class TestApproachDirectionDiversity:
    """
    Tests for Bug 2 fix: side/front approach diversity.

    Before the fix, _get_approach_basis was called once per approach batch, so
    side_sign/front_sign were drawn once and all 8 candidates used the same
    direction (all +X or all -X for side; all +Z or all -Z for front).

    After the fix, _get_approach_basis is called per candidate, so each
    candidate independently samples its sign - giving mix of +X/-X and +Z/-Z.
    """

    def test_side_approach_has_both_positive_and_negative_x_directions(self):
        """
        Side approach candidates must include both +X and -X approach directions
        across enough trials (multiple seeds) to confirm independence.

        We run several seeded generators and check that at least one produces
        a mix - if all seeds produce only one direction, the bug has returned.
        """
        config = GraspConfig.create_default()
        # Keep only side approach so we only look at side candidates
        config.enabled_approaches = [
            a for a in config.enabled_approaches if a.approach_type == "side"
        ]
        # Use more candidates to increase probability of observing both signs
        config.candidates_per_approach = 16

        object_position = (0.0, 0.0, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.3, 0.15, 0.0)

        found_mix_in_any_seed = False
        for seed in range(20):
            gen = GraspCandidateGenerator(config, seed=seed)
            candidates = gen.generate_candidates(
                object_position, object_rotation, object_size, gripper_position
            )

            # Extract the X component of each approach direction
            x_directions = [c.approach_direction[0] for c in candidates]  # type: ignore[index]
            has_positive_x = any(x > 0.5 for x in x_directions)
            has_negative_x = any(x < -0.5 for x in x_directions)

            if has_positive_x and has_negative_x:
                found_mix_in_any_seed = True
                break

        assert found_mix_in_any_seed, (
            "No seed produced a mix of +X and -X side approach directions across "
            "20 seeds with 16 candidates each. This indicates all candidates within "
            "a batch share the same direction (Bug 2 has returned)."
        )

    def test_front_approach_has_both_positive_and_negative_z_directions(self):
        """
        Front approach candidates must include both +Z and -Z approach directions
        across enough trials to confirm independence.
        """
        config = GraspConfig.create_default()
        config.enabled_approaches = [
            a for a in config.enabled_approaches if a.approach_type == "front"
        ]
        config.candidates_per_approach = 16

        object_position = (0.0, 0.0, 0.0)
        object_rotation = (0.0, 0.0, 0.0, 1.0)
        object_size = (0.05, 0.05, 0.05)
        gripper_position = (0.3, 0.15, 0.0)

        found_mix_in_any_seed = False
        for seed in range(20):
            gen = GraspCandidateGenerator(config, seed=seed)
            candidates = gen.generate_candidates(
                object_position, object_rotation, object_size, gripper_position
            )

            z_directions = [c.approach_direction[2] for c in candidates]  # type: ignore[index]
            has_positive_z = any(z > 0.5 for z in z_directions)
            has_negative_z = any(z < -0.5 for z in z_directions)

            if has_positive_z and has_negative_z:
                found_mix_in_any_seed = True
                break

        assert found_mix_in_any_seed, (
            "No seed produced a mix of +Z and -Z front approach directions across "
            "20 seeds with 16 candidates each. This indicates all candidates within "
            "a batch share the same direction (Bug 2 has returned)."
        )

    def test_side_approach_both_directions_in_large_sample(self):
        """
        With a fixed seed and enough candidates, we must see both +X and -X
        directions, since each candidate independently draws its sign.

        Expected number of +X candidates ~ Binomial(n=32, p=0.5) - probability
        of all-same sign is (0.5)^31 ≈ 5e-10, so this test is essentially deterministic.
        """
        config = GraspConfig.create_default()
        config.enabled_approaches = [
            a for a in config.enabled_approaches if a.approach_type == "side"
        ]
        config.candidates_per_approach = 32

        gen = GraspCandidateGenerator(config, seed=0)
        candidates = gen.generate_candidates(
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.05, 0.05, 0.05), (0.3, 0.15, 0.0)
        )

        x_directions = [c.approach_direction[0] for c in candidates]  # type: ignore[index]
        assert any(x > 0.5 for x in x_directions), "No +X side approach candidate found"
        assert any(
            x < -0.5 for x in x_directions
        ), "No -X side approach candidate found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
