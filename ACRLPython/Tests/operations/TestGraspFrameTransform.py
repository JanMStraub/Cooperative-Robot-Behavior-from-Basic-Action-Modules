#!/usr/bin/env python3
"""
Unit tests for GraspFrameTransform.py

Tests the RH camera-frame → LH Unity world-frame transformation for
Contact-GraspNet grasp poses.  All tests are pure math — no Unity or
network access required.

Key invariants verified:
- Identity camera pose: position and rotation round-trip correctly
- X-axis negation (handedness flip) is applied to position
- Rotations are composed correctly (cam_rot ⊗ grasp_rot)
- Malformed grasps are silently skipped
- Approach direction is derived correctly from quaternion when not provided
"""

import math
import pytest
import numpy as np

from operations.GraspFrameTransform import (
    transform_graspnet_poses_to_unity,
    _quat_multiply,
    _quat_rotate_vector,
    _normalise_quat,
)


# ============================================================================
# Internal quaternion math tests
# ============================================================================


class TestQuaternionMath:
    """Verify the internal NumPy quaternion helpers are correct."""

    def test_identity_rotation_leaves_vector_unchanged(self):
        """Rotating by identity quaternion must return the original vector."""
        q_id = np.array([0.0, 0.0, 0.0, 1.0])
        v = np.array([1.0, 2.0, 3.0])
        result = _quat_rotate_vector(q_id, v)
        np.testing.assert_allclose(result, v, atol=1e-9)

    def test_180_rotation_around_y_negates_xz(self):
        """180° around Y maps (x,y,z) → (-x,y,-z)."""
        q_180y = np.array([0.0, 1.0, 0.0, 0.0])  # [sin(90°)*Y, cos(90°)] already normalised
        v = np.array([1.0, 0.0, 1.0])
        result = _quat_rotate_vector(q_180y, v)
        np.testing.assert_allclose(result, np.array([-1.0, 0.0, -1.0]), atol=1e-9)

    def test_identity_times_q_equals_q(self):
        """q_id ⊗ q = q for any unit quaternion."""
        q_id = np.array([0.0, 0.0, 0.0, 1.0])
        q = _normalise_quat(np.array([0.1, 0.2, 0.3, 0.9]))
        result = _normalise_quat(_quat_multiply(q_id, q))
        np.testing.assert_allclose(result, q, atol=1e-9)

    def test_normalise_handles_near_zero(self):
        """Near-zero quaternion normalise returns identity, not NaN."""
        q = np.array([0.0, 0.0, 0.0, 0.0])
        result = _normalise_quat(q)
        np.testing.assert_allclose(result, np.array([0.0, 0.0, 0.0, 1.0]))


# ============================================================================
# Frame transform tests
# ============================================================================


class TestTransformGraspnetPosesToUnity:
    """Tests for the public transform API."""

    def test_identity_camera_identity_grasp_maps_correctly(self):
        """With identity camera pose, a grasp at (1,2,3) maps to (-1,2,3) in Unity."""
        cam_pos = [0.0, 0.0, 0.0]
        cam_rot = [0.0, 0.0, 0.0, 1.0]  # identity

        grasps = [
            {
                "position": [1.0, 2.0, 3.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],  # identity
                "score": 0.9,
                "width": 0.05,
            }
        ]

        result = transform_graspnet_poses_to_unity(grasps, cam_pos, cam_rot)

        assert len(result) == 1
        pos = result[0]["position"]
        # X negated (RH→LH flip), Y and Z unchanged, camera at origin so no translation
        np.testing.assert_allclose(pos, [-1.0, 2.0, 3.0], atol=1e-6)

    def test_camera_translation_is_added(self):
        """Camera world position is added after rotating the flipped grasp position."""
        cam_pos = [0.5, 1.0, 2.0]
        cam_rot = [0.0, 0.0, 0.0, 1.0]  # identity — no rotation

        grasps = [
            {
                "position": [0.0, 0.0, 0.0],  # at camera origin in camera frame
                "rotation": [0.0, 0.0, 0.0, 1.0],
            }
        ]

        result = transform_graspnet_poses_to_unity(grasps, cam_pos, cam_rot)

        assert len(result) == 1
        pos = result[0]["position"]
        # Origin flipped = still origin; then cam_pos is added
        np.testing.assert_allclose(pos, [0.5, 1.0, 2.0], atol=1e-6)

    def test_camera_rotation_applied_to_position(self):
        """Camera 180° rotation around Y swaps X/Z signs of transformed position."""
        cam_pos = [0.0, 0.0, 0.0]
        # 180° around Y: quaternion [0, sin(90°), 0, cos(90°)] = [0,1,0,0]
        cam_rot = [0.0, 1.0, 0.0, 0.0]

        grasps = [
            {
                "position": [1.0, 0.0, 1.0],  # camera frame
                "rotation": [0.0, 0.0, 0.0, 1.0],
            }
        ]

        result = transform_graspnet_poses_to_unity(grasps, cam_pos, cam_rot)

        assert len(result) == 1
        pos = result[0]["position"]
        # Step 1: flip X → (-1, 0, 1)
        # Step 2: rotate by 180° around Y → (1, 0, -1)
        # (No translation)
        np.testing.assert_allclose(pos, [1.0, 0.0, -1.0], atol=1e-6)

    def test_grasp_pointing_down_in_camera_is_down_in_world(self):
        """A grasp approach pointing straight down (-Y) in camera frame should
        remain pointing down in world when the camera faces +Z (no tilt)."""
        cam_pos = [0.0, 0.0, 0.0]
        cam_rot = [0.0, 0.0, 0.0, 1.0]  # identity camera

        # Build a quaternion that rotates Z-forward → -Y (pointing down)
        # That is a -90° rotation around X: [sin(-45°),0,0,cos(-45°)]
        angle = -math.pi / 2
        q = [math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)]  # [x,y,z,w]

        grasps = [
            {
                "position": [0.0, 0.0, 0.5],
                "rotation": q,
            }
        ]

        result = transform_graspnet_poses_to_unity(grasps, cam_pos, cam_rot)

        assert len(result) == 1
        approach = result[0]["approach_direction"]
        # Approach should be near (0, -1, 0) in world — pointing down
        np.testing.assert_allclose(
            approach, [0.0, -1.0, 0.0], atol=1e-5
        )

    def test_score_and_width_preserved(self):
        """score and width fields from GraspNet are preserved in the output."""
        cam_pos = [0.0, 0.0, 0.0]
        cam_rot = [0.0, 0.0, 0.0, 1.0]

        grasps = [
            {
                "position": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "score": 0.87,
                "width": 0.042,
            }
        ]

        result = transform_graspnet_poses_to_unity(grasps, cam_pos, cam_rot)

        assert len(result) == 1
        assert result[0]["score"] == pytest.approx(0.87)
        assert result[0]["width"] == pytest.approx(0.042)

    def test_malformed_grasp_silently_skipped(self):
        """A grasp with missing 'position' key is skipped, not raising an exception."""
        cam_pos = [0.0, 0.0, 0.0]
        cam_rot = [0.0, 0.0, 0.0, 1.0]

        grasps = [
            {"rotation": [0.0, 0.0, 0.0, 1.0], "score": 0.5},  # missing position
            {
                "position": [1.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "score": 0.9,
            },
        ]

        result = transform_graspnet_poses_to_unity(grasps, cam_pos, cam_rot)

        # Only the valid grasp should be in the output
        assert len(result) == 1
        assert result[0]["score"] == pytest.approx(0.9)

    def test_empty_input_returns_empty_list(self):
        """Empty grasp list returns empty output without error."""
        result = transform_graspnet_poses_to_unity([], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        assert result == []

    def test_multiple_grasps_all_transformed(self):
        """All grasps in the list are transformed, preserving order."""
        cam_pos = [0.0, 0.0, 0.0]
        cam_rot = [0.0, 0.0, 0.0, 1.0]

        grasps = [
            {"position": [1.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0], "score": 0.1},
            {"position": [2.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0], "score": 0.2},
            {"position": [3.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0], "score": 0.3},
        ]

        result = transform_graspnet_poses_to_unity(grasps, cam_pos, cam_rot)

        assert len(result) == 3
        # X-flip: [1,0,0] → [-1,0,0], [2,0,0] → [-2,0,0], etc.
        np.testing.assert_allclose(result[0]["position"], [-1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(result[1]["position"], [-2.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(result[2]["position"], [-3.0, 0.0, 0.0], atol=1e-6)

    def test_approach_direction_from_provided_field(self):
        """When approach_direction is provided by GraspNet it is transformed too."""
        cam_pos = [0.0, 0.0, 0.0]
        cam_rot = [0.0, 0.0, 0.0, 1.0]  # identity

        grasps = [
            {
                "position": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "approach_direction": [1.0, 0.0, 0.0],  # pointing +X in camera
            }
        ]

        result = transform_graspnet_poses_to_unity(grasps, cam_pos, cam_rot)

        assert len(result) == 1
        approach = result[0]["approach_direction"]
        # X is negated by handedness flip; identity camera rotation leaves the rest
        np.testing.assert_allclose(approach, [-1.0, 0.0, 0.0], atol=1e-6)
