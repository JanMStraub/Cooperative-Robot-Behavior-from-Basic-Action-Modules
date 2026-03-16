#!/usr/bin/env python3
"""
Unit tests for quaternion mathematical operations.

Tests quaternion operations against known values to ensure correctness.
Validates compatibility with Unity's Quaternion system.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.QuaternionMath import (
    quaternion_from_euler,
    euler_from_quaternion,
    quaternion_multiply,
    quaternion_rotate_vector,
    quaternion_angle,
    quaternion_inverse,
    quaternion_normalize,
    quaternion_identity,
    quaternion_from_axis_angle,
)


class TestQuaternionBasics:
    """Test basic quaternion operations."""

    def test_identity_quaternion(self):
        """Test identity quaternion returns (0, 0, 0, 1)."""
        quat = quaternion_identity()
        assert quat == (0.0, 0.0, 0.0, 1.0)

    def test_normalize_unit_quaternion(self):
        """Test normalizing an already unit quaternion."""
        quat = (0.0, 0.0, 0.0, 1.0)
        normalized = quaternion_normalize(quat)
        assert np.allclose(normalized, quat, atol=1e-6)

    def test_normalize_non_unit_quaternion(self):
        """Test normalizing a non-unit quaternion."""
        quat = (1.0, 1.0, 1.0, 1.0)
        normalized = quaternion_normalize(quat)
        magnitude = np.sqrt(sum(x**2 for x in normalized))
        assert np.isclose(magnitude, 1.0, atol=1e-6)


class TestEulerConversions:
    """Test Euler angle to quaternion conversions."""

    def test_zero_rotation(self):
        """Test zero rotation returns identity quaternion."""
        quat = quaternion_from_euler(0.0, 0.0, 0.0)
        assert np.allclose(quat, (0.0, 0.0, 0.0, 1.0), atol=1e-6)

    def test_90_degree_x_rotation(self):
        """Test 90 degree rotation around X-axis."""
        quat = quaternion_from_euler(np.pi / 2, 0.0, 0.0)
        # Expected: sin(45°) on x, cos(45°) on w
        expected_x = np.sin(np.pi / 4)
        expected_w = np.cos(np.pi / 4)
        assert np.isclose(quat[0], expected_x, atol=1e-6)
        assert np.isclose(quat[3], expected_w, atol=1e-6)

    def test_90_degree_y_rotation(self):
        """Test 90 degree rotation around Y-axis."""
        quat = quaternion_from_euler(0.0, np.pi / 2, 0.0)
        expected_y = np.sin(np.pi / 4)
        expected_w = np.cos(np.pi / 4)
        assert np.isclose(quat[1], expected_y, atol=1e-6)
        assert np.isclose(quat[3], expected_w, atol=1e-6)

    def test_90_degree_z_rotation(self):
        """Test 90 degree rotation around Z-axis."""
        quat = quaternion_from_euler(0.0, 0.0, np.pi / 2)
        expected_z = np.sin(np.pi / 4)
        expected_w = np.cos(np.pi / 4)
        assert np.isclose(quat[2], expected_z, atol=1e-6)
        assert np.isclose(quat[3], expected_w, atol=1e-6)

    def test_roundtrip_conversion(self):
        """Test converting euler -> quaternion -> euler."""
        roll, pitch, yaw = 0.5, 0.3, 0.7
        quat = quaternion_from_euler(roll, pitch, yaw)
        roll2, pitch2, yaw2 = euler_from_quaternion(*quat)
        assert np.isclose(roll, roll2, atol=1e-6)
        assert np.isclose(pitch, pitch2, atol=1e-6)
        assert np.isclose(yaw, yaw2, atol=1e-6)


class TestQuaternionMultiplication:
    """Test quaternion multiplication."""

    def test_identity_multiplication(self):
        """Test multiplying by identity quaternion."""
        quat = (0.5, 0.5, 0.5, 0.5)
        identity = quaternion_identity()
        result = quaternion_multiply(quat, identity)
        assert np.allclose(result, quat, atol=1e-6)

    def test_multiplication_order(self):
        """Test that quaternion multiplication is non-commutative."""
        q1 = quaternion_from_euler(np.pi / 4, 0.0, 0.0)
        q2 = quaternion_from_euler(0.0, np.pi / 4, 0.0)

        result1 = quaternion_multiply(q1, q2)
        result2 = quaternion_multiply(q2, q1)

        # Results should be different (non-commutative)
        assert not np.allclose(result1, result2, atol=1e-6)

    def test_inverse_multiplication(self):
        """Test that q * q^(-1) = identity."""
        quat = quaternion_from_euler(0.5, 0.3, 0.7)
        inv_quat = quaternion_inverse(quat)
        result = quaternion_multiply(quat, inv_quat)

        # Should be close to identity
        identity = quaternion_identity()
        assert np.allclose(result, identity, atol=1e-6)


class TestVectorRotation:
    """Test rotating vectors with quaternions."""

    def test_identity_rotation(self):
        """Test rotating vector with identity quaternion."""
        vec = np.array([1.0, 0.0, 0.0])
        identity = quaternion_identity()
        rotated = quaternion_rotate_vector(identity, vec)
        assert np.allclose(rotated, vec, atol=1e-6)

    def test_90_degree_rotation_around_z(self):
        """Test 90 degree rotation around Z-axis."""
        vec = np.array([1.0, 0.0, 0.0])
        quat = quaternion_from_euler(0.0, 0.0, np.pi / 2)
        rotated = quaternion_rotate_vector(quat, vec)

        # X-axis rotated 90° around Z should point to Y-axis
        expected = np.array([0.0, 1.0, 0.0])
        assert np.allclose(rotated, expected, atol=1e-6)

    def test_90_degree_rotation_around_y(self):
        """Test 90 degree rotation around Y-axis."""
        vec = np.array([1.0, 0.0, 0.0])
        quat = quaternion_from_euler(0.0, np.pi / 2, 0.0)
        rotated = quaternion_rotate_vector(quat, vec)

        # X-axis rotated 90° around Y should point to -Z-axis
        expected = np.array([0.0, 0.0, -1.0])
        assert np.allclose(rotated, expected, atol=1e-6)

    def test_180_degree_rotation(self):
        """Test 180 degree rotation."""
        vec = np.array([1.0, 0.0, 0.0])
        quat = quaternion_from_euler(0.0, 0.0, np.pi)
        rotated = quaternion_rotate_vector(quat, vec)

        # X-axis rotated 180° around Z should point to -X-axis
        expected = np.array([-1.0, 0.0, 0.0])
        assert np.allclose(rotated, expected, atol=1e-6)


class TestQuaternionAngle:
    """Test angle calculation between quaternions."""

    def test_same_quaternion_angle(self):
        """Test angle between same quaternion is zero."""
        quat = quaternion_from_euler(0.5, 0.3, 0.7)
        angle = quaternion_angle(quat, quat)
        assert np.isclose(angle, 0.0, atol=1e-6)

    def test_opposite_quaternion_angle(self):
        """Test angle between opposite quaternions."""
        quat1 = (0.0, 0.0, 0.0, 1.0)
        quat2 = (0.0, 0.0, 0.0, -1.0)
        angle = quaternion_angle(quat1, quat2)
        # Opposite quaternions represent same rotation (0 degrees apart)
        assert np.isclose(angle, 0.0, atol=1e-6)

    def test_90_degree_rotation_angle(self):
        """Test angle between quaternions 90 degrees apart."""
        quat1 = quaternion_identity()
        quat2 = quaternion_from_euler(0.0, 0.0, np.pi / 2)
        angle = quaternion_angle(quat1, quat2)
        assert np.isclose(angle, 90.0, atol=1e-3)

    def test_180_degree_rotation_angle(self):
        """Test angle between quaternions 180 degrees apart."""
        quat1 = quaternion_identity()
        quat2 = quaternion_from_euler(0.0, 0.0, np.pi)
        angle = quaternion_angle(quat1, quat2)
        assert np.isclose(angle, 180.0, atol=1e-3)


class TestAxisAngleConversion:
    """Test axis-angle to quaternion conversion."""

    def test_zero_angle(self):
        """Test zero angle rotation."""
        axis = np.array([0.0, 0.0, 1.0])
        quat = quaternion_from_axis_angle(axis, 0.0)
        assert np.allclose(quat, quaternion_identity(), atol=1e-6)

    def test_90_degree_around_z(self):
        """Test 90 degree rotation around Z-axis."""
        axis = np.array([0.0, 0.0, 1.0])
        quat = quaternion_from_axis_angle(axis, np.pi / 2)

        # Compare with euler conversion
        expected = quaternion_from_euler(0.0, 0.0, np.pi / 2)
        assert np.allclose(quat, expected, atol=1e-6)

    def test_arbitrary_axis(self):
        """Test rotation around arbitrary axis."""
        axis = np.array([1.0, 1.0, 1.0])  # Will be normalized
        angle = np.pi / 3
        quat = quaternion_from_axis_angle(axis, angle)

        # Verify it's a unit quaternion
        magnitude = np.sqrt(sum(x**2 for x in quat))
        assert np.isclose(magnitude, 1.0, atol=1e-6)


class TestQuaternionInverse:
    """Test quaternion inverse operation."""

    def test_inverse_of_identity(self):
        """Test inverse of identity is identity."""
        identity = quaternion_identity()
        inv = quaternion_inverse(identity)
        assert np.allclose(inv, identity, atol=1e-6)

    def test_double_inverse(self):
        """Test that (q^-1)^-1 = q."""
        quat = quaternion_from_euler(0.5, 0.3, 0.7)
        inv_inv = quaternion_inverse(quaternion_inverse(quat))
        assert np.allclose(quat, inv_inv, atol=1e-6)

    def test_inverse_rotates_opposite(self):
        """Test that inverse rotates in opposite direction."""
        vec = np.array([1.0, 0.0, 0.0])
        quat = quaternion_from_euler(0.0, 0.0, np.pi / 4)
        inv_quat = quaternion_inverse(quat)

        # Rotate forward then backward should return original
        rotated = quaternion_rotate_vector(quat, vec)
        back = quaternion_rotate_vector(inv_quat, rotated)
        assert np.allclose(back, vec, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
