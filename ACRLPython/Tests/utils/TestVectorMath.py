"""
Test Vector Math Utilities
===========================

Edge-case and correctness tests for utils/VectorMath.py.

Covers:
- vector_angle: zero-length input vectors
- vector_slerp: anti-parallel inputs, boundary values (t=0, t=1)
- vectors_orthonormal_basis: edge-case forward vectors
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from utils.VectorMath import (
    vector_angle,
    vector_slerp,
    vectors_orthonormal_basis,
    vector_normalize,
)


# ---------------------------------------------------------------------------
# vector_angle
# ---------------------------------------------------------------------------

def test_vector_angle_zero_first():
    """Zero-length first vector should return 0.0, not 90° or NaN."""
    result = vector_angle(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert result == 0.0, f"Expected 0.0 for zero first vector, got {result}"


def test_vector_angle_zero_second():
    """Zero-length second vector should return 0.0, not 90° or NaN."""
    result = vector_angle(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    assert result == 0.0, f"Expected 0.0 for zero second vector, got {result}"


def test_vector_angle_both_zero():
    """Both zero-length vectors should return 0.0."""
    result = vector_angle(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    assert result == 0.0, f"Expected 0.0 for both zero vectors, got {result}"


def test_vector_angle_parallel():
    """Parallel vectors should give 0 degrees."""
    result = vector_angle(np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]))
    assert abs(result) < 1e-6, f"Parallel vectors: expected 0°, got {result}"


def test_vector_angle_antiparallel():
    """Anti-parallel vectors should give 180 degrees."""
    result = vector_angle(np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]))
    assert abs(result - 180.0) < 1e-6, f"Anti-parallel: expected 180°, got {result}"


def test_vector_angle_perpendicular():
    """Perpendicular vectors should give 90 degrees."""
    result = vector_angle(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    assert abs(result - 90.0) < 1e-6, f"Perpendicular: expected 90°, got {result}"


def test_vector_angle_no_nan():
    """vector_angle should never produce NaN for any non-degenerate input."""
    result = vector_angle(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    assert not np.isnan(result), "vector_angle produced NaN"


# ---------------------------------------------------------------------------
# vector_slerp: anti-parallel case
# ---------------------------------------------------------------------------

def test_slerp_antiparallel_t0():
    """Slerp between anti-parallel vectors at t=0 should return v1."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([-1.0, 0.0, 0.0])
    result = vector_slerp(v1, v2, 0.0)
    assert np.allclose(result, v1, atol=1e-6), f"t=0 anti-parallel: expected {v1}, got {result}"


def test_slerp_antiparallel_t1():
    """Slerp between anti-parallel vectors at t=1 should return v2."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([-1.0, 0.0, 0.0])
    result = vector_slerp(v1, v2, 1.0)
    assert np.allclose(result, v2, atol=1e-6), f"t=1 anti-parallel: expected {v2}, got {result}"


def test_slerp_antiparallel_no_nan():
    """Slerp between anti-parallel vectors should not produce NaN at any t."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([-1.0, 0.0, 0.0])
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = vector_slerp(v1, v2, t)
        assert not np.any(np.isnan(result)), f"NaN at t={t}: {result}"


def test_slerp_antiparallel_unit_magnitude():
    """Slerp between unit anti-parallel vectors should preserve unit magnitude at t=0.5."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([-1.0, 0.0, 0.0])
    result = vector_slerp(v1, v2, 0.5)
    mag = np.linalg.norm(result)
    assert abs(mag - 1.0) < 1e-6, f"t=0.5 anti-parallel magnitude: expected 1.0, got {mag}"


# ---------------------------------------------------------------------------
# vector_slerp: boundary values (t=0, t=1)
# ---------------------------------------------------------------------------

def test_slerp_t0_returns_v1():
    """Slerp at t=0 should always return v1."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    result = vector_slerp(v1, v2, 0.0)
    assert np.allclose(result, v1, atol=1e-6), f"t=0: expected {v1}, got {result}"


def test_slerp_t1_returns_v2():
    """Slerp at t=1 should always return v2."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    result = vector_slerp(v1, v2, 1.0)
    assert np.allclose(result, v2, atol=1e-6), f"t=1: expected {v2}, got {result}"


def test_slerp_parallel_no_nan():
    """Slerp between parallel vectors should not produce NaN."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    result = vector_slerp(v1, v2, 0.5)
    assert not np.any(np.isnan(result)), f"NaN for parallel slerp: {result}"


def test_slerp_midpoint_perpendicular():
    """Slerp at t=0.5 between X and Y axes should give 45-degree direction."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    result = vector_slerp(v1, v2, 0.5)
    result_norm = vector_normalize(result)
    expected = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    assert np.allclose(result_norm, expected, atol=1e-6), (
        f"Midpoint slerp: expected {expected}, got {result_norm}"
    )


# ---------------------------------------------------------------------------
# vectors_orthonormal_basis
# ---------------------------------------------------------------------------

def test_orthonormal_basis_standard():
    """Standard forward vector should produce orthonormal basis."""
    fwd, right, up = vectors_orthonormal_basis(np.array([0.0, 0.0, 1.0]))
    assert abs(np.dot(fwd, right)) < 1e-10, "fwd and right not orthogonal"
    assert abs(np.dot(fwd, up)) < 1e-10, "fwd and up not orthogonal"
    assert abs(np.dot(right, up)) < 1e-10, "right and up not orthogonal"
    assert abs(np.linalg.norm(fwd) - 1.0) < 1e-10, "fwd not unit length"
    assert abs(np.linalg.norm(right) - 1.0) < 1e-10, "right not unit length"
    assert abs(np.linalg.norm(up) - 1.0) < 1e-10, "up not unit length"


def test_orthonormal_basis_up_forward():
    """Y-up forward vector should switch to X reference to avoid singularity."""
    fwd, right, up = vectors_orthonormal_basis(np.array([0.0, 1.0, 0.0]))
    assert abs(np.dot(fwd, right)) < 1e-10, "fwd and right not orthogonal"
    assert abs(np.dot(fwd, up)) < 1e-10, "fwd and up not orthogonal"
    assert abs(np.dot(right, up)) < 1e-10, "right and up not orthogonal"


def test_orthonormal_basis_diagonal():
    """Diagonal forward vector should produce valid orthonormal basis."""
    fwd, right, up = vectors_orthonormal_basis(np.array([1.0, 1.0, 1.0]))
    assert abs(np.dot(fwd, right)) < 1e-10, "fwd and right not orthogonal"
    assert abs(np.dot(fwd, up)) < 1e-10, "fwd and up not orthogonal"
    assert abs(np.dot(right, up)) < 1e-10, "right and up not orthogonal"
    # fwd should be normalized
    assert abs(np.linalg.norm(fwd) - 1.0) < 1e-10, "fwd not unit length"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
