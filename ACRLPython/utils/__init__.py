#!/usr/bin/env python3
"""
Utility modules for mathematical operations.

This package provides core math utilities for the ACRL project including:
- QuaternionMath: Quaternion operations for 3D rotations
- VectorMath: Vector operations for spatial calculations
- CoordinateTransforms: Coordinate frame transformations
"""

from . import QuaternionMath, VectorMath, CoordinateTransforms

__all__ = ["QuaternionMath", "VectorMath", "CoordinateTransforms"]
