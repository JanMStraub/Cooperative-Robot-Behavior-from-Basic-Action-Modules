#!/usr/bin/env python3
"""
Detection Operations
====================

All previously registered detection operations have been removed:
- detect_objects: Removed (2D pixel-only; detect_object_stereo supersedes)
- estimate_distance_to_object: Removed (marginal; WorldState-dependent)
- estimate_distance_between_objects: Removed (marginal; WorldState-dependent)

Use detect_object_stereo (VisionOperations) for 3D object detection.
"""
