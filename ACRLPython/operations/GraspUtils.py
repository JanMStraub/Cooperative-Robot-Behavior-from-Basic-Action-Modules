#!/usr/bin/env python3
"""
Grasp Utility Helpers
=====================

Shared helpers used by both ``GraspOperations`` and ``VGNClient``.  Placing
them here breaks the circular import that would arise if VGNClient imported
from GraspOperations directly.

Currently contains:
    - ``_build_segmentation_mask`` — project 3D camera points to 2D and mask
      to a YOLO bounding box.
"""

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def _build_segmentation_mask(
    points_camera: "np.ndarray",
    yolo_bbox: tuple,
    image_width: int,
    image_height: int,
    fov: float,
    preferred_approach: str,
) -> "np.ndarray":
    """Project 3D camera-frame points to 2D and build a boolean mask from a YOLO bbox.

    Projects each point back to pixel coordinates using the pinhole camera
    model (inverse of DepthEstimator.pixel_to_world_coords).  Returns a
    bool mask that is True for points falling inside the bounding box.

    The input ``points_camera`` must be in **right-handed camera frame** (i.e.
    the Unity left-handed X-negation must have been undone by the caller before
    passing points here).  Projection formulas follow OpenCV convention:
    ``u = cx + f·(-X)/Z`` (negate X to undo Unity LH→RH flip) and
    ``v = cy - f·Y/Z`` (negate Y for Unity Y-up vs OpenCV Y-down).

    When ``preferred_approach`` is "side" the mask is additionally restricted
    to the lateral halves of the object (left/right).  When it is "top" only
    the top third of the bounding box is kept.

    Args:
        points_camera:   (N, 3) float32 array in right-handed camera frame.
        yolo_bbox:       (x, y, w, h) pixel bounding box from detect_objects().
        image_width:     Width of the stereo image in pixels.
        image_height:    Height of the stereo image in pixels.
        fov:             Horizontal field-of-view in degrees.
        preferred_approach: "auto", "top", "front", or "side".

    Returns:
        Boolean ndarray of shape (N,) — True for points to include.
    """
    import numpy as np

    N = points_camera.shape[0]
    if N == 0:
        return np.zeros(N, dtype=bool)

    bx, by, bw, bh = yolo_bbox
    if bw <= 0 or bh <= 0:
        # Degenerate bbox — return all points
        return np.ones(N, dtype=bool)

    # Pinhole focal length in pixels from horizontal FOV
    f_px = (image_width / 2.0) / math.tan(math.radians(fov / 2.0))

    cx = image_width / 2.0
    cy = image_height / 2.0

    X = points_camera[:, 0]
    Y = points_camera[:, 1]
    Z = points_camera[:, 2]

    # Points in front of camera have negative Z (Unity camera looks in -Z direction).
    # After the caller's X-negation (LH→RH flip), Z remains negative for forward points.
    valid_z = Z < -1e-3

    # Projection for Unity-derived points (camera looks in -Z, Y-up, X-right):
    #   u = cx + f * X / (-Z)   (divide by positive depth = -Z)
    #   v = cy - f * Y / (-Z)   (negate Y: Unity Y-up vs image Y-down)
    depth = np.where(valid_z, -Z, 1.0)  # positive depth
    u = np.where(valid_z, cx + f_px * X / depth, -1.0)
    v = np.where(valid_z, cy - f_px * Y / depth, -1.0)

    # Basic YOLO bounding-box mask
    x0, y0 = float(bx), float(by)
    x1, y1 = x0 + float(bw), y0 + float(bh)
    mask = valid_z & (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)

    approach = preferred_approach.lower()
    if approach == "side":
        # Keep only left/right halves — exclude centre 50 % of bbox width
        obj_cx = (x0 + x1) / 2.0
        half_w = float(bw) * 0.25  # 25 % from each edge
        side_mask = (u <= (obj_cx - half_w)) | (u >= (obj_cx + half_w))
        mask = mask & side_mask
    elif approach == "top":
        # Keep only top third of bbox
        y_thresh = y0 + float(bh) / 3.0
        mask = mask & (v <= y_thresh)

    return mask
