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
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np


def _build_segmentation_mask(
    points_camera: "np.ndarray",
    yolo_bbox: tuple,
    image_width: int,
    image_height: int,
    fov: float,
    preferred_approach: str,
    depth_hint: "Optional[float]" = None,
    depth_margin: float = 0.07,
) -> "np.ndarray":
    """Project 3D camera-frame points to 2D and build a boolean mask from a YOLO bbox.

    Projects each point back to pixel coordinates using the pinhole camera
    model (inverse of DepthEstimator.pixel_to_world_coords).  Returns a
    bool mask that is True for points falling inside the bounding box.

    The input ``points_camera`` must be in the Q-matrix output frame:
    (X-right, Y-up, Z-negative).  This is the frame returned by
    ``generate_point_cloud`` — no axis flip is applied in VGNClient before
    calling this function.  Projection formulas:
    ``u = cx + f·X/depth`` and ``v = cy - f·Y/depth`` where ``depth = -Z``.

    When ``preferred_approach`` is "side" the mask is additionally restricted
    to the lateral halves of the object (left/right).  When it is "top" only
    the top third of the bounding box is kept.

    When ``depth_hint`` is provided, an additional depth-range filter keeps
    only points within ``[depth_hint - depth_margin, depth_hint + depth_margin]``
    metres.  This removes background and table-surface points that project into
    the object's 2D footprint but lie at a different depth, reducing TSDF noise.

    Args:
        points_camera:   (N, 3) float32 array in right-handed camera frame.
        yolo_bbox:       (x, y, w, h) pixel bounding box from detect_objects().
        image_width:     Width of the stereo image in pixels.
        image_height:    Height of the stereo image in pixels.
        fov:             Horizontal field-of-view in degrees.
        preferred_approach: "auto", "top", "front", or "side".
        depth_hint:      Optional expected camera-frame depth in metres
                         (positive = in front of camera, equal to ``-Z`` in the
                         Q-matrix frame).  Derived from WorldState object position
                         converted to camera frame.  ``None`` disables depth
                         filtering and preserves prior behaviour.
        depth_margin:    Half-width of the depth acceptance window in metres.
                         Default 0.07 m (±7 cm) covers a 5 cm cube with ±2 cm
                         WorldState position uncertainty.

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

    # Q-matrix output is (X-right, Y-up, Z-negative).
    # VGNClient no longer negates X, so pts_rh has the same frame.
    # Points in front of the camera have Z < 0.
    valid_z = Z < -1e-3

    # Pinhole projection with Y-up, Z-negative frame:
    #   depth = -Z  (positive)
    #   u = cx + f * X / depth       (X-right maps directly to pixel u)
    #   v = cy - f * Y / depth       (Y-up: positive Y is above centre → smaller v)
    depth = np.where(valid_z, -Z, 1.0)  # positive depth
    u = np.where(valid_z, cx + f_px * X / depth, -1.0)
    v = np.where(valid_z, cy - f_px * Y / depth, -1.0)

    # Basic YOLO bounding-box mask
    x0, y0 = float(bx), float(by)
    x1, y1 = x0 + float(bw), y0 + float(bh)
    mask = valid_z & (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)

    # Optional depth-range filter: remove background/table points that project
    # into the object's 2D bbox but lie at a different depth.  `depth` = -Z
    # (positive for points in front of camera) is already computed above.
    # Safety: if the depth filter would leave fewer points than the 2D-only
    # mask, fall back to 2D-only and widen the margin using the actual bbox
    # depth median, so stereo reconstruction errors don't hard-abort the pipeline.
    if depth_hint is not None:
        import logging as _logging

        _log = _logging.getLogger(__name__)
        _n_2d = int(np.count_nonzero(mask))
        if _n_2d > 0:
            _bbox_depths = depth[mask]
            _depth_median = float(np.median(_bbox_depths))
            _log.info(
                f"[mask] bbox 2D points: {_n_2d}, "
                f"depth range=[{_bbox_depths.min():.3f}, {_bbox_depths.max():.3f}] m, "
                f"median={_depth_median:.3f} m | "
                f"hint={depth_hint:.3f} ± {depth_margin:.3f} m"
            )
            # If the actual median depth differs from the hint by more than the
            # margin, the WorldState hint is stale or the stereo is off — use the
            # actual median as the filter centre instead.
            _effective_hint = depth_hint
            if abs(_depth_median - depth_hint) > depth_margin:
                _log.warning(
                    f"[mask] depth_hint {depth_hint:.3f} m differs from bbox median "
                    f"{_depth_median:.3f} m by {abs(_depth_median - depth_hint):.3f} m "
                    f"(> margin {depth_margin:.3f} m) — using bbox median as filter centre"
                )
                _effective_hint = _depth_median
            depth_mask = (
                mask
                & (depth >= _effective_hint - depth_margin)
                & (depth <= _effective_hint + depth_margin)
            )
            _n_depth = int(np.count_nonzero(depth_mask))
            if _n_depth >= max(10, _n_2d // 4):
                # Depth filter kept at least 25% of 2D points — apply it.
                mask = depth_mask
                _log.info(f"[mask] depth filter applied: {_n_2d} → {_n_depth} points")
            else:
                # Too aggressive — skip depth filter entirely to avoid starvation.
                _log.warning(
                    f"[mask] depth filter would leave only {_n_depth}/{_n_2d} points "
                    f"— skipping depth filter"
                )

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
