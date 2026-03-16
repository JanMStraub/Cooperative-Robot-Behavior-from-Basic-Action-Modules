#!/usr/bin/env python3
"""
StereoReconstruction.py - Stereo point cloud reconstruction

Provides point cloud reconstruction from stereo image pairs using
StereoSGBM disparity estimation.

Functions
---------
stereo_reconstruct(imgL, imgR, ...)
    Reconstruct from two in-memory images (offline / test use).
stereo_reconstruct_stream(imgL, imgR, ...)
    Same as above, but intended for live streaming — no disk I/O.
reconstruct_from_storage(camera_pair_id=None)
    Pull the latest stereo pair from UnifiedImageStorage and reconstruct.

CLI usage (offline test with image files)
------------------------------------------
    cd ACRLPython
    python -m vision.StereoReconstruction \\
        --l path/to/left.png --r path/to/right.png \\
        --fov 60 --cam_dist 0.05

Live test (requires Unity to be streaming stereo pairs)
-------------------------------------------------------
    python -m vision.StereoReconstruction --live
"""

import argparse
import math
import cv2
import numpy as np
from typing import Optional, Dict

from core.LoggingSetup import get_logger

logger = get_logger(__name__)

# Import camera defaults from project config so values stay in sync
# with what Unity sends (baseline, FOV).
try:
    from config.Vision import (
        DEFAULT_STEREO_BASELINE,
        DEFAULT_STEREO_FOV,
    )
except ImportError:
    from ..config.Vision import (
        DEFAULT_STEREO_BASELINE,
        DEFAULT_STEREO_FOV,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _f_from_fov(fov: float) -> float:
    """
    Convert a horizontal field-of-view angle to a normalised focal length.

    Args:
        fov: Horizontal field of view in degrees.

    Returns:
        Normalised focal length (focal_length / sensor_width).
    """
    return 1.0 / (2.0 * math.tan(math.radians(fov / 2.0)))


def _calc_disparity(
    imgL: np.ndarray, imgR: np.ndarray, max_disp: Optional[int] = None
) -> np.ndarray:
    """
    Compute a disparity map between two rectified grayscale images using StereoSGBM.

    Args:
        imgL: Grayscale left image (single-channel).
        imgR: Grayscale right image (single-channel).
        max_disp: Maximum disparity (rounded up to next multiple of 16).
                  Defaults to 128 if not provided.

    Returns:
        Float32 disparity map; invalid pixels are set to NaN.

    Raises:
        ValueError: If images are not single-channel.
    """
    if imgL.ndim != 2 or imgR.ndim != 2:
        raise ValueError(
            "_calc_disparity expects single-channel (grayscale) images. "
            "Convert with cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) first."
        )

    window_size = 5

    if max_disp is None:
        max_disp = 128

    # StereoSGBM requires numDisparities to be divisible by 16.
    if max_disp % 16 != 0:
        max_disp += 16 - (max_disp % 16)

    logger.debug(f"StereoSGBM: max_disp={max_disp}, window_size={window_size}")

    # Tuned parameters for robotic manipulation scenes, including low-texture
    # synthetic (Unity) environments where strict uniqueness filtering would
    # reject most matches on flat/uniform surfaces.
    stereo = cv2.StereoSGBM_create(  # type: ignore
        minDisparity=0,
        numDisparities=max_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size**2,  # OpenCV recommended formula
        P2=32 * 3 * window_size**2,  # OpenCV recommended formula
        disp12MaxDiff=2,  # Relaxed L-R consistency check (was 1)
        uniquenessRatio=0,  # Disabled; synthetic scenes have near-identical scores
        speckleWindowSize=0,  # Disabled speckle filter; flat regions produce large coherent blobs
        speckleRange=0,
        preFilterCap=63,  # Reduce pre-filter aggressiveness for clean synthetic images
        mode=cv2.STEREO_SGBM_MODE_HH,  # Full HH scan; better disparity coverage on flat surfaces
    )
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    return np.where(disp >= 0.0, disp, np.nan)


def _make_3d(
    depth_map: np.ndarray,
    imgL_color: np.ndarray,
    focal_length: float,
    min_disp: float = 0.0,
    cam_dist: float = 0.1,
) -> Dict[str, np.ndarray]:
    """
    Reproject a disparity map into a coloured 3-D point cloud.

    Uses OpenCV's reprojectImageTo3D with a Q-matrix constructed from the
    camera parameters.  The X axis is negated to match Unity's left-handed
    coordinate system.

    Args:
        depth_map: Float32 disparity map (pixels with value <= 0 are ignored).
        imgL_color: BGR colour image used to colour each point.
        focal_length: Pixel focal length (normalised_f * image_width).
        min_disp: Minimum disparity threshold; points below this are discarded.
        cam_dist: Stereo baseline in metres.

    Returns:
        Dict with keys ``'points'`` (Nx3 float32) and ``'colors'`` (Nx3 uint8, RGB).
    """
    depth_map = np.nan_to_num(depth_map, nan=-1.0).astype(np.float32)

    h, w = imgL_color.shape[:2]
    Q = np.array(
        [
            [-1, 0, 0, 0.5 * w],
            [0, 1, 0, -0.5 * h],
            [0, 0, 0, focal_length],
            [0, 0, -1.0 / cam_dist, 0],
        ],
        dtype=np.float32,
    )
    points = cv2.reprojectImageTo3D(depth_map, Q)
    colors = cv2.cvtColor(imgL_color, cv2.COLOR_BGR2RGB)
    mask = depth_map > min_disp
    return {"points": points[mask], "colors": colors[mask]}


def _reconstruct_from_disparity(
    disparity: np.ndarray,
    imgL_color: np.ndarray,
    fov: Optional[float] = None,
    focal_length: Optional[float] = None,
    sensor_width: Optional[float] = None,
    min_disp: float = 0.0,
    cam_dist: float = DEFAULT_STEREO_BASELINE,
) -> Dict[str, np.ndarray]:
    """
    Build a point cloud from a pre-computed disparity map and colour image.

    Args:
        disparity: Float32 disparity map from _calc_disparity.
        imgL_color: BGR colour reference image.
        fov: Horizontal FOV in degrees.  If given, overrides focal_length/sensor_width.
        focal_length: Camera focal length (metres or pixels, combined with sensor_width).
        sensor_width: Camera sensor width (same units as focal_length).
        min_disp: Minimum disparity; lower values are discarded.
        cam_dist: Stereo baseline in metres.

    Returns:
        Dict with ``'points'`` (Nx3) and ``'colors'`` (Nx3) arrays.

    Raises:
        ValueError: If neither fov nor (focal_length + sensor_width) is provided.
    """
    if fov is not None:
        f_norm = _f_from_fov(fov)
    elif focal_length is not None and sensor_width is not None:
        f_norm = focal_length / sensor_width
    else:
        raise ValueError("Provide either fov or both focal_length and sensor_width.")

    f_px = f_norm * disparity.shape[1]
    return _make_3d(disparity, imgL_color, f_px, min_disp=min_disp, cam_dist=cam_dist)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def stereo_reconstruct(
    imgL: np.ndarray,
    imgR: np.ndarray,
    fov: Optional[float] = None,
    focal_length: Optional[float] = None,
    sensor_width: Optional[float] = None,
    min_disp: float = 0.0,
    max_disp: Optional[int] = None,
    cam_dist: float = DEFAULT_STEREO_BASELINE,
    mask_edges: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Reconstruct a point cloud from two BGR stereo images.

    Intended for offline use (files on disk).  For live Unity data use
    ``reconstruct_from_storage`` instead.

    Args:
        imgL: Left camera image (BGR).
        imgR: Right camera image (BGR).
        fov: Horizontal FOV in degrees.
        focal_length: Camera focal length (if fov not given).
        sensor_width: Sensor width (if fov not given).
        min_disp: Minimum disparity threshold.
        max_disp: Maximum disparity (defaults to 128).
        cam_dist: Stereo baseline in metres (default: ``DEFAULT_STEREO_BASELINE``).
        mask_edges: If True, suppress strong depth edges to reduce noise.

    Returns:
        Dict with ``'points'`` (Nx3 float32) and ``'colors'`` (Nx3 uint8, RGB).
    """
    imgL_color = imgL.copy()
    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    logger.info("Computing disparity map…")
    disparity = _calc_disparity(imgL_gray, imgR_gray, max_disp=max_disp)

    if mask_edges:
        disparity = _remove_edges(disparity)

    logger.info("Generating 3-D point cloud…")
    point_cloud = _reconstruct_from_disparity(
        disparity,
        imgL_color,
        fov=fov,
        focal_length=focal_length,
        sensor_width=sensor_width,
        min_disp=min_disp,
        cam_dist=cam_dist,
    )
    logger.info(f"Point cloud: {len(point_cloud['points'])} points")
    return point_cloud


def stereo_reconstruct_stream(
    imgL: np.ndarray,
    imgR: np.ndarray,
    fov: Optional[float] = None,
    focal_length: Optional[float] = None,
    sensor_width: Optional[float] = None,
    min_disp: float = 0.0,
    max_disp: Optional[int] = None,
    cam_dist: float = DEFAULT_STEREO_BASELINE,
) -> Dict[str, np.ndarray]:
    """
    Reconstruct a point cloud from two BGR stereo images received via streaming.

    Identical to ``stereo_reconstruct`` but omits edge masking (not suitable
    for real-time use) and does not write anything to disk.

    Args:
        imgL: Left camera image (BGR).
        imgR: Right camera image (BGR).
        fov: Horizontal FOV in degrees.
        focal_length: Camera focal length (if fov not given).
        sensor_width: Sensor width (if fov not given).
        min_disp: Minimum disparity threshold.
        max_disp: Maximum disparity (defaults to 128).
        cam_dist: Stereo baseline in metres (default: ``DEFAULT_STEREO_BASELINE``).

    Returns:
        Dict with ``'points'`` (Nx3 float32) and ``'colors'`` (Nx3 uint8, RGB).
    """
    imgL_color = imgL.copy()
    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    disparity = _calc_disparity(imgL_gray, imgR_gray, max_disp=max_disp)
    return _reconstruct_from_disparity(
        disparity,
        imgL_color,
        fov=fov,
        focal_length=focal_length,
        sensor_width=sensor_width,
        min_disp=min_disp,
        cam_dist=cam_dist,
    )


def reconstruct_from_storage(
    camera_pair_id: Optional[str] = None,
    fov: Optional[float] = None,
    cam_dist: Optional[float] = None,
    min_disp: float = 0.0,
    max_disp: Optional[int] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Pull the latest stereo pair from UnifiedImageStorage and reconstruct.

    Camera parameters (fov, cam_dist / baseline) are read from the pair's
    stored metadata if available, then fall back to the config defaults
    (``DEFAULT_STEREO_FOV``, ``DEFAULT_STEREO_BASELINE``), then to the
    explicit arguments supplied here.

    Args:
        camera_pair_id: Specific stereo pair ID to use.  If None, the most
                        recently received pair is used.
        fov: Override horizontal FOV in degrees.
        cam_dist: Override stereo baseline in metres.
        min_disp: Minimum disparity threshold.
        max_disp: Maximum disparity (defaults to 128).

    Returns:
        Dict with ``'points'`` and ``'colors'`` arrays, or None if no stereo
        images are available in storage.
    """
    try:
        from core.Imports import get_unified_image_storage

        storage = get_unified_image_storage()
    except ImportError as e:
        logger.error(f"UnifiedImageStorage not available: {e}")
        return None

    # Fetch images + metadata
    if camera_pair_id is not None:
        pair = storage.get_stereo_pair(camera_pair_id)
        if pair is None:
            logger.warning(f"No stereo pair found for id '{camera_pair_id}'")
            return None
        imgL, imgR, _ = pair
        metadata = storage.get_stereo_metadata(camera_pair_id) or {}
    else:
        result = storage.get_latest_stereo_image()
        if result is None:
            logger.warning("No stereo images in storage — is Unity streaming?")
            return None
        imgL, imgR, _, _, metadata = result

    # Resolve camera parameters: explicit arg > metadata > config default
    resolved_fov = fov or metadata.get("fov", DEFAULT_STEREO_FOV)
    resolved_cam_dist = cam_dist or metadata.get("baseline", DEFAULT_STEREO_BASELINE)

    logger.info(
        f"Reconstructing point cloud: fov={resolved_fov}°, "
        f"baseline={resolved_cam_dist}m"
    )
    return stereo_reconstruct_stream(
        imgL,
        imgR,
        fov=resolved_fov,
        cam_dist=resolved_cam_dist,
        min_disp=min_disp,
        max_disp=max_disp,
    )


# ---------------------------------------------------------------------------
# Edge masking (offline quality improvement)
# ---------------------------------------------------------------------------


def _remove_edges(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Suppress the strongest depth edges to reduce stereo matching artefacts.

    Sets the top-10 % of Laplacian-detected edges to NaN so that they are
    excluded from the point cloud.

    Args:
        image: Float32 disparity map (may contain NaN).
        ksize: Kernel size for median blur and Laplacian.

    Returns:
        Copy of the disparity map with edge pixels set to NaN.
    """
    img = np.copy(image)
    img = cv2.medianBlur(img, ksize)
    img = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)

    threshold = np.nanpercentile(np.asarray(img, dtype=np.float64), 90)
    mask = np.where(img < threshold, 0.0, 1.0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    mask = cv2.dilate(mask, kernel, iterations=1)

    return np.where(mask != 0, np.nan, image)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    p = argparse.ArgumentParser(
        description="Stereo point cloud reconstruction test tool."
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="Reconstruct from the latest Unity stereo pair in storage.",
    )
    p.add_argument(
        "--l",
        type=str,
        metavar="LEFT",
        help="Path to left camera image (offline mode).",
    )
    p.add_argument(
        "--r",
        type=str,
        metavar="RIGHT",
        help="Path to right camera image (offline mode).",
    )
    p.add_argument(
        "--downscale",
        type=int,
        default=1,
        help="Downscale factor for offline images (default: 1).",
    )
    p.add_argument(
        "--fov",
        type=float,
        default=DEFAULT_STEREO_FOV,
        help=f"Camera horizontal FOV in degrees (default: {DEFAULT_STEREO_FOV}).",
    )
    p.add_argument(
        "--cam_dist",
        type=float,
        default=DEFAULT_STEREO_BASELINE,
        help=f"Stereo baseline in metres (default: {DEFAULT_STEREO_BASELINE}).",
    )
    p.add_argument(
        "--min_disp",
        type=int,
        default=0,
        help="Minimum disparity threshold (default: 0).",
    )
    p.add_argument(
        "--max_disp", type=int, default=None, help="Maximum disparity (default: 128)."
    )
    p.add_argument(
        "--mask_edges",
        action="store_true",
        help="Suppress strong depth edges (offline mode only).",
    )
    p.add_argument(
        "--visualize",
        action="store_true",
        help="Open an open3d point cloud viewer after reconstruction.",
    )
    return p


def _visualize_point_cloud(point_cloud: Dict[str, np.ndarray]) -> None:
    """
    Display a point cloud with open3d if available, otherwise print a summary.

    Args:
        point_cloud: Dict with ``'points'`` and ``'colors'`` arrays.
    """
    try:
        import open3d as o3d  # type: ignore[import]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            point_cloud["points"].astype(np.float64)
        )
        pcd.colors = o3d.utility.Vector3dVector(
            point_cloud["colors"].astype(np.float64) / 255.0
        )
        logger.info("Opening open3d viewer — close the window to exit.")
        o3d.visualization.draw_geometries([pcd])  # type: ignore[attr-defined]
    except ImportError:
        logger.warning(
            "open3d not installed — cannot visualize. "
            "Install with: pip install open3d"
        )
        pts = point_cloud["points"]
        logger.info(
            f"Point cloud summary: {len(pts)} points, "
            f"X=[{pts[:,0].min():.3f}, {pts[:,0].max():.3f}] "
            f"Y=[{pts[:,1].min():.3f}, {pts[:,1].max():.3f}] "
            f"Z=[{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]"
        )


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    if args.live:
        # --- Live mode: pull from UnifiedImageStorage ---
        logger.info("Live mode: reading latest stereo pair from UnifiedImageStorage…")
        pc = reconstruct_from_storage(
            fov=args.fov,
            cam_dist=args.cam_dist,
            min_disp=args.min_disp,
            max_disp=args.max_disp,
        )
        if pc is None:
            logger.error(
                "No stereo images available. Ensure Unity is running and streaming."
            )
        else:
            logger.info(f"Point cloud: {len(pc['points'])} points")
            if args.visualize:
                _visualize_point_cloud(pc)

    else:
        # --- Offline mode: read from image files ---
        if not args.l or not args.r:
            _build_arg_parser().error("Provide --l and --r image paths, or use --live.")

        imgL = cv2.imread(args.l)
        imgR = cv2.imread(args.r)
        if imgL is None:
            raise ValueError(f"Could not load left image: {args.l}")
        if imgR is None:
            raise ValueError(f"Could not load right image: {args.r}")

        if args.downscale > 1:
            h, w = imgL.shape[:2]
            new_w, new_h = w // args.downscale, h // args.downscale
            imgL = cv2.resize(imgL, (new_w, new_h), interpolation=cv2.INTER_AREA)
            imgR = cv2.resize(imgR, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"Downscaled to {imgL.shape[1]}×{imgL.shape[0]}")

        pc = stereo_reconstruct(
            imgL,
            imgR,
            fov=args.fov,
            cam_dist=args.cam_dist,
            min_disp=args.min_disp,
            max_disp=args.max_disp,
            mask_edges=args.mask_edges,
        )
        logger.info(f"Point cloud: {len(pc['points'])} points")
        if args.visualize:
            _visualize_point_cloud(pc)
