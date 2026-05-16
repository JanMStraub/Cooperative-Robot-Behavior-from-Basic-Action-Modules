#!/usr/bin/env python3
"""Stereo depth estimation using StereoSGBM. Compute disparity once, reuse across detections."""

import logging
import math
from typing import Tuple, Optional, List
from pathlib import Path
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Import configuration - support both direct script and module execution
try:
    from .StereoConfig import (
        CameraConfig,
        ReconstructionConfig,
        SGBMPreset,
        DEFAULT_CAMERA_CONFIG,
        DEFAULT_RECONSTRUCTION_CONFIG,
        SGBM_CLOSE,
        SGBM_MEDIUM,
        SGBM_FAR,
    )
except ImportError as e:
    logger.warning(f"StereoConfig relative import failed, trying absolute: {e}")
    from vision.StereoConfig import (
        CameraConfig,
        ReconstructionConfig,
        SGBMPreset,
        DEFAULT_CAMERA_CONFIG,
        DEFAULT_RECONSTRUCTION_CONFIG,
        SGBM_CLOSE,
        SGBM_MEDIUM,
        SGBM_FAR,
    )

try:
    from config.Vision import (
        SAVE_DEBUG_DISPARITY_MAPS,
        DEBUG_DISPARITY_DIR,
        DEFAULT_SGBM_PRESET,
        DEPTH_SAMPLE_INNER_PERCENT,
        DEPTH_SAMPLING_STRATEGY,
        ENABLE_DISPARITY_CACHE,
        DISPARITY_CACHE_TTL,
    )
except ImportError:
    from ..config.Vision import (
        SAVE_DEBUG_DISPARITY_MAPS,
        DEBUG_DISPARITY_DIR,
        DEFAULT_SGBM_PRESET,
        DEPTH_SAMPLE_INNER_PERCENT,
        DEPTH_SAMPLING_STRATEGY,
        ENABLE_DISPARITY_CACHE,
        DISPARITY_CACHE_TTL,
    )

from core.LoggingSetup import get_logger

logger = get_logger(__name__)

import time as _time

# Module-level disparity cache keyed by image shape + first 64 bytes.
# Avoids re-running SGBM for duplicate frames; full-image hashing would cost more than SGBM itself.
_disparity_cache: dict = {}


def _make_cache_key(imgL: np.ndarray, imgR: np.ndarray) -> tuple:

    def _head(img: np.ndarray) -> bytes:
        flat = img.flat
        return bytes(int(next(flat)) for _ in range(min(64, img.size)))

    return (imgL.shape, imgR.shape, _head(imgL), _head(imgR))


def calc_disparity(
    imgL: np.ndarray,
    imgR: np.ndarray,
    config: Optional[ReconstructionConfig] = None,
) -> np.ndarray:
    """Compute SGBM disparity map. Negative disparities → NaN."""
    if config is None:
        config = DEFAULT_RECONSTRUCTION_CONFIG

    if imgL.shape != imgR.shape:
        raise ValueError(
            f"Image shape mismatch: left {imgL.shape} vs right {imgR.shape}"
        )

    if ENABLE_DISPARITY_CACHE:
        cache_key = _make_cache_key(imgL, imgR)
        now = _time.time()
        cached = _disparity_cache.get(cache_key)
        if cached is not None:
            disp, ts = cached
            if now - ts < DISPARITY_CACHE_TTL:
                logger.debug("Returning cached disparity map")
                return disp

    max_disp = config.max_disparity
    if max_disp is None:
        max_disp = imgL.shape[1] // 8
        logger.debug(f"Estimated maximum disparity: {max_disp}")

    # Must be multiple of 16 (SGBM requirement)
    max_disp = ((max_disp + 15) // 16) * 16

    if max_disp > 256:
        logger.debug(f"Capping max_disparity from {max_disp} to 256 for performance")
        max_disp = 256

    stereo = cv2.StereoSGBM_create(  # type: ignore[attr-defined]
        minDisparity=config.min_disparity,
        numDisparities=max_disp,
        blockSize=config.window_size,
        P1=config.p1_multiplier * 3 * config.window_size**2,
        P2=config.p2_multiplier * 3 * config.window_size**2,
        disp12MaxDiff=config.disp12_max_diff,
        uniquenessRatio=config.uniqueness_ratio,
        speckleWindowSize=config.speckle_window_size,
        speckleRange=config.speckle_range,
        mode=cv2.STEREO_SGBM_MODE_HH,  # type: ignore[attr-defined]
    )

    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    result = np.where(disp >= 0.0, disp, np.nan)

    if ENABLE_DISPARITY_CACHE:
        _disparity_cache[_make_cache_key(imgL, imgR)] = (result, _time.time())

    return result


def select_sgbm_preset(estimated_distance: Optional[float] = None) -> SGBMPreset:
    """Select SGBM preset for distance range. CLOSE<1m, MEDIUM<2m, FAR>=2m."""
    if estimated_distance is None:
        # Use config-specified default preset
        preset_map = {"close": SGBM_CLOSE, "medium": SGBM_MEDIUM, "far": SGBM_FAR}
        return preset_map.get(DEFAULT_SGBM_PRESET, SGBM_MEDIUM)

    if estimated_distance < 1.0:
        logger.debug(f"Selected CLOSE preset for distance {estimated_distance:.2f}m")
        return SGBM_CLOSE
    elif estimated_distance < 2.0:
        logger.debug(f"Selected MEDIUM preset for distance {estimated_distance:.2f}m")
        return SGBM_MEDIUM
    else:
        logger.debug(f"Selected FAR preset for distance {estimated_distance:.2f}m")
        return SGBM_FAR


def calc_disparity_with_preset(
    imgL: np.ndarray,
    imgR: np.ndarray,
    preset: SGBMPreset,
) -> np.ndarray:
    """Convenience wrapper: convert SGBMPreset → ReconstructionConfig and calc disparity."""
    config = ReconstructionConfig(
        window_size=preset.window_size,
        min_disparity=preset.min_disparity,
        max_disparity=preset.max_disparity,
        uniqueness_ratio=preset.uniqueness_ratio,
        speckle_window_size=preset.speckle_window_size,
        speckle_range=preset.speckle_range,
        disp12_max_diff=preset.disp12_max_diff,
        p1_multiplier=preset.p1_multiplier,
        p2_multiplier=preset.p2_multiplier,
    )

    return calc_disparity(imgL, imgR, config)


def calculate_focal_length_from_fov(
    fov_vertical_deg: float, image_width: int, image_height: int
) -> float:
    """Convert Unity vertical FOV to pixel focal length via aspect-ratio horizontal FOV."""
    aspect_ratio = image_width / image_height
    vertical_fov_rad = math.radians(fov_vertical_deg)
    horizontal_fov_rad = 2 * math.atan(math.tan(vertical_fov_rad / 2) * aspect_ratio)
    focal_length_px = (image_width / 2.0) / math.tan(horizontal_fov_rad / 2)

    logger.debug(
        f"FOV conversion: vertical={fov_vertical_deg}° → "
        f"horizontal={math.degrees(horizontal_fov_rad):.1f}° → "
        f"focal_length={focal_length_px:.1f}px"
    )

    return focal_length_px


def get_focal_length_pixels(
    camera_config: CameraConfig, image_width: int, image_height: int
) -> float:
    """Get focal length in pixels from FOV (preferred) or focal_length/sensor_width."""
    if camera_config.fov is not None and camera_config.fov > 0:
        return calculate_focal_length_from_fov(
            camera_config.fov, image_width, image_height
        )
    elif (
        camera_config.focal_length is not None
        and camera_config.sensor_width is not None
    ):
        return camera_config.focal_length / camera_config.sensor_width * image_width
    else:
        raise ValueError(
            "Camera config must provide fov or (focal_length and sensor_width)"
        )


def save_disparity_map_debug(disparity: np.ndarray, output_path: Optional[Path] = None):
    """Save disparity map as 16-bit PNG + colorized JPG if SAVE_DEBUG_DISPARITY_MAPS=True."""
    if not SAVE_DEBUG_DISPARITY_MAPS:
        return

    try:
        import time as _time

        if output_path is None:
            output_dir = Path(DEBUG_DISPARITY_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(_time.time() * 1000)
            output_path = output_dir / f"disparity_{timestamp}.png"

        disp_valid = np.nan_to_num(disparity, nan=0.0)
        cv2.imwrite(str(output_path), (disp_valid * 16).astype(np.uint16))
        # Also save a colorized visualization
        disp_normalized = np.zeros_like(disp_valid, dtype=np.uint8)
        cv2.normalize(
            disp_valid, disp_normalized, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        disp_colored = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
        color_path = str(output_path).replace(".png", "_color.jpg")
        cv2.imwrite(color_path, disp_colored)
        logger.debug(f"Saved disparity map to {output_path}")
    except Exception as e:
        logger.debug(f"Could not save disparity map: {e}")


def estimate_depth_from_disparity(
    disparity: np.ndarray,
    pixel_x: int,
    pixel_y: int,
    camera_config: CameraConfig,
    image_width: int,
    image_height: int,
    window_size: int = 15,
    min_disparity_threshold: float = 5.0,
    max_depth_threshold: float = 10.0,
) -> Optional[float]:
    """Estimate depth at pixel from pre-computed disparity. Expands search window progressively when no valid disparities found."""
    h, w = disparity.shape

    if not (0 <= pixel_x < w and 0 <= pixel_y < h):
        logger.error(f"Pixel ({pixel_x}, {pixel_y}) out of bounds ({w}, {h})")
        return None

    try:
        for attempt, search_window in enumerate(
            [window_size, window_size * 3, window_size * 6, window_size * 10], 1
        ):
            half_window = search_window // 2
            y_min = max(0, pixel_y - half_window)
            y_max = min(h, pixel_y + half_window + 1)
            x_min = max(0, pixel_x - half_window)
            x_max = min(w, pixel_x + half_window + 1)

            disparity_window = disparity[y_min:y_max, x_min:x_max]

            # Filter for valid disparities above threshold
            valid_disparities = disparity_window[~np.isnan(disparity_window)]
            valid_disparities = valid_disparities[
                valid_disparities >= min_disparity_threshold
            ]

            if len(valid_disparities) > 0:
                if attempt > 1:
                    logger.debug(
                        f"Found {len(valid_disparities)} valid disparities on attempt {attempt} "
                        f"(window={search_window}px)"
                    )
                break
        else:
            # Still no valid disparities - try one last time with very relaxed threshold (0.1px)
            logger.debug("Trying with very low threshold (0.1px) in large window")
            half_window = window_size * 15  # Very large window
            y_min = max(0, pixel_y - half_window)
            y_max = min(h, pixel_y + half_window + 1)
            x_min = max(0, pixel_x - half_window)
            x_max = min(w, pixel_x + half_window + 1)

            disparity_window = disparity[y_min:y_max, x_min:x_max]
            valid_disparities = disparity_window[~np.isnan(disparity_window)]
            valid_disparities = valid_disparities[valid_disparities >= 0.1]

            if len(valid_disparities) > 0:
                logger.debug(
                    f"Found {len(valid_disparities)} valid disparities with relaxed threshold "
                    f"(window={window_size * 15}px)"
                )
            else:
                # No valid disparities found anywhere near the pixel
                all_disp = disparity.flatten()
                all_valid = all_disp[~np.isnan(all_disp) & (all_disp > 0)]
                if len(all_valid) > 0:
                    logger.debug(
                        f"No valid disparity at ({pixel_x}, {pixel_y}). "
                        f"Map has {len(all_valid)} values (range: {all_valid.min():.1f}-{all_valid.max():.1f}px)"
                    )
                else:
                    logger.debug(
                        "Entire disparity map is invalid - stereo matching failed"
                    )
                return None

        disparity_value = np.median(valid_disparities)

        if np.isnan(disparity_value) or disparity_value < min_disparity_threshold:
            logger.debug(
                f"Median disparity {disparity_value:.1f}px below threshold {min_disparity_threshold}px"
            )
            return None

        focal_length_px = get_focal_length_pixels(
            camera_config, image_width, image_height
        )

        depth = (focal_length_px * camera_config.baseline) / disparity_value

        if depth > max_depth_threshold:
            logger.debug(
                f"Calculated depth {depth:.2f}m exceeds threshold {max_depth_threshold}m "
                f"(disparity={disparity_value:.1f}px) - rejecting"
            )
            return None

        logger.debug(
            f"Depth at ({pixel_x}, {pixel_y}): {depth:.3f}m (disparity: {disparity_value:.1f}px)"
        )

        return float(depth)

    except Exception as e:
        logger.error(f"Failed to estimate depth from disparity: {e}")
        return None


def estimate_depth_from_bbox(
    disparity_map: np.ndarray,
    bbox: Tuple[int, int, int, int],
    focal_length_px: float,
    baseline: float,
    strategy: str = DEPTH_SAMPLING_STRATEGY,
    min_disparity_threshold: float = 5.0,
    max_depth_threshold: float = 10.0,
    inner_percent: int = DEPTH_SAMPLE_INNER_PERCENT,
) -> Optional[Tuple[float, float, int]]:
    """Sample disparity within bbox inner region. ~50-60% error reduction vs single-point sampling.

    strategy: "median_inner_50pct" (default), "mean_valid", or "max_disparity" (closest point, for grasping).
    Returns (depth_m, median_disparity, num_valid_pixels) or None.
    """
    x, y, w, h = bbox
    height, width = disparity_map.shape

    margin_fraction = (100 - inner_percent) / 200.0  # e.g., 50% → 0.25 margin each side
    margin_x = int(w * margin_fraction)
    margin_y = int(h * margin_fraction)

    roi_x1 = max(0, x + margin_x)
    roi_y1 = max(0, y + margin_y)
    roi_x2 = min(width, x + w - margin_x)
    roi_y2 = min(height, y + h - margin_y)

    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        logger.debug(
            f"ROI too small after margin: bbox=({x},{y},{w},{h}), "
            f"roi=({roi_x1},{roi_y1},{roi_x2},{roi_y2})"
        )
        return None

    roi_disparity = disparity_map[roi_y1:roi_y2, roi_x1:roi_x2]

    valid_mask = (roi_disparity > min_disparity_threshold) & ~np.isnan(roi_disparity)
    valid_disparities = roi_disparity[valid_mask]

    if len(valid_disparities) == 0:
        logger.debug(
            f"No valid disparities in bbox ROI ({roi_x1},{roi_y1},{roi_x2},{roi_y2})"
        )
        return None

    if strategy == "median_inner_50pct":
        disparity = np.median(valid_disparities)
    elif strategy == "mean_valid":
        disparity = np.mean(valid_disparities)
    elif strategy == "max_disparity":
        disparity = np.max(valid_disparities)  # Closest point in bbox
    else:
        logger.warning(f"Unknown strategy '{strategy}', using median")
        disparity = np.median(valid_disparities)

    depth_m = (focal_length_px * baseline) / disparity

    if depth_m > max_depth_threshold:
        logger.debug(
            f"Depth {depth_m:.2f}m exceeds threshold {max_depth_threshold}m "
            f"(disparity={disparity:.1f}px)"
        )
        return None

    logger.debug(
        f"Bbox depth: {depth_m:.3f}m (disparity: {disparity:.1f}px, "
        f"valid_pixels: {len(valid_disparities)}, strategy: {strategy})"
    )

    return (float(depth_m), float(disparity), int(np.sum(valid_mask)))


def estimate_depth_at_point(
    imgL: np.ndarray,
    imgR: np.ndarray,
    pixel_x: int,
    pixel_y: int,
    camera_config: Optional[CameraConfig] = None,
    recon_config: Optional[ReconstructionConfig] = None,
    window_size: int = 15,
    min_disparity_threshold: float = 5.0,
    max_depth_threshold: float = 10.0,
) -> Optional[float]:
    """Single-point depth estimate. For multi-detection scenes use estimate_depth_from_disparity() with pre-computed disparity instead."""
    if camera_config is None:
        camera_config = DEFAULT_CAMERA_CONFIG
    if recon_config is None:
        recon_config = DEFAULT_RECONSTRUCTION_CONFIG

    if len(imgL.shape) == 3:
        imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    else:
        imgL_gray = imgL

    if len(imgR.shape) == 3:
        imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    else:
        imgR_gray = imgR

    h, w = imgL_gray.shape
    if not (0 <= pixel_x < w and 0 <= pixel_y < h):
        logger.error(f"Pixel ({pixel_x}, {pixel_y}) out of bounds ({w}, {h})")
        return None

    try:
        logger.debug(f"Computing disparity for depth at ({pixel_x}, {pixel_y})")
        disparity = calc_disparity(imgL_gray, imgR_gray, recon_config)

        save_disparity_map_debug(disparity)

        return estimate_depth_from_disparity(
            disparity,
            pixel_x,
            pixel_y,
            camera_config,
            w,
            h,
            window_size,
            min_disparity_threshold,
            max_depth_threshold,
        )

    except Exception as e:
        logger.error(f"Failed to estimate depth: {e}")
        return None


def pixel_to_world_coords(
    pixel_x: int,
    pixel_y: int,
    depth: float,
    camera_config: Optional[CameraConfig] = None,
    image_width: int = 640,
    image_height: int = 480,
    camera_rotation: Optional[List[float]] = None,
    camera_position: Optional[List[float]] = None,
) -> Tuple[float, float, float]:
    """Convert 2D pixel + depth to 3D world coords. Supports quaternion [x,y,z,w] or Euler [pitch,yaw,roll] rotation."""
    if camera_config is None:
        camera_config = DEFAULT_CAMERA_CONFIG

    focal_length_px = get_focal_length_pixels(camera_config, image_width, image_height)

    cx = image_width / 2.0
    cy = image_height / 2.0

    # Camera space: X=right, Y=up (negate image Y), Z=forward
    x_cam = (pixel_x - cx) * depth / focal_length_px
    y_cam = -(pixel_y - cy) * depth / focal_length_px
    z_cam = depth

    if camera_rotation is not None and camera_rotation != [0, 0, 0]:
        if len(camera_rotation) == 4:
            # Quaternion [x, y, z, w] from StereoCameraController (since Jan 2026).
            # Sandwich product: q * (0,v) * q_conj
            qx, qy, qz, qw = camera_rotation
            qnorm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
            if qnorm > 1e-9:
                qx, qy, qz, qw = qx / qnorm, qy / qnorm, qz / qnorm, qw / qnorm
            tx = 2.0 * (qy * z_cam - qz * y_cam)
            ty = 2.0 * (qz * x_cam - qx * z_cam)
            tz = 2.0 * (qx * y_cam - qy * x_cam)
            x_rotated = x_cam + qw * tx + qy * tz - qz * ty
            y_rotated = y_cam + qw * ty + qz * tx - qx * tz
            z_rotated = z_cam + qw * tz + qx * ty - qy * tx
        else:
            # Legacy Euler [pitch, yaw, roll] in degrees. Unity order: Y->X->Z
            pitch, yaw, roll = camera_rotation

            pitch_rad = math.radians(pitch)
            yaw_rad = math.radians(yaw)
            roll_rad = math.radians(roll)

            cos_yaw = math.cos(yaw_rad)
            sin_yaw = math.sin(yaw_rad)
            x1 = cos_yaw * x_cam + sin_yaw * z_cam
            y1 = y_cam
            z1 = -sin_yaw * x_cam + cos_yaw * z_cam

            cos_pitch = math.cos(pitch_rad)
            sin_pitch = math.sin(pitch_rad)
            x2 = x1
            y2 = cos_pitch * y1 - sin_pitch * z1
            z2 = sin_pitch * y1 + cos_pitch * z1

            cos_roll = math.cos(roll_rad)
            sin_roll = math.sin(roll_rad)
            x_rotated = cos_roll * x2 - sin_roll * y2
            y_rotated = sin_roll * x2 + cos_roll * y2
            z_rotated = z2

        x_cam, y_cam, z_cam = x_rotated, y_rotated, z_rotated
        logger.debug(
            f"Pixel ({pixel_x}, {pixel_y}) depth {depth:.3f}m → "
            f"Rotated by {camera_rotation} → "
            f"({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f})m in world orientation"
        )

    if camera_position is not None:
        world_x = x_cam + camera_position[0]
        world_y = y_cam + camera_position[1]
        world_z = z_cam + camera_position[2]

        logger.debug(
            f"Camera-relative ({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f}) + "
            f"Camera position ({camera_position[0]:.3f}, {camera_position[1]:.3f}, {camera_position[2]:.3f}) → "
            f"World ({world_x:.3f}, {world_y:.3f}, {world_z:.3f})m"
        )
        return (world_x, world_y, world_z)
    else:
        logger.debug(
            f"Pixel ({pixel_x}, {pixel_y}) depth {depth:.3f}m → "
            f"Camera-relative ({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f})m"
        )
        return (x_cam, y_cam, z_cam)


def estimate_object_world_position_from_disparity(
    disparity: np.ndarray,
    bbox_center_x: int,
    bbox_center_y: int,
    camera_config: CameraConfig,
    image_width: int,
    image_height: int,
    min_disparity: float = 5.0,
    max_depth: float = 10.0,
    camera_rotation: Optional[List[float]] = None,
    camera_position: Optional[List[float]] = None,
) -> Optional[Tuple[float, float, float]]:
    """3D world position from pre-computed disparity. Preferred for multi-detection scenes."""
    depth = estimate_depth_from_disparity(
        disparity,
        bbox_center_x,
        bbox_center_y,
        camera_config,
        image_width,
        image_height,
        min_disparity_threshold=min_disparity,
        max_depth_threshold=max_depth,
    )

    if depth is None:
        return None

    return pixel_to_world_coords(
        bbox_center_x,
        bbox_center_y,
        depth,
        camera_config,
        image_width,
        image_height,
        camera_rotation=camera_rotation,
        camera_position=camera_position,
    )


def estimate_object_world_position(
    imgL: np.ndarray,
    imgR: np.ndarray,
    bbox_center_x: int,
    bbox_center_y: int,
    camera_config: Optional[CameraConfig] = None,
    recon_config: Optional[ReconstructionConfig] = None,
    min_disparity: float = 5.0,
    max_depth: float = 10.0,
    camera_rotation: Optional[List[float]] = None,
    camera_position: Optional[List[float]] = None,
) -> Optional[Tuple[float, float, float]]:
    """Convenience wrapper computing disparity inline. Use estimate_object_world_position_from_disparity() for multi-detection scenes."""
    depth = estimate_depth_at_point(
        imgL,
        imgR,
        bbox_center_x,
        bbox_center_y,
        camera_config,
        recon_config,
        min_disparity_threshold=min_disparity,
        max_depth_threshold=max_depth,
    )

    if depth is None:
        return None

    h, w = imgL.shape[:2]
    world_pos = pixel_to_world_coords(
        bbox_center_x,
        bbox_center_y,
        depth,
        camera_config,
        w,
        h,
        camera_rotation=camera_rotation,
        camera_position=camera_position,
    )

    return world_pos
