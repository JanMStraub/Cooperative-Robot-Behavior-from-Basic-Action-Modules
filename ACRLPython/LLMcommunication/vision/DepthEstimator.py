#!/usr/bin/env python3
"""
DepthEstimator.py - Stereo depth estimation for object localization

Provides functions to:
1. Estimate depth at specific pixel coordinates using stereo disparity
2. Convert 2D pixel coordinates + depth to 3D world coordinates
3. Support robot target navigation with accurate spatial positioning
"""

import logging
import math
import sys
from typing import Tuple, Optional
import numpy as np
import cv2

# Import stereo reconstruction functions
# StereoImageReconstruction should be in sys.path when this module is loaded
# (managed by ObjectDetector.py which imports this module)
from StereoImageReconstruction.Reconstruct import calc_disparity
from StereoImageReconstruction.config import (
    CameraConfig,
    ReconstructionConfig,
    DEFAULT_CAMERA_CONFIG,
    DEFAULT_RECONSTRUCTION_CONFIG,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def estimate_depth_at_point(
    imgL: np.ndarray,
    imgR: np.ndarray,
    pixel_x: int,
    pixel_y: int,
    camera_config: Optional[CameraConfig] = None,
    recon_config: Optional[ReconstructionConfig] = None,
    window_size: int = 5
) -> Optional[float]:
    """
    Estimate depth at a specific pixel using stereo disparity.

    Args:
        imgL: Left camera image (BGR or grayscale)
        imgR: Right camera image (BGR or grayscale)
        pixel_x: X coordinate of target pixel
        pixel_y: Y coordinate of target pixel
        camera_config: Camera calibration parameters
        recon_config: Reconstruction algorithm parameters
        window_size: Window size for averaging disparity around target pixel

    Returns:
        Estimated depth in meters, or None if estimation fails
    """
    if camera_config is None:
        camera_config = DEFAULT_CAMERA_CONFIG
    if recon_config is None:
        recon_config = DEFAULT_RECONSTRUCTION_CONFIG

    # Convert to grayscale if needed
    if len(imgL.shape) == 3:
        imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    else:
        imgL_gray = imgL

    if len(imgR.shape) == 3:
        imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    else:
        imgR_gray = imgR

    # Validate pixel coordinates
    h, w = imgL_gray.shape
    if not (0 <= pixel_x < w and 0 <= pixel_y < h):
        logging.error(f"Pixel coordinates ({pixel_x}, {pixel_y}) out of bounds ({w}, {h})")
        return None

    try:
        # Compute disparity map
        logging.info(f"Computing disparity map for depth estimation at ({pixel_x}, {pixel_y})")
        disparity = calc_disparity(imgL_gray, imgR_gray, recon_config)

        # Extract disparity in window around target pixel
        half_window = window_size // 2
        y_min = max(0, pixel_y - half_window)
        y_max = min(h, pixel_y + half_window + 1)
        x_min = max(0, pixel_x - half_window)
        x_max = min(w, pixel_x + half_window + 1)

        disparity_window = disparity[y_min:y_max, x_min:x_max]

        # Filter out invalid disparities (NaN and <= 0)
        valid_disparities = disparity_window[~np.isnan(disparity_window)]
        valid_disparities = valid_disparities[valid_disparities > 0]

        if len(valid_disparities) == 0:
            logging.warning(f"No valid disparity values found at pixel ({pixel_x}, {pixel_y})")
            return None

        # Use median disparity for robustness
        disparity_value = np.median(valid_disparities)

        # Calculate focal length in pixels
        if camera_config.fov is not None:
            focal_length_px = (w / 2.0) / math.tan(math.radians(camera_config.fov / 2))
        elif camera_config.focal_length is not None and camera_config.sensor_width is not None:
            focal_length_px = camera_config.focal_length / camera_config.sensor_width * w
        else:
            logging.error("Camera config must provide either fov or (focal_length and sensor_width)")
            return None

        # Calculate depth: Z = (focal_length * baseline) / disparity
        depth = (focal_length_px * camera_config.baseline) / disparity_value

        logging.info(f"Estimated depth at ({pixel_x}, {pixel_y}): {depth:.3f}m (disparity: {disparity_value:.1f}px)")

        return float(depth)

    except Exception as e:
        logging.error(f"Failed to estimate depth: {e}")
        return None


def pixel_to_world_coords(
    pixel_x: int,
    pixel_y: int,
    depth: float,
    camera_config: Optional[CameraConfig] = None,
    image_width: int = 640,
    image_height: int = 480
) -> Tuple[float, float, float]:
    """
    Convert 2D pixel coordinates + depth to 3D world coordinates.

    Assumes camera coordinate system:
    - X axis: right (positive = right of image center)
    - Y axis: up (positive = up from image center)
    - Z axis: forward (positive = away from camera)

    Args:
        pixel_x: X pixel coordinate
        pixel_y: Y pixel coordinate
        depth: Depth in meters
        camera_config: Camera calibration parameters
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        Tuple of (world_x, world_y, world_z) in meters
    """
    if camera_config is None:
        camera_config = DEFAULT_CAMERA_CONFIG

    # Calculate focal length in pixels
    if camera_config.fov is not None:
        focal_length_px = (image_width / 2.0) / math.tan(math.radians(camera_config.fov / 2))
    elif camera_config.focal_length is not None and camera_config.sensor_width is not None:
        focal_length_px = camera_config.focal_length / camera_config.sensor_width * image_width
    else:
        logging.error("Camera config must provide either fov or (focal_length and sensor_width)")
        raise ValueError("Invalid camera configuration")

    # Calculate principal point (image center)
    cx = image_width / 2.0
    cy = image_height / 2.0

    # Convert pixel to normalized camera coordinates
    # OpenCV/Unity convention: Y increases downward in image, so we negate
    world_x = (pixel_x - cx) * depth / focal_length_px
    world_y = -(pixel_y - cy) * depth / focal_length_px  # Negate for up-positive convention
    world_z = depth

    logging.info(f"Pixel ({pixel_x}, {pixel_y}) at depth {depth:.3f}m → World ({world_x:.3f}, {world_y:.3f}, {world_z:.3f})")

    return (world_x, world_y, world_z)


def estimate_object_world_position(
    imgL: np.ndarray,
    imgR: np.ndarray,
    bbox_center_x: int,
    bbox_center_y: int,
    camera_config: Optional[CameraConfig] = None,
    recon_config: Optional[ReconstructionConfig] = None
) -> Optional[Tuple[float, float, float]]:
    """
    Estimate 3D world position of an object from its bounding box center.

    Combines depth estimation and coordinate conversion in one call.

    Args:
        imgL: Left camera image
        imgR: Right camera image
        bbox_center_x: X coordinate of bounding box center
        bbox_center_y: Y coordinate of bounding box center
        camera_config: Camera calibration parameters
        recon_config: Reconstruction algorithm parameters

    Returns:
        Tuple of (world_x, world_y, world_z) or None if estimation fails
    """
    # Estimate depth at bounding box center
    depth = estimate_depth_at_point(
        imgL, imgR, bbox_center_x, bbox_center_y,
        camera_config, recon_config
    )

    if depth is None:
        return None

    # Convert to world coordinates
    h, w = imgL.shape[:2]
    world_pos = pixel_to_world_coords(
        bbox_center_x, bbox_center_y, depth,
        camera_config, w, h
    )

    return world_pos


if __name__ == "__main__":
    # Test with sample images
    import argparse

    parser = argparse.ArgumentParser(description="Test stereo depth estimation")
    parser.add_argument("--left", type=str, required=True, help="Left image path")
    parser.add_argument("--right", type=str, required=True, help="Right image path")
    parser.add_argument("--x", type=int, required=True, help="Pixel X coordinate")
    parser.add_argument("--y", type=int, required=True, help="Pixel Y coordinate")
    parser.add_argument("--baseline", type=float, default=0.1, help="Camera baseline in meters")
    parser.add_argument("--fov", type=float, default=60.0, help="Camera FOV in degrees")

    args = parser.parse_args()

    # Load images
    imgL = cv2.imread(args.left)
    imgR = cv2.imread(args.right)

    if imgL is None or imgR is None:
        print("Error: Could not load images")
        sys.exit(1)

    # Create camera config
    cam_config = CameraConfig(fov=args.fov, baseline=args.baseline)

    # Estimate world position
    world_pos = estimate_object_world_position(
        imgL, imgR, args.x, args.y, cam_config
    )

    if world_pos:
        print(f"\n3D World Position: X={world_pos[0]:.3f}m, Y={world_pos[1]:.3f}m, Z={world_pos[2]:.3f}m")
    else:
        print("\nFailed to estimate world position")
