"""
Stereo image reconstruction module.

This module provides functions for generating 3D point clouds from stereoscopic image pairs
using Semi-Global Block Matching (SGBM) algorithm for disparity calculation.
"""

import argparse
import logging
import math
import os
from collections import namedtuple
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .StereoConfig import (
    CameraConfig,
    ReconstructionConfig,
    OutputConfig,
    DEFAULT_CAMERA_CONFIG,
    DEFAULT_RECONSTRUCTION_CONFIG,
    DEFAULT_OUTPUT_CONFIG,
)

# PLY file format header
PLY_HEADER = """ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

# Point cloud data structure
PLY = namedtuple("PLY", ["verts", "colors"])


def load_images(
    path_left: str, path_right: str, downscale: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load two stereoscopic images from the specified paths.

    Args:
        path_left: Path to the image taken from the left camera
        path_right: Path to the image taken from the right camera
        downscale: Downscale factor (1 = original size)

    Returns:
        Tuple of (left_image, right_image) as numpy arrays in BGR format

    Raises:
        FileNotFoundError: If either image file doesn't exist
        ValueError: If images have different dimensions or invalid downscale
    """
    if not os.path.exists(path_left):
        raise FileNotFoundError(f"Left image not found: {path_left}")
    if not os.path.exists(path_right):
        raise FileNotFoundError(f"Right image not found: {path_right}")

    imgL = cv2.imread(path_left)
    imgR = cv2.imread(path_right)

    if imgL is None or imgR is None:
        raise ValueError("Failed to load one or both images")

    if imgL.shape != imgR.shape:
        raise ValueError(
            f"Image dimensions mismatch: left {imgL.shape} vs right {imgR.shape}"
        )

    if downscale < 1:
        raise ValueError(f"Downscale must be >= 1, got {downscale}")

    if downscale > 1:
        shape = (int(imgL.shape[1] / downscale), int(imgL.shape[0] / downscale))
        imgL = cv2.resize(imgL, shape, interpolation=cv2.INTER_AREA)
        imgR = cv2.resize(imgR, shape, interpolation=cv2.INTER_AREA)

    print(f"Loaded images with shape: {imgL.shape}")
    return imgL, imgR


def estimate_max_disparity(
    imgL: np.ndarray, imgR: np.ndarray, percentile: float = 95.0
) -> int:
    """
    Estimate maximum disparity using feature matching.

    Args:
        imgL: Left grayscale image
        imgR: Right grayscale image
        percentile: Percentile of disparities to use for estimation

    Returns:
        Estimated maximum disparity (capped at image_width / 10)
    """
    try:
        # Try both import styles
        try:
            from ACRLPython.StereoImageReconstruction.FeatureMatching import (
                find_matches,
            )
        except ImportError:
            from .FeatureMatching import find_matches

        matches, kp1, kp2 = find_matches(imgL, imgR)

        if len(matches) < 10:
            print(
                f"Warning: Only found {len(matches)} matches, using image-based default"
            )
            # Use image-width-based heuristic: max_disp = width / 8
            default = imgL.shape[1] // 8
            return ((default + 15) // 16) * 16

        disparities = []
        for match in matches:
            x1 = kp1[match.queryIdx].pt[0]
            x2 = kp2[match.trainIdx].pt[0]
            y1 = kp1[match.queryIdx].pt[1]
            y2 = kp2[match.trainIdx].pt[1]
            dist = math.hypot(x1 - x2, y1 - y2)
            disparities.append(dist)

        estimate = int(2 * np.percentile(disparities, percentile))
        max_disp = min(estimate, imgL.shape[1] // 10)

        # Ensure max_disp is a multiple of 16 (required by SGBM)
        max_disp = ((max_disp + 15) // 16) * 16

        return max(16, max_disp)  # Minimum of 16

    except Exception as e:
        print(
            f"Warning: Could not estimate max disparity ({e}), using image-based default"
        )
        # Use image-width-based heuristic: max_disp = width / 8
        default = imgL.shape[1] // 8
        return ((default + 15) // 16) * 16


def calc_disparity(
    imgL: np.ndarray,
    imgR: np.ndarray,
    config: Optional[ReconstructionConfig] = None,
) -> np.ndarray:
    """
    Calculate the disparity map between two images using SGBM algorithm.

    Args:
        imgL: Left grayscale image
        imgR: Right grayscale image
        config: Reconstruction configuration (uses defaults if None)

    Returns:
        Disparity map as float32 array (negative values replaced with NaN)
    """
    if config is None:
        config = DEFAULT_RECONSTRUCTION_CONFIG

    if imgL.shape != imgR.shape:
        raise ValueError(
            f"Image shape mismatch: left {imgL.shape} vs right {imgR.shape}"
        )

    # Estimate max disparity if not provided
    max_disp = config.max_disparity
    if max_disp is None:
        imgL_gray = (
            cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY) if len(imgL.shape) == 3 else imgL
        )
        imgR_gray = (
            cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY) if len(imgR.shape) == 3 else imgR
        )

        # Try feature-based estimation first
        try:
            max_disp = estimate_max_disparity(imgL_gray, imgR_gray)
        except Exception as e:
            # Fallback: use image-width-based heuristic
            # Rule of thumb: max_disparity should be ~1/8 to 1/6 of image width
            logging.warning(
                f"Feature-based disparity estimation failed: {e}. Using heuristic."
            )
            max_disp = imgL.shape[1] // 8

        print(f"Estimated maximum disparity: {max_disp}")

    # Ensure max_disp is a multiple of 16 and reasonable
    max_disp = ((max_disp + 15) // 16) * 16

    # Cap max_disparity for performance (larger values are very slow)
    if max_disp > 256:
        print(f"Warning: Capping max_disparity from {max_disp} to 256 for performance")
        max_disp = 256

    # Create SGBM matcher
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
        mode=cv2.STEREO_SGBM_MODE_HH,
    )

    # Compute disparity
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    # Replace negative disparities with NaN
    return np.where(disp >= 0.0, disp, np.nan)


def calc_normals(depth_map: np.ndarray, grad_scale: float = 50.0) -> np.ndarray:
    """
    Compute normal map from depth map using Sobel gradients.

    Args:
        depth_map: Depth map as 2D array
        grad_scale: Scale factor for gradient normalization

    Returns:
        Normal map as uint8 array (RGB format, 0-255)
    """
    print("Computing normal map...")

    ksize = 1
    grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=ksize)

    # Get maximum absolute gradient
    abs_max = max(
        np.abs(np.nan_to_num(grad_x)).max(), np.abs(np.nan_to_num(grad_y)).max()
    )

    # Create normal vectors (Z, -Y, -X)
    grad_z = np.full_like(depth_map, abs_max / grad_scale)
    normals = np.stack((grad_z, -grad_y, -grad_x), axis=-1)

    # Normalize
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    norm = np.where(norm == 0, 1, norm)  # Avoid division by zero
    normals /= norm

    # Convert to 0-255 range
    res = ((normals + 1) / 2.0) * 255
    return res.astype(np.uint8)


def write_ply(file_path: str, ply: PLY) -> None:
    """
    Write point cloud data to PLY file.

    Args:
        file_path: Output file path
        ply: PLY namedtuple containing vertices and colors
    """
    verts = ply.verts.reshape(-1, 3)
    colors = ply.colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])

    # Ensure output directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as f:
        f.write((PLY_HEADER % dict(vert_num=len(verts))).encode("utf-8"))
        np.savetxt(f, verts, fmt="%f %f %f %d %d %d ")

    print(f"Saved point cloud to: {file_path}")


def make_3d(
    depth_map: np.ndarray,
    imgL: np.ndarray,
    focal_length: float,
    min_disp: float = 0.0,
    cam_dist: float = 0.1,
) -> PLY:
    """
    Generate a 3D point cloud from a depth map.

    Args:
        depth_map: Disparity/depth map (2D array)
        imgL: Reference image to color the point cloud (BGR format)
        focal_length: Focal length of the camera in pixels
        min_disp: Minimum disparity threshold (pixels below this are filtered)
        cam_dist: Baseline distance between the two cameras (meters)

    Returns:
        PLY namedtuple containing vertices and colors
    """
    depth_map = np.nan_to_num(depth_map, nan=-1)

    h, w = imgL.shape[:2]
    f = focal_length
    T_x = cam_dist

    # Reprojection matrix (Q)
    Q = np.array(
        [[-1, 0, 0, 0.5 * w], [0, 1, 0, -0.5 * h], [0, 0, 0, f], [0, 0, -1 / T_x, 0]],
        dtype=np.float32,
    )

    # Reproject to 3D
    points = cv2.reprojectImageTo3D(depth_map, Q)  # type: ignore[call-overload]
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

    # Filter points
    mask = depth_map > min_disp
    out_points = points[mask]
    out_colors = colors[mask]

    return PLY(out_points, out_colors)


def remove_edges(
    image: np.ndarray, config: Optional[ReconstructionConfig] = None
) -> np.ndarray:
    """
    Mask out the strongest edges in the depth map to reduce noise.

    Args:
        image: Input depth map
        config: Reconstruction configuration (uses defaults if None)

    Returns:
        Depth map with strong edges masked as NaN
    """
    if config is None:
        config = DEFAULT_RECONSTRUCTION_CONFIG

    img = np.copy(image)
    img = cv2.medianBlur(img, 5)
    img = cv2.Laplacian(img, cv2.CV_64F, ksize=config.edge_kernel_size)

    # Mask the strongest edges
    threshold_value = np.nanpercentile(img, config.edge_percentile)  # type: ignore[call-overload]
    img = np.where(img < threshold_value, 0.0, 1.0)

    # Dilate edge mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (config.edge_kernel_size, config.edge_kernel_size)
    )
    img = cv2.dilate(img, kernel, iterations=1)

    return np.where(img != 0, np.nan, image)


def f_from_fov(fov: float, image_width: int) -> float:
    """
    Calculate focal length in pixels from field of view.

    Args:
        fov: Field of view in degrees
        image_width: Image width in pixels

    Returns:
        Focal length in pixels
    """
    return (image_width / 2.0) / math.tan(math.radians(fov / 2))


def reconstruct(
    disparity: np.ndarray,
    imgL_rgb: np.ndarray,
    camera_config: Optional[CameraConfig] = None,
    min_disp: float = 0.0,
) -> PLY:
    """
    Reconstruct a point cloud from a disparity map.

    Args:
        disparity: Disparity map between the two images
        imgL_rgb: Reference image to color the point cloud (BGR format)
        camera_config: Camera configuration (uses defaults if None)
        min_disp: Minimum disparity threshold

    Returns:
        PLY namedtuple containing vertices and colors

    Raises:
        ValueError: If camera parameters are not properly specified
    """
    if camera_config is None:
        camera_config = DEFAULT_CAMERA_CONFIG

    # Calculate focal length
    if camera_config.fov is not None:
        f = f_from_fov(camera_config.fov, disparity.shape[1])
    elif (
        camera_config.focal_length is not None
        and camera_config.sensor_width is not None
    ):
        f = camera_config.focal_length / camera_config.sensor_width
        f = f * disparity.shape[1]
    else:
        raise ValueError(
            "Either fov or (focal_length and sensor_width) must be provided in camera_config"
        )

    point_cloud = make_3d(
        disparity, imgL_rgb, f, min_disp=min_disp, cam_dist=camera_config.baseline
    )

    return point_cloud


def stereo_reconstruct(
    imgL: np.ndarray,
    imgR: np.ndarray,
    camera_config: Optional[CameraConfig] = None,
    recon_config: Optional[ReconstructionConfig] = None,
    output_config: Optional[OutputConfig] = None,
) -> PLY:
    """
    Reconstruct a point cloud from two stereoscopic images.

    Args:
        imgL: Image taken from the left camera (BGR format)
        imgR: Image taken from the right camera (BGR format)
        camera_config: Camera configuration (uses defaults if None)
        recon_config: Reconstruction configuration (uses defaults if None)
        output_config: Output configuration (uses defaults if None)

    Returns:
        PLY namedtuple containing vertices and colors
    """
    if camera_config is None:
        camera_config = DEFAULT_CAMERA_CONFIG
    if recon_config is None:
        recon_config = DEFAULT_RECONSTRUCTION_CONFIG
    if output_config is None:
        output_config = DEFAULT_OUTPUT_CONFIG

    imgL_rgb = imgL.copy()
    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    print("Computing disparity...")
    disparity = calc_disparity(imgL_gray, imgR_gray, recon_config)

    # Optional edge masking
    if recon_config.mask_edges:
        disparity = remove_edges(disparity, recon_config)

    # Filter low disparities
    disparity[disparity <= recon_config.min_depth_threshold] = 0.0

    # Save disparity map if requested
    if output_config.save_disparity:
        output_dir = Path(output_config.output_base_dir) / output_config.disparity_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "disparity_stereo_reconstruction.png"

        disparity_vis = disparity / np.nanmax(disparity) * 255
        cv2.imwrite(str(output_path), disparity_vis)
        print(f"Saved disparity map to: {output_path}")

    print("Generating 3D point cloud...")
    point_cloud = reconstruct(
        disparity, imgL_rgb, camera_config, min_disp=recon_config.min_depth_threshold
    )

    # Save point cloud if requested
    if output_config.save_point_cloud:
        output_dir = Path(output_config.output_base_dir) / output_config.point_cloud_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "point_cloud_stereo_reconstruction.ply"
        write_ply(str(output_path), point_cloud)

    return point_cloud


def stereo_reconstruct_stream(
    imgL: np.ndarray,
    imgR: np.ndarray,
    camera_config: Optional[CameraConfig] = None,
    recon_config: Optional[ReconstructionConfig] = None,
) -> PLY:
    """
    Reconstruct a point cloud from streamed stereoscopic images.
    This function does not write data to disk.

    Args:
        imgL: Left camera image (BGR format)
        imgR: Right camera image (BGR format)
        camera_config: Camera configuration (uses defaults if None)
        recon_config: Reconstruction configuration (uses defaults if None)

    Returns:
        PLY namedtuple containing vertices and colors
    """
    if camera_config is None:
        camera_config = DEFAULT_CAMERA_CONFIG
    if recon_config is None:
        recon_config = DEFAULT_RECONSTRUCTION_CONFIG

    imgL_rgb = imgL.copy()
    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    print("Computing disparity...")
    disparity = calc_disparity(imgL_gray, imgR_gray, recon_config)

    print("Generating 3D point cloud...")
    point_cloud = reconstruct(
        disparity, imgL_rgb, camera_config, min_disp=recon_config.min_depth_threshold
    )

    return point_cloud


def main():
    """Command-line interface for stereo reconstruction"""
    parser = argparse.ArgumentParser(
        description="Reconstruct 3D point cloud from stereo images"
    )
    parser.add_argument(
        "--l",
        type=str,
        required=True,
        help="Path to the image taken from the left camera",
    )
    parser.add_argument(
        "--r",
        type=str,
        required=True,
        help="Path to the image taken from the right camera",
    )
    parser.add_argument(
        "--downscale", type=int, default=1, help="Downscale factor (default: 1)"
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=60.0,
        help="Field of view of the camera in degrees (default: 60)",
    )
    parser.add_argument("--focal_length", type=float, help="Focal length of the camera")
    parser.add_argument("--sensor_width", type=float, help="Width of the camera sensor")
    parser.add_argument(
        "--min_disp", type=float, default=0.0, help="Minimum disparity threshold"
    )
    parser.add_argument("--max_disp", type=int, help="Maximum disparity (auto if None)")
    parser.add_argument(
        "--cam_dist",
        type=float,
        default=0.1,
        help="Distance between the two cameras in meters",
    )
    parser.add_argument(
        "--mask_edges", action="store_true", help="Mask out strong edges in depth map"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output", help="Output directory"
    )

    args = parser.parse_args()

    # Load images
    imgL, imgR = load_images(args.l, args.r, args.downscale)

    # Create configurations
    camera_config = CameraConfig(
        fov=args.fov if args.focal_length is None else None,
        focal_length=args.focal_length,
        sensor_width=args.sensor_width,
        baseline=args.cam_dist,
    )

    recon_config = ReconstructionConfig(
        min_depth_threshold=args.min_disp,
        max_disparity=args.max_disp,
        mask_edges=args.mask_edges,
    )

    output_config = OutputConfig(output_base_dir=args.output_dir)

    # Reconstruct
    stereo_reconstruct(imgL, imgR, camera_config, recon_config, output_config)
    print("Reconstruction complete!")


if __name__ == "__main__":
    main()
