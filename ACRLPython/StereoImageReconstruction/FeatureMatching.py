"""
Feature matching module for stereo reconstruction.

This module provides ORB-based feature matching for estimating disparity
and validating stereo calibration.
"""

import math
from typing import List, Tuple, Optional

import cv2
import numpy as np

from .config import FeatureMatchConfig, DEFAULT_FEATURE_CONFIG


def find_matches(
    imgL: np.ndarray,
    imgR: np.ndarray,
    config: Optional[FeatureMatchConfig] = None,
) -> Tuple[List[cv2.DMatch], List[cv2.KeyPoint], List[cv2.KeyPoint]]:
    """
    Find feature matches between two stereo images using ORB and FLANN.

    Args:
        imgL: Left grayscale image
        imgR: Right grayscale image
        config: Feature matching configuration (uses defaults if None)

    Returns:
        Tuple of (matches, keypoints1, keypoints2)
        - matches: List of cv2.DMatch objects (filtered matches)
        - keypoints1: List of keypoints from left image
        - keypoints2: List of keypoints from right image
    """
    if config is None:
        config = DEFAULT_FEATURE_CONFIG

    # Create ORB detector
    orb = cv2.ORB_create(  # type: ignore[attr-defined]
        nfeatures=config.n_features,
        scaleFactor=config.scale_factor,
        nlevels=config.n_levels,
        edgeThreshold=config.edge_threshold,
        firstLevel=config.first_level,
        WTA_K=config.wta_k,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=config.patch_size,
        fastThreshold=config.fast_threshold,
    )

    # Detect and compute features
    keypoints1, descriptors1 = orb.detectAndCompute(imgL, None)
    keypoints2, descriptors2 = orb.detectAndCompute(imgR, None)

    if descriptors1 is None or descriptors2 is None:
        print("Warning: No descriptors found in one or both images")
        return [], keypoints1, keypoints2

    # Use FLANN matcher optimized for ORB (LSH algorithm)
    index_params: dict = dict(  # type: ignore[type-arg]
        algorithm=config.flann_algorithm,
        table_number=config.flann_table_number,
        key_size=config.flann_key_size,
        multi_probe_level=config.flann_multi_probe_level,
    )
    search_params: dict = {}  # type: ignore[type-arg]
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find best 2 matches for Lowe's ratio test
    try:
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    except cv2.error as e:
        print(f"FLANN matching error: {e}")
        return [], keypoints1, keypoints2

    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) < 2:
            continue
        m, n = match_pair
        if m.distance < config.lowe_ratio * n.distance:
            good_matches.append(m)

    # Filter for matches with positive horizontal disparity and minimal vertical disparity
    filtered_matches = []
    for match in good_matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt

        # Check vertical alignment (epipolar constraint for rectified images)
        y_diff = abs(pt1[1] - pt2[1])
        if y_diff > config.max_y_diff:
            continue

        # Check positive horizontal disparity (left point should be to the right of right point)
        if pt1[0] <= pt2[0]:
            continue

        filtered_matches.append(match)

    # Keep only the best matches
    n_keep = int(len(filtered_matches) * config.match_keep_ratio)
    filtered_matches = sorted(filtered_matches, key=lambda x: x.distance)[:n_keep]

    print(
        f"Found {len(keypoints1)} and {len(keypoints2)} keypoints, "
        f"{len(good_matches)} good matches, {len(filtered_matches)} after filtering"
    )

    return filtered_matches, keypoints1, keypoints2


def draw_matches(
    imgL: np.ndarray,
    imgR: np.ndarray,
    matches: List[cv2.DMatch],
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    draw_keypoints: bool = False,
) -> np.ndarray:
    """
    Draw feature matches between two images with color-coded disparity.

    Args:
        imgL: Left grayscale image
        imgR: Right grayscale image
        matches: List of matches
        kp1: Keypoints from left image
        kp2: Keypoints from right image
        draw_keypoints: If True, draw all keypoints from left image

    Returns:
        Visualization image showing matches with anaglyph effect (red/cyan)
    """
    # Create red/cyan anaglyph overlay
    imgL_red = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
    imgR_cyan = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)

    # Make left image red and right image cyan
    imgL_red[:, :, 0] = 0
    imgL_red[:, :, 1] = 0
    imgR_cyan[:, :, 2] = 0
    img = cv2.addWeighted(imgL_red, 1, imgR_cyan, 1, 0)

    # Optionally draw keypoints
    if draw_keypoints:
        img = cv2.drawKeypoints(img, kp1, img, color=(0, 0, 255))  # type: ignore[call-overload]

    # Calculate disparity range for color mapping
    disparities = []
    for match in matches:
        x1, y1 = kp1[match.queryIdx].pt
        x2, y2 = kp2[match.trainIdx].pt
        dist = math.hypot(x1 - x2, y1 - y2)
        disparities.append(dist)

    if not disparities:
        return img

    min_dist = min(disparities)
    max_dist = max(disparities)
    dist_range = max_dist - min_dist if max_dist > min_dist else 1.0

    # Draw matches with color-coded disparity
    for match in matches:
        x1, y1 = kp1[match.queryIdx].pt
        x2, y2 = kp2[match.trainIdx].pt

        dist = math.hypot(x1 - x2, y1 - y2)

        # Map disparity to hue (0=red, 180=cyan)
        hue = int(180 * (dist - min_dist) / dist_range)
        hsv_pixel = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        line_color_bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]  # type: ignore[call-overload]
        line_color = tuple(int(c) for c in line_color_bgr)

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.line(img, (x1, y1), (x2, y2), line_color, 2)

    return img


def feature_disparity(
    imgL: np.ndarray, imgR: np.ndarray, config: Optional[FeatureMatchConfig] = None
) -> np.ndarray:
    """
    Create a sparse disparity map from feature matches.

    Args:
        imgL: Left grayscale image
        imgR: Right grayscale image
        config: Feature matching configuration (uses defaults if None)

    Returns:
        Sparse disparity map (same size as input images, float32)
    """
    matches, kp1, kp2 = find_matches(imgL, imgR, config)

    # Create empty disparity map
    depth_map = np.zeros_like(imgL, dtype=np.float32)

    # Fill in disparity values at keypoint locations
    for match in reversed(matches):
        x1, y1 = kp1[match.queryIdx].pt
        x2, y2 = kp2[match.trainIdx].pt
        disparity = x1 - x2

        # Store disparity at keypoint location
        depth_map[int(y1), int(x1)] = disparity

    return depth_map


def visualize_matches(
    imgL: np.ndarray,
    imgR: np.ndarray,
    config: Optional[FeatureMatchConfig] = None,
    draw_keypoints: bool = False,
) -> None:
    """
    Find and visualize feature matches between stereo images.

    Args:
        imgL: Left grayscale image
        imgR: Right grayscale image
        config: Feature matching configuration (uses defaults if None)
        draw_keypoints: If True, draw all keypoints
    """
    matches, kp1, kp2 = find_matches(imgL, imgR, config)

    img = draw_matches(imgL, imgR, matches, kp1, kp2, draw_keypoints)

    cv2.imshow("Feature Matches", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def estimate_baseline_from_matches(
    imgL: np.ndarray,
    imgR: np.ndarray,
    known_object_width: float,
    object_disparity: float,
    focal_length: float,
    config: Optional[FeatureMatchConfig] = None,
) -> Optional[float]:
    """
    Estimate camera baseline from feature matches of a known object.

    Args:
        imgL: Left grayscale image
        imgR: Right grayscale image
        known_object_width: Known width of the object in meters
        object_disparity: Measured disparity of the object in pixels
        focal_length: Focal length of the camera in pixels
        config: Feature matching configuration (uses defaults if None)

    Returns:
        Estimated baseline distance in meters

    Note:
        Uses the formula: baseline = (object_width * focal_length) / disparity
    """
    matches, kp1, kp2 = find_matches(imgL, imgR, config)

    if len(matches) < 10:
        print("Warning: Too few matches for reliable baseline estimation")
        return None

    # Calculate median disparity from matches
    disparities = []
    for match in matches:
        x1 = kp1[match.queryIdx].pt[0]
        x2 = kp2[match.trainIdx].pt[0]
        disparity = x1 - x2
        if disparity > 0:
            disparities.append(disparity)

    if not disparities:
        print("Warning: No valid disparities found")
        return None

    median_disparity = np.median(disparities)

    # Estimate baseline using known object
    baseline = (known_object_width * focal_length) / object_disparity

    print(f"Median disparity from matches: {median_disparity:.2f} pixels")
    print(f"Estimated baseline: {baseline:.4f} meters")

    return baseline


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path

    if len(sys.argv) < 3:
        print("Usage: python feature_match.py <left_image> <right_image>")
        sys.exit(1)

    left_path = sys.argv[1]
    right_path = sys.argv[2]

    if not Path(left_path).exists() or not Path(right_path).exists():
        print("Error: One or both image files not found")
        sys.exit(1)

    # Load images
    imgL = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    if imgL is None or imgR is None:
        print("Error: Failed to load images")
        sys.exit(1)

    print(f"Loaded images: {imgL.shape}")

    # Visualize matches
    visualize_matches(imgL, imgR, draw_keypoints=False)
