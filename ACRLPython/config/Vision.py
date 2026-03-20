#!/usr/bin/env python3
"""
Vision and Detection Configuration
====================================

YOLO detection, color-based detection, stereo vision, and image processing configuration.
"""

import os
from pathlib import Path
from typing import Optional

# Get the parent directory (ACRLPython/)
_CONFIG_DIR = Path(__file__).parent.parent.absolute()

# ============================================================================
# Image Processing
# ============================================================================

# Image age thresholds (seconds)
MIN_IMAGE_AGE = float(os.environ.get("MIN_IMAGE_AGE", "0.5"))
MAX_IMAGE_AGE = float(os.environ.get("MAX_IMAGE_AGE", "30.0"))

# Monitoring intervals (seconds)
IMAGE_CHECK_INTERVAL = float(os.environ.get("IMAGE_CHECK_INTERVAL", "1.0"))

# Duplicate detection
DUPLICATE_TIME_THRESHOLD = float(os.environ.get("DUPLICATE_TIME_THRESHOLD", "0.1"))

VISION_OPERATION_TIMEOUT = float(os.environ.get("VISION_OPERATION_TIMEOUT", "20.0"))

# ============================================================================
# YOLO vs HSV Detection Toggle
# ============================================================================

USE_YOLO = os.environ.get("USE_YOLO", "true").lower() in ("true", "1", "yes")
YOLO_MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH", str(_CONFIG_DIR / "yolo" / "models" / "field_detector.onnx")
)
YOLO_CONFIDENCE_THRESHOLD = float(os.environ.get("YOLO_CONFIDENCE_THRESHOLD", "0.5"))
YOLO_IOU_THRESHOLD = float(os.environ.get("YOLO_IOU_THRESHOLD", "0.45"))

# ============================================================================
# Color Ranges for HSV Detection
# ============================================================================

# Red color (wraps around in HSV)
RED_HSV_LOWER_1 = (0, 100, 100)
RED_HSV_UPPER_1 = (10, 255, 255)
RED_HSV_LOWER_2 = (170, 100, 100)
RED_HSV_UPPER_2 = (180, 255, 255)

# Blue color
BLUE_HSV_LOWER = (110, 100, 100)
BLUE_HSV_UPPER = (130, 255, 255)

# ============================================================================
# Detection Filters
# ============================================================================

MIN_CUBE_AREA_PX = int(os.environ.get("MIN_CUBE_AREA_PX", "400"))
MAX_CUBE_AREA_PX = int(os.environ.get("MAX_CUBE_AREA_PX", "80000"))
MIN_ASPECT_RATIO = float(os.environ.get("MIN_ASPECT_RATIO", "0.3"))
MAX_ASPECT_RATIO = float(os.environ.get("MAX_ASPECT_RATIO", "3.5"))
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.3"))

# ============================================================================
# Debug Settings
# ============================================================================

ENABLE_DEBUG_IMAGES = os.environ.get("ENABLE_DEBUG_IMAGES", "false").lower() in (
    "true",
    "1",
    "yes",
)
DEBUG_IMAGES_DIR = os.environ.get(
    "DEBUG_IMAGES_DIR", str(_CONFIG_DIR / "debug_detections")
)
# Used in vision/DepthEstimator.py: save_disparity_map_debug() checks this flag
SAVE_DEBUG_DISPARITY_MAPS = os.environ.get(
    "SAVE_DEBUG_DISPARITY_MAPS", "false"
).lower() in ("true", "1", "yes")
DEBUG_DISPARITY_DIR = os.environ.get(
    "DEBUG_DISPARITY_DIR", str(_CONFIG_DIR / "debug_detections")
)
# Used in operations/PointCloudOperations.py: saves binary PLY files (Blender: File → Import → Stanford PLY)
SAVE_DEBUG_POINT_CLOUDS = os.environ.get(
    "SAVE_DEBUG_POINT_CLOUDS", "false"
).lower() in ("true", "1", "yes")
DEBUG_POINT_CLOUD_DIR = os.environ.get(
    "DEBUG_POINT_CLOUD_DIR", str(_CONFIG_DIR / "debug_point_clouds")
)

# ============================================================================
# Camera Identity
# ============================================================================

# Default camera used for perception operations when Unity sends no camera_id.
# Override with the DEFAULT_CAMERA_ID env var to match your scene's camera name.
DEFAULT_CAMERA_ID = os.environ.get("DEFAULT_CAMERA_ID", "TableStereoCamera")

# Maximum long-edge resolution fed to YOLO. Images larger than this are
# downscaled before inference (YOLO letterboxes to 640×640 internally anyway,
# so sending e.g. a 1280×960 image just wastes preprocessing time).
# Set to 0 or "" to disable resizing and pass full-resolution images.
_yolo_size_raw = os.environ.get("YOLO_INPUT_SIZE", "640")
YOLO_INPUT_SIZE: Optional[int] = int(_yolo_size_raw) if _yolo_size_raw.strip().isdigit() and int(_yolo_size_raw) > 0 else None

# Scene-change detection for VisionProcessor.
# A downsampled thumbnail (SCENE_DIFF_THUMB_SIZE × SCENE_DIFF_THUMB_SIZE pixels)
# is computed once on store and compared cheaply each poll iteration.
# If the mean absolute difference is below SCENE_DIFF_THRESHOLD the frame is
# skipped — saving a full YOLO inference + SGBM disparity pass.
#
# Threshold calibration (pixel values 0-255, measured on this scene):
#   Static scene noise floor (JPEG Q75): MAD ≈ 0.6
#   Small object motion (robot joint):   MAD ≈ 15–50
#   Large motion (arm sweep):            MAD ≈ 80–200
# Default of 8.0 gives a 13× safety margin over the measured noise floor.
# Set SCENE_DIFF_THUMB_SIZE=0 to disable scene-change detection entirely.
_thumb_raw = os.environ.get("SCENE_DIFF_THUMB_SIZE", "64")
SCENE_DIFF_THUMB_SIZE: Optional[int] = int(_thumb_raw) if _thumb_raw.strip().isdigit() and int(_thumb_raw) > 0 else None
SCENE_DIFF_THRESHOLD = float(os.environ.get("SCENE_DIFF_THRESHOLD", "8.0"))

# ============================================================================
# Stereo Camera Configuration
# ============================================================================

DEFAULT_STEREO_BASELINE = float(os.environ.get("STEREO_BASELINE", "0.05"))  # meters
DEFAULT_STEREO_FOV = float(os.environ.get("STEREO_FOV", "60.0"))  # degrees

# Default stereo camera pose (must match Unity)
DEFAULT_STEREO_CAMERA_POSITION = [-0.025, 0.4, -0.65]
DEFAULT_STEREO_CAMERA_ROTATION = [20.0, 0.0, 0.0]  # Pitch, yaw, roll

# ============================================================================
# Vision Streaming Configuration
# ============================================================================

ENABLE_VISION_STREAMING = os.environ.get(
    "ENABLE_VISION_STREAMING", "false"
).lower() in ("true", "1", "yes")
VISION_STREAM_FPS = float(os.environ.get("VISION_STREAM_FPS", "5.0"))
# JPEG quality for any Python-side stereo image re-encoding (e.g. debug saves)
STEREO_JPEG_QUALITY = int(os.environ.get("STEREO_JPEG_QUALITY", "75"))

# ============================================================================
# Depth Estimation
# ============================================================================

# Used in vision/DepthEstimator.py: select_sgbm_preset() and estimate_depth_from_bbox()
DEFAULT_SGBM_PRESET = os.environ.get(
    "SGBM_PRESET", "medium"
)  # close, medium, far, auto
ENABLE_ADAPTIVE_SGBM = os.environ.get("ENABLE_ADAPTIVE_SGBM", "false").lower() in (
    "true",
    "1",
    "yes",
)
DEPTH_SAMPLING_STRATEGY = os.environ.get(
    "DEPTH_SAMPLING_STRATEGY", "median_inner_50pct"
)
DEPTH_SAMPLE_INNER_PERCENT = int(os.environ.get("DEPTH_SAMPLE_INNER_PERCENT", "50"))

# ============================================================================
# Stereo Validation
# ============================================================================

ENABLE_STEREO_VALIDATION = os.environ.get(
    "ENABLE_STEREO_VALIDATION", "false"
).lower() in ("true", "1", "yes")
STEREO_MAX_Y_DIFF = int(os.environ.get("STEREO_MAX_Y_DIFF", "10"))
STEREO_MAX_SIZE_RATIO = float(os.environ.get("STEREO_MAX_SIZE_RATIO", "0.3"))
STEREO_MIN_IOU = float(os.environ.get("STEREO_MIN_IOU", "0.0"))

# ============================================================================
# Object Tracking
# ============================================================================

ENABLE_OBJECT_TRACKING = os.environ.get("ENABLE_OBJECT_TRACKING", "true").lower() in (
    "true",
    "1",
    "yes",
)
TRACKING_MAX_AGE = int(os.environ.get("TRACKING_MAX_AGE", "5"))
TRACKING_MIN_IOU = float(os.environ.get("TRACKING_MIN_IOU", "0.3"))

# ============================================================================
# YOLO Advanced Configuration
# ============================================================================

# Used in vision/ObjectDetector.py: task mode ("detect" or "segment") and segmentation model path
YOLO_TASK = os.environ.get("YOLO_TASK", "detect")  # detect or segment
YOLO_SEGMENTATION_MODEL = os.environ.get(
    "YOLO_SEGMENTATION_MODEL",
    str(_CONFIG_DIR / "yolo" / "models" / "field_detector_seg.onnx"),
)

# ============================================================================
# Multi-Robot Vision
# ============================================================================

SHARED_VISION_STATE_ENABLED = os.environ.get(
    "SHARED_VISION_STATE_ENABLED", "true"
).lower() in ("true", "1", "yes")
# Used in vision/ConflictResolver.py and operations/SharedVisionState.py
OBJECT_CLAIM_TIMEOUT = float(os.environ.get("OBJECT_CLAIM_TIMEOUT", "10.0"))
CONFLICT_RESOLUTION_STRATEGY = os.environ.get(
    "CONFLICT_RESOLUTION_STRATEGY", "closest_robot"
)
CONFLICT_MIN_DISTANCE_DIFF = float(os.environ.get("CONFLICT_MIN_DISTANCE_DIFF", "0.05"))

# ============================================================================
# Visualization and Performance
# ============================================================================

ENABLE_VISION_VISUALIZATION = os.environ.get(
    "ENABLE_VISION_VISUALIZATION", "false"
).lower() in ("true", "1", "yes")
ENABLE_DISPARITY_CACHE = os.environ.get("ENABLE_DISPARITY_CACHE", "true").lower() in (
    "true",
    "1",
    "yes",
)
# Used in vision/DepthEstimator.py: cache TTL for disparity maps
DISPARITY_CACHE_TTL = float(os.environ.get("DISPARITY_CACHE_TTL", "0.5"))
ENABLE_PARALLEL_JPEG_ENCODING = os.environ.get(
    "ENABLE_PARALLEL_JPEG_ENCODING", "true"
).lower() in ("true", "1", "yes")
