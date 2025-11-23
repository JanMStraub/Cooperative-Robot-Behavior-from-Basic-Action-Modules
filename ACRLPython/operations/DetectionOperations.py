"""
Detection Operations
====================

This module provides detection operations that can be executed through
the SequenceServer. These operations detect objects in camera images
stored in ImageStorage and return detection results directly.
"""

import time
import logging
import cv2
import numpy as np
from typing import List, Optional

from servers.StreamingServer import ImageStorage
from servers.ResultsServer import ResultsBroadcaster
from .Base import (
    BasicOperation,
    OperationCategory,
    OperationComplexity,
    OperationParameter,
    OperationResult,
)

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Implementation: detect_objects - Single camera object detection
# ============================================================================


def detect_objects(
    robot_id: str,
    camera_id: str = "main",
) -> OperationResult:
    """
    Detect objects in a camera image using color-based detection.

    Retrieves the latest image from ImageStorage and runs detection.
    Returns pixel coordinates of detected objects.

    Args:
        robot_id: Robot identifier (for context, not used in detection)
        camera_id: Camera ID to retrieve image from ImageStorage

    Returns:
        OperationResult with detection data
    """
    try:
        # Get image from storage
        image_storage = ImageStorage.get_instance()
        image = image_storage.get_camera_image(camera_id)

        if image is None:
            return OperationResult.error_result(
                "NO_IMAGE",
                f"No image available from camera '{camera_id}'",
                [
                    "Ensure Unity is sending images via UnifiedPythonSender",
                    "Check that StreamingServer is running (port 5005)",
                    f"Verify camera_id '{camera_id}' is correct",
                ],
            )

        # Import and run detector
        try:
            from vision.ObjectDetector import CubeDetector
        except ImportError:
            from ..vision.ObjectDetector import CubeDetector

        detector = CubeDetector()
        result = detector.detect_cubes(image, camera_id=camera_id)

        # Convert to dictionary format
        detections = [det.to_dict() for det in result.detections]

        logger.info(f"Detected {len(detections)} objects from camera '{camera_id}'")

        return OperationResult.success_result({
            "camera_id": camera_id,
            "detections": detections,
            "count": len(detections),
            "image_width": result.image_width,
            "image_height": result.image_height,
            "timestamp": time.time(),
        })

    except Exception as e:
        logger.error(f"Error in detect_objects: {e}", exc_info=True)
        return OperationResult.error_result(
            "DETECTION_ERROR",
            f"Detection failed: {str(e)}",
            [
                "Check logs for details",
                "Verify image format is correct",
                "Ensure ObjectDetector is properly configured",
            ],
        )


# ============================================================================
# Implementation: detect_with_depth - Stereo camera 3D detection
# ============================================================================


def detect_with_depth(
    robot_id: str,
    left_camera: str = "left",
    right_camera: str = "right",
    baseline: float = 0.1,
    fov: float = 60.0,
) -> OperationResult:
    """
    Detect objects and calculate 3D world positions using stereo vision.

    Retrieves left and right camera images from ImageStorage, runs detection,
    and computes 3D coordinates through disparity-based depth estimation.

    Args:
        robot_id: Robot identifier (for context)
        left_camera: Left camera ID in ImageStorage
        right_camera: Right camera ID in ImageStorage
        baseline: Camera baseline distance in meters
        fov: Camera field of view in degrees

    Returns:
        OperationResult with 3D detection data
    """
    try:
        # Get stereo images from storage
        image_storage = ImageStorage.get_instance()
        left_image = image_storage.get_camera_image(left_camera)
        right_image = image_storage.get_camera_image(right_camera)

        if left_image is None:
            return OperationResult.error_result(
                "NO_LEFT_IMAGE",
                f"No image available from left camera '{left_camera}'",
                [
                    "Ensure Unity is sending stereo images",
                    f"Verify left camera_id '{left_camera}' is correct",
                ],
            )

        if right_image is None:
            return OperationResult.error_result(
                "NO_RIGHT_IMAGE",
                f"No image available from right camera '{right_camera}'",
                [
                    "Ensure Unity is sending stereo images",
                    f"Verify right camera_id '{right_camera}' is correct",
                ],
            )

        # Import detector and config
        try:
            from vision.ObjectDetector import CubeDetector
            from vision.StereoConfig import CameraConfig
        except ImportError:
            from ..vision.ObjectDetector import CubeDetector
            from ..vision.StereoConfig import CameraConfig

        # Create camera config
        camera_config = CameraConfig(
            baseline=baseline,
            fov=fov,
            image_width=left_image.shape[1],
            image_height=left_image.shape[0],
        )

        # Run stereo detection
        detector = CubeDetector()
        result = detector.detect_cubes_stereo(
            left_image,
            right_image,
            camera_config,
            camera_id="stereo",
        )

        # Convert to dictionary format
        detections_3d = [det.to_dict() for det in result.detections]

        logger.info(f"Detected {len(detections_3d)} objects with 3D positions")

        return OperationResult.success_result({
            "detections_3d": detections_3d,
            "count": len(detections_3d),
            "camera_config": {
                "baseline": baseline,
                "fov": fov,
            },
            "image_width": result.image_width,
            "image_height": result.image_height,
            "timestamp": time.time(),
        })

    except Exception as e:
        logger.error(f"Error in detect_with_depth: {e}", exc_info=True)
        return OperationResult.error_result(
            "STEREO_DETECTION_ERROR",
            f"Stereo detection failed: {str(e)}",
            [
                "Check logs for details",
                "Verify stereo images are synchronized",
                "Ensure camera baseline and FOV are correct",
            ],
        )


# ============================================================================
# BasicOperation definitions for RAG/registry
# ============================================================================


def create_detect_objects_operation() -> BasicOperation:
    """Create the BasicOperation definition for detect_objects."""
    return BasicOperation(
        operation_id="perception_detect_objects_001",
        name="detect_objects",
        category=OperationCategory.PERCEPTION,
        complexity=OperationComplexity.BASIC,

        description="Detect colored objects in camera image using HSV color segmentation",

        long_description="""
            This perception operation detects colored cubes (red, blue) in a camera
            image using HSV color segmentation and contour detection.

            The operation:
            1. Retrieves the latest image from ImageStorage for the specified camera
            2. Converts to HSV color space
            3. Applies color masks for red and blue
            4. Finds contours and filters by size/aspect ratio
            5. Returns bounding boxes and pixel coordinates

            This returns 2D pixel coordinates only. For 3D world coordinates,
            use detect_with_depth which uses stereo vision.
        """,

        usage_examples=[
            "detect_objects('Robot1', 'main') - Detect objects from main camera",
            "detect_objects('Robot1', 'overhead') - Detect from overhead camera",
        ],

        parameters=[
            OperationParameter(
                name="robot_id",
                type="str",
                description="Robot identifier (for context)",
                required=True,
            ),
            OperationParameter(
                name="camera_id",
                type="str",
                description="Camera ID to retrieve image from ImageStorage",
                required=False,
                default="main",
            ),
        ],

        preconditions=[
            "Camera image available in ImageStorage",
            "StreamingServer is running and receiving images",
        ],

        postconditions=[
            "Detection results returned with pixel coordinates",
            "Each detection includes color, bounding box, and confidence",
        ],

        average_duration_ms=50,
        success_rate=0.95,

        failure_modes=[
            "No image available - camera not sending images",
            "No objects detected - empty scene or wrong colors",
            "Poor lighting - affects color detection accuracy",
        ],

        required_operations=[],
        commonly_paired_with=["navigation_move_001", "manipulation_grip_001"],
        mutually_exclusive_with=[],

        implementation=detect_objects,
    )


def create_detect_with_depth_operation() -> BasicOperation:
    """Create the BasicOperation definition for detect_with_depth."""
    return BasicOperation(
        operation_id="perception_detect_depth_001",
        name="detect_with_depth",
        category=OperationCategory.PERCEPTION,
        complexity=OperationComplexity.INTERMEDIATE,

        description="Detect objects and calculate 3D world positions using stereo vision",

        long_description="""
            This perception operation uses stereo camera images to detect objects
            and compute their 3D world coordinates through disparity-based depth
            estimation.

            The operation:
            1. Retrieves left and right camera images from ImageStorage
            2. Runs color-based object detection on both images
            3. Computes disparity map between stereo pair
            4. Calculates depth from disparity using baseline and FOV
            5. Projects pixel coordinates to 3D world coordinates

            This is essential for pick-and-place tasks where you need to know
            the exact 3D position of objects in the robot's workspace.
        """,

        usage_examples=[
            "detect_with_depth('Robot1') - Detect with default stereo cameras",
            "detect_with_depth('Robot1', 'cam_L', 'cam_R', 0.12, 65) - Custom camera setup",
        ],

        parameters=[
            OperationParameter(
                name="robot_id",
                type="str",
                description="Robot identifier (for context)",
                required=True,
            ),
            OperationParameter(
                name="left_camera",
                type="str",
                description="Left camera ID in ImageStorage",
                required=False,
                default="left",
            ),
            OperationParameter(
                name="right_camera",
                type="str",
                description="Right camera ID in ImageStorage",
                required=False,
                default="right",
            ),
            OperationParameter(
                name="baseline",
                type="float",
                description="Camera baseline distance in meters",
                required=False,
                default=0.1,
                valid_range=(0.01, 1.0),
            ),
            OperationParameter(
                name="fov",
                type="float",
                description="Camera field of view in degrees",
                required=False,
                default=60.0,
                valid_range=(20.0, 120.0),
            ),
        ],

        preconditions=[
            "Stereo camera images available in ImageStorage",
            "Cameras are properly calibrated with known baseline",
            "Objects are within detection range",
        ],

        postconditions=[
            "Detection results include 3D world coordinates",
            "Each detection includes world_position (x, y, z) in meters",
        ],

        average_duration_ms=100,
        success_rate=0.90,

        failure_modes=[
            "Missing stereo images - one or both cameras not sending",
            "Image size mismatch - cameras not synchronized",
            "Poor depth estimation - objects too close/far or occluded",
            "Stereo calibration error - incorrect baseline/FOV parameters",
        ],

        required_operations=[],
        commonly_paired_with=["navigation_move_001", "manipulation_grip_001"],
        mutually_exclusive_with=[],

        implementation=detect_with_depth,
    )


# ============================================================================
# Create operation instances for export
# ============================================================================

DETECT_OBJECTS_OPERATION = create_detect_objects_operation()
DETECT_WITH_DEPTH_OPERATION = create_detect_with_depth_operation()
