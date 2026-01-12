"""
Detection Operations
====================

This module provides detection operations that can be executed through
the SequenceServer. These operations detect objects in camera images
stored in ImageStorage and return detection results directly.
"""

import time
import logging

# Import from centralized lazy import system (prevents circular dependencies)
try:
    from ..core.Imports import get_unified_image_storage
except ImportError:
    from core.Imports import get_unified_image_storage

# Handle both direct execution and package import
try:
    from .Base import (
        BasicOperation,
        OperationCategory,
        OperationComplexity,
        OperationParameter,
        OperationResult,
    )
except ImportError:
    from operations.Base import (
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
        image_storage = get_unified_image_storage()
        image = image_storage.get_single_image(camera_id)

        if image is None:
            return OperationResult.error_result(
                "NO_IMAGE",
                f"No image available from camera '{camera_id}'",
                [
                    "Ensure Unity is sending images to ImageServer",
                    "Check that ImageServer is running (port 5005)",
                    f"Verify camera_id '{camera_id}' is correct",
                ],
            )

        # Import and run detector
        try:
            from vision.ObjectDetector import CubeDetector
        except ImportError:
            from ..vision.ObjectDetector import CubeDetector

        detector = CubeDetector()
        result = detector.detect_objects(image, camera_id=camera_id)

        # Convert to dictionary format
        detections = [det.to_dict() for det in result.detections]

        logger.info(f"Detected {len(detections)} objects from camera '{camera_id}'")

        return OperationResult.success_result(
            {
                "camera_id": camera_id,
                "detections": detections,
                "count": len(detections),
                "image_width": result.image_width,
                "image_height": result.image_height,
                "timestamp": time.time(),
            }
        )

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


# ============================================================================
# Create operation instances for export
# ============================================================================

DETECT_OBJECTS_OPERATION = create_detect_objects_operation()
