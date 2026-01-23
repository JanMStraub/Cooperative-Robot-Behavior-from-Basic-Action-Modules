"""
Field Detection Operations
===========================

This module provides field detection operations using YOLO model trained to
recognize labeled fields (A, B, C, D, E, F, G, H, I).

YOLO model returns class names like "fielda", "fieldb", etc. with 3D world
coordinates from stereo detection.
"""

import time
import logging
from typing import Optional, Dict, Any, List

# Import from centralized lazy import system
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
# Implementation: detect_field - Detect field by label using YOLO
# ============================================================================


def detect_field(
    camera_id: str,
    field_label: str,
    confidence_threshold: float = 0.5,
) -> OperationResult:
    """
    Detect a labeled field (A-I) using YOLO model and return 3D coordinates.

    This operation uses the trained YOLO model (field_detector.onnx) to detect
    labeled fields in the camera image. The YOLO model returns class names like
    "fielda", "fieldb", etc., and stereo detection provides 3D world coordinates.

    Args:
        camera_id: Camera ID (e.g., "stereo", "main")
        field_label: Field letter to detect (A-I), case-insensitive
        confidence_threshold: Minimum detection confidence (0.0-1.0)

    Returns:
        OperationResult with field detection data including:
        - field_label: Detected field letter (uppercase)
        - center: 3D world coordinates (x, y, z) of field center
        - bounds: Bounding box in image (x, y, w, h)
        - confidence: Detection confidence score

    Example:
        >>> result = detect_field("stereo", "D")
        >>> if result["success"]:
        ...     center = result["result"]["center"]
        ...     print(f"Field D at: {center}")
    """
    try:
        # Normalize field_label to lowercase for YOLO class name
        field_label_lower = field_label.strip().lower()
        if len(field_label_lower) != 1 or not field_label_lower.isalpha():
            return OperationResult.error_result(
                "INVALID_FIELD_LABEL",
                f"Field label must be single letter A-I, got: {field_label}",
                ["Provide field label as 'A', 'B', 'C', etc."],
            )

        # Construct YOLO class name: "field" + lowercase letter
        yolo_class = f"field{field_label_lower}"

        # Import YOLO detector
        try:
            from vision.YOLODetector import YOLODetector
        except ImportError:
            from ..vision.YOLODetector import YOLODetector

        # Get stereo images from storage
        image_storage = get_unified_image_storage()
        stereo_data = image_storage.get_latest_stereo_image()

        if not stereo_data:
            return OperationResult.error_result(
                "NO_STEREO_IMAGES",
                "No stereo images available",
                [
                    "Ensure Unity is sending stereo images to ImageServer",
                    "Check that ImageServer is running (port 5006)",
                    "Verify stereo cameras are active in Unity scene",
                ],
            )

        left_image = stereo_data.get("left_image")
        right_image = stereo_data.get("right_image")

        if left_image is None or right_image is None:
            return OperationResult.error_result(
                "INCOMPLETE_STEREO_PAIR",
                "Stereo image pair incomplete",
                ["Check both stereo cameras are sending images"],
            )

        # Run YOLO detection with field class filter
        detector = YOLODetector()
        detections = detector.detect_objects_stereo(
            left_image=left_image,
            right_image=right_image,
            camera_params=stereo_data.get("camera_params", {}),
            filter_classes=[yolo_class],
            confidence_threshold=confidence_threshold,
        )

        if not detections or len(detections.detections) == 0:
            return OperationResult.error_result(
                "FIELD_NOT_DETECTED",
                f"Field '{field_label.upper()}' not detected in image",
                [
                    f"Verify field {field_label.upper()} is visible to cameras",
                    "Check lighting conditions",
                    f"Try lowering confidence_threshold (current: {confidence_threshold})",
                    "Verify YOLO model is trained for field detection",
                ],
            )

        # Get first (best) detection
        detection = detections.detections[0]

        # Extract field letter from YOLO class name ("fielda" → "A")
        detected_class = detection.class_name.lower()
        if not detected_class.startswith("field"):
            return OperationResult.error_result(
                "INVALID_DETECTION_CLASS",
                f"Unexpected class name: {detection.class_name}",
                ["Verify YOLO model is correct field detector model"],
            )

        detected_letter = detected_class[5:].upper()  # "fielda"[5:] = "a" → "A"

        # Get 3D world position from stereo detection
        world_position = detection.world_position

        if not world_position:
            return OperationResult.error_result(
                "NO_3D_COORDINATES",
                "Stereo detection did not produce 3D coordinates",
                [
                    "Check stereo camera calibration",
                    "Verify depth estimation is working",
                ],
            )

        logger.info(
            f"Detected field {detected_letter} at world position: {world_position}"
        )

        return OperationResult.success_result(
            {
                "field_label": detected_letter,
                "center": world_position,  # 3D world coordinates
                "bounds": detection.bbox,  # Bounding box in image
                "confidence": detection.confidence,
                "camera_id": camera_id,
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.error(f"Error in detect_field: {e}", exc_info=True)
        return OperationResult.error_result(
            "DETECTION_ERROR",
            f"Field detection failed: {str(e)}",
            [
                "Check logs for details",
                "Verify YOLO model is loaded correctly",
                "Ensure stereo images are available",
            ],
        )


# ============================================================================
# Implementation: get_field_center - Get field center coordinates
# ============================================================================


def get_field_center(
    field_label: str,
    camera_id: str = "stereo",
) -> OperationResult:
    """
    Get the 3D center coordinates of a labeled field.

    This is a convenience wrapper around detect_field that returns just
    the center coordinates.

    Args:
        field_label: Field letter (A-I)
        camera_id: Camera ID for detection (default: "stereo")

    Returns:
        OperationResult with center coordinates

    Example:
        >>> result = get_field_center("E")
        >>> if result["success"]:
        ...     center = result["result"]["center"]
        ...     print(f"Field E center: {center}")
    """
    try:
        # Use detect_field to get full detection
        detection_result = detect_field(camera_id, field_label)

        if not detection_result.success:
            return detection_result  # Forward error

        # Extract center coordinates
        center = detection_result.result.get("center")

        return OperationResult.success_result(
            {
                "field_label": field_label.upper(),
                "center": center,
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.error(f"Error in get_field_center: {e}", exc_info=True)
        return OperationResult.error_result(
            "OPERATION_ERROR",
            f"Failed to get field center: {str(e)}",
            ["Check logs for details"],
        )


# ============================================================================
# Implementation: detect_all_fields - Detect all visible fields
# ============================================================================


def detect_all_fields(
    camera_id: str = "stereo",
    confidence_threshold: float = 0.5,
) -> OperationResult:
    """
    Detect all visible labeled fields in the image.

    This operation detects all fields (A-I) visible in the camera view
    and returns their positions.

    Args:
        camera_id: Camera ID (default: "stereo")
        confidence_threshold: Minimum detection confidence (0.0-1.0)

    Returns:
        OperationResult with list of detected fields

    Example:
        >>> result = detect_all_fields("stereo")
        >>> if result["success"]:
        ...     fields = result["result"]["fields"]
        ...     for field in fields:
        ...         print(f"Field {field['label']} at {field['center']}")
    """
    try:
        # Import YOLO detector
        try:
            from vision.YOLODetector import YOLODetector
        except ImportError:
            from ..vision.YOLODetector import YOLODetector

        # Get stereo images
        image_storage = get_unified_image_storage()
        stereo_data = image_storage.get_latest_stereo_image()

        if not stereo_data:
            return OperationResult.error_result(
                "NO_STEREO_IMAGES",
                "No stereo images available",
                ["Ensure Unity is sending stereo images"],
            )

        left_image = stereo_data.get("left_image")
        right_image = stereo_data.get("right_image")

        if left_image is None or right_image is None:
            return OperationResult.error_result(
                "INCOMPLETE_STEREO_PAIR",
                "Stereo image pair incomplete",
                ["Check both stereo cameras are active"],
            )

        # Run YOLO detection with all field classes (fielda-fieldi)
        field_classes = [f"field{chr(ord('a') + i)}" for i in range(9)]  # fielda-fieldi

        detector = YOLODetector()
        detections = detector.detect_objects_stereo(
            left_image=left_image,
            right_image=right_image,
            camera_params=stereo_data.get("camera_params", {}),
            filter_classes=field_classes,
            confidence_threshold=confidence_threshold,
        )

        if not detections or len(detections.detections) == 0:
            return OperationResult.success_result(
                {
                    "fields": [],
                    "count": 0,
                    "camera_id": camera_id,
                    "timestamp": time.time(),
                }
            )

        # Process all detections
        fields = []
        for detection in detections.detections:
            # Extract field letter from class name
            detected_class = detection.class_name.lower()
            if detected_class.startswith("field"):
                field_letter = detected_class[5:].upper()  # "fielda" → "A"

                fields.append(
                    {
                        "label": field_letter,
                        "center": detection.world_position,
                        "bounds": detection.bbox,
                        "confidence": detection.confidence,
                    }
                )

        logger.info(f"Detected {len(fields)} fields in image")

        return OperationResult.success_result(
            {
                "fields": fields,
                "count": len(fields),
                "camera_id": camera_id,
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.error(f"Error in detect_all_fields: {e}", exc_info=True)
        return OperationResult.error_result(
            "DETECTION_ERROR",
            f"Field detection failed: {str(e)}",
            ["Check logs for details"],
        )


# ============================================================================
# BasicOperation Definitions
# ============================================================================


def create_detect_field_operation() -> BasicOperation:
    """Create the BasicOperation definition for detect_field."""
    return BasicOperation(
        operation_id="perception_detect_field_004",
        name="detect_field",
        category=OperationCategory.PERCEPTION,
        complexity=OperationComplexity.BASIC,
        description="Detect labeled field (A-I) using YOLO and return 3D coordinates",
        long_description="""
            This perception operation detects labeled fields (A-I) using a trained
            YOLO model. The model recognizes field labels and returns 3D world
            coordinates via stereo detection.

            YOLO model trained to recognize class names: fielda, fieldb, fieldc, etc.

            Critical for field-based pick-and-place operations: "Pick cube from
            field D, place on field E".
        """,
        usage_examples=[
            "detect_field('stereo', 'D') - Detect field D",
            "detect_field('stereo', 'A', confidence_threshold=0.7) - Higher confidence",
            "Use for: Pick cube from field X, place on field Y",
        ],
        parameters=[
            OperationParameter(
                name="camera_id",
                type="str",
                description="Camera ID (e.g., 'stereo', 'main')",
                required=True,
            ),
            OperationParameter(
                name="field_label",
                type="str",
                description="Field letter to detect (A-I)",
                required=True,
            ),
            OperationParameter(
                name="confidence_threshold",
                type="float",
                description="Minimum detection confidence (0.0-1.0)",
                required=False,
                default=0.5,
            ),
        ],
        preconditions=[
            "YOLO model loaded (field_detector.onnx)",
            "Stereo images available",
            "Field visible in camera view",
        ],
        postconditions=[
            "Field detected with 3D world coordinates",
            "Bounding box and confidence returned",
        ],
        average_duration_ms=100,
        success_rate=0.92,
        failure_modes=[
            "Field not visible",
            "Poor lighting",
            "YOLO model not loaded",
            "Stereo images unavailable",
        ],
        implementation=detect_field,
    )


def create_get_field_center_operation() -> BasicOperation:
    """Create the BasicOperation definition for get_field_center."""
    return BasicOperation(
        operation_id="perception_get_field_center_005",
        name="get_field_center",
        category=OperationCategory.PERCEPTION,
        complexity=OperationComplexity.BASIC,
        description="Get 3D center coordinates of a labeled field",
        long_description="""
            Convenience wrapper around detect_field that returns just the
            center coordinates of a field.

            Useful for navigation operations: move to field center before
            placing object.
        """,
        usage_examples=[
            "get_field_center('E') - Get field E center coordinates",
            "center = get_field_center('D')['result']['center']",
        ],
        parameters=[
            OperationParameter(
                name="field_label",
                type="str",
                description="Field letter (A-I)",
                required=True,
            ),
            OperationParameter(
                name="camera_id",
                type="str",
                description="Camera ID",
                required=False,
                default="stereo",
            ),
        ],
        preconditions=["Field visible", "YOLO model loaded"],
        postconditions=["Center coordinates returned"],
        average_duration_ms=100,
        success_rate=0.92,
        failure_modes=["Field not detected"],
        implementation=get_field_center,
    )


def create_detect_all_fields_operation() -> BasicOperation:
    """Create the BasicOperation definition for detect_all_fields."""
    return BasicOperation(
        operation_id="perception_detect_all_fields_006",
        name="detect_all_fields",
        category=OperationCategory.PERCEPTION,
        complexity=OperationComplexity.BASIC,
        description="Detect all visible labeled fields (A-I) in image",
        long_description="""
            This operation detects all visible fields in the camera view
            and returns their positions.

            Useful for scene understanding and multi-field operations.
        """,
        usage_examples=[
            "detect_all_fields('stereo') - Find all visible fields",
            "Get field layout for planning multi-step operations",
        ],
        parameters=[
            OperationParameter(
                name="camera_id",
                type="str",
                description="Camera ID",
                required=False,
                default="stereo",
            ),
            OperationParameter(
                name="confidence_threshold",
                type="float",
                description="Minimum confidence",
                required=False,
                default=0.5,
            ),
        ],
        preconditions=["Stereo images available", "YOLO model loaded"],
        postconditions=["List of detected fields with positions"],
        average_duration_ms=150,
        success_rate=0.90,
        failure_modes=["No fields visible", "Poor lighting"],
        implementation=detect_all_fields,
    )


# ============================================================================
# Create operation instances for export
# ============================================================================

DETECT_FIELD_OPERATION = create_detect_field_operation()
GET_FIELD_CENTER_OPERATION = create_get_field_center_operation()
DETECT_ALL_FIELDS_OPERATION = create_detect_all_fields_operation()
