"""
Depth Detection Operation
=========================

This module provides depth detection operations for stereo camera systems.
These operations are PERCEPTION category and operate on cameras/game objects,
not robots directly.
"""

from typing import List, Optional
import time
import logging
from servers.ResultsServer import ResultsBroadcaster
from .Base import (
    BasicOperation,
    OperationCategory,
    OperationComplexity,
    OperationParameter,
    OperationResult,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Implementation: Calculate Object Coordinates via Stereo Vision
# ============================================================================


def calculate_object_coordinates(
    # Required parameters - note: camera_id, not robot_id
    camera_id: str = "stereo_main",
    # Optional parameters
    object_types: Optional[List[str]] = None,
    min_confidence: float = 0.5,
    max_distance: float = 5.0,
) -> OperationResult:
    """
    Calculate 3D world coordinates of detected objects using stereo vision.

    This perception operation uses a stereo camera pair to detect objects and
    compute their 3D positions through disparity-based depth estimation. It does
    not control any robot - it only provides perception data that can be used
    by other operations.

    The operation sends a request to Unity's stereo camera system, which captures
    images from both cameras, runs object detection, and computes depth.

    Args:
        camera_id: ID of the stereo camera pair (e.g., "stereo_main", "overhead_stereo")
        object_types: List of object types to detect (e.g., ["red_cube", "green_cube"]).
                     If None, detects all supported object types.
        min_confidence: Minimum detection confidence threshold, range: [0.0, 1.0]
        max_distance: Maximum detection distance in meters, range: [0.1, 10.0]

    Returns:
        Dict with the following structure:
        {
            "success": bool,
            "result": dict or None,
            "error": dict or None
        }

        Success result structure:
        {
            "camera_id": str,
            "object_types": list,
            "min_confidence": float,
            "max_distance": float,
            "status": str,
            "timestamp": float
        }

        Error structure:
        {
            "code": str,
            "message": str,
            "recovery_suggestions": list
        }

    Example:
        >>> # Detect all objects with default settings
        >>> result = calculate_object_coordinates("stereo_main")
        >>> if result["success"]:
        ...     print(f"Detection request sent: {result['result']}")

        >>> # Detect specific object types with custom thresholds
        >>> result = calculate_object_coordinates(
        ...     camera_id="overhead_stereo",
        ...     object_types=["red_cube", "blue_cube"],
        ...     min_confidence=0.7,
        ...     max_distance=2.0
        ... )
    """
    try:
        # ==================================================================
        # VALIDATION: Validate all parameters before execution
        # ==================================================================

        # Validate camera_id
        if not camera_id or not isinstance(camera_id, str):
            return OperationResult.error_result(
                "INVALID_CAMERA_ID",
                f"Camera ID must be a non-empty string, got: {camera_id}",
                [
                    "Provide a valid camera ID (e.g., 'stereo_main', 'overhead_stereo')",
                    "Check StereoCameraController in Unity for available camera IDs",
                ],
            )

        # Validate min_confidence
        if not (0.0 <= min_confidence <= 1.0):
            return OperationResult.error_result(
                "INVALID_CONFIDENCE",
                f"min_confidence {min_confidence} out of range [0.0, 1.0]",
                [
                    "Adjust min_confidence to be between 0.0 and 1.0",
                    "Use 0.5 for balanced detection, 0.7+ for high precision",
                ],
            )

        # Validate max_distance
        if not (0.1 <= max_distance <= 10.0):
            return OperationResult.error_result(
                "INVALID_DISTANCE",
                f"max_distance {max_distance} out of range [0.1, 10.0]",
                [
                    "Adjust max_distance to be between 0.1 and 10.0 meters",
                    "Typical robot workspace is 0.5-2.0 meters",
                ],
            )

        # Validate object_types if provided
        if object_types is not None:
            if not isinstance(object_types, list):
                return OperationResult.error_result(
                    "INVALID_OBJECT_TYPES",
                    f"object_types must be a list, got: {type(object_types)}",
                    [
                        "Provide object_types as a list: ['red_cube', 'green_cube']",
                        "Use None to detect all supported object types",
                    ],
                )

        # ==================================================================
        # EXECUTION: Construct and send command
        # ==================================================================

        command = {
            "command_type": "calculate_object_coordinates",
            "target_type": "camera",  # Indicates this targets a camera, not a robot
            "camera_id": camera_id,
            "parameters": {
                "object_types": object_types,
                "min_confidence": min_confidence,
                "max_distance": max_distance,
            },
            "timestamp": time.time(),
        }

        # Send to Unity via ResultsBroadcaster
        logger.info(f"Sending depth detection command to camera {camera_id}")

        success = ResultsBroadcaster.send_result(command)

        if not success:
            return OperationResult.error_result(
                "COMMUNICATION_FAILED",
                "Failed to send command to Unity - no clients connected",
                [
                    "Ensure Unity is running with UnifiedPythonReceiver active",
                    "Verify ResultsServer is running (port 5010)",
                    "Check Unity console for connection errors",
                ],
            )

        # ==================================================================
        # SUCCESS: Return success result
        # ==================================================================

        logger.info(f"Successfully sent depth detection command to camera {camera_id}")

        return OperationResult.success_result({
            "camera_id": camera_id,
            "object_types": object_types,
            "min_confidence": min_confidence,
            "max_distance": max_distance,
            "status": "command_sent",
            "timestamp": time.time(),
        })

    except Exception as e:
        # ==================================================================
        # ERROR HANDLING: Catch unexpected errors
        # ==================================================================

        logger.error(f"Unexpected error in calculate_object_coordinates: {e}", exc_info=True)
        return OperationResult.error_result(
            "UNEXPECTED_ERROR",
            f"Unexpected error occurred: {str(e)}",
            [
                "Check logs for detailed error information",
                "Verify all parameters are correct types",
                "Retry the operation",
                "Report bug if error persists",
            ],
        )


# ============================================================================
# BasicOperation definition for RAG system
# ============================================================================


def create_calculate_object_coordinates_operation() -> BasicOperation:
    """
    Create the BasicOperation definition for calculate_object_coordinates.

    This provides rich metadata for RAG retrieval and LLM task planning.
    """
    return BasicOperation(
        # =================================================================
        # IDENTITY: Unique identifiers and categorization
        # =================================================================
        operation_id="perception_depth_detect_001",
        name="calculate_object_coordinates",
        category=OperationCategory.PERCEPTION,
        complexity=OperationComplexity.INTERMEDIATE,

        # =================================================================
        # DESCRIPTIONS: Natural language for RAG retrieval
        # =================================================================
        description="Calculate 3D world coordinates of objects using stereo vision depth estimation",

        long_description="""
            This perception operation uses a stereo camera pair to detect objects
            and compute their 3D world coordinates through disparity-based depth
            estimation.

            The operation works by:
            1. Capturing synchronized images from left and right cameras
            2. Running object detection on both images
            3. Computing disparity between matched detections
            4. Converting disparity to depth using camera baseline and focal length
            5. Projecting to 3D world coordinates

            This is a perception-only operation - it does not control any robot.
            The results can be used by navigation or manipulation operations to
            interact with detected objects.

            Typical use cases:
            - Finding objects for pick-and-place tasks
            - Obstacle detection for path planning
            - Scene understanding for task planning

            Note: This operation targets a camera system, not a robot. The camera_id
            parameter identifies which stereo camera pair to use.
        """,

        usage_examples=[
            "calculate_object_coordinates('stereo_main') - Detect all objects with default settings",
            "calculate_object_coordinates('stereo_main', object_types=['red_cube']) - Detect only red cubes",
            "calculate_object_coordinates('overhead_stereo', min_confidence=0.8, max_distance=1.5) - High precision detection in near range",
        ],

        # =================================================================
        # PARAMETERS: Define all operation parameters
        # =================================================================
        parameters=[
            OperationParameter(
                name="camera_id",
                type="str",
                description="ID of the stereo camera pair (e.g., 'stereo_main', 'overhead_stereo')",
                required=False,
                default="stereo_main",
            ),
            OperationParameter(
                name="object_types",
                type="List[str]",
                description="List of object types to detect (e.g., ['red_cube', 'green_cube']). None for all types.",
                required=False,
                default=None,
            ),
            OperationParameter(
                name="min_confidence",
                type="float",
                description="Minimum detection confidence threshold",
                required=False,
                default=0.5,
                valid_range=(0.0, 1.0),
            ),
            OperationParameter(
                name="max_distance",
                type="float",
                description="Maximum detection distance in meters",
                required=False,
                default=5.0,
                valid_range=(0.1, 10.0),
            ),
        ],

        # =================================================================
        # CONDITIONS: Pre and post conditions
        # =================================================================
        preconditions=[
            "Unity is running with StereoCameraController active",
            "Stereo camera pair is properly calibrated",
            "ResultsServer connection is established (port 5010)",
            "Objects to detect are within camera field of view",
        ],

        postconditions=[
            "Detection command has been sent to Unity via TCP",
            "Unity will process stereo images and return 3D coordinates",
            "Results will be broadcast via DepthResultsReceiver (port 5007)",
        ],

        # =================================================================
        # PERFORMANCE METRICS: For LLM decision making
        # =================================================================
        average_duration_ms=150,  # Typical stereo processing time
        success_rate=0.92,

        failure_modes=[
            "Communication failed - Unity not connected",
            "Camera not found - invalid camera_id",
            "No objects detected - empty field of view or wrong object_types",
            "Poor depth estimation - objects too close or too far",
            "Stereo calibration error - incorrect 3D coordinates",
        ],

        # =================================================================
        # RELATIONSHIPS: How this relates to other operations
        # =================================================================
        required_operations=[],  # Perception operations typically have no prerequisites

        commonly_paired_with=[
            "navigation_move_001",  # Move to detected object
            "manipulation_grip_001",  # Grasp detected object
        ],

        mutually_exclusive_with=[],  # Can run alongside other operations

        # =================================================================
        # IMPLEMENTATION: Link to the actual function
        # =================================================================
        implementation=calculate_object_coordinates,
    )


# ============================================================================
# Create the operation instance for export
# ============================================================================

CALCULATE_OBJECT_COORDINATES_OPERATION = create_calculate_object_coordinates_operation()
