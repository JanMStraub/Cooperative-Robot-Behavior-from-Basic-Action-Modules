#!/usr/bin/env python3
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
        OperationRelationship,
        OperationResult,
    )
except ImportError:
    from operations.Base import (
        BasicOperation,
        OperationCategory,
        OperationComplexity,
        OperationParameter,
        OperationRelationship,
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
    request_id: int = 0,
) -> OperationResult:
    """
    Detect objects in a camera image using YOLO.

    Retrieves the latest image from ImageStorage and runs YOLO detection.
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
        description="Detect objects in camera image using YOLO",
        long_description="""
            This perception operation detects objects (cubes, robot parts, field markers)
            in a camera image using YOLO object detection.

            The operation:
            1. Retrieves the latest image from ImageStorage for the specified camera
            2. Runs YOLO inference on the image
            3. Filters detections by confidence and area thresholds
            4. Returns bounding boxes and pixel coordinates

            This returns 2D pixel coordinates only. For 3D world coordinates,
            use detect_object_stereo which uses stereo vision.
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
        preconditions=[],
        postconditions=[],
        average_duration_ms=50,
        success_rate=0.95,
        failure_modes=[
            "No image available - camera not sending images",
            "No objects detected - empty scene or objects not in YOLO training classes",
            "Poor lighting - affects YOLO detection accuracy",
        ],
        relationships=OperationRelationship(
            operation_id="perception_detect_objects_001",
            required_operations=[],
            commonly_paired_with=[
                "motion_move_to_coord_001",
                "manipulation_control_gripper_001",
            ],
            pairing_reasons={
                "motion_move_to_coord_001": "Move robot toward detected object coordinates",
                "manipulation_control_gripper_001": "Grip detected object after navigation",
            },
            typical_after=["status_check_robot_001"],
            typical_before=["motion_move_to_coord_001"],
        ),
        implementation=detect_objects,
    )


# ============================================================================
# Implementation: Estimate Distance to Object
# ============================================================================


def estimate_distance_to_object(
    robot_id: str,
    object_id: str,
    request_id: int = 0,
) -> OperationResult:
    """
    Estimate the distance from robot end effector to a detected object.

    This operation calculates the Euclidean distance between the robot's
    current end effector position and a detected object's position.

    Args:
        robot_id: Robot identifier
        object_id: ID or label of the detected object

    Returns:
        OperationResult with distance information

    Example:
        >>> result = estimate_distance_to_object("Robot1", "RedCube")
        >>> if result["success"]:
        ...     print(f"Distance: {result['result']['distance']:.3f}m")
    """
    try:
        # Import WorldState for robot and object positions
        try:
            from .WorldState import WorldState
        except ImportError:
            from operations.WorldState import WorldState

        world_state = WorldState()

        # Get robot position
        robot_state = world_state.get_robot_state(robot_id)
        if not robot_state:
            return OperationResult.error_result(
                "ROBOT_NOT_FOUND",
                f"Robot '{robot_id}' not found in world state",
                [
                    "Verify robot_id is correct",
                    "Ensure robot is registered in Unity",
                    "Check WorldStatePublisher is active",
                ],
            )

        # Get object position
        obj_state = world_state.get_object_state(object_id)
        if not obj_state:
            return OperationResult.error_result(
                "OBJECT_NOT_FOUND",
                f"Object '{object_id}' not found in world state",
                [
                    "Run detect_object first to locate the object",
                    "Verify object_id matches detection result",
                    "Check object is in scene",
                ],
            )

        # Calculate Euclidean distance
        import math

        robot_pos = robot_state.get("end_effector_position", robot_state.get("position"))  # type: ignore[union-attr]
        object_pos = obj_state.get("position")  # type: ignore[union-attr]

        if not robot_pos or not object_pos:
            return OperationResult.error_result(
                "POSITION_DATA_MISSING",
                "Robot or object position data missing",
                ["Ensure WorldStatePublisher is sending position data"],
            )

        distance = math.sqrt(
            (robot_pos["x"] - object_pos["x"]) ** 2
            + (robot_pos["y"] - object_pos["y"]) ** 2
            + (robot_pos["z"] - object_pos["z"]) ** 2
        )

        logger.info(f"Distance from {robot_id} to {object_id}: {distance:.3f}m")

        return OperationResult.success_result(
            {
                "robot_id": robot_id,
                "object_id": object_id,
                "distance": distance,
                "robot_position": robot_pos,
                "object_position": object_pos,
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.error(f"Error in estimate_distance_to_object: {e}", exc_info=True)
        return OperationResult.error_result(
            "ESTIMATION_ERROR",
            f"Distance estimation failed: {str(e)}",
            ["Check logs for details", "Verify world state is updated"],
        )


def estimate_distance_between_objects(
    object_id1: str,
    object_id2: str,
    request_id: int = 0,
) -> OperationResult:
    """
    Estimate the distance between two detected objects.

    This operation calculates the Euclidean distance between two objects
    in the scene.

    Args:
        object_id1: ID of first object
        object_id2: ID of second object

    Returns:
        OperationResult with distance information

    Example:
        >>> result = estimate_distance_between_objects("RedCube", "BlueCube")
        >>> if result["success"]:
        ...     print(f"Distance: {result['result']['distance']:.3f}m")
    """
    try:
        # Import WorldState
        try:
            from .WorldState import WorldState
        except ImportError:
            from operations.WorldState import WorldState

        world_state = WorldState()

        # Get first object position
        obj1_state = world_state.get_object_state(object_id1)
        if not obj1_state:
            return OperationResult.error_result(
                "OBJECT_NOT_FOUND",
                f"Object '{object_id1}' not found in world state",
                ["Run detect_object to locate the object", "Verify object_id"],
            )

        # Get second object position
        obj2_state = world_state.get_object_state(object_id2)
        if not obj2_state:
            return OperationResult.error_result(
                "OBJECT_NOT_FOUND",
                f"Object '{object_id2}' not found in world state",
                ["Run detect_object to locate the object", "Verify object_id"],
            )

        # Calculate Euclidean distance
        import math

        pos1 = obj1_state.get("position")
        pos2 = obj2_state.get("position")

        if not pos1 or not pos2:
            return OperationResult.error_result(
                "POSITION_DATA_MISSING",
                "Object position data missing",
                ["Ensure objects have been detected with 3D coordinates"],
            )

        distance = math.sqrt(
            (pos1["x"] - pos2["x"]) ** 2
            + (pos1["y"] - pos2["y"]) ** 2
            + (pos1["z"] - pos2["z"]) ** 2
        )

        logger.info(f"Distance between {object_id1} and {object_id2}: {distance:.3f}m")

        return OperationResult.success_result(
            {
                "object_id1": object_id1,
                "object_id2": object_id2,
                "distance": distance,
                "position1": pos1,
                "position2": pos2,
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.error(f"Error in estimate_distance_between_objects: {e}", exc_info=True)
        return OperationResult.error_result(
            "ESTIMATION_ERROR",
            f"Distance estimation failed: {str(e)}",
            ["Check logs for details", "Verify both objects are detected"],
        )


# ============================================================================
# BasicOperation Definitions
# ============================================================================


def create_estimate_distance_to_object_operation() -> BasicOperation:
    """Create the BasicOperation definition for estimate_distance_to_object."""
    return BasicOperation(
        operation_id="perception_distance_to_object_002",
        name="estimate_distance_to_object",
        category=OperationCategory.PERCEPTION,
        complexity=OperationComplexity.BASIC,
        description="Calculate distance from robot end effector to detected object",
        long_description="""
            This perception operation calculates the Euclidean distance between
            the robot's current end effector position and a detected object.

            Requires object to be detected with 3D world coordinates (use
            detect_object_stereo or detect_with_depth for 3D detection).
        """,
        usage_examples=[
            "estimate_distance_to_object('Robot1', 'RedCube')",
            "Check if object is within reach before attempting grasp",
        ],
        parameters=[
            OperationParameter(
                name="robot_id",
                type="str",
                description="Robot identifier",
                required=True,
            ),
            OperationParameter(
                name="object_id",
                type="str",
                description="ID or label of detected object",
                required=True,
            ),
        ],
        preconditions=[],
        postconditions=[],
        average_duration_ms=10,
        success_rate=0.99,
        failure_modes=[
            "Object not detected",
            "Robot position unavailable",
            "WorldState not updated",
        ],
        relationships=OperationRelationship(
            operation_id="perception_distance_to_object_002",
            required_operations=["perception_stereo_detect_001"],
            required_reasons={
                "perception_stereo_detect_001": "Object must be detected with 3D coordinates before distance can be calculated",
            },
            commonly_paired_with=[
                "motion_move_to_coord_001",
                "manipulation_grasp_object_001",
            ],
            pairing_reasons={
                "motion_move_to_coord_001": "Check distance before moving to verify object is within reach",
                "manipulation_grasp_object_001": "Confirm distance is within grasp range before executing grasp",
            },
            typical_after=["perception_stereo_detect_001"],
            typical_before=[
                "manipulation_grasp_object_001",
                "motion_move_to_coord_001",
            ],
        ),
        implementation=estimate_distance_to_object,
    )


def create_estimate_distance_between_objects_operation() -> BasicOperation:
    """Create BasicOperation definition for estimate_distance_between_objects."""
    return BasicOperation(
        operation_id="perception_distance_between_objects_003",
        name="estimate_distance_between_objects",
        category=OperationCategory.PERCEPTION,
        complexity=OperationComplexity.BASIC,
        description="Calculate distance between two detected objects",
        long_description="""
            This perception operation calculates the Euclidean distance between
            two objects in the scene.

            Requires both objects to be detected with 3D world coordinates.
        """,
        usage_examples=[
            "estimate_distance_between_objects('RedCube', 'BlueCube')",
            "Check clearance between objects before placing",
        ],
        parameters=[
            OperationParameter(
                name="object_id1",
                type="str",
                description="ID of first object",
                required=True,
            ),
            OperationParameter(
                name="object_id2",
                type="str",
                description="ID of second object",
                required=True,
            ),
        ],
        preconditions=[],
        postconditions=[],
        average_duration_ms=10,
        success_rate=0.99,
        failure_modes=[
            "Objects not detected",
            "Position data missing",
        ],
        relationships=OperationRelationship(
            operation_id="perception_distance_between_objects_003",
            required_operations=["perception_stereo_detect_001"],
            required_reasons={
                "perception_stereo_detect_001": "Both objects must be detected with 3D coordinates before inter-object distance can be measured",
            },
            commonly_paired_with=[
                "manipulation_grasp_object_001",
                "motion_move_between_objects_001",
            ],
            pairing_reasons={
                "manipulation_grasp_object_001": "Verify clearance between objects before attempting grasp or placement",
                "motion_move_between_objects_001": "Measure gap before navigating between two objects",
            },
            typical_after=["perception_stereo_detect_001"],
        ),
        implementation=estimate_distance_between_objects,
    )


# ============================================================================
# Create operation instances for export
# ============================================================================

DETECT_OBJECTS_OPERATION = create_detect_objects_operation()
ESTIMATE_DISTANCE_TO_OBJECT_OPERATION = create_estimate_distance_to_object_operation()
ESTIMATE_DISTANCE_BETWEEN_OBJECTS_OPERATION = (
    create_estimate_distance_between_objects_operation()
)
