#!/usr/bin/env python3
"""
VisionOperations.py - Vision-based operations for object detection and scene analysis

Provides operations that use camera images for perception:
- detect_object: Stereo detection with depth estimation
- analyze_scene: LLM vision analysis
"""

import logging
import time
from typing import Optional

from operations.Base import (
    BasicOperation,
    OperationParameter,
    OperationCategory,
    OperationComplexity,
    OperationResult,
    ParameterFlow,
    OperationRelationship,
)

# Import vision modules
try:
    from vision.ObjectDetector import CubeDetector
    from vision.AnalyzeImage import LMStudioVisionProcessor
    from vision.StereoConfig import CameraConfig
except ImportError:
    from ..vision.ObjectDetector import CubeDetector
    from ..vision.AnalyzeImage import LMStudioVisionProcessor
    from ..vision.StereoConfig import CameraConfig

# Import config
try:
    import LLMConfig as cfg
except ImportError:
    from .. import LLMConfig as cfg

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================


def color_matches(detection_color: str | None, query_color: str | None) -> bool:
    """
    Flexible color matching for both legacy and YOLO detectors.

    Supports:
    - Exact match: "blue" == "blue"
    - Partial match: "blue" matches "blue_cube"
    - Case-insensitive: "Blue" matches "blue_cube"

    Args:
        detection_color: Color from detector (e.g., "blue_cube", "blue", "red_cube")
        query_color: Color to search for (e.g., "blue", "red")

    Returns:
        True if colors match
    """
    if detection_color is None or query_color is None:
        return False

    detection_lower = detection_color.lower()
    query_lower = query_color.lower()

    # Exact match (legacy CubeDetector: "blue" == "blue")
    if detection_lower == query_lower:
        return True

    # Partial match (YOLO: "blue" in "blue_cube")
    if query_lower in detection_lower:
        return True

    return False


# ============================================================================
# Implementation: Detect Object
# ============================================================================


def detect_object(
    color: str = "blue",
    camera_id: str = "StereoCamera",
    baseline: Optional[float] = None,
    fov: Optional[float] = None,
    camera_position: Optional[list] = None,
    camera_rotation: Optional[list] = None,
    **kwargs,
) -> OperationResult:
    """
    Detect a colored object using stereo cameras and return 3D coordinates.

    Args:
        color: Color to detect (red, green, blue)
        camera_id: Stereo camera pair ID
        baseline: Stereo camera baseline in meters
        fov: Camera field of view in degrees
        camera_position: Camera position [x, y, z] in world space
        camera_rotation: Camera rotation [pitch, yaw, roll] in degrees

    Returns:
        OperationResult with 3D coordinates
    """
    if baseline is None:
        baseline = cfg.DEFAULT_STEREO_BASELINE
    if fov is None:
        fov = cfg.DEFAULT_STEREO_FOV
    if camera_position is None:
        camera_position = cfg.DEFAULT_STEREO_CAMERA_POSITION
    if camera_rotation is None:
        camera_rotation = cfg.DEFAULT_STEREO_CAMERA_ROTATION

    try:
        # Get image storage and command broadcaster
        from servers.ImageServer import UnifiedImageStorage
        from servers.CommandServer import get_command_broadcaster

        storage = UnifiedImageStorage()
        broadcaster = get_command_broadcaster()

        # Request stereo image capture from Unity
        request_id = int(time.time() * 1000) % (2**32)
        request_time = time.time()
        logger.info(
            f"Requesting stereo capture from {camera_id} (request_id={request_id})"
        )

        capture_command = {
            "command_type": "capture_stereo_images",
            "target_type": "camera",
            "camera_id": camera_id,
            "request_id": request_id,
        }
        broadcaster.send_command(capture_command, request_id)

        # Wait for images to arrive (poll with timeout)
        timeout = 20.0
        poll_interval = 0.1
        start_time = time.time()
        stereo_data = None

        while time.time() - start_time < timeout:
            stereo_data = storage.get_stereo_pair(camera_id)
            if stereo_data is not None:
                # Check if this is a fresh image (received after our request)
                receive_time = storage.get_stereo_timestamp(camera_id)
                if receive_time is not None and receive_time > request_time:
                    age = time.time() - receive_time
                    logger.info(
                        f"Received stereo images from {camera_id} (age={age:.2f}s)"
                    )
                    break
            time.sleep(poll_interval)

        if stereo_data is None:
            # List available cameras for better debugging
            available = (
                storage.get_all_stereo_ids()
                if hasattr(storage, "get_all_stereo_ids")
                else []
            )
            hints = [
                "Unity may not have received the capture request",
                f"Check that StereoCameraController '{camera_id}' exists in Unity",
                "Ensure PythonCommandHandler is processing 'capture_stereo_images' command",
            ]
            if available:
                hints.append(f"Available stereo cameras: {available}")
            return OperationResult.error_result(
                "NO_IMAGES",
                f"Timeout waiting for stereo images from {camera_id}",
                hints,
            )

        imgL, imgR, prompt = stereo_data

        # Get metadata from storage (contains camera pose from Unity)
        metadata = storage.get_stereo_metadata(camera_id)
        logger.info(f"Metadata for {camera_id}: {metadata}")
        if metadata:
            # Use values from Unity metadata
            if "baseline" in metadata and metadata["baseline"] is not None:
                baseline = float(metadata["baseline"])
            if "fov" in metadata and metadata["fov"] is not None:
                fov = float(metadata["fov"])
            if (
                "camera_position" in metadata
                and metadata["camera_position"] is not None
            ):
                camera_position = metadata["camera_position"]
            if (
                "camera_rotation" in metadata
                and metadata["camera_rotation"] is not None
            ):
                camera_rotation = metadata["camera_rotation"]
            logger.info(
                f"Using metadata from Unity: pos={camera_position}, rot={camera_rotation}"
            )
        else:
            logger.warning(
                f"No metadata received from Unity, using defaults: pos={camera_position}, rot={camera_rotation}"
            )

        # Run detection using CubeDetector with stereo mode
        detector = CubeDetector()
        camera_config = CameraConfig(baseline=float(baseline), fov=float(fov))

        detection_result = detector.detect_objects_stereo(
            imgL,
            imgR,
            camera_config,
            camera_id=camera_id,
            camera_rotation=camera_rotation,
            camera_position=camera_position,
        )

        if not detection_result.detections:
            return OperationResult.error_result(
                "NO_DETECTIONS",
                "No objects detected in scene",
                ["Ensure objects are visible", "Check lighting conditions"],
            )

        # Filter by color (flexible matching for both CubeDetector and YOLODetector)
        matching = [
            d for d in detection_result.detections if color_matches(d.color, color)
        ]
        if not matching:
            detected_colors = [d.color for d in detection_result.detections]
            return OperationResult.error_result(
                "COLOR_NOT_FOUND",
                f"No {color} objects detected",
                [
                    f"Looking for {color} objects",
                    f"Detected colors: {detected_colors}",
                    "Check color parameter",
                ],
            )

        # Get leftmost detection (smallest center_x coordinate)
        best = min(matching, key=lambda d: d.center_x)
        logger.info(
            f"Selected leftmost {color} cube from {len(matching)} detections (center_x={best.center_x})"
        )

        if best.world_position is None:
            return OperationResult.error_result(
                "NO_DEPTH",
                f"Could not estimate depth for {color} object",
                ["Object may be too close or too far", "Check stereo calibration"],
            )

        result = {
            "x": best.world_position[0],
            "y": best.world_position[1],
            "z": best.world_position[2],
            "color": color,
            "confidence": best.confidence,
            "camera_id": camera_id,
        }

        logger.info(
            f"Detected {color} object at ({result['x']:.3f}, {result['y']:.3f}, {result['z']:.3f})"
        )

        return OperationResult.success_result(result)

    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return OperationResult.error_result(
            "DETECTION_FAILED",
            str(e),
            ["Check camera connection", "Ensure Python environment is configured"],
        )


def create_detect_object_operation() -> BasicOperation:
    """Create the BasicOperation definition for detect_object."""
    return BasicOperation(
        operation_id="perception_detect_object_001",
        name="detect_object",
        category=OperationCategory.PERCEPTION,
        complexity=OperationComplexity.INTERMEDIATE,
        description="Detect a colored object and return its 3D world coordinates",
        long_description="""
            This operation uses stereo cameras to detect colored objects (cubes) in the scene
            and calculate their 3D world positions using depth estimation.

            The detector uses HSV color segmentation to find red, green, or blue objects,
            then computes stereo disparity to estimate depth. Results are transformed
            to world coordinates that can be used by move_to_coordinate.

            This operation is commonly used as the first step in pick-and-place sequences.
        """,
        usage_examples=[
            "Detect blue cube: detect_object(color='blue')",
            "Find red object with custom camera: detect_object(color='red', camera_id='MyStereo')",
            "Chain with movement: detect blue cube, then move to it, then close gripper",
        ],
        parameters=[
            OperationParameter(
                name="color",
                type="str",
                description="Color to detect: red, green, or blue",
                required=True,
            ),
            OperationParameter(
                name="camera_id",
                type="str",
                description="Stereo camera pair ID",
                required=False,
                default="StereoCamera",
            ),
            OperationParameter(
                name="baseline",
                type="float",
                description="Stereo camera baseline in meters",
                required=False,
                default=cfg.DEFAULT_STEREO_BASELINE,
            ),
            OperationParameter(
                name="fov",
                type="float",
                description="Camera field of view in degrees",
                required=False,
                default=cfg.DEFAULT_STEREO_FOV,
            ),
            OperationParameter(
                name="camera_position",
                type="list",
                description="Camera position [x, y, z] in world space",
                required=False,
                default=cfg.DEFAULT_STEREO_CAMERA_POSITION,
            ),
            OperationParameter(
                name="camera_rotation",
                type="list",
                description="Camera rotation [pitch, yaw, roll] in degrees",
                required=False,
                default=cfg.DEFAULT_STEREO_CAMERA_ROTATION,
            ),
        ],
        preconditions=[
            "Stereo camera must be connected and sending images",
            "Object must be visible to both cameras",
            "Object color must be red, green, or blue",
        ],
        postconditions=[
            "Returns 3D world coordinates (x, y, z) of detected object",
            "Coordinates can be used by move_to_coordinate operation",
        ],
        average_duration_ms=200.0,
        success_rate=0.9,
        failure_modes=[
            "Object not in camera view",
            "Poor lighting conditions",
            "Object too close or too far for depth estimation",
            "Wrong color specified",
        ],
        commonly_paired_with=["move_to_coordinate", "control_gripper"],
        implementation=detect_object,
    )


# ============================================================================
# Implementation: Analyze Scene
# ============================================================================


def analyze_scene(
    prompt: str = "Describe what you see",
    camera_id: str = "MainCamera",
    model: Optional[str] = None,
    **kwargs,
) -> OperationResult:
    """
    Analyze a scene using LLM vision.

    Args:
        prompt: What to analyze in the scene
        camera_id: Camera to use for analysis
        model: LLM model to use

    Returns:
        OperationResult with analysis text
    """
    if model is None:
        model = cfg.DEFAULT_LMSTUDIO_MODEL

    try:
        # Get image storage
        from servers.ImageServer import UnifiedImageStorage

        storage = UnifiedImageStorage()

        # Try to get single camera image
        image = storage.get_single_image(camera_id)

        if image is None:
            # Fallback to stereo left image
            stereo_data = storage.get_latest_stereo()
            if stereo_data:
                _, imgL, _, _ = stereo_data
                image = imgL
            else:
                return OperationResult.error_result(
                    "NO_IMAGES",
                    "No images available for analysis",
                    ["Ensure camera is connected", "Check camera_id parameter"],
                )

        # Use LMStudioVisionProcessor
        processor = LMStudioVisionProcessor(model=model)

        # Send image for analysis
        llm_result = processor.send_images(
            images=[image], camera_ids=[camera_id], prompt=prompt
        )

        response_text = llm_result.get("response", "")

        result = {
            "analysis": response_text,
            "camera_id": camera_id,
            "model": model,
            "prompt": prompt,
        }

        logger.info(f"Scene analysis completed: {response_text[:100]}...")

        return OperationResult.success_result(result)

    except Exception as e:
        logger.error(f"Scene analysis failed: {e}")
        return OperationResult.error_result(
            "ANALYSIS_FAILED",
            str(e),
            ["Check LM Studio is running", "Verify model is loaded"],
        )


def create_analyze_scene_operation() -> BasicOperation:
    """Create the BasicOperation definition for analyze_scene."""
    return BasicOperation(
        operation_id="perception_analyze_scene_001",
        name="analyze_scene",
        category=OperationCategory.PERCEPTION,
        complexity=OperationComplexity.INTERMEDIATE,
        description="Analyze the current scene using LLM vision",
        long_description="""
            This operation sends a camera image to an LLM (via LM Studio) for
            vision-based scene understanding.

            The LLM can describe objects, identify spatial relationships,
            count items, read text, and answer questions about the scene.

            Useful for high-level task planning and verification.
        """,
        usage_examples=[
            "Describe scene: analyze_scene(prompt='Describe what you see')",
            "Count objects: analyze_scene(prompt='How many cubes are on the table?')",
            "Identify colors: analyze_scene(prompt='What colors are the objects?')",
        ],
        parameters=[
            OperationParameter(
                name="prompt",
                type="str",
                description="What to analyze in the scene",
                required=True,
            ),
            OperationParameter(
                name="camera_id",
                type="str",
                description="Camera to use for analysis",
                required=False,
                default="MainCamera",
            ),
            OperationParameter(
                name="model",
                type="str",
                description="LLM model to use",
                required=False,
                default=cfg.DEFAULT_LMSTUDIO_MODEL,
            ),
        ],
        preconditions=[
            "Camera must be connected",
            "LM Studio must be running with vision model loaded",
        ],
        postconditions=[
            "Returns LLM's analysis of the scene as text",
        ],
        average_duration_ms=3000.0,
        success_rate=0.95,
        failure_modes=[
            "LM Studio not running",
            "Vision model not loaded",
            "Image too dark or blurry",
        ],
        relationships=OperationRelationship(
            operation_id="perception_analyze_scene_001",
            commonly_paired_with=[
                "perception_stereo_detect_001",
                "status_check_robot_001",
            ],
            pairing_reasons={
                "perception_stereo_detect_001": "Verify object detection results or gather additional context",
                "status_check_robot_001": "Verify robot reached target position or analyze workspace state",
            },
            typical_before=[],
            typical_after=[],
        ),
        implementation=analyze_scene,
    )


# ============================================================================
# Implementation: Unified Stereo Detection
# ============================================================================


def detect_object_stereo(
    # Primary parameters
    color: Optional[str] = None,
    camera_id: str = "StereoCamera",
    # Detection options
    request_fresh_capture: bool = True,
    min_confidence: float = 0.5,
    max_distance: Optional[float] = None,
    # Selection strategy when multiple objects found
    selection: str = "leftmost",
    # Camera configuration (optional overrides)
    baseline: Optional[float] = None,
    fov: Optional[float] = None,
    camera_position: Optional[list] = None,
    camera_rotation: Optional[list] = None,
    # Compatibility
    robot_id: Optional[str] = None,
    request_id: int = 0,
    **kwargs,
) -> OperationResult:
    """
    Unified stereo detection operation with 3D coordinate estimation.

    This operation combines the functionality of detect_object, detect_with_depth,
    and calculate_object_coordinates into a single flexible operation.

    Args:
        color: Color to detect (red, green, blue) or None for all colors
        camera_id: Stereo camera pair ID
        request_fresh_capture: True to request new images, False to use cached from ImageStorage
        min_confidence: Minimum detection confidence threshold
        max_distance: Maximum detection distance in meters (None for no limit)
        selection: Selection strategy when multiple objects found:
                  - "leftmost": Select leftmost object (smallest x)
                  - "closest": Select closest object (smallest distance)
                  - "first": Select first detection
                  - "all": Return all detections
        baseline: Stereo camera baseline in meters (override config)
        fov: Camera field of view in degrees (override config)
        camera_position: Camera position [x, y, z] in world space (override config)
        camera_rotation: Camera rotation [pitch, yaw, roll] in degrees (override config)
        robot_id: Robot ID for compatibility (not used in detection)
        request_id: Request ID for tracking

    Returns:
        OperationResult with 3D coordinates and detection info
    """
    # Set defaults from config
    if baseline is None:
        baseline = cfg.DEFAULT_STEREO_BASELINE
    if fov is None:
        fov = cfg.DEFAULT_STEREO_FOV
    if camera_position is None:
        camera_position = cfg.DEFAULT_STEREO_CAMERA_POSITION
    if camera_rotation is None:
        camera_rotation = cfg.DEFAULT_STEREO_CAMERA_ROTATION

    try:
        # Get image storage and command broadcaster
        from servers.ImageServer import UnifiedImageStorage
        from servers.CommandServer import get_command_broadcaster

        storage = UnifiedImageStorage()
        broadcaster = get_command_broadcaster()

        # Get stereo images
        if request_fresh_capture:
            # Request fresh capture from Unity
            request_time = time.time()
            logger.info(
                f"Requesting stereo capture from {camera_id} (request_id={request_id})"
            )

            capture_command = {
                "command_type": "capture_stereo_images",
                "target_type": "camera",
                "camera_id": camera_id,
                "request_id": request_id,
            }
            broadcaster.send_command(capture_command, request_id)

            # Wait for images to arrive
            timeout = 20.0
            poll_interval = 0.1
            start_time = time.time()
            stereo_data = None

            while time.time() - start_time < timeout:
                stereo_data = storage.get_stereo_pair(camera_id)
                if stereo_data is not None:
                    receive_time = storage.get_stereo_timestamp(camera_id)
                    if receive_time is not None and receive_time > request_time:
                        age = time.time() - receive_time
                        logger.info(
                            f"Received stereo images from {camera_id} (age={age:.2f}s)"
                        )
                        break
                time.sleep(poll_interval)

            if stereo_data is None:
                available = (
                    storage.get_all_stereo_ids()
                    if hasattr(storage, "get_all_stereo_ids")
                    else []
                )
                hints = [
                    "Unity may not have received the capture request",
                    f"Check that StereoCameraController '{camera_id}' exists in Unity",
                    "Ensure PythonCommandHandler is processing 'capture_stereo_images' command",
                ]
                if available:
                    hints.append(f"Available stereo cameras: {available}")
                return OperationResult.error_result(
                    "NO_IMAGES",
                    f"Timeout waiting for stereo images from {camera_id}",
                    hints,
                )
        else:
            # Use cached images from storage
            stereo_data = storage.get_stereo_pair(camera_id)
            if stereo_data is None:
                return OperationResult.error_result(
                    "NO_CACHED_IMAGES",
                    f"No cached stereo images available for {camera_id}",
                    [
                        "Request fresh capture with request_fresh_capture=True",
                        "Ensure Unity is sending stereo images",
                    ],
                )

        imgL, imgR, prompt = stereo_data

        # Get metadata from storage (contains camera pose from Unity)
        metadata = storage.get_stereo_metadata(camera_id)
        logger.info(f"Metadata for {camera_id}: {metadata}")
        if metadata:
            # Use values from Unity metadata (override defaults)
            if "baseline" in metadata and metadata["baseline"] is not None:
                baseline = float(metadata["baseline"])
            if "fov" in metadata and metadata["fov"] is not None:
                fov = float(metadata["fov"])
            if (
                "camera_position" in metadata
                and metadata["camera_position"] is not None
            ):
                camera_position = metadata["camera_position"]
            if (
                "camera_rotation" in metadata
                and metadata["camera_rotation"] is not None
            ):
                camera_rotation = metadata["camera_rotation"]
            logger.info(
                f"Using metadata from Unity: pos={camera_position}, rot={camera_rotation}"
            )
        else:
            logger.warning(
                f"No metadata received from Unity, using defaults: pos={camera_position}, rot={camera_rotation}"
            )

        # Run detection using CubeDetector with stereo mode
        detector = CubeDetector()
        camera_config = CameraConfig(baseline=float(baseline), fov=float(fov))

        detection_result = detector.detect_objects_stereo(
            imgL,
            imgR,
            camera_config,
            camera_id=camera_id,
            camera_rotation=camera_rotation,
            camera_position=camera_position,
        )

        if not detection_result.detections:
            return OperationResult.error_result(
                "NO_DETECTIONS",
                "No objects detected in scene",
                ["Ensure objects are visible", "Check lighting conditions"],
            )

        # Filter by color if specified (flexible matching for both CubeDetector and YOLODetector)
        detections = detection_result.detections
        if color is not None:
            detections = [d for d in detections if color_matches(d.color, color)]
            if not detections:
                detected_colors = [d.color for d in detection_result.detections]
                return OperationResult.error_result(
                    "COLOR_NOT_FOUND",
                    f"No {color} objects detected",
                    [
                        f"Looking for {color} objects",
                        f"Detected colors: {detected_colors}",
                        "Check color parameter",
                    ],
                )

        # Filter by confidence
        detections = [d for d in detections if d.confidence >= min_confidence]
        if not detections:
            return OperationResult.error_result(
                "LOW_CONFIDENCE",
                f"No objects detected above confidence threshold {min_confidence}",
                ["Lower min_confidence threshold", "Improve lighting conditions"],
            )

        # Filter by distance if specified
        if max_distance is not None:
            detections_with_distance = []
            for d in detections:
                if d.world_position is not None:
                    distance = (
                        d.world_position[0] ** 2
                        + d.world_position[1] ** 2
                        + d.world_position[2] ** 2
                    ) ** 0.5
                    if distance <= max_distance:
                        detections_with_distance.append(d)
            detections = detections_with_distance

            if not detections:
                return OperationResult.error_result(
                    "OUT_OF_RANGE",
                    f"No objects detected within {max_distance}m",
                    ["Increase max_distance", "Move objects closer"],
                )

        # Apply selection strategy
        if selection == "leftmost":
            best = min(detections, key=lambda d: d.center_x)
            logger.info(
                f"Selected leftmost detection from {len(detections)} (center_x={best.center_x})"
            )
        elif selection == "closest":

            def get_distance(d):
                if d.world_position is None:
                    return float("inf")
                return (
                    d.world_position[0] ** 2
                    + d.world_position[1] ** 2
                    + d.world_position[2] ** 2
                ) ** 0.5

            best = min(detections, key=get_distance)
            logger.info(f"Selected closest detection from {len(detections)}")
        elif selection == "first":
            best = detections[0]
            logger.info(f"Selected first detection from {len(detections)}")
        elif selection == "all":
            # Return all detections
            result = {
                "detections": [
                    {
                        "x": d.world_position[0] if d.world_position else None,
                        "y": d.world_position[1] if d.world_position else None,
                        "z": d.world_position[2] if d.world_position else None,
                        "color": d.color,
                        "confidence": d.confidence,
                    }
                    for d in detections
                ],
                "count": len(detections),
                "camera_id": camera_id,
            }
            logger.info(f"Returning {len(detections)} detections")
            return OperationResult.success_result(result)
        else:
            return OperationResult.error_result(
                "INVALID_SELECTION",
                f"Invalid selection strategy: {selection}",
                ["Use 'leftmost', 'closest', 'first', or 'all'"],
            )

        # Check if selected detection has world position
        if best.world_position is None:
            return OperationResult.error_result(
                "NO_DEPTH",
                f"Could not estimate depth for selected object",
                ["Object may be too close or too far", "Check stereo calibration"],
            )

        # Return single best detection
        result = {
            "x": best.world_position[0],
            "y": best.world_position[1],
            "z": best.world_position[2],
            "color": best.color,
            "confidence": best.confidence,
            "camera_id": camera_id,
            "selection": selection,
        }

        logger.info(
            f"Detected {best.color if best.color else 'object'} at ({result['x']:.3f}, {result['y']:.3f}, {result['z']:.3f})"
        )

        return OperationResult.success_result(result)

    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        return OperationResult.error_result(
            "DETECTION_FAILED",
            str(e),
            ["Check camera connection", "Ensure Python environment is configured"],
        )


def create_detect_object_stereo_operation() -> BasicOperation:
    """Create the BasicOperation definition for detect_object_stereo."""
    return BasicOperation(
        operation_id="perception_stereo_detect_001",
        name="detect_object_stereo",
        category=OperationCategory.PERCEPTION,
        complexity=OperationComplexity.INTERMEDIATE,
        description="Unified stereo detection with 3D coordinates, combining all stereo detection capabilities",
        long_description="""
            This unified operation combines the functionality of three previous detection operations:
            - detect_object (color-filtered, fresh capture)
            - detect_with_depth (all objects, cached images)
            - calculate_object_coordinates (Unity-side processing)

            It uses stereo cameras to detect colored objects and calculate their 3D world positions
            using disparity-based depth estimation. Results are transformed to world coordinates
            that can be used by move_to_coordinate and other navigation operations.

            Key features:
            - Optional color filtering (red, green, blue, or all)
            - Fresh capture or cached images
            - Confidence and distance filtering
            - Multiple selection strategies (leftmost, closest, first, all)
            - Camera pose metadata from Unity for accurate world coordinates
        """,
        usage_examples=[
            "Detect blue cube (default): detect_object_stereo(color='blue')",
            "Detect all objects: detect_object_stereo(color=None)",
            "Use cached images: detect_object_stereo(color='red', request_fresh_capture=False)",
            "Get all detections: detect_object_stereo(color=None, selection='all')",
            "High confidence only: detect_object_stereo(color='blue', min_confidence=0.8)",
            "Nearby objects only: detect_object_stereo(color=None, max_distance=1.0)",
        ],
        parameters=[
            OperationParameter(
                name="color",
                type="str",
                description="Color to detect (None for all colors)",
                required=False,
                default=None,
                valid_values=["red", "green", "blue", None],
            ),
            OperationParameter(
                name="camera_id",
                type="str",
                description="Stereo camera pair ID",
                required=False,
                default="StereoCamera",
            ),
            OperationParameter(
                name="request_fresh_capture",
                type="bool",
                description="True to request new images, False to use cached from ImageStorage",
                required=False,
                default=True,
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
                description="Maximum detection distance in meters (None for no limit)",
                required=False,
                default=None,
            ),
            OperationParameter(
                name="selection",
                type="str",
                description="Selection strategy when multiple objects found",
                required=False,
                default="leftmost",
                valid_values=["leftmost", "closest", "first", "all"],
            ),
        ],
        preconditions=[
            "Stereo camera must be connected and configured",
            "Objects must be visible to both cameras",
            "Unity is running with stereo camera system",
        ],
        postconditions=[
            "Returns 3D world coordinates (x, y, z) of detected object(s)",
            "Coordinates can be used by move_to_coordinate operation",
        ],
        average_duration_ms=200.0,
        success_rate=0.9,
        failure_modes=[
            "Object not in camera view",
            "Poor lighting conditions",
            "Object too close or too far for depth estimation",
            "No objects match color filter",
            "All detections below confidence threshold",
        ],
        commonly_paired_with=["move_to_coordinate", "control_gripper"],
        relationships=OperationRelationship(
            operation_id="perception_stereo_detect_001",
            commonly_paired_with=[
                "motion_move_to_coord_001",
                "manipulation_control_gripper_001",
                "spatial_move_relative_001",
            ],
            pairing_reasons={
                "motion_move_to_coord_001": "Move robot to detected object's 3D position",
                "manipulation_control_gripper_001": "Grasp object after positioning at detected coordinates",
                "spatial_move_relative_001": "Move relative to detected object position (left_of, above, etc.)",
            },
            parameter_flows=[
                ParameterFlow(
                    source_operation="perception_stereo_detect_001",
                    source_output_key="x",
                    target_operation="motion_move_to_coord_001",
                    target_input_param="x",
                    description="Object X coordinate in world space for robot positioning",
                ),
                ParameterFlow(
                    source_operation="perception_stereo_detect_001",
                    source_output_key="y",
                    target_operation="motion_move_to_coord_001",
                    target_input_param="y",
                    description="Object Y coordinate in world space for robot positioning",
                ),
                ParameterFlow(
                    source_operation="perception_stereo_detect_001",
                    source_output_key="z",
                    target_operation="motion_move_to_coord_001",
                    target_input_param="z",
                    description="Object Z coordinate in world space for robot positioning",
                ),
                ParameterFlow(
                    source_operation="perception_stereo_detect_001",
                    source_output_key="x",
                    target_operation="spatial_move_relative_001",
                    target_input_param="object_ref",
                    description="Object position for spatial relative movement",
                ),
            ],
            typical_before=[
                "motion_move_to_coord_001",
                "manipulation_control_gripper_001",
            ],
            typical_after=[],
        ),
        implementation=detect_object_stereo,
    )


# ============================================================================
# Operation Instances (imported by Registry)
# ============================================================================


# Create operation instances
DETECT_OBJECT_OPERATION = create_detect_object_operation()
ANALYZE_SCENE_OPERATION = create_analyze_scene_operation()
DETECT_OBJECT_STEREO_OPERATION = create_detect_object_stereo_operation()
