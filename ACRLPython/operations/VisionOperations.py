#!/usr/bin/env python3
"""
VisionOperations.py - Vision-based operations for object detection and scene analysis

Provides operations that use camera images for perception:
- detect_object: Stereo detection with depth estimation
- analyze_scene: LLM vision analysis
"""

import logging
import time
from typing import Dict, Any, Optional

from operations.Base import (
    BasicOperation,
    OperationParameter,
    OperationCategory,
    OperationComplexity,
    OperationResult,
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
# Implementation: Detect Object
# ============================================================================


def detect_object(
    color: str = "blue",
    camera_id: str = "StereoCamera",
    baseline: Optional[float] = None,
    fov: Optional[float] = None,
    camera_position: Optional[list] = None,
    camera_rotation: Optional[list] = None,
    **kwargs
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
        logger.info(f"Requesting stereo capture from {camera_id} (request_id={request_id})")

        capture_command = {
            "command_type": "capture_stereo_images",
            "target_type": "camera",
            "camera_id": camera_id,
            "request_id": request_id
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
                    logger.info(f"Received stereo images from {camera_id} (age={age:.2f}s)")
                    break
            time.sleep(poll_interval)

        if stereo_data is None:
            # List available cameras for better debugging
            available = storage.get_all_stereo_ids() if hasattr(storage, 'get_all_stereo_ids') else []
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
                hints
            )

        imgL, imgR, prompt = stereo_data

        # Get metadata from storage (contains camera pose from Unity)
        metadata = storage.get_stereo_metadata(camera_id)
        logger.info(f"Metadata for {camera_id}: {metadata}")
        if metadata:
            # Use values from Unity metadata
            if 'baseline' in metadata and metadata['baseline'] is not None:
                baseline = float(metadata['baseline'])
            if 'fov' in metadata and metadata['fov'] is not None:
                fov = float(metadata['fov'])
            if 'camera_position' in metadata and metadata['camera_position'] is not None:
                camera_position = metadata['camera_position']
            if 'camera_rotation' in metadata and metadata['camera_rotation'] is not None:
                camera_rotation = metadata['camera_rotation']
            logger.info(f"Using metadata from Unity: pos={camera_position}, rot={camera_rotation}")
        else:
            logger.warning(f"No metadata received from Unity, using defaults: pos={camera_position}, rot={camera_rotation}")

        # Run detection using CubeDetector with stereo mode
        detector = CubeDetector()
        camera_config = CameraConfig(baseline=float(baseline), fov=float(fov))

        detection_result = detector.detect_cubes_stereo(
            imgL, imgR, camera_config, camera_id=camera_id,
            camera_rotation=camera_rotation, camera_position=camera_position
        )

        if not detection_result.detections:
            return OperationResult.error_result(
                "NO_DETECTIONS",
                "No objects detected in scene",
                ["Ensure objects are visible", "Check lighting conditions"]
            )

        # Filter by color
        matching = [d for d in detection_result.detections if d.color == color]
        if not matching:
            return OperationResult.error_result(
                "COLOR_NOT_FOUND",
                f"No {color} objects detected",
                [f"Looking for {color} objects", "Check color parameter"]
            )

        # Get leftmost detection (smallest center_x coordinate)
        best = min(matching, key=lambda d: d.center_x)
        logger.info(f"Selected leftmost {color} cube from {len(matching)} detections (center_x={best.center_x})")

        if best.world_position is None:
            return OperationResult.error_result(
                "NO_DEPTH",
                f"Could not estimate depth for {color} object",
                ["Object may be too close or too far", "Check stereo calibration"]
            )

        result = {
            "x": best.world_position[0],
            "y": best.world_position[1],
            "z": best.world_position[2],
            "color": color,
            "confidence": best.confidence,
            "camera_id": camera_id
        }

        logger.info(f"Detected {color} object at ({result['x']:.3f}, {result['y']:.3f}, {result['z']:.3f})")

        return OperationResult.success_result(result)

    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return OperationResult.error_result(
            "DETECTION_FAILED",
            str(e),
            ["Check camera connection", "Ensure Python environment is configured"]
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
    **kwargs
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
                    ["Ensure camera is connected", "Check camera_id parameter"]
                )

        # Use LMStudioVisionProcessor
        processor = LMStudioVisionProcessor(model=model)

        # Send image for analysis
        llm_result = processor.send_images(
            images=[image],
            camera_ids=[camera_id],
            prompt=prompt
        )

        response_text = llm_result.get("response", "")

        result = {
            "analysis": response_text,
            "camera_id": camera_id,
            "model": model,
            "prompt": prompt
        }

        logger.info(f"Scene analysis completed: {response_text[:100]}...")

        return OperationResult.success_result(result)

    except Exception as e:
        logger.error(f"Scene analysis failed: {e}")
        return OperationResult.error_result(
            "ANALYSIS_FAILED",
            str(e),
            ["Check LM Studio is running", "Verify model is loaded"]
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
        implementation=analyze_scene,
    )


# ============================================================================
# Operation Instances (imported by Registry)
# ============================================================================


# Create operation instances
DETECT_OBJECT_OPERATION = create_detect_object_operation()
ANALYZE_SCENE_OPERATION = create_analyze_scene_operation()
