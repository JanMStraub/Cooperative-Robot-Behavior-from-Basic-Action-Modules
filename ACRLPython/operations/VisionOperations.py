#!/usr/bin/env python3
"""Vision-based perception operations: stereo object detection and LLM scene analysis."""

import logging
import time
from typing import Optional, cast

from .Base import (
    BasicOperation,
    OperationParameter,
    OperationCategory,
    OperationComplexity,
    OperationResult,
    ParameterFlow,
    OperationRelationship,
)

from ._imports import get_unified_image_storage, get_command_broadcaster
from ._imports import (
    ENABLE_VISION_STREAMING,
    VISION_OPERATION_TIMEOUT,
    DEFAULT_CAMERA_ID,
    DEFAULT_LMSTUDIO_MODEL,
)

logger = logging.getLogger(__name__)


def color_matches(detection_color: Optional[str], query_color: Optional[str]) -> bool:
    if detection_color is None or query_color is None:
        return False

    detection_lower = detection_color.lower()
    query_lower = query_color.lower()

    if detection_lower == query_lower:
        return True
    if query_lower in detection_lower:
        return True
    return False


def analyze_scene(
    prompt: str = "Describe what you see",
    camera_id: str = "MainCamera",
    model: Optional[str] = None,
    **kwargs,
) -> OperationResult:
    if model is None:
        model = DEFAULT_LMSTUDIO_MODEL

    try:
        storage = get_unified_image_storage()

        stereo_data = storage.get_latest_stereo()
        if stereo_data is None:
            return OperationResult.error_result(
                "NO_IMAGES",
                "No stereo images available for analysis",
                [
                    "Ensure StereoCameraController is sending images",
                    "Check camera_id parameter",
                ],
            )
        _, image, _, _ = stereo_data

        from vision.AnalyzeImage import LMStudioVisionProcessor

        processor = LMStudioVisionProcessor(model=model)
        llm_result = processor.send_images(
            images=[image], camera_ids=[camera_id], prompt=prompt
        )

        response_text = (llm_result or {}).get("response", "")

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
    return BasicOperation(
        operation_id="perception_analyze_scene_001",
        name="analyze_scene",
        category=OperationCategory.PERCEPTION,
        complexity=OperationComplexity.INTERMEDIATE,
        description=(
            "Analyze the current scene using LLM vision. Trigger phrases: "
            "'analyze the scene', 'what do you see', 'describe the workspace', "
            "'what's on the table', 'scan the scene'. "
            "Do NOT substitute detect_object_stereo for this operation."
        ),
        usage_examples=[
            "Analyze scene: analyze_scene(prompt='Describe what you see')",
            "What objects are in the scene: analyze_scene(prompt='What objects are on the table?')",
            "Describe the scene: analyze_scene(prompt='Describe the workspace')",
            "What do you see: analyze_scene(prompt='What can you see in front of you?')",
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
                default=DEFAULT_LMSTUDIO_MODEL,
            ),
        ],
        preconditions=[],
        postconditions=[],
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


def detect_object_stereo(
    color: Optional[str] = None,
    camera_id: str = DEFAULT_CAMERA_ID,
    request_fresh_capture: bool = True,
    min_confidence: float = 0.5,
    max_distance: Optional[float] = None,
    selection: str = "left",
    baseline: Optional[float] = None,
    fov: Optional[float] = None,
    camera_position: Optional[list] = None,
    camera_rotation: Optional[list] = None,
    robot_id: Optional[str] = None,
    request_id: int = 0,
    **kwargs,
) -> OperationResult:
    if color == "None":  # LLM sometimes passes string "None"
        color = None

    # If the robot is already holding an object matching the requested color,
    # vision will fail to find it (it's in the gripper). Return WorldState position directly.
    if robot_id and color:
        try:
            from core.Imports import get_world_state as _gws

            _ws = _gws()
            _held = [
                obj
                for obj in _ws.get_all_objects()
                if obj.grasped_by == robot_id and color_matches(obj.color, color)
            ]
            if _held:
                obj = _held[0]
                logger.info(
                    f"Robot {robot_id} already holds {obj.color} object - skipping vision detection"
                )
                return OperationResult.success_result(
                    {
                        "x": obj.position[0],
                        "y": obj.position[1],
                        "z": obj.position[2],
                        "color": obj.color,
                        "confidence": obj.confidence,
                        "camera_id": camera_id,
                        "selection": selection,
                        "source": "held_object",
                    }
                )
        except Exception:
            pass  # WorldState unavailable - fall through to normal detection

    try:
        storage = get_unified_image_storage()
        broadcaster = get_command_broadcaster()

        # streaming keeps images fresh - skip fresh capture round-trip
        if ENABLE_VISION_STREAMING:
            request_fresh_capture = False

        if request_fresh_capture:
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

            timeout = VISION_OPERATION_TIMEOUT
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
                msg = f"Timeout waiting for stereo images from {camera_id}"
                if available:
                    msg += f" (available: {available})"
                return OperationResult.error_result(
                    "NO_IMAGES",
                    msg,
                    hints,
                )
        else:
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

        imgL, imgR, _ = stereo_data

        metadata = storage.get_stereo_metadata(camera_id)
        logger.debug(f"Metadata for {camera_id}: {metadata}")

        from operations.StereoUtils import camera_config_from_metadata

        stereo_params = camera_config_from_metadata(
            metadata,
            baseline=baseline,
            fov=fov,
            camera_position=camera_position,
            camera_rotation=camera_rotation,
        )
        baseline = stereo_params.camera_config.baseline
        fov = stereo_params.camera_config.fov
        camera_position = stereo_params.camera_position
        camera_rotation = stereo_params.camera_rotation
        if metadata:
            logger.info(
                f"Using metadata from Unity: pos={camera_position}, rot={camera_rotation}"
            )
        else:
            logger.warning(
                f"No metadata received from Unity, using defaults: pos={camera_position}, rot={camera_rotation}"
            )

        from vision.ObjectDetector import CubeDetector
        from vision.StereoConfig import CameraConfig

        enable_streaming = ENABLE_VISION_STREAMING
        use_cached = False
        detection_result = None

        if enable_streaming:
            try:  # try cached detections from SharedVisionState first
                from operations.SharedVisionState import SharedVisionState

                shared_state = SharedVisionState()
                cached_objects = shared_state.get_available_objects()

                if cached_objects:
                    from vision.DetectionDataModels import (
                        DetectionObject,
                        DetectionResult,
                    )

                    # Unity-streamed collider dims are more accurate than synthetic zero-area bbox
                    try:
                        from core.Imports import get_world_state as _gws

                        _ws_for_dims = _gws()
                    except Exception:
                        _ws_for_dims = None

                    cached_detections = []
                    for idx, obj in enumerate(cached_objects):
                        inherited_dims = None
                        if _ws_for_dims is not None:
                            inherited_dims = _ws_for_dims.get_object_dimensions(
                                obj.color
                            )

                        # bbox=(0,0,0,0): no pixel coords for cached results
                        det = DetectionObject(
                            object_id=idx,
                            color=obj.color,
                            bbox=(0, 0, 0, 0),
                            confidence=obj.confidence,
                            world_position=obj.world_position,
                            depth_m=obj.depth_m,
                            track_id=obj.track_id,
                            dimensions=inherited_dims,
                        )
                        cached_detections.append(det)

                    detection_result = DetectionResult(
                        camera_id=camera_id,
                        image_width=0,
                        image_height=0,
                        detections=cached_detections,
                    )
                    use_cached = True
                    logger.info(
                        f"Using cached detections from SharedVisionState "
                        f"({len(cached_detections)} objects)"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to retrieve cached detections, falling back to on-demand: {e}"
                )

        if not use_cached:
            logger.info("Running on-demand stereo detection")
            detector = CubeDetector()
            camera_config = CameraConfig(
                baseline=float(baseline),
                fov=float(fov if fov is not None else 60.0),
            )

            detection_result = detector.detect_objects_stereo(
                imgL,
                imgR,
                camera_config,
                camera_id=camera_id,
                camera_rotation=camera_rotation,
                camera_position=camera_position,
            )

            # Retry once (0.4 s delay) if nothing was found - streaming frames
            # can briefly miss a held/elevated object between captures.
            # Re-fetch the stereo pair so Unity's latest frame is used, not the
            # stale one captured before the sleep.
            if not detection_result or not detection_result.detections:
                logger.info("No detections on first attempt, retrying after 0.4s")
                time.sleep(0.4)
                _retry_stereo = storage.get_stereo_pair(camera_id)
                if _retry_stereo is not None:
                    _rL, _rR, _ = _retry_stereo
                    detection_result = detector.detect_objects_stereo(
                        _rL,
                        _rR,
                        camera_config,
                        camera_id=camera_id,
                        camera_rotation=camera_rotation,
                        camera_position=camera_position,
                    )

        if detection_result is None:
            return OperationResult.error_result(
                "DETECTION_ERROR",
                "Failed to get detection result",
                ["Internal error: detection_result was not assigned"],
            )

        if not detection_result.detections:
            return OperationResult.error_result(
                "NO_DETECTIONS",
                "No objects detected in scene",
                ["Ensure objects are visible", "Check lighting conditions"],
            )

        logger.debug(
            f"Total detections before filtering: {len(detection_result.detections)}"
        )
        for idx, d in enumerate(detection_result.detections):
            world_pos_str = (
                f"({d.world_position[0]:.3f}, {d.world_position[1]:.3f}, {d.world_position[2]:.3f})"
                if d.world_position
                else "None"
            )
            logger.debug(
                f"  Detection {idx+1}: color={d.color}, world_pos={world_pos_str}, conf={d.confidence:.2f}"
            )

        detections = detection_result.detections
        if color is not None:
            detections = [d for d in detections if color_matches(d.color, color)]
            logger.debug(
                f"After color filter ('{color}'): {len(detections)} detections"
            )
            if not detections:
                detected_colors = [d.color for d in detection_result.detections]
                return OperationResult.error_result(
                    "COLOR_NOT_FOUND",
                    f"No {color} objects detected (found: {detected_colors})",
                    [
                        f"Looking for {color} objects",
                        f"Detected colors: {detected_colors}",
                        "Check color parameter",
                    ],
                )

        detections = [d for d in detections if d.confidence >= min_confidence]
        if not detections:
            return OperationResult.error_result(
                "LOW_CONFIDENCE",
                f"No objects detected above confidence threshold {min_confidence}",
                ["Lower min_confidence threshold", "Improve lighting conditions"],
            )

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

        if selection == "left":
            valid_detections = [d for d in detections if d.world_position is not None]
            if not valid_detections:
                return OperationResult.error_result(
                    "NO_DEPTH",
                    "No detections have valid world positions",
                    ["Object may be too close or too far", "Check stereo calibration"],
                )
            best = min(valid_detections, key=lambda d: cast(tuple, d.world_position)[0])

            for idx, d in enumerate(
                sorted(valid_detections, key=lambda d: cast(tuple, d.world_position)[0])
            ):
                wp = cast(tuple, d.world_position)
                logger.debug(
                    f"Candidate {idx+1}: world_pos=({wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f}), pixel_x={d.center_x}, color={d.color}, conf={d.confidence:.2f}"
                )
            logger.debug(
                f"Selected leftmost detection from {len(detections)} (world_x={cast(tuple, best.world_position)[0]:.3f}, pixel_x={best.center_x})"
            )
        elif selection == "right":
            valid_detections = [d for d in detections if d.world_position is not None]
            if not valid_detections:
                return OperationResult.error_result(
                    "NO_DEPTH",
                    "No detections have valid world positions",
                    ["Check stereo calibration", "Objects may be too close/far"],
                )
            best = max(valid_detections, key=lambda d: cast(tuple, d.world_position)[0])
            logger.debug(
                f"Selected rightmost detection from {len(detections)} (world_x={cast(tuple, best.world_position)[0]:.3f}, pixel_x={best.center_x})"
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
            logger.debug(f"Selected closest detection from {len(detections)}")
        elif selection == "first":
            best = detections[0]
            logger.debug(f"Selected first detection from {len(detections)}")
        elif selection == "all":
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
            return OperationResult.success_result(result)
        else:
            return OperationResult.error_result(
                "INVALID_SELECTION",
                f"Invalid selection strategy: {selection}",
                ["Use 'left', 'right', 'closest', 'first', or 'all'"],
            )

        if best.world_position is None:
            return OperationResult.error_result(
                "NO_DEPTH",
                "Could not estimate depth for selected object",
                ["Check stereo calibration", "Objects may be too close/far"],
            )

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

        try:
            from core.Imports import get_world_state

            world_state = get_world_state()
            ws_object_id = best.color if best.color else "unknown_object"
            # don't overwrite field entries - those are owned by detect_field
            is_field = ws_object_id.lower().startswith("field_")
            existing = (
                world_state.get_object_state(ws_object_id) if not is_field else None
            )
            if not is_field and not (
                existing and existing.get("object_type") == "field"
            ):
                world_state.update_object_position(
                    object_id=ws_object_id,
                    position=(
                        best.world_position[0],
                        best.world_position[1],
                        best.world_position[2],
                    ),
                    color=best.color,
                    object_type="cube",
                    confidence=best.confidence,
                    dimensions=best.dimensions,
                )
                dim_str = (
                    f" dims=({best.dimensions[0]:.3f}, {best.dimensions[1]:.3f}, {best.dimensions[2]:.3f})m"
                    if best.dimensions
                    else ""
                )
                logger.info(
                    f"WorldState updated: key='{ws_object_id}' at "
                    f"({best.world_position[0]:.3f}, {best.world_position[1]:.3f}, {best.world_position[2]:.3f}){dim_str}"
                )
        except Exception as e:
            # ERROR not WARNING - WorldState write failure breaks next grasp_object call
            logger.error(
                f"Failed to update WorldState after detection: {e}", exc_info=True
            )

        # sync to KG now - don't wait for WorldStatePublisher 10Hz cycle
        try:
            from config.KnowledgeGraph import KNOWLEDGE_GRAPH_ENABLED

            if KNOWLEDGE_GRAPH_ENABLED:
                from core.Imports import get_graph_query_engine

                qe = get_graph_query_engine()
                if qe is not None:
                    kg_object_id = best.color if best.color else "unknown_object"
                    pos = (
                        best.world_position[0],
                        best.world_position[1],
                        best.world_position[2],
                    )
                    qe._graph.add_node(
                        kg_object_id,
                        node_type="object",
                        position=pos,
                        color=best.color or "unknown",
                        object_type="cube",
                        confidence=best.confidence,
                        stale=False,
                        grasped_by=None,
                    )
                    logger.debug(f"KG updated: object node '{kg_object_id}' at {pos}")
        except Exception as e:
            logger.debug(f"KG object sync skipped: {e}")

        return OperationResult.success_result(result)

    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        return OperationResult.error_result(
            "DETECTION_FAILED",
            str(e),
            ["Check camera connection", "Ensure Python environment is configured"],
        )


def create_detect_object_stereo_operation() -> BasicOperation:
    return BasicOperation(
        operation_id="perception_stereo_detect_001",
        name="detect_object_stereo",
        category=OperationCategory.PERCEPTION,
        complexity=OperationComplexity.INTERMEDIATE,
        description="Stereo detection with 3D world coordinates - optional color filter, configurable selection strategy",
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
                valid_values=[
                    "red",
                    "green",
                    "blue",
                    "yellow",
                    "purple",
                    "orange",
                    "cyan",
                    "magenta",
                    None,
                ],
            ),
            OperationParameter(
                name="camera_id",
                type="str",
                description="Stereo camera pair ID",
                required=False,
                default=DEFAULT_CAMERA_ID,
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
                default="left",
                valid_values=["left", "right", "closest", "first", "all"],
            ),
        ],
        preconditions=[],
        postconditions=[],
        average_duration_ms=200.0,
        success_rate=0.9,
        failure_modes=[
            "Object not in camera view",
            "Poor lighting conditions",
            "Object too close or too far for depth estimation",
            "No objects match color filter",
            "All detections below confidence threshold",
        ],
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
                "spatial_move_relative_001": "Move relative to detected object position (left_of, right_of, above, etc.)",
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


ANALYZE_SCENE_OPERATION = create_analyze_scene_operation()
DETECT_OBJECT_STEREO_OPERATION = create_detect_object_stereo_operation()
