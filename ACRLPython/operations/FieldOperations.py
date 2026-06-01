#!/usr/bin/env python3


"""Field detection using YOLO (field_a–field_i classes) with stereo 3D coordinates."""

import time
import logging

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
        ParameterFlow,
    )
except ImportError:
    from operations.Base import (
        BasicOperation,
        OperationCategory,
        OperationComplexity,
        OperationParameter,
        OperationRelationship,
        OperationResult,
        ParameterFlow,
    )

logger = logging.getLogger(__name__)


def detect_field(
    robot_id: str,
    field_label: str,
    camera_id: str = "stereo",
    confidence_threshold: float = 0.5,
    request_id: int = 0,
) -> OperationResult:
    """Detect a labeled field (A-I) with YOLO and return 3D world coordinates."""
    try:
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string, got: {robot_id}",
                ["Provide a valid robot ID (e.g., 'Robot1', 'AR4_Robot')"],
            )

        field_label_lower = field_label.strip().lower()
        if len(field_label_lower) != 1 or not field_label_lower.isalpha():
            return OperationResult.error_result(
                "INVALID_FIELD_LABEL",
                f"Field label must be single letter A-I, got: {field_label}",
                ["Provide field label as 'A', 'B', 'C', etc."],
            )

        yolo_class = f"field_{field_label_lower}"  # trained class names: "field_a" etc.

        try:
            from vision.YOLODetector import YOLODetector
        except ImportError:
            from ..vision.YOLODetector import YOLODetector

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

        left_image, right_image, _, _, stereo_metadata = stereo_data

        if left_image is None or right_image is None:
            return OperationResult.error_result(
                "INCOMPLETE_STEREO_PAIR",
                "Stereo image pair incomplete",
                ["Check both stereo cameras are sending images"],
            )

        try:
            from .StereoUtils import camera_config_from_metadata
        except ImportError:
            from operations.StereoUtils import camera_config_from_metadata

        stereo_params = camera_config_from_metadata(stereo_metadata)
        camera_config = stereo_params.camera_config
        camera_position = stereo_params.camera_position
        camera_rotation = stereo_params.camera_rotation

        try:
            from config.Vision import YOLO_MODEL_PATH
        except ImportError:
            from ..config.Vision import YOLO_MODEL_PATH
        detector = YOLODetector(model_path=YOLO_MODEL_PATH)
        detector.conf_threshold = confidence_threshold
        detections = detector.detect_objects_stereo(
            imgL=left_image,
            imgR=right_image,
            camera_config=camera_config,
            camera_position=camera_position,
            camera_rotation=camera_rotation,
            filter_classes=[yolo_class],
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

        detection = detections.detections[0]

        # .color holds the YOLO class string; "field_a" → letter "A"
        detected_class = detection.color.lower()
        if not detected_class.startswith("field_"):
            return OperationResult.error_result(
                "INVALID_DETECTION_CLASS",
                f"Unexpected class name: {detection.color}",
                ["Verify YOLO model is correct field detector model"],
            )

        detected_letter = detected_class[6:].upper()

        if detected_letter != field_label.strip().upper():
            return OperationResult.error_result(
                "FIELD_LABEL_MISMATCH",
                f"Requested field '{field_label.upper()}' but YOLO returned '{detected_letter}' — filter leak or model error",
                [
                    f"Verify YOLO model correctly distinguishes field_{field_label.lower()} from adjacent fields",
                    "Check that filter_classes is respected by the detector",
                ],
            )

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

        if isinstance(world_position, tuple) and len(world_position) == 3:
            center_dict = {
                "x": world_position[0],
                "y": world_position[1],
                "z": world_position[2],
            }
        elif isinstance(world_position, dict):
            center_dict = world_position
        else:
            center_dict = {"x": 0.0, "y": 0.0, "z": 0.0}

        # "field" type skips confidence decay — fields don't move
        try:
            from core.Imports import get_world_state

            world_state = get_world_state()
            ws_key = f"field_{detected_letter.lower()}"
            pos_tuple = (center_dict["x"], center_dict["y"], center_dict["z"])
            world_state.update_object_position(
                object_id=ws_key,
                position=pos_tuple,
                color=ws_key,
                object_type="field",
                confidence=detection.confidence,
            )
            logger.info(f"WorldState updated: key='{ws_key}' at {pos_tuple}")
        except Exception as e:
            logger.error(
                f"Failed to update WorldState after field detection: {e}", exc_info=True
            )

        return OperationResult.success_result(
            {
                "field_label": detected_letter,
                "center": center_dict,
                "bounds": (
                    detection.bbox_x,
                    detection.bbox_y,
                    detection.bbox_w,
                    detection.bbox_h,
                ),
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


def detect_all_fields(
    robot_id: str,
    camera_id: str = "stereo",
    confidence_threshold: float = 0.5,
    request_id: int = 0,
) -> OperationResult:
    """Detect all visible labeled fields (A-I) in the image and return their 3D positions."""
    try:
        if not robot_id or not isinstance(robot_id, str):
            return OperationResult.error_result(
                "INVALID_ROBOT_ID",
                f"Robot ID must be a non-empty string, got: {robot_id}",
                ["Provide a valid robot ID (e.g., 'Robot1', 'AR4_Robot')"],
            )
        try:
            from vision.YOLODetector import YOLODetector
        except ImportError:
            from ..vision.YOLODetector import YOLODetector

        image_storage = get_unified_image_storage()
        stereo_data = image_storage.get_latest_stereo_image()

        if not stereo_data:
            return OperationResult.error_result(
                "NO_STEREO_IMAGES",
                "No stereo images available",
                ["Ensure Unity is sending stereo images"],
            )

        left_image, right_image, _, _, stereo_metadata = stereo_data

        if left_image is None or right_image is None:
            return OperationResult.error_result(
                "INCOMPLETE_STEREO_PAIR",
                "Stereo image pair incomplete",
                ["Check both stereo cameras are active"],
            )

        try:
            from .StereoUtils import camera_config_from_metadata
        except ImportError:
            from operations.StereoUtils import camera_config_from_metadata

        stereo_params = camera_config_from_metadata(stereo_metadata)

        field_classes = [
            f"field_{chr(ord('a') + i)}" for i in range(9)
        ]  # field_a–field_i

        try:
            from config.Vision import YOLO_MODEL_PATH
        except ImportError:
            from ..config.Vision import YOLO_MODEL_PATH
        detector = YOLODetector(model_path=YOLO_MODEL_PATH)
        detector.conf_threshold = confidence_threshold
        detections = detector.detect_objects_stereo(
            imgL=left_image,
            imgR=right_image,
            camera_config=stereo_params.camera_config,
            camera_position=stereo_params.camera_position,
            camera_rotation=stereo_params.camera_rotation,
            filter_classes=field_classes,
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

        from core.Imports import get_world_state

        world_state = get_world_state()

        fields = []
        for detection in detections.detections:
            detected_class = detection.color.lower()
            if detected_class.startswith("field_"):
                field_letter = detected_class[6:].upper()

                world_pos = detection.world_position
                if isinstance(world_pos, tuple) and len(world_pos) == 3:
                    center_dict = {
                        "x": world_pos[0],
                        "y": world_pos[1],
                        "z": world_pos[2],
                    }
                elif isinstance(world_pos, dict):
                    center_dict = world_pos
                else:
                    center_dict = {"x": 0.0, "y": 0.0, "z": 0.0}

                fields.append(
                    {
                        "label": field_letter,
                        "center": center_dict,
                        "bounds": (
                            detection.bbox_x,
                            detection.bbox_y,
                            detection.bbox_w,
                            detection.bbox_h,
                        ),
                        "confidence": detection.confidence,
                    }
                )

                # Mirror detect_field: persist result to WorldState so downstream
                # operations can look up this field without re-detecting.
                try:
                    ws_key = f"field_{field_letter.lower()}"
                    pos_tuple = (center_dict["x"], center_dict["y"], center_dict["z"])
                    world_state.update_object_position(
                        object_id=ws_key,
                        position=pos_tuple,
                        color=ws_key,
                        object_type="field",
                        confidence=detection.confidence,
                    )
                    logger.info(
                        f"WorldState updated (detect_all_fields): key='{ws_key}' at {pos_tuple}"
                    )
                except Exception as exc:
                    logger.warning(
                        f"WorldState write failed for field_{field_letter.lower()}: {exc}",
                        exc_info=True,
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


def create_detect_field_operation() -> BasicOperation:
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
            "detect_field('Robot1', 'D') - Detect field D",
            "detect_field('Robot1', 'A', confidence_threshold=0.7) - Higher confidence",
            "Use for: Pick cube from field X, place on field Y",
        ],
        parameters=[
            OperationParameter(
                name="robot_id",
                type="str",
                description="Robot identifier (for context)",
                required=True,
            ),
            OperationParameter(
                name="field_label",
                type="str",
                description="Field letter to detect (A-I)",
                required=True,
            ),
            OperationParameter(
                name="camera_id",
                type="str",
                description="Camera ID (e.g., 'stereo', 'main')",
                required=False,
                default="stereo",
            ),
            OperationParameter(
                name="confidence_threshold",
                type="float",
                description="Minimum detection confidence (0.0-1.0)",
                required=False,
                default=0.5,
            ),
        ],
        preconditions=[],
        postconditions=[],
        average_duration_ms=100,
        success_rate=0.92,
        failure_modes=[
            "Field not visible",
            "Poor lighting",
            "YOLO model not loaded",
            "Stereo images unavailable",
        ],
        relationships=OperationRelationship(
            operation_id="perception_detect_field_004",
            required_operations=[],
            commonly_paired_with=[
                "motion_move_to_coord_001",
                "manipulation_grasp_object_001",
                "perception_get_field_center_005",
            ],
            pairing_reasons={
                "motion_move_to_coord_001": "Move robot to detected field coordinates for pick/place",
                "manipulation_grasp_object_001": "Grasp object located at detected field position",
                "perception_get_field_center_005": "Get precise center of detected field for navigation",
            },
            typical_before=[
                "motion_move_to_coord_001",
                "manipulation_grasp_object_001",
            ],
            parameter_flows=[
                ParameterFlow(
                    source_operation="detect_field",
                    source_output_key="center.x",
                    target_operation="motion_move_to_coord_001",
                    target_input_param="x",
                    description="Field center X coordinate for move_to_coordinate",
                ),
                ParameterFlow(
                    source_operation="detect_field",
                    source_output_key="center.y",
                    target_operation="motion_move_to_coord_001",
                    target_input_param="y",
                    description="Field center Y coordinate for move_to_coordinate",
                ),
                ParameterFlow(
                    source_operation="detect_field",
                    source_output_key="center.z",
                    target_operation="motion_move_to_coord_001",
                    target_input_param="z",
                    description="Field center Z coordinate for move_to_coordinate",
                ),
            ],
        ),
        implementation=detect_field,
    )


def create_detect_all_fields_operation() -> BasicOperation:
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
            "detect_all_fields('Robot1') - Find all visible fields",
            "Get field layout for planning multi-step operations",
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
        preconditions=[],
        postconditions=[],
        average_duration_ms=150,
        success_rate=0.90,
        failure_modes=["No fields visible", "Poor lighting"],
        relationships=OperationRelationship(
            operation_id="perception_detect_all_fields_006",
            required_operations=[],
            commonly_paired_with=[
                "perception_detect_field_004",
                "motion_move_to_coord_001",
            ],
            pairing_reasons={
                "perception_detect_field_004": "Use detect_field for precise single-field detection after scene overview",
                "motion_move_to_coord_001": "Move to a specific field identified in the scene overview",
            },
            typical_before=["perception_detect_field_004", "motion_move_to_coord_001"],
        ),
        implementation=detect_all_fields,
    )


DETECT_FIELD_OPERATION = create_detect_field_operation()
DETECT_ALL_FIELDS_OPERATION = create_detect_all_fields_operation()
