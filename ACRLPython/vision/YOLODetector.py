#!/usr/bin/env python3
"""
YOLODetector.py - YOLO-based object detection for robot vision

Uses YOLOv8 for robust ML-based object detection.
Designed as drop-in replacement for HSV-based CubeDetector.

Supports detection of 16+ object classes including:
- Cubes: red_cube, blue_cube
- Field markers: Field_a, Field_b, Field_c (or field_a, field_b, field_c)
- Robot parts: Robot, Plate, Base, Shoulder, Elbow, Wrist1, Wrist2
- Gripper components: Gripper_Joint, Gripper_Base, Gripper_Jaw_Left, Gripper_Jaw_Right

Note: Both lowercase_underscore and PascalCase naming conventions are supported
for compatibility with different model training configurations.

Features:
- Automatic class name extraction from model metadata
- Class filtering to detect only specific object types
- Stereo depth estimation for 3D world positions
- Debug visualization with automatic color generation
- ONNX and PyTorch model support

Usage:
    # Basic detection (IMPORTANT: specify task='detect' for ONNX models)
    detector = YOLODetector(
        model_path="models/robot_detector.onnx",
        task='detect'  # Fixes ONNX warning
    )
    result = detector.detect_objects(image, camera_id="AR4Left")

    # Filtered detection (class names depend on model training)
    result = detector.detect_objects(
        image,
        filter_classes=['red_cube', 'blue_cube']  # or ['Red_Cube', 'Blue_Cube']
    )

    # Filter robot parts
    result = detector.detect_objects(
        image,
        filter_classes=['Robot', 'Base', 'Shoulder']  # PascalCase convention
    )

    # Stereo mode with 3D positions
    result = detector.detect_objects_stereo(imgL, imgR, camera_config)
"""

import logging
import math
from typing import List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2

try:
    from ultralytics import YOLO  # type: ignore

    YOLO_AVAILABLE = True
except ImportError:
    logging.warning(
        "Ultralytics YOLO not available. Install with: pip install ultralytics"
    )
    YOLO_AVAILABLE = False
    YOLO = None  # type: ignore

# Import config
try:
    from config.Vision import (
        YOLO_CONFIDENCE_THRESHOLD,
        YOLO_IOU_THRESHOLD,
        MIN_CUBE_AREA_PX,
        MAX_CUBE_AREA_PX,
        ENABLE_DEBUG_IMAGES,
        DEBUG_IMAGES_DIR,
        ENABLE_STEREO_VALIDATION,
        STEREO_MAX_Y_DIFF,
        STEREO_MAX_SIZE_RATIO,
        STEREO_MIN_IOU,
        ENABLE_ADAPTIVE_SGBM,
        DEPTH_SAMPLING_STRATEGY,
        DEPTH_SAMPLE_INNER_PERCENT,
    )
except ImportError:
    from ..config.Vision import (
        YOLO_CONFIDENCE_THRESHOLD,
        YOLO_IOU_THRESHOLD,
        MIN_CUBE_AREA_PX,
        MAX_CUBE_AREA_PX,
        ENABLE_DEBUG_IMAGES,
        DEBUG_IMAGES_DIR,
        ENABLE_STEREO_VALIDATION,
        STEREO_MAX_Y_DIFF,
        STEREO_MAX_SIZE_RATIO,
        STEREO_MIN_IOU,
        ENABLE_ADAPTIVE_SGBM,
        DEPTH_SAMPLING_STRATEGY,
        DEPTH_SAMPLE_INNER_PERCENT,
    )

# Import detection data models from shared module
try:
    from .DetectionDataModels import DetectionObject, DetectionResult
except ImportError:
    from vision.DetectionDataModels import DetectionObject, DetectionResult

# Import stereo depth estimation if available
try:
    try:
        from .StereoConfig import CameraConfig, DEFAULT_CAMERA_CONFIG
    except ImportError:
        from vision.StereoConfig import CameraConfig, DEFAULT_CAMERA_CONFIG

    try:
        from .DepthEstimator import (
            calc_disparity,
            estimate_object_world_position_from_disparity,
            save_disparity_map_debug,
        )
    except ImportError:
        from vision.DepthEstimator import (
            calc_disparity,
            estimate_object_world_position_from_disparity,
            save_disparity_map_debug,
        )
    STEREO_AVAILABLE = True
except ImportError:
    STEREO_AVAILABLE = False
    # Type stubs for when stereo is not available
    CameraConfig = None  # type: ignore
    DEFAULT_CAMERA_CONFIG = None
    calc_disparity = None  # type: ignore
    estimate_object_world_position_from_disparity = None  # type: ignore
    save_disparity_map_debug = None  # type: ignore


class YOLODetector:
    """
    YOLO-based object detector for robot vision system

    Supports detection of 16+ object classes using YOLOv8, including cubes,
    field markers, robot parts, and gripper components.

    Compatible with CubeDetector interface for drop-in replacement.

    Key Features:
    - ONNX and PyTorch (.pt) model support
    - Automatic class name extraction from model metadata
    - Class filtering to detect only specific object types
    - Stereo depth estimation for 3D world positions
    - Debug visualization with 16+ predefined colors
    - Explicit task parameter to avoid ONNX warnings

    Example:
        # Basic usage with ONNX model
        detector = YOLODetector(
            model_path="models/robot_detector.onnx",
            task='detect'  # Avoids ONNX warning
        )

        # Detect all objects
        result = detector.detect_objects(image)

        # Filter by class (class names depend on model training)
        result = detector.detect_objects(
            image,
            filter_classes=['red_cube', 'blue_cube']  # lowercase convention
        )
        # or
        result = detector.detect_objects(
            image,
            filter_classes=['Robot', 'Base', 'Shoulder']  # PascalCase convention
        )
    """

    # Class name mapping for all detected object types.
    # Mirrors the class list embedded in field_detector.onnx (the authoritative source).
    # This fallback is only used when the model file has no embedded metadata.
    DEFAULT_CLASS_MAPPING = {
        0: "red_cube",
        1: "blue_cube",
        2: "green_cube",
        3: "yellow_cube",
        4: "purple_cube",
        5: "orange_cube",
        6: "cyan_cube",
        7: "magenta_cube",
        8: "field_a",
        9: "field_b",
        10: "field_c",
        11: "field_d",
        12: "field_e",
        13: "field_f",
        14: "field_g",
        15: "field_h",
        16: "field_i",
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        class_mapping: Optional[dict] = None,
        task: str = "detect",  # Maybe use segment for object boundaries
    ):
        """
        Initialize YOLO detector for robot vision system

        IMPORTANT: Always specify task='detect' when using ONNX models to avoid
        the "Unable to automatically guess model task" warning.

        Args:
            model_path: Path to YOLO model file (.pt or .onnx)
                       If None, uses pretrained YOLOv8n (for testing only, not for robot detection)
                       Recommended: "models/robot_detector.onnx" trained on cube_dataset.yaml
            class_mapping: Dict mapping YOLO class IDs to class names (optional)
                          If None, automatically extracts from model metadata or uses DEFAULT_CLASS_MAPPING
                          Example: {0: "red_cube", 1: "blue_cube", ...}
            task: YOLO task type ('detect', 'segment', 'classify', 'pose', 'obb')
                 Default: 'detect' - for object detection (bounding boxes)
                 MUST be specified for ONNX models to avoid warning

        Raises:
            ImportError: If ultralytics YOLO package is not installed
            Exception: If model file cannot be loaded

        Example:
            detector = YOLODetector(
                model_path="models/robot_detector.onnx",
                task='detect'  # Fixes ONNX warning
            )
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "YOLO not available. Install with: "
                "pip install ultralytics torch torchvision"
            )

        # Load YOLO model
        if model_path is None:
            # Use pretrained model for testing (will detect common objects, not robot/cubes)
            logging.warning(
                "No model_path provided. Using pretrained YOLOv8n. "
                "For robot detection, use a custom model trained on cube_dataset.yaml!"
            )
            model_path = "yolov8n.pt"

        self.model_path = Path(model_path)
        self.task = task
        logging.info(f"Loading YOLO model from: {self.model_path} (task={task})")

        try:
            if YOLO is None:
                raise ImportError("YOLO not available")

            # Load model with explicit task to avoid ONNX warning
            # This fixes: "WARNING ⚠️ Unable to automatically guess model task"
            self.model = YOLO(str(self.model_path), task=task)
            logging.info(f"YOLO model loaded successfully")

            # Try to extract class names from model metadata
            if hasattr(self.model, "names") and self.model.names:
                model_classes = self.model.names
                logging.info(f"Loaded {len(model_classes)} classes from model metadata")
            else:
                model_classes = None

        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            raise

        # Set class mapping priority:
        # 1. User-provided class_mapping (highest priority)
        # 2. Model metadata names
        # 3. DEFAULT_CLASS_MAPPING (fallback)
        if class_mapping is not None:
            self.class_mapping = class_mapping
            logging.info(
                f"Using user-provided class mapping: {list(class_mapping.values())}"
            )
        elif model_classes is not None:
            self.class_mapping = model_classes
            logging.info(
                f"Using class mapping from model: {list(model_classes.values())}"
            )
        else:
            self.class_mapping = self.DEFAULT_CLASS_MAPPING
            logging.info(
                f"Using default class mapping: {list(self.DEFAULT_CLASS_MAPPING.values())}"
            )

        # Detection thresholds
        self.conf_threshold = YOLO_CONFIDENCE_THRESHOLD
        self.iou_threshold = YOLO_IOU_THRESHOLD
        self.min_area = MIN_CUBE_AREA_PX
        self.max_area = MAX_CUBE_AREA_PX

        # Debug settings
        self.enable_debug = ENABLE_DEBUG_IMAGES
        if self.enable_debug:
            self.debug_dir = Path(DEBUG_IMAGES_DIR)
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        # Log initialization summary
        num_classes = len(self.class_mapping)
        class_names = (
            list(self.class_mapping.values())
            if isinstance(self.class_mapping, dict)
            else list(self.class_mapping)
        )
        logging.info(
            f"YOLODetector initialized: "
            f"model={self.model_path.name}, task={self.task}, "
            f"conf={self.conf_threshold}, iou={self.iou_threshold}, "
            f"classes={num_classes} ({', '.join(class_names[:5])}{'...' if num_classes > 5 else ''})"
        )

    def get_class_name(self, class_id: int) -> str:
        """
        Get class name for a given class ID

        Args:
            class_id: YOLO class ID

        Returns:
            Class name string
        """
        if isinstance(self.class_mapping, dict):
            return self.class_mapping.get(class_id, f"unknown_{class_id}")
        else:
            # Handle YOLO model.names format (can be dict-like or list-like)
            try:
                return self.class_mapping[class_id]
            except (KeyError, IndexError):
                return f"unknown_{class_id}"

    def get_all_class_names(self) -> list:
        """
        Get list of all class names

        Returns:
            List of class name strings
        """
        if isinstance(self.class_mapping, dict):
            return list(self.class_mapping.values())
        else:
            return list(self.class_mapping)

    def detect_objects(
        self,
        image: np.ndarray,
        camera_id: str = "unknown",
        filter_classes: Optional[List[str]] = None,
    ) -> DetectionResult:
        """
        Detect objects in an image using YOLO

        Detects all configured object classes (cubes, robot parts, field markers, etc.)
        or filters to specific classes if filter_classes is provided.

        Args:
            image: OpenCV image (BGR format, typically 640x480 or higher)
            camera_id: ID of the camera for metadata and logging
            filter_classes: Optional list of class names to filter detections
                          If None, returns all detected objects
                          If specified, only returns objects matching these class names
                          Example: ['red_cube', 'blue_cube'] - only detect cubes
                          Example: ['robot', 'base', 'shoulder'] - only detect robot parts

        Returns:
            DetectionResult containing all detected objects with:
            - camera_id: Camera identifier
            - image_width, image_height: Image dimensions
            - detections: List of DetectionObject instances with:
                - object_id: Unique ID within this frame
                - color: Class name (e.g., "red_cube", "robot", "base")
                - bbox: Bounding box (x, y, width, height) in pixels
                - confidence: Detection confidence (0.0-1.0)
                - center_x, center_y: Center point in pixels

        Example:
            # Detect all objects
            result = detector.detect_objects(image, camera_id="AR4Left")

            # Detect only cubes
            result = detector.detect_objects(
                image,
                camera_id="AR4Left",
                filter_classes=['red_cube', 'blue_cube']
            )

            # Detect only robot parts
            result = detector.detect_objects(
                image,
                filter_classes=['Robot', 'Base', 'Shoulder', 'Elbow']
            )
        """
        if image is None or image.size == 0:
            logging.warning("Empty image provided to YOLO detector")
            return DetectionResult(camera_id, 0, 0, [])

        height, width = image.shape[:2]

        # Run YOLO inference
        try:
            results = self.model.predict(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,  # Suppress YOLO logging
            )
        except Exception as e:
            logging.error(f"YOLO inference failed: {e}")
            return DetectionResult(camera_id, width, height, [])

        # Parse YOLO results
        detections = []
        object_id = 0

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                # Get bounding box coordinates (x1, y1, x2, y2)
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)

                # Convert to (x, y, w, h) format
                x, y = x1, y1
                w, h = x2 - x1, y2 - y1
                area = w * h

                # Filter by area
                if area < self.min_area or area > self.max_area:
                    logging.debug(
                        f"Detection {i}: Rejected by area ({area}px, "
                        f"need {self.min_area}-{self.max_area})"
                    )
                    continue

                # Get class ID and confidence
                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())

                # Map class ID to class name
                class_name = self.get_class_name(class_id)

                # Filter by class name if filter_classes is provided
                if filter_classes is not None and class_name not in filter_classes:
                    logging.debug(
                        f"Detection {i}: Rejected by filter (class={class_name}, "
                        f"filter={filter_classes})"
                    )
                    continue

                # Create detection object
                # Extract segmentation mask if available (task=segment)
                mask = None
                if results[0].masks is not None and hasattr(results[0].masks, "data"):
                    try:
                        masks_array = results[0].masks.data.cpu().numpy()  # type: ignore[union-attr]
                        if object_id < masks_array.shape[0]:
                            mask = masks_array[object_id]  # shape (H, W)
                    except Exception as mask_err:
                        logging.debug(
                            f"Could not extract mask for detection {object_id}: {mask_err}"
                        )

                det = DetectionObject(
                    object_id=object_id,
                    color=class_name,  # Use full class name (e.g., "red_cube", "robot", "base")
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    mask=mask,
                )

                detections.append(det)
                object_id += 1

                logging.debug(
                    f"Detection {i}: {class_name} at ({det.center_x}, {det.center_y}) "
                    f"- confidence: {confidence:.3f}, area: {area}px"
                )

        # Save debug image if enabled
        if self.enable_debug:
            self._save_debug_image(image, detections, camera_id)

        logging.debug(f"YOLO detected {len(detections)} objects")

        return DetectionResult(camera_id, width, height, detections)

    def detect_objects_stereo(
        self,
        imgL: np.ndarray,
        imgR: np.ndarray,
        camera_config: Optional[Any] = None,  # CameraConfig type when STEREO_AVAILABLE
        camera_id: str = "stereo",
        camera_rotation: Optional[List[float]] = None,
        camera_position: Optional[List[float]] = None,
        filter_classes: Optional[List[str]] = None,
    ) -> DetectionResult:
        """
        Detect objects in stereo images and estimate 3D world positions using YOLO

        Performs object detection on left camera image, then uses stereo disparity
        to compute 3D world coordinates for each detected object.

        Args:
            imgL: Left camera image (BGR format, must match imgR dimensions)
            imgR: Right camera image (BGR format, must match imgL dimensions)
            camera_config: Camera calibration parameters (baseline, FOV, etc.)
                          If None, uses DEFAULT_CAMERA_CONFIG
            camera_id: ID of the camera for metadata and logging
            camera_rotation: Camera rotation [pitch, yaw, roll] in degrees (optional)
                           Used to transform from camera space to world space
            camera_position: Camera position [x, y, z] in world space (optional)
                           Used to transform from camera space to world space
            filter_classes: Optional list of class names to filter detections
                          If None, returns all detected objects
                          Example: ['red_cube', 'blue_cube'] - only detect cubes
                          Example: ['robot', 'base'] - only detect robot parts

        Returns:
            DetectionResult containing detected objects with 3D world positions:
            - Each detection includes:
                - All 2D fields from detect_objects()
                - world_position: (x, y, z) in meters (world coordinates)
                - depth_m: Distance from camera in meters
                - disparity: Disparity value in pixels

        Example:
            # Detect all objects with 3D positions
            result = detector.detect_objects_stereo(imgL, imgR, camera_config)

            # Detect only cubes with 3D positions
            result = detector.detect_objects_stereo(
                imgL, imgR,
                camera_config=camera_config,
                filter_classes=['red_cube', 'blue_cube']
            )

            # Access 3D positions
            for det in result.detections:
                if det.world_position:
                    x, y, z = det.world_position
                    print(f"{det.color}: ({x:.3f}, {y:.3f}, {z:.3f})m")
        """
        if not STEREO_AVAILABLE:
            logging.error(
                "Stereo depth estimation not available - missing dependencies"
            )
            return DetectionResult(camera_id, 0, 0, [])

        if imgL is None or imgR is None:
            logging.warning("Empty stereo images provided to YOLO detector")
            return DetectionResult(camera_id, 0, 0, [])

        if imgL.shape != imgR.shape:
            logging.error(f"Stereo image size mismatch: {imgL.shape} vs {imgR.shape}")
            return DetectionResult(camera_id, 0, 0, [])

        if camera_config is None:
            camera_config = DEFAULT_CAMERA_CONFIG
            if camera_config is not None:
                logging.info(
                    f"Using default camera config: baseline={camera_config.baseline}m, "
                    f"FOV={camera_config.fov}°"
                )
            else:
                logging.error("No camera config available")
                return DetectionResult(camera_id, 0, 0, [])

        # Stereo validation: Detect in both images and match (optional)
        enable_stereo_validation = ENABLE_STEREO_VALIDATION

        if enable_stereo_validation:
            logging.debug("Stereo validation enabled - detecting in both L/R images")

            # Detect in left image
            detection_result_left = self.detect_objects(
                imgL, camera_id=camera_id + "_L", filter_classes=filter_classes
            )

            # Detect in right image
            detection_result_right = self.detect_objects(
                imgR, camera_id=camera_id + "_R", filter_classes=filter_classes
            )

            # Match detections between left and right
            max_y_diff = STEREO_MAX_Y_DIFF
            max_size_ratio = STEREO_MAX_SIZE_RATIO
            min_iou = STEREO_MIN_IOU

            matched_pairs = self._match_stereo_detections(
                detection_result_left.detections,
                detection_result_right.detections,
                max_y_diff=max_y_diff,
                max_size_ratio=max_size_ratio,
                min_iou=min_iou,
            )

            # Use only validated detections (left image detections that matched right)
            validated_detections = [pair[0] for pair in matched_pairs]
            detection_result = DetectionResult(
                camera_id,
                detection_result_left.image_width,
                detection_result_left.image_height,
                validated_detections,
            )

            logging.info(
                f"Stereo validation: {len(detection_result_left.detections)} left, "
                f"{len(detection_result_right.detections)} right → "
                f"{len(validated_detections)} validated"
            )
        else:
            # Standard mode: detect only in left image
            detection_result = self.detect_objects(
                imgL, camera_id=camera_id, filter_classes=filter_classes
            )

        # If no detections, return early
        if len(detection_result.detections) == 0:
            logging.info("No objects detected in stereo images")
            return detection_result

        # Compute disparity map once for all detections (major performance optimization)
        logging.debug(
            f"Computing disparity map for {len(detection_result.detections)} detections"
        )

        # Convert to grayscale if needed
        imgL_gray = (
            cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY) if len(imgL.shape) == 3 else imgL
        )
        imgR_gray = (
            cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY) if len(imgR.shape) == 3 else imgR
        )

        # Compute disparity map using default reconstruction config (or adaptive preset)
        from .StereoConfig import DEFAULT_RECONSTRUCTION_CONFIG

        # Runtime check for stereo functions (sanity check)
        if (
            calc_disparity is None
            or estimate_object_world_position_from_disparity is None
        ):
            logging.error(
                "Stereo functions not available despite STEREO_AVAILABLE=True"
            )
            return DetectionResult(camera_id, 0, 0, [])

        # Calculate stereo disparity map (shared across all detections for efficiency)
        # Use adaptive SGBM if enabled, otherwise use default config
        if ENABLE_ADAPTIVE_SGBM:
            # Import SGBM preset functions
            try:
                from .DepthEstimator import (
                    select_sgbm_preset,
                    calc_disparity_with_preset,
                )
            except ImportError:
                from vision.DepthEstimator import (
                    select_sgbm_preset,
                    calc_disparity_with_preset,
                )

            # Use medium preset by default (no distance estimate yet)
            preset = select_sgbm_preset(estimated_distance=None)
            disparity = calc_disparity_with_preset(imgL_gray, imgR_gray, preset)
        else:
            disparity = calc_disparity(
                imgL_gray, imgR_gray, DEFAULT_RECONSTRUCTION_CONFIG
            )

        # Save disparity map for debugging (if enabled in config)
        if save_disparity_map_debug is not None:
            save_disparity_map_debug(disparity)

        # Estimate 3D world position for each detected object
        detections_with_depth = []
        h, w = imgL.shape[:2]

        # Import bbox-guided depth function if enabled
        use_bbox_sampling = DEPTH_SAMPLING_STRATEGY is not None

        # Import functions unconditionally to avoid "possibly unbound" errors
        try:
            from .DepthEstimator import (
                estimate_depth_from_bbox,
                get_focal_length_pixels,
            )
        except ImportError:
            from vision.DepthEstimator import (
                estimate_depth_from_bbox,
                get_focal_length_pixels,
            )

        for det in detection_result.detections:
            # Bbox-guided depth sampling (NEW - more accurate than center point)
            if use_bbox_sampling:
                focal_length_px = get_focal_length_pixels(camera_config, w, h)

                bbox = (det.bbox_x, det.bbox_y, det.bbox_w, det.bbox_h)
                strategy = (
                    DEPTH_SAMPLING_STRATEGY
                    if DEPTH_SAMPLING_STRATEGY
                    else "median_inner_50pct"
                )
                inner_pct = DEPTH_SAMPLE_INNER_PERCENT

                bbox_result = estimate_depth_from_bbox(
                    disparity,
                    bbox,
                    focal_length_px,
                    camera_config.baseline,
                    strategy=strategy,
                    min_disparity_threshold=1.0,
                    max_depth_threshold=10.0,
                    inner_percent=inner_pct,
                )

                if bbox_result is not None:
                    depth_m, disp_value, num_valid_pixels = bbox_result
                    logging.debug(
                        f"Bbox-guided depth: {depth_m:.3f}m from {num_valid_pixels} pixels "
                        f"(strategy: {strategy})"
                    )
                else:
                    depth_m = None
                    disp_value = None
            else:
                # Legacy center-point sampling (for backward compatibility)
                disp_value = None
                if (
                    0 <= det.center_y < disparity.shape[0]
                    and 0 <= det.center_x < disparity.shape[1]
                ):
                    disp_value = float(disparity[det.center_y, det.center_x])

                # Calculate depth from disparity using camera parameters
                # Depth = (baseline * focal_length) / disparity
                depth_m = None
                if (
                    disp_value is not None and disp_value > 1.0
                ):  # Valid disparity threshold
                    # focal_length = (image_width / 2) / tan(FOV / 2)
                    focal_length = (w / 2.0) / math.tan(
                        math.radians(camera_config.fov / 2.0)
                    )
                    depth_m = (camera_config.baseline * focal_length) / disp_value

            # Estimate 3D world position using pre-computed disparity map (optimized)
            world_pos = estimate_object_world_position_from_disparity(
                disparity,
                det.center_x,
                det.center_y,
                camera_config,
                w,
                h,
                min_disparity=1.0,
                max_depth=10.0,
                camera_rotation=camera_rotation,
                camera_position=camera_position,
            )

            # Create new detection object with 3D world position, depth, and disparity
            det_with_depth = DetectionObject(
                object_id=det.object_id,
                color=det.color,  # Class name (e.g., "red_cube", "robot", "base")
                bbox=(det.bbox_x, det.bbox_y, det.bbox_w, det.bbox_h),
                confidence=det.confidence,
                world_position=world_pos,  # (x, y, z) in meters
                depth_m=depth_m,  # Z distance from camera in meters
                disparity=disp_value,  # Disparity in pixels
            )

            detections_with_depth.append(det_with_depth)

            # Log detection with 3D coordinates
            if world_pos:
                depth_str = f"{depth_m:.3f}m" if depth_m is not None else "N/A"
                disp_str = f"{disp_value:.1f}px" if disp_value is not None else "N/A"
                logging.debug(
                    f"{det.color.upper()}: pixel ({det.center_x}, {det.center_y}) "
                    f"→ world ({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f})m, "
                    f"depth={depth_str}, disp={disp_str}"
                )
            else:
                logging.debug(
                    f"{det.color.upper()} at pixel ({det.center_x}, {det.center_y}) "
                    f"- failed to estimate 3D position"
                )

        return DetectionResult(
            camera_id,
            detection_result.image_width,
            detection_result.image_height,
            detections_with_depth,
        )

    def _get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """
        Get visualization color for a class name (for debug image annotations)

        Returns predefined BGR colors for known classes, or generates a
        deterministic color from the class name hash for unknown classes.

        Args:
            class_name: Name of the detected class (e.g., "red_cube", "robot", "base")

        Returns:
            BGR color tuple (B, G, R) for OpenCV visualization
            Values range from 0-255 for each channel

        Example:
            color = detector._get_class_color("red_cube")  # Returns (0, 0, 255) - Red
            color = detector._get_class_color("robot")     # Returns (255, 165, 0) - Orange
        """
        # Predefined BGR colors for all 16 dataset classes + extras
        # Format: class_name: (B, G, R) tuple
        # Supports both naming conventions (lowercase with underscores and PascalCase)
        color_map = {
            # Cubes
            "red_cube": (0, 0, 255),  # Red
            "blue_cube": (255, 0, 0),  # Blue
            "green_cube": (0, 255, 0),  # Green
            "yellow_cube": (0, 255, 255),  # Yellow
            # Field markers (support both conventions)
            "field_a": (128, 128, 128),  # Gray
            "Field_a": (128, 128, 128),  # Gray (PascalCase)
            "field_b": (150, 150, 150),  # Light gray
            "Field_b": (150, 150, 150),  # Light gray (PascalCase)
            "field_c": (100, 100, 100),  # Dark gray
            "Field_c": (100, 100, 100),  # Dark gray (PascalCase)
            # Robot parts (support both conventions)
            "robot": (255, 165, 0),  # Orange
            "Robot": (255, 165, 0),  # Orange (PascalCase)
            "plate": (203, 192, 255),  # Pink
            "Plate": (203, 192, 255),  # Pink (PascalCase)
            "base": (42, 42, 165),  # Brown
            "Base": (42, 42, 165),  # Brown (PascalCase)
            "shoulder": (147, 20, 255),  # Deep pink
            "Shoulder": (147, 20, 255),  # Deep pink (PascalCase)
            "elbow": (0, 165, 255),  # Orange-red
            "Elbow": (0, 165, 255),  # Orange-red (PascalCase)
            "wrist1": (255, 255, 0),  # Cyan
            "Wrist1": (255, 255, 0),  # Cyan (PascalCase)
            "wrist2": (180, 105, 255),  # Hot pink
            "Wrist2": (180, 105, 255),  # Hot pink (PascalCase)
            # Gripper components (support both conventions)
            "gripperjoint": (76, 153, 0),  # Dark green
            "Gripper_Joint": (76, 153, 0),  # Dark green (PascalCase)
            "gripperbase": (255, 144, 30),  # Dodger blue
            "Gripper_Base": (255, 144, 30),  # Dodger blue (PascalCase)
            "gripperjawleft": (238, 130, 238),  # Violet
            "Gripper_Jaw_Left": (238, 130, 238),  # Violet (PascalCase)
            "gripperjawright": (221, 160, 221),  # Plum
            "Gripper_Jaw_Right": (221, 160, 221),  # Plum (PascalCase)
        }

        # Return predefined color if available
        if class_name in color_map:
            return color_map[class_name]

        # For unknown classes, generate deterministic color from class name hash
        # This ensures same class always gets same color across runs
        hash_val = hash(class_name) % 360  # Map to hue angle (0-360°)

        # Use HSV color space for better color distribution
        # H=hash_val, S=0.8 (high saturation), V=0.9 (high brightness)
        import colorsys

        rgb = colorsys.hsv_to_rgb(hash_val / 360.0, 0.8, 0.9)

        # Convert RGB to BGR for OpenCV
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        return bgr

    def _calculate_iou(
        self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate Intersection over Union (IOU) of two bounding boxes.

        Args:
            bbox1: First bbox as (x, y, width, height)
            bbox2: Second bbox as (x, y, width, height)

        Returns:
            IOU value between 0.0 and 1.0
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection rectangle
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        # Calculate intersection area
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # Calculate union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _match_stereo_detections(
        self,
        detections_left: List[DetectionObject],
        detections_right: List[DetectionObject],
        max_y_diff: int = 10,
        max_size_ratio: float = 0.3,
        min_iou: float = 0.0,
    ) -> List[Tuple[DetectionObject, DetectionObject]]:
        """
        Match detections between left and right stereo images.

        Matching criteria (all must be satisfied):
        - Same class (color/object type)
        - Similar Y coordinate (±max_y_diff pixels, assuming rectified cameras)
        - Similar bbox size (within max_size_ratio fraction)
        - Right detection is LEFT of left detection (positive disparity)
        - Optional: Minimum IOU threshold

        Args:
            detections_left: Detections from left image
            detections_right: Detections from right image
            max_y_diff: Maximum Y coordinate difference (pixels)
            max_size_ratio: Maximum bbox size difference (fraction, e.g., 0.3 = 30%)
            min_iou: Minimum IOU for match (0.0 = disabled)

        Returns:
            List of matched pairs: [(left_det, right_det), ...]
        """
        matched_pairs = []
        used_right = set()

        for det_l in detections_left:
            best_match = None
            best_score = 0

            for idx, det_r in enumerate(detections_right):
                if idx in used_right:
                    continue

                # 1. Same class?
                if det_l.color != det_r.color:
                    continue

                # 2. Similar Y coordinate? (rectified cameras should have same Y)
                y_diff = abs(det_l.center_y - det_r.center_y)
                if y_diff > max_y_diff:
                    continue

                # 3. Similar size?
                l_area = det_l.bbox_w * det_l.bbox_h
                r_area = det_r.bbox_w * det_r.bbox_h
                size_ratio = abs(l_area - r_area) / max(l_area, r_area)
                if size_ratio > max_size_ratio:
                    continue

                # 4. Positive disparity? (right image LEFT of left image)
                if det_r.center_x >= det_l.center_x:
                    continue

                # 5. Optional IOU check
                if min_iou > 0:
                    bbox_l = (det_l.bbox_x, det_l.bbox_y, det_l.bbox_w, det_l.bbox_h)
                    bbox_r = (det_r.bbox_x, det_r.bbox_y, det_r.bbox_w, det_r.bbox_h)
                    iou = self._calculate_iou(bbox_l, bbox_r)
                    if iou < min_iou:
                        continue

                # Score based on Y difference (lower is better)
                score = 1.0 / (1.0 + y_diff)

                if score > best_score:
                    best_score = score
                    best_match = (det_r, idx)

            if best_match:
                matched_pairs.append((det_l, best_match[0]))
                used_right.add(best_match[1])

        return matched_pairs

    def _save_debug_image(
        self, image: np.ndarray, detections: List[DetectionObject], camera_id: str
    ):
        """
        Save annotated image with bounding boxes for debugging and visualization

        Creates a copy of the input image and draws:
        - Colored bounding boxes (one color per class)
        - Center point markers
        - Labels with class name and confidence score

        Saved to debug_dir with timestamp in filename.

        Args:
            image: Original image (BGR format)
            detections: List of detected objects to visualize
            camera_id: Camera ID for filename (e.g., "AR4Left", "stereo")

        Output:
            Saves image to: {debug_dir}/yolo_{camera_id}_{timestamp}.jpg
        """
        debug_image = image.copy()

        # Draw bounding boxes and labels for each detection
        for det in detections:
            # Get class-specific color
            color = self._get_class_color(det.color)

            # Draw bounding box rectangle
            cv2.rectangle(
                debug_image,
                (det.bbox_x, det.bbox_y),
                (det.bbox_x + det.bbox_w, det.bbox_y + det.bbox_h),
                color,
                thickness=2,
            )

            # Draw center point marker
            cv2.circle(
                debug_image,
                (det.center_x, det.center_y),
                radius=5,
                color=color,
                thickness=-1,  # Filled circle
            )

            # Draw label with class name and confidence
            label = f"{det.color} {det.confidence:.2f}"
            cv2.putText(
                debug_image,
                label,
                (det.bbox_x, det.bbox_y - 10),  # Position above bounding box
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=color,
                thickness=2,
            )

        # Save annotated image to debug directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.debug_dir / f"yolo_{camera_id}_{timestamp}.jpg"
        cv2.imwrite(str(filename), debug_image)
        logging.debug(f"Saved YOLO debug image: {filename}")


def main():
    """
    Command-line interface for YOLO object detection

    Supports:
    - Detection on single images
    - Class filtering (e.g., only detect cubes or robot parts)
    - Adjustable confidence threshold
    - Debug mode with annotated output images
    - Verbose logging for debugging

    Usage:
        python -m vision.YOLODetector <image_path> [options]

    Examples:
        # Basic detection
        python -m vision.YOLODetector test.jpg --model models/robot_detector.onnx --task detect

        # Detect only cubes (class names depend on model training)
        python -m vision.YOLODetector test.jpg --filter red_cube blue_cube

        # Detect robot parts (PascalCase convention)
        python -m vision.YOLODetector test.jpg --filter Robot Base Shoulder

        # High confidence with debug output
        python -m vision.YOLODetector test.jpg --conf 0.7 --debug --verbose
    """
    import sys
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="YOLO-based object detector for robot vision system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic detection with ONNX model (IMPORTANT: specify --task detect)
  python -m vision.YOLODetector test.jpg --model models/robot_detector.onnx --task detect

  # Detect only cubes (class names depend on model training)
  python -m vision.YOLODetector test.jpg --filter red_cube blue_cube

  # Detect robot parts (PascalCase convention)
  python -m vision.YOLODetector test.jpg --filter Robot Base Shoulder Elbow Wrist1 Wrist2

  # High confidence detection with debug visualization
  python -m vision.YOLODetector test.jpg --conf 0.7 --debug --verbose
        """,
    )
    parser.add_argument("image_path", help="Path to input image (JPEG, PNG, etc.)")
    parser.add_argument(
        "--model",
        default="models/robot_detector.onnx",
        help="Path to YOLO model (.pt or .onnx). Default: models/robot_detector.onnx",
    )
    parser.add_argument(
        "--task",
        default="detect",
        choices=["detect", "segment", "classify", "pose", "obb"],
        help="YOLO task type - REQUIRED for ONNX models to avoid warning. Default: detect",
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        metavar="CLASS",
        help="Filter detections by class names. Example: --filter red_cube blue_cube",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        metavar="THRESHOLD",
        help="Minimum confidence threshold (0.0-1.0). Lower = more detections. Default: 0.5",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (saves annotated images to ./debug/)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (shows detection details)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # Load image
    image = cv2.imread(args.image_path)
    if image is None:
        logging.error(f"Could not read image from {args.image_path}")
        sys.exit(1)

    logging.info(f"Loaded image: {args.image_path} ({image.shape[1]}x{image.shape[0]})")

    # Initialize YOLO detector
    try:
        logging.info("Initializing YOLO detector...")
        detector = YOLODetector(model_path=args.model, task=args.task)

        # Override config settings from command-line arguments
        if args.conf != 0.5:
            detector.conf_threshold = args.conf
            logging.info(f"Confidence threshold: {args.conf}")

        if args.debug:
            detector.enable_debug = True
            detector.debug_dir = Path("./debug")
            detector.debug_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Debug mode enabled (output: {detector.debug_dir})")

    except Exception as e:
        logging.error(f"Failed to initialize YOLO detector: {e}")
        logging.error("Make sure the model file exists and YOLO is installed:")
        logging.error("  pip install ultralytics torch torchvision")
        sys.exit(1)

    # Run object detection
    logging.info("Running YOLO detection...")
    result = detector.detect_objects(
        image, camera_id="test", filter_classes=args.filter
    )

    # Print formatted results
    print(f"\n{'='*60}")
    print(f"YOLO Detection Results")
    print(f"{'='*60}")
    print(f"Detected {len(result.detections)} objects")

    if args.filter:
        print(f"Filter: {', '.join(args.filter)}")

    print(f"\nDetections:")
    for i, det in enumerate(result.detections, 1):
        print(
            f"  {i}. {det.color:20s} | "
            f"Position: ({det.center_x:4d}, {det.center_y:4d}) | "
            f"BBox: {det.bbox_w}x{det.bbox_h} px | "
            f"Conf: {det.confidence:.3f}"
        )

    if len(result.detections) == 0:
        print("  (no detections)")
        print(f"\nTip: Try lowering confidence with --conf 0.3")

    print(f"{'='*60}")

    # Print all available classes from model
    print(
        f"\nAvailable classes in model ({len(detector.get_all_class_names())} total):"
    )
    all_classes = detector.get_all_class_names()
    for i in range(0, len(all_classes), 4):
        classes_row = all_classes[i : i + 4]
        print(f"  {', '.join(f'{c:20s}' for c in classes_row)}")

    if args.debug:
        print(f"\n✓ Debug images saved to: {detector.debug_dir}")
        print(f"  Files: yolo_test_*.jpg")


if __name__ == "__main__":
    main()
