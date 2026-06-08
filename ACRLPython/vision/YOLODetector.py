#!/usr/bin/env python3
"""YOLOv8-based object detector. Drop-in replacement for HSV-based CubeDetector. Supports ONNX/PyTorch, stereo depth, class filtering."""

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

try:
    from .DetectionDataModels import DetectionObject, DetectionResult
except ImportError:
    from vision.DetectionDataModels import DetectionObject, DetectionResult

try:
    try:
        from .StereoConfig import DEFAULT_CAMERA_CONFIG
    except ImportError:
        from vision.StereoConfig import DEFAULT_CAMERA_CONFIG

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
    """YOLO-based object detector. Supports ONNX/PyTorch, class filtering, stereo depth."""

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
        task: str = "detect",  # Specify explicitly for ONNX models to avoid "Unable to guess task" warning
    ):
        if not YOLO_AVAILABLE:
            raise ImportError(
                "YOLO not available. Install with: "
                "pip install ultralytics torch torchvision"
            )

        if model_path is None:
            # Use pretrained model for testing (will detect common objects, not robot/cubes)
            logging.warning(
                "No model_path provided. Using pretrained YOLOv8n. "
                "For robot detection, use a custom model trained on cube_dataset.yaml!"
            )
            model_path = "yolov8n.pt"

        self.model_path = Path(model_path)
        self.task = task
        logging.debug(f"Loading YOLO model from: {self.model_path} (task={task})")

        try:
            if YOLO is None:
                raise ImportError("YOLO not available")

            self.model = YOLO(str(self.model_path), task=task)
            logging.debug(f"YOLO model loaded successfully")

            if hasattr(self.model, "names") and self.model.names:
                model_classes = self.model.names
                logging.debug(
                    f"Loaded {len(model_classes)} classes from model metadata"
                )
            else:
                model_classes = None

        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            raise

        if class_mapping is not None:
            self.class_mapping = class_mapping
            logging.debug(
                f"Using user-provided class mapping: {list(class_mapping.values())}"
            )
        elif model_classes is not None:
            self.class_mapping = model_classes
            logging.debug(
                f"Using class mapping from model: {list(model_classes.values())}"
            )
        else:
            self.class_mapping = self.DEFAULT_CLASS_MAPPING
            logging.debug(
                f"Using default class mapping: {list(self.DEFAULT_CLASS_MAPPING.values())}"
            )

        self.conf_threshold = YOLO_CONFIDENCE_THRESHOLD
        self.iou_threshold = YOLO_IOU_THRESHOLD
        self.min_area = MIN_CUBE_AREA_PX
        self.max_area = MAX_CUBE_AREA_PX

        self.enable_debug = ENABLE_DEBUG_IMAGES
        if self.enable_debug:
            self.debug_dir = Path(DEBUG_IMAGES_DIR)
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        num_classes = len(self.class_mapping)
        class_names = (
            list(self.class_mapping.values())
            if isinstance(self.class_mapping, dict)
            else list(self.class_mapping)
        )
        logging.debug(
            f"YOLODetector initialized: "
            f"model={self.model_path.name}, task={self.task}, "
            f"conf={self.conf_threshold}, iou={self.iou_threshold}, "
            f"classes={num_classes} ({', '.join(class_names[:5])}{'...' if num_classes > 5 else ''})"
        )

    def get_class_name(self, class_id: int) -> str:
        if isinstance(self.class_mapping, dict):
            return self.class_mapping.get(class_id, f"unknown_{class_id}")
        else:
            # Handle YOLO model.names format (can be dict-like or list-like)
            try:
                return self.class_mapping[class_id]
            except (KeyError, IndexError):
                return f"unknown_{class_id}"

    def get_all_class_names(self) -> list:
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
        if image is None or image.size == 0:
            logging.warning("Empty image provided to YOLO detector")
            return DetectionResult(camera_id, 0, 0, [])

        height, width = image.shape[:2]

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

        detections = []
        object_id = 0

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)

                x, y = x1, y1
                w, h = x2 - x1, y2 - y1
                area = w * h

                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())

                if area < self.min_area or area > self.max_area:
                    logging.debug(
                        f"Detection {i}: Rejected by area ({area}px, "
                        f"need {self.min_area}-{self.max_area})"
                    )
                    continue

                class_name = self.get_class_name(class_id)

                if filter_classes is not None and class_name not in filter_classes:
                    logging.debug(
                        f"Detection {i}: Rejected by filter (class={class_name}, "
                        f"filter={filter_classes})"
                    )
                    continue

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

        enable_stereo_validation = ENABLE_STEREO_VALIDATION

        if enable_stereo_validation:
            logging.debug("Stereo validation enabled - detecting in both L/R images")

            detection_result_left = self.detect_objects(
                imgL, camera_id=camera_id + "_L", filter_classes=filter_classes
            )

            detection_result_right = self.detect_objects(
                imgR, camera_id=camera_id + "_R", filter_classes=filter_classes
            )

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
            detection_result = self.detect_objects(
                imgL, camera_id=camera_id, filter_classes=filter_classes
            )

        if len(detection_result.detections) == 0:
            logging.info("No objects detected in stereo images")
            return detection_result

        logging.debug(
            f"Computing disparity map for {len(detection_result.detections)} detections"
        )

        imgL_gray = (
            cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY) if len(imgL.shape) == 3 else imgL
        )
        imgR_gray = (
            cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY) if len(imgR.shape) == 3 else imgR
        )

        from .StereoConfig import DEFAULT_RECONSTRUCTION_CONFIG

        if (
            calc_disparity is None
            or estimate_object_world_position_from_disparity is None
        ):
            logging.error(
                "Stereo functions not available despite STEREO_AVAILABLE=True"
            )
            return DetectionResult(camera_id, 0, 0, [])

        if ENABLE_ADAPTIVE_SGBM:
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

            estimated_distance = None
            if camera_position is not None:
                try:
                    from core.Imports import get_world_state

                    ws = get_world_state()
                    known_objects = ws.get_all_objects()
                    if known_objects:
                        cx, cy, cz = (
                            camera_position[0],
                            camera_position[1],
                            camera_position[2],
                        )
                        distances = [
                            math.sqrt(
                                (obj.position[0] - cx) ** 2
                                + (obj.position[1] - cy) ** 2
                                + (obj.position[2] - cz) ** 2
                            )
                            for obj in known_objects
                            if obj.position is not None
                        ]
                        if distances:
                            distances.sort()
                            estimated_distance = distances[
                                len(distances) // 2
                            ]  # median
                            logging.debug(
                                f"Adaptive SGBM: prior distance={estimated_distance:.2f}m "
                                f"(from {len(distances)} WorldState objects)"
                            )
                except Exception as e:
                    logging.debug(f"Could not query WorldState for SGBM prior: {e}")

            preset = select_sgbm_preset(estimated_distance=estimated_distance)
            disparity = calc_disparity_with_preset(imgL_gray, imgR_gray, preset)
        else:
            disparity = calc_disparity(
                imgL_gray, imgR_gray, DEFAULT_RECONSTRUCTION_CONFIG
            )

        if save_disparity_map_debug is not None:
            save_disparity_map_debug(disparity)

        detections_with_depth = []
        h, w = imgL.shape[:2]

        use_bbox_sampling = DEPTH_SAMPLING_STRATEGY is not None

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

                # depth = (baseline * focal_length) / disparity
                depth_m = None
                if (
                    disp_value is not None and disp_value > 1.0
                ):  # Valid disparity threshold
                    # focal_length = (image_width / 2) / tan(FOV / 2)
                    focal_length = (w / 2.0) / math.tan(
                        math.radians(camera_config.fov / 2.0)
                    )
                    depth_m = (camera_config.baseline * focal_length) / disp_value

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

            det_with_depth = DetectionObject(
                object_id=det.object_id,
                color=det.color,
                bbox=(det.bbox_x, det.bbox_y, det.bbox_w, det.bbox_h),
                confidence=det.confidence,
                world_position=world_pos,
                depth_m=depth_m,
                disparity=disp_value,
            )

            detections_with_depth.append(det_with_depth)

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
        """Return BGR debug color for class. Predefined for known classes, hash-derived for unknown."""
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

        if class_name in color_map:
            return color_map[class_name]

        hash_val = hash(class_name) % 360
        import colorsys

        rgb = colorsys.hsv_to_rgb(hash_val / 360.0, 0.8, 0.9)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        return bgr

    def _calculate_iou(
        self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
    ) -> float:
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

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
        matched_pairs = []
        used_right = set()

        for det_l in detections_left:
            best_match = None
            best_score = 0

            for idx, det_r in enumerate(detections_right):
                if idx in used_right:
                    continue

                if det_l.color != det_r.color:
                    continue

                # Rectified cameras: same Y within tolerance
                y_diff = abs(det_l.center_y - det_r.center_y)
                if y_diff > max_y_diff:
                    continue

                l_area = det_l.bbox_w * det_l.bbox_h
                r_area = det_r.bbox_w * det_r.bbox_h
                size_ratio = abs(l_area - r_area) / max(l_area, r_area)
                if size_ratio > max_size_ratio:
                    continue

                # Positive disparity: right detection must be left of left detection
                if det_r.center_x >= det_l.center_x:
                    continue

                if min_iou > 0:
                    bbox_l = (det_l.bbox_x, det_l.bbox_y, det_l.bbox_w, det_l.bbox_h)
                    bbox_r = (det_r.bbox_x, det_r.bbox_y, det_r.bbox_w, det_r.bbox_h)
                    iou = self._calculate_iou(bbox_l, bbox_r)
                    if iou < min_iou:
                        continue

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
        debug_image = image.copy()

        for det in detections:
            color = self._get_class_color(det.color)

            cv2.rectangle(
                debug_image,
                (det.bbox_x, det.bbox_y),
                (det.bbox_x + det.bbox_w, det.bbox_y + det.bbox_h),
                color,
                thickness=2,
            )

            cv2.circle(
                debug_image,
                (det.center_x, det.center_y),
                radius=5,
                color=color,
                thickness=-1,  # Filled circle
            )

            label = f"{det.color} {det.confidence:.2f}"
            cv2.putText(
                debug_image,
                label,
                (det.bbox_x, det.bbox_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=color,
                thickness=2,
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.debug_dir / f"yolo_{camera_id}_{timestamp}.jpg"
        cv2.imwrite(str(filename), debug_image)
        logging.debug(f"Saved YOLO debug image: {filename}")


def main():
    """CLI for YOLO detection on a single image."""
    import sys
    import argparse

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

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    image = cv2.imread(args.image_path)
    if image is None:
        logging.error(f"Could not read image from {args.image_path}")
        sys.exit(1)

    logging.info(f"Loaded image: {args.image_path} ({image.shape[1]}x{image.shape[0]})")

    try:
        logging.info("Initializing YOLO detector...")
        detector = YOLODetector(model_path=args.model, task=args.task)

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

    logging.info("Running YOLO detection...")
    result = detector.detect_objects(
        image, camera_id="test", filter_classes=args.filter
    )

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

    print(
        f"\nAvailable classes in model ({len(detector.get_all_class_names())} total):"
    )
    all_classes = detector.get_all_class_names()
    for i in range(0, len(all_classes), 4):
        classes_row = all_classes[i : i + 4]
        print(f"  {', '.join(f'{c:20s}' for c in classes_row)}")

    if args.debug:
        print(f"\nDebug images saved to: {detector.debug_dir}")
        print(f"  Files: yolo_test_*.jpg")


if __name__ == "__main__":
    main()
