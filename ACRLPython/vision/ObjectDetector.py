#!/usr/bin/env python3
"""HSV-based color object detector with YOLO fallback and stereo depth support."""

import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2

try:
    from config.Vision import (
        USE_YOLO,
        YOLO_MODEL_PATH,
        YOLO_TASK,
        YOLO_SEGMENTATION_MODEL,
        RED_HSV_LOWER_1,
        RED_HSV_UPPER_1,
        RED_HSV_LOWER_2,
        RED_HSV_UPPER_2,
        BLUE_HSV_LOWER,
        BLUE_HSV_UPPER,
        GREEN_HSV_LOWER,
        GREEN_HSV_UPPER,
        YELLOW_HSV_LOWER,
        YELLOW_HSV_UPPER,
        ORANGE_HSV_LOWER,
        ORANGE_HSV_UPPER,
        PURPLE_HSV_LOWER,
        PURPLE_HSV_UPPER,
        CYAN_HSV_LOWER,
        CYAN_HSV_UPPER,
        MAGENTA_HSV_LOWER,
        MAGENTA_HSV_UPPER,
        MIN_CUBE_AREA_PX,
        MAX_CUBE_AREA_PX,
        MIN_ASPECT_RATIO,
        MAX_ASPECT_RATIO,
        MIN_CONFIDENCE,
        ENABLE_DEBUG_IMAGES,
        DEBUG_IMAGES_DIR,
    )
    from config.Servers import LOG_FORMAT
except ImportError:
    from ..config.Vision import (
        USE_YOLO,
        YOLO_MODEL_PATH,
        YOLO_TASK,
        YOLO_SEGMENTATION_MODEL,
        RED_HSV_LOWER_1,
        RED_HSV_UPPER_1,
        RED_HSV_LOWER_2,
        RED_HSV_UPPER_2,
        BLUE_HSV_LOWER,
        BLUE_HSV_UPPER,
        GREEN_HSV_LOWER,
        GREEN_HSV_UPPER,
        YELLOW_HSV_LOWER,
        YELLOW_HSV_UPPER,
        ORANGE_HSV_LOWER,
        ORANGE_HSV_UPPER,
        PURPLE_HSV_LOWER,
        PURPLE_HSV_UPPER,
        CYAN_HSV_LOWER,
        CYAN_HSV_UPPER,
        MAGENTA_HSV_LOWER,
        MAGENTA_HSV_UPPER,
        MIN_CUBE_AREA_PX,
        MAX_CUBE_AREA_PX,
        MIN_ASPECT_RATIO,
        MAX_ASPECT_RATIO,
        MIN_CONFIDENCE,
        ENABLE_DEBUG_IMAGES,
        DEBUG_IMAGES_DIR,
    )
    from ..config.Servers import LOG_FORMAT

try:
    from .DetectionDataModels import DetectionObject, DetectionResult
except ImportError:
    from vision.DetectionDataModels import DetectionObject, DetectionResult

YOLO_AVAILABLE = False
if USE_YOLO:
    try:
        from .YOLODetector import YOLODetector

        YOLO_AVAILABLE = True
        logging.info("YOLO detection enabled")
    except ImportError as e:
        logging.error(f"YOLO enabled in config but import failed: {e}")
        logging.error(
            "Falling back to HSV color detection — install ultralytics to enable YOLO"
        )

try:
    try:
        from .StereoConfig import (
            CameraConfig,
            ReconstructionConfig,
            DEFAULT_CAMERA_CONFIG,
            DEFAULT_RECONSTRUCTION_CONFIG,
        )
    except ImportError:
        from vision.StereoConfig import (
            CameraConfig,
            ReconstructionConfig,
            DEFAULT_CAMERA_CONFIG,
            DEFAULT_RECONSTRUCTION_CONFIG,
        )

    try:
        from .DepthEstimator import (
            calc_disparity,
            estimate_depth_from_bbox,
            estimate_object_world_position_from_disparity,
            save_disparity_map_debug,
        )
    except ImportError:
        from vision.DepthEstimator import (
            calc_disparity,
            estimate_depth_from_bbox,
            estimate_object_world_position_from_disparity,
            save_disparity_map_debug,
        )

    STEREO_AVAILABLE = True
    logging.debug("Stereo depth estimation available")
except Exception as e:
    logging.warning(f"Stereo depth estimation not available: {e}")
    STEREO_AVAILABLE = False
    CameraConfig = type("CameraConfig", (), {})
    DEFAULT_CAMERA_CONFIG = None
    DEFAULT_RECONSTRUCTION_CONFIG = None
    ReconstructionConfig = type("ReconstructionConfig", (), {})

    def estimate_depth_from_bbox(*args, **kwargs) -> Optional[Tuple[float, float, int]]:
        return None

    def estimate_object_world_position_from_disparity(
        *args, **kwargs
    ) -> Optional[Tuple[float, float, float]]:
        return None

    def save_disparity_map_debug(*args, **kwargs) -> None:
        pass

    def calc_disparity(*args, **kwargs) -> np.ndarray:
        return np.zeros((0, 0), dtype=np.float32)


def estimate_object_dimensions_from_bbox(
    bbox: Tuple[int, int, int, int],
    depth_m: float,
    focal_length_px: float,
    camera_config: Optional["CameraConfig"] = None,  # type: ignore
) -> Tuple[float, float, float]:
    """Pinhole model: pixel bbox + depth → (width_m, height_m, depth_m). Depth dim heuristic: min(w,h)*0.8."""
    x, y, w_px, h_px = bbox

    width_m = (w_px * depth_m) / focal_length_px
    height_m = (h_px * depth_m) / focal_length_px

    depth_m_est = min(width_m, height_m) * 0.8

    return (width_m, height_m, depth_m_est)


class CubeDetector:
    """
    YOLO-based object detector with HSV fallback.

    Uses YOLODetector when USE_YOLO=true (default) and ultralytics is installed.
    Falls back to HSV color segmentation only if YOLO is unavailable.
    """

    def __init__(self):
        # Read USE_YOLO dynamically for testability
        try:
            import config.Vision as vision_cfg

            use_yolo_config = vision_cfg.USE_YOLO
        except (ImportError, AttributeError):
            use_yolo_config = USE_YOLO
        self.use_yolo = YOLO_AVAILABLE and use_yolo_config

        self._segmentation_model = None

        if self.use_yolo and YOLO_AVAILABLE:
            try:
                model_path = (
                    YOLO_SEGMENTATION_MODEL
                    if YOLO_TASK == "segment"
                    else YOLO_MODEL_PATH
                )
                self.yolo_detector = YOLODetector(model_path=model_path)  # type: ignore[name-defined]
            except Exception as e:
                logging.error(f"Failed to initialize YOLO detector: {e}")
                logging.error(
                    "Falling back to HSV color detection — YOLO is unavailable"
                )
                self.use_yolo = False

        # Red needs two ranges because hue wraps around in HSV
        self.red_lower_1 = np.array(RED_HSV_LOWER_1, dtype=np.uint8)
        self.red_upper_1 = np.array(RED_HSV_UPPER_1, dtype=np.uint8)
        self.red_lower_2 = np.array(RED_HSV_LOWER_2, dtype=np.uint8)
        self.red_upper_2 = np.array(RED_HSV_UPPER_2, dtype=np.uint8)

        self.blue_lower = np.array(BLUE_HSV_LOWER, dtype=np.uint8)
        self.blue_upper = np.array(BLUE_HSV_UPPER, dtype=np.uint8)
        self.green_lower = np.array(GREEN_HSV_LOWER, dtype=np.uint8)
        self.green_upper = np.array(GREEN_HSV_UPPER, dtype=np.uint8)
        self.yellow_lower = np.array(YELLOW_HSV_LOWER, dtype=np.uint8)
        self.yellow_upper = np.array(YELLOW_HSV_UPPER, dtype=np.uint8)
        self.orange_lower = np.array(ORANGE_HSV_LOWER, dtype=np.uint8)
        self.orange_upper = np.array(ORANGE_HSV_UPPER, dtype=np.uint8)
        self.purple_lower = np.array(PURPLE_HSV_LOWER, dtype=np.uint8)
        self.purple_upper = np.array(PURPLE_HSV_UPPER, dtype=np.uint8)
        self.cyan_lower = np.array(CYAN_HSV_LOWER, dtype=np.uint8)
        self.cyan_upper = np.array(CYAN_HSV_UPPER, dtype=np.uint8)
        self.magenta_lower = np.array(MAGENTA_HSV_LOWER, dtype=np.uint8)
        self.magenta_upper = np.array(MAGENTA_HSV_UPPER, dtype=np.uint8)

        self.min_area = MIN_CUBE_AREA_PX
        self.max_area = MAX_CUBE_AREA_PX
        self.min_aspect = MIN_ASPECT_RATIO
        self.max_aspect = MAX_ASPECT_RATIO
        self.min_confidence = MIN_CONFIDENCE

        self.enable_debug = ENABLE_DEBUG_IMAGES
        if self.enable_debug:
            self.debug_dir = Path(DEBUG_IMAGES_DIR)
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        if not self.use_yolo:
            logging.debug("CubeDetector initialized with HSV color detection")

    def detect_objects(
        self, image: np.ndarray, camera_id: str = "unknown"
    ) -> DetectionResult:
        if self.use_yolo:
            return self.yolo_detector.detect_objects(image, camera_id)

        if image is None or image.size == 0:
            logging.warning("Empty image provided to detector")
            return DetectionResult(camera_id, 0, 0, [])

        height, width = image.shape[:2]

        detections = self._detect_all_objects(image)

        all_detections = []
        object_id = 0

        for det in detections:
            all_detections.append(
                DetectionObject(
                    object_id=object_id,
                    color=det["color"],
                    bbox=det["bbox"],
                    confidence=det["confidence"],
                )
            )
            object_id += 1

        if self.enable_debug:
            self._save_debug_image(image, all_detections, camera_id)

        logging.debug(f"Detected {len(all_detections)} objects")

        return DetectionResult(camera_id, width, height, all_detections)

    def detect_objects_stereo(
        self,
        imgL: np.ndarray,
        imgR: np.ndarray,
        camera_config: Optional["CameraConfig"] = None,  # type: ignore
        camera_id: str = "stereo",
        camera_rotation: Optional[List[float]] = None,
        camera_position: Optional[List[float]] = None,
    ) -> DetectionResult:
        if self.use_yolo:
            return self.yolo_detector.detect_objects_stereo(
                imgL, imgR, camera_config, camera_id, camera_rotation, camera_position
            )
        if not STEREO_AVAILABLE:
            logging.error(
                "Stereo depth estimation not available - missing dependencies"
            )
            return DetectionResult(camera_id, 0, 0, [])

        if imgL is None or imgR is None:
            logging.warning("Empty stereo images provided to detector")
            return DetectionResult(camera_id, 0, 0, [])

        if imgL.shape != imgR.shape:
            logging.error(f"Stereo image size mismatch: {imgL.shape} vs {imgR.shape}")
            return DetectionResult(camera_id, 0, 0, [])

        if camera_config is None:
            camera_config = DEFAULT_CAMERA_CONFIG
            if camera_config is not None:
                logging.info(
                    f"Using default camera config: baseline={camera_config.baseline}m, FOV={camera_config.fov}°"
                )
            else:
                logging.error("No camera config available")
                return DetectionResult(camera_id, 0, 0, [])

        detection_result = self.detect_objects(imgL, camera_id=camera_id)

        if len(detection_result.detections) == 0:
            logging.info("No objects detected in stereo images")
            return detection_result

        logging.debug(
            f"Computing disparity map for {len(detection_result.detections)} detections"
        )

        if len(imgL.shape) == 3:
            imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        else:
            imgL_gray = imgL

        if len(imgR.shape) == 3:
            imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        else:
            imgR_gray = imgR

        recon_config = DEFAULT_RECONSTRUCTION_CONFIG
        disparity = calc_disparity(imgL_gray, imgR_gray, recon_config)

        save_disparity_map_debug(disparity)

        detections_with_depth = []
        h, w = imgL.shape[:2]

        import math

        focal_length = (w / 2.0) / math.tan(math.radians(camera_config.fov / 2.0))

        for det in detection_result.detections:
            depth_result = estimate_depth_from_bbox(
                disparity,
                (det.bbox_x, det.bbox_y, det.bbox_w, det.bbox_h),
                focal_length,
                camera_config.baseline,
            )
            depth_m = depth_result[0] if depth_result is not None else None
            disp_value = depth_result[1] if depth_result is not None else None

            world_pos = estimate_object_world_position_from_disparity(
                disparity,
                det.center_x,
                det.center_y,
                camera_config,
                w,
                h,
                min_disparity=1.0,  # Lower threshold for tabletop scenes
                max_depth=10.0,
                camera_rotation=camera_rotation,
                camera_position=camera_position,
            )

            # fallback: derive depth from Euclidean distance when bbox sampling failed
            fallback_depth_m = depth_m
            if (
                fallback_depth_m is None
                and world_pos is not None
                and camera_position is not None
            ):
                try:
                    dx = world_pos[0] - camera_position[0]
                    dy = world_pos[1] - camera_position[1]
                    dz = world_pos[2] - camera_position[2]
                    fallback_depth_m = math.sqrt(dx * dx + dy * dy + dz * dz)
                except Exception:
                    pass

            dimensions = None
            if fallback_depth_m is not None and focal_length is not None:
                dimensions = estimate_object_dimensions_from_bbox(
                    (det.bbox_x, det.bbox_y, det.bbox_w, det.bbox_h),
                    fallback_depth_m,
                    focal_length,
                    camera_config,
                )

            det_with_depth = DetectionObject(
                object_id=det.object_id,
                color=det.color,
                bbox=(det.bbox_x, det.bbox_y, det.bbox_w, det.bbox_h),
                confidence=det.confidence,
                world_position=world_pos,
                depth_m=depth_m,
                disparity=disp_value,
                dimensions=dimensions,
            )

            detections_with_depth.append(det_with_depth)

            if world_pos:
                depth_str = f"{depth_m:.3f}m" if depth_m is not None else "N/A"
                disp_str = f"{disp_value:.1f}px" if disp_value is not None else "N/A"
                logging.debug(
                    f"{det.color.upper()} cube: pixel ({det.center_x}, {det.center_y}) "
                    f"→ world ({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f})m, "
                    f"depth={depth_str}, disp={disp_str}"
                )
            else:
                logging.debug(
                    f"{det.color.upper()} cube at pixel ({det.center_x}, {det.center_y}) "
                    f"- failed to estimate depth"
                )

        return DetectionResult(
            camera_id,
            detection_result.image_width,
            detection_result.image_height,
            detections_with_depth,
        )

    def detect_objects_segmented(
        self, image: np.ndarray, camera_id: str = "unknown"
    ) -> DetectionResult:
        """YOLO segmentation detection. Falls back to bbox-only when YOLO_TASK!='segment'."""
        if not self.use_yolo or not YOLO_AVAILABLE:
            logging.warning(
                "YOLO not available; falling back to bbox-only detect_objects"
            )
            return self.detect_objects(image, camera_id)

        if YOLO_TASK != "segment":
            logging.debug(
                f"YOLO_TASK='{YOLO_TASK}' is not 'segment'; returning bbox-only result"
            )
            return self.detect_objects(image, camera_id)

        try:
            result = self.yolo_detector.detect_objects(image, camera_id)  # type: ignore[name-defined]
            return result
        except Exception as e:
            logging.error(
                f"Segmentation detection failed: {e}; falling back to detect_objects"
            )
            return self.detect_objects(image, camera_id)

    def _detect_all_objects(self, image: np.ndarray) -> List[Dict]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        detections = []

        mask_red_1 = cv2.inRange(hsv, self.red_lower_1, self.red_upper_1)
        mask_red_2 = cv2.inRange(hsv, self.red_lower_2, self.red_upper_2)
        mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)

        mask_blue = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        mask_green = cv2.inRange(hsv, self.green_lower, self.green_upper)
        mask_yellow = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        mask_orange = cv2.inRange(hsv, self.orange_lower, self.orange_upper)
        mask_purple = cv2.inRange(hsv, self.purple_lower, self.purple_upper)
        mask_cyan = cv2.inRange(hsv, self.cyan_lower, self.cyan_upper)
        mask_magenta = cv2.inRange(hsv, self.magenta_lower, self.magenta_upper)

        for color_name, mask in [
            ("red", mask_red),
            ("blue", mask_blue),
            ("green", mask_green),
            ("yellow", mask_yellow),
            ("orange", mask_orange),
            ("purple", mask_purple),
            ("cyan", mask_cyan),
            ("magenta", mask_magenta),
        ]:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            accepted_count = 0
            logging.debug(f"  {color_name.upper()}: Analyzing {len(contours)} contours")

            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h

                if area < self.min_area or area > self.max_area:
                    logging.debug(
                        f"    Contour {i}: Rejected by area ({area}px, need {self.min_area}-{self.max_area})"
                    )
                    continue

                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < self.min_aspect or aspect_ratio > self.max_aspect:
                    logging.debug(
                        f"    Contour {i}: Rejected by aspect ratio ({aspect_ratio:.2f}, need {self.min_aspect}-{self.max_aspect})"
                    )
                    continue

                # Calculate confidence based on how well contour fills bounding box
                contour_area = cv2.contourArea(contour)
                bbox_area = w * h
                fill_ratio = contour_area / bbox_area if bbox_area > 0 else 0

                # Also consider how many pixels in bbox match the color
                roi_mask = mask[y : y + h, x : x + w]
                color_ratio = np.sum(roi_mask > 0) / bbox_area if bbox_area > 0 else 0

                # Confidence is combination of shape and color match
                confidence = min((fill_ratio * 0.5 + color_ratio * 0.5) * 1.2, 1.0)

                # Filter by confidence
                if confidence < self.min_confidence:
                    logging.debug(
                        f"    Contour {i}: Rejected by confidence ({confidence:.2f}, need >={self.min_confidence})"
                    )
                    continue

                accepted_count += 1
                logging.debug(
                    f"    Contour {i}: ACCEPTED - area={area}px, aspect={aspect_ratio:.2f}, conf={confidence:.2f}"
                )

                detections.append(
                    {
                        "color": color_name,
                        "bbox": (x, y, w, h),
                        "confidence": confidence,
                    }
                )

            # Summary log for each color
            if accepted_count > 0:
                logging.info(
                    f"  {color_name.upper()}: {accepted_count} cube(s) detected"
                )

        return detections

    def _save_debug_image(
        self, image: np.ndarray, detections: List[DetectionObject], camera_id: str
    ):
        """
        Save annotated image with bounding boxes for debugging

        Args:
            image: Original image
            detections: List of detected objects
            camera_id: Camera ID for filename
        """
        debug_image = image.copy()

        # Color map for visualization
        color_map = {
            "red": (0, 0, 255),  # BGR
            "blue": (255, 0, 0),  # BGR
            "green": (0, 255, 0),  # BGR
            "yellow": (0, 255, 255),  # BGR
            "purple": (128, 0, 128),  # BGR
            "orange": (0, 165, 255),  # BGR
            "cyan": (255, 255, 0),  # BGR
            "magenta": (255, 0, 255),  # BGR
        }

        # Draw bounding boxes
        for det in detections:
            color = color_map.get(det.color, (0, 255, 0))

            # Draw rectangle
            cv2.rectangle(
                debug_image,
                (det.bbox_x, det.bbox_y),
                (det.bbox_x + det.bbox_w, det.bbox_y + det.bbox_h),
                color,
                2,
            )

            # Draw center point
            cv2.circle(debug_image, (det.center_x, det.center_y), 5, color, -1)

            # Draw label
            label = f"{det.color} {det.confidence:.2f}"
            cv2.putText(
                debug_image,
                label,
                (det.bbox_x, det.bbox_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Save to debug directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.debug_dir / f"{camera_id}_{timestamp}.jpg"
        cv2.imwrite(str(filename), debug_image)
        logging.debug(f"Saved debug image: {filename}")


def main():
    """
    Test the detector on a sample image
    """
    import sys

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    if len(sys.argv) < 2:
        print("Usage: python ObjectDetector.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read image from {image_path}")
        sys.exit(1)

    detector = CubeDetector()
    result = detector.detect_objects(image, camera_id="test")

    print(f"\nDetected {len(result.detections)} cubes:")
    for det in result.detections:
        print(
            f"  {det.color.upper()} cube at ({det.center_x}, {det.center_y}) - confidence: {det.confidence:.2f}"
        )


if __name__ == "__main__":
    main()
