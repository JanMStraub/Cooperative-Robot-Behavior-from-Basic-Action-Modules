#!/usr/bin/env python3
"""
ObjectDetector.py - General object detection using contours

Detects any colored objects using contour detection and edge detection.
Returns bounding boxes and centroids in pixel coordinates.

Supports stereo mode: Detects objects and estimates 3D world positions using
stereo disparity.

Usage:
    detector = CubeDetector()
    result = detector.detect_objects(image, camera_id="AR4Left")

    # Stereo mode
    result = detector.detect_objects_stereo(imgL, imgR, camera_config)
"""

import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2

# Import config - try both import styles
try:
    import LLMConfig as cfg
except ImportError:
    from .. import LLMConfig as cfg

# Import shared detection data models
try:
    from .DetectionDataModels import DetectionObject, DetectionResult
except ImportError:
    from vision.DetectionDataModels import DetectionObject, DetectionResult

# Import YOLO detector if enabled
YOLO_AVAILABLE = False
if getattr(cfg, "USE_YOLO", False):
    try:
        from .YOLODetector import YOLODetector

        YOLO_AVAILABLE = True
        logging.info("YOLO detection enabled")
    except ImportError as e:
        logging.warning(f"YOLO enabled in config but import failed: {e}")
        logging.warning("Falling back to HSV color detection")

# Import stereo depth estimation
try:
    try:
        from .StereoConfig import (
            CameraConfig,
            ReconstructionConfig,
            DEFAULT_CAMERA_CONFIG,
            DEFAULT_RECONSTRUCTION_CONFIG,
        )
    except ImportError:
        from StereoConfig import (
            CameraConfig,
            ReconstructionConfig,
            DEFAULT_CAMERA_CONFIG,
            DEFAULT_RECONSTRUCTION_CONFIG,
        )

    # Use depth estimator with integrated disparity calculation
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
    logging.debug("Stereo depth estimation available")
except Exception as e:
    logging.warning(f"Stereo depth estimation not available: {e}")
    STEREO_AVAILABLE = False
    # Define dummy types for type hints when stereo is not available
    CameraConfig = type("CameraConfig", (), {})
    DEFAULT_CAMERA_CONFIG = None
    DEFAULT_RECONSTRUCTION_CONFIG = None
    ReconstructionConfig = type("ReconstructionConfig", (), {})

    # Define dummy functions when stereo is not available
    def estimate_object_world_position_from_disparity(
        *args, **kwargs
    ) -> Optional[Tuple[float, float, float]]:
        """Dummy function when stereo depth estimation is not available"""
        return None

    def save_disparity_map_debug(*args, **kwargs) -> None:
        """Dummy function when stereo depth estimation is not available"""
        pass

    def calc_disparity(*args, **kwargs) -> np.ndarray:
        """Dummy function when stereo depth estimation is not available"""
        return np.zeros((0, 0), dtype=np.float32)


class CubeDetector:
    """
    Color-based cube detector using HSV segmentation
    """

    def __init__(self):
        """
        Initialize the cube detector with color ranges from config

        If USE_YOLO is enabled in config, uses YOLODetector.
        Otherwise, uses HSV color-based detection.
        """
        # Check if YOLO should be used
        self.use_yolo = YOLO_AVAILABLE and getattr(cfg, "USE_YOLO", False)

        if self.use_yolo and YOLO_AVAILABLE:
            # Initialize YOLO detector (only if import succeeded)
            model_path = getattr(cfg, "YOLO_MODEL_PATH", None)
            try:
                self.yolo_detector = YOLODetector(model_path=model_path)  # type: ignore[name-defined]
                logging.info(
                    f"CubeDetector initialized with YOLO (model: {model_path})"
                )
            except Exception as e:
                logging.error(f"Failed to initialize YOLO detector: {e}")
                logging.info("Falling back to HSV color detection")
                self.use_yolo = False

        # Always initialize HSV detector (for fallback or direct use)
        # Red color ranges (HSV wraps around, so we need two ranges)
        self.red_lower_1 = np.array(cfg.RED_HSV_LOWER_1, dtype=np.uint8)
        self.red_upper_1 = np.array(cfg.RED_HSV_UPPER_1, dtype=np.uint8)
        self.red_lower_2 = np.array(cfg.RED_HSV_LOWER_2, dtype=np.uint8)
        self.red_upper_2 = np.array(cfg.RED_HSV_UPPER_2, dtype=np.uint8)

        # Blue color range
        self.blue_lower = np.array(cfg.BLUE_HSV_LOWER, dtype=np.uint8)
        self.blue_upper = np.array(cfg.BLUE_HSV_UPPER, dtype=np.uint8)

        # Detection thresholds
        self.min_area = cfg.MIN_CUBE_AREA_PX
        self.max_area = cfg.MAX_CUBE_AREA_PX
        self.min_aspect = cfg.MIN_ASPECT_RATIO
        self.max_aspect = cfg.MAX_ASPECT_RATIO
        self.min_confidence = cfg.MIN_CONFIDENCE

        # Debug settings
        self.enable_debug = cfg.ENABLE_DEBUG_IMAGES
        if self.enable_debug:
            self.debug_dir = Path(cfg.DEBUG_IMAGES_DIR)
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        if not self.use_yolo:
            logging.debug("CubeDetector initialized with HSV color detection")

    def detect_objects(
        self, image: np.ndarray, camera_id: str = "unknown"
    ) -> DetectionResult:
        """
        Detect any colored objects in an image

        Uses YOLO if enabled in config, otherwise uses HSV color detection.

        Args:
            image: OpenCV image (BGR format)
            camera_id: ID of the camera for metadata

        Returns:
            DetectionResult containing all detected objects
        """
        # Delegate to YOLO if enabled
        if self.use_yolo:
            return self.yolo_detector.detect_objects(image, camera_id)

        # Otherwise use HSV color detection
        if image is None or image.size == 0:
            logging.warning("Empty image provided to detector")
            return DetectionResult(camera_id, 0, 0, [])

        height, width = image.shape[:2]

        # Detect all objects using edge and contour detection
        detections = self._detect_all_objects(image)

        # Assign IDs
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

        # Save debug image if enabled (save even with 0 detections for debugging)
        if self.enable_debug:
            self._save_debug_image(image, all_detections, camera_id)

        logging.info(f"Detected {len(all_detections)} objects")

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
        """
        Detect cubes in stereo images and estimate 3D world positions.

        Uses YOLO if enabled in config, otherwise uses HSV color detection.
        Detects objects in the left image and computes depth using stereo disparity.

        Args:
            imgL: Left camera image (BGR format)
            imgR: Right camera image (BGR format)
            camera_config: Camera calibration parameters (baseline, FOV, etc.)
            camera_id: ID of the camera for metadata
            camera_rotation: Camera rotation [pitch, yaw, roll] in degrees
            camera_position: Camera position [x, y, z] in world space

        Returns:
            DetectionResult containing detected cubes with 3D world positions
        """
        # Delegate to YOLO if enabled
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

        # First, detect cubes in the left image (using existing 2D detection)
        detection_result = self.detect_objects(imgL, camera_id=camera_id)

        # If no detections, return early
        if len(detection_result.detections) == 0:
            logging.info("No objects detected in stereo images")
            return detection_result

        # OPTIMIZATION: Compute disparity map ONCE for all detections
        # This provides 80-95% speedup for multi-object scenes
        logging.debug(
            f"Computing disparity map for {len(detection_result.detections)} detections"
        )

        # Convert to grayscale if needed
        if len(imgL.shape) == 3:
            imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        else:
            imgL_gray = imgL

        if len(imgR.shape) == 3:
            imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        else:
            imgR_gray = imgR

        # Compute disparity once using default reconstruction config
        recon_config = DEFAULT_RECONSTRUCTION_CONFIG
        disparity = calc_disparity(imgL_gray, imgR_gray, recon_config)

        # Save disparity map for debugging (if enabled in config)
        save_disparity_map_debug(disparity)

        # Now estimate 3D world position for each detection using pre-computed disparity
        detections_with_depth = []
        h, w = imgL.shape[:2]

        for det in detection_result.detections:
            # Extract disparity value at detection center
            disp_value = None
            if (
                0 <= det.center_y < disparity.shape[0]
                and 0 <= det.center_x < disparity.shape[1]
            ):
                disp_value = float(disparity[det.center_y, det.center_x])

            # Calculate depth from disparity
            depth_m = None
            if disp_value is not None and disp_value > 1.0:  # Valid disparity
                # Depth = (baseline * focal_length) / disparity
                # focal_length = (image_width / 2) / tan(fov/2)
                import math

                focal_length = (w / 2.0) / math.tan(
                    math.radians(camera_config.fov / 2.0)
                )
                depth_m = (camera_config.baseline * focal_length) / disp_value

            # Estimate world position using pre-computed disparity (OPTIMIZED)
            # Use lower min_disparity (1.0px) to handle distant objects better
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

            # Create new detection object with world position, depth, and disparity
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

    def _detect_all_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect colored cubes in image using HSV color segmentation

        Args:
            image: OpenCV image (BGR format)

        Returns:
            List of detection dictionaries
        """
        # Convert to HSV for color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        detections = []

        # Detect red cubes (two ranges because red wraps around HSV)
        mask_red_1 = cv2.inRange(hsv, self.red_lower_1, self.red_upper_1)
        mask_red_2 = cv2.inRange(hsv, self.red_lower_2, self.red_upper_2)
        mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)

        # Detect blue/cyan cubes
        mask_blue = cv2.inRange(hsv, self.blue_lower, self.blue_upper)

        # Process each color mask
        for color_name, mask in [("red", mask_red), ("blue", mask_blue)]:
            # Apply morphological operations to clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find contours in the mask
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            accepted_count = 0
            logging.debug(f"  {color_name.upper()}: Analyzing {len(contours)} contours")

            for i, contour in enumerate(contours):
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h

                # Filter by area
                if area < self.min_area or area > self.max_area:
                    logging.debug(
                        f"    Contour {i}: Rejected by area ({area}px, need {self.min_area}-{self.max_area})"
                    )
                    continue

                # Filter by aspect ratio
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
                    f"  {color_name.upper()}: ✓ {accepted_count} cube(s) detected"
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
        color_map = {"red": (0, 0, 255), "blue": (0, 255, 255)}

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

    logging.basicConfig(level=logging.INFO, format=cfg.LOG_FORMAT)

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
