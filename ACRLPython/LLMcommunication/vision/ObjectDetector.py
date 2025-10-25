#!/usr/bin/env python3
"""
ObjectDetector.py - Color-based cube detection in pixel space

Detects red and blue cubes using HSV color segmentation and contour detection.
Returns bounding boxes and centroids in pixel coordinates. Unity handles the
conversion from pixel coordinates to world coordinates using raycasting.

Supports stereo mode: Detects objects and estimates 3D world positions using
stereo disparity.

Usage:
    detector = CubeDetector()
    result = detector.detect_cubes(image, camera_id="AR4Left")

    # Stereo mode
    result = detector.detect_cubes_stereo(imgL, imgR, camera_config)
"""

import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import sys
import importlib.util

# Setup paths carefully to avoid config name conflicts
_stereo_path = Path(__file__).parent.parent.parent / "StereoImageReconstruction"
_package_dir = Path(__file__).parent.parent
_acrl_root = Path(__file__).parent.parent.parent  # ACRLPython directory

# Add paths to sys.path for imports
if str(_package_dir) not in sys.path:
    sys.path.insert(0, str(_package_dir))
if str(_acrl_root) not in sys.path:
    sys.path.insert(0, str(_acrl_root))

# Import stereo depth estimation
try:
    # Import using the full module path
    from StereoImageReconstruction.config import (
        CameraConfig,
        DEFAULT_CAMERA_CONFIG,
    )
    from LLMCommunication.vision.DepthEstimator import estimate_object_world_position

    STEREO_AVAILABLE = True
    logging.info("Stereo depth estimation available")
except Exception as e:
    logging.warning(f"Stereo depth estimation not available: {e}")
    STEREO_AVAILABLE = False
    # Define dummy types for type hints when stereo is not available
    CameraConfig = type('CameraConfig', (), {})
    DEFAULT_CAMERA_CONFIG = None
    # Define dummy function when stereo is not available
    def estimate_object_world_position(*args, **kwargs) -> Optional[Tuple[float, float, float]]:
        """Dummy function when stereo depth estimation is not available"""
        return None

# NOW import LLMCommunication config (after stereo setup is done)
# Import config module directly to avoid circular import through package __init__
_config_path = Path(__file__).parent.parent / "config.py"
config_spec = importlib.util.spec_from_file_location("llm_config", _config_path)
if config_spec is None or config_spec.loader is None:
    raise ImportError("Failed to load LLMCommunication config")
cfg = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(cfg)


class DetectionObject:
    """
    Represents a single detected cube in pixel space (and optionally 3D world space)
    """

    def __init__(
        self,
        object_id: int,
        color: str,
        bbox: Tuple[int, int, int, int],  # (x, y, width, height)
        confidence: float,
        world_position: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Initialize a detected object

        Args:
            object_id: Unique ID for this detection in the current frame
            color: Color of the cube ('red' or 'blue')
            bbox: Bounding box as (x, y, width, height) in pixels
            confidence: Detection confidence score (0.0-1.0)
            world_position: Optional 3D world position (x, y, z) in meters
        """
        self.object_id = object_id
        self.color = color
        self.bbox_x, self.bbox_y, self.bbox_w, self.bbox_h = bbox
        self.confidence = confidence
        self.world_position = world_position

        # Calculate center point
        self.center_x = int(self.bbox_x + self.bbox_w / 2)
        self.center_y = int(self.bbox_y + self.bbox_h / 2)

    def to_dict(self) -> Dict:
        """
        Convert to dictionary for JSON serialization

        Returns:
            Dictionary representation of the detection
        """
        result = {
            "id": self.object_id,
            "color": self.color,
            "bbox_px": {
                "x": self.bbox_x,
                "y": self.bbox_y,
                "width": self.bbox_w,
                "height": self.bbox_h,
            },
            "center_px": {"x": self.center_x, "y": self.center_y},
            "confidence": round(self.confidence, 3),
        }

        # Add world position if available
        if self.world_position is not None:
            result["world_position"] = {
                "x": round(self.world_position[0], 4),
                "y": round(self.world_position[1], 4),
                "z": round(self.world_position[2], 4),
            }

        return result


class DetectionResult:
    """
    Container for all detections in a single frame
    """

    def __init__(
        self,
        camera_id: str,
        image_width: int,
        image_height: int,
        detections: List[DetectionObject],
    ):
        """
        Initialize detection result

        Args:
            camera_id: ID of the camera that captured the image
            image_width: Width of the source image in pixels
            image_height: Height of the source image in pixels
            detections: List of detected objects
        """
        self.camera_id = camera_id
        self.image_width = image_width
        self.image_height = image_height
        self.detections = detections
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """
        Convert to dictionary for JSON serialization

        Returns:
            Dictionary representation of all detections
        """
        return {
            "success": True,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "detections": [d.to_dict() for d in self.detections],
        }


class CubeDetector:
    """
    Color-based cube detector using HSV segmentation
    """

    def __init__(self):
        """
        Initialize the cube detector with color ranges from config
        """
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

        logging.info("CubeDetector initialized")

    def detect_cubes(
        self, image: np.ndarray, camera_id: str = "unknown"
    ) -> DetectionResult:
        """
        Detect red and blue cubes in an image

        Args:
            image: OpenCV image (BGR format)
            camera_id: ID of the camera for metadata

        Returns:
            DetectionResult containing all detected cubes
        """
        if image is None or image.size == 0:
            logging.warning("Empty image provided to detector")
            return DetectionResult(camera_id, 0, 0, [])

        height, width = image.shape[:2]

        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Detect red cubes
        red_detections = self._detect_color(
            hsv,
            color_name="red",
            lower_ranges=[self.red_lower_1, self.red_lower_2],
            upper_ranges=[self.red_upper_1, self.red_upper_2],
        )

        # Detect blue cubes
        blue_detections = self._detect_color(
            hsv, color_name="blue", lower_ranges=[self.blue_lower], upper_ranges=[self.blue_upper]
        )

        # Combine and assign IDs
        all_detections = []
        object_id = 0

        for det in red_detections + blue_detections:
            all_detections.append(
                DetectionObject(
                    object_id=object_id,
                    color=det["color"],
                    bbox=det["bbox"],
                    confidence=det["confidence"],
                )
            )
            object_id += 1

        # Save debug image if enabled
        if self.enable_debug and len(all_detections) > 0:
            self._save_debug_image(image, all_detections, camera_id)

        logging.info(
            f"Detected {len(all_detections)} cubes ({len(red_detections)} red, {len(blue_detections)} blue)"
        )

        return DetectionResult(camera_id, width, height, all_detections)

    def detect_cubes_stereo(
        self,
        imgL: np.ndarray,
        imgR: np.ndarray,
        camera_config: Optional["CameraConfig"] = None,  # type: ignore
        camera_id: str = "stereo",
    ) -> DetectionResult:
        """
        Detect cubes in stereo images and estimate 3D world positions.

        Detects objects in the left image and computes depth using stereo disparity.

        Args:
            imgL: Left camera image (BGR format)
            imgR: Right camera image (BGR format)
            camera_config: Camera calibration parameters (baseline, FOV, etc.)
            camera_id: ID of the camera for metadata

        Returns:
            DetectionResult containing detected cubes with 3D world positions
        """
        if not STEREO_AVAILABLE:
            logging.error("Stereo depth estimation not available - missing dependencies")
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
                logging.info(f"Using default camera config: baseline={camera_config.baseline}m, FOV={camera_config.fov}°")
            else:
                logging.error("No camera config available")
                return DetectionResult(camera_id, 0, 0, [])

        # First, detect cubes in the left image (using existing 2D detection)
        detection_result = self.detect_cubes(imgL, camera_id=camera_id)

        # Now estimate 3D world position for each detection
        detections_with_depth = []

        for det in detection_result.detections:
            # Estimate world position at bounding box center
            world_pos = estimate_object_world_position(
                imgL, imgR, det.center_x, det.center_y, camera_config
            )

            # Create new detection object with world position
            det_with_depth = DetectionObject(
                object_id=det.object_id,
                color=det.color,
                bbox=(det.bbox_x, det.bbox_y, det.bbox_w, det.bbox_h),
                confidence=det.confidence,
                world_position=world_pos,
            )

            detections_with_depth.append(det_with_depth)

            if world_pos:
                logging.info(
                    f"{det.color.upper()} cube at pixel ({det.center_x}, {det.center_y}) "
                    f"→ world pos ({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f})m"
                )
            else:
                logging.warning(
                    f"{det.color.upper()} cube at pixel ({det.center_x}, {det.center_y}) "
                    f"- failed to estimate depth"
                )

        return DetectionResult(
            camera_id, detection_result.image_width, detection_result.image_height, detections_with_depth
        )

    def _detect_color(
        self,
        hsv_image: np.ndarray,
        color_name: str,
        lower_ranges: List[np.ndarray],
        upper_ranges: List[np.ndarray],
    ) -> List[Dict]:
        """
        Detect objects of a specific color

        Args:
            hsv_image: Image in HSV color space
            color_name: Name of the color ('red' or 'blue')
            lower_ranges: List of lower HSV bounds
            upper_ranges: List of upper HSV bounds

        Returns:
            List of detection dictionaries
        """
        # Create combined mask for all color ranges
        mask = None
        for lower, upper in zip(lower_ranges, upper_ranges):
            range_mask = cv2.inRange(hsv_image, lower, upper)
            if mask is None:
                mask = range_mask
            else:
                mask = cv2.bitwise_or(mask, range_mask)

        # Check if mask was created
        if mask is None:
            logging.warning(f"No color ranges provided for {color_name} detection")
            return []

        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue

            # Filter by aspect ratio (allow some perspective distortion)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < self.min_aspect or aspect_ratio > self.max_aspect:
                continue

            # Calculate confidence based on how well the contour fills the bounding box
            contour_area = cv2.contourArea(contour)
            bbox_area = w * h
            fill_ratio = contour_area / bbox_area if bbox_area > 0 else 0
            confidence = min(fill_ratio * 1.2, 1.0)  # Scale up slightly

            # Filter by confidence
            if confidence < self.min_confidence:
                continue

            detections.append(
                {"color": color_name, "bbox": (x, y, w, h), "confidence": confidence}
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
        color_map = {"red": (0, 0, 255), "blue": (255, 0, 0)}

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
    result = detector.detect_cubes(image, camera_id="test")

    print(f"\nDetected {len(result.detections)} cubes:")
    for det in result.detections:
        print(
            f"  {det.color.upper()} cube at ({det.center_x}, {det.center_y}) - confidence: {det.confidence:.2f}"
        )


if __name__ == "__main__":
    main()
