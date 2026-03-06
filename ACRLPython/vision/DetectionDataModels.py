#!/usr/bin/env python3
"""
DetectionDataModels.py - Shared data models for object detection

Contains common data structures used by both HSV-based and YOLO-based detectors.
Separated to avoid circular imports between ObjectDetector and YOLODetector.
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime


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
        depth_m: Optional[float] = None,
        disparity: Optional[float] = None,
        track_id: Optional[int] = None,
        dimensions: Optional[Tuple[float, float, float]] = None,
        mask=None,  # Optional[np.ndarray]: segmentation mask (H x W), task=segment only
    ):
        """
        Initialize a detected object

        Args:
            object_id: Unique ID for this detection in the current frame
            color: Color of the cube ('red' or 'blue')
            bbox: Bounding box as (x, y, width, height) in pixels
            confidence: Detection confidence score (0.0-1.0)
            world_position: Optional 3D world position (x, y, z) in meters
            depth_m: Optional depth in meters (Z distance from camera)
            disparity: Optional disparity value in pixels
            track_id: Optional persistent track ID across frames (for object tracking)
            dimensions: Optional 3D dimensions (width, height, depth) in meters
        """
        self.object_id = object_id
        self.color = color
        self.bbox_x, self.bbox_y, self.bbox_w, self.bbox_h = bbox
        self.confidence = confidence
        self.world_position = world_position
        self.depth_m = depth_m
        self.disparity = disparity
        self.track_id = track_id
        self.dimensions = dimensions
        self.mask = mask

        # Calculate center point
        self.center_x = int(self.bbox_x + self.bbox_w / 2)
        self.center_y = int(self.bbox_y + self.bbox_h / 2)

    def to_dict(self) -> Dict:
        """
        Convert to dictionary for JSON serialization

        Adapts field names based on whether 3D data is present:
        - With 3D data: Uses Unity DepthResult format (bbox, pixel_center)
        - Without 3D data: Uses Unity DetectionResult format (bbox_px, center_px)

        Returns:
            Dictionary representation of the detection
        """
        # Use Unity DepthResult format if 3D data is present
        if self.world_position is not None:
            result = {
                "color": self.color,
                "confidence": round(self.confidence, 3),
                # Unity DepthResult uses "bbox" (not "bbox_px")
                "bbox": {
                    "x": self.bbox_x,
                    "y": self.bbox_y,
                    "width": self.bbox_w,
                    "height": self.bbox_h,
                },
                # Unity DepthResult uses "pixel_center" (not "center_px")
                "pixel_center": {"x": self.center_x, "y": self.center_y},
                "world_position": {
                    "x": round(self.world_position[0], 4),
                    "y": round(self.world_position[1], 4),
                    "z": round(self.world_position[2], 4),
                },
            }

            # Add depth_m and disparity if available
            if self.depth_m is not None:
                result["depth_m"] = round(self.depth_m, 4)
            if self.disparity is not None:
                result["disparity"] = round(self.disparity, 2)

            # Add dimensions if available
            if self.dimensions is not None:
                result["dimensions"] = {
                    "width": round(self.dimensions[0], 4),
                    "height": round(self.dimensions[1], 4),
                    "depth": round(self.dimensions[2], 4),
                }
        else:
            # Use Unity DetectionResult format (2D only)
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

        # Add track_id if available (for object tracking)
        if self.track_id is not None:
            result["track_id"] = self.track_id

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
