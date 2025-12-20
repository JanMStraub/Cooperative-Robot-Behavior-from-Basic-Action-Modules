#!/usr/bin/env python3
"""
ObjectTracker.py - Simple IOU-based object tracking for persistent IDs across frames

Implements a lightweight tracking system suitable for controlled indoor environments
with relatively static objects. Uses Intersection-over-Union (IOU) for track-detection
association, providing persistent object IDs for velocity estimation and smoother
multi-robot coordination.

Features:
- IOU-based track-detection matching
- Automatic track aging and cleanup
- Configurable max age and minimum IOU thresholds
- Position history for velocity estimation
- Simple and efficient for controlled workspaces

Usage:
    from vision.ObjectTracker import ObjectTracker
    from vision.DetectionDataModels import DetectionObject

    # Initialize tracker
    tracker = ObjectTracker(max_age=5, min_iou=0.3)

    # Update with new detections each frame
    for frame in video_stream:
        detections = detector.detect(frame)
        tracked_detections = tracker.update(detections)

        # tracked_detections now have persistent track_id
        for det in tracked_detections:
            print(f"Track {det.track_id}: {det.color} at ({det.center_x}, {det.center_y})")
"""

import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

# Import detection data models
try:
    from .DetectionDataModels import DetectionObject
except ImportError:
    from vision.DetectionDataModels import DetectionObject

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


@dataclass
class Track:
    """
    Represents a tracked object over multiple frames.

    Attributes:
        track_id: Unique persistent ID for this track
        bbox: Current bounding box (x, y, width, height)
        color: Object class/color name
        age: Frames since last detection (0 = just detected)
        hits: Total number of detections matched to this track
        position_history: Recent center positions for velocity estimation
    """

    track_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    color: str
    age: int = 0
    hits: int = 1
    position_history: List[Tuple[int, int]] = field(default_factory=list)

    def update(self, detection: DetectionObject):
        """
        Update track with new detection.

        Args:
            detection: New detection to update this track with
        """
        self.bbox = (
            detection.bbox_x,
            detection.bbox_y,
            detection.bbox_w,
            detection.bbox_h,
        )
        self.color = detection.color
        self.age = 0
        self.hits += 1

        # Update position history (keep last 10 positions for velocity estimation)
        self.position_history.append((detection.center_x, detection.center_y))
        if len(self.position_history) > 10:
            self.position_history.pop(0)

    def predict_next_position(self) -> Optional[Tuple[int, int]]:
        """
        Predict next position based on velocity from position history.

        Returns:
            Predicted (x, y) position or None if insufficient history
        """
        if len(self.position_history) < 2:
            return None

        # Simple linear prediction based on last 2 positions
        x1, y1 = self.position_history[-2]
        x2, y2 = self.position_history[-1]

        # Velocity
        vx = x2 - x1
        vy = y2 - y1

        # Predict next position
        pred_x = x2 + vx
        pred_y = y2 + vy

        return (int(pred_x), int(pred_y))

    def get_velocity(self) -> Optional[Tuple[float, float]]:
        """
        Calculate current velocity in pixels per frame.

        Returns:
            (vx, vy) velocity or None if insufficient history
        """
        if len(self.position_history) < 2:
            return None

        x1, y1 = self.position_history[-2]
        x2, y2 = self.position_history[-1]

        return (float(x2 - x1), float(y2 - y1))


class ObjectTracker:
    """
    Simple IOU-based tracker for persistent object IDs across frames.

    Suitable for controlled indoor environments with relatively static objects.
    Uses Intersection-over-Union (IOU) for track-detection association.

    Attributes:
        max_age: Max frames a track survives without detection before deletion
        min_iou: Minimum IOU for track-detection association (0.0-1.0)
        tracks: List of active Track objects
        next_id: Next track ID to assign

    Example:
        tracker = ObjectTracker(max_age=5, min_iou=0.3)

        for frame in video:
            detections = detector.detect(frame)
            tracked_detections = tracker.update(detections)

            for det in tracked_detections:
                print(f"Object {det.track_id}: {det.color}")
    """

    def __init__(self, max_age: int = 5, min_iou: float = 0.3):
        """
        Initialize object tracker.

        Args:
            max_age: Max frames a track survives without detection (default: 5)
            min_iou: Minimum IOU for track-detection association (default: 0.3)
        """
        self.max_age = max_age
        self.min_iou = min_iou
        self.tracks: List[Track] = []
        self.next_id = 1

        logging.info(f"ObjectTracker initialized: max_age={max_age}, min_iou={min_iou}")

    def update(self, detections: List[DetectionObject]) -> List[DetectionObject]:
        """
        Update tracks with new detections and assign persistent IDs.

        Process:
        1. Match detections to existing tracks using IOU
        2. Update matched tracks
        3. Create new tracks for unmatched detections
        4. Age out stale tracks
        5. Assign track IDs to detections

        Args:
            detections: List of new detections from current frame

        Returns:
            List of detections with track_id assigned
        """
        # Match detections to existing tracks
        matched_tracks, matched_detections, unmatched_detections = (
            self._associate_detections(detections)
        )

        # Update matched tracks
        for track_idx, det_idx in zip(matched_tracks, matched_detections):
            self.tracks[track_idx].update(detections[det_idx])

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self._create_new_track(detections[det_idx])

        # Age tracks and remove stale ones
        self._age_tracks()

        # Assign track IDs to detections
        # Create a mapping from detection index to track index
        det_to_track = {}
        for track_idx, det_idx in zip(matched_tracks, matched_detections):
            det_to_track[det_idx] = self.tracks[track_idx].track_id

        # For unmatched detections, find their newly created tracks
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            # Find newly created track (last created tracks)
            for track in self.tracks:
                if track.hits == 1:  # Newly created track
                    det_bbox = (det.bbox_x, det.bbox_y, det.bbox_w, det.bbox_h)
                    if track.bbox == det_bbox and track.color == det.color:
                        det_to_track[det_idx] = track.track_id
                        break

        # Create tracked detections with track IDs
        tracked_detections = []
        for idx, det in enumerate(detections):
            track_id = det_to_track.get(idx, None)

            tracked_det = DetectionObject(
                object_id=det.object_id,
                color=det.color,
                bbox=(det.bbox_x, det.bbox_y, det.bbox_w, det.bbox_h),
                confidence=det.confidence,
                world_position=det.world_position,
                depth_m=det.depth_m,
                disparity=det.disparity,
                track_id=track_id,  # Add persistent track ID
            )
            tracked_detections.append(tracked_det)

        logging.debug(
            f"Tracking: {len(detections)} detections → {len(self.tracks)} active tracks"
        )

        return tracked_detections

    def _associate_detections(
        self, detections: List[DetectionObject]
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Associate detections to existing tracks using IOU.

        Args:
            detections: List of new detections

        Returns:
            Tuple of (matched_tracks, matched_detections, unmatched_detections)
            - matched_tracks: Indices of tracks that were matched
            - matched_detections: Indices of detections that were matched
            - unmatched_detections: Indices of detections without match
        """
        if len(self.tracks) == 0:
            # No existing tracks, all detections are unmatched
            return ([], [], list(range(len(detections))))

        if len(detections) == 0:
            # No new detections
            return ([], [], [])

        # Build IOU matrix (tracks x detections)
        import numpy as np

        iou_matrix = np.zeros((len(self.tracks), len(detections)))

        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                # Only match same class
                if track.color != det.color:
                    continue

                det_bbox = (det.bbox_x, det.bbox_y, det.bbox_w, det.bbox_h)
                iou = self._calculate_iou(track.bbox, det_bbox)
                iou_matrix[t_idx, d_idx] = iou

        # Greedy matching: match highest IOU pairs first
        matched_tracks = []
        matched_detections = []
        used_tracks = set()
        used_detections = set()

        # Flatten IOU matrix and sort by IOU (highest first)
        track_det_pairs = []
        for t_idx in range(len(self.tracks)):
            for d_idx in range(len(detections)):
                if iou_matrix[t_idx, d_idx] >= self.min_iou:
                    track_det_pairs.append((t_idx, d_idx, iou_matrix[t_idx, d_idx]))

        track_det_pairs.sort(key=lambda x: x[2], reverse=True)  # Sort by IOU

        # Match greedily
        for t_idx, d_idx, iou in track_det_pairs:
            if t_idx not in used_tracks and d_idx not in used_detections:
                matched_tracks.append(t_idx)
                matched_detections.append(d_idx)
                used_tracks.add(t_idx)
                used_detections.add(d_idx)

        # Find unmatched detections
        unmatched_detections = [
            d_idx for d_idx in range(len(detections)) if d_idx not in used_detections
        ]

        return (matched_tracks, matched_detections, unmatched_detections)

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

    def _create_new_track(self, detection: DetectionObject):
        """
        Create new track for unmatched detection.

        Args:
            detection: Detection to create track for
        """
        bbox = (detection.bbox_x, detection.bbox_y, detection.bbox_w, detection.bbox_h)
        track = Track(
            track_id=self.next_id,
            bbox=bbox,
            color=detection.color,
            age=0,
            hits=1,
        )
        track.position_history.append((detection.center_x, detection.center_y))

        self.tracks.append(track)
        self.next_id += 1

        logging.debug(f"Created new track {track.track_id} for {detection.color}")

    def _age_tracks(self):
        """
        Increment age for unmatched tracks and remove stale ones.
        """
        # Age all tracks that weren't just updated
        for track in self.tracks:
            track.age += 1

        # Remove tracks that are too old
        before_count = len(self.tracks)
        self.tracks = [track for track in self.tracks if track.age <= self.max_age]
        removed_count = before_count - len(self.tracks)

        if removed_count > 0:
            logging.debug(
                f"Removed {removed_count} stale tracks (age > {self.max_age})"
            )

    def get_active_tracks(self) -> List[Track]:
        """
        Get list of all active tracks.

        Returns:
            List of Track objects
        """
        return self.tracks.copy()

    def reset(self):
        """
        Reset tracker, clearing all tracks.
        """
        self.tracks.clear()
        self.next_id = 1
        logging.info("ObjectTracker reset")


# ===========================
# Main (for testing)
# ===========================


def main():
    """Test ObjectTracker with synthetic detections"""
    import numpy as np

    print("=== ObjectTracker Test ===\n")

    tracker = ObjectTracker(max_age=3, min_iou=0.3)

    # Simulate 10 frames with moving objects
    for frame_idx in range(10):
        print(f"Frame {frame_idx}:")

        # Simulate 2-3 detections per frame
        detections = []

        # Object 1: Moving left to right
        if frame_idx < 8:
            x = 100 + frame_idx * 50
            det1 = DetectionObject(
                object_id=frame_idx * 10 + 1,
                color="red_cube",
                bbox=(x, 100, 50, 50),
                confidence=0.9,
            )
            detections.append(det1)

        # Object 2: Stationary
        if frame_idx > 2:
            det2 = DetectionObject(
                object_id=frame_idx * 10 + 2,
                color="blue_cube",
                bbox=(300, 200, 50, 50),
                confidence=0.85,
            )
            detections.append(det2)

        # Update tracker
        tracked_dets = tracker.update(detections)

        # Print results
        for det in tracked_dets:
            track_id_str = f"Track {det.track_id}" if det.track_id else "No track"
            print(f"  {track_id_str}: {det.color} at ({det.center_x}, {det.center_y})")

        print(f"  Active tracks: {len(tracker.get_active_tracks())}\n")

    print("=== Test Complete ===")


if __name__ == "__main__":
    main()
