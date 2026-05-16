#!/usr/bin/env python3
"""IOU-based object tracker for persistent IDs across frames."""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

try:
    from .DetectionDataModels import DetectionObject
except ImportError:
    from vision.DetectionDataModels import DetectionObject

from core.LoggingSetup import get_logger

logger = get_logger(__name__)


@dataclass
class Track:
    """Tracked object with persistent ID and position history."""

    track_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    color: str
    age: int = 0
    hits: int = 1
    position_history: List[Tuple[int, int]] = field(default_factory=list)

    def update(self, detection: DetectionObject):
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
        """Linear prediction from last 2 positions. None if insufficient history."""
        if len(self.position_history) < 2:
            return None

        x1, y1 = self.position_history[-2]
        x2, y2 = self.position_history[-1]

        vx = x2 - x1
        vy = y2 - y1

        pred_x = x2 + vx
        pred_y = y2 + vy

        return (int(pred_x), int(pred_y))

    def get_velocity(self) -> Optional[Tuple[float, float]]:
        if len(self.position_history) < 2:
            return None

        x1, y1 = self.position_history[-2]
        x2, y2 = self.position_history[-1]

        return (float(x2 - x1), float(y2 - y1))


class ObjectTracker:
    """IOU-based tracker assigning persistent IDs across frames."""

    def __init__(self, max_age: int = 5, min_iou: float = 0.3):
        self.max_age = max_age
        self.min_iou = min_iou
        self.tracks: List[Track] = []
        self.next_id = 1

        logger.debug(f"ObjectTracker initialized: max_age={max_age}, min_iou={min_iou}")

    def update(self, detections: List[DetectionObject]) -> List[DetectionObject]:
        """Match detections to tracks via IOU, age out stale tracks, return detections with track_id."""
        matched_tracks, matched_detections, unmatched_detections = (
            self._associate_detections(detections)
        )

        for track_idx, det_idx in zip(matched_tracks, matched_detections):
            self.tracks[track_idx].update(detections[det_idx])

        for det_idx in unmatched_detections:
            self._create_new_track(detections[det_idx])

        # Must map before _age_tracks() removes tracks and invalidates indices
        det_to_track = {}
        for track_idx, det_idx in zip(matched_tracks, matched_detections):
            det_to_track[det_idx] = self.tracks[track_idx].track_id

        for det_idx in unmatched_detections:
            det = detections[det_idx]
            # Find newly created track (last created tracks)
            for track in self.tracks:
                if track.hits == 1:
                    det_bbox = (det.bbox_x, det.bbox_y, det.bbox_w, det.bbox_h)
                    if track.bbox == det_bbox and track.color == det.color:
                        det_to_track[det_idx] = track.track_id
                        break

        self._age_tracks()

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
                track_id=track_id,
            )
            tracked_detections.append(tracked_det)

        logger.debug(
            f"Tracking: {len(detections)} detections → {len(self.tracks)} active tracks"
        )

        return tracked_detections

    def _associate_detections(
        self, detections: List[DetectionObject]
    ) -> Tuple[List[int], List[int], List[int]]:
        if len(self.tracks) == 0:
            return ([], [], list(range(len(detections))))

        if len(detections) == 0:
            return ([], [], [])

        iou_matrix = np.zeros((len(self.tracks), len(detections)))

        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                # Only match same class
                if track.color != det.color:
                    continue

                det_bbox = (det.bbox_x, det.bbox_y, det.bbox_w, det.bbox_h)
                iou = self._calculate_iou(track.bbox, det_bbox)
                iou_matrix[t_idx, d_idx] = iou

        matched_tracks = []
        matched_detections = []
        used_tracks = set()
        used_detections = set()

        track_det_pairs = []
        for t_idx in range(len(self.tracks)):
            for d_idx in range(len(detections)):
                if iou_matrix[t_idx, d_idx] >= self.min_iou:
                    track_det_pairs.append((t_idx, d_idx, iou_matrix[t_idx, d_idx]))

        track_det_pairs.sort(key=lambda x: x[2], reverse=True)

        for t_idx, d_idx, iou in track_det_pairs:
            if t_idx not in used_tracks and d_idx not in used_detections:
                matched_tracks.append(t_idx)
                matched_detections.append(d_idx)
                used_tracks.add(t_idx)
                used_detections.add(d_idx)

        unmatched_detections = [
            d_idx for d_idx in range(len(detections)) if d_idx not in used_detections
        ]

        return (matched_tracks, matched_detections, unmatched_detections)

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

    def _create_new_track(self, detection: DetectionObject):
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

        logger.debug(f"Created new track {track.track_id} for {detection.color}")

    def _age_tracks(self):
        for track in self.tracks:
            track.age += 1

        before_count = len(self.tracks)
        self.tracks = [track for track in self.tracks if track.age <= self.max_age]
        removed_count = before_count - len(self.tracks)

        if removed_count > 0:
            logger.debug(f"Removed {removed_count} stale tracks (age > {self.max_age})")

    def get_active_tracks(self) -> List[Track]:
        return self.tracks.copy()

    def reset(self):
        self.tracks.clear()
        self.next_id = 1
        logger.info("ObjectTracker reset")


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
