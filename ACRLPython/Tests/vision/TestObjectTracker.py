#!/usr/bin/env python3
"""
test_object_tracker.py - Unit tests for ObjectTracker

Tests the IOU-based object tracking system for persistent object IDs across frames.
"""

import sys
import os
import unittest

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.ObjectTracker import ObjectTracker, Track
from vision.DetectionDataModels import DetectionObject


class TestTrack(unittest.TestCase):
    """Test Track class"""

    def test_track_initialization(self):
        """Test Track initialization"""
        det = DetectionObject(
            object_id=1,
            color="red_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
        )

        track = Track(
            track_id=1,
            bbox=(det.bbox_x, det.bbox_y, det.bbox_w, det.bbox_h),
            color=det.color,
        )

        self.assertEqual(track.track_id, 1)
        self.assertEqual(track.bbox, (100, 100, 50, 50))
        self.assertEqual(track.color, "red_cube")
        self.assertEqual(track.age, 0)
        self.assertEqual(track.hits, 1)
        self.assertEqual(len(track.position_history), 0)

    def test_track_update(self):
        """Test track update with new detection"""
        # Create initial track
        track = Track(track_id=1, bbox=(100, 100, 50, 50), color="red_cube")

        # Create new detection
        det = DetectionObject(
            object_id=2, color="red_cube", bbox=(110, 105, 50, 50), confidence=0.95
        )

        # Update track
        track.update(det)

        self.assertEqual(track.bbox, (110, 105, 50, 50))
        self.assertEqual(track.age, 0)
        self.assertEqual(track.hits, 2)
        self.assertEqual(len(track.position_history), 1)
        self.assertEqual(track.position_history[0], (det.center_x, det.center_y))

    def test_position_history_limit(self):
        """Test position history is limited to 10 entries"""
        track = Track(track_id=1, bbox=(100, 100, 50, 50), color="red_cube")

        # Add 15 detections
        for i in range(15):
            det = DetectionObject(
                object_id=i,
                color="red_cube",
                bbox=(100 + i * 10, 100, 50, 50),
                confidence=0.9,
            )
            track.update(det)

        # Should only keep last 10
        self.assertEqual(len(track.position_history), 10)

    def test_predict_next_position(self):
        """Test velocity-based position prediction"""
        track = Track(track_id=1, bbox=(100, 100, 50, 50), color="red_cube")

        # Add two positions: (125, 125) -> (135, 130)
        det1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        det2 = DetectionObject(
            object_id=2, color="red_cube", bbox=(110, 105, 50, 50), confidence=0.9
        )

        track.update(det1)
        track.update(det2)

        # Should predict next position based on velocity
        pred = track.predict_next_position()
        self.assertIsNotNone(pred)

        # Velocity: (135-125, 130-125) = (10, 5)
        # Prediction: (135+10, 130+5) = (145, 135)
        expected_x = det2.center_x + (det2.center_x - det1.center_x)
        expected_y = det2.center_y + (det2.center_y - det1.center_y)
        self.assertEqual(pred, (expected_x, expected_y))

    def test_get_velocity(self):
        """Test velocity calculation"""
        track = Track(track_id=1, bbox=(100, 100, 50, 50), color="red_cube")

        # Insufficient history
        self.assertIsNone(track.get_velocity())

        # Add two positions
        det1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        det2 = DetectionObject(
            object_id=2, color="red_cube", bbox=(110, 105, 50, 50), confidence=0.9
        )

        track.update(det1)
        track.update(det2)

        velocity = track.get_velocity()
        self.assertIsNotNone(velocity)

        vx = det2.center_x - det1.center_x
        vy = det2.center_y - det1.center_y
        self.assertEqual(velocity, (float(vx), float(vy)))


class TestObjectTracker(unittest.TestCase):
    """Test ObjectTracker class"""

    def test_tracker_initialization(self):
        """Test tracker initialization"""
        tracker = ObjectTracker(max_age=5, min_iou=0.3)

        self.assertEqual(tracker.max_age, 5)
        self.assertEqual(tracker.min_iou, 0.3)
        self.assertEqual(len(tracker.tracks), 0)
        self.assertEqual(tracker.next_id, 1)

    def test_single_object_tracking(self):
        """Test tracking a single object across frames"""
        tracker = ObjectTracker(max_age=3, min_iou=0.3)

        # Frame 1: Object appears
        det1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        tracked1 = tracker.update([det1])

        self.assertEqual(len(tracked1), 1)
        self.assertEqual(tracked1[0].track_id, 1)
        self.assertEqual(len(tracker.tracks), 1)

        # Frame 2: Object moves slightly
        det2 = DetectionObject(
            object_id=2, color="red_cube", bbox=(105, 102, 50, 50), confidence=0.9
        )
        tracked2 = tracker.update([det2])

        self.assertEqual(len(tracked2), 1)
        self.assertEqual(tracked2[0].track_id, 1)  # Same track ID
        self.assertEqual(len(tracker.tracks), 1)

        # Frame 3: Object moves again
        det3 = DetectionObject(
            object_id=3, color="red_cube", bbox=(110, 104, 50, 50), confidence=0.9
        )
        tracked3 = tracker.update([det3])

        self.assertEqual(len(tracked3), 1)
        self.assertEqual(tracked3[0].track_id, 1)  # Still same track ID

    def test_multiple_objects_tracking(self):
        """Test tracking multiple objects simultaneously"""
        tracker = ObjectTracker(max_age=3, min_iou=0.3)

        # Frame 1: Two objects appear
        det1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        det2 = DetectionObject(
            object_id=2, color="blue_cube", bbox=(300, 200, 50, 50), confidence=0.85
        )
        tracked1 = tracker.update([det1, det2])

        self.assertEqual(len(tracked1), 2)
        self.assertEqual(tracked1[0].track_id, 1)
        self.assertEqual(tracked1[1].track_id, 2)
        self.assertEqual(len(tracker.tracks), 2)

        # Frame 2: Both objects move
        det3 = DetectionObject(
            object_id=3, color="red_cube", bbox=(105, 102, 50, 50), confidence=0.9
        )
        det4 = DetectionObject(
            object_id=4, color="blue_cube", bbox=(305, 202, 50, 50), confidence=0.85
        )
        tracked2 = tracker.update([det3, det4])

        self.assertEqual(len(tracked2), 2)
        # Should maintain track IDs
        track_ids = {t.track_id for t in tracked2}
        self.assertEqual(track_ids, {1, 2})

    def test_track_aging_and_cleanup(self):
        """Test tracks age out after max_age frames without detection"""
        tracker = ObjectTracker(max_age=3, min_iou=0.3)

        # Frame 1: Object appears
        det1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        tracker.update([det1])
        self.assertEqual(len(tracker.tracks), 1)

        # Frames 2-5: Object disappears (empty detections)
        for i in range(4):
            tracker.update([])

        # Track should be removed after max_age=3 frames
        self.assertEqual(len(tracker.tracks), 0)

    def test_new_track_creation(self):
        """Test new tracks are created for unmatched detections"""
        tracker = ObjectTracker(max_age=3, min_iou=0.3)

        # Frame 1: Red cube
        det1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        tracker.update([det1])
        self.assertEqual(tracker.next_id, 2)  # Next ID should be 2

        # Frame 2: Blue cube appears (new object)
        det2 = DetectionObject(
            object_id=2, color="blue_cube", bbox=(300, 200, 50, 50), confidence=0.85
        )
        tracked = tracker.update([det1, det2])

        self.assertEqual(len(tracker.tracks), 2)
        self.assertEqual(tracker.next_id, 3)  # Next ID should be 3

    def test_iou_matching(self):
        """Test tracks are matched using IOU threshold"""
        tracker = ObjectTracker(max_age=3, min_iou=0.5)  # Higher threshold

        # Frame 1: Object
        det1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        tracker.update([det1])

        # Frame 2: Object moves significantly (low IOU)
        det2 = DetectionObject(
            object_id=2,
            color="red_cube",
            bbox=(200, 200, 50, 50),  # Far from original
            confidence=0.9,
        )
        tracked = tracker.update([det2])

        # Should create new track (IOU too low)
        self.assertEqual(len(tracker.tracks), 2)

    def test_class_filtering(self):
        """Test tracks only match detections of same class"""
        tracker = ObjectTracker(max_age=3, min_iou=0.3)

        # Frame 1: Red cube
        det1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        tracker.update([det1])

        # Frame 2: Same location but blue cube (different class)
        det2 = DetectionObject(
            object_id=2, color="blue_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        tracked = tracker.update([det2])

        # Should create new track (different class)
        self.assertEqual(len(tracker.tracks), 2)

    def test_get_active_tracks(self):
        """Test get_active_tracks returns copy of tracks"""
        tracker = ObjectTracker(max_age=3, min_iou=0.3)

        det1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        tracker.update([det1])

        tracks = tracker.get_active_tracks()
        self.assertEqual(len(tracks), 1)

        # Should be a copy (modifying shouldn't affect tracker)
        tracks.clear()
        self.assertEqual(len(tracker.get_active_tracks()), 1)

    def test_reset(self):
        """Test tracker reset clears all tracks"""
        tracker = ObjectTracker(max_age=3, min_iou=0.3)

        # Add some tracks
        det1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        det2 = DetectionObject(
            object_id=2, color="blue_cube", bbox=(300, 200, 50, 50), confidence=0.85
        )
        tracker.update([det1, det2])

        self.assertEqual(len(tracker.tracks), 2)
        self.assertEqual(tracker.next_id, 3)

        # Reset
        tracker.reset()

        self.assertEqual(len(tracker.tracks), 0)
        self.assertEqual(tracker.next_id, 1)

    def test_calculate_iou(self):
        """Test IOU calculation accuracy"""
        tracker = ObjectTracker()

        # Identical boxes: IOU = 1.0
        bbox1 = (100, 100, 50, 50)
        bbox2 = (100, 100, 50, 50)
        iou = tracker._calculate_iou(bbox1, bbox2)
        self.assertAlmostEqual(iou, 1.0)

        # No overlap: IOU = 0.0
        bbox1 = (100, 100, 50, 50)
        bbox2 = (200, 200, 50, 50)
        iou = tracker._calculate_iou(bbox1, bbox2)
        self.assertAlmostEqual(iou, 0.0)

        # Partial overlap
        bbox1 = (100, 100, 50, 50)
        bbox2 = (125, 100, 50, 50)  # 25x50 overlap (only X shifted)
        iou = tracker._calculate_iou(bbox1, bbox2)

        # Intersection: 25*50 = 1250
        # Union: 50*50 + 50*50 - 1250 = 3750
        # IOU: 1250/3750 = 0.333...
        self.assertAlmostEqual(iou, 1250.0 / 3750.0, places=3)


class TestObjectTrackerIntegration(unittest.TestCase):
    """Integration tests for ObjectTracker"""

    def test_realistic_tracking_scenario(self):
        """Test realistic scenario with moving objects"""
        tracker = ObjectTracker(max_age=3, min_iou=0.3)

        # Simulate 10 frames with moving objects
        for frame_idx in range(10):
            detections = []

            # Object 1: Moving slowly left to right
            if frame_idx < 8:
                x = 100 + frame_idx * 10  # Slower movement (10px/frame)
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
            tracked = tracker.update(detections)

            # Verify tracking continuity
            if frame_idx == 0:
                self.assertEqual(len(tracked), 1)  # Only object 1
                self.assertEqual(tracked[0].track_id, 1)

            if frame_idx == 3:
                self.assertEqual(len(tracked), 2)  # Both objects
                # Should have track IDs 1 and 2
                track_ids = {t.track_id for t in tracked}
                self.assertEqual(track_ids, {1, 2})

            if frame_idx == 9:
                # Object 1 disappeared at frame 8
                self.assertEqual(len(tracked), 1)  # Only object 2


if __name__ == "__main__":
    unittest.main()
