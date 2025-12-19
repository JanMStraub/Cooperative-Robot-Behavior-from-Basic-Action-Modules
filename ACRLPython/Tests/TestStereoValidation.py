#!/usr/bin/env python3
"""
test_stereo_validation.py - Unit tests for stereo L/R validation

Tests detection matching between left and right stereo images to reduce false positives.
"""

import sys
import os
import unittest
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.YOLODetector import YOLODetector
from vision.DetectionDataModels import DetectionObject


class TestStereoValidation(unittest.TestCase):
    """Test stereo L/R detection matching"""

    def setUp(self):
        """Create YOLODetector instance for testing"""
        # Note: We'll test the matching functions directly without loading YOLO model
        self.detector = YOLODetector.__new__(
            YOLODetector
        )  # Create without __init__

    def test_calculate_iou_identical(self):
        """Test IOU calculation for identical bboxes"""
        bbox1 = (100, 100, 50, 50)
        bbox2 = (100, 100, 50, 50)

        iou = self.detector._calculate_iou(bbox1, bbox2)

        self.assertAlmostEqual(iou, 1.0)

    def test_calculate_iou_no_overlap(self):
        """Test IOU for non-overlapping bboxes"""
        bbox1 = (100, 100, 50, 50)
        bbox2 = (200, 200, 50, 50)

        iou = self.detector._calculate_iou(bbox1, bbox2)

        self.assertAlmostEqual(iou, 0.0)

    def test_calculate_iou_partial_overlap(self):
        """Test IOU for partially overlapping bboxes"""
        bbox1 = (100, 100, 50, 50)
        bbox2 = (125, 100, 50, 50)  # 25x50 overlap (only X shifted)

        iou = self.detector._calculate_iou(bbox1, bbox2)

        # Intersection: 25*50 = 1250
        # Union: 50*50 + 50*50 - 1250 = 3750
        # IOU: 1250/3750 = 0.333...
        expected_iou = 1250.0 / 3750.0
        self.assertAlmostEqual(iou, expected_iou, places=3)

    def test_calculate_iou_contained(self):
        """Test IOU when one bbox contains another"""
        bbox1 = (100, 100, 100, 100)  # Large
        bbox2 = (120, 120, 40, 40)  # Small, inside large

        iou = self.detector._calculate_iou(bbox1, bbox2)

        # Intersection: 40*40 = 1600
        # Union: 100*100 = 10000
        # IOU: 1600/10000 = 0.16
        expected_iou = 1600.0 / 10000.0
        self.assertAlmostEqual(iou, expected_iou, places=3)

    def test_match_stereo_detections_perfect_match(self):
        """Test matching with perfect stereo pair"""
        # Create left detection
        det_left = DetectionObject(
            object_id=1,
            color="red_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
        )

        # Create right detection (shifted left due to disparity)
        det_right = DetectionObject(
            object_id=2,
            color="red_cube",
            bbox=(80, 100, 50, 50),  # 20 pixels left (positive disparity)
            confidence=0.9,
        )

        matches = self.detector._match_stereo_detections(
            [det_left], [det_right], max_y_diff=10, max_size_ratio=0.3, min_iou=0.0
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0], det_left)
        self.assertEqual(matches[0][1], det_right)

    def test_match_stereo_detections_different_class(self):
        """Test matching fails for different classes"""
        det_left = DetectionObject(
            object_id=1,
            color="red_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
        )

        det_right = DetectionObject(
            object_id=2,
            color="blue_cube",  # Different class
            bbox=(80, 100, 50, 50),
            confidence=0.9,
        )

        matches = self.detector._match_stereo_detections(
            [det_left], [det_right], max_y_diff=10, max_size_ratio=0.3
        )

        # Should not match (different class)
        self.assertEqual(len(matches), 0)

    def test_match_stereo_detections_y_diff_threshold(self):
        """Test Y coordinate difference threshold"""
        det_left = DetectionObject(
            object_id=1,
            color="red_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
        )

        # Right detection with large Y difference
        det_right = DetectionObject(
            object_id=2,
            color="red_cube",
            bbox=(80, 120, 50, 50),  # Y=120 (diff=20)
            confidence=0.9,
        )

        # Should not match (Y diff > max_y_diff=10)
        matches = self.detector._match_stereo_detections(
            [det_left], [det_right], max_y_diff=10, max_size_ratio=0.3
        )

        self.assertEqual(len(matches), 0)

        # Should match with higher threshold
        matches = self.detector._match_stereo_detections(
            [det_left], [det_right], max_y_diff=25, max_size_ratio=0.3
        )

        self.assertEqual(len(matches), 1)

    def test_match_stereo_detections_size_ratio_threshold(self):
        """Test bbox size difference threshold"""
        det_left = DetectionObject(
            object_id=1,
            color="red_cube",
            bbox=(100, 100, 50, 50),  # Area = 2500, center at (125, 125)
            confidence=0.9,
        )

        # Right detection significantly larger, but Y-aligned
        det_right = DetectionObject(
            object_id=2,
            color="red_cube",
            bbox=(50, 105, 80, 40),  # Area = 3200, center at (90, 125)
            confidence=0.9,
        )

        # Size ratio: |2500 - 3200| / 3200 = 0.21875 (within 0.3)
        # Y diff: 0 (both centers at Y=125)
        # X: right center (90) < left center (125) - positive disparity OK
        # Should match with max_size_ratio=0.3
        matches = self.detector._match_stereo_detections(
            [det_left], [det_right], max_y_diff=10, max_size_ratio=0.3
        )

        self.assertEqual(len(matches), 1)

        # Test with very restrictive size ratio
        det_right_large = DetectionObject(
            object_id=3,
            color="red_cube",
            bbox=(40, 105, 100, 40),  # Area = 4000, center at (90, 125)
            confidence=0.9,
        )

        # Size ratio: |2500 - 4000| / 4000 = 0.375 (exceeds 0.3)
        # Should NOT match with max_size_ratio=0.3
        matches = self.detector._match_stereo_detections(
            [det_left], [det_right_large], max_y_diff=10, max_size_ratio=0.3
        )

        self.assertEqual(len(matches), 0)

    def test_match_stereo_detections_positive_disparity(self):
        """Test that right detection must be LEFT of left detection"""
        det_left = DetectionObject(
            object_id=1,
            color="red_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
        )

        # Right detection to the RIGHT (negative disparity - invalid)
        det_right = DetectionObject(
            object_id=2,
            color="red_cube",
            bbox=(120, 100, 50, 50),  # center_x > left.center_x
            confidence=0.9,
        )

        matches = self.detector._match_stereo_detections(
            [det_left], [det_right], max_y_diff=10, max_size_ratio=0.3
        )

        # Should not match (negative disparity)
        self.assertEqual(len(matches), 0)

    def test_match_stereo_detections_iou_threshold(self):
        """Test optional IOU threshold for matching"""
        det_left = DetectionObject(
            object_id=1,
            color="red_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
        )

        # Right detection with low overlap
        det_right = DetectionObject(
            object_id=2,
            color="red_cube",
            bbox=(70, 100, 50, 50),  # Small overlap
            confidence=0.9,
        )

        # Calculate actual IOU
        iou = self.detector._calculate_iou(
            (det_left.bbox_x, det_left.bbox_y, det_left.bbox_w, det_left.bbox_h),
            (det_right.bbox_x, det_right.bbox_y, det_right.bbox_w, det_right.bbox_h),
        )

        # Should match without IOU threshold
        matches = self.detector._match_stereo_detections(
            [det_left], [det_right], max_y_diff=10, max_size_ratio=0.3, min_iou=0.0
        )
        self.assertEqual(len(matches), 1)

        # Should not match with high IOU threshold
        matches = self.detector._match_stereo_detections(
            [det_left], [det_right], max_y_diff=10, max_size_ratio=0.3, min_iou=0.5
        )
        self.assertEqual(len(matches), 0)

    def test_match_stereo_detections_multiple_objects(self):
        """Test matching multiple objects in stereo pair"""
        # Left detections
        det_left_1 = DetectionObject(
            object_id=1,
            color="red_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
        )
        det_left_2 = DetectionObject(
            object_id=2,
            color="blue_cube",
            bbox=(300, 200, 50, 50),
            confidence=0.85,
        )

        # Right detections (with disparity)
        det_right_1 = DetectionObject(
            object_id=3,
            color="red_cube",
            bbox=(80, 100, 50, 50),  # Matches left_1
            confidence=0.9,
        )
        det_right_2 = DetectionObject(
            object_id=4,
            color="blue_cube",
            bbox=(275, 200, 50, 50),  # Matches left_2
            confidence=0.85,
        )

        matches = self.detector._match_stereo_detections(
            [det_left_1, det_left_2],
            [det_right_1, det_right_2],
            max_y_diff=10,
            max_size_ratio=0.3,
        )

        # Should match both pairs
        self.assertEqual(len(matches), 2)

        # Verify correct matching
        matched_colors = {(m[0].color, m[1].color) for m in matches}
        self.assertIn(("red_cube", "red_cube"), matched_colors)
        self.assertIn(("blue_cube", "blue_cube"), matched_colors)

    def test_match_stereo_detections_one_sided(self):
        """Test when detection appears in only one image"""
        det_left = DetectionObject(
            object_id=1,
            color="red_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
        )

        # No right detections (false positive in left)
        matches = self.detector._match_stereo_detections(
            [det_left], [], max_y_diff=10, max_size_ratio=0.3
        )

        # Should have no matches
        self.assertEqual(len(matches), 0)

    def test_match_stereo_detections_ambiguous_matching(self):
        """Test greedy matching with multiple candidates"""
        det_left = DetectionObject(
            object_id=1,
            color="red_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
        )

        # Two right detections that could match
        det_right_1 = DetectionObject(
            object_id=2,
            color="red_cube",
            bbox=(80, 100, 50, 50),  # Closer Y match
            confidence=0.9,
        )
        det_right_2 = DetectionObject(
            object_id=3,
            color="red_cube",
            bbox=(85, 105, 50, 50),  # Different Y
            confidence=0.9,
        )

        matches = self.detector._match_stereo_detections(
            [det_left],
            [det_right_1, det_right_2],
            max_y_diff=10,
            max_size_ratio=0.3,
        )

        # Should match only once (greedy: best match by Y distance)
        self.assertEqual(len(matches), 1)
        # Should match with closer Y (det_right_1)
        self.assertEqual(matches[0][1], det_right_1)

    def test_match_stereo_detections_no_duplicate_matches(self):
        """Test that each detection is matched at most once"""
        # Two left detections
        det_left_1 = DetectionObject(
            object_id=1,
            color="red_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
        )
        det_left_2 = DetectionObject(
            object_id=2,
            color="red_cube",
            bbox=(105, 102, 50, 50),
            confidence=0.9,
        )

        # One right detection (could match both)
        det_right = DetectionObject(
            object_id=3,
            color="red_cube",
            bbox=(80, 100, 50, 50),
            confidence=0.9,
        )

        matches = self.detector._match_stereo_detections(
            [det_left_1, det_left_2],
            [det_right],
            max_y_diff=10,
            max_size_ratio=0.3,
        )

        # Should match only once (to best match)
        self.assertEqual(len(matches), 1)


class TestStereoValidationIntegration(unittest.TestCase):
    """Integration tests for stereo validation"""

    def setUp(self):
        """Create YOLODetector instance"""
        self.detector = YOLODetector.__new__(YOLODetector)

    def test_false_positive_reduction(self):
        """Test that stereo validation reduces false positives"""
        # Simulate scenario: Left image has false positive, right doesn't

        # Left: 2 detections (1 real, 1 false positive)
        det_left_real = DetectionObject(
            object_id=1,
            color="red_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
        )
        det_left_false = DetectionObject(
            object_id=2,
            color="blue_cube",
            bbox=(300, 200, 50, 50),
            confidence=0.7,  # Lower confidence
        )

        # Right: Only 1 detection (matching the real one)
        det_right_real = DetectionObject(
            object_id=3,
            color="red_cube",
            bbox=(80, 100, 50, 50),
            confidence=0.9,
        )

        # Match
        matches = self.detector._match_stereo_detections(
            [det_left_real, det_left_false],
            [det_right_real],
            max_y_diff=10,
            max_size_ratio=0.3,
        )

        # Should only match the real detection
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0], det_left_real)

        # False positive reduction: 50% (1 out of 2 left detections filtered)
        reduction = (1 - len(matches) / 2) * 100
        self.assertEqual(reduction, 50.0)

    def test_typical_stereo_scene(self):
        """Test realistic stereo scene with multiple objects"""
        # Create typical scene: 3 objects at different depths

        # Object 1: Close (high disparity)
        det_left_1 = DetectionObject(
            object_id=1,
            color="red_cube",
            bbox=(200, 150, 60, 60),
            confidence=0.95,
        )
        det_right_1 = DetectionObject(
            object_id=11,
            color="red_cube",
            bbox=(160, 150, 60, 60),  # 40px disparity
            confidence=0.95,
        )

        # Object 2: Medium distance (medium disparity)
        det_left_2 = DetectionObject(
            object_id=2,
            color="blue_cube",
            bbox=(500, 300, 50, 50),
            confidence=0.88,
        )
        det_right_2 = DetectionObject(
            object_id=12,
            color="blue_cube",
            bbox=(475, 300, 50, 50),  # 25px disparity
            confidence=0.88,
        )

        # Object 3: Far (low disparity)
        det_left_3 = DetectionObject(
            object_id=3,
            color="green_cube",
            bbox=(800, 400, 45, 45),
            confidence=0.82,
        )
        det_right_3 = DetectionObject(
            object_id=13,
            color="green_cube",
            bbox=(790, 400, 45, 45),  # 10px disparity
            confidence=0.82,
        )

        # Add false positive in left only
        det_left_false = DetectionObject(
            object_id=4,
            color="yellow_cube",
            bbox=(1000, 500, 40, 40),
            confidence=0.65,
        )

        matches = self.detector._match_stereo_detections(
            [det_left_1, det_left_2, det_left_3, det_left_false],
            [det_right_1, det_right_2, det_right_3],
            max_y_diff=10,
            max_size_ratio=0.3,
        )

        # Should match 3 real objects, filter false positive
        self.assertEqual(len(matches), 3)

        # Verify all real objects matched
        matched_colors = {m[0].color for m in matches}
        self.assertEqual(matched_colors, {"red_cube", "blue_cube", "green_cube"})


if __name__ == "__main__":
    unittest.main()
