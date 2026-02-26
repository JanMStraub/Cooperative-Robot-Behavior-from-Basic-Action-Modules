#!/usr/bin/env python3
"""
test_bbox_depth.py - Unit tests for bbox-guided depth estimation

Tests ROI-based depth sampling for improved accuracy over single-point sampling.
"""

import sys
import os
import unittest
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.DepthEstimator import estimate_depth_from_bbox


class TestBboxDepthEstimation(unittest.TestCase):
    """Test bbox-guided depth estimation"""

    def test_median_inner_50pct_strategy(self):
        """Test median sampling of inner 50% of bbox"""
        # Create synthetic disparity map (100x100)
        disparity_map = np.zeros((100, 100), dtype=np.float32)

        # Create bbox area with known disparity
        # Bbox: (20, 20, 40, 40)
        # Inner 50%: (30, 30, 20, 20) - center 20x20 region
        disparity_map[30:50, 30:50] = 50.0  # Good disparity in center

        # Add noise to edges (should be ignored)
        disparity_map[20:30, 20:60] = 10.0
        disparity_map[50:60, 20:60] = 10.0
        disparity_map[20:60, 20:30] = 10.0
        disparity_map[20:60, 50:60] = 10.0

        # Test with median strategy
        focal_length_px = 800.0
        baseline = 0.05  # 5cm

        result = estimate_depth_from_bbox(
            disparity_map,
            bbox=(20, 20, 40, 40),
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="median_inner_50pct",
            min_disparity_threshold=5.0,
            inner_percent=50,
        )

        self.assertIsNotNone(result)
        assert result is not None  # Type guard for Pylance
        depth_m, median_disparity, num_valid = result

        # Should use median of center region (50.0)
        self.assertAlmostEqual(median_disparity, 50.0, places=1)

        # Depth = (f * b) / d = (800 * 0.05) / 50 = 0.8m
        expected_depth = (focal_length_px * baseline) / 50.0
        self.assertAlmostEqual(depth_m, expected_depth, places=2)

        # Number of valid pixels should be ~400 (20x20 inner region)
        self.assertGreater(num_valid, 300)

    def test_mean_valid_strategy(self):
        """Test mean sampling strategy"""
        # Create disparity map
        disparity_map = np.zeros((100, 100), dtype=np.float32)

        # Inner region with varying disparities
        disparity_map[30:50, 30:50] = np.random.uniform(45.0, 55.0, (20, 20))

        focal_length_px = 800.0
        baseline = 0.05

        result = estimate_depth_from_bbox(
            disparity_map,
            bbox=(20, 20, 40, 40),
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="mean_valid",
            inner_percent=50,
        )

        self.assertIsNotNone(result)
        assert result is not None  # Type guard for Pylance
        depth_m, mean_disparity, num_valid = result

        # Mean should be around 50.0
        self.assertGreater(mean_disparity, 45.0)
        self.assertLess(mean_disparity, 55.0)

    def test_max_disparity_strategy(self):
        """Test max disparity (closest point) strategy"""
        # Create disparity map
        disparity_map = np.zeros((100, 100), dtype=np.float32)

        # Inner region with varying disparities
        disparity_map[30:50, 30:50] = 40.0
        # Add one high disparity pixel (closest point)
        disparity_map[40, 40] = 60.0

        focal_length_px = 800.0
        baseline = 0.05

        result = estimate_depth_from_bbox(
            disparity_map,
            bbox=(20, 20, 40, 40),
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="max_disparity",
            inner_percent=50,
        )

        self.assertIsNotNone(result)
        assert result is not None  # Type guard for Pylance
        depth_m, max_disparity, num_valid = result

        # Should pick max disparity (closest)
        self.assertAlmostEqual(max_disparity, 60.0, places=1)

    def test_inner_percent_parameter(self):
        """Test different inner_percent values"""
        disparity_map = np.zeros((100, 100), dtype=np.float32)
        disparity_map[20:60, 20:60] = 50.0

        focal_length_px = 800.0
        baseline = 0.05

        # Inner 80% should sample larger region
        result_80 = estimate_depth_from_bbox(
            disparity_map,
            bbox=(20, 20, 40, 40),
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="median_inner_50pct",
            inner_percent=80,
        )

        # Inner 20% should sample smaller region
        result_20 = estimate_depth_from_bbox(
            disparity_map,
            bbox=(20, 20, 40, 40),
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="median_inner_50pct",
            inner_percent=20,
        )

        self.assertIsNotNone(result_80)
        self.assertIsNotNone(result_20)
        assert result_80 is not None  # Type guard for Pylance
        assert result_20 is not None  # Type guard for Pylance

        # 80% should have more valid pixels
        self.assertGreater(result_80[2], result_20[2])

    def test_min_disparity_threshold(self):
        """Test minimum disparity threshold filtering"""
        # Create disparity map with low disparities
        disparity_map = np.ones((100, 100), dtype=np.float32) * 3.0  # Below threshold

        focal_length_px = 800.0
        baseline = 0.05

        result = estimate_depth_from_bbox(
            disparity_map,
            bbox=(20, 20, 40, 40),
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="median_inner_50pct",
            min_disparity_threshold=5.0,  # Higher than disparity values
        )

        # Should return None (no valid disparities)
        self.assertIsNone(result)

    def test_max_depth_threshold(self):
        """Test maximum depth threshold filtering"""
        # Create disparity map with low disparities (far objects)
        disparity_map = np.ones((100, 100), dtype=np.float32) * 2.0

        focal_length_px = 800.0
        baseline = 0.05

        result = estimate_depth_from_bbox(
            disparity_map,
            bbox=(20, 20, 40, 40),
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="median_inner_50pct",
            min_disparity_threshold=1.0,
            max_depth_threshold=10.0,  # Depth would be (800*0.05)/2 = 20m
        )

        # Should return None (depth exceeds threshold)
        self.assertIsNone(result)

    def test_bbox_at_image_edge(self):
        """Test bbox sampling at image edges"""
        disparity_map = np.ones((100, 100), dtype=np.float32) * 50.0

        focal_length_px = 800.0
        baseline = 0.05

        # Bbox at top-left corner
        result_corner = estimate_depth_from_bbox(
            disparity_map,
            bbox=(0, 0, 40, 40),
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="median_inner_50pct",
        )

        # Should handle edge case gracefully
        self.assertIsNotNone(result_corner)

    def test_small_bbox(self):
        """Test with very small bbox (edge case)"""
        disparity_map = np.ones((100, 100), dtype=np.float32) * 50.0

        focal_length_px = 800.0
        baseline = 0.05

        # Very small bbox (10x10)
        result = estimate_depth_from_bbox(
            disparity_map,
            bbox=(40, 40, 10, 10),
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="median_inner_50pct",
            inner_percent=50,
        )

        # Should still work
        self.assertIsNotNone(result)

    def test_nan_disparity_handling(self):
        """Test handling of NaN values in disparity map"""
        disparity_map = np.ones((100, 100), dtype=np.float32) * 50.0

        # Add some NaN values
        disparity_map[30:35, 30:35] = np.nan

        focal_length_px = 800.0
        baseline = 0.05

        result = estimate_depth_from_bbox(
            disparity_map,
            bbox=(20, 20, 40, 40),
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="median_inner_50pct",
        )

        # Should filter out NaN values
        self.assertIsNotNone(result)
        assert result is not None  # Type guard for Pylance
        depth_m, median_disparity, num_valid = result

        # Should have valid result despite NaN values
        self.assertFalse(np.isnan(median_disparity))
        self.assertFalse(np.isnan(depth_m))

    def test_empty_roi(self):
        """Test when ROI has no valid disparities"""
        # All zeros (below min threshold)
        disparity_map = np.zeros((100, 100), dtype=np.float32)

        focal_length_px = 800.0
        baseline = 0.05

        result = estimate_depth_from_bbox(
            disparity_map,
            bbox=(20, 20, 40, 40),
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="median_inner_50pct",
            min_disparity_threshold=5.0,
        )

        # Should return None
        self.assertIsNone(result)


class TestBboxDepthAccuracy(unittest.TestCase):
    """Test accuracy improvements vs single-point sampling"""

    def test_noisy_disparity_robustness(self):
        """Test bbox sampling is more robust to noise than single-point"""
        # Create disparity map with noise
        np.random.seed(42)
        disparity_map = np.ones((100, 100), dtype=np.float32) * 50.0

        # Add gaussian noise
        noise = np.random.normal(0, 5.0, (100, 100))
        disparity_map += noise

        # Add outlier at center (simulating single-point error)
        disparity_map[40, 40] = 100.0  # Bad value at center

        focal_length_px = 800.0
        baseline = 0.05

        # Bbox sampling (should ignore outlier via median)
        result_bbox = estimate_depth_from_bbox(
            disparity_map,
            bbox=(20, 20, 40, 40),
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="median_inner_50pct",
        )

        # Single-point sampling (center of bbox)
        center_x, center_y = 40, 40
        single_point_disparity = disparity_map[center_y, center_x]
        single_point_depth = (focal_length_px * baseline) / single_point_disparity

        self.assertIsNotNone(result_bbox)
        assert result_bbox is not None  # Type guard for Pylance
        bbox_depth, bbox_disparity, _ = result_bbox

        # Bbox sampling should be closer to true value (50.0 disparity)
        true_disparity = 50.0
        true_depth = (focal_length_px * baseline) / true_disparity

        bbox_error = abs(bbox_depth - true_depth)
        single_point_error = abs(single_point_depth - true_depth)

        # Bbox should have lower error (more robust to outlier)
        self.assertLess(bbox_error, single_point_error)

    def test_edge_effect_mitigation(self):
        """Test bbox sampling avoids edge effects"""
        # Create disparity map with bad edges (common in SGBM)
        disparity_map = np.ones((100, 100), dtype=np.float32) * 50.0

        # Bad disparities at edges of object
        bbox = (20, 20, 40, 40)
        x, y, w, h = bbox

        # Outer edge has bad values
        disparity_map[y : y + 5, x : x + w] = 5.0  # Top edge
        disparity_map[y + h - 5 : y + h, x : x + w] = 5.0  # Bottom edge
        disparity_map[y : y + h, x : x + 5] = 5.0  # Left edge
        disparity_map[y : y + h, x + w - 5 : x + w] = 5.0  # Right edge

        focal_length_px = 800.0
        baseline = 0.05

        # Sample inner 50% (should avoid bad edges)
        result = estimate_depth_from_bbox(
            disparity_map,
            bbox=bbox,
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="median_inner_50pct",
            inner_percent=50,
        )

        self.assertIsNotNone(result)
        assert result is not None  # Type guard for Pylance
        depth_m, median_disparity, num_valid = result

        # Should get good disparity from center (50.0)
        self.assertAlmostEqual(median_disparity, 50.0, places=1)


class TestBboxDepthRealWorld(unittest.TestCase):
    """Real-world scenario tests"""

    def test_typical_stereo_disparity(self):
        """Test with typical stereo camera setup"""
        # Simulate typical SGBM output for object at 0.8m
        # Camera: 5cm baseline, 60° FOV, 1280x960
        # Object at 0.8m: disparity ~= (focal * baseline) / depth

        focal_length_px = 800.0
        baseline = 0.05  # 5cm
        true_depth = 0.8  # meters

        # Expected disparity
        expected_disparity = (focal_length_px * baseline) / true_depth  # 50 pixels

        # Create disparity map with realistic noise
        np.random.seed(42)
        disparity_map = np.zeros((960, 1280), dtype=np.float32)

        # Object bbox area (YOLO detection)
        bbox = (500, 400, 100, 80)
        x, y, w, h = bbox

        # Fill bbox with expected disparity + noise
        disparity_map[y : y + h, x : x + w] = expected_disparity + np.random.normal(
            0, 2.0, (h, w)
        )

        # Estimate depth
        result = estimate_depth_from_bbox(
            disparity_map,
            bbox=bbox,
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="median_inner_50pct",
        )

        self.assertIsNotNone(result)
        assert result is not None  # Type guard for Pylance
        estimated_depth, estimated_disparity, num_valid = result

        # Should be close to true depth
        depth_error = abs(estimated_depth - true_depth)
        depth_error_percent = (depth_error / true_depth) * 100

        # Should have <10% error (goal from plan)
        self.assertLess(depth_error_percent, 10.0)

    def test_close_range_object(self):
        """Test close-range object (<1m) - key requirement"""
        # Object at 0.5m (close range)
        focal_length_px = 800.0
        baseline = 0.05
        true_depth = 0.5

        expected_disparity = (focal_length_px * baseline) / true_depth  # 80 pixels

        # Create disparity map
        np.random.seed(42)
        disparity_map = np.zeros((960, 1280), dtype=np.float32)
        bbox = (500, 400, 120, 100)
        x, y, w, h = bbox

        disparity_map[y : y + h, x : x + w] = expected_disparity + np.random.normal(
            0, 3.0, (h, w)
        )

        result = estimate_depth_from_bbox(
            disparity_map,
            bbox=bbox,
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="median_inner_50pct",
        )

        self.assertIsNotNone(result)
        assert result is not None  # Type guard for Pylance
        estimated_depth, _, _ = result

        depth_error_percent = (abs(estimated_depth - true_depth) / true_depth) * 100

        # Close range goal: 5-10% error (vs 15-20% with single-point)
        self.assertLess(depth_error_percent, 10.0)


if __name__ == "__main__":
    unittest.main()
