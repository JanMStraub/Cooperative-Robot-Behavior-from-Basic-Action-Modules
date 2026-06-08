import pytest
import numpy as np

from vision.DepthEstimator import estimate_depth_from_bbox


class TestBboxDepthEstimation:

    def test_median_inner_50pct_strategy(self):
        disparity_map = np.zeros((100, 100), dtype=np.float32)

        disparity_map[30:50, 30:50] = 50.0

        disparity_map[20:30, 20:60] = 10.0
        disparity_map[50:60, 20:60] = 10.0
        disparity_map[20:60, 20:30] = 10.0
        disparity_map[20:60, 50:60] = 10.0

        focal_length_px = 800.0
        baseline = 0.05

        result = estimate_depth_from_bbox(
            disparity_map,
            bbox=(20, 20, 40, 40),
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="median_inner_50pct",
            min_disparity_threshold=5.0,
            inner_percent=50,
        )

        assert result is not None
        depth_m, median_disparity, num_valid = result

        assert abs(median_disparity - 50.0) < 10**-1

        expected_depth = (focal_length_px * baseline) / 50.0
        assert abs(depth_m - expected_depth) < 10**-2

        assert num_valid > 300

    def test_mean_valid_strategy(self):
        disparity_map = np.zeros((100, 100), dtype=np.float32)

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

        assert result is not None
        depth_m, mean_disparity, num_valid = result

        assert mean_disparity > 45.0
        assert mean_disparity < 55.0

    def test_max_disparity_strategy(self):
        disparity_map = np.zeros((100, 100), dtype=np.float32)

        disparity_map[30:50, 30:50] = 40.0
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

        assert result is not None
        depth_m, max_disparity, num_valid = result

        assert abs(max_disparity - 60.0) < 10**-1

    def test_inner_percent_parameter(self):
        disparity_map = np.zeros((100, 100), dtype=np.float32)
        disparity_map[20:60, 20:60] = 50.0

        focal_length_px = 800.0
        baseline = 0.05

        result_80 = estimate_depth_from_bbox(
            disparity_map,
            bbox=(20, 20, 40, 40),
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="median_inner_50pct",
            inner_percent=80,
        )

        result_20 = estimate_depth_from_bbox(
            disparity_map,
            bbox=(20, 20, 40, 40),
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="median_inner_50pct",
            inner_percent=20,
        )

        assert result_80 is not None
        assert result_20 is not None

        assert result_80[2] > result_20[2]

    def test_min_disparity_threshold(self):
        disparity_map = np.ones((100, 100), dtype=np.float32) * 3.0

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

        assert result is None

    def test_max_depth_threshold(self):
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
            max_depth_threshold=10.0,
        )

        assert result is None

    def test_bbox_at_image_edge(self):
        disparity_map = np.ones((100, 100), dtype=np.float32) * 50.0

        focal_length_px = 800.0
        baseline = 0.05

        result_corner = estimate_depth_from_bbox(
            disparity_map,
            bbox=(0, 0, 40, 40),
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="median_inner_50pct",
        )

        assert result_corner is not None

    def test_small_bbox(self):
        disparity_map = np.ones((100, 100), dtype=np.float32) * 50.0

        focal_length_px = 800.0
        baseline = 0.05

        result = estimate_depth_from_bbox(
            disparity_map,
            bbox=(40, 40, 10, 10),
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="median_inner_50pct",
            inner_percent=50,
        )

        assert result is not None

    def test_nan_disparity_handling(self):
        disparity_map = np.ones((100, 100), dtype=np.float32) * 50.0

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

        assert result is not None
        depth_m, median_disparity, num_valid = result

        assert not np.isnan(median_disparity)
        assert not np.isnan(depth_m)

    def test_empty_roi(self):
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

        assert result is None


class TestBboxDepthAccuracy:

    def test_noisy_disparity_robustness(self):
        np.random.seed(42)
        disparity_map = np.ones((100, 100), dtype=np.float32) * 50.0

        noise = np.random.normal(0, 5.0, (100, 100))
        disparity_map += noise

        disparity_map[40, 40] = 100.0

        focal_length_px = 800.0
        baseline = 0.05

        result_bbox = estimate_depth_from_bbox(
            disparity_map,
            bbox=(20, 20, 40, 40),
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="median_inner_50pct",
        )

        center_x, center_y = 40, 40
        single_point_disparity = disparity_map[center_y, center_x]
        single_point_depth = (focal_length_px * baseline) / single_point_disparity

        assert result_bbox is not None
        bbox_depth, bbox_disparity, _ = result_bbox

        true_disparity = 50.0
        true_depth = (focal_length_px * baseline) / true_disparity

        bbox_error = abs(bbox_depth - true_depth)
        single_point_error = abs(single_point_depth - true_depth)

        assert bbox_error < single_point_error

    def test_edge_effect_mitigation(self):
        disparity_map = np.ones((100, 100), dtype=np.float32) * 50.0

        bbox = (20, 20, 40, 40)
        x, y, w, h = bbox

        disparity_map[y : y + 5, x : x + w] = 5.0
        disparity_map[y + h - 5 : y + h, x : x + w] = 5.0
        disparity_map[y : y + h, x : x + 5] = 5.0
        disparity_map[y : y + h, x + w - 5 : x + w] = 5.0

        focal_length_px = 800.0
        baseline = 0.05

        result = estimate_depth_from_bbox(
            disparity_map,
            bbox=bbox,
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="median_inner_50pct",
            inner_percent=50,
        )

        assert result is not None
        depth_m, median_disparity, num_valid = result

        assert abs(median_disparity - 50.0) < 10**-1


class TestBboxDepthRealWorld:

    def test_typical_stereo_disparity(self):
        focal_length_px = 800.0
        baseline = 0.05
        true_depth = 0.8

        expected_disparity = (focal_length_px * baseline) / true_depth

        np.random.seed(42)
        disparity_map = np.zeros((960, 1280), dtype=np.float32)

        bbox = (500, 400, 100, 80)
        x, y, w, h = bbox

        disparity_map[y : y + h, x : x + w] = expected_disparity + np.random.normal(
            0, 2.0, (h, w)
        )

        result = estimate_depth_from_bbox(
            disparity_map,
            bbox=bbox,
            focal_length_px=focal_length_px,
            baseline=baseline,
            strategy="median_inner_50pct",
        )

        assert result is not None
        estimated_depth, estimated_disparity, num_valid = result

        depth_error = abs(estimated_depth - true_depth)
        depth_error_percent = (depth_error / true_depth) * 100

        assert depth_error_percent < 10.0

    def test_close_range_object(self):
        focal_length_px = 800.0
        baseline = 0.05
        true_depth = 0.5

        expected_disparity = (focal_length_px * baseline) / true_depth

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

        assert result is not None
        estimated_depth, _, _ = result

        depth_error_percent = (abs(estimated_depth - true_depth) / true_depth) * 100

        assert depth_error_percent < 10.0
