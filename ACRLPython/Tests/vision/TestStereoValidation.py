from vision.YOLODetector import YOLODetector
from vision.DetectionDataModels import DetectionObject


class TestStereoValidation:

    def setup_method(self):
        self.detector = YOLODetector.__new__(YOLODetector)

    def test_calculate_iou_identical(self):
        iou = self.detector._calculate_iou((100, 100, 50, 50), (100, 100, 50, 50))
        assert abs(iou - 1.0) < 1e-7

    def test_calculate_iou_no_overlap(self):
        iou = self.detector._calculate_iou((100, 100, 50, 50), (200, 200, 50, 50))
        assert abs(iou - 0.0) < 1e-7

    def test_calculate_iou_partial_overlap(self):
        # Intersection: 25*50=1250, Union: 50*50+50*50-1250=3750, IOU=1250/3750
        iou = self.detector._calculate_iou((100, 100, 50, 50), (125, 100, 50, 50))
        assert abs(iou - 1250.0 / 3750.0) < 1e-3

    def test_calculate_iou_contained(self):
        # Intersection: 40*40=1600, Union: 100*100=10000, IOU=0.16
        iou = self.detector._calculate_iou((100, 100, 100, 100), (120, 120, 40, 40))
        assert abs(iou - 1600.0 / 10000.0) < 1e-3

    def test_match_stereo_detections_perfect_match(self):
        det_left = DetectionObject(
            object_id=1,
            color="red_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
        )
        det_right = DetectionObject(
            object_id=2,
            color="red_cube",
            bbox=(80, 100, 50, 50),
            confidence=0.9,
        )

        matches = self.detector._match_stereo_detections(
            [det_left], [det_right], max_y_diff=10, max_size_ratio=0.3, min_iou=0.0
        )

        assert len(matches) == 1
        assert matches[0][0] == det_left
        assert matches[0][1] == det_right

    def test_match_stereo_detections_different_class(self):
        det_left = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        det_right = DetectionObject(
            object_id=2, color="blue_cube", bbox=(80, 100, 50, 50), confidence=0.9
        )

        matches = self.detector._match_stereo_detections(
            [det_left], [det_right], max_y_diff=10, max_size_ratio=0.3
        )

        assert len(matches) == 0

    def test_match_stereo_detections_y_diff_threshold(self):
        det_left = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        det_right = DetectionObject(
            object_id=2, color="red_cube", bbox=(80, 120, 50, 50), confidence=0.9
        )

        matches = self.detector._match_stereo_detections(
            [det_left], [det_right], max_y_diff=10, max_size_ratio=0.3
        )
        assert len(matches) == 0

        matches = self.detector._match_stereo_detections(
            [det_left], [det_right], max_y_diff=25, max_size_ratio=0.3
        )
        assert len(matches) == 1

    def test_match_stereo_detections_size_ratio_threshold(self):
        det_left = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        # Area ratio: |2500-3200|/3200=0.21875 — within 0.3, Y-aligned
        det_right = DetectionObject(
            object_id=2, color="red_cube", bbox=(50, 105, 80, 40), confidence=0.9
        )

        matches = self.detector._match_stereo_detections(
            [det_left], [det_right], max_y_diff=10, max_size_ratio=0.3
        )
        assert len(matches) == 1

        # Area ratio: |2500-4000|/4000=0.375 — exceeds 0.3
        det_right_large = DetectionObject(
            object_id=3, color="red_cube", bbox=(40, 105, 100, 40), confidence=0.9
        )

        matches = self.detector._match_stereo_detections(
            [det_left], [det_right_large], max_y_diff=10, max_size_ratio=0.3
        )
        assert len(matches) == 0

    def test_match_stereo_detections_positive_disparity(self):
        det_left = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        det_right = DetectionObject(
            object_id=2, color="red_cube", bbox=(120, 100, 50, 50), confidence=0.9
        )

        matches = self.detector._match_stereo_detections(
            [det_left], [det_right], max_y_diff=10, max_size_ratio=0.3
        )

        assert len(matches) == 0

    def test_match_stereo_detections_iou_threshold(self):
        det_left = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        det_right = DetectionObject(
            object_id=2, color="red_cube", bbox=(70, 100, 50, 50), confidence=0.9
        )

        matches = self.detector._match_stereo_detections(
            [det_left], [det_right], max_y_diff=10, max_size_ratio=0.3, min_iou=0.0
        )
        assert len(matches) == 1

        matches = self.detector._match_stereo_detections(
            [det_left], [det_right], max_y_diff=10, max_size_ratio=0.3, min_iou=0.5
        )
        assert len(matches) == 0

    def test_match_stereo_detections_multiple_objects(self):
        det_left_1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        det_left_2 = DetectionObject(
            object_id=2, color="blue_cube", bbox=(300, 200, 50, 50), confidence=0.85
        )
        det_right_1 = DetectionObject(
            object_id=3, color="red_cube", bbox=(80, 100, 50, 50), confidence=0.9
        )
        det_right_2 = DetectionObject(
            object_id=4, color="blue_cube", bbox=(275, 200, 50, 50), confidence=0.85
        )

        matches = self.detector._match_stereo_detections(
            [det_left_1, det_left_2],
            [det_right_1, det_right_2],
            max_y_diff=10,
            max_size_ratio=0.3,
        )

        assert len(matches) == 2
        matched_colors = {(m[0].color, m[1].color) for m in matches}
        assert ("red_cube", "red_cube") in matched_colors
        assert ("blue_cube", "blue_cube") in matched_colors

    def test_match_stereo_detections_one_sided(self):
        det_left = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )

        matches = self.detector._match_stereo_detections(
            [det_left], [], max_y_diff=10, max_size_ratio=0.3
        )

        assert len(matches) == 0

    def test_match_stereo_detections_ambiguous_matching(self):
        det_left = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        det_right_1 = DetectionObject(
            object_id=2, color="red_cube", bbox=(80, 100, 50, 50), confidence=0.9
        )
        det_right_2 = DetectionObject(
            object_id=3, color="red_cube", bbox=(85, 105, 50, 50), confidence=0.9
        )

        matches = self.detector._match_stereo_detections(
            [det_left],
            [det_right_1, det_right_2],
            max_y_diff=10,
            max_size_ratio=0.3,
        )

        assert len(matches) == 1
        assert matches[0][1] == det_right_1

    def test_match_stereo_detections_no_duplicate_matches(self):
        det_left_1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        det_left_2 = DetectionObject(
            object_id=2, color="red_cube", bbox=(105, 102, 50, 50), confidence=0.9
        )
        det_right = DetectionObject(
            object_id=3, color="red_cube", bbox=(80, 100, 50, 50), confidence=0.9
        )

        matches = self.detector._match_stereo_detections(
            [det_left_1, det_left_2],
            [det_right],
            max_y_diff=10,
            max_size_ratio=0.3,
        )

        assert len(matches) == 1


class TestStereoValidationIntegration:

    def setup_method(self):
        self.detector = YOLODetector.__new__(YOLODetector)

    def test_false_positive_reduction(self):
        det_left_real = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        det_left_false = DetectionObject(
            object_id=2, color="blue_cube", bbox=(300, 200, 50, 50), confidence=0.7
        )
        det_right_real = DetectionObject(
            object_id=3, color="red_cube", bbox=(80, 100, 50, 50), confidence=0.9
        )

        matches = self.detector._match_stereo_detections(
            [det_left_real, det_left_false],
            [det_right_real],
            max_y_diff=10,
            max_size_ratio=0.3,
        )

        assert len(matches) == 1
        assert matches[0][0] == det_left_real

        reduction = (1 - len(matches) / 2) * 100
        assert reduction == 50.0

    def test_typical_stereo_scene(self):
        det_left_1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(200, 150, 60, 60), confidence=0.95
        )
        det_right_1 = DetectionObject(
            object_id=11, color="red_cube", bbox=(160, 150, 60, 60), confidence=0.95
        )
        det_left_2 = DetectionObject(
            object_id=2, color="blue_cube", bbox=(500, 300, 50, 50), confidence=0.88
        )
        det_right_2 = DetectionObject(
            object_id=12, color="blue_cube", bbox=(475, 300, 50, 50), confidence=0.88
        )
        det_left_3 = DetectionObject(
            object_id=3, color="green_cube", bbox=(800, 400, 45, 45), confidence=0.82
        )
        det_right_3 = DetectionObject(
            object_id=13, color="green_cube", bbox=(790, 400, 45, 45), confidence=0.82
        )
        det_left_false = DetectionObject(
            object_id=4, color="yellow_cube", bbox=(1000, 500, 40, 40), confidence=0.65
        )

        matches = self.detector._match_stereo_detections(
            [det_left_1, det_left_2, det_left_3, det_left_false],
            [det_right_1, det_right_2, det_right_3],
            max_y_diff=10,
            max_size_ratio=0.3,
        )

        assert len(matches) == 3
        matched_colors = {m[0].color for m in matches}
        assert matched_colors == {"red_cube", "blue_cube", "green_cube"}
