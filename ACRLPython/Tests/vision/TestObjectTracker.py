from vision.ObjectTracker import ObjectTracker, Track
from vision.DetectionDataModels import DetectionObject


class TestTrack:

    def test_track_initialization(self):
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

        assert track.track_id == 1
        assert track.bbox == (100, 100, 50, 50)
        assert track.color == "red_cube"
        assert track.age == 0
        assert track.hits == 1
        assert len(track.position_history) == 0

    def test_track_update(self):
        track = Track(track_id=1, bbox=(100, 100, 50, 50), color="red_cube")

        det = DetectionObject(
            object_id=2, color="red_cube", bbox=(110, 105, 50, 50), confidence=0.95
        )

        track.update(det)

        assert track.bbox == (110, 105, 50, 50)
        assert track.age == 0
        assert track.hits == 2
        assert len(track.position_history) == 1
        assert track.position_history[0] == (det.center_x, det.center_y)

    def test_position_history_limit(self):
        track = Track(track_id=1, bbox=(100, 100, 50, 50), color="red_cube")

        for i in range(15):
            det = DetectionObject(
                object_id=i,
                color="red_cube",
                bbox=(100 + i * 10, 100, 50, 50),
                confidence=0.9,
            )
            track.update(det)

        assert len(track.position_history) == 10

    def test_predict_next_position(self):
        track = Track(track_id=1, bbox=(100, 100, 50, 50), color="red_cube")

        det1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        det2 = DetectionObject(
            object_id=2, color="red_cube", bbox=(110, 105, 50, 50), confidence=0.9
        )

        track.update(det1)
        track.update(det2)

        pred = track.predict_next_position()
        assert pred is not None

        # Prediction based on velocity
        expected_x = det2.center_x + (det2.center_x - det1.center_x)
        expected_y = det2.center_y + (det2.center_y - det1.center_y)
        assert pred == (expected_x, expected_y)

    def test_get_velocity(self):
        track = Track(track_id=1, bbox=(100, 100, 50, 50), color="red_cube")

        assert track.get_velocity() is None

        det1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        det2 = DetectionObject(
            object_id=2, color="red_cube", bbox=(110, 105, 50, 50), confidence=0.9
        )

        track.update(det1)
        track.update(det2)

        velocity = track.get_velocity()
        assert velocity is not None

        vx = det2.center_x - det1.center_x
        vy = det2.center_y - det1.center_y
        assert velocity == (float(vx), float(vy))


class TestObjectTracker:

    def test_tracker_initialization(self):
        tracker = ObjectTracker(max_age=5, min_iou=0.3)

        assert tracker.max_age == 5
        assert tracker.min_iou == 0.3
        assert len(tracker.tracks) == 0
        assert tracker.next_id == 1

    def test_single_object_tracking(self):
        tracker = ObjectTracker(max_age=3, min_iou=0.3)

        det1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        tracked1 = tracker.update([det1])

        assert len(tracked1) == 1
        assert tracked1[0].track_id == 1
        assert len(tracker.tracks) == 1

        det2 = DetectionObject(
            object_id=2, color="red_cube", bbox=(105, 102, 50, 50), confidence=0.9
        )
        tracked2 = tracker.update([det2])

        assert len(tracked2) == 1
        assert tracked2[0].track_id == 1
        assert len(tracker.tracks) == 1

        det3 = DetectionObject(
            object_id=3, color="red_cube", bbox=(110, 104, 50, 50), confidence=0.9
        )
        tracked3 = tracker.update([det3])

        assert len(tracked3) == 1
        assert tracked3[0].track_id == 1

    def test_multiple_objects_tracking(self):
        tracker = ObjectTracker(max_age=3, min_iou=0.3)

        det1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        det2 = DetectionObject(
            object_id=2, color="blue_cube", bbox=(300, 200, 50, 50), confidence=0.85
        )
        tracked1 = tracker.update([det1, det2])

        assert len(tracked1) == 2
        assert tracked1[0].track_id == 1
        assert tracked1[1].track_id == 2
        assert len(tracker.tracks) == 2

        det3 = DetectionObject(
            object_id=3, color="red_cube", bbox=(105, 102, 50, 50), confidence=0.9
        )
        det4 = DetectionObject(
            object_id=4, color="blue_cube", bbox=(305, 202, 50, 50), confidence=0.85
        )
        tracked2 = tracker.update([det3, det4])

        assert len(tracked2) == 2
        track_ids = {t.track_id for t in tracked2}
        assert track_ids == {1, 2}

    def test_track_aging_and_cleanup(self):
        tracker = ObjectTracker(max_age=3, min_iou=0.3)

        det1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        tracker.update([det1])
        assert len(tracker.tracks) == 1

        for _ in range(4):
            tracker.update([])

        assert len(tracker.tracks) == 0

    def test_new_track_creation(self):
        tracker = ObjectTracker(max_age=3, min_iou=0.3)

        det1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        tracker.update([det1])
        assert tracker.next_id == 2

        det2 = DetectionObject(
            object_id=2, color="blue_cube", bbox=(300, 200, 50, 50), confidence=0.85
        )
        tracker.update([det1, det2])

        assert len(tracker.tracks) == 2
        assert tracker.next_id == 3

    def test_iou_matching(self):
        tracker = ObjectTracker(max_age=3, min_iou=0.5)

        det1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        tracker.update([det1])

        det2 = DetectionObject(
            object_id=2,
            color="red_cube",
            bbox=(200, 200, 50, 50),
            confidence=0.9,
        )
        tracker.update([det2])

        assert len(tracker.tracks) == 2

    def test_class_filtering(self):
        tracker = ObjectTracker(max_age=3, min_iou=0.3)

        det1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        tracker.update([det1])

        det2 = DetectionObject(
            object_id=2, color="blue_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        tracker.update([det2])

        assert len(tracker.tracks) == 2

    def test_get_active_tracks(self):
        tracker = ObjectTracker(max_age=3, min_iou=0.3)

        det1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        tracker.update([det1])

        tracks = tracker.get_active_tracks()
        assert len(tracks) == 1

        tracks.clear()
        assert len(tracker.get_active_tracks()) == 1

    def test_reset(self):
        tracker = ObjectTracker(max_age=3, min_iou=0.3)

        det1 = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        det2 = DetectionObject(
            object_id=2, color="blue_cube", bbox=(300, 200, 50, 50), confidence=0.85
        )
        tracker.update([det1, det2])

        assert len(tracker.tracks) == 2
        assert tracker.next_id == 3

        tracker.reset()

        assert len(tracker.tracks) == 0
        assert tracker.next_id == 1

    def test_calculate_iou(self):
        tracker = ObjectTracker()

        iou = tracker._calculate_iou((100, 100, 50, 50), (100, 100, 50, 50))
        assert abs(iou - 1.0) < 1e-7

        iou = tracker._calculate_iou((100, 100, 50, 50), (200, 200, 50, 50))
        assert abs(iou - 0.0) < 1e-7

        # Intersection: 25*50=1250, Union: 50*50+50*50-1250=3750, IOU=1250/3750
        iou = tracker._calculate_iou((100, 100, 50, 50), (125, 100, 50, 50))
        assert abs(iou - 1250.0 / 3750.0) < 1e-3


class TestObjectTrackerIntegration:

    def test_realistic_tracking_scenario(self):
        tracker = ObjectTracker(max_age=3, min_iou=0.3)

        for frame_idx in range(10):
            detections = []

            if frame_idx < 8:
                x = 100 + frame_idx * 10
                det1 = DetectionObject(
                    object_id=frame_idx * 10 + 1,
                    color="red_cube",
                    bbox=(x, 100, 50, 50),
                    confidence=0.9,
                )
                detections.append(det1)

            if frame_idx > 2:
                det2 = DetectionObject(
                    object_id=frame_idx * 10 + 2,
                    color="blue_cube",
                    bbox=(300, 200, 50, 50),
                    confidence=0.85,
                )
                detections.append(det2)

            tracked = tracker.update(detections)

            if frame_idx == 0:
                assert len(tracked) == 1
                assert tracked[0].track_id == 1

            if frame_idx == 3:
                assert len(tracked) == 2
                track_ids = {t.track_id for t in tracked}
                assert track_ids == {1, 2}

            if frame_idx == 9:
                assert len(tracked) == 1
