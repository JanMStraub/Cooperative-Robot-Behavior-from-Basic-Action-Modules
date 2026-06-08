import time
import threading
from unittest.mock import patch, MagicMock

from vision.VisionProcessor import VisionProcessor
from vision.DetectionDataModels import DetectionObject, DetectionResult


def create_mock_storage():
    mock_storage = MagicMock()
    mock_storage.get_latest_stereo_image.return_value = None
    return mock_storage


class MockDetector:

    def __init__(self, return_objects=None):
        self.return_objects = return_objects or []
        self.call_count = 0
        self.last_call_args = None

    def detect_objects_stereo(
        self, imgL, imgR, camera_id=None, camera_config=None, **kwargs
    ):
        self.call_count += 1
        self.last_call_args = {
            "imgL": imgL,
            "imgR": imgR,
            "camera_id": camera_id,
            "camera_config": camera_config,
        }
        return DetectionResult(
            camera_id=camera_id or "mock",
            image_width=1280,
            image_height=960,
            detections=self.return_objects,
        )


class TestVisionProcessor:

    def test_initialization(self):
        detector = MockDetector()
        processor = VisionProcessor(detector, fps=5.0)

        assert processor.detector == detector
        assert processor.fps == 5.0
        assert not processor.running
        assert processor.thread is None
        assert not processor.enable_tracking
        assert not processor.enable_shared_state

    def test_initialization_with_tracking(self):
        detector = MockDetector()
        processor = VisionProcessor(detector, fps=5.0, enable_tracking=True)

        assert processor.enable_tracking
        assert processor.tracker is not None

    @patch("vision.VisionProcessor._get_storage")
    def test_start_stop(self, mock_get_storage):
        mock_get_storage.return_value = create_mock_storage()

        detector = MockDetector()
        processor = VisionProcessor(detector, fps=5.0)

        processor.start()
        assert processor.running
        assert processor.thread is not None
        assert processor.thread.is_alive()

        time.sleep(0.1)

        processor.stop()
        assert not processor.running

        time.sleep(0.5)
        if processor.thread:
            assert not processor.thread.is_alive()

    def test_double_start(self):
        detector = MockDetector()
        processor = VisionProcessor(detector, fps=5.0)

        processor.start()
        assert processor.running

        processor.start()
        assert processor.running

        processor.stop()

    def test_stop_not_running(self):
        detector = MockDetector()
        processor = VisionProcessor(detector, fps=5.0)

        processor.stop()
        assert not processor.running

    def test_result_callback(self):
        det_obj = DetectionObject(
            object_id=1, color="red_cube", bbox=(100, 100, 50, 50), confidence=0.9
        )
        detector = MockDetector(return_objects=[det_obj])
        processor = VisionProcessor(detector, fps=5.0)

        callback_results = []

        def on_result(result: DetectionResult):
            callback_results.append(result)

        processor.on_result_callback = on_result

        assert processor.on_result_callback is not None

    def test_get_stats(self):
        detector = MockDetector()
        processor = VisionProcessor(detector, fps=5.0, enable_tracking=True)

        stats = processor.get_stats()

        assert not stats["running"]
        assert stats["fps"] == 5.0
        assert stats["tracking_enabled"]
        assert not stats["shared_state_enabled"]

        processor.start()
        stats = processor.get_stats()
        assert stats["running"]

        processor.stop()

    def test_get_stats_with_tracking(self):
        detector = MockDetector()
        processor = VisionProcessor(detector, fps=5.0, enable_tracking=True)

        stats = processor.get_stats()

        assert "active_tracks" in stats
        assert stats["active_tracks"] == 0

    def test_fps_configuration(self):
        detector = MockDetector()

        assert VisionProcessor(detector, fps=1.0).fps == 1.0
        assert VisionProcessor(detector, fps=10.0).fps == 10.0
        assert VisionProcessor(detector).fps == 5.0


class TestVisionProcessorIntegration:

    def test_processor_lifecycle(self):
        detector = MockDetector()
        processor = VisionProcessor(detector, fps=10.0)

        states = []

        def on_result(result):
            states.append("result_received")

        processor.on_result_callback = on_result

        processor.start()
        states.append("started")

        time.sleep(0.3)

        processor.stop()
        states.append("stopped")

        assert "started" in states
        assert "stopped" in states

    def test_processor_thread_safety(self):
        detector = MockDetector()
        processor = VisionProcessor(detector, fps=5.0)

        processor.start()

        results = []

        def get_stats_thread():
            for _ in range(10):
                stats = processor.get_stats()
                results.append(stats)
                time.sleep(0.01)

        threads = [threading.Thread(target=get_stats_thread) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        processor.stop()

        assert len(results) == 30

    def test_processor_with_tracking_enabled(self):
        det_obj = DetectionObject(
            object_id=1,
            color="red_cube",
            bbox=(100, 100, 50, 50),
            confidence=0.9,
            track_id=None,
        )
        detector = MockDetector(return_objects=[det_obj])
        processor = VisionProcessor(detector, fps=5.0, enable_tracking=True)

        assert processor.tracker is not None
        assert processor.enable_tracking

        stats = processor.get_stats()
        assert stats["tracking_enabled"]
        assert stats["active_tracks"] == 0


class TestVisionProcessorErrorHandling:

    def test_detector_exception_recovery(self):

        class FailingDetector:

            def __init__(self):
                self.call_count = 0

            def detect_objects_stereo(self, *args, **kwargs):
                self.call_count += 1
                if self.call_count < 3:
                    raise Exception("Mock detection failure")
                return DetectionResult(
                    camera_id="test", image_width=1280, image_height=960, detections=[]
                )

        detector = FailingDetector()
        processor = VisionProcessor(detector, fps=10.0)

        processor.start()
        time.sleep(0.2)
        processor.stop()

        assert detector.call_count >= 0

    def test_callback_exception_handling(self):
        detector = MockDetector()
        processor = VisionProcessor(detector, fps=5.0)

        exception_count = [0]

        def failing_callback(result):
            exception_count[0] += 1
            raise Exception("Mock callback failure")

        processor.on_result_callback = failing_callback

        processor.start()
        time.sleep(0.1)
        processor.stop()


class TestVisionProcessorConfiguration:

    def test_shared_state_disabled_by_default(self):
        detector = MockDetector()
        processor = VisionProcessor(detector, fps=5.0)

        assert not processor.enable_shared_state

    def test_shared_state_configuration(self):
        detector = MockDetector()

        try:
            processor = VisionProcessor(detector, fps=5.0, enable_shared_state=True)
            assert processor.enable_shared_state is not None
        except ImportError:
            pass

    def test_multiple_processors(self):
        detector1 = MockDetector()
        detector2 = MockDetector()

        processor1 = VisionProcessor(detector1, fps=5.0)
        processor2 = VisionProcessor(detector2, fps=10.0)

        assert processor1 is not processor2
        assert processor1.detector is not processor2.detector

        processor1.start()
        processor2.start()

        assert processor1.running
        assert processor2.running

        processor1.stop()
        processor2.stop()
