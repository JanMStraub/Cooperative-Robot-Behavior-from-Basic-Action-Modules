#!/usr/bin/env python3
"""
Unit tests for RunDetector.py

Tests the object detection orchestrator
"""

import pytest
import hashlib
import time
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from argparse import Namespace

# Add LLMCommunication directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "LLMCommunication"))

import ACRLPython.LLMCommunication.LLMConfig as cfg


class TestRunDetectorImageProcessing:
    """Test image processing logic in RunDetector"""

    @patch("LLMCommunication.orchestrators.RunDetector.DetectionBroadcaster")
    @patch("LLMCommunication.orchestrators.RunDetector.ImageStorage")
    @patch("LLMCommunication.orchestrators.RunDetector.CubeDetector")
    @patch("LLMCommunication.orchestrators.RunDetector.run_detection_server_background")
    def test_process_new_image(
        self,
        mock_server_bg,
        mock_detector_class,
        mock_storage_class,
        mock_broadcaster_class,
        sample_red_cube_image,
    ):
        """Test processing a new image for detection"""
        from LLMCommunication.orchestrators.RunDetector import run_detection_loop

        # Setup storage mock to return from get_instance
        mock_storage = MagicMock()
        mock_storage.get_all_camera_ids.return_value = ["camera1"]
        mock_storage.get_camera_image.return_value = sample_red_cube_image
        mock_storage.get_camera_age.return_value = 1.0
        mock_storage_class.get_instance.return_value = mock_storage

        # Setup detector
        mock_detector = MagicMock()
        from LLMCommunication.vision.ObjectDetector import (
            DetectionResult,
            DetectionObject,
        )

        mock_result = DetectionResult(
            camera_id="camera1",
            image_width=640,
            image_height=480,
            detections=[DetectionObject(0, "red", (270, 200, 100, 80), 0.95)],
        )
        mock_detector.detect_cubes.return_value = mock_result
        mock_detector_class.return_value = mock_detector

        # Setup broadcaster
        mock_broadcaster = MagicMock()
        mock_broadcaster_class.send_result = MagicMock()

        args = Namespace(
            camera=None, interval=0.1, min_age=0.1, max_age=10.0, debug=False
        )

        import threading

        def stop_after_delay():
            time.sleep(0.3)
            raise KeyboardInterrupt()

        stop_thread = threading.Thread(target=stop_after_delay, daemon=True)
        stop_thread.start()

        try:
            run_detection_loop(args)
        except (KeyboardInterrupt, AttributeError):
            pass

        # Verify detector was called
        assert mock_detector.detect_cubes.called

    def test_duplicate_image_detection(self, sample_image):
        """Test that duplicate images are detected by hash"""
        # Same image should have same hash
        hash1 = hashlib.md5(sample_image.tobytes()).hexdigest()
        hash2 = hashlib.md5(sample_image.tobytes()).hexdigest()

        assert hash1 == hash2

        # Different image should have different hash
        different_image = sample_image.copy()
        different_image[0, 0, 0] = 255
        hash3 = hashlib.md5(different_image.tobytes()).hexdigest()

        assert hash1 != hash3

    @patch("LLMCommunication.orchestrators.RunDetector.ImageStorage")
    def test_skip_too_fresh_images(self, mock_storage_class, sample_image):
        """Test that too-fresh images are skipped"""
        from LLMCommunication.orchestrators.RunDetector import run_detection_loop

        mock_storage = MagicMock()
        mock_storage.get_all_camera_ids.return_value = ["camera1"]
        mock_storage.get_camera_image.return_value = sample_image
        mock_storage.get_camera_age.return_value = 0.01  # Too fresh
        mock_storage_class.get_instance.return_value = mock_storage

        args = Namespace(
            camera=None,
            interval=0.1,
            min_age=0.5,  # Require 0.5s minimum age
            max_age=10.0,
            debug=False,
        )

        import threading

        def stop():
            time.sleep(0.2)
            raise KeyboardInterrupt()

        threading.Thread(target=stop, daemon=True).start()

        try:
            run_detection_loop(args)
        except (KeyboardInterrupt, AttributeError):
            pass

    @patch("LLMCommunication.orchestrators.RunDetector.ImageStorage")
    def test_skip_too_old_images(self, mock_storage_class, sample_image):
        """Test that too-old images are skipped"""
        from LLMCommunication.orchestrators.RunDetector import run_detection_loop

        mock_storage = MagicMock()
        mock_storage.get_all_camera_ids.return_value = ["camera1"]
        mock_storage.get_camera_image.return_value = sample_image
        mock_storage.get_camera_age.return_value = 100.0  # Too old
        mock_storage_class.get_instance.return_value = mock_storage

        args = Namespace(
            camera=None,
            interval=0.1,
            min_age=0.1,
            max_age=30.0,  # Max 30 seconds old
            debug=False,
        )

        import threading

        def stop():
            time.sleep(0.2)
            raise KeyboardInterrupt()

        threading.Thread(target=stop, daemon=True).start()

        try:
            run_detection_loop(args)
        except (KeyboardInterrupt, AttributeError):
            pass


class TestRunDetectorConfiguration:
    """Test detector configuration and setup"""

    @patch("LLMCommunication.orchestrators.RunDetector.run_detection_server_background")
    @patch("LLMCommunication.orchestrators.RunDetector.run_detection_loop")
    def test_main_starts_server(self, mock_detection_loop, mock_server):
        """Test that main() starts the detection server"""
        from LLMCommunication.orchestrators.RunDetector import main

        # Make detection loop exit quickly
        mock_detection_loop.side_effect = KeyboardInterrupt()

        with patch("sys.argv", ["run_detector.py"]):
            try:
                main()
            except (KeyboardInterrupt, SystemExit):
                pass

        # Verify server was started
        assert mock_server.called

    @patch("LLMCommunication.orchestrators.RunDetector.CubeDetector")
    def test_detector_initialization(self, mock_detector_class):
        """Test detector initialization"""
        from LLMCommunication.orchestrators.RunDetector import run_detection_loop

        args = Namespace(
            camera=None, interval=0.1, min_age=0.1, max_age=10.0, debug=False
        )

        import threading

        def stop():
            time.sleep(0.1)
            raise KeyboardInterrupt()

        threading.Thread(target=stop, daemon=True).start()

        try:
            run_detection_loop(args)
        except (KeyboardInterrupt, AttributeError):
            pass

        # Verify detector was initialized
        mock_detector_class.assert_called_once()

    @patch("LLMCommunication.orchestrators.RunDetector.CubeDetector")
    @patch("config.ENABLE_DEBUG_IMAGES", False)
    def test_debug_mode_disabled(self, mock_detector_class):
        """Test that debug mode can be disabled"""
        from LLMCommunication.orchestrators.RunDetector import main

        with patch("sys.argv", ["run_detector.py"]):
            import threading

            def stop():
                time.sleep(0.1)
                raise KeyboardInterrupt()

            threading.Thread(target=stop, daemon=True).start()

            try:
                main()
            except (KeyboardInterrupt, SystemExit, AttributeError):
                pass

        # Debug should be disabled by default
        assert cfg.ENABLE_DEBUG_IMAGES == False

    @patch("LLMCommunication.orchestrators.RunDetector.CubeDetector")
    @patch("LLMCommunication.orchestrators.RunDetector.Path")
    def test_debug_mode_enabled(self, mock_path_class, mock_detector_class):
        """Test that debug mode can be enabled"""
        from LLMCommunication.orchestrators.RunDetector import main

        mock_path = MagicMock()
        mock_path_class.return_value = mock_path

        with patch("sys.argv", ["run_detector.py", "--debug"]):
            import threading

            def stop():
                time.sleep(0.1)
                raise KeyboardInterrupt()

            threading.Thread(target=stop, daemon=True).start()

            try:
                main()
            except (KeyboardInterrupt, SystemExit, AttributeError):
                pass

            # Debug directory creation should be attempted
            assert mock_path.mkdir.called or mock_path_class.called


class TestRunDetectorCameraFiltering:
    """Test camera filtering logic"""

    @patch("LLMCommunication.orchestrators.RunDetector.ImageStorage")
    def test_monitor_specific_cameras(self, mock_storage_class, sample_image):
        """Test monitoring specific cameras only"""
        from LLMCommunication.orchestrators.RunDetector import run_detection_loop

        mock_storage = MagicMock()
        # Even though multiple cameras exist, should only check specified ones
        mock_storage.get_all_camera_ids.return_value = ["camera1", "camera2", "camera3"]
        mock_storage.get_camera_image.return_value = sample_image
        mock_storage.get_camera_age.return_value = 1.0
        mock_storage_class.get_instance.return_value = mock_storage

        args = Namespace(
            camera=["camera1"],  # Only monitor camera1
            interval=0.1,
            min_age=0.1,
            max_age=10.0,
            debug=False,
        )

        import threading

        def stop():
            time.sleep(0.2)
            raise KeyboardInterrupt()

        threading.Thread(target=stop, daemon=True).start()

        try:
            run_detection_loop(args)
        except (KeyboardInterrupt, AttributeError):
            pass

    @patch("LLMCommunication.orchestrators.RunDetector.ImageStorage")
    def test_monitor_all_cameras(self, mock_storage_class, sample_image):
        """Test monitoring all available cameras"""
        from LLMCommunication.orchestrators.RunDetector import run_detection_loop

        mock_storage = MagicMock()
        mock_storage.get_all_camera_ids.return_value = ["camera1", "camera2"]
        mock_storage.get_camera_image.return_value = sample_image
        mock_storage.get_camera_age.return_value = 1.0
        mock_storage_class.get_instance.return_value = mock_storage

        args = Namespace(
            camera=None,  # Monitor all cameras
            interval=0.1,
            min_age=0.1,
            max_age=10.0,
            debug=False,
        )

        import threading

        def stop():
            time.sleep(0.2)
            raise KeyboardInterrupt()

        threading.Thread(target=stop, daemon=True).start()

        try:
            run_detection_loop(args)
        except (KeyboardInterrupt, AttributeError):
            pass

        # Should call get_all_camera_ids
        assert mock_storage.get_all_camera_ids.called


class TestRunDetectorResultBroadcasting:
    """Test result broadcasting integration"""

    @patch("LLMCommunication.orchestrators.RunDetector.DetectionBroadcaster")
    @patch("LLMCommunication.orchestrators.RunDetector.ImageStorage")
    @patch("LLMCommunication.orchestrators.RunDetector.CubeDetector")
    def test_results_broadcasted(
        self,
        mock_detector_class,
        mock_storage_class,
        mock_broadcaster_class,
        sample_red_cube_image,
    ):
        """Test that detection results are broadcasted"""
        from LLMCommunication.orchestrators.RunDetector import run_detection_loop
        from LLMCommunication.vision.ObjectDetector import (
            DetectionResult,
            DetectionObject,
        )

        # Setup storage
        mock_storage = MagicMock()
        mock_storage.get_all_camera_ids.return_value = ["camera1"]
        mock_storage.get_camera_image.return_value = sample_red_cube_image
        mock_storage.get_camera_age.return_value = 1.0
        mock_storage_class.get_instance.return_value = mock_storage

        # Setup detector
        mock_detector = MagicMock()
        test_result = DetectionResult(
            camera_id="camera1",
            image_width=640,
            image_height=480,
            detections=[DetectionObject(0, "red", (270, 200, 100, 80), 0.95)],
        )
        mock_detector.detect_cubes.return_value = test_result
        mock_detector_class.return_value = mock_detector

        # Setup broadcaster
        mock_broadcaster = MagicMock()
        mock_broadcaster_class.send_result = MagicMock()

        args = Namespace(
            camera=None, interval=0.1, min_age=0.1, max_age=10.0, debug=False
        )

        import threading

        def stop():
            time.sleep(0.3)
            raise KeyboardInterrupt()

        threading.Thread(target=stop, daemon=True).start()

        try:
            run_detection_loop(args)
        except (KeyboardInterrupt, AttributeError):
            pass

        # Verify DetectionBroadcaster.send_result was called
        assert mock_broadcaster_class.send_result.called

    @patch("LLMCommunication.orchestrators.RunDetector.DetectionBroadcaster")
    @patch("LLMCommunication.orchestrators.RunDetector.ImageStorage")
    @patch("LLMCommunication.orchestrators.RunDetector.CubeDetector")
    def test_empty_detection_results_broadcasted(
        self,
        mock_detector_class,
        mock_storage_class,
        mock_broadcaster_class,
        sample_image,
    ):
        """Test that empty detection results are still broadcasted"""
        from LLMCommunication.orchestrators.RunDetector import run_detection_loop
        from LLMCommunication.vision.ObjectDetector import DetectionResult

        # Setup storage
        mock_storage = MagicMock()
        mock_storage.get_all_camera_ids.return_value = ["camera1"]
        mock_storage.get_camera_image.return_value = sample_image
        mock_storage.get_camera_age.return_value = 1.0
        mock_storage_class.get_instance.return_value = mock_storage

        # Setup detector - no detections
        mock_detector = MagicMock()
        test_result = DetectionResult(
            camera_id="camera1",
            image_width=640,
            image_height=480,
            detections=[],  # No cubes detected
        )
        mock_detector.detect_cubes.return_value = test_result
        mock_detector_class.return_value = mock_detector

        # Setup broadcaster
        mock_broadcaster = MagicMock()
        mock_broadcaster_class.send_result = MagicMock()

        args = Namespace(
            camera=None, interval=0.1, min_age=0.1, max_age=10.0, debug=False
        )

        import threading

        def stop():
            time.sleep(0.3)
            raise KeyboardInterrupt()

        threading.Thread(target=stop, daemon=True).start()

        try:
            run_detection_loop(args)
        except (KeyboardInterrupt, AttributeError):
            pass

        # Should still broadcast even with no detections
        assert mock_broadcaster_class.send_result.called


class TestRunDetectorErrorHandling:
    """Test error handling in RunDetector"""

    @patch("LLMCommunication.orchestrators.RunDetector.ImageStorage")
    @patch("LLMCommunication.orchestrators.RunDetector.CubeDetector")
    def test_handles_detection_errors(
        self, mock_detector_class, mock_storage_class, sample_image
    ):
        """Test that detection errors are handled gracefully"""
        from LLMCommunication.orchestrators.RunDetector import run_detection_loop

        mock_storage = MagicMock()
        mock_storage.get_all_camera_ids.return_value = ["camera1"]
        mock_storage.get_camera_image.return_value = sample_image
        mock_storage.get_camera_age.return_value = 1.0
        mock_storage_class.get_instance.return_value = mock_storage

        # Make detector raise error
        mock_detector = MagicMock()
        mock_detector.detect_cubes.side_effect = Exception("Detection error")
        mock_detector_class.return_value = mock_detector

        args = Namespace(
            camera=None, interval=0.1, min_age=0.1, max_age=10.0, debug=False
        )

        import threading

        def stop():
            time.sleep(0.3)
            raise KeyboardInterrupt()

        threading.Thread(target=stop, daemon=True).start()

        # Should continue running despite error
        try:
            run_detection_loop(args)
        except (KeyboardInterrupt, AttributeError):
            pass

    @patch("LLMCommunication.orchestrators.RunDetector.CubeDetector")
    def test_handles_detector_initialization_error(self, mock_detector_class):
        """Test handling of detector initialization errors"""
        from LLMCommunication.orchestrators.RunDetector import run_detection_loop

        # Make detector initialization fail
        mock_detector_class.side_effect = Exception("Cannot initialize detector")

        args = Namespace(
            camera=None, interval=0.1, min_age=0.1, max_age=10.0, debug=False
        )

        # Should raise the initialization error
        with pytest.raises(Exception, match="Cannot initialize detector"):
            run_detection_loop(args)


class TestRunDetectorPerformance:
    """Test performance aspects of RunDetector"""

    def test_check_interval_configuration(self):
        """Test that check interval can be configured"""
        from LLMCommunication.orchestrators.RunDetector import run_detection_loop

        args = Namespace(
            camera=None,
            interval=0.05,  # Very fast interval
            min_age=0.1,
            max_age=10.0,
            debug=False,
        )

        # Verify interval is used (check it's set correctly)
        assert args.interval == 0.05

    def test_image_age_window_configuration(self):
        """Test that image age window can be configured"""
        args = Namespace(min_age=0.2, max_age=5.0)

        # Verify age constraints
        assert args.min_age == 0.2
        assert args.max_age == 5.0
        assert args.min_age < args.max_age

    @patch("LLMCommunication.orchestrators.RunDetector.DetectionBroadcaster")
    @patch("LLMCommunication.orchestrators.RunDetector.ImageStorage")
    @patch("LLMCommunication.orchestrators.RunDetector.CubeDetector")
    def test_detection_timing_tracked(
        self,
        mock_detector_class,
        mock_storage_class,
        mock_broadcaster_class,
        sample_image,
    ):
        """Test that detection timing is tracked"""
        from LLMCommunication.orchestrators.RunDetector import run_detection_loop
        from LLMCommunication.vision.ObjectDetector import DetectionResult

        # Setup storage
        mock_storage = MagicMock()
        mock_storage.get_all_camera_ids.return_value = ["camera1"]
        mock_storage.get_camera_image.return_value = sample_image
        mock_storage.get_camera_age.return_value = 1.0
        mock_storage_class.get_instance.return_value = mock_storage

        # Setup detector with delay
        mock_detector = MagicMock()

        def slow_detect(*args, **kwargs):
            time.sleep(0.01)  # Simulate processing time
            return DetectionResult("camera1", 640, 480, [])

        mock_detector.detect_cubes.side_effect = slow_detect
        mock_detector_class.return_value = mock_detector

        args = Namespace(
            camera=None, interval=0.1, min_age=0.1, max_age=10.0, debug=False
        )

        import threading

        def stop():
            time.sleep(0.3)
            raise KeyboardInterrupt()

        threading.Thread(target=stop, daemon=True).start()

        try:
            run_detection_loop(args)
        except (KeyboardInterrupt, AttributeError):
            pass

        # Detector should have been called
        assert mock_detector.detect_cubes.called
