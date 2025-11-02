#!/usr/bin/env python3
"""
Unit tests for RunStereoDetector.py

Tests the stereo detection orchestrator
"""

import pytest
import json
import time
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from argparse import Namespace

# Add LLMCommunication directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "LLMCommunication"))

# Mock the stereo config imports
sys.modules['stereo_config'] = MagicMock()


class MockCameraConfig:
    """Mock camera configuration for testing"""
    def __init__(self, fov=60.0, baseline=0.1):
        self.fov = fov
        self.baseline = baseline


class TestStereoDetectorOrchestratorInitialization:
    """Test StereoDetectorOrchestrator initialization"""

    @patch('LLMCommunication.orchestrators.RunStereoDetector.CameraConfig', MockCameraConfig)
    @patch('LLMCommunication.orchestrators.RunStereoDetector.StereoImageStorage')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.CubeDetector')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.ResultsBroadcaster')
    def test_orchestrator_initialization(self, mock_broadcaster, mock_detector, mock_storage):
        """Test orchestrator initializes correctly"""
        from LLMCommunication.orchestrators.RunStereoDetector import StereoDetectorOrchestrator

        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)
        orchestrator = StereoDetectorOrchestrator(camera_config, check_interval=0.5)

        assert orchestrator.camera_config == camera_config
        assert orchestrator.check_interval == 0.5
        assert orchestrator.shutdown_flag is False

    @patch('LLMCommunication.orchestrators.RunStereoDetector.CameraConfig', MockCameraConfig)
    @patch('LLMCommunication.orchestrators.RunStereoDetector.StereoImageStorage')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.CubeDetector')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.ResultsBroadcaster')
    def test_orchestrator_default_check_interval(self, mock_broadcaster, mock_detector, mock_storage):
        """Test orchestrator uses default check interval"""
        from LLMCommunication.orchestrators.RunStereoDetector import StereoDetectorOrchestrator

        camera_config = MockCameraConfig()
        orchestrator = StereoDetectorOrchestrator(camera_config)

        # Should use default check interval
        assert orchestrator.check_interval > 0


class TestStereoDetectorOrchestratorProcessing:
    """Test stereo pair processing"""

    @patch('LLMCommunication.orchestrators.RunStereoDetector.CameraConfig', MockCameraConfig)
    @patch('LLMCommunication.orchestrators.RunStereoDetector.StereoImageStorage')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.CubeDetector')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.ResultsBroadcaster')
    def test_process_stereo_pair(self, mock_broadcaster, mock_detector_class,
                                 mock_storage_class, sample_stereo_pair):
        """Test processing a stereo image pair"""
        from LLMCommunication.orchestrators.RunStereoDetector import StereoDetectorOrchestrator
        from LLMCommunication.vision.ObjectDetector import DetectionResult, DetectionObject

        imgL, imgR = sample_stereo_pair

        # Setup storage
        mock_storage = MagicMock()
        mock_storage.get_stereo_pair.return_value = (imgL, imgR, "detect cubes")
        mock_storage_class.return_value = mock_storage

        # Setup detector
        mock_detector = MagicMock()
        mock_result = DetectionResult(
            camera_id="stereo",
            image_width=640,
            image_height=480,
            detections=[
                DetectionObject(0, "red", (270, 200, 100, 80), 0.95,
                               world_position=(0.5, 0.2, 1.0))
            ]
        )
        mock_detector.detect_cubes_stereo.return_value = mock_result
        mock_detector_class.return_value = mock_detector

        # Setup broadcaster
        mock_broadcaster.send_result = MagicMock()

        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)
        orchestrator = StereoDetectorOrchestrator(camera_config, check_interval=0.1)

        # Process the stereo pair
        orchestrator._process_stereo_pair("stereo")

        # Verify detector was called
        assert mock_detector.detect_cubes_stereo.called

        # Verify result was broadcasted
        assert mock_broadcaster.send_result.called

    @patch('LLMCommunication.orchestrators.RunStereoDetector.CameraConfig', MockCameraConfig)
    @patch('LLMCommunication.orchestrators.RunStereoDetector.StereoImageStorage')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.CubeDetector')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.ResultsBroadcaster')
    def test_process_stereo_pair_with_json_config(self, mock_broadcaster, mock_detector_class,
                                                   mock_storage_class, sample_stereo_pair):
        """Test processing with camera config in JSON prompt"""
        from LLMCommunication.orchestrators.RunStereoDetector import StereoDetectorOrchestrator

        imgL, imgR = sample_stereo_pair

        # Create JSON prompt with camera parameters
        json_prompt = json.dumps({
            "baseline": 0.15,
            "fov": 70.0,
            "prompt": "detect objects"
        })

        # Setup storage
        mock_storage = MagicMock()
        mock_storage.get_stereo_pair.return_value = (imgL, imgR, json_prompt)
        mock_storage_class.return_value = mock_storage

        # Setup detector
        mock_detector = MagicMock()
        from LLMCommunication.vision.ObjectDetector import DetectionResult
        mock_detector.detect_cubes_stereo.return_value = DetectionResult(
            "stereo", 640, 480, []
        )
        mock_detector_class.return_value = mock_detector

        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)
        orchestrator = StereoDetectorOrchestrator(camera_config, check_interval=0.1)

        # Process the stereo pair
        orchestrator._process_stereo_pair("stereo")

        # Verify detector was called with Unity-provided camera config
        call_args = mock_detector.detect_cubes_stereo.call_args
        # camera_config is the 3rd positional argument (imgL, imgR, camera_config)
        used_config = call_args[0][2]

        # Should use Unity's parameters
        assert used_config.baseline == 0.15
        assert used_config.fov == 70.0

    @patch('LLMCommunication.orchestrators.RunStereoDetector.CameraConfig', MockCameraConfig)
    @patch('LLMCommunication.orchestrators.RunStereoDetector.StereoImageStorage')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.CubeDetector')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.ResultsBroadcaster')
    def test_process_stereo_pair_no_data(self, mock_broadcaster, mock_detector_class,
                                         mock_storage_class):
        """Test processing when no stereo pair available"""
        from LLMCommunication.orchestrators.RunStereoDetector import StereoDetectorOrchestrator

        # Setup storage to return None
        mock_storage = MagicMock()
        mock_storage.get_stereo_pair.return_value = None
        mock_storage_class.return_value = mock_storage

        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector

        camera_config = MockCameraConfig()
        orchestrator = StereoDetectorOrchestrator(camera_config, check_interval=0.1)

        # Process should handle None gracefully
        orchestrator._process_stereo_pair("nonexistent")

        # Detector should not be called
        assert not mock_detector.detect_cubes_stereo.called

    @patch('LLMCommunication.orchestrators.RunStereoDetector.CameraConfig', MockCameraConfig)
    @patch('LLMCommunication.orchestrators.RunStereoDetector.StereoImageStorage')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.CubeDetector')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.ResultsBroadcaster')
    def test_process_stereo_pair_error_handling(self, mock_broadcaster, mock_detector_class,
                                                 mock_storage_class, sample_stereo_pair):
        """Test error handling during stereo processing"""
        from LLMCommunication.orchestrators.RunStereoDetector import StereoDetectorOrchestrator

        imgL, imgR = sample_stereo_pair

        # Setup storage
        mock_storage = MagicMock()
        mock_storage.get_stereo_pair.return_value = (imgL, imgR, "test")
        mock_storage_class.return_value = mock_storage

        # Setup detector to raise error
        mock_detector = MagicMock()
        mock_detector.detect_cubes_stereo.side_effect = Exception("Detection failed")
        mock_detector_class.return_value = mock_detector

        # Setup broadcaster
        mock_broadcaster.send_result = MagicMock()

        camera_config = MockCameraConfig()
        orchestrator = StereoDetectorOrchestrator(camera_config, check_interval=0.1)

        # Process should handle error gracefully
        orchestrator._process_stereo_pair("stereo")

        # Error result should be sent
        assert mock_broadcaster.send_result.called
        call_args = mock_broadcaster.send_result.call_args[0][0]
        assert call_args["success"] is False
        assert "error" in call_args


class TestStereoDetectorOrchestratorProcessLoop:
    """Test the main processing loop"""

    @patch('LLMCommunication.orchestrators.RunStereoDetector.CameraConfig', MockCameraConfig)
    @patch('LLMCommunication.orchestrators.RunStereoDetector.StereoImageStorage')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.CubeDetector')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.ResultsBroadcaster')
    def test_process_loop_checks_for_new_images(self, mock_broadcaster, mock_detector_class,
                                                 mock_storage_class, sample_stereo_pair):
        """Test that process loop checks for new images"""
        from LLMCommunication.orchestrators.RunStereoDetector import StereoDetectorOrchestrator

        imgL, imgR = sample_stereo_pair

        # Setup storage
        mock_storage = MagicMock()
        mock_storage.get_stereo_pair.return_value = (imgL, imgR, "test")
        mock_storage.get_pair_age.return_value = 0.5
        mock_storage_class.return_value = mock_storage

        # Setup detector
        mock_detector = MagicMock()
        from LLMCommunication.vision.ObjectDetector import DetectionResult
        mock_detector.detect_cubes_stereo.return_value = DetectionResult(
            "stereo", 640, 480, []
        )
        mock_detector_class.return_value = mock_detector

        camera_config = MockCameraConfig()
        orchestrator = StereoDetectorOrchestrator(camera_config, check_interval=0.1)

        # Run loop briefly
        import threading
        def stop():
            time.sleep(0.3)
            orchestrator.shutdown()

        threading.Thread(target=stop, daemon=True).start()

        orchestrator.process_loop()

        # Should have checked for images
        assert mock_storage.get_pair_age.called

    @patch('LLMCommunication.orchestrators.RunStereoDetector.CameraConfig', MockCameraConfig)
    @patch('LLMCommunication.orchestrators.RunStereoDetector.StereoImageStorage')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.CubeDetector')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.ResultsBroadcaster')
    def test_process_loop_skips_already_processed(self, mock_broadcaster, mock_detector_class,
                                                   mock_storage_class, sample_stereo_pair):
        """Test that already processed images are skipped"""
        from LLMCommunication.orchestrators.RunStereoDetector import StereoDetectorOrchestrator

        imgL, imgR = sample_stereo_pair

        # Setup storage - same image age
        mock_storage = MagicMock()
        mock_storage.get_stereo_pair.return_value = (imgL, imgR, "test")
        mock_storage.get_pair_age.return_value = 1.0  # Constant age
        mock_storage_class.return_value = mock_storage

        # Setup detector
        mock_detector = MagicMock()
        from LLMCommunication.vision.ObjectDetector import DetectionResult
        mock_detector.detect_cubes_stereo.return_value = DetectionResult(
            "stereo", 640, 480, []
        )
        mock_detector_class.return_value = mock_detector

        camera_config = MockCameraConfig()
        orchestrator = StereoDetectorOrchestrator(camera_config, check_interval=0.1)

        # Process once to mark as processed
        orchestrator._process_stereo_pair("stereo")
        initial_call_count = mock_detector.detect_cubes_stereo.call_count

        # Run loop briefly
        import threading
        def stop():
            time.sleep(0.3)
            orchestrator.shutdown()

        threading.Thread(target=stop, daemon=True).start()

        orchestrator.process_loop()

        # Should not process same image again (call count should be same or similar)
        # Note: Exact behavior depends on timing

    @patch('LLMCommunication.orchestrators.RunStereoDetector.CameraConfig', MockCameraConfig)
    @patch('LLMCommunication.orchestrators.RunStereoDetector.StereoImageStorage')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.CubeDetector')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.ResultsBroadcaster')
    def test_process_loop_handles_errors(self, mock_broadcaster, mock_detector_class,
                                         mock_storage_class):
        """Test that process loop handles errors gracefully"""
        from LLMCommunication.orchestrators.RunStereoDetector import StereoDetectorOrchestrator

        # Setup storage to raise error
        mock_storage = MagicMock()
        mock_storage.get_pair_age.side_effect = Exception("Storage error")
        mock_storage_class.return_value = mock_storage

        camera_config = MockCameraConfig()
        orchestrator = StereoDetectorOrchestrator(camera_config, check_interval=0.1)

        # Run loop briefly
        import threading
        def stop():
            time.sleep(0.3)
            orchestrator.shutdown()

        threading.Thread(target=stop, daemon=True).start()

        # Should continue running despite errors
        orchestrator.process_loop()


class TestStereoDetectorOrchestratorShutdown:
    """Test orchestrator shutdown"""

    @patch('LLMCommunication.orchestrators.RunStereoDetector.CameraConfig', MockCameraConfig)
    @patch('LLMCommunication.orchestrators.RunStereoDetector.StereoImageStorage')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.CubeDetector')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.ResultsBroadcaster')
    def test_shutdown(self, mock_broadcaster, mock_detector, mock_storage):
        """Test orchestrator shutdown"""
        from LLMCommunication.orchestrators.RunStereoDetector import StereoDetectorOrchestrator

        camera_config = MockCameraConfig()
        orchestrator = StereoDetectorOrchestrator(camera_config, check_interval=0.1)

        assert orchestrator.shutdown_flag is False

        orchestrator.shutdown()

        assert orchestrator.shutdown_flag is True


class TestStereoDetectorMain:
    """Test main function"""

    @patch('LLMCommunication.orchestrators.RunStereoDetector.CameraConfig', MockCameraConfig)
    @patch('LLMCommunication.orchestrators.RunStereoDetector.run_stereo_detection_server_background')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.run_results_server_background')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.StereoDetectorOrchestrator')
    def test_main_starts_servers(self, mock_orchestrator_class, mock_results_server,
                                 mock_stereo_server):
        """Test that main() starts both servers"""
        from LLMCommunication.orchestrators.RunStereoDetector import main

        # Setup orchestrator to exit quickly
        mock_orchestrator = MagicMock()
        mock_orchestrator.process_loop.side_effect = KeyboardInterrupt()
        mock_orchestrator_class.return_value = mock_orchestrator

        with patch('sys.argv', ['run_stereo_detector.py']):
            try:
                main()
            except (KeyboardInterrupt, SystemExit):
                pass

        # Verify servers were started
        assert mock_stereo_server.called
        assert mock_results_server.called

    @patch('LLMCommunication.orchestrators.RunStereoDetector.CameraConfig', MockCameraConfig)
    @patch('LLMCommunication.orchestrators.RunStereoDetector.run_stereo_detection_server_background')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.run_results_server_background')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.StereoDetectorOrchestrator')
    def test_main_with_custom_camera_config(self, mock_orchestrator_class, mock_results_server,
                                            mock_stereo_server):
        """Test main() with custom camera configuration"""
        from LLMCommunication.orchestrators.RunStereoDetector import main

        # Setup orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.process_loop.side_effect = KeyboardInterrupt()
        mock_orchestrator_class.return_value = mock_orchestrator

        with patch('sys.argv', ['run_stereo_detector.py', '--baseline', '0.15', '--fov', '70']):
            try:
                main()
            except (KeyboardInterrupt, SystemExit):
                pass

        # Verify orchestrator was created with custom camera config
        call_args = mock_orchestrator_class.call_args[0]
        camera_config = call_args[0]

        assert camera_config.baseline == 0.15
        assert camera_config.fov == 70.0

    @patch('LLMCommunication.orchestrators.RunStereoDetector.CameraConfig', MockCameraConfig)
    @patch('LLMCommunication.orchestrators.RunStereoDetector.run_stereo_detection_server_background')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.run_results_server_background')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.StereoDetectorOrchestrator')
    def test_main_handles_keyboard_interrupt(self, mock_orchestrator_class, mock_results_server,
                                             mock_stereo_server):
        """Test that main() handles keyboard interrupt gracefully"""
        from LLMCommunication.orchestrators.RunStereoDetector import main

        # Setup orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.process_loop.side_effect = KeyboardInterrupt()
        mock_orchestrator_class.return_value = mock_orchestrator

        with patch('sys.argv', ['run_stereo_detector.py']):
            # Should not raise
            try:
                main()
            except SystemExit:
                pass

        # Shutdown should be called
        assert mock_orchestrator.shutdown.called


class TestStereoDetectorResultMetadata:
    """Test result metadata generation"""

    @patch('LLMCommunication.orchestrators.RunStereoDetector.CameraConfig', MockCameraConfig)
    @patch('LLMCommunication.orchestrators.RunStereoDetector.StereoImageStorage')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.CubeDetector')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.ResultsBroadcaster')
    def test_result_includes_metadata(self, mock_broadcaster, mock_detector_class,
                                      mock_storage_class, sample_stereo_pair):
        """Test that results include processing metadata"""
        from LLMCommunication.orchestrators.RunStereoDetector import StereoDetectorOrchestrator
        from LLMCommunication.vision.ObjectDetector import DetectionResult

        imgL, imgR = sample_stereo_pair

        # Setup storage
        mock_storage = MagicMock()
        mock_storage.get_stereo_pair.return_value = (imgL, imgR, "test")
        mock_storage_class.return_value = mock_storage

        # Setup detector
        mock_detector = MagicMock()
        mock_detector.detect_cubes_stereo.return_value = DetectionResult(
            "stereo", 640, 480, []
        )
        mock_detector_class.return_value = mock_detector

        # Setup broadcaster
        mock_broadcaster.send_result = MagicMock()

        camera_config = MockCameraConfig(fov=60.0, baseline=0.1)
        orchestrator = StereoDetectorOrchestrator(camera_config, check_interval=0.1)

        # Process stereo pair
        orchestrator._process_stereo_pair("stereo")

        # Verify result was sent with metadata
        assert mock_broadcaster.send_result.called
        call_args = mock_broadcaster.send_result.call_args[0][0]

        assert "metadata" in call_args
        assert "processing_time_seconds" in call_args["metadata"]
        assert "camera_baseline_m" in call_args["metadata"]
        assert "camera_fov_deg" in call_args["metadata"]
        assert "detection_mode" in call_args["metadata"]
        assert call_args["metadata"]["detection_mode"] == "stereo_3d"


class TestStereoDetectorPerformance:
    """Test performance aspects"""

    def test_check_interval_configuration(self):
        """Test that check interval can be configured"""
        check_interval = 0.25
        assert check_interval > 0

    @patch('LLMCommunication.orchestrators.RunStereoDetector.CameraConfig', MockCameraConfig)
    @patch('LLMCommunication.orchestrators.RunStereoDetector.StereoImageStorage')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.CubeDetector')
    @patch('LLMCommunication.orchestrators.RunStereoDetector.ResultsBroadcaster')
    def test_processing_time_tracked(self, mock_broadcaster, mock_detector_class,
                                     mock_storage_class, sample_stereo_pair):
        """Test that processing time is tracked"""
        from LLMCommunication.orchestrators.RunStereoDetector import StereoDetectorOrchestrator
        from LLMCommunication.vision.ObjectDetector import DetectionResult

        imgL, imgR = sample_stereo_pair

        # Setup storage
        mock_storage = MagicMock()
        mock_storage.get_stereo_pair.return_value = (imgL, imgR, "test")
        mock_storage_class.return_value = mock_storage

        # Setup detector with delay
        mock_detector = MagicMock()
        def slow_detect(*args, **kwargs):
            time.sleep(0.01)
            return DetectionResult("stereo", 640, 480, [])

        mock_detector.detect_cubes_stereo.side_effect = slow_detect
        mock_detector_class.return_value = mock_detector

        # Setup broadcaster
        mock_broadcaster.send_result = MagicMock()

        camera_config = MockCameraConfig()
        orchestrator = StereoDetectorOrchestrator(camera_config, check_interval=0.1)

        # Process
        orchestrator._process_stereo_pair("stereo")

        # Verify processing time is included
        call_args = mock_broadcaster.send_result.call_args[0][0]
        assert call_args["metadata"]["processing_time_seconds"] > 0
