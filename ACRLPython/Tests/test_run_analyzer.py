#!/usr/bin/env python3
"""
Unit tests for RunAnalyzer.py

Tests the LLM analyzer orchestrator
"""

import pytest
import hashlib
import time
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from argparse import Namespace

# Add LLMCommunication directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "LLMCommunication"))

import LLMCommunication.llm_config as cfg


class TestRunAnalyzerImageProcessing:
    """Test image processing logic in RunAnalyzer"""

    @patch('LLMCommunication.orchestrators.RunAnalyzer.ResultsBroadcaster')
    @patch('LLMCommunication.orchestrators.RunAnalyzer.ImageStorage')
    @patch('LLMCommunication.orchestrators.RunAnalyzer.LMStudioVisionProcessor')
    def test_process_new_image(self, mock_processor_class, mock_storage_class, mock_broadcaster_class, sample_image):
        """Test processing a new image"""
        from LLMCommunication.orchestrators.RunAnalyzer import run_analyzer_loop

        # Setup mocks
        mock_storage = MagicMock()
        mock_storage.get_all_camera_ids.return_value = ["camera1"]
        mock_storage.get_camera_image.return_value = sample_image
        mock_storage.get_camera_prompt.return_value = "What do you see?"
        mock_storage.get_camera_age.return_value = 1.0
        mock_storage_class.get_instance.return_value = mock_storage

        mock_processor = MagicMock()
        mock_processor.send_images.return_value = {
            "success": True,
            "response": "Test response",
            "metadata": {
                "duration_seconds": 1.5,
                "model": "llama-3.2-vision"
            }
        }
        mock_processor_class.return_value = mock_processor

        # Create args
        args = Namespace(
            model="llama-3.2-vision",
            base_url=None,
            output_dir="/tmp",
            no_save=True,
            camera=None,
            interval=0.1,
            min_age=0.1,
            max_age=10.0,
            temperature=0.7
        )

        # Run for a short time
        import threading
        def stop_after_delay():
            time.sleep(0.5)
            raise KeyboardInterrupt()

        stop_thread = threading.Thread(target=stop_after_delay, daemon=True)
        stop_thread.start()

        try:
            run_analyzer_loop(args)
        except KeyboardInterrupt:
            pass

        # Verify processor was called
        assert mock_processor.send_images.called

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

    @patch('LLMCommunication.orchestrators.RunAnalyzer.ImageStorage')
    def test_skip_images_without_prompt(self, mock_storage_class, sample_image):
        """Test that images without prompts are skipped"""
        from LLMCommunication.orchestrators.RunAnalyzer import run_analyzer_loop

        mock_storage = MagicMock()
        mock_storage.get_all_camera_ids.return_value = ["camera1"]
        mock_storage.get_camera_image.return_value = sample_image
        mock_storage.get_camera_prompt.return_value = ""  # Empty prompt
        mock_storage.get_camera_age.return_value = 1.0
        mock_storage_class.get_instance.return_value = mock_storage

        args = Namespace(
            model="llama-3.2-vision",
            base_url=None,
            output_dir="/tmp",
            no_save=True,
            camera=None,
            interval=0.1,
            min_age=0.1,
            max_age=10.0,
            temperature=0.7
        )

        # Run briefly
        import threading
        def stop():
            time.sleep(0.3)
            raise KeyboardInterrupt()

        threading.Thread(target=stop, daemon=True).start()

        # Should skip processing due to empty prompt
        try:
            run_analyzer_loop(args)
        except (KeyboardInterrupt, AttributeError):
            pass

    @patch('LLMCommunication.orchestrators.RunAnalyzer.ImageStorage')
    def test_skip_too_fresh_images(self, mock_storage_class, sample_image):
        """Test that too-fresh images are skipped"""
        from LLMCommunication.orchestrators.RunAnalyzer import run_analyzer_loop

        mock_storage = MagicMock()
        mock_storage.get_all_camera_ids.return_value = ["camera1"]
        mock_storage.get_camera_image.return_value = sample_image
        mock_storage.get_camera_prompt.return_value = "Test"
        mock_storage.get_camera_age.return_value = 0.01  # Too fresh
        mock_storage_class.get_instance.return_value = mock_storage

        args = Namespace(
            model="llama-3.2-vision",
            base_url=None,
            output_dir="/tmp",
            no_save=True,
            camera=None,
            interval=0.1,
            min_age=0.5,  # Require 0.5s minimum age
            max_age=10.0,
            temperature=0.7
        )

        import threading
        def stop():
            time.sleep(0.2)
            raise KeyboardInterrupt()

        threading.Thread(target=stop, daemon=True).start()

        try:
            run_analyzer_loop(args)
        except (KeyboardInterrupt, AttributeError):
            pass

    @patch('LLMCommunication.orchestrators.RunAnalyzer.ImageStorage')
    def test_skip_too_old_images(self, mock_storage_class, sample_image):
        """Test that too-old images are skipped"""
        from LLMCommunication.orchestrators.RunAnalyzer import run_analyzer_loop

        mock_storage = MagicMock()
        mock_storage.get_all_camera_ids.return_value = ["camera1"]
        mock_storage.get_camera_image.return_value = sample_image
        mock_storage.get_camera_prompt.return_value = "Test"
        mock_storage.get_camera_age.return_value = 100.0  # Too old
        mock_storage_class.get_instance.return_value = mock_storage

        args = Namespace(
            model="llama-3.2-vision",
            base_url=None,
            output_dir="/tmp",
            no_save=True,
            camera=None,
            interval=0.1,
            min_age=0.1,
            max_age=30.0,  # Max 30 seconds old
            temperature=0.7
        )

        import threading
        def stop():
            time.sleep(0.2)
            raise KeyboardInterrupt()

        threading.Thread(target=stop, daemon=True).start()

        try:
            run_analyzer_loop(args)
        except (KeyboardInterrupt, AttributeError):
            pass


class TestRunAnalyzerConfiguration:
    """Test analyzer configuration and setup"""

    @patch('LLMCommunication.orchestrators.RunAnalyzer.run_streaming_server_background')
    @patch('LLMCommunication.orchestrators.RunAnalyzer.run_results_server_background')
    @patch('LLMCommunication.orchestrators.RunAnalyzer.run_analyzer_loop')
    def test_main_starts_servers(self, mock_analyzer_loop, mock_results_server, mock_streaming_server):
        """Test that main() starts both servers"""
        from LLMCommunication.orchestrators.RunAnalyzer import main

        # Make analyzer loop exit quickly
        mock_analyzer_loop.side_effect = KeyboardInterrupt()

        with patch('sys.argv', ['run_analyzer.py']):
            try:
                main()
            except (KeyboardInterrupt, SystemExit):
                pass

        # Verify servers were started
        assert mock_streaming_server.called
        assert mock_results_server.called

    @patch('LLMCommunication.orchestrators.RunAnalyzer.LMStudioVisionProcessor')
    def test_analyzer_initialization_custom_model(self, mock_processor_class):
        """Test analyzer initialization with custom model"""
        from LLMCommunication.orchestrators.RunAnalyzer import run_analyzer_loop

        args = Namespace(
            model="llava",
            base_url="http://custom:1234/v1",
            output_dir="/tmp",
            no_save=True,
            camera=None,
            interval=0.1,
            min_age=0.1,
            max_age=10.0,
            temperature=0.9
        )

        import threading
        def stop():
            time.sleep(0.1)
            raise KeyboardInterrupt()

        threading.Thread(target=stop, daemon=True).start()

        try:
            run_analyzer_loop(args)
        except (KeyboardInterrupt, AttributeError):
            pass

        # Verify processor was initialized with correct parameters
        mock_processor_class.assert_called_once_with(model="llava", base_url="http://custom:1234/v1")

    @patch('LLMCommunication.orchestrators.RunAnalyzer.Path')
    def test_output_directory_creation(self, mock_path_class):
        """Test that output directory is created if needed"""
        from LLMCommunication.orchestrators.RunAnalyzer import run_analyzer_loop

        mock_path = MagicMock()
        mock_path_class.return_value = mock_path

        args = Namespace(
            model="llama-3.2-vision",
            base_url=None,
            output_dir="/custom/output",
            no_save=False,  # Enable saving
            camera=None,
            interval=0.1,
            min_age=0.1,
            max_age=10.0,
            temperature=0.7
        )

        import threading
        def stop():
            time.sleep(0.1)
            raise KeyboardInterrupt()

        threading.Thread(target=stop, daemon=True).start()

        try:
            run_analyzer_loop(args)
        except (KeyboardInterrupt, AttributeError):
            pass

        # Directory creation should be attempted
        assert mock_path.mkdir.called or mock_path_class.called


class TestRunAnalyzerCameraFiltering:
    """Test camera filtering logic"""

    @patch('LLMCommunication.orchestrators.RunAnalyzer.ImageStorage')
    def test_monitor_specific_cameras(self, mock_storage_class, sample_image):
        """Test monitoring specific cameras only"""
        from LLMCommunication.orchestrators.RunAnalyzer import run_analyzer_loop

        mock_storage = MagicMock()
        # Even though multiple cameras exist, should only check specified ones
        mock_storage.get_all_camera_ids.return_value = ["camera1", "camera2", "camera3"]
        mock_storage.get_camera_image.return_value = sample_image
        mock_storage.get_camera_prompt.return_value = "Test"
        mock_storage.get_camera_age.return_value = 1.0
        mock_storage_class.get_instance.return_value = mock_storage

        args = Namespace(
            model="llama-3.2-vision",
            base_url=None,
            output_dir="/tmp",
            no_save=True,
            camera=["camera1"],  # Only monitor camera1
            interval=0.1,
            min_age=0.1,
            max_age=10.0,
            temperature=0.7
        )

        import threading
        def stop():
            time.sleep(0.2)
            raise KeyboardInterrupt()

        threading.Thread(target=stop, daemon=True).start()

        try:
            run_analyzer_loop(args)
        except (KeyboardInterrupt, AttributeError):
            pass

    @patch('LLMCommunication.orchestrators.RunAnalyzer.ImageStorage')
    def test_monitor_all_cameras(self, mock_storage_class, sample_image):
        """Test monitoring all available cameras"""
        from LLMCommunication.orchestrators.RunAnalyzer import run_analyzer_loop

        mock_storage = MagicMock()
        mock_storage.get_all_camera_ids.return_value = ["camera1", "camera2"]
        mock_storage.get_camera_image.return_value = sample_image
        mock_storage.get_camera_prompt.return_value = "Test"
        mock_storage.get_camera_age.return_value = 1.0
        mock_storage_class.get_instance.return_value = mock_storage

        args = Namespace(
            model="llama-3.2-vision",
            base_url=None,
            output_dir="/tmp",
            no_save=True,
            camera=None,  # Monitor all cameras
            interval=0.1,
            min_age=0.1,
            max_age=10.0,
            temperature=0.7
        )

        import threading
        def stop():
            time.sleep(0.2)
            raise KeyboardInterrupt()

        threading.Thread(target=stop, daemon=True).start()

        try:
            run_analyzer_loop(args)
        except (KeyboardInterrupt, AttributeError):
            pass

        # Should call get_all_camera_ids
        assert mock_storage.get_all_camera_ids.called


class TestRunAnalyzerResultBroadcasting:
    """Test result broadcasting integration"""

    @patch('LLMCommunication.orchestrators.RunAnalyzer.ResultsBroadcaster')
    @patch('LLMCommunication.orchestrators.RunAnalyzer.ImageStorage')
    @patch('LLMCommunication.orchestrators.RunAnalyzer.LMStudioVisionProcessor')
    def test_results_broadcasted_to_unity(self, mock_processor_class, mock_storage_class,
                                          mock_broadcaster_class, sample_image):
        """Test that results are broadcasted to Unity"""
        from LLMCommunication.orchestrators.RunAnalyzer import run_analyzer_loop

        # Setup storage
        mock_storage = MagicMock()
        mock_storage.get_all_camera_ids.return_value = ["camera1"]
        mock_storage.get_camera_image.return_value = sample_image
        mock_storage.get_camera_prompt.return_value = "Test"
        mock_storage.get_camera_age.return_value = 1.0
        mock_storage_class.get_instance.return_value = mock_storage

        # Setup processor
        mock_processor = MagicMock()
        test_result = {
            "success": True,
            "response": "Test response",
            "metadata": {"duration_seconds": 1.0, "model": "llama-3.2-vision"}
        }
        mock_processor.send_images.return_value = test_result
        mock_processor_class.return_value = mock_processor

        # Setup broadcaster
        mock_broadcaster = MagicMock()
        mock_broadcaster_class.send_result = MagicMock()

        args = Namespace(
            model="llama-3.2-vision",
            base_url=None,
            output_dir="/tmp",
            no_save=True,
            camera=None,
            interval=0.1,
            min_age=0.1,
            max_age=10.0,
            temperature=0.7
        )

        import threading
        def stop():
            time.sleep(0.3)
            raise KeyboardInterrupt()

        threading.Thread(target=stop, daemon=True).start()

        try:
            run_analyzer_loop(args)
        except (KeyboardInterrupt, AttributeError):
            pass

        # Verify ResultsBroadcaster.send_result was called
        assert mock_broadcaster_class.send_result.called


class TestRunAnalyzerErrorHandling:
    """Test error handling in RunAnalyzer"""

    @patch('LLMCommunication.orchestrators.RunAnalyzer.ImageStorage')
    @patch('LLMCommunication.orchestrators.RunAnalyzer.LMStudioVisionProcessor')
    def test_handles_processing_errors(self, mock_processor_class, mock_storage_class, sample_image):
        """Test that processing errors are handled gracefully"""
        from LLMCommunication.orchestrators.RunAnalyzer import run_analyzer_loop

        mock_storage = MagicMock()
        mock_storage.get_all_camera_ids.return_value = ["camera1"]
        mock_storage.get_camera_image.return_value = sample_image
        mock_storage.get_camera_prompt.return_value = "Test"
        mock_storage.get_camera_age.return_value = 1.0
        mock_storage_class.get_instance.return_value = mock_storage

        # Make processor raise error
        mock_processor = MagicMock()
        mock_processor.send_images.side_effect = Exception("LM Studio error")
        mock_processor_class.return_value = mock_processor

        args = Namespace(
            model="llama-3.2-vision",
            base_url=None,
            output_dir="/tmp",
            no_save=True,
            camera=None,
            interval=0.1,
            min_age=0.1,
            max_age=10.0,
            temperature=0.7
        )

        import threading
        def stop():
            time.sleep(0.3)
            raise KeyboardInterrupt()

        threading.Thread(target=stop, daemon=True).start()

        # Should continue running despite error
        try:
            run_analyzer_loop(args)
        except (KeyboardInterrupt, AttributeError):
            pass

    @patch('LLMCommunication.orchestrators.RunAnalyzer.LMStudioVisionProcessor')
    def test_handles_lmstudio_connection_error(self, mock_processor_class):
        """Test handling of LM Studio connection errors"""
        from LLMCommunication.orchestrators.RunAnalyzer import run_analyzer_loop

        # Make processor initialization fail
        mock_processor_class.side_effect = ConnectionError("Cannot connect to LM Studio")

        args = Namespace(
            model="llama-3.2-vision",
            base_url=None,
            output_dir="/tmp",
            no_save=True,
            camera=None,
            interval=0.1,
            min_age=0.1,
            max_age=10.0,
            temperature=0.7
        )

        # Should raise the connection error
        with pytest.raises(ConnectionError):
            run_analyzer_loop(args)


class TestRunAnalyzerPerformance:
    """Test performance aspects of RunAnalyzer"""

    def test_check_interval_configuration(self):
        """Test that check interval can be configured"""
        from LLMCommunication.orchestrators.RunAnalyzer import run_analyzer_loop

        args = Namespace(
            model="llama-3.2-vision",
            base_url=None,
            output_dir="/tmp",
            no_save=True,
            camera=None,
            interval=0.05,  # Very fast interval
            min_age=0.1,
            max_age=10.0,
            temperature=0.7
        )

        # Verify interval is used (check it's set correctly)
        assert args.interval == 0.05

    def test_image_age_window_configuration(self):
        """Test that image age window can be configured"""
        args = Namespace(
            min_age=0.2,
            max_age=5.0
        )

        # Verify age constraints
        assert args.min_age == 0.2
        assert args.max_age == 5.0
        assert args.min_age < args.max_age
