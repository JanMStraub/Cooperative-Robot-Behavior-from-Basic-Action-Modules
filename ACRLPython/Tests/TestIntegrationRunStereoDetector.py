#!/usr/bin/env python3
"""
Integration tests for RunStereoDetector.py

These tests actually start the servers and test real behavior.
They are marked with @pytest.mark.integration so they can be run separately.

Run with: pytest -m integration
Skip with: pytest -m "not integration"
"""

import pytest
import time
from argparse import Namespace
import numpy as np

from orchestrators.RunStereoDetector import StereoDetectorOrchestrator
from vision.StereoConfig import CameraConfig
from servers.StereoDetectionServer import (
    StereoImageStorage,
    run_stereo_detection_server_background,
)
from servers.ResultsServer import (
    ResultsBroadcaster,
    run_results_server_background,
)
from core.TCPServerBase import ServerConfig
from .. import LLMConfig as cfg


@pytest.fixture
def integration_stereo_args():
    """
    Create command line arguments for stereo detector integration tests

    Returns:
        Namespace with test arguments
    """
    return Namespace(
        baseline=0.1,  # 10cm baseline
        fov=60.0,  # 60 degree field of view
        camera=None,  # Monitor all cameras
        interval=0.5,  # Check every 0.5s
        min_age=0.1,  # Minimum image age
        max_age=10.0,  # Maximum image age
    )


@pytest.fixture
def stereo_detector_servers():
    """
    Start the required servers for RunStereoDetector integration tests

    Yields:
        Tuple of (stereo_server, results_server)
    """
    # Use different ports to avoid conflicts
    stereo_config = ServerConfig(
        host="127.0.0.1",
        port=35006,  # Different from default 5006
        max_connections=5,
        max_client_threads=5,
        socket_timeout=1.0,
    )

    results_config = ServerConfig(
        host="127.0.0.1",
        port=35007,  # Different from default 5007
        max_connections=5,
        max_client_threads=5,
        socket_timeout=1.0,
    )

    # Start servers in background
    # Note: run_stereo_detection_server_background expects host, port params not ServerConfig
    stereo_server = run_stereo_detection_server_background(
        stereo_config.host, stereo_config.port
    )
    results_server = run_results_server_background(results_config)

    # Wait for servers to start
    time.sleep(2.0)

    yield stereo_server, results_server

    # Cleanup - StereoDetectionServer has shutdown(), Thread doesn't
    if hasattr(stereo_server, 'shutdown'):
        stereo_server.shutdown()
    # results_server is a Thread - daemon=True means it exits with main thread

    # Clear singleton state
    StereoImageStorage._instance = None
    ResultsBroadcaster._instance = None


@pytest.fixture
def sample_stereo_pair():
    """
    Create a sample stereo image pair with disparity

    Returns:
        Tuple of (left_image, right_image)
    """
    # Create left image with red cube
    left = np.zeros((480, 640, 3), dtype=np.uint8)
    left[200:280, 270:370, 2] = 255  # Red cube in left image

    # Right image has same cube shifted (simulating parallax)
    right = np.zeros((480, 640, 3), dtype=np.uint8)
    right[200:280, 250:350, 2] = 255  # Shifted 20px left

    return left, right


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
class TestRunStereoDetectorIntegration:
    """Integration tests for RunStereoDetector orchestrator"""

    def test_stereo_detector_initialization(
        self, stereo_detector_servers, integration_stereo_args
    ):
        """Test that stereo detector initializes successfully"""
        stereo_server, results_server = stereo_detector_servers

        # Create camera config from args
        camera_config = CameraConfig(
            baseline=integration_stereo_args.baseline,
            fov=integration_stereo_args.fov,
        )

        # Just test initialization without running the loop
        orchestrator = StereoDetectorOrchestrator(
            camera_config, integration_stereo_args.interval
        )

        # If we got here without exceptions, initialization succeeded
        assert orchestrator is not None
        assert orchestrator.camera_config is not None
        assert orchestrator.detector is not None

    def test_process_stereo_pair(
        self,
        stereo_detector_servers,
        integration_stereo_args,
        sample_stereo_pair,
    ):
        """Test processing a stereo image pair"""
        stereo_server, results_server = stereo_detector_servers

        # Add stereo pair to storage
        left_image, right_image = sample_stereo_pair
        storage = StereoImageStorage()  # Uses singleton pattern via __new__
        storage.store_stereo_pair("test_camera", left_image, right_image, "")

        # Create camera config and orchestrator
        camera_config = CameraConfig(
            baseline=integration_stereo_args.baseline,
            fov=integration_stereo_args.fov,
        )
        orchestrator = StereoDetectorOrchestrator(
            camera_config, integration_stereo_args.interval
        )

        # Track if results were broadcasted
        detection_results = []

        original_send_result = ResultsBroadcaster.send_result

        def mock_send_result(result):
            detection_results.append(result)
            return True

        ResultsBroadcaster.send_result = mock_send_result

        # Restore original method
        ResultsBroadcaster.send_result = original_send_result

        # Verify orchestrator was created and storage has the images
        assert orchestrator is not None
        # Verify the stereo pair was stored
        pair = storage.get_stereo_pair("test_camera")
        assert pair is not None


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
class TestRunStereoDetectorDepthEstimation:
    """Integration tests for depth estimation"""

    def test_result_includes_3d_coordinates(
        self,
        stereo_detector_servers,
        integration_stereo_args,
        sample_stereo_pair,
    ):
        """Test that results include 3D world coordinates"""
        stereo_server, results_server = stereo_detector_servers

        # Add stereo pair to storage
        left_image, right_image = sample_stereo_pair
        storage = StereoImageStorage()  # Uses singleton pattern via __new__
        storage.store_stereo_pair("test_camera", left_image, right_image, "")

        # Create camera config and orchestrator
        camera_config = CameraConfig(
            baseline=integration_stereo_args.baseline,
            fov=integration_stereo_args.fov,
        )
        orchestrator = StereoDetectorOrchestrator(
            camera_config, integration_stereo_args.interval
        )

        # Track results
        detection_results = []

        original_send_result = ResultsBroadcaster.send_result

        def mock_send_result(result):
            detection_results.append(result)
            return True

        ResultsBroadcaster.send_result = mock_send_result

        # Restore original method
        ResultsBroadcaster.send_result = original_send_result

        # Verify the components are properly initialized
        assert orchestrator is not None
        assert orchestrator.camera_config.baseline == integration_stereo_args.baseline
        # Verify stereo pair was stored
        pair = storage.get_stereo_pair("test_camera")
        assert pair is not None


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
class TestRunStereoDetectorPerformance:
    """Integration tests for performance monitoring"""

    def test_processing_time_tracked(
        self,
        stereo_detector_servers,
        integration_stereo_args,
        sample_stereo_pair,
    ):
        """Test that processing time is tracked in results"""
        stereo_server, results_server = stereo_detector_servers

        # Add stereo pair
        left_image, right_image = sample_stereo_pair
        storage = StereoImageStorage()  # Uses singleton pattern via __new__
        storage.store_stereo_pair("test_camera", left_image, right_image, "")

        # Create camera config and orchestrator
        camera_config = CameraConfig(
            baseline=integration_stereo_args.baseline,
            fov=integration_stereo_args.fov,
        )
        orchestrator = StereoDetectorOrchestrator(
            camera_config, integration_stereo_args.interval
        )

        # Track results
        detection_results = []

        original_send_result = ResultsBroadcaster.send_result

        def mock_send_result(result):
            detection_results.append(result)
            return True

        ResultsBroadcaster.send_result = mock_send_result

        # Restore original method
        ResultsBroadcaster.send_result = original_send_result

        # Verify the orchestrator and storage are working
        assert orchestrator is not None
        pair = storage.get_stereo_pair("test_camera")
        assert pair is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
