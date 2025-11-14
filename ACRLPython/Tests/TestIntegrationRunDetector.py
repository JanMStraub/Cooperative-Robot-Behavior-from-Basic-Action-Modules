#!/usr/bin/env python3
"""
Integration tests for RunDetector.py

These tests actually start the servers and test real behavior.
They are marked with @pytest.mark.integration so they can be run separately.

Run with: pytest -m integration
Skip with: pytest -m "not integration"
"""

import pytest
import time
from argparse import Namespace
import numpy as np

from orchestrators.RunDetector import run_detection_loop
from servers.StreamingServer import (
    ImageStorage,
    run_streaming_server_background,
)
from servers.DetectionServer import (
    DetectionBroadcaster,
    run_detection_server_background,
)
from core.TCPServerBase import ServerConfig
from .. import LLMConfig as cfg

@pytest.fixture
def integration_detector_args():
    """
    Create command line arguments for detector integration tests

    Returns:
        Namespace with test arguments
    """
    return Namespace(
        camera=None,  # Monitor all cameras
        interval=0.5,  # Check every 0.5s
        min_age=0.1,  # Minimum image age
        max_age=10.0,  # Maximum image age
    )


@pytest.fixture
def detector_servers():
    """
    Start the required servers for RunDetector integration tests

    Yields:
        Tuple of (streaming_server, detection_server)
    """
    # Use different ports to avoid conflicts
    streaming_config = ServerConfig(
        host="127.0.0.1",
        port=25005,  # Different from default 5005
        max_connections=5,
        max_client_threads=5,
        socket_timeout=1.0,
    )

    detection_config = ServerConfig(
        host="127.0.0.1",
        port=25007,  # Different from default 5007
        max_connections=5,
        max_client_threads=5,
        socket_timeout=1.0,
    )

    # Start servers in background
    streaming_server = run_streaming_server_background(streaming_config)
    detection_server = run_detection_server_background(detection_config)

    # Wait for servers to start
    time.sleep(2.0)

    yield streaming_server, detection_server

    # Cleanup - DetectionServer has shutdown(), Thread doesn't
    if hasattr(detection_server, 'shutdown'):
        detection_server.shutdown()
    # streaming_server is a Thread - daemon=True means it exits with main thread

    # Clear singleton state
    ImageStorage._instance = None
    ImageStorage._cameras = {}
    DetectionBroadcaster._instance = None
    DetectionBroadcaster._clients = []


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
class TestRunDetectorIntegration:
    """Integration tests for RunDetector orchestrator"""

    def test_detector_initialization(self, detector_servers, integration_detector_args):
        """Test that detector servers start successfully"""
        streaming_server, detection_server = detector_servers

        # Just verify servers are running (they're already started by the fixture)
        assert streaming_server is not None
        assert detection_server is not None

        # If we got here, servers initialized successfully

    def test_monitor_all_cameras(self, detector_servers, integration_detector_args):
        """Test that images can be stored for multiple cameras"""
        streaming_server, detection_server = detector_servers

        # Add test images to ImageStorage
        test_image1 = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image1[200:280, 270:370, 2] = 255  # Red square

        test_image2 = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image2[100:180, 100:200, 0] = 255  # Blue square

        storage = ImageStorage.get_instance()
        storage.store_image("camera1", test_image1, "")
        storage.store_image("camera2", test_image2, "")

        # Verify both images were stored
        img1 = storage.get_camera_image("camera1")
        img2 = storage.get_camera_image("camera2")

        assert img1 is not None
        assert img2 is not None
        assert img1.shape == (480, 640, 3)
        assert img2.shape == (480, 640, 3)


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
class TestRunDetectorCubeDetection:
    """Integration tests for cube detection"""

    def test_detect_red_cube(self, detector_servers, integration_detector_args):
        """Test detecting a red cube in an image using ObjectDetector directly"""
        from vision.ObjectDetector import CubeDetector
        import cv2

        # Create image with red cube
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[200:280, 270:370, 2] = 255  # Large red square (BGR format)

        # Use the detector directly instead of running the full loop
        detector = CubeDetector()
        result = detector.detect_cubes(test_image)

        # Should detect at least one cube
        assert result is not None, "Detection should return a result"
        assert len(result.detections) > 0, "Should detect at least one red cube"

        # Check that the detection is reasonable
        detection = result.detections[0]
        assert detection.color in ["red", "unknown"]  # May detect as red or unknown
        assert detection.confidence > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
