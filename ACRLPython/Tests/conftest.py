#!/usr/bin/env python3
"""
Shared pytest fixtures for LLMcommunication tests
"""

import pytest
import numpy as np
import socket
from unittest.mock import Mock, MagicMock
from pathlib import Path


@pytest.fixture
def mock_socket():
    """
    Create a mock socket object for testing network code

    Returns:
        Mock socket with common socket methods
    """
    sock = Mock(spec=socket.socket)
    sock.recv = Mock(return_value=b"test_data")
    sock.sendall = Mock(return_value=None)
    sock.close = Mock(return_value=None)
    sock.settimeout = Mock(return_value=None)
    sock.bind = Mock(return_value=None)
    sock.listen = Mock(return_value=None)
    sock.accept = Mock(return_value=(Mock(spec=socket.socket), ("127.0.0.1", 12345)))
    return sock


@pytest.fixture
def sample_image():
    """
    Create a sample RGB image for testing

    Returns:
        Numpy array representing a 640x480 RGB image
    """
    # Create a simple gradient image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :, 0] = np.linspace(0, 255, 640, dtype=np.uint8)  # Red gradient
    image[:, :, 1] = 128  # Constant green
    image[:, :, 2] = np.linspace(255, 0, 640, dtype=np.uint8)  # Blue gradient
    return image


@pytest.fixture
def sample_red_cube_image():
    """
    Create a test image with a red cube for object detection testing

    Returns:
        Numpy array with a red square in the center
    """
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a red cube (BGR format)
    image[200:280, 270:370, 2] = 255  # Red channel
    return image


@pytest.fixture
def sample_blue_cube_image():
    """
    Create a test image with a blue cube for object detection testing

    Returns:
        Numpy array with a blue square in the center
    """
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a blue cube (BGR format)
    image[200:280, 270:370, 0] = 255  # Blue channel
    return image


@pytest.fixture
def sample_stereo_pair():
    """
    Create a sample stereo image pair for depth estimation testing

    Returns:
        Tuple of (left_image, right_image)
    """
    # Create identical images for simplicity (real stereo would have disparity)
    left = np.zeros((480, 640, 3), dtype=np.uint8)
    left[200:280, 270:370, 2] = 255  # Red cube in left image

    # Right image has same cube slightly shifted (simulating parallax)
    right = np.zeros((480, 640, 3), dtype=np.uint8)
    right[200:280, 250:350, 2] = 255  # Shifted 20px left

    return left, right


@pytest.fixture
def server_config():
    """
    Create a test server configuration

    Returns:
        ServerConfig instance with test settings
    """
    from LLMCommunication.core.TCPServerBase import ServerConfig
    return ServerConfig(
        host="127.0.0.1",
        port=9999,  # Use non-standard port for testing
        max_connections=2,
        max_client_threads=2,
        socket_timeout=0.1  # Short timeout for tests
    )


@pytest.fixture
def detection_result_dict():
    """
    Create a sample detection result dictionary

    Returns:
        Dict representing a detection result
    """
    return {
        "success": True,
        "camera_id": "test_camera",
        "timestamp": "2025-01-01T12:00:00",
        "image_width": 640,
        "image_height": 480,
        "detections": [
            {
                "id": 0,
                "color": "red",
                "bbox_px": {"x": 270, "y": 200, "width": 100, "height": 80},
                "center_px": {"x": 320, "y": 240},
                "confidence": 0.95
            }
        ]
    }


@pytest.fixture
def llm_result_dict():
    """
    Create a sample LLM result dictionary

    Returns:
        Dict representing an LLM analysis result
    """
    return {
        "success": True,
        "response": "I see a red cube on the table.",
        "camera_id": "AR4Left",
        "timestamp": "2025-01-01T12:00:00",
        "metadata": {
            "model": "llama-3.2-vision",
            "duration_seconds": 2.5,
            "image_count": 1,
            "camera_ids": ["AR4Left"],
            "prompt": "What do you see?"
        }
    }


@pytest.fixture
def mock_lmstudio_client():
    """
    Create a mock LM Studio (OpenAI-compatible) client for testing

    Returns:
        Mock OpenAI client with mocked chat.completions.create method
    """
    client = MagicMock()

    # Mock models.list() for connection testing
    client.models.list = Mock(return_value=[])

    # Mock chat.completions.create() for vision API
    mock_choice = MagicMock()
    mock_choice.message.content = "This is a test response from the LLM."

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    client.chat.completions.create = Mock(return_value=mock_response)

    return client


@pytest.fixture
def cleanup_singletons():
    """
    Fixture to clean up singleton instances between tests

    Yields control to the test, then resets singletons
    """
    yield

    # Reset singleton instances
    try:
        from LLMCommunication.servers.StreamingServer import ImageStorage
        ImageStorage._instance = None
    except:
        pass

    try:
        from LLMCommunication.servers.ResultsServer import ResultsBroadcaster
        ResultsBroadcaster._instance = None
        ResultsBroadcaster._server = None
    except:
        pass

    try:
        from LLMCommunication.servers.DetectionServer import DetectionBroadcaster
        DetectionBroadcaster._instance = None
    except:
        pass

    try:
        from LLMCommunication.servers.StereoDetectionServer import StereoImageStorage
        StereoImageStorage._instance = None
    except:
        pass


@pytest.fixture
def temp_output_dir(tmp_path):
    """
    Create a temporary output directory for test files

    Args:
        tmp_path: Pytest's built-in temporary directory fixture

    Returns:
        Path to temporary output directory
    """
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir
