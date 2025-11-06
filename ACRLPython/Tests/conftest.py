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
        socket_timeout=0.1,  # Short timeout for tests
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
                "confidence": 0.95,
            }
        ],
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
            "prompt": "What do you see?",
        },
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
        ImageStorage._cameras = {}
    except:
        pass

    try:
        from LLMCommunication.servers.ResultsServer import ResultsBroadcaster

        ResultsBroadcaster._instance = None
        ResultsBroadcaster._server = None
        ResultsBroadcaster._result_queue = []
    except:
        pass

    try:
        from LLMCommunication.servers.DetectionServer import DetectionBroadcaster
        import queue
        import ACRLPython.LLMCommunication.LLMConfig as cfg

        DetectionBroadcaster._instance = None
        DetectionBroadcaster._clients = []
        DetectionBroadcaster._result_queue = queue.Queue(
            maxsize=cfg.MAX_RESULT_QUEUE_SIZE
        )
    except:
        pass

    try:
        from LLMCommunication.servers.StereoDetectionServer import StereoImageStorage

        StereoImageStorage._instance = None
        # Reset will be done through _init_storage when new instance is created
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


# ============================================================================
# RAG System Fixtures
# ============================================================================


@pytest.fixture
def mock_lmstudio_embeddings_client():
    """
    Create a mock LM Studio client for embedding generation

    Returns:
        Mock OpenAI client with mocked embeddings.create method
    """
    client = MagicMock()

    # Mock embeddings.create() for embedding API
    mock_embedding_data = MagicMock()
    mock_embedding_data.embedding = [0.1] * 768  # 768-dim embedding

    mock_response = MagicMock()
    mock_response.data = [mock_embedding_data]

    client.embeddings.create = Mock(return_value=mock_response)

    return client


@pytest.fixture
def sample_operation():
    """
    Create a sample BasicOperation for testing

    Returns:
        Mock BasicOperation instance
    """
    from LLMCommunication.operations.Base import OperationCategory, OperationComplexity

    op = Mock()
    op.operation_id = "test_op_001"
    op.name = "test_operation"
    op.category = Mock(value="navigation")
    op.complexity = Mock(value="basic")
    op.description = "A test operation"
    op.average_duration_ms = 1000.0
    op.success_rate = 0.95
    op.to_rag_document = Mock(return_value="Test operation RAG document")

    return op


@pytest.fixture
def mock_operation_registry(sample_operation):
    """
    Create a mock operation registry with sample operations

    Args:
        sample_operation: Sample operation fixture

    Returns:
        Mock OperationRegistry instance
    """
    registry = Mock()
    registry.get_all_operations = Mock(return_value=[sample_operation])
    registry.get_operation = Mock(return_value=sample_operation)

    return registry


@pytest.fixture
def temp_vector_store_path(tmp_path):
    """
    Create a temporary path for vector store persistence

    Args:
        tmp_path: Pytest's built-in temporary directory fixture

    Returns:
        Path to temporary vector store file
    """
    return tmp_path / "test_vector_store.pkl"


@pytest.fixture
def cleanup_rag_singletons():
    """
    Fixture to clean up RAG system singleton instances between tests

    Yields control to the test, then resets singletons
    """
    yield

    # Reset RAG singleton instances if they exist
    try:
        from LLMCommunication.operations.Registry import _global_registry

        _global_registry = None
    except:
        pass
