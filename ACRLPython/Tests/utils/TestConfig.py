#!/usr/bin/env python3
"""
Unit tests for config modules

Tests configuration constants from the modular config system:
- config/Servers.py - Network, ports, LLM settings, logging
- config/Vision.py - Image processing, detection, stereo
- config/Rag.py - RAG system settings
- config/Robot.py - Multi-robot coordination
"""

import os

from config.Servers import (
    DEFAULT_HOST,
    STREAMING_SERVER_PORT,
    STEREO_DETECTION_PORT,
    DEPTH_RESULTS_PORT,
    LLM_RESULTS_PORT,
    RAG_SERVER_PORT,
    STATUS_SERVER_PORT,
    SEQUENCE_SERVER_PORT,
    MAX_CONNECTIONS_BACKLOG,
    MAX_CLIENT_THREADS,
    SOCKET_ACCEPT_TIMEOUT,
    MAX_STRING_LENGTH,
    MAX_IMAGE_SIZE,
    DEFAULT_LMSTUDIO_MODEL,
    DEFAULT_TEMPERATURE,
    VISION_MODELS,
    LMSTUDIO_BASE_URL,
    MAX_RESULT_QUEUE_SIZE,
    LOG_FORMAT,
    LOG_LEVEL,
    DEFAULT_OUTPUT_DIR,
    SERVER_INIT_WAIT_TIME,
    LLM_REQUEST_TIMEOUT,
    WORLDSTATE_CHECK_INTERVAL,
)

from config.Vision import (
    MIN_IMAGE_AGE,
    MAX_IMAGE_AGE,
    IMAGE_CHECK_INTERVAL,
    DUPLICATE_TIME_THRESHOLD,
    RED_HSV_LOWER_1,
    RED_HSV_UPPER_1,
    BLUE_HSV_LOWER,
    BLUE_HSV_UPPER,
    MIN_CUBE_AREA_PX,
    MAX_CUBE_AREA_PX,
    MIN_ASPECT_RATIO,
    MAX_ASPECT_RATIO,
    MIN_CONFIDENCE,
    DETECTION_CHECK_INTERVAL,
    DEFAULT_STEREO_BASELINE,
    DEFAULT_STEREO_FOV,
    STEREO_CHECK_INTERVAL,
    VISION_OPERATION_TIMEOUT,
)


class TestConfigConstants:
    """Test configuration constants have expected values"""

    def test_network_config(self):
        """Test network configuration constants"""
        assert DEFAULT_HOST == "127.0.0.1"
        assert STREAMING_SERVER_PORT == 5005
        assert STEREO_DETECTION_PORT == 5006
        assert DEPTH_RESULTS_PORT == 5007
        assert LLM_RESULTS_PORT == 5010
        assert RAG_SERVER_PORT == 5011
        assert STATUS_SERVER_PORT == 5012
        assert SEQUENCE_SERVER_PORT == 5013
        assert MAX_CONNECTIONS_BACKLOG > 0
        assert MAX_CLIENT_THREADS > 0
        assert SOCKET_ACCEPT_TIMEOUT > 0

    def test_protocol_limits(self):
        """Test protocol limit constants"""
        assert MAX_STRING_LENGTH == 256
        assert MAX_IMAGE_SIZE == 10 * 1024 * 1024  # 10MB
        assert MAX_STRING_LENGTH > 0
        assert MAX_IMAGE_SIZE > 0

    def test_image_processing_config(self):
        """Test image processing configuration"""
        assert MIN_IMAGE_AGE >= 0
        assert MAX_IMAGE_AGE > MIN_IMAGE_AGE
        assert IMAGE_CHECK_INTERVAL > 0
        assert SERVER_INIT_WAIT_TIME > 0
        assert DUPLICATE_TIME_THRESHOLD >= 0
        assert LLM_REQUEST_TIMEOUT > 0
        assert WORLDSTATE_CHECK_INTERVAL > 0
        assert VISION_OPERATION_TIMEOUT > 0

    def test_llm_config(self):
        """Test LLM configuration constants"""
        assert DEFAULT_LMSTUDIO_MODEL in VISION_MODELS
        assert DEFAULT_TEMPERATURE >= 0.0
        assert DEFAULT_TEMPERATURE <= 2.0
        assert len(VISION_MODELS) > 0
        # LMSTUDIO_BASE_URL should use environment variable or default to localhost
        assert LMSTUDIO_BASE_URL is not None
        assert isinstance(LMSTUDIO_BASE_URL, str)
        assert LMSTUDIO_BASE_URL.startswith("http")

    def test_lmstudio_url_default(self):
        """Default LMStudio URL must be a valid http URL.

        When no LMSTUDIO_BASE_URL env var is set the fallback is the
        project's LM Studio host (192.168.178.53).
        """
        import importlib
        from unittest.mock import patch

        with patch.dict("os.environ", {}, clear=True):
            # Remove LMSTUDIO_BASE_URL from env so the module uses the default
            os.environ.pop("LMSTUDIO_BASE_URL", None)
            # Reload the module to re-evaluate the module-level constant
            import config.Servers as servers_mod
            importlib.reload(servers_mod)
            url = servers_mod.LMSTUDIO_BASE_URL

        assert url.startswith("http"), (
            f"Default LMSTUDIO_BASE_URL must be an http URL, got: {url!r}"
        )

    def test_queue_config(self):
        """Test queue configuration"""
        assert MAX_RESULT_QUEUE_SIZE > 0

    def test_logging_config(self):
        """Test logging configuration"""
        assert LOG_FORMAT is not None
        assert LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert DEFAULT_OUTPUT_DIR is not None

    def test_detection_config(self):
        """Test object detection configuration"""
        # Color ranges
        assert len(RED_HSV_LOWER_1) == 3
        assert len(RED_HSV_UPPER_1) == 3
        assert len(BLUE_HSV_LOWER) == 3
        assert len(BLUE_HSV_UPPER) == 3

        # Detection filters
        assert MIN_CUBE_AREA_PX > 0
        assert MAX_CUBE_AREA_PX > MIN_CUBE_AREA_PX
        assert 0 < MIN_ASPECT_RATIO < MAX_ASPECT_RATIO
        assert 0 <= MIN_CONFIDENCE <= 1.0

        # Processing intervals
        assert DETECTION_CHECK_INTERVAL > 0

    def test_stereo_config(self):
        """Test stereo reconstruction configuration"""
        assert DEFAULT_STEREO_BASELINE > 0
        assert DEFAULT_STEREO_FOV > 0
        assert DEFAULT_STEREO_FOV < 180  # FOV should be reasonable
        assert STEREO_CHECK_INTERVAL > 0


class TestConfigModuleStructure:
    """Test that config modules are properly structured"""

    def test_servers_module_imports(self):
        """Test that Servers module exports expected constants"""
        from config import Servers

        # Network constants
        assert hasattr(Servers, 'DEFAULT_HOST')
        assert hasattr(Servers, 'STREAMING_SERVER_PORT')
        assert hasattr(Servers, 'SEQUENCE_SERVER_PORT')

        # LLM constants
        assert hasattr(Servers, 'LMSTUDIO_BASE_URL')
        assert hasattr(Servers, 'DEFAULT_LMSTUDIO_MODEL')

    def test_vision_module_imports(self):
        """Test that Vision module exports expected constants"""
        from config import Vision

        # Detection constants
        assert hasattr(Vision, 'USE_YOLO')
        assert hasattr(Vision, 'MIN_CUBE_AREA_PX')
        assert hasattr(Vision, 'ENABLE_DEBUG_IMAGES')

        # Stereo constants
        assert hasattr(Vision, 'DEFAULT_STEREO_BASELINE')

    def test_rag_module_imports(self):
        """Test that Rag module exports expected constants"""
        from config import Rag

        assert hasattr(Rag, 'RAG_LM_STUDIO_URL')
        assert hasattr(Rag, 'RAG_EMBEDDING_DIMENSION')
        assert hasattr(Rag, 'RAG_DEFAULT_TOP_K')

    def test_robot_module_imports(self):
        """Test that Robot module exports expected constants"""
        from config import Robot

        assert hasattr(Robot, 'WORKSPACE_REGIONS')
        assert hasattr(Robot, 'ROBOT_BASE_POSITIONS')
        assert hasattr(Robot, 'MIN_ROBOT_SEPARATION')
