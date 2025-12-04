#!/usr/bin/env python3
"""
Unit tests for config.py

Tests configuration constants and helper functions
"""

from .. import LLMConfig as cfg

class TestConfigConstants:
    """Test configuration constants have expected values"""

    def test_network_config(self):
        """Test network configuration constants"""
        assert cfg.DEFAULT_HOST == "127.0.0.1"
        # Legacy ports (kept for backward compatibility)
        assert cfg.STREAMING_SERVER_PORT == 5005
        assert cfg.STEREO_DETECTION_PORT == 5006
        assert cfg.DEPTH_RESULTS_PORT == 5007
        # Active ports
        assert cfg.LLM_RESULTS_PORT == 5010
        assert cfg.RAG_SERVER_PORT == 5011
        assert cfg.STATUS_SERVER_PORT == 5012
        assert cfg.SEQUENCE_SERVER_PORT == 5013
        # Aliases
        assert cfg.RESULTS_SERVER_PORT == cfg.LLM_RESULTS_PORT
        assert cfg.DETECTION_SERVER_PORT == cfg.DEPTH_RESULTS_PORT
        assert cfg.MAX_CONNECTIONS_BACKLOG > 0
        assert cfg.MAX_CLIENT_THREADS > 0
        assert cfg.SOCKET_ACCEPT_TIMEOUT > 0
        assert cfg.SOCKET_RECEIVE_TIMEOUT > 0

    def test_protocol_limits(self):
        """Test protocol limit constants"""
        assert cfg.MAX_STRING_LENGTH == 256
        assert cfg.MAX_IMAGE_SIZE == 10 * 1024 * 1024  # 10MB
        assert cfg.MAX_STRING_LENGTH > 0
        assert cfg.MAX_IMAGE_SIZE > 0

    def test_image_processing_config(self):
        """Test image processing configuration"""
        assert cfg.MIN_IMAGE_AGE >= 0
        assert cfg.MAX_IMAGE_AGE > cfg.MIN_IMAGE_AGE
        assert cfg.IMAGE_CHECK_INTERVAL > 0
        assert cfg.SERVER_INIT_WAIT_TIME > 0
        assert cfg.DUPLICATE_TIME_THRESHOLD >= 0

    def test_llm_config(self):
        """Test LLM configuration constants"""
        assert cfg.DEFAULT_LMSTUDIO_MODEL in cfg.VISION_MODELS
        assert cfg.DEFAULT_TEMPERATURE >= 0.0
        assert cfg.DEFAULT_TEMPERATURE <= 2.0
        assert len(cfg.VISION_MODELS) > 0

    def test_queue_config(self):
        """Test queue configuration"""
        assert cfg.MAX_RESULT_QUEUE_SIZE > 0

    def test_logging_config(self):
        """Test logging configuration"""
        assert cfg.LOG_FORMAT is not None
        assert cfg.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert cfg.DEFAULT_OUTPUT_DIR is not None

    def test_detection_config(self):
        """Test object detection configuration"""
        # Color ranges
        assert len(cfg.RED_HSV_LOWER_1) == 3
        assert len(cfg.RED_HSV_UPPER_1) == 3
        assert len(cfg.BLUE_HSV_LOWER) == 3
        assert len(cfg.BLUE_HSV_UPPER) == 3

        # Detection filters
        assert cfg.MIN_CUBE_AREA_PX > 0
        assert cfg.MAX_CUBE_AREA_PX > cfg.MIN_CUBE_AREA_PX
        assert 0 < cfg.MIN_ASPECT_RATIO < cfg.MAX_ASPECT_RATIO
        assert 0 <= cfg.MIN_CONFIDENCE <= 1.0

        # Processing intervals
        assert cfg.DETECTION_CHECK_INTERVAL > 0

    def test_stereo_config(self):
        """Test stereo reconstruction configuration"""
        assert cfg.DEFAULT_STEREO_BASELINE > 0
        assert cfg.DEFAULT_STEREO_FOV > 0
        assert cfg.DEFAULT_STEREO_FOV < 180  # FOV should be reasonable
        assert cfg.STEREO_CHECK_INTERVAL > 0


class TestConfigHelpers:
    """Test configuration helper functions"""

    def test_get_server_config_defaults(self):
        """Test get_server_config with default parameters"""
        config = cfg.get_server_config()

        assert config.host == cfg.DEFAULT_HOST
        assert config.port == cfg.STREAMING_SERVER_PORT
        assert config.max_connections == cfg.MAX_CONNECTIONS_BACKLOG
        assert config.max_client_threads == cfg.MAX_CLIENT_THREADS
        assert config.socket_timeout == cfg.SOCKET_ACCEPT_TIMEOUT

    def test_get_server_config_custom(self):
        """Test get_server_config with custom parameters"""
        custom_config = cfg.get_server_config(
            port=8888,
            host="192.168.1.1",
            max_connections=10,
            max_threads=20,
            timeout=5.0,
        )

        assert custom_config.host == "192.168.1.1"
        assert custom_config.port == 8888
        assert custom_config.max_connections == 10
        assert custom_config.max_client_threads == 20
        assert custom_config.socket_timeout == 5.0

    def test_get_streaming_config(self):
        """Test get_streaming_config returns correct port"""
        config = cfg.get_streaming_config()

        assert config.host == cfg.DEFAULT_HOST
        assert config.port == cfg.STREAMING_SERVER_PORT
        assert hasattr(config, "max_connections")
        assert hasattr(config, "max_client_threads")
        assert hasattr(config, "socket_timeout")

    def test_get_results_config(self):
        """Test get_results_config returns correct port"""
        config = cfg.get_results_config()

        assert config.host == cfg.DEFAULT_HOST
        assert config.port == cfg.RESULTS_SERVER_PORT
        assert hasattr(config, "max_connections")
        assert hasattr(config, "max_client_threads")
        assert hasattr(config, "socket_timeout")

    def test_get_rag_config(self):
        """Test get_rag_config returns correct port"""
        config = cfg.get_rag_config()

        assert config.host == cfg.DEFAULT_HOST
        assert config.port == cfg.RAG_SERVER_PORT
        assert hasattr(config, "max_connections")
        assert hasattr(config, "max_client_threads")
        assert hasattr(config, "socket_timeout")

    def test_get_status_config(self):
        """Test get_status_config returns correct port"""
        config = cfg.get_status_config()

        assert config.host == cfg.DEFAULT_HOST
        assert config.port == cfg.STATUS_SERVER_PORT
        assert hasattr(config, "max_connections")
        assert hasattr(config, "max_client_threads")
        assert hasattr(config, "socket_timeout")

    def test_get_sequence_config(self):
        """Test get_sequence_config returns correct port"""
        config = cfg.get_sequence_config()

        assert config.host == cfg.DEFAULT_HOST
        assert config.port == cfg.SEQUENCE_SERVER_PORT
        assert hasattr(config, "max_connections")
        assert hasattr(config, "max_client_threads")
        assert hasattr(config, "socket_timeout")

    def test_config_consistency(self):
        """Test that related config values are consistent"""
        # Streaming and results should use same host by default
        streaming = cfg.get_streaming_config()
        results = cfg.get_results_config()
        assert streaming.host == results.host

        # Ports should be different
        assert streaming.port != results.port

        # Timeouts should be positive
        assert streaming.socket_timeout > 0
        assert results.socket_timeout > 0
