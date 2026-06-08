import os

from config.Servers import (
    COMMAND_SERVER_PORT,
    DEFAULT_HOST,
    DEFAULT_LMSTUDIO_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TEMPERATURE,
    LLM_REQUEST_TIMEOUT,
    LMSTUDIO_BASE_URL,
    LOG_FORMAT,
    LOG_LEVEL,
    MAX_CLIENT_THREADS,
    MAX_CONNECTIONS_BACKLOG,
    MAX_IMAGE_SIZE,
    MAX_RESULT_QUEUE_SIZE,
    MAX_STRING_LENGTH,
    SEQUENCE_SERVER_PORT,
    SOCKET_ACCEPT_TIMEOUT,
    STEREO_DETECTION_PORT,
    VISION_MODELS,
    WORLDSTATE_CHECK_INTERVAL,
)
from config.Vision import (
    BLUE_HSV_LOWER,
    BLUE_HSV_UPPER,
    DEFAULT_STEREO_BASELINE,
    DEFAULT_STEREO_FOV,
    IMAGE_CHECK_INTERVAL,
    MAX_ASPECT_RATIO,
    MAX_CUBE_AREA_PX,
    MAX_IMAGE_AGE,
    MIN_ASPECT_RATIO,
    MIN_CONFIDENCE,
    MIN_CUBE_AREA_PX,
    MIN_IMAGE_AGE,
    RED_HSV_LOWER_1,
    RED_HSV_UPPER_1,
    VISION_OPERATION_TIMEOUT,
)


class TestConfigConstants:

    def test_network_config(self):
        assert DEFAULT_HOST == "127.0.0.1"
        assert STEREO_DETECTION_PORT == 5006
        assert COMMAND_SERVER_PORT == 5007
        assert SEQUENCE_SERVER_PORT == 5008
        assert MAX_CONNECTIONS_BACKLOG > 0
        assert MAX_CLIENT_THREADS > 0
        assert SOCKET_ACCEPT_TIMEOUT > 0

    def test_protocol_limits(self):
        assert MAX_STRING_LENGTH == 256
        assert MAX_IMAGE_SIZE == 10 * 1024 * 1024  # 10MB
        assert MAX_STRING_LENGTH > 0
        assert MAX_IMAGE_SIZE > 0

    def test_image_processing_config(self):
        assert MIN_IMAGE_AGE >= 0
        assert MAX_IMAGE_AGE > MIN_IMAGE_AGE
        assert IMAGE_CHECK_INTERVAL > 0
        assert LLM_REQUEST_TIMEOUT > 0
        assert WORLDSTATE_CHECK_INTERVAL > 0
        assert VISION_OPERATION_TIMEOUT > 0

    def test_llm_config(self):
        assert DEFAULT_LMSTUDIO_MODEL in VISION_MODELS
        assert DEFAULT_TEMPERATURE >= 0.0
        assert DEFAULT_TEMPERATURE <= 2.0
        assert len(VISION_MODELS) > 0
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

        assert url.startswith(
            "http"
        ), f"Default LMSTUDIO_BASE_URL must be an http URL, got: {url!r}"

    def test_queue_config(self):
        assert MAX_RESULT_QUEUE_SIZE > 0

    def test_logging_config(self):
        assert LOG_FORMAT is not None
        assert LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert DEFAULT_OUTPUT_DIR is not None

    def test_detection_config(self):
        assert len(RED_HSV_LOWER_1) == 3
        assert len(RED_HSV_UPPER_1) == 3
        assert len(BLUE_HSV_LOWER) == 3
        assert len(BLUE_HSV_UPPER) == 3

        assert MIN_CUBE_AREA_PX > 0
        assert MAX_CUBE_AREA_PX > MIN_CUBE_AREA_PX
        assert 0 < MIN_ASPECT_RATIO < MAX_ASPECT_RATIO
        assert 0 <= MIN_CONFIDENCE <= 1.0

    def test_stereo_config(self):
        assert DEFAULT_STEREO_BASELINE > 0
        assert DEFAULT_STEREO_FOV > 0
        assert DEFAULT_STEREO_FOV < 180  # FOV should be reasonable


class TestConfigModuleStructure:

    def test_servers_module_imports(self):
        from config import Servers

        assert hasattr(Servers, "DEFAULT_HOST")
        assert hasattr(Servers, "STEREO_DETECTION_PORT")
        assert hasattr(Servers, "SEQUENCE_SERVER_PORT")
        assert hasattr(Servers, "LMSTUDIO_BASE_URL")
        assert hasattr(Servers, "DEFAULT_LMSTUDIO_MODEL")

    def test_vision_module_imports(self):
        from config import Vision

        assert hasattr(Vision, "USE_YOLO")
        assert hasattr(Vision, "MIN_CUBE_AREA_PX")
        assert hasattr(Vision, "ENABLE_DEBUG_IMAGES")
        assert hasattr(Vision, "DEFAULT_STEREO_BASELINE")

    def test_rag_module_imports(self):
        from config import Rag

        assert hasattr(Rag, "RAG_LM_STUDIO_URL")
        assert hasattr(Rag, "RAG_EMBEDDING_DIMENSION")
        assert hasattr(Rag, "RAG_DEFAULT_TOP_K")

    def test_robot_module_imports(self):
        from config import Robot

        assert hasattr(Robot, "WORKSPACE_REGIONS")
        assert hasattr(Robot, "ROBOT_BASE_POSITIONS")
        assert hasattr(Robot, "MIN_ROBOT_SEPARATION")
