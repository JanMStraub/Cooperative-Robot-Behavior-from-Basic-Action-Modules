"""
Server Network Configuration
=============================

Network, ports, and server-related configuration.
"""

import os

# ============================================================================
# Network Configuration
# ============================================================================

# Server host (usually localhost for security)
DEFAULT_HOST = os.environ.get("ACRL_HOST", "127.0.0.1")

# Port assignments (can be overridden via environment variables)
STREAMING_SERVER_PORT = int(os.environ.get("STREAMING_SERVER_PORT", "5005"))  # Receives images from Unity
STEREO_DETECTION_PORT = int(os.environ.get("STEREO_DETECTION_PORT", "5006"))  # Receives stereo image pairs
LLM_RESULTS_PORT = int(os.environ.get("LLM_RESULTS_PORT", "5010"))  # Sends LLM analysis results
DEPTH_RESULTS_PORT = int(os.environ.get("DEPTH_RESULTS_PORT", "5007"))  # Sends depth detection results
RAG_SERVER_PORT = int(os.environ.get("RAG_SERVER_PORT", "5011"))  # RAG semantic search server
STATUS_SERVER_PORT = int(os.environ.get("STATUS_SERVER_PORT", "5012"))  # Status query server
SEQUENCE_SERVER_PORT = int(os.environ.get("SEQUENCE_SERVER_PORT", "5013"))  # Sequence server
WORLD_STATE_PORT = int(os.environ.get("WORLD_STATE_PORT", "5014"))  # World state streaming (Unity → Python)
AUTORT_SERVER_PORT = int(os.environ.get("AUTORT_SERVER_PORT", "5015"))  # AutoRT task generation server

# Legacy port names for backward compatibility
RESULTS_SERVER_PORT = LLM_RESULTS_PORT
DETECTION_SERVER_PORT = DEPTH_RESULTS_PORT

# ============================================================================
# Connection Limits
# ============================================================================

MAX_CONNECTIONS_BACKLOG = int(os.environ.get("MAX_CONNECTIONS_BACKLOG", "5"))
MAX_CLIENT_THREADS = int(os.environ.get("MAX_CLIENT_THREADS", "10"))

# ============================================================================
# Timeout Settings (seconds)
# ============================================================================

SOCKET_ACCEPT_TIMEOUT = float(os.environ.get("SOCKET_ACCEPT_TIMEOUT", "1.0"))
SOCKET_RECEIVE_TIMEOUT = float(os.environ.get("SOCKET_RECEIVE_TIMEOUT", "300.0"))

# ============================================================================
# Protocol Limits
# ============================================================================

MAX_STRING_LENGTH = int(os.environ.get("MAX_STRING_LENGTH", "256"))
MAX_IMAGE_SIZE = int(os.environ.get("MAX_IMAGE_SIZE", str(10 * 1024 * 1024)))  # 10MB
MAX_METADATA_SIZE = int(os.environ.get("MAX_METADATA_SIZE", str(10 * 1024)))  # 10KB

# ============================================================================
# Result Queue
# ============================================================================

MAX_RESULT_QUEUE_SIZE = int(os.environ.get("MAX_RESULT_QUEUE_SIZE", "100"))

# ============================================================================
# Monitoring Intervals (seconds)
# ============================================================================

THREAD_CLEANUP_INTERVAL = float(os.environ.get("THREAD_CLEANUP_INTERVAL", "10.0"))
RESULTS_SERVER_KEEPALIVE = float(os.environ.get("RESULTS_SERVER_KEEPALIVE", "1.0"))
STREAMING_SERVER_MONITOR = float(os.environ.get("STREAMING_SERVER_MONITOR", "60.0"))
RAG_SERVER_TIMEOUT = float(os.environ.get("RAG_SERVER_TIMEOUT", "20.0"))
SERVER_INIT_WAIT_TIME = float(os.environ.get("SERVER_INIT_WAIT_TIME", "2.0"))

# ============================================================================
# LLM Configuration
# ============================================================================

LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://192.168.178.53:1234/v1")
DEFAULT_LMSTUDIO_MODEL = os.environ.get("DEFAULT_LMSTUDIO_MODEL", "ministral-3-14b-reasoning")
DEFAULT_TEMPERATURE = float(os.environ.get("DEFAULT_TEMPERATURE", "0.1"))

# Popular vision models (for reference)
VISION_MODELS = [
    "gemma-3-12b",
    "llama-3.2-vision",
    "qwen3-vl-8b",
    "mistral-3-3b",
    "ministral-3-14b-reasoning",
]

# ============================================================================
# Logging Configuration
# ============================================================================

from pathlib import Path

_CONFIG_DIR = Path(__file__).parent.parent.absolute()

DEFAULT_OUTPUT_DIR = os.environ.get("DEFAULT_OUTPUT_DIR", str(_CONFIG_DIR / "llm_responses"))
LOG_DIR = os.environ.get("LOG_DIR", str(_CONFIG_DIR / "logs"))

LOG_FORMAT = os.environ.get("LOG_FORMAT", "%(asctime)s [%(levelname)s] %(message)s")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

ENABLE_FILE_LOGGING = os.environ.get("ENABLE_FILE_LOGGING", "true").lower() in ("true", "1", "yes")
LOG_FILE_NAME = os.environ.get("LOG_FILE_NAME", "server_logs.txt")
LOG_FILE_MAX_BYTES = int(os.environ.get("LOG_FILE_MAX_BYTES", str(10 * 1024 * 1024)))
LOG_FILE_BACKUP_COUNT = int(os.environ.get("LOG_FILE_BACKUP_COUNT", "20"))
