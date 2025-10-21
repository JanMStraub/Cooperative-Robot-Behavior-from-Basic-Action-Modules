#!/usr/bin/env python3
"""
config.py - Centralized configuration for LLM communication system

This module contains all configuration constants and defaults for the
StreamingServer, ResultsServer, and image analysis systems.
"""

# ===========================
# Network Configuration
# ===========================

# Server host (usually localhost for security)
DEFAULT_HOST = "127.0.0.1"

# Port assignments
STREAMING_SERVER_PORT = 5005  # Receives images from Unity
RESULTS_SERVER_PORT = 5006  # Sends LLM results to Unity

# Connection limits
MAX_CONNECTIONS_BACKLOG = 5  # Max pending connections in listen() queue
MAX_CLIENT_THREADS = 10  # Max concurrent client handler threads

# Timeout settings (seconds)
SOCKET_ACCEPT_TIMEOUT = 1.0  # Timeout for accept() to allow shutdown checks
SOCKET_RECEIVE_TIMEOUT = 30.0  # Timeout for receiving data (prevents hangs)


# ===========================
# Protocol Configuration
# ===========================

# Wire protocol limits
MAX_STRING_LENGTH = 256  # Max length for camera_id and prompt strings
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB max image size


# ===========================
# Image Processing
# ===========================

# Image age thresholds (seconds)
MIN_IMAGE_AGE = 0.5  # Minimum age before processing (avoid partial uploads)
MAX_IMAGE_AGE = 30.0  # Maximum age to consider "fresh"

# Monitoring intervals (seconds)
IMAGE_CHECK_INTERVAL = 1.0  # How often to check for new images
SERVER_INIT_WAIT_TIME = 2.0  # Wait time for servers to initialize

# Duplicate detection
DUPLICATE_TIME_THRESHOLD = 0.1  # Time threshold for detecting duplicate sends (seconds)


# ===========================
# LLM Configuration
# ===========================

# Default Ollama model
DEFAULT_OLLAMA_MODEL = "gemma3"

# LLM generation parameters
DEFAULT_TEMPERATURE = 0.7  # Sampling temperature (0.0-2.0)

# Popular vision models (for reference)
VISION_MODELS = [
    "llava",
    "llama3.2-vision",
    "gemma3",
]


# ===========================
# Queue Configuration
# ===========================

# ResultsBroadcaster queue settings
MAX_RESULT_QUEUE_SIZE = 100  # Max queued results when no clients connected


# ===========================
# Logging Configuration
# ===========================

# Output directories
DEFAULT_OUTPUT_DIR = (
    "ACRLPython/LLMCommunication/llm_responses"  # Where to save LLM responses
)

# Logging format
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_LEVEL = "INFO"  # Can be: DEBUG, INFO, WARNING, ERROR, CRITICAL


# ===========================
# Performance Configuration
# ===========================

# Thread monitoring
THREAD_CLEANUP_INTERVAL = 10.0  # How often to clean up completed threads (seconds)

# Keepalive intervals
RESULTS_SERVER_KEEPALIVE = 1.0  # Results server monitoring interval (seconds)
STREAMING_SERVER_MONITOR = 5.0  # Streaming server monitoring interval (seconds)


# ===========================
# Helper Functions
# ===========================


def get_server_config(
    port: int = STREAMING_SERVER_PORT,
    host: str = DEFAULT_HOST,
    max_connections: int = MAX_CONNECTIONS_BACKLOG,
    max_threads: int = MAX_CLIENT_THREADS,
    timeout: float = SOCKET_ACCEPT_TIMEOUT,
):
    """
    Get a ServerConfig dict with defaults.

    Args:
        port: Server port
        host: Server host
        max_connections: Max backlog for listen()
        max_threads: Max concurrent client threads
        timeout: Socket accept timeout

    Returns:
        Dictionary with server configuration
    """
    return {
        "host": host,
        "port": port,
        "max_connections": max_connections,
        "max_client_threads": max_threads,
        "socket_timeout": timeout,
    }


def get_streaming_config():
    """Get default configuration for StreamingServer"""
    return get_server_config(port=STREAMING_SERVER_PORT)


def get_results_config():
    """Get default configuration for ResultsServer"""
    return get_server_config(port=RESULTS_SERVER_PORT)
