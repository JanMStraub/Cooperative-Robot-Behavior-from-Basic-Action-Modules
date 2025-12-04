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
STREAMING_SERVER_PORT = 5005  # Receives images from Unity (RunAnalyzer)
STEREO_DETECTION_PORT = 5006  # Receives stereo image pairs from Unity (RunStereoDetector)
LLM_RESULTS_PORT = 5010  # Sends LLM analysis results to Unity (RunAnalyzer)
DEPTH_RESULTS_PORT = 5007  # Sends depth detection results with 3D coordinates to Unity (RunStereoDetector)
RAG_SERVER_PORT = 5011  # RAG semantic search server for operation queries
STATUS_SERVER_PORT = 5012  # Status query server for robot status information
SEQUENCE_SERVER_PORT = 5013  # Sequence server for multi-command execution

# Legacy port names for backward compatibility
RESULTS_SERVER_PORT = LLM_RESULTS_PORT  # Default for RunAnalyzer
DETECTION_SERVER_PORT = DEPTH_RESULTS_PORT  # Default for RunStereoDetector


# Connection limits
MAX_CONNECTIONS_BACKLOG = 5  # Max pending connections in listen() queue
MAX_CLIENT_THREADS = 10  # Max concurrent client handler threads

# Timeout settings (seconds)
SOCKET_ACCEPT_TIMEOUT = 1.0  # Timeout for accept() to allow shutdown checks
SOCKET_RECEIVE_TIMEOUT = 300.0  # Timeout for idle connections (5 minutes)
# Note: Individual servers may override this for persistent connections


# ===========================
# Protocol Configuration
# ===========================

# Wire protocol limits
MAX_STRING_LENGTH = 256  # Max length for camera_id and prompt strings
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB max image size
MAX_METADATA_SIZE = 10 * 1024  # 10KB max metadata JSON size


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

# LM Studio server configuration
LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1" # local
# LMSTUDIO_BASE_URL = "http://192.168.178.53:1234" # GPU

# Default LM Studio model (use model name shown in LM Studio)
DEFAULT_LMSTUDIO_MODEL = "qwen3-vl-8b"

# LLM generation parameters
DEFAULT_TEMPERATURE = 0.2  # Sampling temperature (0.0-2.0)

# Popular vision models compatible with LM Studio (for reference)
VISION_MODELS = [
    "gemma-3-12b",
    "llama-3.2-vision",
    "llava",
    "qwen3-vl-8b",
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
DEFAULT_OUTPUT_DIR = "./llm_responses"  # Where to save LLM responses

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
STREAMING_SERVER_MONITOR = 60.0  # Streaming server monitoring interval (seconds)
RAG_SERVER_TIMEOUT = 60.0  # RAG server query timeout (seconds)


# ===========================
# Object Detection Configuration
# ===========================

# Color Ranges for Cube Detection (HSV)
# OpenCV HSV: H=0-180, S=0-255, V=0-255
# Red (RGB 255,0,0): H≈0-10 or H≈170-180 (wraps around)
# Allow some tolerance for lighting/shadows
RED_HSV_LOWER_1 = (0, 100, 100)  # Red with some tolerance for lighting
RED_HSV_UPPER_1 = (10, 255, 255)
RED_HSV_LOWER_2 = (170, 100, 100)  # Red wraparound with tolerance
RED_HSV_UPPER_2 = (180, 255, 255)

# Blue (RGB 0,0,255): H≈110-130 in OpenCV HSV space
# Allow some tolerance for lighting/shadows
BLUE_HSV_LOWER = (110, 100, 100)  # Blue with tolerance for lighting
BLUE_HSV_UPPER = (130, 255, 255)

# Detection Filters
MIN_CUBE_AREA_PX = 400  # Minimum bounding box area in pixels
MAX_CUBE_AREA_PX = 80000  # Maximum bounding box area in pixels
MIN_ASPECT_RATIO = 0.3  # Minimum width/height ratio (allow some perspective distortion)
MAX_ASPECT_RATIO = 3.5  # Maximum width/height ratio (allow elongated views)
MIN_CONFIDENCE = 0.3  # Minimum detection confidence threshold (lower to catch more)

# Detection Processing
DETECTION_CHECK_INTERVAL = 1.0  # Check for new images every N seconds
ENABLE_DEBUG_IMAGES = True  # Save annotated images for debugging
DEBUG_IMAGES_DIR = "./debug_detections"

# Disparity Map Debugging
SAVE_DEBUG_DISPARITY_MAPS = False  # Set to True to save disparity maps for debugging
DEBUG_DISPARITY_DIR = "./debug_detections"


# ===========================
# Stereo Reconstruction Configuration
# ===========================

# Default stereo camera parameters
DEFAULT_STEREO_BASELINE = 0.05  # meters, distance between stereo cameras
DEFAULT_STEREO_FOV = 60.0  # degrees, field of view of cameras

# Default stereo camera pose (must match Unity camera transform)
# Position: [x, y, z] in world space
# Rotation: [pitch, yaw, roll] in degrees (Unity convention)
DEFAULT_STEREO_CAMERA_POSITION = [-0.025, 0.1, -0.65]  # Example: camera above and behind origin
DEFAULT_STEREO_CAMERA_ROTATION = [0.0, 0.0, 0.0]  # Example: looking down at 30 degrees

# Stereo processing
STEREO_CHECK_INTERVAL = 0.5  # Check for new stereo pairs every N seconds


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
    Get a ServerConfig object with defaults.

    Args:
        port: Server port
        host: Server host
        max_connections: Max backlog for listen()
        max_threads: Max concurrent client threads
        timeout: Socket accept timeout

    Returns:
        ServerConfig object
    """
    from core.TCPServerBase import ServerConfig
    return ServerConfig(
        host=host,
        port=port,
        max_connections=max_connections,
        max_client_threads=max_threads,
        socket_timeout=timeout,
    )


def get_streaming_config():
    """Get default configuration for StreamingServer"""
    return get_server_config(port=STREAMING_SERVER_PORT)


def get_results_config():
    """Get default configuration for ResultsServer"""
    return get_server_config(port=RESULTS_SERVER_PORT)


def get_rag_config():
    """Get default configuration for RAGServer"""
    return get_server_config(port=RAG_SERVER_PORT)


def get_status_config():
    """Get default configuration for StatusServer"""
    return get_server_config(port=STATUS_SERVER_PORT)


def get_sequence_config():
    """Get default configuration for SequenceServer"""
    return get_server_config(port=SEQUENCE_SERVER_PORT)
