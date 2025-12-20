#!/usr/bin/env python3
"""
config.py - Centralized configuration for LLM communication system

This module contains all configuration constants and defaults for the
StreamingServer, ResultsServer, and image analysis systems.
"""

from pathlib import Path

# Get the directory containing this config file
_CONFIG_DIR = Path(__file__).parent.absolute()

# ===========================
# Network Configuration
# ===========================

# Server host (usually localhost for security)
DEFAULT_HOST = "127.0.0.1"

# Port assignments
STREAMING_SERVER_PORT = 5005  # Receives images from Unity (RunAnalyzer)
STEREO_DETECTION_PORT = (
    5006  # Receives stereo image pairs from Unity (RunStereoDetector)
)
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
# LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"  # local
LMSTUDIO_BASE_URL = "http://192.168.178.53:1234/v1"  # GPU

# Default LM Studio model (use model name shown in LM Studio)
DEFAULT_LMSTUDIO_MODEL = "mistralai/ministral-3-14b-reasoning"

# LLM generation parameters
DEFAULT_TEMPERATURE = 0.0  # Sampling temperature (0.0-2.0)

# Popular vision models compatible with LM Studio (for reference)
VISION_MODELS = [
    "gemma-3-12b",
    "llama-3.2-vision",
    "llava",
    "qwen3-vl-8b",
    "mistral-3-3b",
    "mistralai/ministral-3-14b-reasoning",
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
DEFAULT_OUTPUT_DIR = str(_CONFIG_DIR / "llm_responses")  # Where to save LLM responses

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
RAG_SERVER_TIMEOUT = 90.0  # RAG server query timeout (seconds)


# ===========================
# Object Detection Configuration
# ===========================

# YOLO vs HSV Detection Toggle
USE_YOLO = (
    True  # Set to True to use YOLO detection, False for HSV color-based detection
)
YOLO_MODEL_PATH = str(
    _CONFIG_DIR / "yolo" / "models" / "robot_detector.onnx"
)  # Path to trained YOLO model for cubes
YOLO_CONFIDENCE_THRESHOLD = 0.5  # YOLO detection confidence threshold (0.0-1.0)
YOLO_IOU_THRESHOLD = 0.45  # YOLO IoU threshold for NMS (Non-Maximum Suppression)

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
DEBUG_IMAGES_DIR = str(_CONFIG_DIR / "debug_detections")

# Disparity Map Debugging
SAVE_DEBUG_DISPARITY_MAPS = True  # Set to True to save disparity maps for debugging
DEBUG_DISPARITY_DIR = str(_CONFIG_DIR / "debug_detections")


# ===========================
# Stereo Reconstruction Configuration
# ===========================

# Default stereo camera parameters
DEFAULT_STEREO_BASELINE = 0.05  # meters, distance between stereo cameras
DEFAULT_STEREO_FOV = 60.0  # degrees, field of view of cameras

# Default stereo camera pose (must match Unity camera transform)
# Position: [x, y, z] in world space
# Rotation: [pitch, yaw, roll] in degrees (Unity convention)
DEFAULT_STEREO_CAMERA_POSITION = [
    -0.025,
    0.1,
    -0.65,
]  # Example: camera above and behind origin
DEFAULT_STEREO_CAMERA_ROTATION = [0.0, 0.0, 0.0]  # Example: looking down at 30 degrees

# Stereo processing
STEREO_CHECK_INTERVAL = 0.5  # Check for new stereo pairs every N seconds


# ===========================
# Vision System Configuration (NEW)
# ===========================
# NOTE: Vision streaming settings only apply when RunRobotController starts
# Changing these requires restarting the Python backend

# ===== Streaming =====
ENABLE_VISION_STREAMING = True  # Master switch for continuous vision processing (default: False for backward compat)
VISION_STREAM_FPS = 5.0  # Streaming rate in frames per second (conservative default)
STEREO_JPEG_QUALITY = (
    75  # JPEG compression quality (reduced from 85 for faster encoding)
)

# ===== Depth Estimation =====
DEFAULT_SGBM_PRESET = (
    "medium"  # SGBM preset: "close" (<1m), "medium" (0.5-2m), "far" (>2m), "auto"
)
ENABLE_ADAPTIVE_SGBM = (
    False  # Auto-select SGBM preset based on estimated distance (default: False)
)
DEPTH_SAMPLING_STRATEGY = "median_inner_50pct"  # Depth sampling: "median_inner_50pct", "mean_valid", "max_disparity"
DEPTH_SAMPLE_INNER_PERCENT = (
    50  # Percentage of bbox to sample (default: 50 = inner 50%)
)

# ===== Stereo Validation =====
ENABLE_STEREO_VALIDATION = (
    True  # Validate detections in both L/R images (default: False for backward compat)
)
STEREO_MAX_Y_DIFF = 10  # Max Y coordinate difference for stereo matching (pixels)
STEREO_MAX_SIZE_RATIO = (
    0.3  # Max bbox size difference for stereo matching (fraction, 0.3 = 30%)
)
STEREO_MIN_IOU = 0.0  # Minimum IOU for stereo matching (0.0 = disabled)

# ===== Object Tracking =====
ENABLE_OBJECT_TRACKING = True  # Enable persistent object tracking across frames (default: False for backward compat)
TRACKING_MAX_AGE = 5  # Max frames a track survives without detection
TRACKING_MIN_IOU = 0.3  # Minimum IOU for track-detection association

# ===== YOLO Model =====
YOLO_MODEL_PATH = str(
    _CONFIG_DIR / "yolo" / "models" / "robot_detector.onnx"
)  # Path to ONNX detection model for vision streaming
YOLO_TASK = "detect"  # YOLO task type: "detect" (bounding boxes) or "segment" (segmentation masks)
YOLO_SEGMENTATION_MODEL = str(
    _CONFIG_DIR / "yolo" / "models" / "robot_detector_seg.onnx"
)
YOLO_INPUT_SIZE = None  # Downscale images before YOLO inference for speed (None = use original, e.g., (640, 480))

# ===== Multi-Robot Vision =====
SHARED_VISION_STATE_ENABLED = (
    True  # Enable shared vision state for multi-robot coordination (default: False)
)
OBJECT_CLAIM_TIMEOUT = (
    10.0  # Object claim timeout in seconds (auto-release stale claims)
)
CONFLICT_RESOLUTION_STRATEGY = (
    "closest_robot"  # Conflict resolution: "closest_robot", "first_claim"
)
CONFLICT_MIN_DISTANCE_DIFF = (
    0.05  # Min distance difference for "closest_robot" strategy (meters, 5cm)
)

# ===== Visualization =====
ENABLE_VISION_VISUALIZATION = (
    True  # Show live video window with YOLO detections (default: False)
)

# ===== Performance Optimizations =====
ENABLE_DISPARITY_CACHE = True  # Cache disparity maps for repeat detections (default: False for backward compat)
DISPARITY_CACHE_TTL = 0.5  # Disparity cache time-to-live in seconds
ENABLE_PARALLEL_JPEG_ENCODING = (
    True  # Enable parallel JPEG encoding in Unity (default: False for backward compat)
)


# ===========================
# Workspace and Coordination Configuration
# ===========================

# Workspace region definitions (meters, world coordinates)
WORKSPACE_REGIONS = {
    "left_workspace": {
        "x_min": -0.5,
        "x_max": -0.15,
        "y_min": 0.0,
        "y_max": 0.6,
        "z_min": -0.45,
        "z_max": 0.45,
    },
    "right_workspace": {
        "x_min": 0.15,
        "x_max": 0.5,
        "y_min": 0.0,
        "y_max": 0.6,
        "z_min": -0.45,
        "z_max": 0.45,
    },
    "shared_zone": {
        "x_min": -0.15,
        "x_max": 0.15,
        "y_min": 0.0,
        "y_max": 0.6,
        "z_min": -0.45,
        "z_max": 0.45,
    },
    "center": {
        "x_min": -0.15,
        "x_max": 0.15,
        "y_min": 0.0,
        "y_max": 0.5,
        "z_min": -0.1,
        "z_max": 0.1,
    },
}

# Robot workspace assignments (default allocation)
ROBOT_WORKSPACE_ASSIGNMENTS = {
    "Robot1": "left_workspace",
    "Robot2": "right_workspace",
}

# Robot base positions (world coordinates, meters)
ROBOT_BASE_POSITIONS = {
    "Robot1": (-0.4, 0.0, 0.0),
    "Robot2": (0.4, 0.0, 0.0),
}

# Multi-robot coordination safety parameters
COLLISION_SAFETY_MARGIN = 0.01  # Minimum safe distance between robots (meters)
MIN_ROBOT_SEPARATION = (
    0.2  # Minimum distance to maintain between robot end effectors (meters)
)
MAX_ROBOT_REACH = 0.8  # Maximum reach distance from base (meters)

# State caching configuration
ROBOT_STATUS_CACHE_TTL = 0.5  # Time-to-live for cached robot status (seconds)
WORLD_STATE_UPDATE_INTERVAL = 0.1  # How often to update world state (seconds)


# ===========================
# RAG System Configuration
# ===========================

# LM Studio connection settings for RAG
RAG_LM_STUDIO_URL = LMSTUDIO_BASE_URL  # Reuse main LM Studio URL
RAG_LM_STUDIO_MODEL = "nomic-embed-text"  # Embedding model (must match LM Studio)
RAG_LM_STUDIO_API_KEY = "lm-studio"  # LM Studio doesn't require real key

# Embedding settings
RAG_EMBEDDING_DIMENSION = 768  # Embedding vector dimension
RAG_EMBEDDING_BATCH_SIZE = 10  # Batch size for embedding generation
RAG_EMBEDDING_TIMEOUT = 30  # Timeout for embedding requests (seconds)

# Vector store settings
RAG_VECTOR_STORE_PATH = str(
    _CONFIG_DIR / "rag" / ".rag_index.pkl"
)  # Path to cached index
RAG_AUTO_SAVE_INDEX = True  # Automatically save index after building

# Search settings
RAG_DEFAULT_TOP_K = 5  # Default number of search results to return
RAG_MIN_SIMILARITY_SCORE = 0.5  # Minimum similarity score for search results (0.0-1.0)

# Confidence scoring settings
RAG_CONFIDENCE_STRATEGY = "balanced"  # Options: "strict", "balanced", "permissive"
RAG_ENABLE_CONFIDENCE_SCORING = True  # Enable confidence score computation
RAG_CONFIDENCE_TIERS = {
    "high": 0.75,
    "medium": 0.5,
    "low": 0.25,
}

# Fallback settings
RAG_USE_TFIDF_FALLBACK = True  # Use TF-IDF fallback when embeddings unavailable
RAG_TFIDF_MAX_FEATURES = 500  # Max features for TF-IDF vectorization


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
