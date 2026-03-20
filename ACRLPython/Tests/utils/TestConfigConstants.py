#!/usr/bin/env python3
"""
Test Configuration Constants
=============================

Centralized configuration constants for test suite.
Eliminates magic numbers and makes test behavior explicit.

Usage:
    from tests.TestConfigConstants import THREAD_START_DELAY_MS, DEFAULT_TIMEOUT_MS

    time.sleep(THREAD_START_DELAY_MS / 1000.0)
    result = operation(timeout=DEFAULT_TIMEOUT_MS)
"""

# ============================================================================
# Timing Constants
# ============================================================================

# Thread synchronization delays
THREAD_START_DELAY_MS = 100  # Time to wait for threads to start (milliseconds)
THREAD_SYNC_DELAY_MS = 10  # Delay for thread synchronization (milliseconds)
THREAD_CONTENTION_DELAY_MS = 1  # Minimal delay for contention tests (milliseconds)

# Operation timeouts
DEFAULT_TIMEOUT_MS = 500  # Default operation timeout (milliseconds)
LONG_TIMEOUT_MS = 2000  # Timeout for slow operations (milliseconds)
VERY_LONG_TIMEOUT_MS = 10000  # Timeout for integration tests (milliseconds)

# Cache and TTL settings
SHORT_TTL_SECONDS = 0.1  # Short TTL for expiration tests (seconds)
MEDIUM_TTL_SECONDS = 1.0  # Medium TTL for caching tests (seconds)
LONG_TTL_SECONDS = 10.0  # Long TTL for stable caching (seconds)

# ============================================================================
# Concurrency Constants
# ============================================================================

# Thread counts for concurrency tests
CONCURRENCY_THREAD_COUNT_SMALL = 5  # Small thread count for basic tests
CONCURRENCY_THREAD_COUNT_MEDIUM = 20  # Medium thread count for stress tests
CONCURRENCY_THREAD_COUNT_LARGE = 100  # Large thread count for extreme tests

# Iteration counts
STRESS_TEST_ITERATIONS = 1000  # Iterations for stress tests
PERFORMANCE_TEST_ITERATIONS = 10000  # Iterations for performance benchmarks

# ============================================================================
# Network Constants
# ============================================================================

# Server ports (should match production config)
IMAGE_SERVER_PORT = 5005
STEREO_SERVER_PORT = 5006
COMMAND_SERVER_PORT = 5007
SEQUENCE_SERVER_PORT = 5011
ROS_TCP_ENDPOINT_PORT = 10000

# Network timeouts
SOCKET_CONNECT_TIMEOUT_SEC = 1.0  # Socket connection timeout (seconds)
SOCKET_READ_TIMEOUT_SEC = 5.0  # Socket read timeout (seconds)
SERVER_STARTUP_WAIT_SEC = 0.5  # Wait for server to start (seconds)

# Message sizes
MAX_MESSAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB maximum message size
LARGE_MESSAGE_SIZE_BYTES = 5 * 1024  # 5KB large message threshold

# ============================================================================
# Spatial/Position Constants
# ============================================================================

# Distance thresholds
POSITION_TOLERANCE_MM = 1.0  # Position accuracy tolerance (millimeters)
POSITION_TOLERANCE_M = 0.001  # Position accuracy tolerance (meters)
POSITION_TOLERANCE_CM = 0.01  # Position accuracy tolerance (centimeters)

# Robot reach distances
MIN_ROBOT_REACH_M = 0.1  # Minimum robot reach (meters)
MAX_ROBOT_REACH_M = 0.6  # Maximum robot reach (meters)
SAFE_ROBOT_DISTANCE_M = 0.2  # Safe distance between robots (meters)

# Workspace dimensions
WORKSPACE_MIN_X = 0.0
WORKSPACE_MAX_X = 1.0
WORKSPACE_MIN_Y = 0.0
WORKSPACE_MAX_Y = 1.0
WORKSPACE_MIN_Z = 0.0
WORKSPACE_MAX_Z = 0.5

# ============================================================================
# Force/Grasp Constants
# ============================================================================

# Grasp force thresholds
MIN_GRASP_FORCE_N = 5.0  # Minimum grasp force (Newtons)
MAX_GRASP_FORCE_N = 50.0  # Maximum grasp force (Newtons)
TYPICAL_GRASP_FORCE_N = 8.5  # Typical grasp force (Newtons)
FORCE_BALANCE_TOLERANCE_N = 1.0  # Tolerance for force balance (Newtons)

# Contact durations
MIN_CONTACT_DURATION_MS = 100  # Minimum stable contact duration (milliseconds)
TYPICAL_CONTACT_DURATION_MS = 150  # Typical contact duration (milliseconds)

# Pre-grasp and retreat distances
DEFAULT_PRE_GRASP_DISTANCE_M = 0.1  # Default pre-grasp offset (meters)
DEFAULT_RETREAT_DISTANCE_M = 0.12  # Default retreat distance (meters)

# ============================================================================
# Performance Benchmarks
# ============================================================================

# Expected performance thresholds
MAX_INDEX_BUILD_TIME_100_OPS_SEC = 5.0  # Max time to index 100 operations (seconds)
MAX_INDEX_BUILD_TIME_1000_OPS_SEC = 30.0  # Max time to index 1000 operations (seconds)
MAX_QUERY_TIME_SEC = 1.0  # Max time for RAG query (seconds)
MAX_SPATIAL_CALC_TIME_100_OBJS_SEC = (
    0.01  # Max time for 100 distance calculations (seconds)
)

# Database/cache operations
MAX_DB_WRITE_TIME_MS = 10  # Max time for database write (milliseconds)
MAX_CACHE_LOOKUP_TIME_MS = 1  # Max time for cache lookup (milliseconds)

# ============================================================================
# Test Data Scales
# ============================================================================

# Object counts for scaling tests
SMALL_OBJECT_COUNT = 10  # Small number of objects
MEDIUM_OBJECT_COUNT = 100  # Medium number of objects
LARGE_OBJECT_COUNT = 1000  # Large number of objects

# Robot counts
SMALL_ROBOT_COUNT = 2  # Small robot fleet
MEDIUM_ROBOT_COUNT = 5  # Medium robot fleet
LARGE_ROBOT_COUNT = 15  # Large robot fleet

# Operation counts
SMALL_OPERATION_COUNT = 10  # Small number of operations
MEDIUM_OPERATION_COUNT = 100  # Medium number of operations
LARGE_OPERATION_COUNT = 1000  # Large number of operations

# ============================================================================
# Image/Vision Constants
# ============================================================================

# Image dimensions
MIN_IMAGE_WIDTH = 64
MAX_IMAGE_WIDTH = 4096
MIN_IMAGE_HEIGHT = 64
MAX_IMAGE_HEIGHT = 4096
TEST_IMAGE_WIDTH = 640  # Standard test image width
TEST_IMAGE_HEIGHT = 480  # Standard test image height

# Image processing
IMAGE_CHANNELS = 3  # RGB channels
IMAGE_DTYPE_BITS = 8  # 8-bit image depth

# ============================================================================
# Verification/Safety Constants
# ============================================================================

# Verification confidence scores
MIN_CONFIDENCE_SCORE = 0.0
HIGH_CONFIDENCE_SCORE = 0.85
MAX_CONFIDENCE_SCORE = 1.0

# Stability scores
UNSTABLE_GRASP_THRESHOLD = 0.5  # Below this = unstable
STABLE_GRASP_THRESHOLD = 0.8  # Above this = stable

# ============================================================================
# RAG/Embedding Constants
# ============================================================================

# Embedding dimensions
TYPICAL_EMBEDDING_DIM = 384  # Typical embedding dimension
ALTERNATIVE_EMBEDDING_DIM = 256  # Alternative embedding dimension

# RAG search parameters
DEFAULT_RAG_TOP_K = 3  # Default number of RAG results
MAX_RAG_TOP_K = 10  # Maximum RAG results
DEFAULT_RAG_MIN_SCORE = 0.5  # Minimum similarity score

# ============================================================================
# Workspace Allocation Constants
# ============================================================================

# Workspace timeouts
WORKSPACE_ALLOCATION_TIMEOUT_SEC = 2.0  # Timeout for workspace allocation
WORKSPACE_RELEASE_TIMEOUT_SEC = 1.0  # Timeout for workspace release

# Workspace counts
SMALL_WORKSPACE_COUNT = 3  # Small number of workspaces
MEDIUM_WORKSPACE_COUNT = 5  # Medium number of workspaces
LARGE_WORKSPACE_COUNT = 10  # Large number of workspaces

# ============================================================================
# Command Parsing Constants
# ============================================================================

# Retry settings
MAX_GRASP_RETRIES = 3  # Maximum grasp retry attempts
RETRY_DELAY_SEC = 0.5  # Delay between retries (seconds)

# Command queue sizes
SMALL_QUEUE_SIZE = 5  # Small command queue
MEDIUM_QUEUE_SIZE = 50  # Medium command queue
LARGE_QUEUE_SIZE = 500  # Large command queue

# ============================================================================
# Logging/Cleanup Constants
# ============================================================================

# Command retention
COMMAND_RETENTION_SEC = 300  # Keep completed commands for 5 minutes
OLD_COMMAND_THRESHOLD_SEC = 400  # Commands older than this are cleaned

# Log rotation
LOG_FILE_MAX_SIZE_MB = 10.0  # Max log file size (megabytes)
LOG_FILE_MAX_BACKUPS = 20  # Max number of backup log files
