#!/usr/bin/env python3
"""
Server Network Configuration
=============================

Network, ports, and server-related configuration.
"""

import os

# ============================================================================
# Network Configuration
# ============================================================================

DEFAULT_HOST = os.environ.get("ACRL_HOST", "127.0.0.1")

STEREO_DETECTION_PORT = int(
    os.environ.get("STEREO_DETECTION_PORT", "5006")
)  # Receives stereo image pairs
COMMAND_SERVER_PORT = int(
    os.environ.get("COMMAND_SERVER_PORT", "5007")
)  # Bidirectional command server (commands & results)
SEQUENCE_SERVER_PORT = int(
    os.environ.get("SEQUENCE_SERVER_PORT", "5008")
)  # Sequence server
WORLD_STATE_PORT = int(
    os.environ.get("WORLD_STATE_PORT", "5009")
)  # World state streaming (Unity → Python)
AUTORT_SERVER_PORT = int(
    os.environ.get("AUTORT_SERVER_PORT", "5010")
)  # AutoRT task generation server

# When True, WorldStateServer (port 5009) is NOT started even in sim mode.
# WorldState is populated entirely by FK (from joint angles) and stereo perception.
# Useful for testing real-robot code paths in the sim environment without Unity
# broadcasting ground-truth positions.
PERCEPTION_ONLY_MODE = os.environ.get("PERCEPTION_ONLY_MODE", "false").lower() in (
    "true",
    "1",
    "yes",
)

MAX_CONNECTIONS_BACKLOG = int(os.environ.get("MAX_CONNECTIONS_BACKLOG", "5"))
MAX_CLIENT_THREADS = int(os.environ.get("MAX_CLIENT_THREADS", "10"))

SOCKET_ACCEPT_TIMEOUT = float(os.environ.get("SOCKET_ACCEPT_TIMEOUT", "1.0"))
SERVER_HEARTBEAT_INTERVAL = float(os.environ.get("SERVER_HEARTBEAT_INTERVAL", "30.0"))

MAX_STRING_LENGTH = int(os.environ.get("MAX_STRING_LENGTH", "256"))
MAX_IMAGE_SIZE = int(os.environ.get("MAX_IMAGE_SIZE", str(10 * 1024 * 1024)))  # 10MB

MAX_RESULT_QUEUE_SIZE = int(os.environ.get("MAX_RESULT_QUEUE_SIZE", "100"))

LLM_REQUEST_TIMEOUT = float(os.environ.get("LLM_REQUEST_TIMEOUT", "90.0"))
WORLDSTATE_CHECK_INTERVAL = float(os.environ.get("WORLDSTATE_CHECK_INTERVAL", "5.0"))

# Reflection retry: re-parse failed commands with error context injected into the LLM prompt.
REFLECTION_ENABLED = os.environ.get("REFLECTION_ENABLED", "true").lower() in (
    "true",
    "1",
    "yes",
)
REFLECTION_MAX_RETRIES = int(os.environ.get("REFLECTION_MAX_RETRIES", "2"))
REFLECTION_SELF_REFLECT_ENABLED = REFLECTION_ENABLED and os.environ.get(
    "REFLECTION_SELF_REFLECT", "true"
).lower() in ("true", "1", "yes")

# Minimum contact force (N) to accept as a confirmed grasp when gripper_has_contact=False.
# After Unity attaches the object, parent-child collision disables force callbacks, so
# gripper_has_contact and gripper_contact_force both read 0. Set to 0.0 to pass on any
# force reading; raise for real-robot deployments where force feedback is reliable.
GRASP_VERIFY_MIN_FORCE = float(os.environ.get("GRASP_VERIFY_MIN_FORCE", "0.0"))

# Intermediate motion layer (RT-H style): when True, CommandParser sends the
# command to the LLM twice - Stage 1 decomposes to motion strings, Stage 2
# maps those to operations. Disabled by default to preserve existing behaviour.
USE_MOTION_LAYER = os.environ.get("PARSER_USE_MOTION_LAYER", "true").lower() in (
    "true",
    "1",
    "yes",
)

# ============================================================================
# LLM Configuration
# ============================================================================

_lmstudio_raw = os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234").rstrip("/")
LMSTUDIO_BASE_URL = (
    _lmstudio_raw if _lmstudio_raw.endswith("/v1") else _lmstudio_raw + "/v1"
)

DEFAULT_LMSTUDIO_MODEL = os.environ.get(
    "DEFAULT_LMSTUDIO_MODEL", "mistralai/magistral-small-2509"
)
DEFAULT_TEMPERATURE = float(os.environ.get("DEFAULT_TEMPERATURE", "0.3"))

# Maximum thinking tokens for reasoning models (e.g. ministral-3-14b-reasoning).
# LM Studio exposes this as `budget_tokens` inside the `thinking` block.
# Set to 0 to disable thinking entirely (fastest); increase for harder tasks.
# Has no effect on non-reasoning models.
LLM_THINKING_BUDGET = int(os.environ.get("LLM_THINKING_BUDGET", "8192"))

# Set to True to enable thinking for reasoning models (e.g. ministral-3-14b-reasoning).
LLM_THINKING_ENABLED = os.environ.get("LLM_THINKING_ENABLED", "true").lower() == "true"

# Maximum tokens for LLM responses. Must be large enough to cover thinking budget + JSON response.
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "16384"))

# ============================================================================
# VGN (Volumetric Grasp Network) - Local Mac Inference
# ============================================================================

# Path to the VGN checkpoint file (vgn_conv.pth).
# Relative paths are resolved from the ACRLPython/ root directory.
# Download from: https://github.com/ethz-asl/vgn (Google Drive link in README)
VGN_MODEL_PATH = os.environ.get("VGN_MODEL_PATH", "vgn/models/vgn_conv.pth")

VGN_TOP_K = int(os.environ.get("VGN_TOP_K", "20"))

# Master toggle: set to "false" to skip VGN entirely and use geometric fallback.
VGN_ENABLED = os.environ.get("VGN_ENABLED", "true").lower() in ("true", "1", "yes")

# When False (default), skip the LM Studio VLM bbox-refinement step inside
# VGNClient.predict_grasps() and use the raw YOLO bbox directly.  Set to "true"
# only when LM Studio is running; otherwise every grasp attempt will hang for
# the HTTP client default timeout (~30 s) before falling back.
VGN_USE_VLM_REFINEMENT = os.environ.get("VGN_USE_VLM_REFINEMENT", "false").lower() in (
    "true",
    "1",
    "yes",
)

# A concise domain preamble injected into every LLM call as the system message.
# Individual role-specific prompts (CommandParser, RobotLLMAgent, etc.) extend
# this with their own instructions - they should NOT repeat this context.
SYSTEM_PROMPT_BASE = "You are an AI robot controller for a dual-arm AR4 robotic system running inside a Unity simulation. The workspace is a tabletop environment with two 6-DOF robot arms: Robot1 (left side, base at x = -0.475) and Robot2 (right side, base at x = +0.475). Workspace bounds: x between -0.6 and 0.6, y between 0.0 and 0.6, z between -0.5 and 0.5. Operations are executed sequentially or in named parallel_groups. Robots communicate via signal/wait_for_signal events. You must ONLY use operations, object IDs, and coordinate values explicitly provided in the user message and never invent names, IDs, or positions. Output only valid JSON. Never include markdown fences, reasoning text, or [THINK] tags."

# Popular vision models (for reference)
VISION_MODELS = [
    "gemma-3-12b",
    "llama-3.2-vision",
    "qwen/qwen3-vl-4b",
    "mistralai/ministral-3-14b-reasoning",
    "mistralai/magistral-small-2509",
]

# ============================================================================
# Logging Configuration
# ============================================================================

from pathlib import Path

_CONFIG_DIR = Path(__file__).parent.parent.absolute()

DEFAULT_OUTPUT_DIR = os.environ.get(
    "DEFAULT_OUTPUT_DIR", str(_CONFIG_DIR / "llm_responses")
)
LOG_DIR = os.environ.get("LOG_DIR", str(_CONFIG_DIR / "logs"))

LOG_FORMAT = os.environ.get("LOG_FORMAT", "%(asctime)s [%(levelname)s] %(message)s")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

ENABLE_FILE_LOGGING = os.environ.get("ENABLE_FILE_LOGGING", "true").lower() in (
    "true",
    "1",
    "yes",
)
LOG_FILE_BACKUP_COUNT = int(os.environ.get("LOG_FILE_BACKUP_COUNT", "20"))
