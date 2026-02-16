"""
AutoRT configuration.

Extends the existing config pattern (config/Vision.py, config/Servers.py).
Uses environment variables for overrides where appropriate.
"""
import os
from config.Servers import LMSTUDIO_BASE_URL, DEFAULT_LMSTUDIO_MODEL

# LLM configuration (inherits from existing config/Servers.py)
LM_STUDIO_URL = os.environ.get("AUTORT_LM_STUDIO_URL", LMSTUDIO_BASE_URL)
TASK_GENERATION_MODEL = os.environ.get("AUTORT_TASK_MODEL", DEFAULT_LMSTUDIO_MODEL)
SAFETY_VALIDATION_MODEL = os.environ.get("AUTORT_SAFETY_MODEL", DEFAULT_LMSTUDIO_MODEL)

# AutoRT loop settings
MAX_TASK_CANDIDATES = int(os.environ.get("AUTORT_MAX_TASKS", "3"))
LOOP_DELAY_SECONDS = float(os.environ.get("AUTORT_LOOP_DELAY", "5.0"))
HUMAN_IN_LOOP_DEFAULT = os.environ.get("AUTORT_HUMAN_IN_LOOP", "true").lower() == "true"
USE_VLM_REASONING = os.environ.get("AUTORT_USE_VLM", "true").lower() == "true"

# Safety settings
ENABLE_SAFETY_VALIDATION = os.environ.get("AUTORT_ENABLE_SAFETY", "true").lower() == "false"
WORKSPACE_BOUNDS = {
    'min_corner': (-0.6, 0.0, -0.6),
    'max_corner': (0.6, 0.6, 0.6),
}
MAX_VELOCITY = 2.0       # m/s
MIN_ROBOT_SEPARATION = 0.2  # meters
MAX_GRIPPER_FORCE = 50.0    # Newtons

# Multi-robot settings
DEFAULT_ROBOTS = ["Robot1", "Robot2"]
ENABLE_COLLABORATIVE_TASKS = False

# Robot spatial layout (used for task generation context)
ROBOT_SPATIAL_LAYOUT = {
    "Robot1": {
        "position": "LEFT side of workspace",
        "x_range": "negative X coordinates (approximately -0.65 to -0.1)",
        "workspace_region": "Left_Workspace",
    },
    "Robot2": {
        "position": "RIGHT side of workspace",
        "x_range": "positive X coordinates (approximately 0.1 to 0.65)",
        "workspace_region": "Right_Workspace",
    },
}

# JSON parsing
MAX_JSON_RETRIES = 1

# Unity integration settings
UNITY_INTEGRATION_ENABLED = os.environ.get("AUTORT_UNITY_INTEGRATION", "true").lower() == "true"
TASK_CACHE_SIZE = int(os.environ.get("AUTORT_TASK_CACHE_SIZE", "50"))  # Max cached tasks
TASK_EXPIRATION_SECONDS = float(os.environ.get("AUTORT_TASK_EXPIRATION", "300"))  # 5 minutes
