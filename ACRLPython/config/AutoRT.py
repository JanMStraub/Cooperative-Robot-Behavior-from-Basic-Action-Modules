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
MAX_TASK_CANDIDATES = int(os.environ.get("AUTORT_MAX_TASKS", "5"))
LOOP_DELAY_SECONDS = float(os.environ.get("AUTORT_LOOP_DELAY", "5.0"))
HUMAN_IN_LOOP_DEFAULT = os.environ.get("AUTORT_HUMAN_IN_LOOP", "true").lower() == "true"
USE_VLM_REASONING = os.environ.get("AUTORT_USE_VLM", "false").lower() == "true"

# Safety settings
WORKSPACE_BOUNDS = {
    'min_corner': (-1.0, -1.0, 0.0),
    'max_corner': (1.0, 1.0, 1.5),
}
MAX_VELOCITY = 2.0       # m/s
MIN_ROBOT_SEPARATION = 0.2  # meters
MAX_GRIPPER_FORCE = 50.0    # Newtons

# Multi-robot settings
DEFAULT_ROBOTS = ["Robot1", "Robot2"]
ENABLE_COLLABORATIVE_TASKS = True

# JSON parsing
MAX_JSON_RETRIES = 3
