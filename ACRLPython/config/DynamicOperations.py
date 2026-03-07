"""
Dynamic Operation Generation Configuration
=============================================

Configuration for the opt-in dynamic operation learning feature.
When enabled, the system generates new operations via LLM when no
fitting operation is found in the RAG system.
"""

import os

# ============================================================================
# Feature Toggle
# ============================================================================

ENABLE_DYNAMIC_OPERATIONS = os.environ.get(
    "ENABLE_DYNAMIC_OPERATIONS", "true"
).lower() in ("true", "1", "yes")

# ============================================================================
# Generation Triggers
# ============================================================================

# Minimum RAG similarity score to consider a match adequate.
# Scores below this trigger generation of a new operation.
GENERATION_TRIGGER_THRESHOLD = float(
    os.environ.get("GENERATION_TRIGGER_THRESHOLD", "0.4")
)

# Maximum number of generated operations allowed (safety limit)
MAX_GENERATED_OPERATIONS = int(
    os.environ.get("MAX_GENERATED_OPERATIONS", "50")
)

# ============================================================================
# Review & Approval
# ============================================================================

# When True, generated operations require human approval before activation
REQUIRE_USER_REVIEW = os.environ.get(
    "REQUIRE_USER_REVIEW", "true"
).lower() in ("true", "1", "yes")

# ============================================================================
# RAG Integration
# ============================================================================

# Automatically rebuild the RAG index after generating a new operation
AUTO_REBUILD_INDEX = os.environ.get(
    "AUTO_REBUILD_INDEX", "true"
).lower() in ("true", "1", "yes")

# ============================================================================
# Safety: Restricted Modules
# ============================================================================

# Python modules that generated operations are NOT allowed to import
RESTRICTED_MODULES = [
    "os",
    "subprocess",
    "sys",
    "shutil",
    "pathlib",
    "importlib",
    "ctypes",
    "socket",
]

# Python builtins that generated operations are NOT allowed to use
RESTRICTED_BUILTINS = [
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "globals",
    "locals",
    "breakpoint",
]

# ============================================================================
# Sandbox Settings
# ============================================================================

# Timeout for sandbox execution of generated operations (seconds)
SANDBOX_TIMEOUT = float(os.environ.get("SANDBOX_TIMEOUT", "5.0"))

# ============================================================================
# Generated Operations Directory
# ============================================================================

from pathlib import Path

GENERATED_OPERATIONS_DIR = os.environ.get(
    "GENERATED_OPERATIONS_DIR",
    str(Path(__file__).parent.parent / "operations" / "generated"),
)
