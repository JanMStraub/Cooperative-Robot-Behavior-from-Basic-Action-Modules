"""
Multi-Robot Negotiation Configuration
=======================================

Configuration constants for the LLM-based multi-robot negotiation system.
Controls negotiation behavior, timeouts, and collaboration detection.
"""

import os

# ============================================================================
# Negotiation System Toggle
# ============================================================================

NEGOTIATION_ENABLED = os.environ.get("NEGOTIATION_ENABLED", "false").lower() in ("true", "1", "yes")

# ============================================================================
# Negotiation Protocol Parameters
# ============================================================================

MAX_NEGOTIATION_ROUNDS = int(os.environ.get("MAX_NEGOTIATION_ROUNDS", "3"))
NEGOTIATION_TIMEOUT = float(os.environ.get("NEGOTIATION_TIMEOUT", "120.0"))  # seconds

# ============================================================================
# LLM Agent Parameters
# ============================================================================

AGENT_LLM_TIMEOUT = float(os.environ.get("AGENT_LLM_TIMEOUT", "30.0"))  # seconds per LLM call
NEGOTIATION_TEMPERATURE = float(os.environ.get("NEGOTIATION_TEMPERATURE", "0.3"))

# ============================================================================
# Collaboration Detection
# ============================================================================

COLLABORATION_KEYWORDS = [
    "both", "together", "cooperate", "collaborate", "coordinate",
    "simultaneously", "handoff", "hand off", "pass to", "transfer",
    "help each other", "work together", "dual", "two robots",
    "both robots", "all robots", "jointly", "synchronize",
]

# ============================================================================
# LLM Structured Output
# ============================================================================

# When True, passes response_format={"type": "json_object"} to LM Studio, which
# forces the model to emit valid JSON directly (no prose or Markdown wrapping).
# Set to False for models that don't support the structured output API.
USE_STRUCTURED_OUTPUT = os.environ.get("USE_STRUCTURED_OUTPUT", "false").lower() in ("true", "1", "yes")

# ============================================================================
# Plan Validation
# ============================================================================

VERIFY_NEGOTIATED_PLANS = os.environ.get("VERIFY_NEGOTIATED_PLANS", "true").lower() in ("true", "1", "yes")
MAX_PLAN_LENGTH = int(os.environ.get("MAX_PLAN_LENGTH", "50"))
