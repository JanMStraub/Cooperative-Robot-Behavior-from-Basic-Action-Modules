"""
Memory System Configuration
============================

Controls the persistent Markdown-based memory system that records
operation outcomes per robot and injects cross-session context into
negotiation agent prompts (RobotLLMAgent).

Toggle with environment variable:
    MEMORY_ENABLED=true python -m orchestrators.RunRobotController
"""

import os

# Whether the memory system is active.
# When False: no files are written, no OutcomeTracker.record() calls from
# SequenceExecutor, no memory injected into RobotLLMAgent prompts.
MEMORY_ENABLED: bool = os.environ.get("MEMORY_ENABLED", "false").lower() in (
    "true",
    "1",
    "yes",
)

# Directory where per-robot memory files are stored.
# Files are named memory_{robot_id}.md (e.g. data/memory/memory_Robot1.md).
MEMORY_DIR: str = os.environ.get("MEMORY_DIR", "data/memory")

# Maximum number of lesson lines (operation stats) written to the Lessons section.
MEMORY_MAX_LESSONS: int = int(os.environ.get("MEMORY_MAX_LESSONS", "20"))

# Maximum number of recent error lines written to the Recent Errors section.
MEMORY_MAX_ERRORS: int = int(os.environ.get("MEMORY_MAX_ERRORS", "10"))
