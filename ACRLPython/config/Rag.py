"""
RAG System Configuration
==========================

Configuration for Retrieval-Augmented Generation system.
"""

import os
from pathlib import Path

# Get the parent directory (ACRLPython/)
_CONFIG_DIR = Path(__file__).parent.parent.absolute()

# ============================================================================
# LM Studio Connection for RAG
# ============================================================================

from config.Servers import LMSTUDIO_BASE_URL as _LMSTUDIO_BASE_URL

RAG_LM_STUDIO_URL = os.environ.get("RAG_LM_STUDIO_URL", _LMSTUDIO_BASE_URL)
RAG_LM_STUDIO_MODEL = os.environ.get("RAG_LM_STUDIO_MODEL", "nomic-embed-text")
RAG_LM_STUDIO_API_KEY = os.environ.get("RAG_LM_STUDIO_API_KEY", "lm-studio")

# ============================================================================
# Embedding Settings
# ============================================================================

RAG_EMBEDDING_DIMENSION = int(os.environ.get("RAG_EMBEDDING_DIMENSION", "768"))
RAG_EMBEDDING_BATCH_SIZE = int(os.environ.get("RAG_EMBEDDING_BATCH_SIZE", "10"))
RAG_EMBEDDING_TIMEOUT = int(os.environ.get("RAG_EMBEDDING_TIMEOUT", "30"))

# ============================================================================
# Vector Store Settings
# ============================================================================

RAG_VECTOR_STORE_PATH = os.environ.get(
    "RAG_VECTOR_STORE_PATH",
    str(_CONFIG_DIR / "rag" / ".rag_index.pkl")
)
RAG_AUTO_SAVE_INDEX = os.environ.get("RAG_AUTO_SAVE_INDEX", "true").lower() in ("true", "1", "yes")

# ============================================================================
# Search Settings
# ============================================================================

RAG_DEFAULT_TOP_K = int(os.environ.get("RAG_DEFAULT_TOP_K", "5"))
RAG_MIN_SIMILARITY_SCORE = float(os.environ.get("RAG_MIN_SIMILARITY_SCORE", "0.5"))

# ============================================================================
# Confidence Scoring
# ============================================================================

RAG_CONFIDENCE_STRATEGY = os.environ.get("RAG_CONFIDENCE_STRATEGY", "balanced")  # strict, balanced, permissive
RAG_ENABLE_CONFIDENCE_SCORING = os.environ.get("RAG_ENABLE_CONFIDENCE_SCORING", "true").lower() in ("true", "1", "yes")
RAG_CONFIDENCE_TIERS = {
    "high": 0.75,
    "medium": 0.5,
    "low": 0.25,
}

# ============================================================================
# Fallback Settings
# ============================================================================

RAG_USE_TFIDF_FALLBACK = os.environ.get("RAG_USE_TFIDF_FALLBACK", "true").lower() in ("true", "1", "yes")
RAG_TFIDF_MAX_FEATURES = int(os.environ.get("RAG_TFIDF_MAX_FEATURES", "500"))
