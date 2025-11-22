"""
RAG System Configuration
=========================

Configuration settings for the RAG system using LM Studio for embeddings.
"""

import os
from typing import Optional


class RAGConfig:
    """Configuration for RAG system with LM Studio integration"""

    def __init__(self):
        """Initialize configuration with values from environment or defaults"""
        # LM Studio connection settings
        self.LM_STUDIO_BASE_URL: str = os.getenv(
            "LM_STUDIO_BASE_URL", "http://localhost:1234/v1"
        )
        self.LM_STUDIO_API_KEY: str = os.getenv(
            "LM_STUDIO_API_KEY", "lm-studio"
        )  # LM Studio doesn't require real key
        self.LM_STUDIO_MODEL: str = os.getenv(
            "LM_STUDIO_EMBEDDING_MODEL", "nomic-embed-text"
        )  # Must match model loaded in LM Studio

        # Embedding settings
        self.EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))
        self.EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
        self.EMBEDDING_TIMEOUT: int = int(os.getenv("EMBEDDING_TIMEOUT", "30"))

        # Vector store settings
        self.VECTOR_STORE_PATH: str = os.getenv(
            "VECTOR_STORE_PATH",
            os.path.join(os.path.dirname(__file__), ".rag_index.pkl")
        )
        self.AUTO_SAVE_INDEX: bool = os.getenv("AUTO_SAVE_INDEX", "true").lower() == "true"

        # Search settings
        self.DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
        self.MIN_SIMILARITY_SCORE: float = float(os.getenv("MIN_SIMILARITY_SCORE", "0.5"))

        # Confidence scoring settings
        self.CONFIDENCE_STRATEGY: str = os.getenv("CONFIDENCE_STRATEGY", "balanced")  # strict, balanced, permissive
        self.ENABLE_CONFIDENCE_SCORING: bool = os.getenv("ENABLE_CONFIDENCE_SCORING", "true").lower() == "true"
        self.CONFIDENCE_TIERS: dict = {
            "high": 0.75,
            "medium": 0.5,
            "low": 0.25,
        }

        # Fallback settings
        self.USE_TFIDF_FALLBACK: bool = os.getenv("USE_TFIDF_FALLBACK", "true").lower() == "true"
        self.TFIDF_MAX_FEATURES: int = int(os.getenv("TFIDF_MAX_FEATURES", "500"))

    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate configuration settings.

        Returns:
            (is_valid, error_message)
        """
        if self.EMBEDDING_DIMENSION <= 0:
            return False, "EMBEDDING_DIMENSION must be positive"

        if self.EMBEDDING_BATCH_SIZE <= 0:
            return False, "EMBEDDING_BATCH_SIZE must be positive"

        if self.DEFAULT_TOP_K <= 0:
            return False, "DEFAULT_TOP_K must be positive"

        if not (0.0 <= self.MIN_SIMILARITY_SCORE <= 1.0):
            return False, "MIN_SIMILARITY_SCORE must be between 0.0 and 1.0"

        if self.CONFIDENCE_STRATEGY not in ["strict", "balanced", "permissive"]:
            return False, "CONFIDENCE_STRATEGY must be 'strict', 'balanced', or 'permissive'"

        return True, None

    def get_summary(self) -> str:
        """Get a summary of current configuration"""
        return f"""
RAG System Configuration:
========================
LM Studio:
  - Base URL: {self.LM_STUDIO_BASE_URL}
  - Model: {self.LM_STUDIO_MODEL}
  - Timeout: {self.EMBEDDING_TIMEOUT}s

Embeddings:
  - Dimension: {self.EMBEDDING_DIMENSION}
  - Batch Size: {self.EMBEDDING_BATCH_SIZE}

Vector Store:
  - Path: {self.VECTOR_STORE_PATH}
  - Auto-save: {self.AUTO_SAVE_INDEX}

Search:
  - Default Top-K: {self.DEFAULT_TOP_K}
  - Min Similarity: {self.MIN_SIMILARITY_SCORE}

Confidence:
  - Scoring Enabled: {self.ENABLE_CONFIDENCE_SCORING}
  - Strategy: {self.CONFIDENCE_STRATEGY}

Fallback:
  - TF-IDF Enabled: {self.USE_TFIDF_FALLBACK}
"""


# Global config instance
config = RAGConfig()
