"""
RAG System Configuration
=========================

Configuration settings for the RAG system using LM Studio for embeddings.
"""

import os
from typing import Optional


class RAGConfig:
    """Configuration for RAG system with LM Studio integration"""

    # LM Studio connection settings
    LM_STUDIO_BASE_URL: str = os.getenv(
        "LM_STUDIO_BASE_URL", "http://localhost:1234/v1"
    )
    LM_STUDIO_API_KEY: str = os.getenv(
        "LM_STUDIO_API_KEY", "lm-studio"
    )  # LM Studio doesn't require real key
    LM_STUDIO_MODEL: str = os.getenv(
        "LM_STUDIO_EMBEDDING_MODEL", "nomic-embed-text"
    )  # Must match model loaded in LM Studio

    # Embedding settings
    EMBEDDING_DIMENSION: int = (
        768  # Default for nomic-embed-text, adjust based on model
    )
    EMBEDDING_BATCH_SIZE: int = 10  # Number of texts to embed in one request
    EMBEDDING_TIMEOUT: int = 30  # Timeout in seconds for embedding requests

    # Vector store settings
    VECTOR_STORE_PATH: str = os.path.join(os.path.dirname(__file__), ".rag_index.pkl")
    AUTO_SAVE_INDEX: bool = True  # Automatically save index after building

    # Search settings
    DEFAULT_TOP_K: int = 5  # Default number of results to return
    MIN_SIMILARITY_SCORE: float = 0.5  # Minimum cosine similarity to include in results

    # Fallback settings
    USE_TFIDF_FALLBACK: bool = True  # Use TF-IDF if LM Studio unavailable
    TFIDF_MAX_FEATURES: int = 500  # Maximum features for TF-IDF vectorizer

    @classmethod
    def validate(cls) -> tuple[bool, Optional[str]]:
        """
        Validate configuration settings.

        Returns:
            (is_valid, error_message)
        """
        if cls.EMBEDDING_DIMENSION <= 0:
            return False, "EMBEDDING_DIMENSION must be positive"

        if cls.EMBEDDING_BATCH_SIZE <= 0:
            return False, "EMBEDDING_BATCH_SIZE must be positive"

        if cls.DEFAULT_TOP_K <= 0:
            return False, "DEFAULT_TOP_K must be positive"

        if not (0.0 <= cls.MIN_SIMILARITY_SCORE <= 1.0):
            return False, "MIN_SIMILARITY_SCORE must be between 0.0 and 1.0"

        return True, None

    @classmethod
    def get_summary(cls) -> str:
        """Get a summary of current configuration"""
        return f"""
RAG System Configuration:
========================
LM Studio:
  - Base URL: {cls.LM_STUDIO_BASE_URL}
  - Model: {cls.LM_STUDIO_MODEL}
  - Timeout: {cls.EMBEDDING_TIMEOUT}s

Embeddings:
  - Dimension: {cls.EMBEDDING_DIMENSION}
  - Batch Size: {cls.EMBEDDING_BATCH_SIZE}

Vector Store:
  - Path: {cls.VECTOR_STORE_PATH}
  - Auto-save: {cls.AUTO_SAVE_INDEX}

Search:
  - Default Top-K: {cls.DEFAULT_TOP_K}
  - Min Similarity: {cls.MIN_SIMILARITY_SCORE}

Fallback:
  - TF-IDF Enabled: {cls.USE_TFIDF_FALLBACK}
"""


# Global config instance
config = RAGConfig()
