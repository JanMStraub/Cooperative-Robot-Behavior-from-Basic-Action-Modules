"""
Test Cases for RAG Configuration
=================================

Tests for the RAG system configuration module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import os
from LLMCommunication.rag.Config import RAGConfig, config


class TestRAGConfig:
    """Test RAG configuration class"""

    def test_default_values(self):
        """Test default configuration values are set correctly"""
        assert config.LM_STUDIO_BASE_URL == "http://localhost:1234/v1"
        assert config.LM_STUDIO_MODEL == "nomic-embed-text"
        assert config.EMBEDDING_DIMENSION == 768
        assert config.EMBEDDING_BATCH_SIZE == 10
        assert config.DEFAULT_TOP_K == 5
        assert config.MIN_SIMILARITY_SCORE == 0.5
        assert config.USE_TFIDF_FALLBACK is True

    def test_validation_success(self):
        """Test configuration validation with valid settings"""
        is_valid, error_msg = config.validate()
        assert is_valid is True
        assert error_msg is None

    def test_validation_invalid_embedding_dimension(self):
        """Test validation fails with invalid embedding dimension"""
        original = config.EMBEDDING_DIMENSION
        try:
            config.EMBEDDING_DIMENSION = -1
            is_valid, error_msg = config.validate()
            assert is_valid is False
            assert error_msg is not None and "EMBEDDING_DIMENSION" in error_msg
        finally:
            config.EMBEDDING_DIMENSION = original

    def test_validation_invalid_similarity_score(self):
        """Test validation fails with invalid similarity score"""
        original = config.MIN_SIMILARITY_SCORE
        try:
            config.MIN_SIMILARITY_SCORE = 1.5  # Out of range
            is_valid, error_msg = config.validate()
            assert is_valid is False
            assert error_msg is not None and "MIN_SIMILARITY_SCORE" in error_msg
        finally:
            config.MIN_SIMILARITY_SCORE = original

    def test_get_summary(self):
        """Test configuration summary generation"""
        summary = config.get_summary()
        assert "LM Studio" in summary
        assert "Embeddings" in summary
        assert "Vector Store" in summary
        assert config.LM_STUDIO_BASE_URL in summary

    def test_vector_store_path(self):
        """Test vector store path is correctly set"""
        assert config.VECTOR_STORE_PATH.endswith(".rag_index.pkl")
        assert os.path.isabs(config.VECTOR_STORE_PATH) or ".." not in config.VECTOR_STORE_PATH

    def test_environment_variable_override(self):
        """Test configuration can be overridden with environment variables"""
        # Set environment variable
        test_url = "http://test:5000/v1"
        os.environ["LM_STUDIO_BASE_URL"] = test_url

        # Create new config instance
        test_config = RAGConfig()
        assert test_config.LM_STUDIO_BASE_URL == test_url

        # Cleanup
        del os.environ["LM_STUDIO_BASE_URL"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
