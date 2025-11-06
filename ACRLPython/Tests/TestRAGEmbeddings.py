"""
Test Cases for RAG Embeddings
==============================

Tests for the embedding generation module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from unittest.mock import Mock, patch
from LLMCommunication.rag.Embeddings import EmbeddingGenerator
from LLMCommunication.rag.Config import config


class TestEmbeddingGenerator:
    """Test embedding generator with LM Studio and TF-IDF fallback"""

    @patch("LLMCommunication.rag.Embeddings.OpenAI")
    def test_initialization_with_lm_studio(self, mock_openai):
        """Test initialization successfully connects to LM Studio"""
        # Mock successful LM Studio connection
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 768)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        generator = EmbeddingGenerator()

        assert generator.use_lm_studio is True
        assert generator.client is not None

    @patch("LLMCommunication.rag.Embeddings.OpenAI")
    def test_initialization_fallback_to_tfidf(self, mock_openai):
        """Test fallback to TF-IDF when LM Studio unavailable"""
        # Mock LM Studio connection failure
        mock_openai.side_effect = Exception("Connection failed")

        generator = EmbeddingGenerator()

        assert generator.use_lm_studio is False
        assert generator.tfidf_vectorizer is not None

    @patch("LLMCommunication.rag.Embeddings.OpenAI")
    def test_generate_single_embedding_lm_studio(self, mock_openai):
        """Test generating single embedding via LM Studio"""
        # Mock LM Studio
        mock_client = Mock()
        mock_response = Mock()
        test_embedding = [0.1, 0.2, 0.3] + [0.0] * 765
        mock_response.data = [Mock(embedding=test_embedding)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        generator = EmbeddingGenerator()
        embedding = generator.generate_embedding("test text")

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 768
        assert embedding[0] == pytest.approx(0.1, rel=1e-3)

    def test_generate_embeddings_tfidf(self):
        """Test generating embeddings with TF-IDF fallback"""
        # Force TF-IDF by mocking failed LM Studio connection
        with patch("LLMCommunication.rag.Embeddings.OpenAI") as mock_openai:
            mock_openai.side_effect = Exception("No LM Studio")

            generator = EmbeddingGenerator()
            texts = ["robot movement", "pick up object", "grasp cube"]
            embeddings = generator.generate_embeddings(texts)

            assert len(embeddings) == 3
            assert all(isinstance(e, np.ndarray) for e in embeddings)
            assert all(len(e) == config.TFIDF_MAX_FEATURES for e in embeddings)

    @patch("LLMCommunication.rag.Embeddings.OpenAI")
    def test_batch_embedding_generation(self, mock_openai):
        """Test batch embedding generation"""
        # Mock LM Studio batch response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 768),
            Mock(embedding=[0.2] * 768),
            Mock(embedding=[0.3] * 768),
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        generator = EmbeddingGenerator()
        texts = ["text 1", "text 2", "text 3"]
        embeddings = generator.generate_embeddings(texts)

        assert len(embeddings) == 3
        assert all(isinstance(e, np.ndarray) for e in embeddings)

    def test_empty_text_list(self):
        """Test handling of empty text list"""
        with patch("LLMCommunication.rag.Embeddings.OpenAI") as mock_openai:
            mock_openai.side_effect = Exception("No LM Studio")

            generator = EmbeddingGenerator()
            embeddings = generator.generate_embeddings([])

            assert embeddings == []

    @patch("LLMCommunication.rag.Embeddings.OpenAI")
    def test_get_embedding_dimension(self, mock_openai):
        """Test getting embedding dimension"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 768)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        generator = EmbeddingGenerator()
        dim = generator.get_embedding_dimension()

        assert dim == config.EMBEDDING_DIMENSION

    def test_is_using_lm_studio(self):
        """Test checking if using LM Studio or TF-IDF"""
        with patch("LLMCommunication.rag.Embeddings.OpenAI") as mock_openai:
            # Test with LM Studio
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 768)]
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client

            generator = EmbeddingGenerator()
            assert generator.is_using_lm_studio() is True

    def test_repr(self):
        """Test string representation"""
        with patch("LLMCommunication.rag.Embeddings.OpenAI") as mock_openai:
            mock_openai.side_effect = Exception("No LM Studio")

            generator = EmbeddingGenerator()
            repr_str = repr(generator)

            assert "EmbeddingGenerator" in repr_str
            assert "tfidf" in repr_str or "lm_studio" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
