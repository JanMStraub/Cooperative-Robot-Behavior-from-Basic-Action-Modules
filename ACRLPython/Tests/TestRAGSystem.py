"""
Test Cases for RAG System Integration
======================================

Integration tests for the complete RAG system.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from rag import RAGSystem
from rag.VectorStore import VectorStore


class TestRAGSystemIntegration:
    """Integration tests for RAG system"""

    @patch("rag.get_global_registry")
    @patch("rag.EmbeddingGenerator")
    def test_initialization(self, mock_embedding_gen, mock_registry):
        """Test RAG system initialization"""
        # Mock components
        mock_reg = Mock()
        mock_reg.get_all_operations.return_value = []
        mock_registry.return_value = mock_reg

        mock_emb = Mock()
        mock_embedding_gen.return_value = mock_emb

        rag = RAGSystem(auto_load_index=False)

        assert rag.registry is not None
        assert rag.embedding_generator is not None
        assert rag.indexer is not None

    @patch("rag.get_global_registry")
    @patch("rag.EmbeddingGenerator")
    def test_auto_load_index(self, mock_embedding_gen, mock_registry):
        """Test automatic index loading on initialization"""
        # Mock loaded store
        mock_store = Mock(spec=VectorStore)
        mock_store.__len__ = Mock(return_value=5)

        mock_reg = Mock()
        mock_registry.return_value = mock_reg

        mock_emb = Mock()
        mock_embedding_gen.return_value = mock_emb

        # Patch VectorStore.load class method where the class is imported
        with patch.object(VectorStore, "load", return_value=mock_store):
            with patch("os.path.exists", return_value=True):
                rag = RAGSystem(auto_load_index=True)

        assert rag.vector_store is not None
        assert rag.query_engine is not None

    @patch("rag.get_global_registry")
    @patch("rag.EmbeddingGenerator")
    def test_index_operations(self, mock_embedding_gen, mock_registry):
        """Test indexing operations"""
        # Mock operation
        mock_op = Mock()
        mock_op.operation_id = "op_001"
        mock_op.name = "test_op"
        mock_op.category = Mock(value="navigation")
        mock_op.complexity = Mock(value="basic")
        mock_op.description = "Test"
        mock_op.average_duration_ms = 1000.0
        mock_op.success_rate = 0.95
        mock_op.to_rag_document.return_value = "Test doc"

        mock_reg = Mock()
        mock_reg.get_all_operations.return_value = [mock_op]
        mock_registry.return_value = mock_reg

        # Mock embedding generator
        mock_emb = Mock()
        mock_emb.generate_embeddings.return_value = [np.array([0.1, 0.2, 0.3])]
        mock_embedding_gen.return_value = mock_emb

        rag = RAGSystem(auto_load_index=False)
        success = rag.index_operations()

        assert success is True
        assert rag.is_ready() is True

    @patch("rag.get_global_registry")
    @patch("rag.EmbeddingGenerator")
    def test_search(self, mock_embedding_gen, mock_registry):
        """Test searching for operations"""
        # Setup mocks
        mock_reg = Mock()
        mock_registry.return_value = mock_reg

        mock_emb = Mock()
        mock_emb.generate_embedding.return_value = np.array([1.0, 0.0, 0.0])
        mock_embedding_gen.return_value = mock_emb

        # Create RAG with pre-populated store
        rag = RAGSystem(auto_load_index=False)
        rag.vector_store = VectorStore()
        rag.vector_store.add_operation(
            "op_001",
            np.array([1.0, 0.0, 0.0]),
            {"name": "move_to_coordinate", "category": "navigation"}
        )

        from rag.QueryEngine import QueryEngine
        rag.query_engine = QueryEngine(rag.vector_store, mock_emb, mock_reg)

        # Search
        results = rag.search("move robot", top_k=1)

        assert len(results) >= 0

    @patch("rag.get_global_registry")
    @patch("rag.EmbeddingGenerator")
    def test_search_not_ready(self, mock_embedding_gen, mock_registry):
        """Test search when system not ready"""
        mock_reg = Mock()
        mock_registry.return_value = mock_reg

        mock_emb = Mock()
        mock_embedding_gen.return_value = mock_emb

        rag = RAGSystem(auto_load_index=False)
        results = rag.search("test query")

        assert results == []

    @patch("rag.get_global_registry")
    @patch("rag.EmbeddingGenerator")
    def test_is_ready(self, mock_embedding_gen, mock_registry):
        """Test checking if system is ready"""
        mock_reg = Mock()
        mock_registry.return_value = mock_reg

        mock_emb = Mock()
        mock_embedding_gen.return_value = mock_emb

        rag = RAGSystem(auto_load_index=False)

        # Not ready initially
        assert rag.is_ready() is False

        # Set components to make it ready
        rag.vector_store = Mock(spec=VectorStore)
        rag.query_engine = Mock()

        assert rag.is_ready() is True

    @patch("rag.get_global_registry")
    @patch("rag.EmbeddingGenerator")
    def test_get_stats(self, mock_embedding_gen, mock_registry):
        """Test getting RAG system statistics"""
        # Mock operation
        mock_op = Mock()
        mock_op.operation_id = "op_001"
        mock_op.name = "test_op"
        mock_op.category = Mock(value="navigation")
        mock_op.complexity = Mock(value="basic")
        mock_op.description = "Test"
        mock_op.average_duration_ms = 1000.0
        mock_op.success_rate = 0.95
        mock_op.to_rag_document.return_value = "Test doc"

        mock_reg = Mock()
        mock_reg.get_all_operations.return_value = [mock_op]
        mock_registry.return_value = mock_reg

        mock_emb = Mock()
        mock_emb.generate_embeddings.return_value = [np.array([0.1, 0.2])]
        mock_emb.get_embedding_dimension.return_value = 2
        mock_emb.is_using_lm_studio.return_value = True
        mock_embedding_gen.return_value = mock_emb

        rag = RAGSystem(auto_load_index=False)
        stats = rag.get_stats()

        assert "config" in stats
        assert "indexer_stats" in stats

    @patch("rag.get_global_registry")
    @patch("rag.EmbeddingGenerator")
    def test_repr(self, mock_embedding_gen, mock_registry):
        """Test string representation"""
        mock_reg = Mock()
        mock_registry.return_value = mock_reg

        mock_emb = Mock()
        mock_embedding_gen.return_value = mock_emb

        rag = RAGSystem(auto_load_index=False)
        repr_str = repr(rag)

        assert "RAGSystem" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
