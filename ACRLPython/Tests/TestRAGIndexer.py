"""
Test Cases for RAG Indexer
===========================

Tests for the operation indexing module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from rag.Indexer import OperationIndexer


class TestOperationIndexer:
    """Test operation indexer"""

    @patch("rag.Indexer.get_global_registry")
    @patch("rag.Indexer.EmbeddingGenerator")
    def test_build_index(self, mock_embedding_gen, mock_registry):
        """Test building index from operations"""
        # Mock registry with operations
        mock_op = Mock()
        mock_op.operation_id = "op_001"
        mock_op.name = "test_op"
        mock_op.category = Mock(value="navigation")
        mock_op.complexity = Mock(value="basic")
        mock_op.description = "Test operation"
        mock_op.average_duration_ms = 1000.0
        mock_op.success_rate = 0.95
        mock_op.to_rag_document.return_value = "Test operation document"

        mock_reg = Mock()
        mock_reg.get_all_operations.return_value = [mock_op]
        mock_registry.return_value = mock_reg

        # Mock embedding generator
        mock_emb = Mock()
        mock_emb.generate_embeddings.return_value = [np.array([0.1, 0.2, 0.3])]
        mock_embedding_gen.return_value = mock_emb

        # Build index
        indexer = OperationIndexer()
        store = indexer.build_index(save=False)

        assert len(store) == 1
        assert "op_001" in store.operation_ids

    @patch("rag.Indexer.get_global_registry")
    @patch("rag.Indexer.EmbeddingGenerator")
    def test_build_index_empty_registry(self, mock_embedding_gen, mock_registry):
        """Test building index with empty registry"""
        mock_reg = Mock()
        mock_reg.get_all_operations.return_value = []
        mock_registry.return_value = mock_reg

        indexer = OperationIndexer()
        store = indexer.build_index(save=False)

        assert len(store) == 0

    @patch("rag.Indexer.get_global_registry")
    @patch("rag.Indexer.EmbeddingGenerator")
    def test_rebuild_index(self, mock_embedding_gen, mock_registry):
        """Test rebuilding index"""
        mock_reg = Mock()
        mock_reg.get_all_operations.return_value = []
        mock_registry.return_value = mock_reg

        indexer = OperationIndexer()
        store = indexer.rebuild_index()

        assert len(store) == 0

    @patch("rag.Indexer.get_global_registry")
    @patch("rag.Indexer.EmbeddingGenerator")
    def test_get_indexer_stats(self, mock_embedding_gen, mock_registry):
        """Test getting indexer statistics"""
        mock_reg = Mock()
        mock_reg.get_all_operations.return_value = [Mock()]
        mock_registry.return_value = mock_reg

        mock_emb = Mock()
        mock_emb.get_embedding_dimension.return_value = 768
        mock_emb.is_using_lm_studio.return_value = True
        mock_embedding_gen.return_value = mock_emb

        indexer = OperationIndexer()
        stats = indexer.get_indexer_stats()

        assert "num_operations" in stats
        assert "embedding_dimension" in stats
        assert "using_lm_studio" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
