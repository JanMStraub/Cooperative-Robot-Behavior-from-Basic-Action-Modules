"""
Test Cases for RAG Query Engine
================================

Tests for the query and search module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from rag.QueryEngine import QueryEngine
from rag.VectorStore import VectorStore


class TestQueryEngine:
    """Test query engine functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        # Create a mock vector store with test operations
        self.vector_store = VectorStore()
        self.vector_store.add_operation(
            "op_001",
            np.array([1.0, 0.0, 0.0]),
            {"name": "move_to_coordinate", "category": "navigation", "description": "Move robot"}
        )
        self.vector_store.add_operation(
            "op_002",
            np.array([0.0, 1.0, 0.0]),
            {"name": "grip_object", "category": "manipulation", "description": "Grip object"}
        )

    @patch("rag.QueryEngine.EmbeddingGenerator")
    @patch("operations.Registry.get_global_registry")
    def test_search_basic(self, mock_registry, mock_embedding_gen):
        """Test basic search functionality"""
        # Mock embedding generator
        mock_emb = Mock()
        mock_emb.generate_embedding.return_value = np.array([1.0, 0.0, 0.0])
        mock_embedding_gen.return_value = mock_emb

        engine = QueryEngine(self.vector_store, mock_emb)
        results = engine.search("move robot", top_k=2)

        assert len(results) <= 2
        assert all("operation_id" in r for r in results)
        assert all("score" in r for r in results)

    @patch("rag.QueryEngine.EmbeddingGenerator")
    def test_search_empty_query(self, mock_embedding_gen):
        """Test search with empty query"""
        mock_emb = Mock()
        mock_embedding_gen.return_value = mock_emb

        engine = QueryEngine(self.vector_store, mock_emb)
        results = engine.search("")

        assert results == []

    @patch("rag.QueryEngine.EmbeddingGenerator")
    @patch("operations.Registry.get_global_registry")
    def test_search_with_category_filter(self, mock_registry, mock_embedding_gen):
        """Test search with category filtering"""
        mock_emb = Mock()
        mock_emb.generate_embedding.return_value = np.array([1.0, 0.0, 0.0])
        mock_embedding_gen.return_value = mock_emb

        engine = QueryEngine(self.vector_store, mock_emb)
        results = engine.search("robot", category_filter="navigation")

        assert all(r["metadata"]["category"] == "navigation" for r in results)

    @patch("rag.QueryEngine.EmbeddingGenerator")
    @patch("operations.Registry.get_global_registry")
    def test_get_operation_context(self, mock_registry, mock_embedding_gen):
        """Test getting full operation context"""
        # Mock registry
        mock_op = Mock()
        mock_op.operation_id = "op_001"
        mock_op.name = "move_to_coordinate"
        mock_op.category = Mock(value="navigation")
        mock_op.complexity = Mock(value="basic")
        mock_op.description = "Move robot"
        mock_op.parameters = []
        mock_op.usage_examples = ["example 1"]
        mock_op.preconditions = ["precondition 1"]
        mock_op.postconditions = ["postcondition 1"]
        mock_op.failure_modes = ["failure 1"]
        # Mock relationships to be None (no relationships)
        mock_op.relationships = None

        mock_reg = Mock()
        mock_reg.get_operation.return_value = mock_op
        mock_registry.return_value = mock_reg

        # Mock embedding generator
        mock_emb = Mock()
        mock_emb.generate_embedding.return_value = np.array([1.0, 0.0, 0.0])
        mock_embedding_gen.return_value = mock_emb

        engine = QueryEngine(self.vector_store, mock_emb, mock_reg)
        context = engine.get_operation_context("move robot", top_k=1)

        assert "query" in context
        assert "operations" in context
        assert "summary" in context
        assert context["num_results"] >= 0

    @patch("rag.QueryEngine.EmbeddingGenerator")
    def test_find_similar_operations(self, mock_embedding_gen):
        """Test finding similar operations"""
        mock_emb = Mock()
        mock_embedding_gen.return_value = mock_emb

        engine = QueryEngine(self.vector_store, mock_emb)
        similar = engine.find_similar_operations("op_001", top_k=1)

        # Should not include the operation itself
        assert all(r["operation_id"] != "op_001" for r in similar)

    @patch("rag.QueryEngine.EmbeddingGenerator")
    def test_get_stats(self, mock_embedding_gen):
        """Test getting query engine statistics"""
        mock_emb = Mock()
        mock_emb.is_using_lm_studio.return_value = True
        mock_embedding_gen.return_value = mock_emb

        engine = QueryEngine(self.vector_store, mock_emb)
        stats = engine.get_stats()

        assert "vector_store_stats" in stats
        assert "using_lm_studio" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
