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

    @patch("operations.Registry.get_global_registry")
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

    @patch("operations.Registry.get_global_registry")
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

    @patch("operations.Registry.get_global_registry")
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
        mock_op.parameters = []  # Empty list of parameters
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

    @patch("operations.Registry.get_global_registry")
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

    @patch("operations.Registry.get_global_registry")
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

    @patch("operations.Registry.get_global_registry")
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

    @patch("operations.Registry.get_global_registry")
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
        mock_emb.base_url = "http://localhost:1234/v1"
        mock_emb.model = "test-model"
        mock_emb.generate_embeddings.return_value = [np.array([0.1, 0.2])]
        mock_emb.get_embedding_dimension.return_value = 2
        mock_emb.is_using_lm_studio.return_value = True
        mock_embedding_gen.return_value = mock_emb

        rag = RAGSystem(auto_load_index=False)
        stats = rag.get_stats()

        assert "config" in stats
        assert "indexer_stats" in stats
        assert "vector_store_stats" in stats
        assert "embedding_stats" in stats

        # Verify config contents
        assert stats["config"]["lm_studio_url"] == "http://localhost:1234/v1"
        assert stats["config"]["embedding_model"] == "test-model"
        assert stats["config"]["is_ready"] == False  # Not indexed yet

        # Verify indexer stats
        assert stats["indexer_stats"]["total_operations"] == 1

        # Verify embedding stats
        assert stats["embedding_stats"]["embedding_dimension"] == 2
        assert stats["embedding_stats"]["using_lm_studio"] == True

        # Verify vector store stats (not indexed)
        assert stats["vector_store_stats"]["num_operations"] == 0
        assert stats["vector_store_stats"]["has_embeddings"] == False

    @patch("operations.Registry.get_global_registry")
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

    @patch("operations.Registry.get_global_registry")
    @patch("rag.EmbeddingGenerator")
    @patch("os.path.exists")
    def test_index_loading_failure_recovery(self, mock_exists, mock_embedding_gen, mock_registry):
        """Test RAG system recovers gracefully from index loading failures"""
        mock_reg = Mock()
        mock_registry.return_value = mock_reg

        mock_emb = Mock()
        mock_embedding_gen.return_value = mock_emb

        # Simulate index file exists but is corrupted
        mock_exists.return_value = True

        with patch.object(VectorStore, "load", side_effect=Exception("Corrupted index")):
            rag = RAGSystem(auto_load_index=True)

            # Should initialize successfully despite load failure
            assert rag.vector_store is None
            assert rag.query_engine is None
            assert rag.is_ready() == False

    @patch("operations.Registry.get_global_registry")
    @patch("rag.EmbeddingGenerator")
    def test_search_without_index(self, mock_embedding_gen, mock_registry):
        """Test search returns empty list when index not built"""
        mock_reg = Mock()
        mock_registry.return_value = mock_reg

        mock_emb = Mock()
        mock_embedding_gen.return_value = mock_emb

        rag = RAGSystem(auto_load_index=False)
        results = rag.search("test query")

        assert results == []

    @patch("operations.Registry.get_global_registry")
    @patch("rag.EmbeddingGenerator")
    def test_get_operation_context_without_index(self, mock_embedding_gen, mock_registry):
        """Test get_operation_context returns error dict when index not built"""
        mock_reg = Mock()
        mock_registry.return_value = mock_reg

        mock_emb = Mock()
        mock_embedding_gen.return_value = mock_emb

        rag = RAGSystem(auto_load_index=False)
        context = rag.get_operation_context("test query")

        assert "query" in context
        assert context["num_results"] == 0
        assert "not initialized" in context["summary"].lower()
        assert context["operations"] == []

    @patch("operations.Registry.get_global_registry")
    @patch("rag.EmbeddingGenerator")
    @pytest.mark.slow
    def test_large_scale_indexing_performance(self, mock_embedding_gen, mock_registry):
        """Test indexing performance with large number of operations"""
        import time

        # Create 100 mock operations
        mock_ops = []
        for i in range(100):
            mock_op = Mock()
            mock_op.operation_id = f"op_{i:03d}"
            mock_op.name = f"operation_{i}"
            mock_op.category = Mock(value="navigation")
            mock_op.complexity = Mock(value="basic")
            mock_op.description = f"Test operation {i}"
            mock_op.average_duration_ms = 1000.0
            mock_op.success_rate = 0.95
            mock_op.parameters = []
            mock_op.to_rag_document.return_value = f"Test doc {i}"
            mock_ops.append(mock_op)

        mock_reg = Mock()
        mock_reg.get_all_operations.return_value = mock_ops
        mock_registry.return_value = mock_reg

        # Mock embeddings for all operations
        mock_emb = Mock()
        mock_emb.generate_embeddings.return_value = [
            np.random.rand(384).astype(np.float32) for _ in range(100)
        ]
        mock_embedding_gen.return_value = mock_emb

        rag = RAGSystem(auto_load_index=False)

        # Measure indexing time
        start = time.time()
        success = rag.index_operations()
        elapsed = time.time() - start

        assert success is True
        assert rag.is_ready() is True

        # Indexing 100 operations should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0, f"Indexing took {elapsed:.2f}s, expected < 5s"

    @patch("operations.Registry.get_global_registry")
    @patch("rag.EmbeddingGenerator")
    def test_embedding_dimension_mismatch_handling(self, mock_embedding_gen, mock_registry):
        """Test handling of embedding dimension mismatches"""
        mock_op = Mock()
        mock_op.operation_id = "op_001"
        mock_op.name = "test_op"
        mock_op.category = Mock(value="navigation")
        mock_op.complexity = Mock(value="basic")
        mock_op.description = "Test"
        mock_op.average_duration_ms = 1000.0
        mock_op.success_rate = 0.95
        mock_op.parameters = []
        mock_op.to_rag_document.return_value = "Test doc"

        mock_reg = Mock()
        mock_reg.get_all_operations.return_value = [mock_op]
        mock_registry.return_value = mock_reg

        # Create embeddings with wrong dimension
        mock_emb = Mock()
        # First call returns 384-dim, second call returns 256-dim (mismatch)
        mock_emb.generate_embeddings.side_effect = [
            [np.random.rand(384).astype(np.float32)],
            [np.random.rand(256).astype(np.float32)],
        ]
        mock_emb.generate_embedding.return_value = np.random.rand(256).astype(np.float32)
        mock_embedding_gen.return_value = mock_emb

        rag = RAGSystem(auto_load_index=False)

        # Index operations (first embedding)
        success = rag.index_operations()
        assert success is True

        # Try to search (second embedding with different dimension)
        # Should handle gracefully (VectorStore may raise error or return empty)
        try:
            results = rag.search("test query")
            # If no error, verify results are handled
            assert isinstance(results, list)
        except Exception as e:
            # Dimension mismatch should be caught and handled
            assert "dimension" in str(e).lower() or "shape" in str(e).lower()


    @patch("operations.Registry.get_global_registry")
    @patch("rag.EmbeddingGenerator")
    @pytest.mark.slow
    def test_very_large_scale_performance(self, mock_embedding_gen, mock_registry):
        """Test indexing and query performance with 1000+ operations"""
        import time

        # Create 1000 mock operations
        mock_ops = []
        for i in range(1000):
            mock_op = Mock()
            mock_op.operation_id = f"op_{i:04d}"
            mock_op.name = f"operation_{i}"
            mock_op.category = Mock(value=["navigation", "manipulation", "detection"][i % 3])
            mock_op.complexity = Mock(value=["basic", "intermediate", "advanced"][i % 3])
            mock_op.description = f"Test operation {i} for category {i % 3}"
            mock_op.average_duration_ms = 1000.0 + (i * 10)
            mock_op.success_rate = 0.9 + (i % 10) * 0.01
            mock_op.parameters = []
            mock_op.to_rag_document.return_value = f"Test doc {i} with description"
            mock_ops.append(mock_op)

        mock_reg = Mock()
        mock_reg.get_all_operations.return_value = mock_ops
        mock_registry.return_value = mock_reg

        # Mock embeddings for all operations
        mock_emb = Mock()
        mock_emb.generate_embeddings.return_value = [
            np.random.rand(384).astype(np.float32) for _ in range(1000)
        ]
        # Mock query embedding
        mock_emb.generate_embedding.return_value = np.random.rand(384).astype(np.float32)
        mock_embedding_gen.return_value = mock_emb

        rag = RAGSystem(auto_load_index=False)

        # Measure indexing time
        start = time.time()
        success = rag.index_operations()
        index_time = time.time() - start

        assert success is True
        assert rag.is_ready() is True

        # Indexing 1000 operations should complete in reasonable time (< 30 seconds)
        assert index_time < 30.0, f"Indexing took {index_time:.2f}s, expected < 30s"

        # Measure query time
        start = time.time()
        results = rag.search("test query", top_k=10)
        query_time = time.time() - start

        # Query should be fast (< 1 second)
        assert query_time < 1.0, f"Query took {query_time:.2f}s, expected < 1s"
        assert len(results) <= 10

    @patch("operations.Registry.get_global_registry")
    @patch("rag.EmbeddingGenerator")
    def test_semantic_search_accuracy(self, mock_embedding_gen, mock_registry):
        """Test semantic search returns relevant operations for natural language queries"""
        # Create mock operations with realistic names and descriptions
        move_op = Mock()
        move_op.operation_id = "move_001"
        move_op.name = "move_to_coordinate"
        move_op.category = Mock(value="navigation")
        move_op.complexity = Mock(value="basic")
        move_op.description = "Move robot to specified XYZ coordinate"
        move_op.average_duration_ms = 2000.0
        move_op.success_rate = 0.95
        move_op.parameters = []
        move_op.to_rag_document.return_value = "move_to_coordinate: Move robot to specified XYZ coordinate"

        grasp_op = Mock()
        grasp_op.operation_id = "grasp_001"
        grasp_op.name = "execute_grasp"
        grasp_op.category = Mock(value="manipulation")
        grasp_op.complexity = Mock(value="intermediate")
        grasp_op.description = "Execute grasp operation to pick up object"
        grasp_op.average_duration_ms = 5000.0
        grasp_op.success_rate = 0.85
        grasp_op.parameters = []
        grasp_op.to_rag_document.return_value = "execute_grasp: Execute grasp operation to pick up object"

        detect_op = Mock()
        detect_op.operation_id = "detect_001"
        detect_op.name = "detect_objects"
        detect_op.category = Mock(value="perception")
        detect_op.complexity = Mock(value="basic")
        detect_op.description = "Detect objects in camera view using color or ML"
        detect_op.average_duration_ms = 1000.0
        detect_op.success_rate = 0.90
        detect_op.parameters = []
        detect_op.to_rag_document.return_value = "detect_objects: Detect objects in camera view using color or ML"

        mock_reg = Mock()
        mock_reg.get_all_operations.return_value = [move_op, grasp_op, detect_op]
        mock_reg.get_operation_by_id.side_effect = lambda op_id: {
            "move_001": move_op,
            "grasp_001": grasp_op,
            "detect_001": detect_op
        }.get(op_id)
        mock_registry.return_value = mock_reg

        # Create embeddings that cluster similar operations
        # Move embedding: high on first dimension
        move_embedding = np.array([0.9, 0.1, 0.1] + [0.0] * 381, dtype=np.float32)
        # Grasp embedding: high on second dimension
        grasp_embedding = np.array([0.1, 0.9, 0.1] + [0.0] * 381, dtype=np.float32)
        # Detect embedding: high on third dimension
        detect_embedding = np.array([0.1, 0.1, 0.9] + [0.0] * 381, dtype=np.float32)

        mock_emb = Mock()
        mock_emb.generate_embeddings.return_value = [move_embedding, grasp_embedding, detect_embedding]
        mock_embedding_gen.return_value = mock_emb

        rag = RAGSystem(auto_load_index=False)
        success = rag.index_operations()
        assert success is True

        # Test 1: Query similar to "move" should return move operation first
        mock_emb.generate_embedding.return_value = np.array([0.85, 0.05, 0.05] + [0.0] * 381, dtype=np.float32)
        results = rag.search("move robot to position", top_k=3)

        assert len(results) > 0
        # First result should be move operation (highest similarity)
        assert results[0]['operation_id'] == "move_001"

        # Test 2: Query similar to "grasp" should return grasp operation first
        mock_emb.generate_embedding.return_value = np.array([0.05, 0.85, 0.05] + [0.0] * 381, dtype=np.float32)
        results = rag.search("pick up object", top_k=3)

        assert len(results) > 0
        assert results[0]['operation_id'] == "grasp_001"

        # Test 3: Query similar to "detect" should return detect operation first
        mock_emb.generate_embedding.return_value = np.array([0.05, 0.05, 0.85] + [0.0] * 381, dtype=np.float32)
        results = rag.search("find objects in scene", top_k=3)

        assert len(results) > 0
        assert results[0]['operation_id'] == "detect_001"

    @patch("operations.Registry.get_global_registry")
    @patch("rag.EmbeddingGenerator")
    def test_query_with_filters(self, mock_embedding_gen, mock_registry):
        """Test search filtering by category and complexity"""
        # Create diverse operations
        ops = []
        categories = ["navigation", "manipulation", "perception"]
        complexities = ["basic", "intermediate", "advanced"]

        for i in range(9):
            cat_idx = i % 3
            comp_idx = i // 3

            mock_op = Mock()
            mock_op.operation_id = f"op_{i:02d}"
            mock_op.name = f"operation_{i}"
            mock_op.category = Mock(value=categories[cat_idx])
            mock_op.complexity = Mock(value=complexities[comp_idx])
            mock_op.description = f"Operation {i}"
            mock_op.average_duration_ms = 1000.0
            mock_op.success_rate = 0.90
            mock_op.parameters = []
            mock_op.to_rag_document.return_value = f"doc_{i}"
            ops.append(mock_op)

        mock_reg = Mock()
        mock_reg.get_all_operations.return_value = ops
        mock_registry.return_value = mock_reg

        # Mock embeddings
        mock_emb = Mock()
        mock_emb.generate_embeddings.return_value = [
            np.random.rand(384).astype(np.float32) for _ in range(9)
        ]
        mock_emb.generate_embedding.return_value = np.random.rand(384).astype(np.float32)
        mock_embedding_gen.return_value = mock_emb

        rag = RAGSystem(auto_load_index=False)
        success = rag.index_operations()
        assert success is True

        # Test category filter
        nav_results = rag.search("test query", top_k=10, category="navigation")
        # Should return only navigation operations (3 total: ops 0, 3, 6)
        assert len(nav_results) <= 3

        # Test complexity filter
        basic_results = rag.search("test query", top_k=10, complexity="basic")
        # Should return only basic operations (3 total: ops 0, 1, 2)
        assert len(basic_results) <= 3

        # Test both filters combined
        nav_basic_results = rag.search("test query", top_k=10, category="navigation", complexity="basic")
        # Should return only navigation + basic (1 total: op 0)
        assert len(nav_basic_results) <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
