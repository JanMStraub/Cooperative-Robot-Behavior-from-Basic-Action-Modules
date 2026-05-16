#!/usr/bin/env python3
"""Semantic search and retrieval for robot operations."""

from typing import Optional, List, Dict, Any, cast
import os


def _get_registry():
    from core.Imports import get_global_registry

    return get_global_registry()


from config.Rag import RAG_VECTOR_STORE_PATH

from .Embeddings import EmbeddingGenerator
from .VectorStore import VectorStore as _VectorStore
from .Indexer import OperationIndexer
from .QueryEngine import QueryEngine as _QueryEngine
from .ConfidenceScorer import (
    compute_confidence_score,
    get_confidence_level,
    apply_confidence_boosting,
    ConfidenceLevel,
)

from core.LoggingSetup import get_logger

logger = get_logger(__name__)


class RAGSystem:
    """
    Complete RAG system for robot operations.

    This is the main entry point for the RAG system, providing a simple
    API for indexing operations and performing semantic search.
    """

    def __init__(
        self,
        lm_studio_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        registry: Optional[Any] = None,  # Changed to Any to avoid circular import
        auto_load_index: bool = True,
    ):
        """
        Initialize the RAG system.

        Args:
            lm_studio_url: LM Studio base URL (default from config)
            embedding_model: Embedding model name (default from config)
            registry: Operation registry (default: global registry)
            auto_load_index: Automatically load cached index if available

        Example:
            >>> rag = RAGSystem()
            Connected to LM Studio at http://localhost:1234/v1
            Loaded vector store from .rag_index.pkl (5 operations)
        """
        self.registry = registry or _get_registry()
        self.embedding_generator = EmbeddingGenerator(
            base_url=lm_studio_url, model=embedding_model
        )

        self.vector_store: Optional[_VectorStore] = None
        self.query_engine: Optional[_QueryEngine] = None
        self.indexer = OperationIndexer(
            registry=self.registry, embedding_generator=self.embedding_generator
        )

        # Try to load existing index
        if auto_load_index:
            self._try_load_index()

    def _try_load_index(self):
        """Try to load existing index from disk."""
        if os.path.exists(RAG_VECTOR_STORE_PATH):
            try:
                self.vector_store = _VectorStore.load()
                self.query_engine = _QueryEngine(
                    vector_store=self.vector_store,
                    embedding_generator=self.embedding_generator,
                    registry=self.registry,
                )
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                self.vector_store = None
                self.query_engine = None
        else:
            pass

    def index_operations(self, rebuild: bool = False) -> bool:
        """Build or rebuild the operation index."""
        try:
            if rebuild or self.vector_store is None:
                self.vector_store = self.indexer.build_index(save=True)
            else:
                assert self.vector_store is not None
                self.vector_store = self.indexer.update_index(self.vector_store)

            self.query_engine = _QueryEngine(
                vector_store=self.vector_store,
                embedding_generator=self.embedding_generator,
                registry=self.registry,
            )

            return True

        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            return False

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        category: Optional[str] = None,
        complexity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for operations using natural language."""
        if self.query_engine is None:
            logger.error("Query engine not initialized. Call index_operations() first.")
            return []

        qe = cast(_QueryEngine, self.query_engine)
        return qe.search(
            query=query,
            top_k=top_k,
            min_score=min_score,
            category_filter=category,
            complexity_filter=complexity,
        )

    def get_operation_context(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Get full operation context for LLM consumption."""
        if self.query_engine is None:
            logger.error("Query engine not initialized. Call index_operations() first.")
            return {
                "query": query,
                "num_results": 0,
                "summary": "RAG system not initialized",
                "operations": [],
            }

        qe = cast(_QueryEngine, self.query_engine)
        return qe.get_operation_context(query, top_k=top_k)

    def get_operations_by_category(
        self, category: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all operations in a specific category."""
        if self.query_engine is None:
            logger.error("Query engine not initialized. Call index_operations() first.")
            return []

        qe = cast(_QueryEngine, self.query_engine)
        return qe.search_by_category(category, top_k=top_k)

    def find_similar_operations(
        self, operation_id: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find operations similar to a given operation."""
        if self.query_engine is None:
            logger.error("Query engine not initialized. Call index_operations() first.")
            return []

        qe = cast(_QueryEngine, self.query_engine)
        return qe.find_similar_operations(operation_id, top_k=top_k)

    def search_by_type(
        self, query: str, result_type: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for items of a specific type."""
        all_results = self.search(query, top_k=top_k * 3)
        filtered = [
            r for r in all_results if r.get("metadata", {}).get("type") == result_type
        ]
        return filtered[:top_k]

    def is_ready(self) -> bool:
        """Check if RAG system is ready for queries"""
        return self.query_engine is not None and self.vector_store is not None

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about RAG system."""
        stats = {
            "config": {
                "lm_studio_url": self.embedding_generator.base_url,
                "embedding_model": self.embedding_generator.model,
                "vector_store_path": RAG_VECTOR_STORE_PATH,
                "is_ready": self.is_ready(),
            },
            "indexer_stats": {
                "total_operations": (
                    len(self.registry.get_all_operations()) if self.registry else 0
                ),
            },
            "embedding_stats": {
                "embedding_dimension": (
                    self.embedding_generator.get_embedding_dimension()
                    if hasattr(self.embedding_generator, "get_embedding_dimension")
                    else None
                ),
                "using_lm_studio": (
                    self.embedding_generator.is_using_lm_studio()
                    if hasattr(self.embedding_generator, "is_using_lm_studio")
                    else None
                ),
            },
        }

        # Add vector store stats if available
        if self.vector_store:
            _vs = cast(_VectorStore, self.vector_store)
            stats["vector_store_stats"] = {
                "num_operations": len(_vs),
                "has_embeddings": len(_vs) > 0,
            }
        else:
            stats["vector_store_stats"] = {
                "num_operations": 0,
                "has_embeddings": False,
            }

        return stats

    def __repr__(self) -> str:
        ready = "ready" if self.is_ready() else "not indexed"
        num_ops = (
            len(cast(_VectorStore, self.vector_store))
            if self.vector_store is not None
            else 0
        )
        return f"RAGSystem({ready}, operations={num_ops})"


VectorStore = _VectorStore
QueryEngine = _QueryEngine

__all__ = [
    "RAGSystem",
    "EmbeddingGenerator",
    "VectorStore",
    "OperationIndexer",
    "QueryEngine",
    "compute_confidence_score",
    "get_confidence_level",
    "apply_confidence_boosting",
    "ConfidenceLevel",
]
