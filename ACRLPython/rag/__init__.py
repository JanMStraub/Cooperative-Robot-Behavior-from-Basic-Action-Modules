"""
RAG System for Robot Operations
================================

Semantic search and retrieval system for robot operations using LM Studio embeddings.

Usage:
    >>> from rag import RAGSystem
    >>>
    >>> # Initialize RAG system
    >>> rag = RAGSystem()
    >>>
    >>> # Build index (first time or when operations change)
    >>> rag.index_operations()
    >>>
    >>> # Search for operations
    >>> results = rag.search("move robot to position", top_k=3)
    >>> print(results[0]['metadata']['name'])
    'move_to_coordinate'
    >>>
    >>> # Get full context for LLM
    >>> context = rag.get_operation_context("pick up object")
    >>> print(context['summary'])
    'Found 3 relevant operations for: pick up object'
"""

from typing import Optional, List, Dict, Any
import os
import logging

from operations.Registry import OperationRegistry, get_global_registry

# Import config
# Import config - try both import styles
try:
    import LLMConfig as config
except ImportError:
    from .. import LLMConfig as config

from .Embeddings import EmbeddingGenerator
from .VectorStore import VectorStore
from .Indexer import OperationIndexer
from .QueryEngine import QueryEngine
from .ConfidenceScorer import (
    compute_confidence_score,
    get_confidence_level,
    apply_confidence_boosting,
    ConfidenceLevel,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        registry: Optional[OperationRegistry] = None,
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
            ✓ Connected to LM Studio at http://localhost:1234/v1
            Loaded vector store from .rag_index.pkl (5 operations)
        """
        self.registry = registry or get_global_registry()
        self.embedding_generator = EmbeddingGenerator(
            base_url=lm_studio_url, model=embedding_model
        )

        self.vector_store: Optional[VectorStore] = None
        self.query_engine: Optional[QueryEngine] = None
        self.indexer = OperationIndexer(
            registry=self.registry, embedding_generator=self.embedding_generator
        )

        # Try to load existing index
        if auto_load_index:
            self._try_load_index()

    def _try_load_index(self):
        """Try to load existing index from disk"""
        if os.path.exists(config.RAG_VECTOR_STORE_PATH):
            try:
                self.vector_store = VectorStore.load()
                self.query_engine = QueryEngine(
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
        """
        Build or rebuild the operation index.

        This generates embeddings for all operations in the registry
        and creates a searchable index.

        Args:
            rebuild: Force rebuild even if index exists (default: False)

        Returns:
            True if indexing succeeded, False otherwise

        Example:
            >>> rag = RAGSystem()
            >>> rag.index_operations()
            Building index for 5 operations...
            ✓ Index built with 5 operations
            True
        """
        try:
            if rebuild or self.vector_store is None:
                self.vector_store = self.indexer.build_index(save=True)
            else:
                self.vector_store = self.indexer.update_index(self.vector_store)

            # Create query engine with new vector store
            self.query_engine = QueryEngine(
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
        """
        Search for operations using natural language.

        Args:
            query: Natural language search query
            top_k: Number of results to return (default from config)
            min_score: Minimum similarity score (default from config)
            category: Filter by category (e.g., "navigation")
            complexity: Filter by complexity (e.g., "basic")

        Returns:
            List of dicts with operation_id, score, metadata

        Example:
            >>> results = rag.search("move robot to position")
            >>> results[0]['metadata']['name']
            'move_to_coordinate'
        """
        if self.query_engine is None:
            logger.error("Query engine not initialized. Call index_operations() first.")
            return []

        return self.query_engine.search(
            query=query,
            top_k=top_k,
            min_score=min_score,
            category_filter=category,
            complexity_filter=complexity,
        )

    def get_operation_context(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Get full operation context for LLM consumption.

        Returns comprehensive information about relevant operations
        including parameters, examples, preconditions, etc.

        Args:
            query: Natural language query describing the task
            top_k: Number of operations to include in context

        Returns:
            Dict with query, operations (full details), and summary

        Example:
            >>> context = rag.get_operation_context("move robot to pick up object")
            >>> context['summary']
            'Found 3 relevant operations for: move robot to pick up object'
            >>> context['operations'][0]['name']
            'move_to_coordinate'
        """
        if self.query_engine is None:
            logger.error("Query engine not initialized. Call index_operations() first.")
            return {
                "query": query,
                "num_results": 0,
                "summary": "RAG system not initialized",
                "operations": [],
            }

        return self.query_engine.get_operation_context(query, top_k=top_k)

    def get_operations_by_category(
        self, category: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all operations in a specific category.

        Args:
            category: Category name (e.g., "navigation", "manipulation")
            top_k: Maximum number to return

        Returns:
            List of operations in that category
        """
        if self.query_engine is None:
            logger.error("Query engine not initialized. Call index_operations() first.")
            return []

        return self.query_engine.search_by_category(category, top_k=top_k)

    def find_similar_operations(
        self, operation_id: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find operations similar to a given operation.

        Args:
            operation_id: Operation ID to find similar operations for
            top_k: Number of similar operations to return

        Returns:
            List of similar operations
        """
        if self.query_engine is None:
            logger.error("Query engine not initialized. Call index_operations() first.")
            return []

        return self.query_engine.find_similar_operations(operation_id, top_k=top_k)

    def is_ready(self) -> bool:
        """Check if RAG system is ready for queries"""
        return self.query_engine is not None and self.vector_store is not None

    def __repr__(self) -> str:
        ready = "ready" if self.is_ready() else "not indexed"
        num_ops = len(self.vector_store) if self.vector_store else 0
        return f"RAGSystem({ready}, operations={num_ops})"


# Export main classes and functions
__all__ = [
    "RAGSystem",
    "EmbeddingGenerator",
    "VectorStore",
    "OperationIndexer",
    "QueryEngine",
    "config",
    "compute_confidence_score",
    "get_confidence_level",
    "apply_confidence_boosting",
    "ConfidenceLevel",
]
