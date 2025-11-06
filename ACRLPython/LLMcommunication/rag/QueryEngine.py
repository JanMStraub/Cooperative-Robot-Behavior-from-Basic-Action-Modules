"""
Query Engine for RAG System
============================

Semantic search and retrieval over operations.
"""

from typing import List, Dict, Any, Optional
import logging
from ..operations.Registry import OperationRegistry, get_global_registry
from .Embeddings import EmbeddingGenerator
from .VectorStore import VectorStore
from .Config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Semantic search engine for robot operations.

    This class provides natural language search over operations,
    returning relevant operations ranked by similarity.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        registry: Optional[OperationRegistry] = None,
    ):
        """
        Initialize the query engine.

        Args:
            vector_store: Vector store with indexed operations
            embedding_generator: Embedding generator (default: new instance)
            registry: Operation registry for full operation details
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.registry = registry or get_global_registry()

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        category_filter: Optional[str] = None,
        complexity_filter: Optional[str] = None,
        include_full_operation: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for operations using natural language query.

        Args:
            query: Natural language search query
            top_k: Number of results to return (default from config)
            min_score: Minimum similarity score (default from config)
            category_filter: Filter by category (e.g., "navigation")
            complexity_filter: Filter by complexity (e.g., "basic")
            include_full_operation: Include full BasicOperation objects in results

        Returns:
            List of dicts with operation_id, score, metadata, and optionally full operation

        Example:
            >>> engine = QueryEngine(vector_store)
            >>> results = engine.search("move robot to position", top_k=3)
            >>> results[0]['metadata']['name']
            'move_to_coordinate'
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        # Generate query embedding
        logger.debug(f"Searching for: '{query}'")
        query_embedding = self.embedding_generator.generate_embedding(query)

        # Search vector store
        k = top_k if top_k is not None else config.DEFAULT_TOP_K
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=k,
            min_score=min_score,
            category_filter=category_filter,
            complexity_filter=complexity_filter,
        )

        # Optionally include full operation objects
        if include_full_operation:
            for result in results:
                op_id = result["operation_id"]
                operation = self.registry.get_operation(op_id)
                if operation:
                    result["operation"] = operation

        logger.debug(f"Found {len(results)} results for query")
        return results

    def get_operation_context(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Get full context for LLM consumption.

        Returns operation details, parameters, examples, etc. for the
        most relevant operations based on the query.

        Args:
            query: Natural language query
            top_k: Number of operations to include

        Returns:
            Dict with query, results (with full operation data), and summary

        Example:
            >>> engine = QueryEngine(vector_store)
            >>> context = engine.get_operation_context("move robot")
            >>> context['summary']
            'Found 3 relevant operations for: move robot'
        """
        results = self.search(query, top_k=top_k, include_full_operation=True)

        # Build context with full operation details
        operations_context = []
        for result in results:
            operation = result.get("operation")
            if operation:
                operations_context.append(
                    {
                        "operation_id": operation.operation_id,
                        "name": operation.name,
                        "category": operation.category.value,
                        "complexity": operation.complexity.value,
                        "description": operation.description,
                        "parameters": [
                            {
                                "name": p.name,
                                "type": p.type,
                                "description": p.description,
                                "required": p.required,
                                "default": p.default,
                                "valid_range": p.valid_range,
                            }
                            for p in operation.parameters
                        ],
                        "usage_examples": operation.usage_examples,
                        "preconditions": operation.preconditions,
                        "postconditions": operation.postconditions,
                        "failure_modes": operation.failure_modes,
                        "similarity_score": result["score"],
                    }
                )

        return {
            "query": query,
            "num_results": len(operations_context),
            "summary": f"Found {len(operations_context)} relevant operations for: {query}",
            "operations": operations_context,
        }

    def search_by_category(
        self, category: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all operations in a specific category.

        Args:
            category: Category name (e.g., "navigation", "manipulation")
            top_k: Maximum number to return

        Returns:
            List of operation results
        """
        # Use a generic query since we're filtering by category
        return self.search(
            query=f"{category} operations",
            top_k=top_k or 50,  # Large number to get all in category
            category_filter=category,
        )

    def find_similar_operations(
        self, operation_id: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find operations similar to a given operation.

        Args:
            operation_id: Operation to find similar operations for
            top_k: Number of similar operations to return

        Returns:
            List of similar operations (excluding the input operation)
        """
        # Get the operation from vector store
        op_data = self.vector_store.get_operation(operation_id)
        if not op_data:
            logger.warning(f"Operation '{operation_id}' not found in vector store")
            return []

        # Search using the operation's embedding
        results = self.vector_store.search(
            query_embedding=op_data["embedding"], top_k=top_k + 1  # +1 to exclude self
        )

        # Filter out the input operation itself
        return [r for r in results if r["operation_id"] != operation_id][:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the query engine.

        Returns:
            Dict with vector store stats and embedding info
        """
        return {
            "vector_store_stats": self.vector_store.get_stats(),
            "embedding_generator": repr(self.embedding_generator),
            "using_lm_studio": self.embedding_generator.is_using_lm_studio(),
        }
