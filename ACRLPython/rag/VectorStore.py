"""
Vector Store for RAG System
============================

In-memory vector storage with pickle-based persistence and cosine similarity search.
"""

from typing import List, Dict, Any, Optional
import pickle
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .ConfidenceScorer import apply_confidence_boosting, get_category_min_score

# Import config
# Import config - try both import styles
try:
    import LLMConfig as config
except ImportError:
    from .. import LLMConfig as config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    In-memory vector store with cosine similarity search.

    Stores operation embeddings with metadata and provides efficient
    similarity search functionality.
    """

    def __init__(self):
        """Initialize empty vector store"""
        self.vectors: np.ndarray = np.array([])  # Shape: (n_ops, embedding_dim)
        self.operation_ids: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.embedding_dimension: Optional[int] = None

    def add_operation(
        self, operation_id: str, embedding: np.ndarray, metadata: Dict[str, Any]
    ):
        """
        Add an operation to the vector store.

        Args:
            operation_id: Unique operation identifier
            embedding: Embedding vector
            metadata: Operation metadata (name, category, etc.)

        Example:
            >>> store = VectorStore()
            >>> embedding = np.array([0.1, 0.2, 0.3])
            >>> store.add_operation(
            ...     "op_001",
            ...     embedding,
            ...     {"name": "move_to_coordinate", "category": "navigation"}
            ... )
        """
        # Set embedding dimension on first add
        if self.embedding_dimension is None:
            self.embedding_dimension = len(embedding)
        elif len(embedding) != self.embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {len(embedding)}"
            )

        # Add to lists
        self.operation_ids.append(operation_id)
        self.metadata.append(metadata)

        # Add to vectors array
        if len(self.vectors) == 0:
            self.vectors = embedding.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, embedding.reshape(1, -1)])

        logger.debug(f"Added operation '{operation_id}' to vector store")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_score: Optional[float] = None,
        category_filter: Optional[str] = None,
        complexity_filter: Optional[str] = None,
        query_text: str = "",
        enable_confidence: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar operations using cosine similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            min_score: Minimum similarity score to include
            category_filter: Filter by operation category
            complexity_filter: Filter by operation complexity
            query_text: Original query for confidence scoring
            enable_confidence: Whether to apply confidence scoring

        Returns:
            List of dicts with keys: operation_id, score, metadata, confidence

        Example:
            >>> store = VectorStore()
            >>> # ... add operations ...
            >>> query = np.array([0.15, 0.25, 0.35])
            >>> results = store.search(query, top_k=3)
            >>> results[0]['operation_id']
            'op_001'
        """
        if len(self.vectors) == 0:
            logger.warning("Vector store is empty")
            return []

        if len(query_embedding) != self.embedding_dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.embedding_dimension}, got {len(query_embedding)}"
            )

        # Compute cosine similarity
        query = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query, self.vectors)[0]

        # Create results with metadata
        results = []
        for idx, score in enumerate(similarities):
            # Apply filters
            if (
                category_filter
                and self.metadata[idx].get("category") != category_filter
            ):
                continue
            if (
                complexity_filter
                and self.metadata[idx].get("complexity") != complexity_filter
            ):
                continue

            # Apply min score threshold (use category-specific if available)
            if min_score is not None:
                min_threshold = min_score
            elif category_filter:
                min_threshold = get_category_min_score(category_filter)
            else:
                min_threshold = config.RAG_MIN_SIMILARITY_SCORE

            if score < min_threshold:
                continue

            results.append(
                {
                    "operation_id": self.operation_ids[idx],
                    "score": float(score),
                    "metadata": self.metadata[idx],
                }
            )

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        # Apply confidence scoring if enabled
        if enable_confidence and config.RAG_ENABLE_CONFIDENCE_SCORING and results:
            results = apply_confidence_boosting(
                results,
                query_text=query_text,
                category_filter=category_filter,
                complexity_filter=complexity_filter,
            )

        # Return top-k
        return results[:top_k]

    def get_operation(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get operation by ID.

        Args:
            operation_id: Operation identifier

        Returns:
            Dict with operation_id, embedding, metadata or None if not found
        """
        try:
            idx = self.operation_ids.index(operation_id)
            return {
                "operation_id": operation_id,
                "embedding": self.vectors[idx],
                "metadata": self.metadata[idx],
            }
        except ValueError:
            return None

    def save(self, file_path: Optional[str] = None):
        """
        Save vector store to pickle file.

        Args:
            file_path: Path to save file (default from config)

        Example:
            >>> store = VectorStore()
            >>> # ... add operations ...
            >>> store.save()
            Saved vector store to .rag_index.pkl
        """
        path = file_path or config.RAG_VECTOR_STORE_PATH

        data = {
            "vectors": self.vectors,
            "operation_ids": self.operation_ids,
            "metadata": self.metadata,
            "embedding_dimension": self.embedding_dimension,
        }

        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise

    @classmethod
    def load(cls, file_path: Optional[str] = None) -> "VectorStore":
        """
        Load vector store from pickle file.

        Args:
            file_path: Path to load file (default from config)

        Returns:
            Loaded VectorStore instance

        Example:
            >>> store = VectorStore.load()
            Loaded vector store from .rag_index.pkl (5 operations)
        """
        path = file_path or config.RAG_VECTOR_STORE_PATH

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            store = cls()
            store.vectors = data["vectors"]
            store.operation_ids = data["operation_ids"]
            store.metadata = data["metadata"]
            store.embedding_dimension = data["embedding_dimension"]

            return store

        except FileNotFoundError:
            logger.warning(f"Vector store file not found: {path}")
            return cls()  # Return empty store
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dict with num_operations, embedding_dimension, categories, etc.
        """
        categories = {}
        complexities = {}

        for meta in self.metadata:
            cat = meta.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

            comp = meta.get("complexity", "unknown")
            complexities[comp] = complexities.get(comp, 0) + 1

        return {
            "num_operations": len(self.operation_ids),
            "embedding_dimension": self.embedding_dimension,
            "categories": categories,
            "complexities": complexities,
            "operation_ids": self.operation_ids,
        }

    def clear(self):
        """Clear all data from the vector store"""
        self.vectors = np.array([])
        self.operation_ids = []
        self.metadata = []
        self.embedding_dimension = None
        logger.info("Cleared vector store")

    def __len__(self) -> int:
        """Return number of operations in store"""
        return len(self.operation_ids)

    def __repr__(self) -> str:
        return f"VectorStore(operations={len(self)}, dim={self.embedding_dimension})"
