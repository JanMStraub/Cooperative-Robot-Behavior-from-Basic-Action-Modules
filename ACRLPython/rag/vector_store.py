#!/usr/bin/env python3
"""In-memory vector store with pickle-based persistence and cosine similarity search."""

from typing import List, Dict, Any, Optional
import pickle
import threading
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .ConfidenceScorer import apply_confidence_boosting, get_category_min_score

try:
    from config.Rag import (
        RAG_MIN_SIMILARITY_SCORE,
        RAG_ENABLE_CONFIDENCE_SCORING,
        RAG_VECTOR_STORE_PATH,
        RAG_EMBEDDING_DIMENSION,
    )
except ImportError:
    from ..config.Rag import (
        RAG_MIN_SIMILARITY_SCORE,
        RAG_ENABLE_CONFIDENCE_SCORING,
        RAG_VECTOR_STORE_PATH,
        RAG_EMBEDDING_DIMENSION,
    )

from core.LoggingSetup import get_logger

logger = get_logger(__name__)


class VectorStore:
    """In-memory vector store with cosine similarity search."""

    def __init__(self):
        self.vectors: np.ndarray = np.array([])  # Shape: (n_ops, embedding_dim)
        self.operation_ids: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.embedding_dimension: Optional[int] = None
        self._lock = threading.RLock()  # Thread safety for concurrent access
        # Staging list for deferred vstack: embeddings are appended here and
        # flushed into self.vectors via _flush_pending_vectors() to avoid O(n²)
        # repeated vstack calls during bulk add_operation() sequences.
        self._pending_vectors: List[np.ndarray] = []

    def add_operation(
        self, operation_id: str, embedding: np.ndarray, metadata: Dict[str, Any]
    ):
        """Add an operation embedding and metadata to the store."""
        with self._lock:
            # Set embedding dimension on first add; validate against config
            if self.embedding_dimension is None:
                self.embedding_dimension = len(embedding)
                if (
                    RAG_EMBEDDING_DIMENSION
                    and self.embedding_dimension != RAG_EMBEDDING_DIMENSION
                ):
                    logger.warning(
                        f"Embedding dimension {self.embedding_dimension} != configured "
                        f"RAG_EMBEDDING_DIMENSION={RAG_EMBEDDING_DIMENSION}. "
                        "Check that the embedding model matches the config."
                    )
            elif len(embedding) != self.embedding_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {len(embedding)}"
                )

            self.operation_ids.append(operation_id)
            self.metadata.append(metadata)

            # Stage the embedding for deferred vstack; call _flush_pending_vectors()
            # when done with a bulk insert to materialize self.vectors in one pass.
            self._pending_vectors.append(embedding.reshape(1, -1))

            logger.debug(f"Added operation '{operation_id}' to vector store")

    def _flush_pending_vectors(self):
        """
        Materialize staged embeddings into self.vectors.

        Collects all embeddings appended since the last flush and builds the
        final matrix with a single np.vstack() call, avoiding the O(n²)
        cost of stacking inside add_operation() for bulk inserts.

        Must be called with self._lock already held.
        """
        if not self._pending_vectors:
            return

        if len(self.vectors) == 0:
            self.vectors = np.vstack(self._pending_vectors)
        else:
            self.vectors = np.vstack([self.vectors] + self._pending_vectors)
        self._pending_vectors = []

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
        """Search for similar operations using cosine similarity."""
        with self._lock:
            self._flush_pending_vectors()

            if len(self.vectors) == 0:
                logger.warning("Vector store is empty")
                return []

            if len(query_embedding) != self.embedding_dimension:
                raise ValueError(
                    f"Query embedding dimension mismatch: expected {self.embedding_dimension}, got {len(query_embedding)}"
                )

            # Compute cosine similarity; suppress divide/overflow warnings that
            # arise from zero-norm TF-IDF vectors (e.g. all-stopword documents).
            # nan_to_num converts those NaN scores to 0.0 (no match).
            query = query_embedding.reshape(1, -1)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                similarities = cosine_similarity(query, self.vectors)[0]
            similarities = np.nan_to_num(similarities, nan=0.0, posinf=1.0, neginf=0.0)

            results = []
            for idx, score in enumerate(similarities):
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
                    min_threshold = RAG_MIN_SIMILARITY_SCORE

                if score < min_threshold:
                    continue

                results.append(
                    {
                        "operation_id": self.operation_ids[idx],
                        "score": float(score),
                        "metadata": self.metadata[idx],
                    }
                )

            results.sort(key=lambda x: x["score"], reverse=True)

            if enable_confidence and RAG_ENABLE_CONFIDENCE_SCORING and results:
                results = apply_confidence_boosting(
                    results,
                    query_text=query_text,
                    category_filter=category_filter,
                    complexity_filter=complexity_filter,
                )

            return results[:top_k]

    def get_operation(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Return operation by ID, or None if not found."""
        with self._lock:
            self._flush_pending_vectors()
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
        """Save vector store to pickle file."""
        path = file_path or RAG_VECTOR_STORE_PATH

        with self._lock:
            self._flush_pending_vectors()
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
        """Load vector store from pickle file, or return empty store if not found."""
        path = file_path or RAG_VECTOR_STORE_PATH

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            store = cls()
            with store._lock:
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

    def update_operation_metadata(
        self, operation_id: str, metadata_update: Dict[str, Any]
    ) -> bool:
        """Merge new fields into the metadata of an existing operation."""
        with self._lock:
            try:
                idx = self.operation_ids.index(operation_id)
                self.metadata[idx].update(metadata_update)
                return True
            except ValueError:
                return False

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the vector store."""
        with self._lock:
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
        with self._lock:
            self.vectors = np.array([])
            self.operation_ids = []
            self.metadata = []
            self.embedding_dimension = None
            self._pending_vectors = []
            logger.info("Cleared vector store")

    def __len__(self) -> int:
        with self._lock:
            return len(self.operation_ids)

    def __repr__(self) -> str:
        return f"VectorStore(operations={len(self)}, dim={self.embedding_dimension})"
