"""
Index Builder for RAG System
=============================

Build searchable index from operations registry.
"""

from typing import Optional
import logging

from operations.Registry import OperationRegistry, get_global_registry

from .Embeddings import EmbeddingGenerator
from .VectorStore import VectorStore
from .Config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OperationIndexer:
    """
    Build searchable index from operations registry.

    This class takes operations from the registry, generates embeddings
    for their RAG documents, and stores them in a vector store.
    """

    def __init__(
        self,
        registry: Optional[OperationRegistry] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
    ):
        """
        Initialize the indexer.

        Args:
            registry: Operation registry (default: global registry)
            embedding_generator: Embedding generator (default: new instance)
        """
        self.registry = registry or get_global_registry()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()

    def build_index(self, save: bool = True) -> VectorStore:
        """
        Build index from all operations in the registry.

        Args:
            save: Whether to save the index to disk (default: True)

        Returns:
            Populated VectorStore

        Example:
            >>> indexer = OperationIndexer()
            >>> store = indexer.build_index()
            Building index for 5 operations...
            Generated embeddings for 5 operations
            Saved vector store to .rag_index.pkl (5 operations)
        """
        operations = self.registry.get_all_operations()

        if not operations:
            logger.warning("No operations found in registry")
            return VectorStore()

        logger.info(f"Building index for {len(operations)} operations...")

        # Create new vector store
        store = VectorStore()

        # Collect texts to embed
        texts_to_embed = []
        operation_data = []

        for op in operations:
            # Generate RAG document text
            rag_text = op.to_rag_document()
            texts_to_embed.append(rag_text)

            # Store operation data
            operation_data.append(
                {
                    "operation_id": op.operation_id,
                    "metadata": {
                        "name": op.name,
                        "category": op.category.value,
                        "complexity": op.complexity.value,
                        "description": op.description,
                        "average_duration_ms": op.average_duration_ms,
                        "success_rate": op.success_rate,
                    },
                }
            )

        # Generate embeddings for all operations
        logger.info(f"Generating embeddings for {len(texts_to_embed)} operations...")
        embeddings = self.embedding_generator.generate_embeddings(texts_to_embed)

        # Add to vector store
        for data, embedding in zip(operation_data, embeddings):
            store.add_operation(
                operation_id=data["operation_id"],
                embedding=embedding,
                metadata=data["metadata"],
            )

        logger.info(f"✓ Index built with {len(store)} operations")

        # Save to disk
        if save and config.AUTO_SAVE_INDEX:
            store.save()

        return store

    def rebuild_index(self) -> VectorStore:
        """
        Rebuild index from scratch (clears existing index).

        Returns:
            New VectorStore with fresh index
        """
        logger.info("Rebuilding index from scratch...")
        return self.build_index(save=True)

    def update_index(self, existing_store: VectorStore) -> VectorStore:
        """
        Update existing index with new/changed operations.

        Args:
            existing_store: Existing vector store to update

        Returns:
            Updated VectorStore

        Note:
            Currently rebuilds entire index. Incremental updates
            could be added in the future.
        """
        logger.info("Updating index (full rebuild)...")
        # For now, just rebuild the entire index
        # Future enhancement: Implement incremental updates for better performance
        return self.build_index(save=True)

    def get_indexer_stats(self) -> dict:
        """
        Get statistics about the indexer and its components.

        Returns:
            Dict with registry info and embedding generator info
        """
        operations = self.registry.get_all_operations()

        return {
            "num_operations": len(operations),
            "embedding_generator": repr(self.embedding_generator),
            "embedding_dimension": self.embedding_generator.get_embedding_dimension(),
            "using_lm_studio": self.embedding_generator.is_using_lm_studio(),
        }


def build_index_from_registry(
    registry: Optional[OperationRegistry] = None,
    save_path: Optional[str] = None,
) -> VectorStore:
    """
    Convenience function to build index from registry.

    Args:
        registry: Operation registry (default: global registry)
        save_path: Path to save index (default from config)

    Returns:
        Populated VectorStore

    Example:
        >>> from rag.indexer import build_index_from_registry
        >>> store = build_index_from_registry()
        Building index for 5 operations...
        ✓ Index built with 5 operations
    """
    indexer = OperationIndexer(registry=registry)
    store = indexer.build_index(save=True)

    # Save to custom path if specified
    if save_path:
        store.save(save_path)

    return store
