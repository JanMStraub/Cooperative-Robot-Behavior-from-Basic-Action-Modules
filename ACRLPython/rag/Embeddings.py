"""
Embedding Generation via LM Studio
===================================

This module provides embedding generation using LM Studio's OpenAI-compatible API.
Falls back to TF-IDF if LM Studio is unavailable.
"""

from typing import List, Optional
import numpy as np
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

# Import config
try:
    from config.Rag import (
        RAG_LM_STUDIO_URL,
        RAG_LM_STUDIO_API_KEY,
        RAG_LM_STUDIO_MODEL,
        RAG_EMBEDDING_TIMEOUT,
        RAG_EMBEDDING_DIMENSION,
        RAG_EMBEDDING_BATCH_SIZE,
        RAG_USE_TFIDF_FALLBACK,
        RAG_TFIDF_MAX_FEATURES,
    )
except ImportError:
    from ..config.Rag import (
        RAG_LM_STUDIO_URL,
        RAG_LM_STUDIO_API_KEY,
        RAG_LM_STUDIO_MODEL,
        RAG_EMBEDDING_TIMEOUT,
        RAG_EMBEDDING_DIMENSION,
        RAG_EMBEDDING_BATCH_SIZE,
        RAG_USE_TFIDF_FALLBACK,
        RAG_TFIDF_MAX_FEATURES,
    )

# Configure logging
from core.LoggingSetup import get_logger
logger = get_logger(__name__)


class EmbeddingGenerator:
    """
    Generate embeddings using LM Studio or TF-IDF fallback.

    This class provides a unified interface for embedding generation,
    using LM Studio's local API when available and falling back to TF-IDF.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the embedding generator.

        Args:
            base_url: LM Studio base URL (default from config)
            api_key: API key (default from config, LM Studio doesn't need real key)
            model: Embedding model name (default from config)
        """
        self.base_url = base_url or RAG_LM_STUDIO_URL
        self.api_key = api_key or RAG_LM_STUDIO_API_KEY
        self.model = model or RAG_LM_STUDIO_MODEL

        self.client: Optional[OpenAI] = None
        self.use_lm_studio = True
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None

        # Try to initialize LM Studio client
        self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenAI client pointing to LM Studio"""
        try:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

            # Test connection with a simple embedding
            assert self.client is not None  # For type checker
            test_response = self.client.embeddings.create(
                input="test", model=self.model, timeout=RAG_EMBEDDING_TIMEOUT
            )

            if test_response and test_response.data:
                self.use_lm_studio = True
            else:
                raise Exception("Invalid response from LM Studio")

        except Exception as e:
            logger.warning(f"⚠ Failed to connect to LM Studio: {e}")
            if RAG_USE_TFIDF_FALLBACK:
                logger.info("  Falling back to TF-IDF embeddings")
                self.use_lm_studio = False
                self._initialize_tfidf()
            else:
                raise Exception(f"LM Studio unavailable and fallback disabled: {e}")

    def _initialize_tfidf(self):
        """Initialize TF-IDF vectorizer as fallback"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=RAG_TFIDF_MAX_FEATURES,
            stop_words="english",
            ngram_range=(1, 2),  # Unigrams and bigrams
        )

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as numpy array

        Example:
            >>> generator = EmbeddingGenerator()
            >>> embedding = generator.generate_embedding("move robot to position")
            >>> embedding.shape
            (768,)
        """
        return self.generate_embeddings([text])[0]

    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors as numpy arrays

        Example:
            >>> generator = EmbeddingGenerator()
            >>> embeddings = generator.generate_embeddings([
            ...     "move robot to position",
            ...     "grip the object"
            ... ])
            >>> len(embeddings)
            2
        """
        if not texts:
            return []

        if self.use_lm_studio:
            return self._generate_lm_studio_embeddings(texts)
        else:
            return self._generate_tfidf_embeddings(texts)

    def _generate_lm_studio_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using LM Studio API"""
        embeddings = []

        # Check if client is available
        if self.client is None:
            logger.error("LM Studio client not initialized")
            return [
                np.zeros(RAG_EMBEDDING_DIMENSION, dtype=np.float32)
                for _ in texts
            ]

        # Process in batches
        batch_size = RAG_EMBEDDING_BATCH_SIZE
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model,
                    timeout=RAG_EMBEDDING_TIMEOUT,
                )

                for data in response.data:
                    embeddings.append(np.array(data.embedding, dtype=np.float32))

                logger.debug(
                    f"Generated {len(batch)} embeddings (batch {i // batch_size + 1})"
                )

            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}")
                # Return zero vectors for failed batch
                for _ in batch:
                    embeddings.append(
                        np.zeros(RAG_EMBEDDING_DIMENSION, dtype=np.float32)
                    )

        return embeddings

    def _generate_tfidf_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using TF-IDF vectorizer"""
        if self.tfidf_vectorizer is None:
            self._initialize_tfidf()

        # Check again after initialization
        if self.tfidf_vectorizer is None:
            logger.error("TF-IDF vectorizer initialization failed")
            return [
                np.zeros(RAG_TFIDF_MAX_FEATURES, dtype=np.float32) for _ in texts
            ]

        try:
            # Fit and transform if not already fitted
            if not hasattr(self.tfidf_vectorizer, "vocabulary_"):
                vectors = self.tfidf_vectorizer.fit_transform(texts)
            else:
                vectors = self.tfidf_vectorizer.transform(texts)

            # Convert sparse matrix to dense numpy arrays and pad to max_features
            embeddings = []
            for i in range(vectors.shape[0]):
                embedding = (
                    np.array(vectors.getrow(i).todense()).flatten().astype(np.float32)
                )

                # Pad to TFIDF_MAX_FEATURES if necessary
                if len(embedding) < RAG_TFIDF_MAX_FEATURES:
                    embedding = np.pad(
                        embedding,
                        (0, RAG_TFIDF_MAX_FEATURES - len(embedding)),
                        mode="constant",
                        constant_values=0,
                    )

                embeddings.append(embedding)

            logger.debug(f"Generated {len(embeddings)} TF-IDF embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating TF-IDF embeddings: {e}")
            # Return zero vectors
            return [
                np.zeros(RAG_TFIDF_MAX_FEATURES, dtype=np.float32) for _ in texts
            ]

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this generator.

        Returns:
            Embedding dimension size
        """
        if self.use_lm_studio:
            return RAG_EMBEDDING_DIMENSION
        else:
            return RAG_TFIDF_MAX_FEATURES

    def is_using_lm_studio(self) -> bool:
        """Check if using LM Studio (True) or TF-IDF fallback (False)"""
        return self.use_lm_studio

    def __repr__(self) -> str:
        if self.use_lm_studio:
            return f"EmbeddingGenerator(lm_studio, model={self.model}, url={self.base_url})"
        else:
            return f"EmbeddingGenerator(tfidf, max_features={RAG_TFIDF_MAX_FEATURES})"
