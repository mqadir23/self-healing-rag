"""
embedder.py — Sentence-transformer embeddings for the Self-Healing RAG pipeline.

Uses all-mpnet-base-v2 (768-dim) with configurable mini-batch encoding
to control memory usage on large document sets.
"""

import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Wraps a SentenceTransformer model to encode text chunks into dense vectors.

    Attributes:
        model_name: The HuggingFace model identifier.
        batch_size: Number of texts to encode per mini-batch.
    """

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        print(f"[Embedder] Loading model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] Model loaded — dimension: {self.dimension}")

    def _embed_texts_sync(self, texts: list[str]) -> np.ndarray:
        """Synchronously encode texts."""
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)

        all_embeddings = []
        total = len(texts)

        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch = texts[start:end]
            embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalize for cosine similarity
            )
            all_embeddings.append(embeddings)
            print(f"[Embedder] Encoded batch {start + 1}–{end} / {total}")

        result = np.vstack(all_embeddings).astype(np.float32)
        print(f"[Embedder] Finished — {result.shape[0]} embeddings, dim={result.shape[1]}")
        return result

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of text strings into a NumPy matrix of embeddings asynchronously.
        """
        return await asyncio.to_thread(self._embed_texts_sync, texts)

    def _embed_query_sync(self, query: str) -> np.ndarray:
        """Synchronously encode query."""
        embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)

    async def embed_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string into an embedding vector asynchronously.
        """
        return await asyncio.to_thread(self._embed_query_sync, query)
