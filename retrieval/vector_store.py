"""
vector_store.py — FAISS HNSW vector store for the Self-Healing RAG pipeline.

Uses IndexHNSWFlat for fast, near-exact approximate nearest neighbor search.
Stores chunk metadata alongside embeddings for retrieval traceability.
"""

import asyncio
import faiss
import numpy as np


class VectorStore:
    """
    FAISS-backed vector store using HNSW (Hierarchical Navigable Small World) index.

    HNSW provides near-exact recall (~99%+) with fast retrieval — ideal for
    a self-healing pipeline where retrieval quality directly impacts whether
    the healing loop gets triggered.

    Attributes:
        dimension: Embedding vector dimension (768 for all-mpnet-base-v2).
        top_k: Default number of results to return per search.
        index: The underlying FAISS HNSW index.
        chunks: Stored chunk objects for metadata retrieval.
    """

    def __init__(
        self,
        dimension: int = 768,
        top_k: int = 5,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
    ):
        """
        Initialize the HNSW vector store.

        Args:
            dimension: Size of embedding vectors.
            top_k: Default number of neighbors to retrieve.
            M: HNSW graph connectivity (higher = better recall, more memory).
            ef_construction: Build-time search depth (higher = better index quality).
            ef_search: Query-time search depth (higher = better recall, slower).
        """
        self.dimension = dimension
        self.top_k = top_k

        # Build HNSW index with inner product (cosine sim on L2-normalized vectors)
        self.index = faiss.IndexHNSWFlat(dimension, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

        # Parallel storage for chunk objects (metadata + content)
        self.chunks = []

        print(f"[VectorStore] Initialized HNSW index — dim={dimension}, M={M}, "
              f"efConstruction={ef_construction}, efSearch={ef_search}")

    async def add(self, embeddings: np.ndarray, chunks: list) -> None:
        """
        Add embeddings and their corresponding chunks to the store asynchronously.

        Args:
            embeddings: NumPy array of shape (n, dimension).
            chunks: List of Chunk objects (from chunker.py), same length as embeddings.
        """
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks"
            )

        await asyncio.to_thread(self.index.add, embeddings)
        self.chunks.extend(chunks)
        print(f"[VectorStore] Added {len(chunks)} vectors — total: {self.index.ntotal}")

    async def search(self, query_embedding: np.ndarray, top_k: int = None) -> list[dict]:
        """
        Search for the most similar chunks to a query embedding asynchronously.

        Args:
            query_embedding: NumPy array of shape (1, dimension).
            top_k: Number of results to return. Defaults to self.top_k.

        Returns:
            List of dicts, each containing:
              - "content": The chunk text.
              - "metadata": The chunk metadata dict.
              - "score": Similarity score (higher = more similar).
              - "rank": 1-indexed rank in results.
        """
        if self.index.ntotal == 0:
            print("[VectorStore] Warning: Index is empty, no results to return.")
            return []

        k = top_k or self.top_k
        # Clamp k to the number of stored vectors
        k = min(k, self.index.ntotal)

        distances, indices = await asyncio.to_thread(self.index.search, query_embedding, k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
            if idx == -1:
                continue  # FAISS returns -1 for missing neighbors
            chunk = self.chunks[idx]
            results.append({
                "content": chunk.content,
                "metadata": chunk.metadata,
                "score": float(score),
                "rank": rank,
            })

        return results

    async def reset(self) -> None:
        """Clear the index and all stored chunks asynchronously."""
        await asyncio.to_thread(self.index.reset)
        self.chunks.clear()
        print("[VectorStore] Index reset.")
