"""
vector_store.py — Hybrid vector store for the Self-Healing RAG pipeline.

Combines three retrieval methods into a single unified search interface:

  1. DENSE (FAISS HNSW): Semantic nearest-neighbor search on L2-normalised
     all-mpnet-base-v2 embeddings. Excellent for conceptual/paraphrase queries.

  2. SPARSE (BM25 Okapi): Classical TF-IDF-style keyword scoring. Excellent for
     exact terms: model IDs, names, codes, legal clauses.

  3. RECIPROCAL RANK FUSION (RRF): Merges the dense and sparse candidate lists
     into a single ranking without needing calibrated scores:
         score(d) = w_dense / (60 + rank_dense(d))
                  + w_sparse / (60 + rank_sparse(d))
     The Healer can shift w_dense / w_sparse when it detects bad_retrieval.

  4. CROSS-ENCODER RE-RANKING: The merged RRF top candidates are scored by a
     lightweight cross-encoder (ms-marco-MiniLM-L-6-v2, ~80 MB) that jointly
     encodes the query and each passage for much more accurate relevance scoring.

Reliability: BM25 and Cross-Encoder are initialized lazily; a missing index
returns an empty list rather than raising.
"""

import asyncio
import logging
import re
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# How many candidates to pass to the cross-encoder before final top-k cut
_RERANK_POOL = 20
_RRF_K = 60          # RRF constant — controls the slope of the rank penalty


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


class VectorStore:
    """
    Hybrid FAISS-HNSW + BM25 vector store with RRF fusion and cross-encoder
    re-ranking for the Self-Healing RAG pipeline.

    Attributes:
        dimension: Embedding vector dimension (768 for all-mpnet-base-v2).
        top_k: Default number of results to return per search.
        index: The underlying FAISS HNSW index.
        chunks: Stored chunk objects for metadata retrieval.
        bm25: BM25Okapi index rebuilt after every ingestion.
        cross_encoder: Sentence-transformer CrossEncoder for re-ranking.
    """

    def __init__(
        self,
        dimension: int = 768,
        top_k: int = 5,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Initialize the hybrid vector store.

        Args:
            dimension: Size of embedding vectors.
            top_k: Default number of neighbors to retrieve.
            M: HNSW graph connectivity.
            ef_construction: Build-time search depth.
            ef_search: Query-time search depth.
            reranker_model: HuggingFace cross-encoder model identifier.
        """
        self.dimension = dimension
        self.top_k = top_k

        # Dense index
        self.index = faiss.IndexHNSWFlat(dimension, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

        # Parallel storage for chunk objects (metadata + content)
        self.chunks: list = []

        # Sparse index — built lazily after ingestion
        self.bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

        # Cross-encoder for re-ranking
        print(f"[VectorStore] Loading cross-encoder '{reranker_model}'...")
        self.cross_encoder = CrossEncoder(reranker_model)
        print("[VectorStore] Cross-encoder loaded.")

        print(
            f"[VectorStore] Initialized HNSW index — dim={dimension}, M={M}, "
            f"efConstruction={ef_construction}, efSearch={ef_search}"
        )

    async def add(self, embeddings: np.ndarray, chunks: list) -> None:
        """
        Add embeddings and their corresponding chunks to the store asynchronously.
        Rebuilds the BM25 index to include the new chunks.

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

        # Rebuild BM25 with all current chunks
        self._tokenized_corpus = [_tokenize(c.content) for c in self.chunks]
        self.bm25 = BM25Okapi(self._tokenized_corpus)

        print(
            f"[VectorStore] Added {len(chunks)} vectors — total: {self.index.ntotal} | "
            f"BM25 index rebuilt with {len(self.chunks)} docs."
        )

    async def search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int | None = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> list[dict]:
        """
        Hybrid search: dense FAISS + sparse BM25, fused with RRF, re-ranked by
        a cross-encoder, asynchronously.

        Args:
            query_embedding: NumPy array of shape (1, dimension).
            query_text: Raw query string for BM25 and cross-encoder scoring.
            top_k: Number of final results to return. Defaults to self.top_k.
            dense_weight: RRF weight for FAISS dense candidates (0.0–1.0).
            sparse_weight: RRF weight for BM25 sparse candidates (0.0–1.0).

        Returns:
            List of dicts with 'content', 'metadata', 'score', 'rank'.
        """
        if self.index.ntotal == 0:
            print("[VectorStore] Warning: Index is empty, no results to return.")
            return []

        k = top_k or self.top_k
        pool = min(_RERANK_POOL, self.index.ntotal)

        # ── 1. DENSE retrieval ─────────────────────────────────────────────────
        distances, indices = await asyncio.to_thread(
            self.index.search, query_embedding, pool
        )
        dense_hits: list[int] = [
            int(idx) for idx in indices[0] if idx != -1
        ]

        # ── 2. SPARSE retrieval (BM25) ─────────────────────────────────────────
        sparse_hits: list[int] = []
        if self.bm25 is not None:
            query_tokens = _tokenize(query_text)
            bm25_scores = await asyncio.to_thread(
                self.bm25.get_scores, query_tokens
            )
            # Rank by descending score, take top pool candidates
            sparse_hits = list(
                np.argsort(bm25_scores)[::-1][:pool]
            )
            logger.debug(
                "[VectorStore] BM25 top-3 scores: %s",
                bm25_scores[sparse_hits[:3]],
            )
        else:
            print("[VectorStore] BM25 index not available — using dense only.")
            sparse_hits = dense_hits[:]

        # ── 3. RECIPROCAL RANK FUSION ──────────────────────────────────────────
        rrf_scores: dict[int, float] = {}
        for rank, idx in enumerate(dense_hits, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + dense_weight / (_RRF_K + rank)
        for rank, idx in enumerate(sparse_hits, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + sparse_weight / (_RRF_K + rank)

        # Sort by RRF score descending; take top pool candidates for re-ranking
        rrf_ranked: list[int] = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:pool]

        print(
            f"[VectorStore] RRF merged {len(dense_hits)} dense + {len(sparse_hits)} sparse "
            f"→ {len(rrf_ranked)} candidates (w_dense={dense_weight}, w_sparse={sparse_weight})"
        )

        # ── 4. CROSS-ENCODER RE-RANKING ────────────────────────────────────────
        candidate_chunks = [self.chunks[i] for i in rrf_ranked]
        pairs = [(query_text, c.content) for c in candidate_chunks]

        ce_scores: list[float] = await asyncio.to_thread(
            self.cross_encoder.predict, pairs
        )

        # Sort by cross-encoder score descending and take final top_k
        reranked = sorted(
            zip(rrf_ranked, candidate_chunks, ce_scores),
            key=lambda x: x[2],
            reverse=True,
        )[:k]

        results = []
        for rank, (idx, chunk, ce_score) in enumerate(reranked, start=1):
            results.append({
                "content": chunk.content,
                "metadata": chunk.metadata,
                "score": float(ce_score),
                "rank": rank,
            })

        print(
            f"[VectorStore] Re-ranked to top {len(results)} — "
            f"best CE score: {results[0]['score']:.3f}" if results else "[VectorStore] No results."
        )
        return results

    async def reset(self) -> None:
        """Clear the FAISS index, BM25 index, and all stored chunks asynchronously."""
        await asyncio.to_thread(self.index.reset)
        self.chunks.clear()
        self._tokenized_corpus.clear()
        self.bm25 = None
        print("[VectorStore] Index reset (FAISS + BM25 cleared).")
