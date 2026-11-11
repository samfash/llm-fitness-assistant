"""Hybrid retrieval: BM25 + optional dense embeddings, fused with Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache

from .bm25 import BM25
from .corpus import DEFAULT_DIR, Chunk, load_chunks
from .dense import CrossEncoderReranker, DenseIndex, Embedder, SentenceTransformerEmbedder
from .text import tokenize

log = logging.getLogger(__name__)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25: float = 0.0
    dense: float | None = None
    ranks: dict = field(default_factory=dict)


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None, reranker=None,
                 rrf_k: int = 60, bm25_min: float = 2.0, dense_min: float = 0.30, max_per_doc: int = 2):
        self.chunks = chunks
        self.bm25 = BM25([tokenize(c.index_text) for c in chunks])
        self.dense = DenseIndex(embedder, [c.index_text for c in chunks]) if embedder else None
        self.reranker, self.rrf_k = reranker, rrf_k
        self.bm25_min, self.dense_min, self.max_per_doc = bm25_min, dense_min, max_per_doc

    @property
    def mode(self) -> str:
        return "hybrid" if self.dense else "bm25"

    def search(self, query: str, k: int = 4, fetch_k: int = 20) -> list[Hit]:
        q_tokens = tokenize(query)
        if not q_tokens and not self.dense:
            return []
        b = self.bm25.scores(q_tokens) if q_tokens else [0.0] * len(self.chunks)
        d = self.dense.scores(query) if self.dense else None

        by_bm25 = [i for i in sorted(range(len(b)), key=lambda i: b[i], reverse=True) if b[i] > 0][:fetch_k]
        ranks: dict[int, dict] = {i: {"bm25": r} for r, i in enumerate(by_bm25, start=1)}
        if d is not None:
            for r, i in enumerate(sorted(range(len(d)), key=lambda i: d[i], reverse=True)[:fetch_k], start=1):
                ranks.setdefault(i, {})["dense"] = r

        hits = []
        for i, rk in ranks.items():
            # relevance gate: RRF has no absolute scale, so require real lexical OR semantic evidence
            if b[i] >= self.bm25_min or (d is not None and d[i] >= self.dense_min):
                score = sum(1.0 / (self.rrf_k + r) for r in rk.values())
                hits.append(Hit(self.chunks[i], score, b[i], d[i] if d is not None else None, rk))
        hits.sort(key=lambda h: h.score, reverse=True)

        if self.reranker and hits:
            hits = self.reranker.rerank(query, hits[:fetch_k])

        out, per_doc = [], {}
        for h in hits:  # diversity: at most `max_per_doc` chunks from one document
            if per_doc.get(h.chunk.doc_id, 0) < self.max_per_doc:
                out.append(h)
                per_doc[h.chunk.doc_id] = per_doc.get(h.chunk.doc_id, 0) + 1
            if len(out) == k:
                break
        return out


def build_retriever(directory=DEFAULT_DIR, dense: bool | None = None, rerank: bool | None = None) -> HybridRetriever:
    """dense=None -> try real embeddings unless RAG_DENSE=off; falls back to BM25-only if unavailable."""
    chunks = load_chunks(directory)
    want_dense = (os.getenv("RAG_DENSE", "on").lower() != "off") if dense is None else dense
    embedder = None
    if want_dense:
        try:
            embedder = SentenceTransformerEmbedder()
        except Exception as e:  # not installed, offline, download failed...
            log.warning("Dense embeddings unavailable (%s); using BM25 only", e)
    want_rerank = (os.getenv("RAG_RERANK", "0") == "1") if rerank is None else rerank
    reranker = None
    if want_rerank:
        try:
            reranker = CrossEncoderReranker()
        except Exception as e:
            log.warning("Reranker unavailable (%s); skipping", e)
    return HybridRetriever(chunks, embedder, reranker)


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return build_retriever()
