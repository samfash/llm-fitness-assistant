"""Dense retrieval: pluggable embedders. Embeddings must be L2-normalised so dot product == cosine."""
from __future__ import annotations

import math
import zlib
from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Real semantic embeddings (downloads the model on first use)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True).tolist()


class HashingEmbedder:
    """Deterministic hashed bag-of-words (+bigrams). Lexical, NOT semantic: for tests and plumbing only."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts):
        from .text import tokenize

        out = []
        for t in texts:
            toks = tokenize(t)
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
            v = [0.0] * self.dim
            for f in feats:
                v[zlib.crc32(f.encode()) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


class DenseIndex:
    def __init__(self, embedder: Embedder, texts: list[str]):
        self.embedder = embedder
        self.vecs = embedder.embed(texts)

    def scores(self, query: str) -> list[float]:
        q = self.embedder.embed([query])[0]
        return [sum(a * b for a, b in zip(q, v, strict=True)) for v in self.vecs]


class CrossEncoderReranker:
    """Optional second-stage reranker (off by default; set RAG_RERANK=1).

    NOTE: requires sentence-transformers and a model download; not exercised by the offline test suite.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query, hits):
        if not hits:
            return hits
        scores = self.model.predict([(query, h.chunk.index_text) for h in hits])
        for h, s in zip(hits, scores, strict=True):
            h.score = float(s)
        return sorted(hits, key=lambda h: h.score, reverse=True)
