"""Okapi BM25 in pure Python (no dependencies; corpus sizes here are tiny)."""
import math
from collections import Counter


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs_tokens]
        self.len = [len(d) for d in docs_tokens]
        self.n = len(docs_tokens)
        self.avg = (sum(self.len) / self.n) if self.n else 0.0
        df = Counter(t for d in self.tf for t in d)
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query_tokens: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.tf, self.len, strict=True):
            s = 0.0
            for t in set(query_tokens):
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            out.append(s)
        return out
