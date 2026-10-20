from functools import lru_cache

DOCS = [
    "Protein-rich meals help repair muscle tissue after workouts.",
    "Hydration improves recovery and energy levels.",
    "Cardio workouts like running and cycling improve heart health.",
    "Resistance training builds muscle strength and tone.",
    "Eating complex carbs before a workout provides sustained energy.",
]


@lru_cache(maxsize=1)
def _store():
    """Built lazily on first query, not at import time."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection(name="fitness_knowledge")
    col.upsert(
        ids=[f"doc_{i}" for i in range(len(DOCS))],
        documents=DOCS,
        embeddings=model.encode(DOCS).tolist(),
    )
    return model, col


def retrieve_context(query, top_k=2):
    model, col = _store()
    res = col.query(query_embeddings=[model.encode(query).tolist()], n_results=top_k)
    docs = res.get("documents") or [[]]
    return " ".join(docs[0])
