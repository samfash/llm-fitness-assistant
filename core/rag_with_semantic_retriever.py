"""Backwards-compatible entry points used by the app and API (implementation lives in core/rag/)."""
from .rag import format_context, get_retriever
from .tracing import span


def retrieve_with_sources(query: str, top_k: int = 4) -> tuple[str, list[dict]]:
    """Return (numbered context for the prompt, source list for the UI)."""
    with span("rag.retrieve") as sp:
        retriever = get_retriever()
        hits = retriever.search(query, k=top_k)
        sp.set(mode=retriever.mode, hits=len(hits), top_doc=hits[0].chunk.doc_id if hits else None)
        return format_context(hits)


def retrieve_context(query: str, top_k: int = 4) -> str:
    return retrieve_with_sources(query, top_k)[0]
