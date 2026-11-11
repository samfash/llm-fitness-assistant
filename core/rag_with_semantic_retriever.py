"""Backwards-compatible entry points used by the app and API (implementation lives in core/rag/)."""
from .rag import format_context, get_retriever


def retrieve_with_sources(query: str, top_k: int = 4) -> tuple[str, list[dict]]:
    """Return (numbered context for the prompt, source list for the UI)."""
    return format_context(get_retriever().search(query, k=top_k))


def retrieve_context(query: str, top_k: int = 4) -> str:
    return retrieve_with_sources(query, top_k)[0]
