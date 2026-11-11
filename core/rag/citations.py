"""Citation helpers: number the retrieved chunks and strip citation markers the model invented."""
import re

_CITE = re.compile(r"\s?\[(\d{1,2})\]")


def format_context(hits) -> tuple[str, list[dict]]:
    """Return (text for the prompt, sources for the UI). Numbering starts at 1."""
    sources, lines = [], []
    for n, h in enumerate(hits, start=1):
        c = h.chunk
        sources.append({"n": n, "doc_id": c.doc_id, "title": c.title, "source": c.source, "text": c.text})
        lines.append(f"[{n}] {c.title} ({c.source}): {c.text}")
    return "\n\n".join(lines), sources


def valid_ids_from_context(knowledge: str) -> set[int]:
    return {int(x) for x in re.findall(r"^\[(\d+)\]", knowledge or "", re.M)}


def cited_ids(reply: str) -> list[int]:
    return [int(m.group(1)) for m in _CITE.finditer(reply)]


def sanitize_citations(reply: str, valid_ids: set[int]) -> tuple[str, list[int]]:
    """Remove [n] markers that do not refer to a provided source. Returns (reply, removed ids)."""
    removed: list[int] = []

    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        if n in valid_ids:
            return m.group(0)
        removed.append(n)
        return ""

    return _CITE.sub(repl, reply), removed
