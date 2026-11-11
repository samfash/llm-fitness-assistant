"""Load markdown documents (with simple front matter) and split them into chunks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "knowledge"


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    title: str
    source: str
    index: int
    text: str

    @property
    def index_text(self) -> str:
        """What gets indexed: the title is prepended so short chunks keep their topic."""
        return f"{self.title}. {self.text}"


def parse_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        raise ValueError(f"{path.name}: missing front matter")
    _, header, body = raw.split("---", 2)
    meta = {}
    for line in header.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    missing = {"id", "title", "source"} - set(meta)
    if missing:
        raise ValueError(f"{path.name}: front matter missing {sorted(missing)}")
    return Document(meta["id"], meta["title"], meta["source"], body.strip())


def load_documents(directory: Path | str = DEFAULT_DIR) -> list[Document]:
    docs = [parse_document(p) for p in sorted(Path(directory).glob("*.md")) if p.name.lower() != "readme.md"]
    ids = [d.id for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate document ids in knowledge base")
    return docs


def chunk_document(doc: Document, target_words: int = 90, max_words: int = 170) -> list[Chunk]:
    """Pack whole paragraphs into chunks of about `target_words`; split oversized paragraphs by sentence."""
    units: list[str] = []
    for para in (p.strip() for p in doc.text.split("\n\n") if p.strip()):
        if len(para.split()) <= max_words:
            units.append(para)
            continue
        buf: list[str] = []
        for sent in para.replace("\n", " ").split(". "):
            buf.append(sent)
            if len(" ".join(buf).split()) >= target_words:
                units.append(". ".join(buf).rstrip(".") + ".")
                buf = []
        if buf:
            units.append(". ".join(buf))

    chunks, cur = [], []
    for u in units:
        cur.append(u)
        if len(" ".join(cur).split()) >= target_words:
            chunks.append("\n\n".join(cur))
            cur = []
    if cur:
        if chunks and len(" ".join(cur).split()) < target_words // 3:  # avoid a tiny orphan tail
            chunks[-1] += "\n\n" + "\n\n".join(cur)
        else:
            chunks.append("\n\n".join(cur))
    return [Chunk(doc.id, doc.title, doc.source, i, t) for i, t in enumerate(chunks)]


def load_chunks(directory: Path | str = DEFAULT_DIR) -> list[Chunk]:
    return [c for d in load_documents(directory) for c in chunk_document(d)]
