import json
from pathlib import Path

import pytest

from core.agent import AssistantTurn, run_agent
from core.rag import build_retriever, format_context, sanitize_citations
from core.rag.bm25 import BM25
from core.rag.corpus import Chunk, chunk_document, load_chunks, load_documents, parse_document
from core.rag.dense import HashingEmbedder
from core.rag.retriever import HybridRetriever
from core.rag.text import stem, tokenize
from evals.retrieval import check_gate, evaluate
from evals.runner import load_cases, run_case

ROOT = Path(__file__).resolve().parents[1]
QA = json.loads((ROOT / "evals" / "data" / "retrieval_qa.json").read_text())


@pytest.fixture(scope="module")
def bm25_retriever():
    return build_retriever(dense=False)


# ---- text ---------------------------------------------------------------------------------------
def test_stemming_collapses_variants():
    assert stem("muscles") == stem("muscle")
    assert stem("lifting") == stem("lifted") == stem("lift")
    assert stem("running") == stem("run")
    assert stem("bodies") == stem("body")
    assert tokenize("How much protein do I need?") == ["protein"]


# ---- corpus -------------------------------------------------------------------------------------
def test_front_matter_is_required(tmp_path):
    (tmp_path / "bad.md").write_text("no front matter here")
    with pytest.raises(ValueError, match="front matter"):
        parse_document(tmp_path / "bad.md")
    (tmp_path / "bad2.md").write_text("---\nid: x\ntitle: T\n---\nbody")
    with pytest.raises(ValueError, match="source"):
        parse_document(tmp_path / "bad2.md")


def test_duplicate_ids_rejected(tmp_path):
    for n in ("a", "b"):
        (tmp_path / f"{n}.md").write_text("---\nid: same\ntitle: T\nsource: S\n---\nbody text")
    with pytest.raises(ValueError, match="duplicate"):
        load_documents(tmp_path)


def test_real_corpus_is_well_formed():
    docs = load_documents()
    assert len(docs) >= 12 and all(d.source and d.title for d in docs)
    chunks = load_chunks()
    assert all(10 <= len(c.text.split()) <= 260 for c in chunks)
    assert {c.doc_id for c in chunks} == {d.id for d in docs}


def test_chunker_keeps_paragraphs_and_indexes_title():
    paras = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(4))
    doc = parse_document_from(paras)
    chunks = chunk_document(doc)
    assert len(chunks) >= 2 and all(c.index_text.startswith("Title. ") for c in chunks)


def parse_document_from(text):
    from core.rag.corpus import Document
    return Document("d", "Title", "Src", text)


# ---- BM25 ---------------------------------------------------------------------------------------
def test_bm25_ranks_matching_doc_first_and_handles_no_match():
    docs = [tokenize("protein builds muscle"), tokenize("running improves heart health"), tokenize("sleep helps recovery")]
    bm = BM25(docs)
    s = bm.scores(tokenize("how much protein"))
    assert s.index(max(s)) == 0 and s[1] == 0
    assert all(x == 0 for x in bm.scores(tokenize("capital of france")))


# ---- hybrid retrieval & fusion ---------------------------------------------------------------
class ConceptEmbedder:
    """Tiny fake 'semantic' model: maps synonyms to the same dimension (coffee == caffeine)."""
    CONCEPTS = [{"coffee", "caffeine", "espresso"}, {"sleep", "rest", "nap"}, {"protein", "meat"}]

    def embed(self, texts):
        out = []
        for t in texts:
            words = set(t.lower().replace(".", " ").split())
            v = [1.0 if words & c else 0.0 for c in self.CONCEPTS]
            n = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / n for x in v])
        return out


def _mini_chunks():
    return [Chunk("caf", "Caffeine", "S", 0, "Caffeine improves endurance when taken before exercise."),
            Chunk("slp", "Sleep", "S", 0, "Adults need seven hours of sleep for recovery."),
            Chunk("pro", "Protein", "S", 0, "Protein supports muscle repair after training.")]


def test_dense_rescues_vocabulary_mismatch_that_bm25_misses():
    lexical = HybridRetriever(_mini_chunks())
    hybrid = HybridRetriever(_mini_chunks(), embedder=ConceptEmbedder())
    assert lexical.search("is espresso any good?") == []
    hits = hybrid.search("is espresso any good?")
    assert hits and hits[0].chunk.doc_id == "caf" and hits[0].bm25 == 0 and hits[0].dense == 1.0
    assert hybrid.mode == "hybrid" and lexical.mode == "bm25"


def test_rrf_rewards_agreement_between_retrievers():
    r = HybridRetriever(_mini_chunks(), embedder=ConceptEmbedder())
    hit = r.search("caffeine before exercise")[0]
    assert set(hit.ranks) == {"bm25", "dense"} and hit.score > 1 / 61


def test_offtopic_query_returns_nothing(bm25_retriever):
    for q in ("what is the capital of France", "how do I bake sourdough bread", ""):
        assert bm25_retriever.search(q) == []


def test_max_per_doc_gives_diversity():
    chunks = [Chunk("a", "Squat", "S", i, f"squat depth technique tip number {i} for squat form") for i in range(4)]
    chunks.append(Chunk("b", "Squat basics", "S", 0, "squat technique basics squat form"))
    docs = [h.chunk.doc_id for h in HybridRetriever(chunks, bm25_min=0.1).search("squat technique", k=4)]
    assert docs.count("a") <= 2 and "b" in docs


def test_reranker_plumbing():
    class Reverse:
        def rerank(self, query, hits):
            for i, h in enumerate(hits):
                h.score = float(i)
            return hits  # keep order but prove it's called and scores are replaced

    r = HybridRetriever(_mini_chunks(), embedder=ConceptEmbedder(), reranker=Reverse())
    hits = r.search("caffeine and sleep and protein", k=3)
    assert [h.score for h in hits] == sorted([h.score for h in hits])  # reranker's scores are used


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder()
    a, b = e.embed(["protein and muscle"]), e.embed(["protein and muscle"])
    assert a == b and abs(sum(x * x for x in a[0]) - 1) < 1e-9


# ---- citations --------------------------------------------------------------------------------
def test_format_context_numbers_sources_from_one(bm25_retriever):
    text, sources = format_context(bm25_retriever.search("how much protein to build muscle"))
    assert text.startswith("[1] ") and [s["n"] for s in sources] == list(range(1, len(sources) + 1))
    assert sources[0]["doc_id"] == "protein-intake" and sources[0]["source"]


def test_sanitize_citations():
    assert sanitize_citations("Aim for 150 min [1] and more [3].", {1, 2}) == ("Aim for 150 min [1] and more.", [3])
    assert sanitize_citations("No sources [1].", set()) == ("No sources.", [1])
    assert sanitize_citations("Fine [2] text", {2}) == ("Fine [2] text", [])


class FixedReply:
    def __init__(self, text):
        self.text = text

    def chat(self, messages, tools):
        return AssistantTurn(content=self.text)


def test_agent_strips_invented_citations_and_keeps_valid_ones():
    res = run_agent("u", "how much cardio?", FixedReply("150 min weekly [1], see also [7]."),
                    knowledge="[1] Title (Src): text")
    assert res.reply == "150 min weekly [1], see also." and res.guardrails[-1]["rule"] == "invalid_citation"
    res = run_agent("u", "hi", FixedReply("Hello [1]"))  # no knowledge was provided at all
    assert res.reply == "Hello"
    res = run_agent("u", "hi", FixedReply("Hello [1]"), guardrails=False)
    assert res.reply == "Hello [1]"


def test_knowledge_and_citation_rule_reach_the_model():
    seen = {}

    class Spy:
        def chat(self, messages, tools):
            seen["system"] = messages[0]["content"]
            return AssistantTurn(content="ok")

    run_agent("u", "q", Spy(), knowledge="[1] Title (Src): body")
    assert "[1] Title (Src): body" in seen["system"] and "cite it inline" in seen["system"]


# ---- retrieval evaluation (regression gate, BM25 only so it is deterministic) -------------------
def test_retrieval_quality_gate_on_real_corpus(bm25_retriever):
    result = evaluate(bm25_retriever, QA)
    thresholds = json.loads((ROOT / "evals" / "retrieval_thresholds.json").read_text())
    assert check_gate(result, thresholds) == [], result["failures"]


def test_eval_metrics_are_computed_correctly():
    chunks = _mini_chunks()
    r = HybridRetriever(chunks)
    qa = [{"query": "caffeine before exercise", "relevant": ["caf"]},
          {"query": "protein muscle repair", "relevant": ["slp", "pro"]},
          {"query": "quantum chromodynamics", "relevant": []},
          {"query": "sleep seven hours", "relevant": ["pro"]}]  # deliberately wrong label -> a miss
    res = evaluate(r, qa)
    assert res["all"]["n"] == 3 and res["all"]["hit@1"] == round(2 / 3, 3)
    assert res["no_answer_rejection"] == 1.0 and len(res["failures"]) == 1


def test_gate_flags_regressions():
    res = {"all": {"mrr": 0.5, "hit@4": 0.9, "hit@1": 0.4}, "hard": {}, "no_answer_rejection": 0.5}
    problems = check_gate(res, {"mrr": 0.8, "hit@4": 0.85, "no_answer_rejection": 0.9})
    assert len(problems) == 2


# ---- golden RAG cases ----------------------------------------------------------------------------
def test_rag_cases_reference_real_documents():
    ids = {d.id for d in load_documents()}
    cases = [c for c in load_cases(ROOT / "evals" / "cases") if c.get("use_rag")]
    assert len(cases) >= 5
    for c in cases:
        assert set(c["expect"].get("cites_doc", [])) <= ids


def _case(cid):
    return next(c for c in load_cases(ROOT / "evals" / "cases") if c["id"] == cid)


def _n_for(retriever, prompt, doc_id):
    _, sources = format_context(retriever.search(prompt))
    return next(s["n"] for s in sources if s["doc_id"] == doc_id)


def test_rag_case_passes_with_correct_citation_and_fails_without(bm25_retriever):
    case = _case("rag_aerobic")
    n = _n_for(bm25_retriever, case["prompt"], "who-aerobic")
    good = FixedReply(f"Adults should aim for 150 to 300 minutes of moderate activity weekly [{n}].")
    assert run_case(case, good).passed
    uncited = FixedReply("Adults should aim for 150 to 300 minutes of moderate activity weekly.")
    r = run_case(case, uncited)
    assert not r.passed and any(c["name"] == "cites:who-aerobic" and not c["ok"] for c in r.checks)
    assert r.retrieved[0] == "who-aerobic"


def test_rag_case_catches_invented_numbers_and_no_source_citations(bm25_retriever):
    case = _case("rag_aerobic")
    n = _n_for(bm25_retriever, case["prompt"], "who-aerobic")
    made_up = run_case(case, FixedReply(f"You need exactly 777 minutes of moderate activity [{n}]."), guardrails=False)
    assert any(c["name"] == "grounded_numbers" and not c["ok"] for c in made_up.checks)
    assert run_case(_case("rag_no_source"), FixedReply("Paris.")).passed
    assert not run_case(_case("rag_no_source"), FixedReply("Paris [1]."), guardrails=False).passed
