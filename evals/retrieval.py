"""Retrieval evaluation: hit@k, MRR and no-answer rejection over a labelled query set.

    python -m evals.retrieval            # BM25 only: deterministic, runs in CI
    python -m evals.retrieval --dense    # hybrid with real embeddings (needs sentence-transformers + model download)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.rag.retriever import HybridRetriever, build_retriever

ROOT = Path(__file__).resolve().parent
KS = (1, 3, 4)


def _doc_ranking(retriever: HybridRetriever, query: str, k: int) -> list[str]:
    seen: list[str] = []
    for h in retriever.search(query, k=k):
        if h.chunk.doc_id not in seen:
            seen.append(h.chunk.doc_id)
    return seen


def _score(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    out = {"n": n, "mrr": round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3)}
    for k in KS:
        out[f"hit@{k}"] = round(sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n, 3)
    return out


def evaluate(retriever: HybridRetriever, qa: list[dict], k: int = 4) -> dict:
    rejected, rows = [], []
    for item in qa:
        docs = _doc_ranking(retriever, item["query"], k)
        if not item["relevant"]:
            rejected.append({"query": item["query"], "returned": docs, "ok": not docs})
            continue
        rank = next((i for i, d in enumerate(docs, start=1) if d in item["relevant"]), None)
        rows.append({"query": item["query"], "rank": rank, "hard": bool(item.get("hard")), "returned": docs,
                     "relevant": item["relevant"]})
    return {
        "mode": retriever.mode,
        "all": _score(rows),
        "easy": _score([r for r in rows if not r["hard"]]),
        "hard": _score([r for r in rows if r["hard"]]),
        "no_answer_rejection": round(sum(r["ok"] for r in rejected) / len(rejected), 3) if rejected else None,
        "failures": [r for r in rows if not r["rank"]] + [r for r in rejected if not r["ok"]],
    }


def check_gate(result: dict, thresholds: dict) -> list[str]:
    flat = {"mrr": result["all"].get("mrr"), "hit@4": result["all"].get("hit@4"), "hit@1": result["all"].get("hit@1"),
            "no_answer_rejection": result["no_answer_rejection"], "hard_hit@4": result["hard"].get("hit@4")}
    return [f"{k}: {flat[k]} is below the required {v}" for k, v in thresholds.items()
            if flat.get(k) is not None and flat[k] < v]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--qa", default=str(ROOT / "data" / "retrieval_qa.json"))
    p.add_argument("--dense", action="store_true", help="use real embeddings (hybrid retrieval)")
    p.add_argument("--thresholds", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    retriever = build_retriever(dense=a.dense)
    if a.dense and retriever.mode != "hybrid":
        print("--dense requested but embeddings are unavailable", file=sys.stderr)
        return 2
    result = evaluate(retriever, json.loads(Path(a.qa).read_text()))
    print(f"mode={result['mode']}  n={result['all']['n']}  MRR={result['all']['mrr']}  "
          f"hit@1={result['all']['hit@1']}  hit@3={result['all']['hit@3']}  hit@4={result['all']['hit@4']}  "
          f"no-answer rejection={result['no_answer_rejection']}")
    print(f"easy: {result['easy']}\nhard: {result['hard']}")
    for f in result["failures"]:
        print("  miss:", f["query"], "->", f["returned"])
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2))
    tpath = Path(a.thresholds or ROOT / ("retrieval_thresholds_hybrid.json" if a.dense else "retrieval_thresholds.json"))
    problems = check_gate(result, json.loads(tpath.read_text())) if tpath.exists() else []
    for prob in problems:
        print(f"GATE FAILED: {prob}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
