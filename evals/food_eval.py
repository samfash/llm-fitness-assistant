"""Food-search evaluation: does a natural query surface an acceptable USDA entry near the top?

A result is acceptable if its description matches any regex in the item's `accept` list.

    python -m evals.food_eval --split dev     # tune against this
    python -m evals.food_eval --split test    # report this; do not tune against it
    python -m evals.food_eval                 # both + gate (used in CI)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from core.foods import FoodDB, get_food_db

ROOT = Path(__file__).resolve().parent
KS = (1, 3, 5)


def _rank(db: FoodDB, item: dict, k: int) -> tuple[int | None, list[str]]:
    results = db.search(item["query"], limit=k)
    rxs = [re.compile(p, re.I) for p in item["accept"]]
    for i, f in enumerate(results, start=1):
        if any(r.search(f.description) for r in rxs):
            return i, [x.description for x in results]
    return None, [x.description for x in results]


def evaluate(db: FoodDB, qa: list[dict], split: str = "all", k: int = 5) -> dict:
    items = [q for q in qa if split == "all" or q.get("split") == split]
    rows = []
    for it in items:
        rank, returned = _rank(db, it, k)
        rows.append({"query": it["query"], "rank": rank, "returned": returned})
    n = len(rows)
    res = {"split": split, "n": n, "mrr": round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3) if n else 0.0}
    for kk in KS:
        res[f"hit@{kk}"] = round(sum(1 for r in rows if r["rank"] and r["rank"] <= kk) / n, 3) if n else 0.0
    res["failures"] = [r for r in rows if not r["rank"]]
    res["rows"] = rows
    return res


def check_gate(result: dict, thresholds: dict) -> list[str]:
    return [f"{k}: {result[k]} is below the required {v}" for k, v in thresholds.items() if result.get(k, 1) < v]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--qa", default=str(ROOT / "data" / "food_qa.json"))
    p.add_argument("--split", default="all", choices=["dev", "test", "all"])
    p.add_argument("--thresholds", default=str(ROOT / "food_thresholds.json"))
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)
    qa = json.loads(Path(a.qa).read_text())
    db = get_food_db()
    if not len(db):
        print("food data not found", file=sys.stderr)
        return 2
    res = evaluate(db, qa, a.split)
    print(f"[{res['split']}] n={res['n']}  MRR={res['mrr']}  hit@1={res['hit@1']}  hit@3={res['hit@3']}  hit@5={res['hit@5']}")
    for r in res["rows"] if a.verbose else res["failures"]:
        mark = "  ok " if r["rank"] else " miss"
        print(f"{mark} #{r['rank']}  {r['query']!r} -> {r['returned'][:3]}")
    t = Path(a.thresholds)
    problems = check_gate(res, json.loads(t.read_text())) if (t.exists() and a.split == "all") else []
    for prob in problems:
        print(f"GATE FAILED: {prob}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
