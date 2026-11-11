"""CLI.  Examples:
    python -m evals.run --mode live --record --report evals/results/report.md   # needs HF_TOKEN
    python -m evals.run --mode replay                                            # no network, uses recordings
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

from core.llm_engine import REMOTE_MODEL
from core.rag import build_retriever

from .runner import RecordingLLM, ReplayLLM, check_gate, load_cases, render_report, run_case, summarize

ROOT = Path(__file__).resolve().parent


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["live", "replay"], default="replay")
    p.add_argument("--cases", default=str(ROOT / "cases"))
    p.add_argument("--recordings", default=str(ROOT / "recordings"))
    p.add_argument("--record", action="store_true", help="live mode: save model turns for later replay")
    p.add_argument("--model", default=None)
    p.add_argument("--no-guardrails", action="store_true", help="A/B: bypass the deterministic safety layer")
    p.add_argument("--dense", action="store_true", help="use hybrid retrieval for RAG cases (breaks replay determinism)")
    p.add_argument("--only", default=None, help="substring filter on case id or category")
    p.add_argument("--out", default=str(ROOT / "results" / "latest.json"))
    p.add_argument("--report", default=None)
    p.add_argument("--thresholds", default=str(ROOT / "thresholds.json"))
    p.add_argument("--baseline", default=None, help="previous results JSON; fail on regression")
    p.add_argument("--max-drop", type=float, default=0.05)
    a = p.parse_args(argv)

    # read the baseline first: --baseline and --out may point at the same file
    baseline = json.loads(Path(a.baseline).read_text()) if a.baseline and Path(a.baseline).exists() else None
    cases = load_cases(Path(a.cases))
    if a.only:
        cases = [c for c in cases if a.only in c["id"] or a.only in c["category"]]
    rec_dir = Path(a.recordings)
    model = a.model or REMOTE_MODEL
    client = None

    if a.mode == "live":
        if not (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")):
            print("live mode needs an HF_TOKEN environment variable", file=sys.stderr)
            return 2
        from core.agent import HFToolClient
        client = HFToolClient(model=model)

    retriever = build_retriever(dense=True) if a.dense else None
    results, skipped = [], []
    for case in cases:
        rec_file = rec_dir / f"{case['id']}.json"
        if a.mode == "replay":
            if not rec_file.exists():
                skipped.append(case["id"])
                continue
            data = json.loads(rec_file.read_text())
            model = data.get("model", model)
            llm = ReplayLLM(data["turns"])
        else:
            llm = RecordingLLM(client) if a.record else client
        r = run_case(case, llm, guardrails=not a.no_guardrails, retriever=retriever)
        if a.mode == "live" and a.record and not r.error:
            rec_dir.mkdir(parents=True, exist_ok=True)
            rec_file.write_text(json.dumps({"model": model, "recorded_at": date.today().isoformat(),
                                            "turns": llm.turns}, indent=2))
        print(f"{'PASS' if r.passed else 'FAIL'}  {r.category}/{r.id}  ({r.latency_s:.1f}s)")
        results.append(r)

    if not results:
        print(f"No cases ran ({len(skipped)} without recordings). Run `--mode live --record` first.")
        return 0

    summary = summarize(results)
    meta = {"mode": a.mode, "model": model, "date": date.today().isoformat(), "skipped": skipped,
            "guardrails": not a.no_guardrails}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, **summary, "cases": [asdict(r) for r in results]}, indent=2, default=str))
    report = render_report(summary, results, meta)
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(report)
    print("\n" + report.split("## Failures")[0])
    if skipped:
        print(f"(skipped {len(skipped)} case(s) without recordings)")

    thresholds = json.loads(Path(a.thresholds).read_text()) if Path(a.thresholds).exists() else {}
    problems = check_gate(summary, thresholds, baseline, a.max_drop)
    for prob in problems:
        print(f"GATE FAILED: {prob}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
