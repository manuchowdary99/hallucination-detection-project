"""Evaluate the VeriAI pipeline on the labelled claim dataset using the real retrieval + NLI stack.

Usage (from backend/):  .venv/bin/python -m eval.run_eval [--skip-correction] [--limit N]
Requires network access to Wikipedia/Wikidata and the NLI model (downloaded on first run).
Writes eval/results.json and eval/results.md with the metrics that were actually measured.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.verification import ClaimStatus
from app.services.llm_provider import DemoProvider
from app.services.verification_pipeline import VerificationPipeline

HERE = Path(__file__).parent
LABELS = ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]


def metrics(gold: list[str], pred: list[str]) -> dict:
    matrix = {g: {p: 0 for p in LABELS + ["VERIFICATION_UNAVAILABLE"]} for g in LABELS}
    for g, p in zip(gold, pred):
        matrix[g][p] += 1
    per_class = {}
    for c in LABELS:
        tp = matrix[c][c]
        fp = sum(matrix[g][c] for g in LABELS if g != c)
        fn = sum(v for p, v in matrix[c].items() if p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3), "support": tp + fn}
    acc = sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else 0.0
    macro_f1 = statistics.mean(v["f1"] for v in per_class.values())
    # Hallucination detection view: positive = CONTRADICTED.
    tp = sum(g == "CONTRADICTED" and p == "CONTRADICTED" for g, p in zip(gold, pred))
    fp = sum(g != "CONTRADICTED" and p == "CONTRADICTED" for g, p in zip(gold, pred))
    fn = sum(g == "CONTRADICTED" and p != "CONTRADICTED" for g, p in zip(gold, pred))
    hp = tp / (tp + fp) if tp + fp else 0.0
    hr = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": round(acc, 3),
        "macro_f1": round(macro_f1, 3),
        "per_class": per_class,
        "hallucination_detection": {"precision": round(hp, 3), "recall": round(hr, 3),
                                    "f1": round(2 * hp * hr / (hp + hr), 3) if hp + hr else 0.0},
        "confusion_matrix": matrix,
    }


async def evaluate_claims(pipeline: VerificationPipeline, items: list[dict]) -> tuple[list[dict], list[float]]:
    rows, latencies = [], []
    for item in items:
        t0 = time.perf_counter()
        result = await pipeline.verify_text(item["text"])
        latencies.append((time.perf_counter() - t0) * 1000)
        if result.claims:
            c = result.claims[0]
            pred, conf, expl = c.status.value, c.confidence, c.explanation
        else:
            pred, conf, expl = ClaimStatus.UNVERIFIABLE.value, None, "No claim extracted."
        rows.append({"text": item["text"], "gold": item["label"], "pred": pred, "confidence": conf,
                     "explanation": expl, "latency_ms": round(latencies[-1]), "warnings": result.warnings})
        mark = "OK " if pred == item["label"] else "ERR"
        print(f"{mark} {item['label']:<12} -> {pred:<24} {item['text']}")
    return rows, latencies


async def evaluate_correction(pipeline: VerificationPipeline, questions: list[str]) -> dict:
    provider = DemoProvider()
    runs = []
    for q in questions:
        answer = await provider.generate(q)
        original = await pipeline.verify_text(answer)
        info, revised, corrected = await pipeline.correct(provider, answer, original)
        before = original.score.counts.contradicted
        after = corrected.score.counts.contradicted if corrected else before
        runs.append({"question": q, "status": info.status.value, "score_before": info.score_before,
                     "score_after": info.score_after, "contradicted_before": before, "contradicted_after": after})
        print(f"CORR {q!r}: {info.status.value} score {info.score_before} -> {info.score_after}, "
              f"contradicted {before} -> {after}")
    attempted = [r for r in runs if r["contradicted_before"] > 0]
    fixed = [r for r in attempted if r["contradicted_after"] < r["contradicted_before"]]
    return {
        "provider": "demo (extractive)",
        "runs": runs,
        "answers_with_contradictions": len(attempted),
        "answers_with_fewer_contradictions_after_correction": len(fixed),
        "correction_success_rate": round(len(fixed) / len(attempted), 3) if attempted else None,
    }


def to_markdown(report: dict) -> str:
    m = report["metrics"]
    lines = [
        "# VeriAI evaluation results", "",
        f"Run at {report['run_at']} on {report['n_claims']} labelled claims (live Wikipedia/Wikidata + {report['nli_model']}).", "",
        f"- Accuracy: **{m['accuracy']}**",
        f"- Macro F1: **{m['macro_f1']}**",
        f"- Hallucination (CONTRADICTED) detection — precision {m['hallucination_detection']['precision']}, "
        f"recall {m['hallucination_detection']['recall']}, F1 {m['hallucination_detection']['f1']}",
        f"- Latency per claim — mean {report['latency_ms']['mean']} ms, median {report['latency_ms']['median']} ms, "
        f"max {report['latency_ms']['max']} ms", "",
        "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|",
    ]
    lines += [f"| {c} | {v['precision']} | {v['recall']} | {v['f1']} | {v['support']} |" for c, v in m["per_class"].items()]
    cols = LABELS + ["VERIFICATION_UNAVAILABLE"]
    lines += ["", "Confusion matrix (rows = gold, columns = predicted):", "",
              "| gold \\ pred | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    lines += [f"| {g} | " + " | ".join(str(m["confusion_matrix"][g][p]) for p in cols) + " |" for g in LABELS]
    if report.get("correction"):
        c = report["correction"]
        lines += ["", "## Self-correction (demo provider)", "",
                  f"{c['answers_with_fewer_contradictions_after_correction']} of {c['answers_with_contradictions']} "
                  f"demo answers containing contradicted claims had fewer contradicted claims after correction "
                  f"(success rate {c['correction_success_rate']}).", "",
                  "| Question | Status | Score before | Score after | Contradicted before → after |", "|---|---|---|---|---|"]
        lines += [f"| {r['question']} | {r['status']} | {r['score_before']} | {r['score_after']} | "
                  f"{r['contradicted_before']} → {r['contradicted_after']} |" for r in c["runs"]]
    errors = [r for r in report["claims"] if r["gold"] != r["pred"]]
    if errors:
        lines += ["", "## Misclassified claims", "", "| Claim | Gold | Predicted |", "|---|---|---|"]
        lines += [f"| {r['text']} | {r['gold']} | {r['pred']} |" for r in errors]
    lines += ["", "Small, hand-built dataset: treat these numbers as a sanity check, not a benchmark result.", ""]
    return "\n".join(lines)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-correction", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    data = json.loads((HERE / "dataset.json").read_text())
    items = data["claims"][: args.limit] if args.limit else data["claims"]
    pipeline = VerificationPipeline()
    if pipeline.nli.status[0] not in ("loaded", "ok"):
        pipeline.nli.load()
    rows, lat = await evaluate_claims(pipeline, items)
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nli_model": pipeline.settings.nli_model if hasattr(pipeline, "settings") else "NLI model",
        "n_claims": len(rows),
        "metrics": metrics([r["gold"] for r in rows], [r["pred"] for r in rows]),
        "latency_ms": {"mean": round(statistics.mean(lat)), "median": round(statistics.median(lat)), "max": round(max(lat))},
        "claims": rows,
    }
    if not args.skip_correction:
        report["correction"] = await evaluate_correction(pipeline, data["correction_questions"])
    (HERE / "results.json").write_text(json.dumps(report, indent=2))
    (HERE / "results.md").write_text(to_markdown(report))
    print(json.dumps({k: report["metrics"][k] for k in ("accuracy", "macro_f1", "hallucination_detection")}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
