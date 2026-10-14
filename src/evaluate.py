"""Field-level accuracy evaluation against a labeled sample set.

Compares the baseline and (optionally) the LLM extractor on the same gold data,
reporting per-field accuracy and an overall score so you can quantify the lift
the LLM provides.
"""

from __future__ import annotations

import json
from pathlib import Path

from .extract import extract_baseline, extract_llm
from .schemas import fields_of, get_schema

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples" / "contacts.json"


def _norm(v) -> str:
    return "".join(ch for ch in str(v or "").lower() if ch.isalnum())


def _match(pred, gold) -> bool:
    if gold is None:
        return pred is None or _norm(pred) == ""
    if pred is None:
        return False
    a, b = _norm(pred), _norm(gold)
    # Accept containment either way (handles "Founder & CEO" vs "Founder CEO").
    return a == b or a in b or b in a


def load_samples(path: Path = SAMPLES) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate(extractor, schema_name: str = "contact",
             samples: list[dict] | None = None) -> dict:
    schema = get_schema(schema_name)
    fields = fields_of(schema)
    samples = samples if samples is not None else load_samples()

    per_field = {f: {"correct": 0, "total": 0} for f in fields}
    exact_rows = 0
    for ex in samples:
        pred = extractor(ex["text"], schema)
        gold = ex["gold"]
        row_ok = True
        for f in fields:
            ok = _match(pred.get(f), gold.get(f))
            per_field[f]["total"] += 1
            per_field[f]["correct"] += int(ok)
            row_ok = row_ok and ok
        exact_rows += int(row_ok)

    field_acc = {f: round(d["correct"] / d["total"], 3) if d["total"] else 0.0
                 for f, d in per_field.items()}
    overall = round(sum(d["correct"] for d in per_field.values())
                    / sum(d["total"] for d in per_field.values()), 3)
    return {
        "n_samples": len(samples),
        "field_accuracy": field_acc,
        "overall_field_accuracy": overall,
        "exact_match_rows": exact_rows,
        "exact_match_rate": round(exact_rows / len(samples), 3) if samples else 0.0,
    }


def compare() -> dict:
    results = {"baseline": evaluate(extract_baseline)}
    from .extract import llm_available
    if llm_available():
        results["llm"] = evaluate(extract_llm)
    return results
