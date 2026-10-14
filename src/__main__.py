"""ExtractIQ CLI.

    python -m src extract "Jane Doe, CEO of Acme, jane@acme.com"   # text -> JSON
    python -m src extract --file signature.txt --no-llm
    python -m src eval                                             # accuracy report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluate import evaluate, load_samples
from .extract import extract, extract_baseline, extract_llm, llm_available
from .schemas import get_schema, validate


def cmd_extract(args: argparse.Namespace) -> int:
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    elif args.text:
        text = args.text
    else:
        print("Provide text or --file.", file=sys.stderr)
        return 2
    schema = get_schema(args.schema)
    use_llm = False if args.no_llm else None
    data = extract(text, schema, use_llm=use_llm)
    errors = validate(data, schema)
    mode = "extractive-baseline" if args.no_llm or not llm_available() else "LLM"
    print(f"[schema={args.schema} | {mode} | valid={not errors}]")
    print(json.dumps(data, indent=2))
    if errors:
        print("Validation errors:", errors, file=sys.stderr)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    samples = load_samples()
    print(f"Evaluating on {len(samples)} labeled samples (schema=contact)\n")

    base = evaluate(extract_baseline, samples=samples)
    _print_report("baseline (regex/heuristic)", base)

    if llm_available():
        llm = evaluate(extract_llm, samples=samples)
        _print_report("LLM (Claude)", llm)
    else:
        print("\n(LLM not evaluated — set ANTHROPIC_API_KEY to compare Claude.)")
    return 0


def _print_report(label: str, r: dict) -> None:
    print(f"== {label} ==")
    print(f"  overall field accuracy: {r['overall_field_accuracy']}")
    print(f"  exact-match rows:       {r['exact_match_rows']}/{r['n_samples']} "
          f"({r['exact_match_rate']})")
    for f, acc in r["field_accuracy"].items():
        print(f"    {f:9} {acc}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="extractiq")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("extract"); e.set_defaults(func=cmd_extract)
    e.add_argument("text", nargs="?")
    e.add_argument("--file")
    e.add_argument("--schema", default="contact")
    e.add_argument("--no-llm", action="store_true")

    v = sub.add_parser("eval"); v.set_defaults(func=cmd_eval)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
