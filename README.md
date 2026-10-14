# ExtractIQ — LLM Structured Extraction  🧾→🔤

Turn messy, unstructured text into **schema-validated JSON** — and *measure* how
well it works. ExtractIQ extracts structured records with Claude, validates every
result against a JSON Schema, and benchmarks the LLM against a regex baseline on
a labeled set. Pulling clean structured data out of free text is one of the most
useful real-world LLM skills.

## What it does

- **Schema-driven extraction** — define a JSON Schema (bundled: `contact`), and
  both extractors target it. Every output is **validated** before it's returned.
- **Two extractors:**
  - **Baseline** — regex + heuristics (email/phone patterns, name/title/company
    cues). Deterministic, offline, and a real yardstick.
  - **LLM (Claude)** — handles messy, inline, and implicit phrasing the regex
    baseline can't.
- **Evaluation** — field-level accuracy + exact-match rate on a labeled sample
  set, so the LLM's lift is quantified, not assumed.

## Results (baseline, on the 6 bundled labeled samples)

| Field | Accuracy |
|-------|:--------:|
| email | 1.00 |
| phone | 1.00 |
| company | 1.00 |
| title | 0.67 |
| name | 0.50 |
| **Overall field accuracy** | **0.83** |
| Exact-match rows | 3 / 6 (0.50) |

The baseline nails regex-shaped fields (email, phone) and domain-derived company
names, but **misses names and titles in inline prose** like *"Reach Michael
Chen, Director of Product, at …"* — it has no regex for "a name mentioned
mid-sentence." That gap is exactly what the LLM path closes:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m src eval      # adds an "LLM (Claude)" row to compare the lift
```

This baseline-vs-LLM framing is the point: it shows you can *measure* an LLM
feature instead of trusting it.

## Quickstart

```bash
cd 03-extractiq
pip install -r requirements.txt

# Extract from text (offline baseline):
python -m src extract "Jane Doe, Senior Engineer at Acme, jane@acme.com (415) 555-0132" --no-llm

# Extract with Claude (needs a key):
export ANTHROPIC_API_KEY=sk-ant-...
python -m src extract --file signature.txt

# Benchmark baseline (and LLM, if a key is set):
python -m src eval
```

Output is validated JSON:

```json
{ "name": "Jane Doe", "email": "jane@acme.com", "phone": "(415) 555-0132",
  "company": "Acme", "title": "Senior Engineer" }
```

## Project layout

```
03-extractiq/
├── README.md
├── requirements.txt
├── samples/contacts.json   # labeled eval set (text + gold JSON)
└── src/
    ├── schemas.py          # JSON Schemas + validation
    ├── extract.py          # baseline (regex) + LLM (Claude) extractors
    ├── evaluate.py         # field accuracy + exact-match scoring
    └── __main__.py         # extract / eval CLI
tests/
└── test_extractiq.py       # 8 tests, no API key required
```

## Tests

```bash
python -m pytest -q      # 8 tests
```

## Design notes

- **Validation is non-negotiable** — the LLM is instructed to return schema-shaped
  JSON, but the output is still parsed and checked against the schema, and
  coerced to always contain the required keys. Never trust raw model JSON.
- **Model:** `claude-opus-5-5` via the official `anthropic` SDK; credentials
  resolve from the environment (no key hardcoded).

## Possible extensions

- More schemas: receipts (line items + totals), resumes, event invites.
- Use the API's **structured outputs** (`output_config.format`) for guaranteed
  schema-valid responses.
- Confidence scores + a human-review queue for low-confidence extractions.
- Batch extraction over a folder of files with a CSV export.
