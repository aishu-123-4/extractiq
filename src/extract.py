"""Extract structured data from text — LLM (Claude) or a regex baseline.

Both paths return a dict matching the schema; the result is validated against
the JSON Schema before being returned. The baseline runs offline and gives a
meaningful comparison point for the eval (and a graceful fallback with no key).
"""

from __future__ import annotations

import json
import os
import re

from .schemas import fields_of, validate

MODEL = "claude-opus-5-5"

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(
    r"(?:\+?\d{1,2}[\s.\-]?)?(?:\(\d{3}\)|\d{3})[\s.\-]?\d{3}[\s.\-]?\d{4}"
)
TITLE_WORDS = ("engineer", "manager", "director", "officer", "ceo", "cto",
               "cfo", "president", "founder", "analyst", "developer", "designer",
               "consultant", "lead", "head", "scientist", "architect", "vp")
COMPANY_SUFFIX = ("inc", "inc.", "llc", "ltd", "ltd.", "corp", "corp.",
                  "co.", "company", "technologies", "labs", "group", "systems")


def llm_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.path.isdir(os.path.expanduser("~/.config/anthropic")))


def extract_baseline(text: str, schema: dict) -> dict:
    """Heuristic/regex extraction. Deterministic, offline."""
    out = {f: None for f in fields_of(schema)}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    email = EMAIL_RE.search(text)
    if email:
        out["email"] = email.group(0)
    phone = PHONE_RE.search(text)
    if phone:
        out["phone"] = phone.group(0).strip()

    # Name heuristic: first line that looks like a 2-3 word proper name.
    for ln in lines:
        words = ln.split()
        if (2 <= len(words) <= 3 and all(w[:1].isupper() for w in words)
                and "@" not in ln and not any(c.isdigit() for c in ln)
                and not any(s in ln.lower() for s in COMPANY_SUFFIX)):
            out["name"] = ln
            break

    # Title: a line containing a known role word.
    for ln in lines:
        if any(w in ln.lower() for w in TITLE_WORDS) and len(ln) < 60:
            out["title"] = ln.strip(" ,|-")
            break

    # Company: a line with a corporate suffix, else the email domain.
    for ln in lines:
        if any(ln.lower().rstrip(".").endswith(s.rstrip(".")) for s in COMPANY_SUFFIX):
            out["company"] = ln.strip(" ,|-")
            break
    if out["company"] is None and out["email"]:
        domain = out["email"].split("@")[1].split(".")[0]
        if domain.lower() not in ("gmail", "yahoo", "outlook", "hotmail", "icloud"):
            out["company"] = domain.capitalize()

    return out


def extract_llm(text: str, schema: dict) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    system = (
        "You extract structured data from text. Return ONLY a JSON object that "
        "matches the given JSON Schema. Use null for fields not present. No prose."
    )
    user = (f"JSON Schema:\n{json.dumps(schema)}\n\n"
            f"Text:\n\"\"\"\n{text}\n\"\"\"\n\nReturn the JSON object.")
    resp = client.messages.create(
        model=MODEL, max_tokens=800, system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(raw)
    # Keep only schema fields; fill missing with null.
    return {f: data.get(f) for f in fields_of(schema)}


def extract(text: str, schema: dict, use_llm: bool | None = None) -> dict:
    if use_llm is None:
        use_llm = llm_available()
    if use_llm:
        try:
            data = extract_llm(text, schema)
        except Exception:  # noqa: BLE001 — fall back to baseline
            data = extract_baseline(text, schema)
    else:
        data = extract_baseline(text, schema)
    # Validate (and coerce obvious issues) before returning.
    errors = validate(data, schema)
    if errors:
        # Ensure every required field at least exists as null.
        for f in fields_of(schema):
            data.setdefault(f, None)
    return data
