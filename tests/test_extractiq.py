"""Tests for the baseline extractor, schema validation, and the eval harness."""

from src.evaluate import evaluate, load_samples
from src.extract import extract, extract_baseline
from src.schemas import fields_of, get_schema, validate


def test_schema_fields():
    assert set(fields_of(get_schema("contact"))) == {
        "name", "email", "phone", "company", "title"
    }


def test_baseline_extracts_email_and_phone():
    schema = get_schema("contact")
    text = "Jane Doe\nSoftware Engineer\nAcme Inc\njane.doe@acme.com | (415) 555-0132"
    out = extract_baseline(text, schema)
    assert out["email"] == "jane.doe@acme.com"
    assert "555-0132" in (out["phone"] or "")
    assert out["name"] == "Jane Doe"


def test_baseline_company_from_domain():
    schema = get_schema("contact")
    out = extract_baseline("contact me at sam@brightlabs.io", schema)
    assert out["company"] == "Brightlabs"


def test_baseline_ignores_freemail_domain_for_company():
    schema = get_schema("contact")
    out = extract_baseline("Raj Patel raj@gmail.com", schema)
    assert out["company"] is None


def test_output_always_valid_against_schema():
    schema = get_schema("contact")
    out = extract("random text with no contact info at all", schema, use_llm=False)
    assert validate(out, schema) == []
    assert set(out.keys()) == set(fields_of(schema))


def test_validate_rejects_extra_keys():
    schema = get_schema("contact")
    bad = {"name": "x", "email": None, "phone": None, "company": None,
           "title": None, "surprise": 1}
    assert validate(bad, schema)  # non-empty errors


def test_eval_baseline_runs_and_scores():
    samples = load_samples()
    assert len(samples) >= 5
    r = evaluate(extract_baseline, samples=samples)
    assert 0.0 <= r["overall_field_accuracy"] <= 1.0
    # The regex baseline should nail email on every sample.
    assert r["field_accuracy"]["email"] == 1.0


def test_unknown_schema_raises():
    import pytest
    with pytest.raises(KeyError):
        get_schema("does-not-exist")
