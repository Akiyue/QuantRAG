"""Tests for the layer that decides which numbers go into the metrics."""

from __future__ import annotations

import json

import pytest

from quantrag.analysis import (
    ItemKey, Record, known_subsets, load, paired, precisions, truncated_entropy,
)
from quantrag.normalize import Label


def mk(precision, kind="generate", label=None, r=None, endorsed="fake",
       fact_id="f1", lang="en", condition="C2", mode="strict", model="m"):
    return Record(
        key=ItemKey(fact_id, lang, condition, mode), model_id=model,
        precision=precision, kind=kind, domain="d", endorsed=endorsed,
        label=label, r=r,
    )


# -- known subsets --------------------------------------------------------

def test_known_all_requires_every_precision():
    """known_all is the primary set precisely because it is not selected on any
    single arm, so regression to the mean cannot manufacture an effect."""
    recs = [
        mk("F16", label=Label.TRUE, fact_id="a"),
        mk("Q4_K_M", label=Label.TRUE, fact_id="a"),
        mk("F16", label=Label.TRUE, fact_id="b"),
        mk("Q4_K_M", label=Label.OTHER, fact_id="b"),
        mk("F16", label=Label.OTHER, fact_id="c"),
        mk("Q4_K_M", label=Label.OTHER, fact_id="c"),
    ]
    s = known_subsets(recs)
    assert ("a", "en") in s["known_all"]
    assert ("b", "en") in s["known_fp16"]
    assert ("c", "en") in s["unknown"]
    assert ("b", "en") not in s["known_all"]


def test_known_subsets_are_per_language():
    recs = [
        mk("F16", label=Label.TRUE, lang="en"),
        mk("Q4_K_M", label=Label.TRUE, lang="en"),
        mk("F16", label=Label.OTHER, lang="vi"),
        mk("Q4_K_M", label=Label.OTHER, lang="vi"),
    ]
    s = known_subsets(recs)
    assert ("f1", "en") in s["known_all"]
    assert ("f1", "vi") in s["unknown"]


# -- reliance orientation -------------------------------------------------

def test_reliance_is_undefined_when_context_endorses_the_truth(tmp_path):
    """Under C1 the context answer and the parametric answer are the same
    string, so R would collapse to zero by construction."""
    rec = {
        "fact_id": "f1", "lang": "en", "condition": "C1", "mode": "strict",
        "kind": "score", "model_id": "m", "precision": "F16", "domain": "d",
        "endorsed": "true",
        "score_fake": {"mean_logprob": -1.0, "n_tokens": 1, "sum_logprob": -1.0},
        "score_true": {"mean_logprob": -0.5, "n_tokens": 1, "sum_logprob": -0.5},
    }
    p = tmp_path / "main__m__F16.jsonl"
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    loaded = load(tmp_path, "main")
    assert loaded[0].r is None


def test_reliance_sign_under_conflict(tmp_path):
    rec = {
        "fact_id": "f1", "lang": "en", "condition": "C2", "mode": "strict",
        "kind": "score", "model_id": "m", "precision": "F16", "domain": "d",
        "endorsed": "fake",
        "score_fake": {"mean_logprob": -0.5, "n_tokens": 1, "sum_logprob": -0.5},
        "score_true": {"mean_logprob": -2.0, "n_tokens": 1, "sum_logprob": -2.0},
    }
    p = tmp_path / "main__m__F16.jsonl"
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    loaded = load(tmp_path, "main")
    assert loaded[0].r == pytest.approx(1.5)     # leaning to context
    assert loaded[0].p_ctx > 0.5


def test_boundary_error_records_are_skipped(tmp_path):
    p = tmp_path / "main__m__F16.jsonl"
    p.write_text(json.dumps({"key": "k", "error": "boundary",
                             "model_id": "m", "precision": "F16"}) + "\n",
                 encoding="utf-8")
    assert load(tmp_path, "main") == []


# -- pairing --------------------------------------------------------------

def test_pairing_keeps_only_items_present_at_both_precisions():
    recs = [
        mk("F16", label=Label.TRUE, fact_id="a"),
        mk("Q4_K_M", label=Label.FAKE, fact_id="a"),
        mk("F16", label=Label.TRUE, fact_id="b"),   # missing at Q4
    ]
    items = paired(recs, "m", "Q4_K_M", condition="C2")
    assert [i.key.fact_id for i in items] == ["a"]


def test_scope_restriction_is_per_fact_and_language():
    recs = [
        mk("F16", label=Label.TRUE, fact_id="a", lang="en"),
        mk("Q4_K_M", label=Label.FAKE, fact_id="a", lang="en"),
        mk("F16", label=Label.TRUE, fact_id="a", lang="vi"),
        mk("Q4_K_M", label=Label.FAKE, fact_id="a", lang="vi"),
    ]
    items = paired(recs, "m", "Q4_K_M", condition="C2", scope={("a", "en")})
    assert [i.key.lang for i in items] == ["en"]


def test_precision_ordering_puts_baseline_first():
    recs = [mk("Q4_K_M"), mk("F16"), mk("Q8_0")]
    assert precisions(recs) == ["F16", "Q8_0", "Q4_K_M"]


# -- calibration ----------------------------------------------------------

def test_truncated_entropy_is_zero_for_a_certain_distribution():
    import math
    assert truncated_entropy([math.log(1.0)]) == pytest.approx(0.0)


def test_truncated_entropy_is_maximal_when_flat():
    import math
    flat = [math.log(0.25)] * 4
    assert truncated_entropy(flat) == pytest.approx(math.log(4))


# -- analysis sections end to end -----------------------------------------

def _synthetic_records(n: int = 60):
    """Two precisions on the same items, with a planted one-sided shift.

    Under Q4 a quarter of the items move from keeping the true answer to
    following the counterfactual document, and none move back. That is the
    directional signal the asymmetry metric exists to detect.
    """
    import random
    rng = random.Random(7)
    recs = []
    for i in range(n):
        fid = f"f{i:03d}"
        flips = i % 4 == 0
        for lang in ("en", "vi"):
            base_r = rng.uniform(-2.0, -0.2)
            recs += [
                mk("F16", label=Label.TRUE, fact_id=fid, lang=lang),
                mk("Q4_K_M", label=Label.FAKE if flips else Label.TRUE,
                   fact_id=fid, lang=lang),
                mk("F16", kind="score", r=base_r, fact_id=fid, lang=lang),
                mk("Q4_K_M", kind="score", r=base_r + (1.5 if flips else 0.05),
                   fact_id=fid, lang=lang),
            ]
    return recs


def test_analysis_sections_produce_populated_tables(tmp_path):
    sys_path_setup = __import__("sys")
    sys_path_setup.path.insert(0, str(__import__("pathlib").Path(__file__)
                                      .resolve().parents[1] / "scripts"))
    import analyze  # noqa: PLC0415

    recs = _synthetic_records()
    analyze.section_margin(recs, None, tmp_path)
    pvals = analyze.section_behaviour(recs, None, tmp_path)
    analyze.section_flips(recs, None, tmp_path)
    pvals |= analyze.section_reliance(recs, None, tmp_path)
    analyze.section_diagnostics(recs, None, tmp_path)
    analyze.section_holm(pvals, tmp_path)

    for name in ("00_margin_control", "01_behaviour", "02_flips", "03_reliance",
                 "05_diagnostics", "06_multiple_comparisons"):
        text = (tmp_path / f"{name}.md").read_text(encoding="utf-8")
        assert "Q4_K_M" in text or "diagnostic" in text.lower(), name

    flips = (tmp_path / "02_flips.md").read_text(encoding="utf-8")
    assert "+0.250" in flips, "one-sided true->fake shift should show as asymmetry"
    assert pvals, "paired tests should have produced p-values"


def test_analysis_reports_reliance_shift_direction(tmp_path):
    sys_path_setup = __import__("sys")
    sys_path_setup.path.insert(0, str(__import__("pathlib").Path(__file__)
                                      .resolve().parents[1] / "scripts"))
    import analyze  # noqa: PLC0415

    analyze.section_reliance(_synthetic_records(), None, tmp_path)
    rows = (tmp_path / "03_reliance.csv").read_text(encoding="utf-8").splitlines()
    deltas = [float(r.split(",")[5]) for r in rows[1:]]
    assert deltas and all(d > 0 for d in deltas), "planted shift is toward context"
