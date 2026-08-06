"""Tests for the pieces where a silent bug would corrupt every downstream number."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from quantrag.backends.base import BoundaryError, MockBackend, split_boundary
from quantrag.metrics import (
    auc,
    ccr,
    classify_flip,
    corrected_attribution,
    holm,
    margin_control,
    override_threshold,
    p_context,
    paired_bootstrap_ci,
    prr,
    quantization_flip_rate,
    reliance_score,
    semantic_gap,
)
from quantrag.normalize import Label, classify, fold, leading_answer, matches_any
from quantrag.prompts import (
    build_prompt,
    candidates_for,
    documents_for,
    dose_documents,
)
from quantrag.runner import run
from quantrag.schema import read_facts

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "facts.sample.jsonl"


@pytest.fixture(scope="module")
def facts():
    return read_facts(SAMPLE)


# -- schema ---------------------------------------------------------------

def test_sample_facts_are_valid(facts):
    for f in facts:
        assert f.validate() == [], f"{f.fact_id}: {f.validate()}"


def test_filter_and_eval_questions_differ(facts):
    """Reusing one question for both filtering and evaluation biases the
    known-subset toward the baseline arm."""
    for f in facts:
        for lg in ("en", "vi"):
            assert f.q_filter[lg] != f.q_eval[lg]


def test_validate_rejects_shared_question(facts):
    import dataclasses

    bad = dataclasses.replace(facts[0], q_eval=dict(facts[0].q_filter))
    assert any("q_filter == q_eval" in p for p in bad.validate())


def test_alias_set_includes_canonical_form(facts):
    f = facts[0]
    assert "London" in f.alias_set("fake", "en")
    assert "Luân Đôn" in f.alias_set("fake", "vi")


# -- prompts --------------------------------------------------------------

def test_conditions_place_expected_documents(facts):
    f = facts[0]
    assert documents_for(f, "C0", "en") == []
    assert documents_for(f, "C1", "en") == [f.evidence_true["en"]]
    assert documents_for(f, "C2", "en") == [f.evidence_fake["en"]]
    assert documents_for(f, "C3_fake_first", "en")[0] == f.evidence_fake["en"]
    assert documents_for(f, "C3_true_first", "en")[0] == f.evidence_true["en"]


def test_prompt_ends_at_assistant_turn(facts):
    p = build_prompt(facts[0], lang="en", mode="strict", condition="C2")
    assert p.endswith("<|im_start|>assistant\n")
    assert "London" in p


def test_modes_produce_different_instructions(facts):
    strict = build_prompt(facts[0], lang="en", mode="strict", condition="C2")
    truth = build_prompt(facts[0], lang="en", mode="truth_seeking", condition="C2")
    assert strict != truth


def test_dropping_evidence_removes_the_document(facts):
    f = facts[0]
    full = build_prompt(f, lang="en", mode="strict", condition="C2")
    ablated = build_prompt(f, lang="en", mode="strict", condition="C2",
                           drop_spans=("evidence",))
    assert "London" in full and "London" not in ablated


def test_candidate_answers_follow_the_condition(facts):
    f = facts[0]
    assert candidates_for(f, "C1", "en").context_answer == "Paris"   # agrees
    assert candidates_for(f, "C2", "en").context_answer == "London"  # conflicts
    assert candidates_for(f, "C2", "en").true == "Paris"


def test_compliance_is_undefined_without_an_endorsing_document(facts):
    """C0 has no document and C4's document asserts neither answer, so there is
    nothing to comply with. Both objects are still scored - that preference is
    the baseline margin the falsification test needs."""
    f = facts[0]
    for cond in ("C0", "C4"):
        c = candidates_for(f, cond, "en")
        assert c.endorsed == "none"
        assert c.context_answer is None
        assert set(c.as_continuations()) == {"Paris", "London"}
        assert c.reliance_pair() == ("London", "Paris")  # stable orientation


def test_dose_ladder_increases_evidence_pressure(facts):
    f = facts[0]
    assert dose_documents(f, "en", 0) == []
    counts = [
        sum(d.count("London") for d in dose_documents(f, "en", k))
        for k in (1, 2, 3, 4)
    ]
    assert counts == sorted(counts) and counts[-1] > counts[0]


def test_paraphrase_condition_requires_human_written_text(facts):
    import dataclasses

    stripped = dataclasses.replace(facts[0], evidence_fake_para={})
    with pytest.raises(ValueError, match="evidence_fake_para"):
        documents_for(stripped, "C2_para", "en")


# -- tokenisation boundary ------------------------------------------------

def test_split_boundary_finds_the_continuation():
    assert split_boundary([0, 4, 9, 12], 12) == 3


def test_split_boundary_rejects_a_merged_token():
    """If a token straddles the prompt/answer boundary, the answer's
    log-probability silently includes part of the prompt."""
    with pytest.raises(BoundaryError):
        split_boundary([0, 4, 9, 11, 15], 12)


# -- normalisation --------------------------------------------------------

def test_fold_is_case_and_punctuation_insensitive():
    assert fold("  Paris.  ") == fold("paris")


def test_diacritic_stripping_handles_vietnamese_d():
    """'đ' does not decompose under NFD and needs its own mapping."""
    assert fold("Luân Đôn", strip_diacritics=True) == "luan don"


def test_tones_are_preserved_by_default():
    assert fold("Luân Đôn") != fold("Luan Don")
    assert fold("Luân Đôn", strip_diacritics=True) == fold("Luan Don", strip_diacritics=True)


def test_matching_respects_word_boundaries():
    assert not matches_any("Viennese pastry", ["Vienna"], strip_diacritics=False)
    assert matches_any("Vienna, Austria", ["Vienna"], strip_diacritics=False)


def test_leading_answer_stops_at_the_first_sentence_break():
    assert leading_answer("Paris. However, the document says London.") == "Paris"


def test_truth_seeking_answer_is_labelled_by_its_leading_span():
    """'Paris. The document appears incorrect - it claims London.' is retention,
    not compliance, even though both entities appear."""
    c = classify(
        "Paris. The document appears incorrect; it claims London.",
        lang="en", aliases_true=["Paris"], aliases_fake=["London"],
    )
    assert c.label is Label.TRUE
    assert c.both_present is True   # flagged for the manual validation sample


def test_refusal_beats_entity_mentions():
    c = classify(
        "I cannot determine this from the document, which says London.",
        lang="en", aliases_true=["Paris"], aliases_fake=["London"],
    )
    assert c.label is Label.REFUSAL


def test_vietnamese_alias_is_matched():
    c = classify("Luân Đôn", lang="vi",
                 aliases_true=["Paris"], aliases_fake=["London", "Luân Đôn"])
    assert c.label is Label.FAKE


def test_unknown_answer_is_other():
    c = classify("Berlin", lang="en", aliases_true=["Paris"], aliases_fake=["London"])
    assert c.label is Label.OTHER


# -- metrics --------------------------------------------------------------

def test_reliance_sign_and_probability():
    assert reliance_score(-1.0, -3.0) > 0      # context preferred
    assert reliance_score(-3.0, -1.0) < 0      # memory preferred
    assert p_context(0.0) == pytest.approx(0.5)
    assert p_context(2.0) > 0.85


def test_rates_are_complementary_under_conflict():
    labels = [Label.FAKE, Label.FAKE, Label.TRUE, Label.OTHER]
    assert ccr(labels) == pytest.approx(0.5)
    assert prr(labels) == pytest.approx(0.25)


def test_flip_classification():
    assert classify_flip(Label.TRUE, Label.TRUE) is None
    assert classify_flip(Label.TRUE, Label.FAKE) == "true_to_fake"
    assert classify_flip(Label.FAKE, Label.TRUE) == "fake_to_true"
    assert classify_flip(Label.TRUE, Label.REFUSAL) == "answer_to_refusal"


def test_flip_asymmetry_survives_a_null_total():
    """Equal flip counts in both directions vs a one-sided shift: QFR cannot
    tell them apart, asymmetry can."""
    balanced = quantization_flip_rate(
        [Label.TRUE, Label.FAKE], [Label.FAKE, Label.TRUE]
    )
    one_sided = quantization_flip_rate(
        [Label.TRUE, Label.TRUE], [Label.FAKE, Label.FAKE]
    )
    assert balanced.qfr == one_sided.qfr == 1.0
    assert balanced.asymmetry() == pytest.approx(0.0)
    assert one_sided.asymmetry() == pytest.approx(1.0)


def test_bootstrap_ci_brackets_the_mean():
    point, lo, hi = paired_bootstrap_ci([0.0, 1.0] * 50, n_boot=2000)
    assert lo <= point <= hi
    assert point == pytest.approx(0.5, abs=0.01)


def test_holm_is_monotone_and_no_smaller_than_raw():
    out = holm({"a": 0.001, "b": 0.02, "c": 0.4})
    assert out["a"]["p_holm"] <= out["b"]["p_holm"] <= out["c"]["p_holm"]
    for k, v in out.items():
        assert v["p_holm"] >= v["p_raw"]


def test_auc_separates_perfectly_and_not_at_all():
    assert auc([1.0, 2.0, 3.0, 4.0], [False, False, True, True]) == pytest.approx(1.0)
    assert auc([1.0, 1.0, 1.0, 1.0], [False, True, False, True]) == pytest.approx(0.5)


def test_margin_control_detects_boundary_noise():
    """The falsification test: if flips happen only where the baseline margin
    was small, the central claim reduces to margin sensitivity."""
    margins = [0.01, 0.02, 0.03, 0.04, 3.0, 3.1, 3.2, 3.3]
    flips = [True, True, True, True, False, False, False, False]
    noise = margin_control(margins, flips)
    assert noise["auc_margin_predicts_flip"] == pytest.approx(1.0)

    # Flipped margins {1,4,5,8} interleave symmetrically with unflipped
    # {2,3,6,7}, so margin carries no information about flipping.
    structural = margin_control(
        [1, 2, 3, 4, 5, 6, 7, 8],
        [True, False, False, True, True, False, False, True],
    )
    assert structural["auc_margin_predicts_flip"] == pytest.approx(0.5)


def test_corrected_attribution_removes_the_length_effect():
    """Deleting any span lowers the log-probability; only the excess over a
    length-matched control counts as evidence."""
    assert corrected_attribution(-1.0, -4.0, -2.0) == pytest.approx(2.0)
    assert corrected_attribution(-1.0, -2.0, -2.0) == pytest.approx(0.0)


def test_semantic_gap_sign():
    assert semantic_gap(2.0, 0.5) > 0    # reliance collapsed under paraphrase
    assert semantic_gap(2.0, 2.0) == 0   # grounding survived rewording


def test_override_threshold_interpolates_the_crossing():
    assert override_threshold([0, 1, 2, 3], [0.1, 0.2, 0.6, 0.9]) == pytest.approx(1.75)
    assert math.isinf(override_threshold([0, 1, 2], [0.1, 0.2, 0.3]))


# -- end to end -----------------------------------------------------------

def test_grid_runs_and_resumes(tmp_path, facts):
    out = tmp_path / "run.jsonl"
    backend = MockBackend(model_id="m", precision="Q4_K_M")

    first = run(backend, facts, out, languages=("en", "vi"),
                conditions=("C0", "C1", "C2"), progress=False)
    assert first["written"] > 0
    assert first["errors"] == 0

    second = run(backend, facts, out, languages=("en", "vi"),
                 conditions=("C0", "C1", "C2"), progress=False)
    assert second["written"] == 0, "resume must skip completed cells"
    assert second["skipped"] == first["written"]


def test_c0_is_not_duplicated_across_modes(tmp_path, facts):
    """C0 has no document, so both instruction modes render the same prompt."""
    out = tmp_path / "run.jsonl"
    run(MockBackend(), facts, out, languages=("en",), conditions=("C0",),
        kinds=("score",), progress=False)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(facts)


def test_records_carry_provenance(tmp_path, facts):
    import json

    out = tmp_path / "run.jsonl"
    run(MockBackend(), facts[:1], out, languages=("en",), conditions=("C2",),
        kinds=("score",), progress=False)
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert rec["prompt_version"] and rec["env"] and rec["precision"]
    assert rec["endorsed"] == "fake"
    # Per-token log-probabilities must survive: re-running the grid later is the
    # expensive path, and the journal extension needs them.
    assert rec["score_fake"]["token_logprobs"]


# -- degenerate output ----------------------------------------------------

def test_degenerate_run_of_punctuation_is_rejected():
    """llama.cpp emits a run of '!' when the logits come back NaN. Recorded as
    OTHER it would be indistinguishable from a genuine wrong answer and would
    sit in the denominator of every rate."""
    from quantrag.backends.base import DegenerateOutput, check_degenerate

    with pytest.raises(DegenerateOutput):
        check_degenerate("!!!!!!!!!!!!!!!!")
    with pytest.raises(DegenerateOutput):
        check_degenerate("  ????????????  ")


def test_real_answers_are_not_mistaken_for_degenerate():
    from quantrag.backends.base import check_degenerate

    for text in ("Paris", "London", "Luân Đôn", "aaa", "!!", "Paris!!!",
                 "Thủ đô của Pháp là London."):
        check_degenerate(text)   # must not raise
