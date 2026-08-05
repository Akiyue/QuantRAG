"""Metrics, paired statistics and causal attribution.

Definitions follow docs/PLAN.md section 5.2. Two conventions are load-bearing:

* Reliance is computed from *length-normalised* log-probabilities. Raw sums are
  retained so the appendix can show conclusions do not flip, but the normalised
  form is what gets reported - otherwise the bilingual comparison measures the
  tokeniser.
* No rate is ever returned without a confidence interval.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .normalize import Label

FlipKind = Literal["true_to_fake", "fake_to_true", "answer_to_refusal",
                   "refusal_to_answer", "answer_to_other", "other"]


# --------------------------------------------------------------------------
# Reliance
# --------------------------------------------------------------------------

def reliance_score(mean_lp_context: float, mean_lp_parametric: float) -> float:
    """R = s(a_ctx) - s(a_par), on length-normalised scores.

    R > 0 leans toward the retrieved context, R < 0 toward parametric memory.
    """
    return mean_lp_context - mean_lp_parametric


def p_context(r: float) -> float:
    """Two-way renormalised preference, sigma(R).

    Reported alongside Delta R because quantization changes calibration: a raw
    Delta R can be driven by a global sharpening or flattening of the output
    distribution rather than by any change in which answer is preferred. This
    form is far less sensitive to that.
    """
    return 1.0 / (1.0 + math.exp(-r))


def delta_r(r_quant: float, r_baseline: float) -> float:
    return r_quant - r_baseline


# --------------------------------------------------------------------------
# Behavioural rates
# --------------------------------------------------------------------------

def rate(labels: Sequence[Label], target: Label) -> float:
    return sum(1 for x in labels if x is target) / len(labels) if labels else float("nan")


def ccr(labels: Sequence[Label], context_label: Label = Label.FAKE) -> float:
    """Context Compliance Rate under conflict.

    NEVER report this without stating the instruction mode. Under strict
    grounding a high CCR means the model followed instructions; under
    truth-seeking the same number means it was steered by misinformation.
    """
    return rate(labels, context_label)


def prr(labels: Sequence[Label]) -> float:
    """Parametric Retention Rate: kept the correct answer despite false evidence."""
    return rate(labels, Label.TRUE)


def classify_flip(before: Label, after: Label) -> FlipKind | None:
    """How an item's answer changed between two precisions. None means stable."""
    if before is after:
        return None
    if before is Label.TRUE and after is Label.FAKE:
        return "true_to_fake"
    if before is Label.FAKE and after is Label.TRUE:
        return "fake_to_true"
    if after is Label.REFUSAL:
        return "answer_to_refusal"
    if before is Label.REFUSAL:
        return "refusal_to_answer"
    if after is Label.OTHER:
        return "answer_to_other"
    return "other"


@dataclass(frozen=True, slots=True)
class FlipReport:
    qfr: float
    n: int
    breakdown: dict[str, int]

    def asymmetry(self) -> float:
        """true->fake minus fake->true, as a share of all items.

        This is the directional claim that survives a null net effect: even if
        the total flip count is unchanged, an imbalance in direction says
        quantization moved the arbitration rather than just adding noise.
        """
        if not self.n:
            return float("nan")
        return (self.breakdown.get("true_to_fake", 0)
                - self.breakdown.get("fake_to_true", 0)) / self.n


def quantization_flip_rate(
    baseline: Sequence[Label], quantized: Sequence[Label]
) -> FlipReport:
    """QFR with its breakdown.

    Reported as an extension of correctness agreement (arXiv 2607.08734) into
    the conflict setting - cite it, do not claim the bare instability finding as
    novel.
    """
    if len(baseline) != len(quantized):
        raise ValueError("paired sequences must have equal length")
    kinds = [classify_flip(b, q) for b, q in zip(baseline, quantized)]
    flips = [k for k in kinds if k is not None]
    n = len(baseline)
    return FlipReport(
        qfr=len(flips) / n if n else float("nan"),
        n=n,
        breakdown=dict(Counter(flips)),
    )


# --------------------------------------------------------------------------
# Paired statistics
# --------------------------------------------------------------------------

def paired_bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 1234,
) -> tuple[float, float, float]:
    """(point estimate, lower, upper) for the mean, resampling items.

    Items are the resampling unit because every configuration is evaluated on
    the same questions - that pairing is the whole reason the design has power.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(lo), float(hi)


def mcnemar(baseline_correct: Sequence[bool], quant_correct: Sequence[bool]) -> dict:
    """Exact McNemar test on paired correct/incorrect outcomes."""
    from scipy.stats import binomtest

    if len(baseline_correct) != len(quant_correct):
        raise ValueError("paired sequences must have equal length")
    b = sum(1 for x, y in zip(baseline_correct, quant_correct) if x and not y)
    c = sum(1 for x, y in zip(baseline_correct, quant_correct) if y and not x)
    n = b + c
    p = binomtest(b, n, 0.5).pvalue if n else 1.0
    return {
        "b_baseline_only": b,
        "c_quant_only": c,
        "n_discordant": n,
        "p_value": float(p),
        # Odds ratio of the discordant pairs; the effect size reviewers expect
        # alongside the p-value.
        "odds_ratio": (b / c) if c else float("inf") if b else float("nan"),
    }


def wilcoxon_paired(x: Sequence[float], y: Sequence[float]) -> dict:
    """Wilcoxon signed-rank with a rank-biserial effect size."""
    from scipy.stats import wilcoxon as _w

    a, b = np.asarray(x, float), np.asarray(y, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    d = a - b
    if np.all(d == 0) or d.size == 0:
        return {"statistic": float("nan"), "p_value": 1.0,
                "rank_biserial": 0.0, "n": int(d.size)}
    res = _w(a, b, zero_method="wilcox", alternative="two-sided")
    nz = d[d != 0]
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1
    r_plus = ranks[nz > 0].sum()
    total = ranks.sum()
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "rank_biserial": float(2 * r_plus / total - 1) if total else 0.0,
        "n": int(nz.size),
    }


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni across a family of tests."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    prev = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        prev = adj
        out[name] = {"p_raw": p, "p_holm": adj, "reject": adj <= alpha}
    return out


# --------------------------------------------------------------------------
# Margin control - the falsification test (PLAN 6.3)
# --------------------------------------------------------------------------

def auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Rank-based AUC (Mann-Whitney), no sklearn dependency."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, s.size + 1)
    # Average ranks within ties so the statistic stays unbiased.
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def margin_control(baseline_r: Sequence[float], flipped: Sequence[bool],
                   n_deciles: int = 10) -> dict:
    """Can flips be explained away as noise near the decision boundary?

    If |R_baseline| predicts flipping almost perfectly, quantization did nothing
    structural - items simply sat close to the boundary and jittered across it,
    and the paper's central claim collapses into a study of margin sensitivity.
    Run this on day 12, before committing week 3 to the attribution analysis.
    """
    margin = np.abs(np.asarray(baseline_r, float))
    flips = np.asarray(flipped, bool)
    # Low margin should predict flipping, so score on the negated margin.
    a = auc(-margin, flips)

    order = np.argsort(margin, kind="mergesort")
    buckets = np.array_split(order, min(n_deciles, max(1, margin.size)))
    by_decile = [
        {
            "decile": i,
            "mean_margin": float(margin[b].mean()) if b.size else float("nan"),
            "flip_rate": float(flips[b].mean()) if b.size else float("nan"),
            "n": int(b.size),
        }
        for i, b in enumerate(buckets)
    ]
    return {
        "auc_margin_predicts_flip": a,
        "by_decile": by_decile,
        "interpretation": (
            "AUC near 1.0: flips are boundary noise, the central claim does not "
            "hold. AUC near 0.5: flips are not reducible to baseline margin."
        ),
    }


# --------------------------------------------------------------------------
# Causal attribution (PLAN 6.1-6.2)
# --------------------------------------------------------------------------

def attribution(full_logprob: float, ablated_logprob: float) -> float:
    """A(s) = log P(y | x) - log P(y | x \\ s)."""
    return full_logprob - ablated_logprob


def corrected_attribution(
    full_logprob: float, evidence_ablated: float, control_ablated: float
) -> float:
    """A*(evidence) = A(evidence) - A(control).

    Deleting any tokens lowers the log-probability, so a bare A(evidence) is
    uninterpretable. The control span is length-matched but carries no
    information about the answer; subtracting it is what turns the number into
    evidence rather than an artefact of prompt length.
    """
    return attribution(full_logprob, evidence_ablated) - attribution(
        full_logprob, control_ablated
    )


def semantic_gap(r_original: float, r_paraphrased: float) -> float:
    """R(evidence) - R(evidence_paraphrased).

    Near zero means the model is grounding on meaning; large and positive means
    it was leaning on the specific surface form, i.e. copying a string. If this
    grows as precision drops, context grounding is degrading from semantic to
    surface - a mechanism-flavoured claim reachable with input-level methods only.
    """
    return r_original - r_paraphrased


def override_threshold(doses: Sequence[int], p_ctx: Sequence[float]) -> float:
    """Smallest evidence dose at which P_ctx crosses 0.5, linearly interpolated.

    Raw material for the journal metric contribution (Context Override
    Threshold). Collect the dose-response data during the conference runs even
    if this is not analysed then: re-running months later loses comparability
    with the conference numbers, and that cost cannot be undone.
    """
    d = np.asarray(doses, float)
    p = np.asarray(p_ctx, float)
    order = np.argsort(d)
    d, p = d[order], p[order]
    for i in range(1, d.size):
        if p[i - 1] < 0.5 <= p[i]:
            span = p[i] - p[i - 1]
            frac = (0.5 - p[i - 1]) / span if span else 0.0
            return float(d[i - 1] + frac * (d[i] - d[i - 1]))
    return float("inf") if p.size and p[-1] < 0.5 else float(d[0]) if p.size else float("nan")
