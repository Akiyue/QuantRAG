"""Turn runs/*.jsonl into the quantities the paper reports.

The metrics in metrics.py are pure functions over numbers. This module is the
layer that decides *which* numbers go into them: which records pair with which,
which conditions a quantity is even defined for, and which items are in scope.

Two rules encoded here rather than left to the caller:

* The reliance score is undefined under C1. There the document endorses the true
  answer, so the "context answer" and the "parametric answer" are the same
  string and R collapses to zero by construction. Reporting that as evidence of
  balanced reliance would be an artefact.
* Compliance is undefined wherever no document endorses either answer (C0, C4).
  Both candidates are still scored there, but only as the baseline margin.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal

from .metrics import p_context, reliance_score
from .normalize import Label

BASELINE_PRECISION = "F16"


@dataclass(frozen=True, slots=True)
class ItemKey:
    """Identifies one measurement across precisions.

    Precision is deliberately absent: this is the unit that gets *paired* when
    comparing F16 against a quantized model, which is what gives the paired
    tests their power.
    """
    fact_id: str
    lang: str
    condition: str
    mode: str

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (self.fact_id, self.lang, self.condition, self.mode)


@dataclass(slots=True)
class Record:
    key: ItemKey
    model_id: str
    precision: str
    kind: str
    domain: str
    endorsed: str
    # scoring
    r: float | None = None
    p_ctx: float | None = None
    lp_fake: float | None = None
    lp_true: float | None = None
    n_tok_fake: int = 0
    n_tok_true: int = 0
    # generation
    label: Label | None = None
    text: str = ""
    both_present: bool = False


def iter_records(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load(run_dir: str | Path, pass_name: str = "main") -> list[Record]:
    """Read every JSONL for one pass and derive the per-record quantities."""
    run_dir = Path(run_dir)
    out: list[Record] = []
    errors = 0

    for path in sorted(run_dir.glob(f"{pass_name}__*.jsonl")):
        if path.name.endswith("manifest.json"):
            continue
        for raw in iter_records(path):
            if "error" in raw:
                errors += 1
                continue
            key = ItemKey(raw["fact_id"], raw["lang"], raw["condition"], raw["mode"])
            rec = Record(
                key=key,
                model_id=raw["model_id"],
                precision=raw["precision"],
                kind=raw["kind"],
                domain=raw.get("domain", ""),
                endorsed=raw.get("endorsed", "none"),
            )
            if raw["kind"] == "score":
                sf, st = raw["score_fake"], raw["score_true"]
                rec.lp_fake, rec.lp_true = sf["mean_logprob"], st["mean_logprob"]
                rec.n_tok_fake, rec.n_tok_true = sf["n_tokens"], st["n_tokens"]
                # Orientation is fixed as fake-minus-true so that R keeps one
                # meaning across conditions; under C2 that is exactly
                # context-minus-parametric.
                if raw.get("endorsed") != "true":
                    rec.r = reliance_score(rec.lp_fake, rec.lp_true)
                    rec.p_ctx = p_context(rec.r)
            else:
                rec.label = Label(raw["label"]) if raw.get("label") else None
                rec.text = raw.get("generation", {}).get("text", "")
                rec.both_present = bool(raw.get("both_present"))
            out.append(rec)

    if errors:
        print(f"  warning: skipped {errors} records with boundary errors")
    return out


# --------------------------------------------------------------- indexing

def index(records: Iterable[Record], kind: str) -> dict[tuple, dict[str, Record]]:
    """{(model, item key...): {precision: Record}} for one record kind."""
    out: dict[tuple, dict[str, Record]] = defaultdict(dict)
    for r in records:
        if r.kind != kind:
            continue
        out[(r.model_id, *r.key.as_tuple())][r.precision] = r
    return out


def precisions(records: Iterable[Record]) -> list[str]:
    """Precision levels, baseline first then by descending nominal width."""
    order = ["F16", "Q8_0", "Q4_K_M", "Q3_K_M", "AWQ4"]
    seen = {r.precision for r in records}
    return [p for p in order if p in seen] + sorted(seen - set(order))


def models(records: Iterable[Record]) -> list[str]:
    return sorted({r.model_id for r in records})


# ------------------------------------------------------------ paired views

@dataclass(slots=True)
class PairedItem:
    """One item measured at the baseline and at one quantized precision."""
    key: ItemKey
    model_id: str
    domain: str
    baseline_label: Label | None = None
    quant_label: Label | None = None
    baseline_r: float | None = None
    quant_r: float | None = None
    baseline_p_ctx: float | None = None
    quant_p_ctx: float | None = None


def paired(
    records: list[Record],
    model_id: str,
    precision: str,
    *,
    condition: str,
    mode: str | None = None,
    lang: str | None = None,
    scope: set[tuple[str, str]] | None = None,
    baseline: str = BASELINE_PRECISION,
) -> list[PairedItem]:
    """Items present at both precisions, optionally restricted to a scope.

    `scope` is a set of (fact_id, lang) - normally the parametric-known subset.
    Restricting here rather than after aggregation matters: the known subset is
    defined per language, and mixing scoped and unscoped items would make the
    bilingual comparison incoherent.
    """
    gen = index(records, "generate")
    sco = index(records, "score")

    keys = {k for k in set(gen) | set(sco) if k[0] == model_id and k[3] == condition}
    out: list[PairedItem] = []

    for k in sorted(keys):
        _, fact_id, lg, cond, md = k
        if mode is not None and md != mode:
            continue
        if lang is not None and lg != lang:
            continue
        if scope is not None and (fact_id, lg) not in scope:
            continue

        g, s = gen.get(k, {}), sco.get(k, {})
        if baseline not in g and baseline not in s:
            continue
        if precision not in g and precision not in s:
            continue

        dom = next((r.domain for r in list(g.values()) + list(s.values()) if r.domain), "")
        out.append(PairedItem(
            key=ItemKey(fact_id, lg, cond, md), model_id=model_id, domain=dom,
            baseline_label=g.get(baseline).label if g.get(baseline) else None,
            quant_label=g.get(precision).label if g.get(precision) else None,
            baseline_r=s.get(baseline).r if s.get(baseline) else None,
            quant_r=s.get(precision).r if s.get(precision) else None,
            baseline_p_ctx=s.get(baseline).p_ctx if s.get(baseline) else None,
            quant_p_ctx=s.get(precision).p_ctx if s.get(precision) else None,
        ))
    return out


def labelled(items: list[PairedItem]) -> tuple[list[Label], list[Label]]:
    """Baseline and quantized labels for items where both exist."""
    both = [(i.baseline_label, i.quant_label) for i in items
            if i.baseline_label is not None and i.quant_label is not None]
    return [b for b, _ in both], [q for _, q in both]


def scored(items: list[PairedItem]) -> tuple[list[float], list[float]]:
    both = [(i.baseline_r, i.quant_r) for i in items
            if i.baseline_r is not None and i.quant_r is not None]
    return [b for b, _ in both], [q for _, q in both]


# --------------------------------------------------------- known subsets

Split = Literal["known_all", "known_fp16", "unknown"]


def known_subsets(filter_records: list[Record]) -> dict[str, set[tuple[str, str]]]:
    """Partition (fact_id, lang) by whether the model knew the fact unprompted.

    Built from the C0 pass on q_filter. Evaluation then uses q_eval, so the
    selection question and the measurement question are different paraphrases.
    Filtering and evaluating on the same question would be selection on the
    baseline arm: items would be chosen partly for baseline noise, and
    regression to the mean alone would make the quantized model look worse.

    known_all  - correct at EVERY precision. The primary analysis set, because
                 it is not selected on any single arm.
    known_fp16 - correct at the baseline only. Reported as a secondary view,
                 with the bias stated.
    unknown    - correct nowhere. The control for pure retrieval use.
    """
    by_item: dict[tuple[str, str], dict[str, Label]] = defaultdict(dict)
    for r in filter_records:
        if r.kind != "generate" or r.label is None:
            continue
        by_item[(r.key.fact_id, r.key.lang)][r.precision] = r.label

    out: dict[str, set[tuple[str, str]]] = {
        "known_all": set(), "known_fp16": set(), "unknown": set()
    }
    for item, per_prec in by_item.items():
        if not per_prec:
            continue
        correct = {p: lb is Label.TRUE for p, lb in per_prec.items()}
        if all(correct.values()):
            out["known_all"].add(item)
        elif correct.get(BASELINE_PRECISION, False):
            out["known_fp16"].add(item)
        elif not any(correct.values()):
            out["unknown"].add(item)
    return out


def load_scope(path: str | Path, split: str) -> set[tuple[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(f, l) for f, l in data[split]}


# ------------------------------------------------------------- calibration

def truncated_entropy(logprobs: Iterable[float]) -> float:
    """Entropy over a truncated (top-k) next-token distribution, in nats.

    Quantization changes how peaked the output distribution is, and a raw
    Delta R can be driven by that global sharpening rather than by any change in
    which answer is preferred. Reporting this alongside is what lets a reader
    tell the two apart; it is also why P_ctx, which is far less scale-sensitive,
    is the metric of record.

    Truncated because only the top-k tail is stored. State that in the paper -
    it is a lower bound on the full entropy, comparable across arms only
    because k is fixed.
    """
    ps = [math.exp(lp) for lp in logprobs]
    total = sum(ps)
    if total <= 0:
        return float("nan")
    return -sum((p / total) * math.log(p / total) for p in ps if p > 0)
