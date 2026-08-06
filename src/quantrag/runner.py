"""Grid runner: resumable, one JSONL line per measurement.

Runs are long and get interrupted - accept that up front rather than
discovering it at 3am. Every record carries its own key; on restart the keys
already on disk are skipped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterator, Sequence

from tqdm import tqdm

from .backends import Backend
from .backends.base import BoundaryError, DegenerateArm, DegenerateOutput
from .normalize import Label, classify
from .prompts import (
    PROMPT_VERSION,
    Condition,
    Mode,
    build_prompt,
    candidates_for,
    dose_documents,
)
from .schema import Fact, Lang, append_jsonl, iter_jsonl


@dataclass(frozen=True, slots=True)
class Cell:
    """One point of the experimental grid."""
    fact_id: str
    lang: Lang
    condition: str
    mode: Mode
    kind: str            # "score" | "generate"
    question_kind: str = "q_eval"

    def key(self, model_id: str, precision: str) -> str:
        return "|".join([
            model_id, precision, self.fact_id, self.lang,
            self.condition, self.mode, self.kind, self.question_kind,
        ])


def grid(
    facts: Sequence[Fact],
    *,
    languages: Sequence[Lang],
    conditions: Sequence[str],
    modes: Sequence[Mode],
    kinds: Sequence[str] = ("score", "generate"),
    question_kind: str = "q_eval",
) -> Iterator[Cell]:
    for fact, lang, cond, mode, kind in product(facts, languages, conditions, modes, kinds):
        # C0 has no document, so the two instruction modes would be identical
        # prompts. Run it once and label it 'strict' to avoid double counting.
        if cond == "C0" and mode != modes[0]:
            continue
        yield Cell(fact.fact_id, lang, cond, mode, kind, question_kind)


def _dose_cells(facts: Sequence[Fact], languages: Sequence[Lang],
                modes: Sequence[Mode], doses: Sequence[int]) -> Iterator[Cell]:
    for fact, lang, mode, dose in product(facts, languages, modes, doses):
        yield Cell(fact.fact_id, lang, f"DOSE{dose}", mode, "score")


def run(
    backend: Backend,
    facts: Sequence[Fact],
    out_path: str | Path,
    *,
    languages: Sequence[Lang] = ("en", "vi"),
    conditions: Sequence[str] = ("C0", "C1", "C2", "C4"),
    modes: Sequence[Mode] = ("strict", "truth_seeking"),
    kinds: Sequence[str] = ("score", "generate"),
    question_kind: str = "q_eval",
    doses: Sequence[int] = (),
    max_tokens: int = 32,
    strip_diacritics: bool = False,
    resume: bool = True,
    progress: bool = True,
) -> dict:
    """Execute the grid, appending to `out_path`.

    Returns a small summary; the numbers themselves live in the JSONL.
    """
    out_path = Path(out_path)
    by_id = {f.fact_id: f for f in facts}

    done: set[str] = set()
    if resume:
        done = {rec["key"] for rec in iter_jsonl(out_path) if "key" in rec}

    cells = list(grid(
        facts, languages=languages, conditions=conditions, modes=modes,
        kinds=kinds, question_kind=question_kind,
    ))
    if doses:
        cells += list(_dose_cells(facts, languages, modes, doses))

    pending = [c for c in cells if c.key(backend.model_id, backend.precision) not in done]

    env = backend.env()
    stats = {"total": len(cells), "skipped": len(cells) - len(pending),
             "written": 0, "errors": 0}

    # An arm that starts producing NaN does not recover on its own, and a run
    # that is half garbage is not a partial result - it is an arm that has to be
    # redone. Stopping loudly beats writing a thousand error records and moving
    # on, which during a long grid would be noticed only at analysis time.
    abort_after = max(20, int(0.05 * len(pending)))

    it = tqdm(pending, desc=f"{backend.model_id}/{backend.precision}",
              disable=not progress)
    for cell in it:
        fact = by_id[cell.fact_id]
        try:
            record = _run_cell(backend, fact, cell, max_tokens=max_tokens,
                               strip_diacritics=strip_diacritics)
        except (BoundaryError, DegenerateOutput) as exc:
            # Both invalidate the cell rather than merely degrading it. Writing
            # a number here would put a tokenisation fault or a GPU glitch into
            # the denominator of every rate, where it is indistinguishable from
            # a genuine wrong answer.
            kind = ("boundary" if isinstance(exc, BoundaryError) else "degenerate")
            stats["errors"] += 1
            stats.setdefault(kind, 0)
            stats[kind] += 1
            append_jsonl(out_path, {
                "key": cell.key(backend.model_id, backend.precision),
                "error": kind, "detail": str(exc),
                "model_id": backend.model_id, "precision": backend.precision,
            })
            if stats.get("degenerate", 0) >= abort_after:
                raise DegenerateArm(
                    f"{backend.model_id} {backend.precision}: "
                    f"{stats['degenerate']} degenerate cells out of "
                    f"{stats['written'] + stats['errors']} attempted. This arm is "
                    f"not producing usable output - NaN logits at this rate are a "
                    f"hardware or memory problem, not a property of the model. "
                    f"Delete its JSONL and rerun once the cause is found; a "
                    f"partially-degenerate arm must not be analysed."
                ) from exc
            continue

        record.update({
            "key": cell.key(backend.model_id, backend.precision),
            "model_id": backend.model_id,
            "precision": backend.precision,
            "prompt_version": PROMPT_VERSION,
            "env": env,
            "ts": time.time(),
        })
        append_jsonl(out_path, record)
        stats["written"] += 1

    return stats


def _run_cell(backend: Backend, fact: Fact, cell: Cell, *,
              max_tokens: int, strip_diacritics: bool) -> dict:
    lang: Lang = cell.lang  # type: ignore[assignment]

    if cell.condition.startswith("DOSE"):
        dose = int(cell.condition[4:])
        docs = dose_documents(fact, lang, dose)
        prompt = build_prompt(fact, lang=lang, mode=cell.mode, documents=docs,
                              question_kind=cell.question_kind)
        cands = candidates_for(fact, "C2", lang)
    else:
        cond: Condition = cell.condition  # type: ignore[assignment]
        prompt = build_prompt(fact, lang=lang, mode=cell.mode, condition=cond,
                              question_kind=cell.question_kind)
        cands = candidates_for(fact, cond, lang)

    base = {
        "fact_id": fact.fact_id, "domain": fact.domain, "lang": lang,
        "condition": cell.condition, "mode": cell.mode, "kind": cell.kind,
        "question_kind": cell.question_kind,
        "candidate_fake": cands.fake, "candidate_true": cands.true,
        # "none" means no document endorses either answer (C0, C4). Compliance
        # is undefined there; analysis code must check this before computing CCR.
        "endorsed": cands.endorsed,
    }

    if cell.kind == "score":
        fake_s, true_s = backend.score(prompt, cands.as_continuations())
        base["score_fake"] = fake_s.to_dict()
        base["score_true"] = true_s.to_dict()
        # R and P_ctx are derived at analysis time from these fields; storing the
        # raw per-token log-probabilities keeps every later re-analysis - including
        # the journal extension - possible without re-running the grid.
        return base

    if cell.kind == "generate":
        gen = backend.generate(prompt, max_tokens=max_tokens)
        cls = classify(
            gen.text, lang=lang,
            aliases_true=fact.alias_set("true", lang),
            aliases_fake=fact.alias_set("fake", lang),
            strip_diacritics=strip_diacritics,
        )
        base["generation"] = gen.to_dict()
        base["label"] = cls.label.value
        base["matched_on"] = cls.matched_on
        base["both_present"] = cls.both_present
        return base

    raise ValueError(f"unknown kind {cell.kind!r}")


def label_of(record: dict) -> Label | None:
    v = record.get("label")
    return Label(v) if v else None
