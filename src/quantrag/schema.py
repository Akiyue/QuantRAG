"""Dataset schema and JSONL I/O.

One `Fact` per line in data/facts.jsonl. Every text field is stored per-language
so that the English and Vietnamese arms are *paired* on identical propositional
content - any EN/VI difference we later measure cannot be an artifact of
different facts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Literal

Lang = Literal["en", "vi"]
LANGS: tuple[Lang, ...] = ("en", "vi")

# Domains come from configs/relations.yaml rather than a hard-coded list.
# Stratification is by *relation*, which is the natural stratum in CounterFact
# and the axis along which Vietnamese label coverage actually varies; topical
# domains would be an invented grouping laid over it.


@dataclass(frozen=True, slots=True)
class Fact:
    fact_id: str
    domain: str
    relation: str

    # Per-language surface forms. Keys are exactly LANGS.
    subject: dict[str, str]
    object_true: dict[str, str]
    object_fake: dict[str, str]

    # q_filter selects the parametric-known subset; q_eval is used for every
    # reported measurement. They MUST be different paraphrases -- filtering and
    # evaluating on the same question is selection on the baseline arm and
    # manufactures a spurious quantization effect (see docs/PLAN.md 3.5).
    q_filter: dict[str, str]
    q_eval: dict[str, str]

    evidence_true: dict[str, str]
    evidence_fake: dict[str, str]
    evidence_irrelevant: dict[str, str]

    # Surface variants used by the XAI analyses.
    # evidence_fake_para: same proposition, fully rewritten surface form.
    # evidence_control: length-matched but information-free (control span).
    evidence_fake_para: dict[str, str] = field(default_factory=dict)
    evidence_control: dict[str, str] = field(default_factory=dict)

    aliases_true: dict[str, list[str]] = field(default_factory=dict)
    aliases_fake: dict[str, list[str]] = field(default_factory=dict)

    # Wikidata provenance. Object QIDs come from CounterFact; the subject QID is
    # recovered by name search and verified against the relation claim, so a
    # record carrying one has been checked against live Wikidata.
    relation_id: str = ""
    subject_qid: str = ""
    object_true_qid: str = ""
    object_fake_qid: str = ""

    # False when the subject has no Vietnamese Wikidata label and keeps its
    # Latin spelling in both languages. That is correct Vietnamese practice for
    # foreign proper names, not a fallback - but it does mean the EN/VI contrast
    # for those items rests on the template and the object alone, so results
    # should be checked for a difference between the two groups.
    subject_localized: bool = False

    source: str = ""
    vi_translation_checked: bool = False

    def alias_set(self, which: Literal["true", "fake"], lang: Lang) -> list[str]:
        """Canonical surface form plus its aliases, for the evaluator."""
        obj = self.object_true if which == "true" else self.object_fake
        aliases = self.aliases_true if which == "true" else self.aliases_fake
        out = [obj[lang], *aliases.get(lang, [])]
        seen: dict[str, None] = {}
        for a in out:
            seen.setdefault(a, None)
        return list(seen)

    def validate(self) -> list[str]:
        """Return a list of problems; empty means the record is usable."""
        problems: list[str] = []
        if not self.domain:
            problems.append("empty domain")

        required = {
            "subject": self.subject,
            "object_true": self.object_true,
            "object_fake": self.object_fake,
            "q_filter": self.q_filter,
            "q_eval": self.q_eval,
            "evidence_true": self.evidence_true,
            "evidence_fake": self.evidence_fake,
            "evidence_irrelevant": self.evidence_irrelevant,
        }
        for name, d in required.items():
            missing = [lg for lg in LANGS if not d.get(lg)]
            if missing:
                problems.append(f"{name} missing languages {missing}")

        for lg in LANGS:
            if self.q_filter.get(lg) and self.q_filter.get(lg) == self.q_eval.get(lg):
                problems.append(
                    f"q_filter == q_eval for {lg!r}: filtering and evaluating on the "
                    f"same question biases the known-subset (see PLAN 3.5)"
                )
            true_o = (self.object_true.get(lg) or "").strip().casefold()
            fake_o = (self.object_fake.get(lg) or "").strip().casefold()
            if true_o and true_o == fake_o:
                problems.append(f"object_true == object_fake for {lg!r}")
            # The counterfactual must actually appear in the counterfactual evidence.
            ev = (self.evidence_fake.get(lg) or "").casefold()
            if fake_o and ev and fake_o not in ev:
                problems.append(f"object_fake not present in evidence_fake for {lg!r}")

        if not self.vi_translation_checked:
            problems.append("vi_translation_checked is False (human review required)")
        return problems

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


def read_facts(path: str | Path) -> list[Fact]:
    facts: list[Fact] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                facts.append(Fact.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return facts


def write_facts(path: str | Path, facts: list[Fact]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for f in facts:
            fh.write(json.dumps(f.to_dict(), ensure_ascii=False) + "\n")


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
