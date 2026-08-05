"""Answer normalisation and classification.

This is the single most likely source of silently wrong numbers in the whole
study. Vietnamese entities have two legitimate surface forms - 'London' and
'Luân Đôn', 'Vienna' and 'Viên' - and diacritics may or may not be typed. An
evaluator that misses those inflates the OTHER bucket and quietly deflates both
CCR and PRR.

Nothing here is trusted on faith: 200 randomly sampled items must be labelled by
hand and the agreement reported in the paper (docs/PLAN.md 5.3).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

from .schema import Lang

# Vietnamese 'đ' does not decompose under NFD, so it needs an explicit mapping.
_EXTRA_FOLD = str.maketrans({"đ": "d", "Đ": "D"})

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")

# A leading answer is whatever precedes the first sentence break. Under
# truth_seeking the model is invited to add a warning after the entity, so we
# must not let that trailing prose decide the label.
_LEAD = re.compile(r"^[^.;:!?\n]+")


class Label(str, Enum):
    TRUE = "TRUE"        # the factually correct object
    FAKE = "FAKE"        # the object asserted by the counterfactual document
    REFUSAL = "REFUSAL"  # declined to answer
    OTHER = "OTHER"      # answered, but neither candidate


_REFUSAL_PATTERNS: dict[Lang, tuple[str, ...]] = {
    "en": (
        "i cannot", "i can't", "i am unable", "i'm unable", "cannot determine",
        "not able to", "no information", "does not contain", "doesn't contain",
        "insufficient information", "unable to answer", "i don't know",
        "i do not know", "not specified", "cannot answer",
    ),
    "vi": (
        "tôi không thể", "không thể xác định", "không có thông tin",
        "không đủ thông tin", "tôi không biết", "không rõ", "không được cung cấp",
        "tài liệu không", "không trả lời được",
    ),
}


def fold(text: str, *, strip_diacritics: bool = False) -> str:
    """Casefold, drop punctuation, squeeze whitespace; optionally strip tone marks.

    Diacritic stripping is off by default. It is a recall/precision trade: it
    rescues 'Luan Don' typed without tones, but in Vietnamese tone marks are
    lexical, so folding them can merge genuinely different words. Report which
    setting produced the reported numbers.
    """
    t = unicodedata.normalize("NFC", text).strip().casefold()
    if strip_diacritics:
        t = t.translate(_EXTRA_FOLD)
        t = "".join(
            c for c in unicodedata.normalize("NFD", t)
            if not unicodedata.combining(c)
        )
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def leading_answer(text: str) -> str:
    """The span the model offers as its answer, before any commentary."""
    m = _LEAD.match(text.strip())
    return (m.group(0) if m else text).strip()


def is_refusal(text: str, lang: Lang) -> bool:
    low = fold(text)
    return any(fold(p) in low for p in _REFUSAL_PATTERNS[lang])


def matches_any(candidate: str, aliases: Iterable[str], *, strip_diacritics: bool) -> bool:
    c = fold(candidate, strip_diacritics=strip_diacritics)
    if not c:
        return False
    for a in aliases:
        fa = fold(a, strip_diacritics=strip_diacritics)
        if not fa:
            continue
        if c == fa:
            return True
        # Word-boundary containment, so 'Vienna' does not match inside 'Viennese'.
        if re.search(rf"(?<!\w){re.escape(fa)}(?!\w)", c):
            return True
    return False


@dataclass(frozen=True, slots=True)
class Classification:
    label: Label
    matched_on: str          # "lead" or "full" - where the match was found
    both_present: bool       # both candidates appeared; manual review target
    raw: str


def classify(
    text: str,
    *,
    lang: Lang,
    aliases_true: Sequence[str],
    aliases_fake: Sequence[str],
    strip_diacritics: bool = False,
) -> Classification:
    """Label one generated answer.

    Order matters. A refusal is checked first because 'I cannot answer, the
    document says London' is a refusal, not compliance. Then the leading span is
    tried, and only if that is inconclusive do we fall back to the full text.
    """
    raw = text.strip()
    if not raw:
        return Classification(Label.REFUSAL, "lead", False, raw)

    if is_refusal(raw, lang):
        return Classification(Label.REFUSAL, "lead", False, raw)

    full_true = matches_any(raw, aliases_true, strip_diacritics=strip_diacritics)
    full_fake = matches_any(raw, aliases_fake, strip_diacritics=strip_diacritics)
    both = full_true and full_fake

    lead = leading_answer(raw)
    lead_true = matches_any(lead, aliases_true, strip_diacritics=strip_diacritics)
    lead_fake = matches_any(lead, aliases_fake, strip_diacritics=strip_diacritics)

    if lead_true != lead_fake:  # exactly one matched in the leading span
        return Classification(
            Label.TRUE if lead_true else Label.FAKE, "lead", both, raw
        )

    if full_true != full_fake:
        return Classification(
            Label.TRUE if full_true else Label.FAKE, "full", both, raw
        )

    # Neither matched, or both did with no way to prefer one.
    return Classification(Label.OTHER, "full", both, raw)
