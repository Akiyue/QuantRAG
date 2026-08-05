"""Prompt construction.

Every arm of the experiment - every precision, every backend - goes through this
module. The chat template is written out literally rather than delegated to a
tokenizer helper so that tier A (llama.cpp) and tier B (HF/AWQ) are guaranteed to
see byte-identical prompts. If the template ever differs between backends the
cross-quantizer robustness check is worthless.

Bump PROMPT_VERSION on any change and re-run affected configs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from .schema import Fact, Lang

PROMPT_VERSION = "v1"

Condition = Literal[
    "C0",              # no context               -> parametric baseline
    "C1",              # supporting evidence      -> ordinary retrieval
    "C2",              # counterfactual evidence  -> the conflict condition
    "C3_fake_first",   # both docs, fake first    -> inter-document conflict
    "C3_true_first",   # both docs, true first    -> + position bias
    "C4",              # irrelevant evidence      -> control for blind copying
    "C2_para",         # counterfactual, paraphrased surface form (XAI 6.4)
]

CONDITIONS: tuple[Condition, ...] = (
    "C0", "C1", "C2", "C3_fake_first", "C3_true_first", "C4", "C2_para",
)

Mode = Literal["strict", "truth_seeking"]
MODES: tuple[Mode, ...] = ("strict", "truth_seeking")

Span = Literal["instruction", "evidence", "question"]


# --------------------------------------------------------------------------
# Instruction text
# --------------------------------------------------------------------------
# The two modes are the conceptual core of the design: they separate
# instruction-following from context reliance. The same CCR number means
# opposite things under the two modes, which is why no metric may ever be
# reported without its mode (see docs/PLAN.md 4.2).

_SYSTEM: dict[Mode, dict[Lang, str]] = {
    "strict": {
        "en": (
            "Answer using ONLY the provided document, even if it contradicts what "
            "you previously knew. Reply with the entity name only."
        ),
        "vi": (
            "Chỉ trả lời dựa trên tài liệu được cung cấp, kể cả khi tài liệu đó trái "
            "với những gì bạn đã biết trước đây. Chỉ trả lời bằng tên thực thể."
        ),
    },
    "truth_seeking": {
        "en": (
            "Use the document as a reference, but prioritise giving the factually "
            "correct answer. If the document appears to be incorrect, say so. Reply "
            "with the entity name only, optionally followed by a brief warning."
        ),
        "vi": (
            "Hãy dùng tài liệu như một nguồn tham khảo, nhưng ưu tiên đưa ra câu trả "
            "lời chính xác. Nếu tài liệu có vẻ sai, hãy nói rõ điều đó. Chỉ trả lời "
            "bằng tên thực thể, có thể kèm một cảnh báo ngắn."
        ),
    },
}

# Used for C0, where there is no document to refer to.
_SYSTEM_NO_CONTEXT: dict[Lang, str] = {
    "en": "Answer the question. Reply with the entity name only.",
    "vi": "Hãy trả lời câu hỏi. Chỉ trả lời bằng tên thực thể.",
}

_DOC_LABEL: dict[Lang, str] = {"en": "Document", "vi": "Tài liệu"}
_QUESTION_LABEL: dict[Lang, str] = {"en": "Question", "vi": "Câu hỏi"}


# --------------------------------------------------------------------------
# Chat template (Qwen2.5 / ChatML)
# --------------------------------------------------------------------------

def _chatml(system: str, user: str) -> str:
    """Render a single-turn ChatML prompt ending at the assistant turn.

    The prompt deliberately ends with a newline after the assistant header so
    that a continuation such as "Paris" tokenises cleanly on its own rather
    than merging with preceding characters. See backends.base.score for the
    boundary assertion that enforces this.
    """
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# --------------------------------------------------------------------------
# Context assembly
# --------------------------------------------------------------------------

def documents_for(fact: Fact, condition: Condition, lang: Lang) -> list[str]:
    """The document list a condition puts in front of the model."""
    match condition:
        case "C0":
            return []
        case "C1":
            return [fact.evidence_true[lang]]
        case "C2":
            return [fact.evidence_fake[lang]]
        case "C2_para":
            para = fact.evidence_fake_para.get(lang)
            if not para:
                raise ValueError(
                    f"{fact.fact_id}: condition C2_para needs evidence_fake_para[{lang!r}]; "
                    f"paraphrases must be human-written (PLAN 6.4)"
                )
            return [para]
        case "C3_fake_first":
            return [fact.evidence_fake[lang], fact.evidence_true[lang]]
        case "C3_true_first":
            return [fact.evidence_true[lang], fact.evidence_fake[lang]]
        case "C4":
            return [fact.evidence_irrelevant[lang]]
        case _:
            raise ValueError(f"unknown condition {condition!r}")


def dose_documents(fact: Fact, lang: Lang, dose: int) -> list[str]:
    """Evidence pressure ladder for the dose-response analysis (PLAN 6.5).

    dose 0 no evidence
         1 one hedged mention
         2 one assertive mention
         3 two assertive mentions
         4 three assertive mentions, bracketing the document

    The metric of interest is the *override threshold*: the smallest dose at
    which P_ctx crosses 0.5. This data is MANDATORY to collect even if the
    conference paper does not analyse it - it is the raw material for the
    journal metric contribution, and re-running later loses comparability with
    the conference numbers.
    """
    if dose == 0:
        return []
    claim = fact.evidence_fake[lang]
    hedge = {
        "en": "According to some sources, ",
        "vi": "Theo một số nguồn, ",
    }[lang]
    filler = fact.evidence_irrelevant[lang]

    if dose == 1:
        return [hedge + claim[0].lower() + claim[1:]]
    if dose == 2:
        return [claim]
    if dose == 3:
        return [claim, filler, claim]
    if dose == 4:
        return [claim, filler, claim, filler, claim]
    raise ValueError(f"dose must be 0..4, got {dose}")


def build_prompt(
    fact: Fact,
    *,
    lang: Lang,
    mode: Mode,
    question_kind: Literal["q_eval", "q_filter"] = "q_eval",
    condition: Condition | None = None,
    documents: Sequence[str] | None = None,
    drop_spans: Sequence[Span] = (),
) -> str:
    """Render one prompt.

    Supply either `condition` (normal experimental grid) or `documents`
    directly (dose-response and ablation variants).

    `drop_spans` removes a span for causal attribution. Note that dropping a
    span is only interpretable against a length-matched control - see
    metrics.attribution.
    """
    if (condition is None) == (documents is None):
        raise ValueError("pass exactly one of `condition` or `documents`")

    docs = list(documents) if documents is not None else documents_for(fact, condition, lang)  # type: ignore[arg-type]
    question = (fact.q_eval if question_kind == "q_eval" else fact.q_filter)[lang]

    if "evidence" in drop_spans:
        docs = []
    if "question" in drop_spans:
        question = ""

    if "instruction" in drop_spans:
        system = ""
    elif not docs:
        system = _SYSTEM_NO_CONTEXT[lang]
    else:
        system = _SYSTEM[mode][lang]

    parts: list[str] = []
    if docs:
        label = _DOC_LABEL[lang]
        for i, doc in enumerate(docs, 1):
            tag = f"{label} {i}" if len(docs) > 1 else label
            parts.append(f"{tag}:\n{doc}")
    if question:
        parts.append(f"{_QUESTION_LABEL[lang]}: {question}")

    return _chatml(system, "\n\n".join(parts))


Endorsement = Literal["true", "fake", "none"]


@dataclass(frozen=True, slots=True)
class Candidates:
    """The two answers scored in every condition, plus what the document says.

    Both objects are always scored, including under C0 and C4 where no document
    endorses either. That is deliberate: the resulting preference is the model's
    prior between the two strings with no evidence for either, which is exactly
    the baseline margin the falsification test in metrics.margin_control needs.

    `endorsed` records which object the retrieved document actually asserts.
    Compliance is only defined when it is "true" or "fake" - under "none" there
    is nothing to comply with, and computing CCR there is a category error.
    """
    true: str
    fake: str
    endorsed: Endorsement

    def as_continuations(self) -> list[str]:
        """Order is fixed as (context-endorsed-or-fake, true) so that stored
        records keep a stable column meaning across conditions."""
        return [self.fake, self.true]

    @property
    def context_answer(self) -> str | None:
        """What following the document would mean here, or None if undefined."""
        if self.endorsed == "true":
            return self.true
        if self.endorsed == "fake":
            return self.fake
        return None

    def reliance_pair(self) -> tuple[str, str]:
        """(a_ctx, a_par) for the reliance score.

        Under "none" the first element is the counterfactual object, so R keeps
        a consistent orientation across conditions and can be used as a margin.
        """
        return (self.context_answer or self.fake, self.true)


_ENDORSEMENT: dict[str, Endorsement] = {
    "C0": "none",
    "C1": "true",
    "C2": "fake",
    "C2_para": "fake",
    "C3_fake_first": "fake",   # both docs present; the conflict is inter-document
    "C3_true_first": "fake",
    "C4": "none",
}


def candidates_for(fact: Fact, condition: Condition, lang: Lang) -> Candidates:
    """The scored answers and what the document endorses in this condition."""
    if condition not in _ENDORSEMENT:
        raise ValueError(f"unknown condition {condition!r}")
    return Candidates(
        true=fact.object_true[lang],
        fake=fact.object_fake[lang],
        endorsed=_ENDORSEMENT[condition],
    )


def continuation(answer: str) -> str:
    """Scored continuations are prefixed with nothing: the ChatML prompt already
    ends with a newline, so the answer begins a fresh token sequence."""
    return answer
