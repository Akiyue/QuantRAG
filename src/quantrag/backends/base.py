"""Backend interface.

Two rules the whole study rests on:

1. Every precision level goes through one runtime. The only thing that differs
   between arms is the number of bits.
2. Scoring is teacher-forced. We do not ask the model to generate and then check
   what it said; we ask how much probability it assigns to each candidate answer.
   That is a continuous signal, independent of decoding quirks, and it is the
   quantity the reliance score is defined on.

Free generation still runs, but only to catch refusals, hallucinations and
answers outside the two candidates - things scoring cannot see.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable


class DegenerateOutput(RuntimeError):
    """The model produced numerical garbage rather than an answer.

    llama.cpp emits a run of '!' when the logits come back NaN - a GPU fault, a
    corrupted weight file, memory pressure. It looks like an answer and it is
    not one.

    This has to be an error rather than a label. Recorded as OTHER it would be
    indistinguishable from a genuine wrong answer, sit in the denominator of
    every rate, and quietly become a finding: an arm that glitched would show
    lower compliance than one that did not.
    """


class BoundaryError(RuntimeError):
    """Raised when a continuation does not begin on a token boundary.

    If the tokeniser merges the last character of the prompt with the first
    character of the answer, the log-probabilities we attribute to the answer
    include part of the prompt, and lengths stop being comparable across
    languages. Silently averaging over that would quietly corrupt every
    downstream number, so it is a hard error.
    """


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """Teacher-forced score of one continuation."""
    text: str
    sum_logprob: float
    n_tokens: int
    token_logprobs: list[float] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)

    @property
    def mean_logprob(self) -> float:
        """Length-normalised score.

        Mandatory for the bilingual arm: 'London' may be one token while
        'Luân Đôn' is three or four. Comparing raw sums across languages
        measures the tokeniser, not the model. The raw sum is kept as well so
        the appendix can confirm conclusions do not flip.
        """
        return self.sum_logprob / self.n_tokens if self.n_tokens else float("-inf")

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "sum_logprob": self.sum_logprob,
            "mean_logprob": self.mean_logprob,
            "n_tokens": self.n_tokens,
            "token_logprobs": self.token_logprobs,
        }


@dataclass(frozen=True, slots=True)
class GenResult:
    text: str
    n_tokens: int = 0
    finish_reason: str = ""

    def to_dict(self) -> dict:
        return {"text": self.text, "n_tokens": self.n_tokens,
                "finish_reason": self.finish_reason}


@runtime_checkable
class Backend(Protocol):
    """What the runner is allowed to assume about a model.

    Deliberately narrow. Analysis code must never branch on which backend is
    running - the moment it does, tier A and tier B stop being comparable.
    """

    model_id: str
    precision: str

    def score(self, prompt: str, continuations: Sequence[str]) -> list[ScoreResult]: ...

    def generate(self, prompt: str, max_tokens: int = 32) -> GenResult: ...

    def env(self) -> dict:
        """Environment fingerprint recorded with every run for reproducibility."""
        ...


def check_degenerate(text: str, min_run: int = 8) -> None:
    """Reject a generation that is one character repeated.

    Narrow on purpose: a real answer is never eight identical non-space
    characters in a row, and anything looser would start rejecting answers.
    """
    stripped = text.strip()
    if len(stripped) >= min_run and len(set(stripped)) == 1 and not stripped[0].isalnum():
        raise DegenerateOutput(
            f"model emitted {stripped[0]!r} x{len(stripped)} - NaN logits, not an "
            f"answer. Check the GGUF checksum and GPU health for this arm."
        )


def split_boundary(text_offsets: Sequence[int], prompt_len: int) -> int:
    """Index of the first token belonging to the continuation.

    `text_offsets` are character offsets of each token within prompt+continuation.
    A clean split requires some token to start exactly at `prompt_len`; if none
    does, the tokeniser merged across the boundary.
    """
    for i, off in enumerate(text_offsets):
        if off == prompt_len:
            return i
        if off > prompt_len:
            raise BoundaryError(
                f"continuation does not start on a token boundary: a token spans "
                f"prompt character {prompt_len} (next offset {off}). Adjust the "
                f"prompt template so it ends on a clean break."
            )
    raise BoundaryError("continuation produced no tokens")


class MockBackend:
    """Deterministic stand-in used to develop and test the pipeline offline.

    Scores are a stable hash of (prompt, continuation), so the runner, the
    metrics and the resume logic can all be exercised on a laptop with no model
    weights present. It is not a model and must never appear in results.
    """

    def __init__(self, model_id: str = "mock", precision: str = "MOCK", **_: object) -> None:
        self.model_id = model_id
        self.precision = precision

    @staticmethod
    def _pseudo_logprob(prompt: str, token: str, salt: str) -> float:
        h = hashlib.sha256(f"{salt}|{prompt}|{token}".encode()).digest()
        u = int.from_bytes(h[:8], "big") / 2**64  # uniform in [0, 1)
        return -0.05 - 3.0 * u  # plausible per-token log-probability

    def score(self, prompt: str, continuations: Sequence[str]) -> list[ScoreResult]:
        out: list[ScoreResult] = []
        for cont in continuations:
            toks = cont.split() or [cont]
            lps = [self._pseudo_logprob(prompt, t, self.precision) for t in toks]
            out.append(ScoreResult(cont, math.fsum(lps), len(toks), lps, toks))
        return out

    def generate(self, prompt: str, max_tokens: int = 32) -> GenResult:
        h = hashlib.sha256(f"{self.precision}|{prompt}".encode()).hexdigest()[:8]
        return GenResult(text=f"mock-{h}", n_tokens=1, finish_reason="stop")

    def env(self) -> dict:
        return {"backend": "mock", "model_id": self.model_id, "precision": self.precision}
