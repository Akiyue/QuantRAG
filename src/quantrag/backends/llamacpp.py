"""Tier A backend: llama.cpp / GGUF.

This is the primary stack. F16, Q8_0, Q4_K_M and Q3_K_M all load through the
same runtime with the same tokeniser and the same sampler, so the only variable
across arms is the bit width. That is what makes the precision ladder a
controlled comparison rather than a comparison of two inference stacks.

Note on terminology: Q4_K_M and Q3_K_M are mixed k-quants, not uniform INT4 or
INT3. The effective average bit width differs from the nominal number and some
tensors are kept at higher precision. Report this precisely in the paper.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Sequence

from .base import BoundaryError, GenResult, ScoreResult, split_boundary


class LlamaCppBackend:
    def __init__(
        self,
        path: str,
        precision: str,
        model_id: str = "",
        n_ctx: int = 2048,
        seed: int = 1234,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        verbose: bool = False,
        **_: object,
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "llama-cpp-python is required for the llamacpp backend.\n"
                '  CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python '
                "--no-binary llama-cpp-python"
            ) from exc

        self.path = str(path)
        if not Path(self.path).exists():
            raise FileNotFoundError(f"GGUF not found: {self.path}")
        self.precision = precision
        self.model_id = model_id or Path(self.path).stem
        self.seed = seed

        self._llm = Llama(
            model_path=self.path,
            n_ctx=n_ctx,
            seed=seed,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            logits_all=True,   # required for echoed log-probabilities
            verbose=verbose,
        )

    # -- scoring ---------------------------------------------------------

    def score(self, prompt: str, continuations: Sequence[str]) -> list[ScoreResult]:
        """Log-probability of each continuation, teacher-forced.

        Implemented by echoing prompt+continuation with max_tokens=0 and reading
        back per-token log-probabilities, then slicing off the prompt using
        character offsets. The offset-based split is what lets us assert the
        answer starts on a real token boundary instead of hoping it does.
        """
        results: list[ScoreResult] = []
        prompt_len = len(prompt)

        for cont in continuations:
            out = self._llm.create_completion(
                prompt=prompt + cont,
                max_tokens=0,
                echo=True,
                logprobs=0,
                temperature=0.0,
            )
            lp = out["choices"][0]["logprobs"]
            offsets: list[int] = lp["text_offset"]
            token_lps: list[float | None] = lp["token_logprobs"]
            tokens: list[str] = lp["tokens"]

            start = split_boundary(offsets, prompt_len)
            sel_lps = token_lps[start:]
            sel_toks = tokens[start:]

            # The very first token of a sequence has no conditional probability;
            # that can only happen here if the prompt were empty, which it never is.
            if any(x is None for x in sel_lps):
                raise BoundaryError(
                    f"missing log-probabilities in continuation {cont!r}"
                )
            vals = [float(x) for x in sel_lps]  # type: ignore[arg-type]
            if not vals:
                raise BoundaryError(f"continuation {cont!r} tokenised to nothing")

            results.append(
                ScoreResult(
                    text=cont,
                    sum_logprob=math.fsum(vals),
                    n_tokens=len(vals),
                    token_logprobs=vals,
                    tokens=sel_toks,
                )
            )
        return results

    # -- generation ------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int = 32) -> GenResult:
        out = self._llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,      # greedy; QFR must not measure sampling noise
            top_p=1.0,
            top_k=1,
            stop=["<|im_end|>", "<|im_start|>", "\n\n"],
        )
        choice = out["choices"][0]
        return GenResult(
            text=choice["text"].strip(),
            n_tokens=int(out.get("usage", {}).get("completion_tokens", 0)),
            finish_reason=str(choice.get("finish_reason", "")),
        )

    # -- provenance ------------------------------------------------------

    def env(self) -> dict:
        import llama_cpp

        return {
            "backend": "llamacpp",
            "model_id": self.model_id,
            "precision": self.precision,
            "gguf_path": self.path,
            "gguf_sha256": _sha256(self.path),
            "llama_cpp_python": getattr(llama_cpp, "__version__", "unknown"),
            "seed": self.seed,
        }


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()
