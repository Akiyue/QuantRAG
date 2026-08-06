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
import os
from pathlib import Path
from typing import Sequence

from .base import (BoundaryError, DegenerateOutput, GenResult, ScoreResult,
                   check_degenerate)


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
        main_gpu: int = 0,
        split_mode: int = 0,       # LLAMA_SPLIT_MODE_NONE
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
        self.n_ctx = n_ctx
        self.main_gpu = main_gpu
        self.split_mode = split_mode
        # Note for anyone tempted to optimise this: reusing the prompt's KV
        # cache between the two candidates looks like it should halve the work,
        # and it does not. Measured on a realistic mix of conditions it came out
        # at 1.02x - llama.cpp already avoids most of the recomputation - while
        # moving a third of the items by up to 4e-2 in the reliance score. The
        # plain path is bit-identical across runs; that property is worth far
        # more than two percent. Measured 2026-08-05, see git history.

        # One model, one GPU, by default. Every model here fits in a single
        # card, so splitting a 1.5B across two devices buys nothing and adds a
        # variable - tensor-split reductions are another place where results
        # can move between otherwise identical runs, which is exactly what the
        # flip rate must not be measuring.
        # To use both cards, run two processes with CUDA_VISIBLE_DEVICES set
        # rather than splitting one model.
        self._llm = Llama(
            model_path=self.path,
            n_ctx=n_ctx,
            seed=seed,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            main_gpu=main_gpu,
            split_mode=split_mode,
            logits_all=True,   # required for echoed log-probabilities
            verbose=verbose,
        )

    # -- scoring ---------------------------------------------------------

    def _tokenize(self, text: str, add_bos: bool) -> list[int]:
        return self._llm.tokenize(text.encode("utf-8"), add_bos=add_bos, special=True)

    def score(self, prompt: str, continuations: Sequence[str]) -> list[ScoreResult]:
        """Log-probability of each continuation, teacher-forced.

        Works on token ids rather than character offsets. `create_completion`
        does expose per-token log-probabilities with `echo=True`, but its
        `text_offset` values are byte offsets into the detokenised output, so
        comparing them against a Python character length never lines up - and
        for Vietnamese the two differ on almost every token.

        Tokenising the prompt and the prompt+continuation separately and
        requiring the former to be a prefix of the latter gives the same
        guarantee directly: if the tokeniser merged the last prompt character
        with the first answer character, the prefix breaks and we refuse to
        return a number.
        """
        import numpy as np

        prompt_ids = self._tokenize(prompt, add_bos=True)
        n_prompt = len(prompt_ids)

        results: list[ScoreResult] = []
        for cont in continuations:
            full_ids = self._tokenize(prompt + cont, add_bos=True)
            if full_ids[:n_prompt] != prompt_ids:
                raise BoundaryError(
                    f"continuation {cont!r} does not start on a token boundary: "
                    "tokenising prompt+answer does not extend the prompt's own "
                    "tokenisation, so the answer's score would absorb part of "
                    "the prompt"
                )
            target_ids = full_ids[n_prompt:]
            if not target_ids:
                raise BoundaryError(f"continuation {cont!r} tokenised to nothing")
            if len(full_ids) > self.n_ctx:
                raise BoundaryError(
                    f"prompt+answer is {len(full_ids)} tokens, over n_ctx={self.n_ctx}"
                )

            self._llm.reset()
            self._llm.eval(full_ids)
            scores = np.asarray(self._llm.scores, dtype=np.float32)
            if scores.ndim != 2 or scores.shape[0] < len(full_ids):
                raise RuntimeError(
                    "llama.cpp did not return per-position logits. The model must "
                    "be loaded with logits_all=True; run scripts/debug_scoring.py."
                )

            # Row i holds the distribution over token i+1, so the rows that
            # predict the continuation start one before it.
            rows = scores[n_prompt - 1: len(full_ids) - 1]
            if not np.isfinite(rows).all():
                raise DegenerateOutput(
                    f"non-finite logits scoring {cont!r}. The model produced NaN "
                    f"rather than a distribution; check the GGUF checksum and GPU "
                    f"health for {self.model_id} {self.precision}."
                )

            rows = rows - rows.max(axis=-1, keepdims=True)
            logprobs = rows - np.log(np.exp(rows).sum(axis=-1, keepdims=True))
            vals = [float(logprobs[i, t]) for i, t in enumerate(target_ids)]
            if not all(v == v and v != float("-inf") for v in vals):
                raise DegenerateOutput(
                    f"non-finite log-probability for {cont!r} in {self.model_id} "
                    f"{self.precision}"
                )

            toks = [self._llm.detokenize([t]).decode("utf-8", errors="replace")
                    for t in target_ids]
            results.append(
                ScoreResult(
                    text=cont,
                    sum_logprob=math.fsum(vals),
                    n_tokens=len(vals),
                    token_logprobs=vals,
                    tokens=toks,
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
        check_degenerate(choice["text"])
        return GenResult(
            text=choice["text"].strip(),
            n_tokens=int(out.get("usage", {}).get("completion_tokens", 0)),
            finish_reason=str(choice.get("finish_reason", "")),
        )

    # -- provenance ------------------------------------------------------

    def env(self) -> dict:
        import llama_cpp

        import os

        return {
            "backend": "llamacpp",
            "model_id": self.model_id,
            "precision": self.precision,
            "gguf_path": self.path,
            "gguf_sha256": _sha256(self.path),
            "llama_cpp_python": getattr(llama_cpp, "__version__", "unknown"),
            "seed": self.seed,
            "main_gpu": self.main_gpu,
            "split_mode": self.split_mode,
            # Which physical device this arm ran on. On a multi-GPU box two
            # arms can land on different cards, and that belongs in the record
            # rather than being reconstructed later from shell history.
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        }


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()
