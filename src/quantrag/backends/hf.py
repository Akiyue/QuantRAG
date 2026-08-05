"""Tier B backend: HuggingFace transformers, used for AWQ.

Exists for one purpose - showing that whatever the precision ladder reveals is
not an artefact of k-quantization specifically. Results from here go in their
own subsection and are never mixed into the tier A tables: this is a different
runtime with a different kernel path, so a difference between tiers is not
attributable to bit width alone.

The prompt string is byte-identical to tier A because both backends receive it
already rendered from prompts.py. If that ever stops being true the comparison
is worthless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .base import BoundaryError, GenResult, ScoreResult


class HFBackend:
    def __init__(
        self,
        path: str,
        precision: str,
        model_id: str = "",
        seed: int = 1234,
        device: str = "cuda",
        dtype: str = "float16",
        **_: object,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                'the tier B backend needs: uv pip install -e ".[hf]"'
            ) from exc

        self.path = str(path)
        self.precision = precision
        self.model_id = model_id or Path(self.path).name
        self.seed = seed
        self.device = device
        self._torch = torch

        torch.manual_seed(seed)
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            torch_dtype=getattr(torch, dtype),
            device_map=device,
        )
        self.model.eval()

    # -- scoring ---------------------------------------------------------

    def score(self, prompt: str, continuations: Sequence[str]) -> list[ScoreResult]:
        """Teacher-forced log-probabilities, sliced on a real token boundary.

        The prompt is tokenised on its own and the continuation is required to
        extend it exactly. If the tokeniser merges the last prompt character
        with the first answer character, the answer's score would silently
        absorb part of the prompt and lengths would stop being comparable
        across languages - so that is an error, not a warning.
        """
        torch = self._torch
        prompt_ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False)
        n_prompt = prompt_ids["input_ids"].shape[1]

        results: list[ScoreResult] = []
        for cont in continuations:
            full = self.tok(prompt + cont, return_tensors="pt",
                            add_special_tokens=False)
            ids = full["input_ids"]
            if ids.shape[1] <= n_prompt:
                raise BoundaryError(f"continuation {cont!r} tokenised to nothing")
            if not torch.equal(ids[0, :n_prompt], prompt_ids["input_ids"][0]):
                raise BoundaryError(
                    f"continuation {cont!r} does not start on a token boundary: "
                    "the tokeniser merged across the prompt/answer split"
                )

            ids = ids.to(self.model.device)
            with torch.no_grad():
                logits = self.model(ids).logits
            logprobs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
            targets = ids[0, 1:]
            picked = logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)

            # logprobs[i] predicts token i+1, so continuation token j (absolute
            # index n_prompt + j) is at picked[n_prompt + j - 1].
            sel = picked[n_prompt - 1:]
            toks = self.tok.convert_ids_to_tokens(targets[n_prompt - 1:].tolist())
            vals = [float(x) for x in sel.tolist()]
            results.append(ScoreResult(cont, sum(vals), len(vals), vals, toks))
        return results

    # -- generation ------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int = 32) -> GenResult:
        torch = self._torch
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False)
        ids = {k: v.to(self.model.device) for k, v in ids.items()}
        with torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=max_tokens,
                do_sample=False,          # greedy, to match tier A
                pad_token_id=self.tok.eos_token_id,
            )
        new = out[0, ids["input_ids"].shape[1]:]
        text = self.tok.decode(new, skip_special_tokens=True)
        return GenResult(text=text.strip(), n_tokens=int(new.shape[0]),
                         finish_reason="stop")

    # -- provenance ------------------------------------------------------

    def env(self) -> dict:
        import torch
        import transformers

        return {
            "backend": "hf_awq",
            "model_id": self.model_id,
            "precision": self.precision,
            "path": self.path,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "device_name": (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else "cpu"),
            "seed": self.seed,
        }
