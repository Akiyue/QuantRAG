"""Show exactly how one prompt and its candidate answers tokenise.

    python scripts/debug_scoring.py

Run this if scoring reports boundary errors. It prints the token ids either side
of the prompt/answer split and checks the prefix property directly, so the
failure is visible rather than inferred from an exception.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantrag.backends import load_backend  # noqa: E402
from quantrag.prompts import build_prompt, candidates_for  # noqa: E402
from quantrag.schema import read_facts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-0.5b")
    ap.add_argument("--precision", default="F16")
    ap.add_argument("--facts", default="data/facts.sample.jsonl")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    model = next(m for m in cfg["models"] if m["id"] == args.model)
    var = next(v for v in model["variants"] if v["precision"] == args.precision)
    backend = load_backend({**var, "model_id": model["id"],
                            **{k: cfg["runtime"][k]
                               for k in ("n_ctx", "seed", "n_gpu_layers")
                               if k in cfg["runtime"]}})

    fact = read_facts(ROOT / args.facts)[0]

    for lang in ("en", "vi"):
        prompt = build_prompt(fact, lang=lang, mode="strict", condition="C2")
        cands = candidates_for(fact, "C2", lang)
        p_ids = backend._tokenize(prompt, add_bos=True)

        print("=" * 68)
        print(f"{lang}  prompt: {len(prompt)} chars, "
              f"{len(prompt.encode('utf-8'))} bytes, {len(p_ids)} tokens")
        tail = [(t, backend._llm.detokenize([t]).decode('utf-8', errors='replace'))
                for t in p_ids[-4:]]
        print(f"  last prompt tokens: {tail}")

        for cont in cands.as_continuations():
            f_ids = backend._tokenize(prompt + cont, add_bos=True)
            prefix_ok = f_ids[:len(p_ids)] == p_ids
            new = f_ids[len(p_ids):] if prefix_ok else []
            mark = "OK  " if prefix_ok else "BAD "
            print(f"  {mark} {cont!r:<16} -> +{len(f_ids) - len(p_ids)} tokens "
                  f"{[(t, backend._llm.detokenize([t]).decode('utf-8', errors='replace')) for t in new]}")
            if not prefix_ok:
                for i, (a, b) in enumerate(zip(p_ids, f_ids)):
                    if a != b:
                        print(f"       diverges at token {i}: prompt {a} vs full {b}")
                        break

        try:
            res = backend.score(prompt, cands.as_continuations())
            for r in res:
                print(f"  scored {r.text!r:<16} sum={r.sum_logprob:8.3f} "
                      f"n={r.n_tokens} mean={r.mean_logprob:7.3f}")
        except Exception as exc:  # noqa: BLE001 - this is the diagnostic
            print(f"  score() raised: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
