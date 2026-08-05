"""Confirm the KV-cache reuse in scoring changes speed and nothing else.

    python scripts/verify_kv_reuse.py

Scoring evaluates the prompt once and rewinds between candidates instead of
re-processing it per candidate. That is worth roughly a factor of two, but it
touches the cache directly, so it needs checking rather than trusting: this runs
both paths on the same prompts and compares log-probabilities.

Anything above ~1e-4 means the cached path is not equivalent. Set
QUANTRAG_NO_KV_REUSE=1 and re-run the grid if so.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]


def build(model_id: str, precision: str, reuse: bool):
    if reuse:
        os.environ.pop("QUANTRAG_NO_KV_REUSE", None)
    else:
        os.environ["QUANTRAG_NO_KV_REUSE"] = "1"
    # Reimport so the flag is read fresh.
    for mod in [m for m in list(sys.modules) if m.startswith("quantrag.backends")]:
        del sys.modules[mod]
    from quantrag.backends import load_backend  # noqa: PLC0415

    cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    model = next(m for m in cfg["models"] if m["id"] == model_id)
    var = next(v for v in model["variants"] if v["precision"] == precision)
    return load_backend({**var, "model_id": model_id,
                         **{k: cfg["runtime"][k]
                            for k in ("n_ctx", "seed", "n_gpu_layers")
                            if k in cfg["runtime"]}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-0.5b")
    ap.add_argument("--precision", default="F16")
    ap.add_argument("--facts", default="data/facts.sample.jsonl")
    args = ap.parse_args()

    from quantrag.prompts import build_prompt, candidates_for  # noqa: PLC0415
    from quantrag.schema import read_facts  # noqa: PLC0415

    facts = read_facts(ROOT / args.facts)
    cases = [(build_prompt(f, lang=lg, mode="strict", condition="C2"),
              candidates_for(f, "C2", lg).as_continuations())
             for f in facts for lg in ("en", "vi")]

    out = {}
    for reuse in (False, True):
        backend = build(args.model, args.precision, reuse)
        t0 = time.perf_counter()
        vals = [[r.mean_logprob for r in backend.score(p, c)] for p, c in cases]
        out[reuse] = (vals, time.perf_counter() - t0)
        del backend

    (plain, t_plain), (cached, t_cached) = out[False], out[True]
    worst = max(abs(a - b)
                for ra, rb in zip(plain, cached)
                for a, b in zip(ra, rb))

    print(f"cases            : {len(cases)}")
    print(f"without reuse    : {t_plain:.2f}s")
    print(f"with reuse       : {t_cached:.2f}s  ({t_plain / max(t_cached, 1e-9):.2f}x)")
    print(f"worst difference : {worst:.3e}")
    print()
    if worst < 1e-4:
        print("Equivalent. The reuse is a speed change only.")
    else:
        print("NOT equivalent. Export QUANTRAG_NO_KV_REUSE=1 before running the grid.")
        sys.exit(1)


if __name__ == "__main__":
    main()
