"""How much does the same measurement move when nothing has changed?

    python scripts/numerical_floor.py

Two questions, and the second only matters because of the first:

  1. Run the identical scoring twice. Any difference is pure numerical wobble -
     kernel scheduling, reduction order, nothing to do with the experiment.
  2. Run it once with the prompt cache reused between candidates. Compare that
     difference against the floor from (1).

If the two are the same size, the cache reuse is not introducing anything the
stack does not already do to itself, and the interesting number is the floor.
If the reuse difference is much larger, it is adding noise and belongs off.

Either way the floor from (1) is a number the paper needs. Every flip rate has
to clear it, and a reader is entitled to know what it was.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]


def load(model_id: str, precision: str, reuse: bool):
    os.environ["QUANTRAG_KV_REUSE"] = "1" if reuse else ""
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


def reliance(backend, cases) -> list[float]:
    out = []
    for prompt, conts in cases:
        s = backend.score(prompt, conts)
        out.append(s[0].mean_logprob - s[1].mean_logprob)
    return out


def summarise(name: str, a: list[float], b: list[float]) -> float:
    d = [abs(x - y) for x, y in zip(a, b)]
    flips = sum((x > 0) != (y > 0) for x, y in zip(a, b))
    identical = sum(1 for x in d if x == 0.0)
    print(f"{name}")
    print(f"  identical      : {identical}/{len(d)}")
    print(f"  median |ΔR|    : {statistics.median(d):.3e}")
    print(f"  worst  |ΔR|    : {max(d):.3e}")
    print(f"  argmax flips   : {flips}/{len(d)}")
    return max(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-0.5b")
    ap.add_argument("--precision", default="F16")
    ap.add_argument("--facts", default="data/facts.sample.jsonl")
    args = ap.parse_args()

    from quantrag.prompts import build_prompt, candidates_for  # noqa: PLC0415
    from quantrag.schema import read_facts  # noqa: PLC0415

    facts = read_facts(ROOT / args.facts)
    cases = [(build_prompt(f, lang=lg, mode=md, condition=c),
              candidates_for(f, c, lg).as_continuations())
             for f in facts for lg in ("en", "vi")
             for c in ("C0", "C1", "C2") for md in ("strict", "truth_seeking")]
    print(f"{len(cases)} cases, {args.model} {args.precision}\n")

    plain = load(args.model, args.precision, reuse=False)
    run1 = reliance(plain, cases)
    run2 = reliance(plain, cases)
    del plain

    floor = summarise("same code path, run twice", run1, run2)
    print()

    cached = load(args.model, args.precision, reuse=True)
    run3 = reliance(cached, cases)
    del cached
    reuse_gap = summarise("plain vs prompt-cache reuse", run1, run3)

    print()
    print(f"numerical floor      : {floor:.3e}")
    print(f"reuse adds           : {reuse_gap:.3e}")
    print()
    if floor == 0.0 and reuse_gap == 0.0:
        print("Fully deterministic. Any flip you later observe is attributable")
        print("to quantization, and the paper can say so plainly.")
    elif reuse_gap <= max(floor * 2, 1e-6):
        print("The reuse sits inside the noise the stack already has. It is not")
        print("adding instability - but it also measured only ~1.02x, so it stays")
        print("off. Report the floor above as the numerical resolution.")
    else:
        print("The reuse moves results well beyond the intrinsic floor. Keep it")
        print("off (it is the default) and report the floor above.")
    print()
    print("Carry this number to ./run.sh pilot: the two-run label disagreement")
    print("there is the same quantity measured on the whole pipeline, and every")
    print("flip rate you report has to be comfortably larger than it.")


if __name__ == "__main__":
    main()
