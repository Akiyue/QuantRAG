"""Confirm the KV-cache reuse in scoring changes speed and nothing that matters.

    python scripts/verify_kv_reuse.py

Scoring evaluates the prompt once and rewinds between candidates instead of
re-processing it per candidate. That touches the cache directly, so it needs
checking rather than trusting.

The two paths will not agree bit for bit and are not expected to. Evaluating a
150-token batch and a 152-token batch splits the matrix work differently, and
floating-point addition is not associative, so the reductions land in a slightly
different place. That is the same class of wobble as running the same
configuration twice.

What matters is therefore not the raw delta but whether it changes a decision:

  * does the preferred answer ever flip?
  * does the reliance score R move enough to matter against the effects the
    paper reports, which are of order 0.1 to 1.0?

A difference of 1e-3 in log-probability with no argmax flips is numerical dust.
The same difference flipping answers would mean the measurement is not stable
enough to report at all - and that would be true of the uncached path too.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]

# R is a difference of two mean log-probabilities and the reported shifts are of
# order 0.1-1.0. A tenth of the smallest interesting effect is generous.
R_TOLERANCE = 0.01


def build(model_id: str, precision: str, reuse: bool):
    if reuse:
        os.environ.pop("QUANTRAG_NO_KV_REUSE", None)
    else:
        os.environ["QUANTRAG_NO_KV_REUSE"] = "1"
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
    ap.add_argument("--conditions", nargs="*", default=["C0", "C1", "C2"])
    args = ap.parse_args()

    from quantrag.prompts import build_prompt, candidates_for  # noqa: PLC0415
    from quantrag.schema import read_facts  # noqa: PLC0415

    facts = read_facts(ROOT / args.facts)
    cases = [(build_prompt(f, lang=lg, mode=md, condition=cond),
              candidates_for(f, cond, lg).as_continuations())
             for f in facts for lg in ("en", "vi")
             for cond in args.conditions for md in ("strict", "truth_seeking")]

    out = {}
    for reuse in (False, True):
        backend = build(args.model, args.precision, reuse)
        t0 = time.perf_counter()
        vals = [[r.mean_logprob for r in backend.score(p, c)] for p, c in cases]
        out[reuse] = (vals, time.perf_counter() - t0)
        del backend

    (plain, t_plain), (cached, t_cached) = out[False], out[True]

    lp_deltas, r_deltas, flips = [], [], 0
    for ra, rb in zip(plain, cached):
        lp_deltas += [abs(a - b) for a, b in zip(ra, rb)]
        # R is fake-minus-true, the quantity every downstream metric is built on.
        r_a, r_b = ra[0] - ra[1], rb[0] - rb[1]
        r_deltas.append(abs(r_a - r_b))
        flips += (r_a > 0) != (r_b > 0)

    print(f"cases               : {len(cases)}")
    print(f"without reuse       : {t_plain:.2f}s")
    print(f"with reuse          : {t_cached:.2f}s  "
          f"({t_plain / max(t_cached, 1e-9):.2f}x)")
    print()
    print(f"worst |Δ log-prob|  : {max(lp_deltas):.3e}")
    print(f"worst |Δ R|         : {max(r_deltas):.3e}   (tolerance {R_TOLERANCE})")
    print(f"median |Δ R|        : {statistics.median(r_deltas):.3e}")
    print(f"preference flips    : {flips}/{len(cases)}")
    print()

    if flips == 0 and max(r_deltas) < R_TOLERANCE:
        print("Equivalent where it counts: no preferred answer changed, and the")
        print("reliance score moved far less than the effects being measured.")
        print("Keep the reuse on.")
        print()
        print(f"Note the size though: |ΔR| up to {max(r_deltas):.1e} is a floor on")
        print("numerical wobble in this stack. The pilot's two-run check should")
        print("come out at least this large; if it comes out much larger, that is")
        print("the run-to-run noise the flip rate must clear.")
        return

    if flips:
        print(f"{flips} preferred answers changed between the two paths.")
        print("That is not acceptable for either path: if the argmax moves under")
        print("a pure batching change, the measurement is too unstable to report.")
        print("Investigate before running the grid - start with a larger n_batch")
        print("or a single-threaded build.")
    else:
        print(f"No flips, but |ΔR| reached {max(r_deltas):.3e}, above {R_TOLERANCE}.")
        print("Run with QUANTRAG_NO_KV_REUSE=1 and report the numerical floor in")
        print("the paper.")
    sys.exit(1)


if __name__ == "__main__":
    main()
