"""How much does a measurement move when nothing has changed?

    python scripts/numerical_floor.py --model qwen2.5-0.5b

Runs the identical scoring twice per precision and reports how far the results
drift. Any difference is pure numerical wobble - kernel scheduling, reduction
order - and has nothing to do with the experiment.

This number is the resolution of every flip rate the paper reports. If the
stack moves answers on its own, the quantization flip rate is partly measuring
the runtime, and a reader is entitled to know by how much. A floor of exactly
zero is the strongest version of that statement and belongs in the
reproducibility section.

Worth running per precision rather than once: the F16 and the k-quant kernels
are different code, so determinism in one does not imply it in the others.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]

GREEN, YELLOW, RED, OFF = "\033[32m", "\033[33m", "\033[31m", "\033[0m"


def reliance(backend, cases) -> list[float]:
    out = []
    for prompt, conts in cases:
        s = backend.score(prompt, conts)
        out.append(s[0].mean_logprob - s[1].mean_logprob)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-0.5b")
    ap.add_argument("--precisions", nargs="*", default=None,
                    help="default: every tier A precision for the model")
    ap.add_argument("--facts", default="data/facts.sample.jsonl")
    args = ap.parse_args()

    from quantrag.backends import load_backend  # noqa: PLC0415
    from quantrag.prompts import build_prompt, candidates_for  # noqa: PLC0415
    from quantrag.schema import read_facts  # noqa: PLC0415

    cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    model = next(m for m in cfg["models"] if m["id"] == args.model)
    variants = [v for v in model["variants"] if v.get("tier", "A") == "A"
                and (not args.precisions or v["precision"] in args.precisions)]

    facts = read_facts(ROOT / args.facts)
    cases = [(build_prompt(f, lang=lg, mode=md, condition=c),
              candidates_for(f, c, lg).as_continuations())
             for f in facts for lg in ("en", "vi")
             for c in ("C0", "C1", "C2") for md in ("strict", "truth_seeking")]
    print(f"{args.model}, {len(cases)} cases per precision\n")

    worst_overall = 0.0
    rows = []
    for var in variants:
        if not (ROOT / var["path"]).exists():
            print(f"  {var['precision']:<8} (missing, skipped)")
            continue
        backend = load_backend({**var, "model_id": model["id"],
                                **{k: cfg["runtime"][k]
                                   for k in ("n_ctx", "seed", "n_gpu_layers")
                                   if k in cfg["runtime"]}})
        a, b = reliance(backend, cases), reliance(backend, cases)
        del backend

        d = [abs(x - y) for x, y in zip(a, b)]
        flips = sum((x > 0) != (y > 0) for x, y in zip(a, b))
        worst = max(d)
        worst_overall = max(worst_overall, worst)
        rows.append((var["precision"], sum(1 for x in d if x == 0.0), len(d),
                     statistics.median(d), worst, flips))

    print(f"  {'precision':<10} {'identical':>12} {'median |ΔR|':>14} "
          f"{'worst |ΔR|':>13} {'flips':>7}")
    for prec, ident, n, med, worst, flips in rows:
        print(f"  {prec:<10} {f'{ident}/{n}':>12} {med:>14.3e} "
              f"{worst:>13.3e} {flips:>7}")

    print()
    if worst_overall == 0.0:
        print(f"{GREEN}Bit-identical across runs at every precision.{OFF}")
        print("Any flip observed between precisions is attributable to")
        print("quantization, not to the runtime. State this in the paper:")
        print()
        print("  \"Repeated evaluation of an identical configuration produced")
        print("   bit-identical log-probabilities, so the reported instance-level")
        print("   changes are not attributable to run-to-run variation.\"")
    elif any(r[5] for r in rows):
        print(f"{RED}Answers changed between identical runs.{OFF}")
        print("The flip rate cannot be reported as a quantization effect until")
        print("this is fixed. Try pinning n_threads and n_batch, or a")
        print("single-GPU, single-stream configuration.")
        sys.exit(1)
    else:
        print(f"{YELLOW}Not bit-identical, but no answer changed.{OFF}")
        print(f"Report {worst_overall:.2e} as the numerical resolution, and only")
        print("treat flip rates comfortably above it as findings.")

    print()
    print("Next: ./run.sh pilot measures the same thing end to end, including")
    print("generation and the evaluator.")


if __name__ == "__main__":
    main()
