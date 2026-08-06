"""Measure the run-to-run noise floor from two identical pilot runs.

    python scripts/check_noise.py runs/pilot_a runs/pilot_b

Greedy decoding is deterministic in principle; GPU kernels are not always
deterministic in practice. If two identical configurations disagree on a
meaningful fraction of items, then a quantization flip rate of the same size is
measuring the hardware rather than the quantization, and QFR cannot be reported
as a finding.

Run this at the pilot stage. Discovering it after the full grid is expensive;
discovering it after the paper is written is worse.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantrag.analysis import index, load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--pass-name", default="main")
    ap.add_argument("--threshold", type=float, default=0.02,
                    help="disagreement above this makes QFR uninterpretable")
    args = ap.parse_args()

    a = load(ROOT / args.run_a, args.pass_name)
    b = load(ROOT / args.run_b, args.pass_name)
    if not a or not b:
        sys.exit("one of the runs is empty")

    ga, gb = index(a, "generate"), index(b, "generate")
    sa, sb = index(a, "score"), index(b, "score")

    disagree = total = 0
    for key, per_prec in ga.items():
        for prec, ra in per_prec.items():
            rb = gb.get(key, {}).get(prec)
            if rb is None or ra.label is None or rb.label is None:
                continue
            total += 1
            disagree += ra.label is not rb.label

    deltas, nans = [], 0
    for key, per_prec in sa.items():
        for prec, ra in per_prec.items():
            rb = sb.get(key, {}).get(prec)
            if rb is None or ra.r is None or rb.r is None:
                continue
            d = abs(ra.r - rb.r)
            # A nan means a score was degenerate, which is its own problem;
            # letting it poison the mean hides both.
            if d != d:
                nans += 1
            else:
                deltas.append(d)

    rate = disagree / total if total else float("nan")
    print(f"label disagreement : {disagree}/{total} = {rate:.4f}")
    if deltas:
        print(f"|ΔR| median        : {statistics.median(deltas):.6f}")
        print(f"|ΔR| max           : {max(deltas):.6f}")
        print(f"|ΔR| exactly zero  : {sum(d == 0 for d in deltas)}/{len(deltas)}")
    if nans:
        print(f"|ΔR| nan           : {nans}  <- degenerate scores, investigate")

    print()
    if total and rate == 0:
        print("Fully deterministic. Any flip you observe is attributable to")
        print("quantization, not to the runtime.")
    elif total and rate <= args.threshold:
        print(f"Noise floor is {rate:.4f}. Report it in the paper and only treat")
        print("flip rates comfortably above it as findings.")
    else:
        print(f"Disagreement is {rate:.4f}, above the {args.threshold} threshold.")
        print("QFR is not interpretable at this noise level. Fix determinism")
        print("(pin kernels, batch size, thread count) before running the grid.")
        sys.exit(1)


if __name__ == "__main__":
    main()
