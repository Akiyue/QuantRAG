"""Explain where two runs of the same configuration disagree.

    python scripts/diff_runs.py runs/pilot_a runs/pilot_b

check_noise.py answers "how much"; this answers "where", which is what tells
you whether you are looking at numerical noise or at a mistake.

Numerical noise is small, unstructured, and spread evenly. A disagreement that
clusters on one model, one precision, or one stretch of wall-clock time is not
noise - it is two runs that were not actually the same run. The commonest cause
is stale output: the runner resumes by design, so a directory left over from an
earlier invocation carries answers produced by an earlier version of the
prompts or the aliases.
"""

from __future__ import annotations

import argparse
import datetime as dt
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantrag.analysis import index, load  # noqa: E402
from quantrag.schema import iter_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def timestamps(run_dir: Path) -> list[float]:
    out = []
    for p in run_dir.glob("*.jsonl"):
        out += [r["ts"] for r in iter_jsonl(p) if "ts" in r]
    return out


def stamp_report(name: str, run_dir: Path) -> float | None:
    ts = timestamps(run_dir)
    if not ts:
        print(f"  {name}: no timestamps")
        return None
    lo, hi = min(ts), max(ts)
    span = hi - lo
    print(f"  {name}: {len(ts)} records, "
          f"{dt.datetime.fromtimestamp(lo):%Y-%m-%d %H:%M} .. "
          f"{dt.datetime.fromtimestamp(hi):%H:%M}  (span {span / 60:.0f} min)")
    return span


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--pass-name", default="main")
    args = ap.parse_args()

    a_dir, b_dir = ROOT / args.run_a, ROOT / args.run_b

    # -- 1. are these even two runs of the same thing? -------------------
    print("wall-clock coverage")
    span_a = stamp_report("A", a_dir)
    span_b = stamp_report("B", b_dir)
    print()

    a, b = load(a_dir, args.pass_name), load(b_dir, args.pass_name)
    versions = Counter(r for d in (a_dir, b_dir) for p in d.glob("*.jsonl")
                       for rec in iter_jsonl(p)
                       for r in [rec.get("prompt_version", "?")])
    print(f"prompt versions present: {dict(versions)}")
    if len(versions) > 1:
        print("  MIXED PROMPT VERSIONS - the runs are not comparable. Delete both")
        print("  directories and rerun; the runner resumes, so stale cells survive.")
    print()

    # -- 2. where do the labels disagree? --------------------------------
    ga, gb = index(a, "generate"), index(b, "generate")
    by_model: Counter[str] = Counter()
    by_prec: Counter[str] = Counter()
    by_cond: Counter[str] = Counter()
    by_lang: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    examples = []

    for key, per_prec in ga.items():
        model, fact_id, lang, cond, mode = key
        for prec, ra in per_prec.items():
            rb = gb.get(key, {}).get(prec)
            if rb is None or ra.label is None or rb.label is None:
                continue
            totals[f"{model}|{prec}"] += 1
            if ra.label is not rb.label:
                by_model[model] += 1
                by_prec[prec] += 1
                by_cond[cond] += 1
                by_lang[lang] += 1
                if len(examples) < 8:
                    examples.append((model, prec, lang, cond, mode,
                                     ra.label.value, rb.label.value,
                                     ra.text[:28], rb.text[:28]))

    total_disagree = sum(by_model.values())
    total_pairs = sum(totals.values())
    print(f"label disagreement: {total_disagree}/{total_pairs} = "
          f"{total_disagree / max(total_pairs, 1):.4f}")

    for name, counter in (("model", by_model), ("precision", by_prec),
                          ("condition", by_cond), ("language", by_lang)):
        if counter:
            print(f"  by {name:<10} " +
                  "  ".join(f"{k}={v}" for k, v in counter.most_common()))

    # Concentration is the tell: noise spreads, mistakes cluster.
    if by_model and len(by_model) > 1:
        share = max(by_model.values()) / total_disagree
        if share > 0.7:
            print(f"  -> {share:.0%} of disagreements sit on one model. That is a")
            print("     mistake, not noise. Suspect stale output or a mid-run edit.")

    # -- 3. how large are the score differences? -------------------------
    sa, sb = index(a, "score"), index(b, "score")
    deltas, nans, zeros = [], 0, 0
    big = []
    for key, per_prec in sa.items():
        for prec, ra in per_prec.items():
            rb = sb.get(key, {}).get(prec)
            if rb is None or ra.r is None or rb.r is None:
                continue
            d = ra.r - rb.r
            if d != d:  # nan
                nans += 1
                continue
            d = abs(d)
            deltas.append(d)
            zeros += d == 0.0
            if d > 0.5 and len(big) < 8:
                big.append((key, prec, ra.r, rb.r))

    print()
    if deltas:
        print(f"score comparisons  : {len(deltas)}")
        print(f"  exactly zero     : {zeros} ({zeros / len(deltas):.1%})")
        print(f"  median |ΔR|      : {statistics.median(deltas):.3e}")
        print(f"  max |ΔR|         : {max(deltas):.3e}")
        print(f"  nan              : {nans}")
        nonzero = [d for d in deltas if d > 0]
        if nonzero:
            print(f"  of the {len(nonzero)} nonzero: median {statistics.median(nonzero):.3e}, "
                  f"max {max(nonzero):.3e}")
            print()
            if statistics.median(nonzero) > 0.1:
                print("  Differences of this size are not floating-point noise. Two")
                print("  evaluations of the same tokens cannot land 0.1 apart in log")
                print("  space; something upstream differed.")

    if big:
        print("\nlargest score differences")
        for (model, fid, lang, cond, mode), prec, ra, rb in big:
            print(f"  {model} {prec} {lang} {cond} {mode}: R {ra:+.3f} vs {rb:+.3f}")

    if examples:
        print("\nlabel disagreements")
        for m, p, lg, c, md, la, lb, ta, tb in examples:
            print(f"  {m} {p} {lg} {c} {md}")
            print(f"      A {la:<8} {ta!r}")
            print(f"      B {lb:<8} {tb!r}")

    print()
    if span_a and span_b and max(span_a, span_b) > 4 * min(span_a, span_b) + 60:
        print("The two runs cover very different stretches of wall-clock time.")
        print("One of them is largely resumed from an earlier invocation.")
    print("If anything above points at stale output: rm -rf the two directories")
    print("and rerun. The resume logic is right for the grid and wrong here -")
    print("a noise measurement needs both halves computed fresh.")


if __name__ == "__main__":
    main()
