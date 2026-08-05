"""Human validation of the automatic evaluator.

    python scripts/evaluator_review.py sample --n 200
    # fill in the `human_label` column of results/evaluator_review.csv
    python scripts/evaluator_review.py score

The automatic evaluator turns generated text into TRUE/FAKE/REFUSAL/OTHER, and
every headline rate is built on it. Vietnamese makes this harder than it looks:
entities have two legitimate surface forms, diacritics may be dropped, and small
models code-switch. The agreement number this produces belongs in the paper.

Sampling deliberately oversamples answers that mention both candidates. Those
are where the evaluator is most likely to be wrong, so a uniform sample would
spend most of its budget confirming easy cases.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantrag.analysis import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "evaluator_review.csv"
FIELDS = ["fact_id", "lang", "condition", "mode", "model_id", "precision",
          "text", "auto_label", "both_present", "human_label"]


def cmd_sample(args) -> None:
    records = [r for r in load(ROOT / args.runs, args.pass_name)
               if r.kind == "generate" and r.label is not None]
    if not records:
        sys.exit(f"no generation records in {args.runs}")

    rng = random.Random(args.seed)
    ambiguous = [r for r in records if r.both_present]
    plain = [r for r in records if not r.both_present]
    rng.shuffle(ambiguous)
    rng.shuffle(plain)

    n_amb = min(len(ambiguous), args.n // 2)
    chosen = ambiguous[:n_amb] + plain[: args.n - n_amb]
    rng.shuffle(chosen)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in chosen:
            w.writerow({
                "fact_id": r.key.fact_id, "lang": r.key.lang,
                "condition": r.key.condition, "mode": r.key.mode,
                "model_id": r.model_id, "precision": r.precision,
                "text": r.text.replace("\n", " ")[:300],
                "auto_label": r.label.value if r.label else "",
                "both_present": int(r.both_present), "human_label": "",
            })
    print(f"wrote {len(chosen)} rows to {OUT.relative_to(ROOT)}")
    print(f"  {n_amb} mention both candidates (deliberately oversampled)")
    print("\nFill in `human_label` with TRUE / FAKE / REFUSAL / OTHER.")
    print("Do not look at auto_label while deciding - it defeats the purpose.")


def cmd_score(args) -> None:
    if not OUT.exists():
        sys.exit(f"{OUT} not found; run `sample` first")
    rows = list(csv.DictReader(open(OUT, encoding="utf-8")))
    done = [r for r in rows if r["human_label"].strip()]
    if not done:
        sys.exit("no human labels filled in yet")

    agree = sum(r["auto_label"] == r["human_label"].strip().upper() for r in done)
    rate = agree / len(done)

    print(f"labelled : {len(done)}/{len(rows)}")
    print(f"agreement: {agree}/{len(done)} = {rate:.4f}")

    conf: Counter[tuple[str, str]] = Counter(
        (r["auto_label"], r["human_label"].strip().upper()) for r in done
    )
    print("\ndisagreements (auto -> human):")
    for (a, h), n in sorted(conf.items(), key=lambda kv: -kv[1]):
        if a != h:
            print(f"  {a:>8} -> {h:<8} {n}")

    amb = [r for r in done if r["both_present"] == "1"]
    if amb:
        amb_ok = sum(r["auto_label"] == r["human_label"].strip().upper() for r in amb)
        print(f"\non answers mentioning both candidates: {amb_ok}/{len(amb)}"
              f" = {amb_ok / len(amb):.4f}")

    print()
    if rate >= 0.95:
        print(f"Agreement {rate:.3f}. Report this number in the paper.")
        print("Next: ./run.sh gate evaluator")
    else:
        print(f"Agreement {rate:.3f} is below 0.95. Fix the evaluator - usually the")
        print("alias lists - and re-run the affected generation cells before")
        print("reporting any rate built on these labels.")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample")
    s.add_argument("--runs", default="runs")
    s.add_argument("--pass-name", default="main")
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--seed", type=int, default=1234)
    s.set_defaults(fn=cmd_sample)

    c = sub.add_parser("score")
    c.set_defaults(fn=cmd_score)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
