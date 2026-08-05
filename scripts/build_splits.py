"""Partition items into parametric-known subsets from the C0 filter pass.

    python scripts/build_splits.py

Writes data/splits.json with three (fact_id, lang) sets. Everything downstream
is scoped by one of them, so this runs before the main grid is analysed - and
before the main grid is even worth running, since a collapsed Vietnamese subset
changes what the paper can claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantrag.analysis import known_subsets, load, models, precisions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="data/splits.json")
    ap.add_argument("--min-vi", type=int, default=150,
                    help="warn if the Vietnamese known_all subset is smaller")
    args = ap.parse_args()

    records = load(ROOT / args.runs, "filter")
    if not records:
        sys.exit(f"no filter-pass records in {args.runs}; run `./run.sh filter` first")

    print(f"models     : {models(records)}")
    print(f"precisions : {precisions(records)}")

    splits = known_subsets(records)

    print("\nsplit sizes")
    per_lang: dict[str, Counter[str]] = {}
    for name, items in splits.items():
        langs = Counter(lang for _, lang in items)
        per_lang[name] = langs
        detail = " ".join(f"{lg}={n}" for lg, n in sorted(langs.items()))
        print(f"  {name:<11} {len(items):>5}   {detail}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({k: sorted(v) for k, v in splits.items()}, ensure_ascii=False,
                   indent=1),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")

    vi = per_lang["known_all"].get("vi", 0)
    en = per_lang["known_all"].get("en", 0)
    if vi < args.min_vi:
        print(f"\n  WARNING: Vietnamese known_all is {vi} (< {args.min_vi}).")
        print("  The bilingual comparison is underpowered at this size. Decide now,")
        print("  before the main grid burns GPU time, whether to demote the")
        print("  Vietnamese arm to a secondary analysis and say so in the paper,")
        print("  or to move to a larger model for that arm (PLAN day 5).")
    elif en and abs(en - vi) / max(en, vi) > 0.3:
        print(f"\n  NOTE: en={en} vs vi={vi} differ by more than 30%. The arms are")
        print("  no longer paired on equal footing; report the intersection as the")
        print("  primary bilingual analysis.")

    print("\nNext: ./run.sh gate known")


if __name__ == "__main__":
    main()
