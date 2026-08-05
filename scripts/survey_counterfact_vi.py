"""Measure Vietnamese coverage on real CounterFact entities.

Reads CounterFact rows (cached, shared with the dataset build) and asks Wikidata
for Vietnamese labels of every true/false object.

This answers the riskiest open question in the project - whether the Vietnamese
arm has enough paired items to be worth running - on day one, using no GPU and
no model weights. If coverage is poor, the bilingual framing has to change
before any compute is spent, not after.

Coverage varies by relation, and that is the actionable part: keeping only the
relations whose objects are near-fully covered gives a paired EN/VI set without
translating a single entity name.

    python scripts/survey_counterfact_vi.py 21919
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantrag.counterfact import load_rows  # noqa: E402
from quantrag.wikidata import Wikidata  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    print("CounterFact rows")
    rows = load_rows(RAW / "counterfact.jsonl", n)
    print(f"  using {len(rows)}\n")

    qids = sorted({r["target_true_id"] for r in rows}
                  | {r["target_false_id"] for r in rows})
    print(f"unique object QIDs: {len(qids)}")
    wd = Wikidata(RAW / "wikidata_cache.json")
    try:
        labels = wd.entities(qids)
    finally:
        wd.save()

    def has_vi(qid: str) -> bool:
        return bool((labels.get(qid) or {}).get("labels", {}).get("vi"))

    both_vi = 0
    ok: Counter[str] = Counter()
    total: Counter[str] = Counter()
    examples: list[tuple[str, str, str, str, str]] = []

    for r in rows:
        t, f = r["target_true_id"], r["target_false_id"]
        good = has_vi(t) and has_vi(f)
        both_vi += good
        total[r["relation_id"]] += 1
        if good:
            ok[r["relation_id"]] += 1
            if len(examples) < 8:
                examples.append((
                    r["subject"].strip(),
                    labels[t]["labels"]["en"] or "", labels[t]["labels"]["vi"] or "",
                    labels[f]["labels"]["en"] or "", labels[f]["labels"]["vi"] or "",
                ))

    pct = 100 * both_vi / len(rows) if rows else 0
    print(f"\nrows where BOTH objects have a Vietnamese label: "
          f"{both_vi}/{len(rows)} ({pct:.1f}%)")

    print("\nby relation, sorted by coverage (n >= 20):")
    ranked = sorted(((rel, ok[rel], tot) for rel, tot in total.items() if tot >= 20),
                    key=lambda x: -x[1] / x[2])
    keep = []
    for rel, good, tot in ranked:
        share = 100 * good / tot
        mark = "keep" if share >= 95 else "    "
        if share >= 95:
            keep.append(rel)
        print(f"  {mark} {rel:>6} {good:>5}/{tot:<6} {share:5.1f}%")

    print(f"\nrelations at >=95% coverage ({len(keep)}):")
    print(f"  {' '.join(keep)}")
    print("\nPut these in configs/relations.yaml. Anything below is not worth the")
    print("items it costs - the Vietnamese arm needs paired facts, not more facts.")

    print("\nexamples:")
    for subj, ten, tvi, fen, fvi in examples:
        print(f"  {subj[:32]:<32} {ten} / {tvi}   vs   {fen} / {fvi}")


if __name__ == "__main__":
    main()
