"""Measure Vietnamese coverage on real CounterFact entities.

Pulls a sample of CounterFact rows straight from the HF datasets server, then
asks Wikidata for Vietnamese labels of every true/false object.

This answers the single riskiest open question in the project - whether the
Vietnamese arm has enough paired items to be worth running - on day 1, using no
GPU and no model weights. If coverage is poor, the bilingual framing has to
change before any compute is spent, not after.

    python scripts/survey_counterfact_vi.py [n_rows]
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from collections import Counter

from probe_wikidata_vi import fetch_labels  # noqa: E402

ROWS_API = "https://datasets-server.huggingface.co/rows"
DATASET = "NeelNanda/counterfact-tracing"
UA = "QuantRAG/0.1 (research; contact via repo)"


def fetch_rows(n: int) -> list[dict]:
    rows: list[dict] = []
    while len(rows) < n:
        params = {
            "dataset": DATASET, "config": "default", "split": "train",
            "offset": len(rows), "length": min(100, n - len(rows)),
        }
        req = urllib.request.Request(
            f"{ROWS_API}?{urllib.parse.urlencode(params)}", headers={"User-Agent": UA}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.load(resp)
        batch = [r["row"] for r in payload.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
    return rows[:n]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rows = fetch_rows(n)
    print(f"fetched {len(rows)} CounterFact rows\n")

    qids = sorted({r["target_true_id"] for r in rows} | {r["target_false_id"] for r in rows})
    print(f"unique object QIDs: {len(qids)}")
    labels = fetch_labels(qids)

    both_vi = 0
    relation_ok: Counter[str] = Counter()
    relation_all: Counter[str] = Counter()
    examples: list[tuple[str, str, str, str, str]] = []

    for r in rows:
        t, f = labels.get(r["target_true_id"], {}), labels.get(r["target_false_id"], {})
        ok = bool(t.get("vi")) and bool(f.get("vi"))
        both_vi += ok
        relation_all[r["relation_id"]] += 1
        if ok:
            relation_ok[r["relation_id"]] += 1
            if len(examples) < 8:
                examples.append((r["subject"], t.get("en") or "", t.get("vi") or "",
                                 f.get("en") or "", f.get("vi") or ""))

    pct = 100 * both_vi / len(rows) if rows else 0
    print(f"\nrows where BOTH objects have a Vietnamese label: {both_vi}/{len(rows)} ({pct:.1f}%)")

    print("\nby relation (top 12 by volume):")
    for rel, total in relation_all.most_common(12):
        ok = relation_ok[rel]
        print(f"  {rel:>6}  {ok:>4}/{total:<4}  {100*ok/total:5.1f}%")

    print("\nexamples:")
    for subj, ten, tvi, fen, fvi in examples:
        print(f"  {subj[:34]:<34} {ten} / {tvi}   vs   {fen} / {fvi}")


if __name__ == "__main__":
    main()
