"""Inspect the actual field layout of the CounterFact variants on HF.

Specifically: is there a Wikidata QID for the *subject*? Objects have one, and
that is what makes the Vietnamese arm translation-free. If subjects do not, the
subject names need a separate resolution path and that changes the dataset
build.

    python scripts/probe_counterfact_schema.py
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

ROWS = "https://datasets-server.huggingface.co/first-rows"
UA = "QuantRAG/0.1 (research; contact via repo)"

DATASETS = [
    ("NeelNanda/counterfact-tracing", "default", "train"),
    ("azhx/counterfact", "default", "train"),
    ("azhx/counterfact-easy", "default", "train"),
]


def first_rows(dataset: str, config: str, split: str) -> dict:
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split})
    req = urllib.request.Request(f"{ROWS}?{q}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def main() -> None:
    for name, config, split in DATASETS:
        print("=" * 72)
        print(name)
        try:
            data = first_rows(name, config, split)
        except Exception as exc:  # noqa: BLE001 - probing, report and continue
            print(f"  unavailable: {exc}")
            continue

        cols = [f["name"] for f in data.get("features", [])]
        print(f"  columns: {cols}")

        rows = data.get("rows", [])
        if rows:
            print("  first row:")
            print("   ", json.dumps(rows[0]["row"], ensure_ascii=False)[:900])

        subj_id_fields = [c for c in cols if "subject" in c.lower() and "id" in c.lower()]
        print(f"  subject-id fields: {subj_id_fields or 'NONE'}")


if __name__ == "__main__":
    main()
