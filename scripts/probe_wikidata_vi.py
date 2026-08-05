"""Probe Vietnamese label coverage on Wikidata for a list of QIDs.

CounterFact carries Wikidata QIDs for both the true and the counterfactual
object. That means the Vietnamese arm does not need the entities translated at
all: pull the official Vietnamese label for each QID and the EN/VI pair is
exact by construction, with no translation noise in the entity names.

Coverage is uneven for tail entities, which is useful rather than annoying - it
gives an objective, model-free filter, and it can be measured on day 1 instead
of discovering the problem when the known-subset collapses on day 5.

    python scripts/probe_wikidata_vi.py Q1860 Q150 Q90 Q84
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request

API = "https://www.wikidata.org/w/api.php"
UA = "QuantRAG/0.1 (research; contact via repo)"


def fetch_labels(
    qids: list[str],
    langs: tuple[str, ...] = ("en", "vi"),
    with_aliases: bool = False,
) -> dict:
    """Batch-fetch labels, optionally with aliases. Up to 50 ids per call.

    Aliases matter more than they look: the evaluator's alias list is the single
    most likely source of silently wrong rates, and Wikidata already knows that
    London is Luân Đôn. Seeding the list from here beats writing it by hand,
    though it still needs a human pass - Wikidata aliases include spellings that
    are attested but not what a model would ever emit.
    """
    props = "labels|aliases" if with_aliases else "labels"
    out: dict[str, dict] = {}
    for i in range(0, len(qids), 50):
        chunk = qids[i:i + 50]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(chunk),
            "props": props,
            "languages": "|".join(langs),
            "format": "json",
        }
        req = urllib.request.Request(
            f"{API}?{urllib.parse.urlencode(params)}", headers={"User-Agent": UA}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        for qid, ent in data.get("entities", {}).items():
            labels = ent.get("labels", {})
            rec: dict = {lg: labels.get(lg, {}).get("value") for lg in langs}
            if with_aliases:
                al = ent.get("aliases", {})
                rec["aliases"] = {
                    lg: [a["value"] for a in al.get(lg, [])] for lg in langs
                }
            out[qid] = rec
    return out


def main() -> None:
    qids = sys.argv[1:] or ["Q1860", "Q150", "Q90", "Q84", "Q142", "Q30"]
    labels = fetch_labels(qids, with_aliases=True)
    have_vi = 0
    for qid in qids:
        rec = labels.get(qid, {})
        en, vi = rec.get("en"), rec.get("vi")
        have_vi += vi is not None
        vi_al = rec.get("aliases", {}).get("vi", [])[:4]
        print(f"{qid:9} en={en!r:22} vi={vi!r:22} vi_aliases={vi_al}")
    print(f"\nVietnamese label coverage: {have_vi}/{len(qids)}")


if __name__ == "__main__":
    main()
