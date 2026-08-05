"""Can we recover subject QIDs by name, and verify them?

CounterFact gives QIDs for the objects but not the subject, so the Vietnamese
subject name has nowhere to come from. Proposed fix: search Wikidata by the
English subject string, then *verify* the candidate by checking that it really
carries the relation claim CounterFact asserts (subject --relation--> target_true).

Verification is what makes this safe. A bare name search would happily return a
different Danielle Darrieux; an entity that also has P103 = Q150 is almost
certainly the right one. It doubles as a freshness check, since a fact that no
longer holds in Wikidata gets dropped rather than quietly aging into the dataset.

    python scripts/probe_subject_resolution.py
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

API = "https://www.wikidata.org/w/api.php"
UA = "QuantRAG/0.1 (research; contact via repo)"

# (subject, relation_id, expected target_true QID) drawn from real CounterFact rows
CASES = [
    ("Danielle Darrieux", "P103", "Q150"),
    ("Edwin of Northumbria", "P140", "Q5043"),
    ("Toko Yasuda", "P1303", "Q6607"),
    ("Autonomous University of Madrid", "P17", "Q29"),
    ("Anaal Nathrakh", "P740", "Q2256"),
    ("Thomas Joannes Stieltjes", "P103", "Q7411"),
]


def _get(params: dict) -> dict:
    req = urllib.request.Request(
        f"{API}?{urllib.parse.urlencode(params)}", headers={"User-Agent": UA}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def search(name: str, limit: int = 5) -> list[str]:
    data = _get({
        "action": "wbsearchentities", "search": name, "language": "en",
        "uselang": "en", "type": "item", "limit": limit, "format": "json",
    })
    return [r["id"] for r in data.get("search", [])]


def claims_and_labels(qid: str) -> tuple[dict, dict]:
    data = _get({
        "action": "wbgetentities", "ids": qid, "props": "claims|labels",
        "languages": "en|vi", "format": "json",
    })
    ent = data.get("entities", {}).get(qid, {})
    return ent.get("claims", {}), ent.get("labels", {})


def relation_targets(claims: dict, pid: str) -> set[str]:
    out: set[str] = set()
    for st in claims.get(pid, []):
        val = st.get("mainsnak", {}).get("datavalue", {}).get("value", {})
        if isinstance(val, dict) and "id" in val:
            out.add(val["id"])
    return out


def resolve(name: str, pid: str, expect_qid: str) -> tuple[str | None, dict]:
    """Return the first candidate whose claims confirm the relation."""
    for qid in search(name):
        claims, labels = claims_and_labels(qid)
        if expect_qid in relation_targets(claims, pid):
            return qid, labels
    return None, {}


def main() -> None:
    ok = 0
    for name, pid, expect in CASES:
        qid, labels = resolve(name, pid, expect)
        if qid:
            ok += 1
            en = labels.get("en", {}).get("value")
            vi = labels.get("vi", {}).get("value")
            print(f"  OK   {name:<34} -> {qid:<9} en={en!r} vi={vi!r}")
        else:
            print(f"  MISS {name:<34} (no candidate carries {pid} = {expect})")
    print(f"\nverified {ok}/{len(CASES)}")


if __name__ == "__main__":
    main()
