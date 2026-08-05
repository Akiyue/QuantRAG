"""Build data/facts.jsonl from CounterFact + Wikidata.

Pipeline:

  1. cache CounterFact rows from the HF datasets server
  2. keep the relations declared in configs/relations.yaml
  3. pull Vietnamese labels and aliases for every object QID
  4. keep rows where BOTH objects have a Vietnamese label
  5. resolve each subject to a QID by name and verify it against the relation
     claim, dropping anything that fails
  6. render the templates and write validated Fact records

Steps 3-5 are the expensive ones, so they run in that order: filter cheaply
first, and pay for subject verification only on the shortlist.

    python scripts/build_dataset.py --target 500

Safe to interrupt and re-run: every Wikidata response is cached on disk.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantrag.schema import Fact, write_facts  # noqa: E402
from quantrag.wikidata import Wikidata  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ROWS_API = "https://datasets-server.huggingface.co/rows"
DATASET = "NeelNanda/counterfact-tracing"
UA = "QuantRAG/0.1 (academic research)"


# --------------------------------------------------------------- counterfact

def cache_counterfact(limit: int) -> list[dict]:
    path = RAW / "counterfact.jsonl"
    if path.exists():
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
        print(f"  cached: {len(rows)} rows")
        return rows

    RAW.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    while len(rows) < limit:
        q = urllib.parse.urlencode({
            "dataset": DATASET, "config": "default", "split": "train",
            "offset": len(rows), "length": 100,
        })
        req = urllib.request.Request(f"{ROWS_API}?{q}", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=90) as resp:
            payload = json.load(resp)
        batch = [r["row"] for r in payload.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        print(f"\r  fetched {len(rows)}", end="", flush=True)
    print()
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


# ------------------------------------------------------------------ helpers

def alias_set(labels: dict, aliases: dict, lang: str) -> list[str]:
    """Alias list for one language, always including the English label.

    Small models code-switch constantly: asked in Vietnamese they will often
    answer 'London' rather than 'Luân Đôn', and Wikidata's Vietnamese aliases
    for London are empty. Without the English form the correct answer lands in
    OTHER and both CCR and PRR are quietly wrong.

    Wikidata aliases still need a human pass: they include epithets ('Kinh đô
    ánh sáng' for Paris) and semantic stretches ('tiếng Mỹ' for English) that
    would cause false positives. That is what the `aliases` gate certifies.
    """
    out = [labels.get(lang), *aliases.get(lang, [])]
    if lang != "en":
        out.append(labels.get("en"))
    seen: dict[str, None] = {}
    for a in out:
        if a:
            seen.setdefault(a, None)
    return list(seen)


def render(tpl: dict, subject: str, obj: str) -> dict[str, str]:
    return {k: v.format(s=subject, o=obj) for k, v in tpl.items()}


def build_fact(row: dict, resolved: dict, objects: dict, cfg: dict,
               index: int) -> Fact | None:
    rel = row["relation_id"]
    subject_en = row["subject"].strip()
    s_vi = (resolved["labels"] or {}).get("vi")
    subject = {"en": subject_en, "vi": s_vi or subject_en}

    t_ent = objects[row["target_true_id"]]
    f_ent = objects[row["target_false_id"]]
    o_true = {lg: t_ent["labels"][lg] or row["target_true"].strip() for lg in ("en", "vi")}
    o_fake = {lg: f_ent["labels"][lg] or row["target_false"].strip() for lg in ("en", "vi")}

    built: dict[str, dict[str, str]] = defaultdict(dict)
    for lg in ("en", "vi"):
        tpl = cfg[lg]
        true_txt = render(tpl, subject[lg], o_true[lg])
        fake_txt = render(tpl, subject[lg], o_fake[lg])
        built["q_filter"][lg] = true_txt["q_filter"]
        built["q_eval"][lg] = true_txt["q_eval"]
        built["evidence_true"][lg] = true_txt["evidence"]
        built["evidence_fake"][lg] = fake_txt["evidence"]
        built["evidence_fake_para"][lg] = fake_txt["evidence_para"]
        built["evidence_irrelevant"][lg] = true_txt["irrelevant"]
        built["evidence_control"][lg] = true_txt["control"]

    fact = Fact(
        fact_id=f"{cfg['name']}_{index:04d}",
        domain=cfg["domain"],
        relation=cfg["name"],
        subject=subject,
        object_true=o_true,
        object_fake=o_fake,
        q_filter=dict(built["q_filter"]),
        q_eval=dict(built["q_eval"]),
        evidence_true=dict(built["evidence_true"]),
        evidence_fake=dict(built["evidence_fake"]),
        evidence_irrelevant=dict(built["evidence_irrelevant"]),
        evidence_fake_para=dict(built["evidence_fake_para"]),
        evidence_control=dict(built["evidence_control"]),
        aliases_true={lg: alias_set(t_ent["labels"], t_ent["aliases"], lg)
                      for lg in ("en", "vi")},
        aliases_fake={lg: alias_set(f_ent["labels"], f_ent["aliases"], lg)
                      for lg in ("en", "vi")},
        relation_id=rel,
        subject_qid=resolved["qid"],
        object_true_qid=row["target_true_id"],
        object_fake_qid=row["target_false_id"],
        subject_localized=bool(s_vi),
        source=f"counterfact-tracing:{rel}",
        # Entities come from Wikidata labels rather than translation. What still
        # needs a person is the template wording and the alias lists, which the
        # `templates` and `aliases` gates cover.
        vi_translation_checked=True,
    )
    return None if fact.validate() else fact


# --------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--pool-factor", type=float, default=3.0,
                    help="candidates shortlisted per kept fact")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="data/facts.jsonl")
    ap.add_argument("--max-rows", type=int, default=25_000,
                    help="CounterFact rows to cache; lower it for a quick trial")
    ap.add_argument("--delay", type=float, default=0.2,
                    help="seconds between Wikidata calls")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rel_cfg = yaml.safe_load(
        (ROOT / "configs" / "relations.yaml").read_text(encoding="utf-8")
    )
    relations = rel_cfg["relations"]

    print("1. CounterFact")
    rows = cache_counterfact(args.max_rows)

    print("2. relation filter")
    rows = [r for r in rows if r["relation_id"] in relations]
    print(f"  {len(rows)} rows across {len(relations)} declared relations")

    print("3. object labels")
    wd = Wikidata(RAW / "wikidata_cache.json", delay=args.delay)
    qids = {r["target_true_id"] for r in rows} | {r["target_false_id"] for r in rows}
    try:
        objects = wd.entities(sorted(qids))
    finally:
        wd.save()
    print(f"  {len(objects)} object entities")

    print("4. Vietnamese object coverage")

    def has_vi(qid: str) -> bool:
        return bool((objects.get(qid) or {}).get("labels", {}).get("vi"))

    rows = [r for r in rows
            if has_vi(r["target_true_id"]) and has_vi(r["target_false_id"])]
    by_rel: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_rel[r["relation_id"]].append(r)
    for rel, rs in sorted(by_rel.items()):
        print(f"  {rel:>6} {len(rs):>6}")
    if not by_rel:
        sys.exit("no rows survived the Vietnamese coverage filter")

    print("5. shortlist and verify subjects")
    per_rel = max(1, int(args.target * args.pool_factor / len(by_rel)))
    shortlist: list[dict] = []
    for rs in by_rel.values():
        rng.shuffle(rs)
        shortlist.extend(rs[:per_rel])
    rng.shuffle(shortlist)

    facts: list[Fact] = []
    kept: Counter[str] = Counter()
    quota = max(1, args.target // len(by_rel))
    checked = verified = 0

    try:
        for row in shortlist:
            if len(facts) >= args.target:
                break
            rel = row["relation_id"]
            if kept[rel] >= quota:
                continue

            checked += 1
            print(f"\r  checked {checked}/{len(shortlist)}, verified {verified},"
                  f" kept {len(facts)}", end="", flush=True)

            resolved = wd.resolve_subject(
                row["subject"].strip(), rel, row["target_true_id"]
            )
            if resolved is None:
                continue
            verified += 1
            if checked % 25 == 0:
                wd.save()

            fact = build_fact(row, resolved, objects, relations[rel], len(facts))
            if fact is not None:
                facts.append(fact)
                kept[rel] += 1
    finally:
        # A full build makes thousands of calls and will get interrupted.
        # Losing the cache would mean paying for all of them again.
        wd.save()
        print()

    print("6. write")
    out = ROOT / args.out
    write_facts(out, facts)
    print(f"  {len(facts)} facts -> {out}")
    print(f"  subject verification rate : {verified}/{checked}"
          f" ({100 * verified / max(1, checked):.1f}%)")
    loc = sum(f.subject_localized for f in facts)
    print(f"  subjects with a VI label  : {loc}/{len(facts)}"
          f" ({100 * loc / max(1, len(facts)):.1f}%)")
    print(f"  per relation              : {dict(kept)}")
    if len(facts) < args.target:
        print(f"\n  short of target ({len(facts)}/{args.target}) -"
              f" raise --pool-factor or --max-rows")
    print("\nNext: review the alias lists and the Vietnamese templates, then")
    print("  ./run.sh gate aliases && ./run.sh gate templates")


if __name__ == "__main__":
    main()
