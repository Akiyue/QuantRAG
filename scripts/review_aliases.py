"""Make the alias review tractable.

    python scripts/review_aliases.py list     # writes data/alias_review.yaml
    # edit that file: set keep: false on anything that is not an answer
    python scripts/review_aliases.py apply    # writes the decisions back

The alias lists decide whether a correct answer is scored as correct, so they
are the single most likely source of a silently wrong rate. But reading 500
facts is not a review, it is a formality that gets skipped at midnight.

So this surfaces only the entries that need a judgement. Wikidata aliases are
mostly harmless variants - 'Viet Nam' for 'Việt Nam' - and those are kept
without asking. What needs a person is the other kind:

  * epithets: 'Kinh đô ánh sáng' is a real alias of Paris and is not an answer
  * semantic stretches: 'tiếng Mỹ' listed under English would count an answer
    of "American" as correct
  * disambiguated forms: 'Vienna, Áo' is a title, not something a model emits

Everything flagged is shown with its reason. Everything else is listed at the
bottom, so the review is auditable rather than hidden.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantrag.schema import Fact, read_facts, write_facts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "data" / "alias_review.yaml"


def strip_tones(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if not unicodedata.combining(c)).casefold()


def words(s: str) -> list[str]:
    return [w for w in re.split(r"\W+", strip_tones(s)) if w]


def risk(label: str, alias: str) -> tuple[int, str]:
    """A sorting hint, not a verdict.

    Telling "another name for this entity" from "the name of a different
    entity" is a semantic judgement, and string comparison cannot make it. The
    tempting heuristics get it backwards in both directions: 'Ba Lê' and 'Mỹ'
    share nothing with 'Paris' and 'Hoa Kỳ' yet are perfectly good answers,
    while 'tiếng Mỹ' shares a word with 'tiếng Anh' and is exactly the alias
    that would score "American" as English.

    So nothing is auto-rejected and nothing risky is auto-accepted. This only
    decides what to read first.
    """
    la, lw = words(label), words(alias)
    if not lw:
        return 100, "empty"
    if any(ch in alias for ch in ",()[]/"):
        return 90, "punctuation - reads like a disambiguated title"
    if len(lw) >= len(la) + 2:
        return 80, f"{len(lw)} words against {len(la)} - reads like a description"
    if len(alias) > max(2 * len(label), len(label) + 12):
        return 70, "much longer than the label"
    shared = set(lw) & set(la)
    if shared and len(lw) == len(la):
        # Same shape, one word swapped: 'tiếng Anh' -> 'tiếng Mỹ'. This is the
        # pattern that produces a plausible-looking alias for a different thing.
        return 60, f"same shape, differs in one word - check it means the same"
    if not shared:
        return 30, "no shared word - usually a legitimate variant, confirm"
    return 10, "minor variant"


def collect(facts: list[Fact]) -> dict:
    """Unique (qid, lang) alias sets. Objects repeat heavily across facts, so
    reviewing per object is a fraction of the work of reviewing per fact."""
    seen: dict[tuple[str, str], dict] = {}
    for f in facts:
        for which, qid in (("true", f.object_true_qid), ("fake", f.object_fake_qid)):
            obj = f.object_true if which == "true" else f.object_fake
            aliases = f.aliases_true if which == "true" else f.aliases_fake
            for lang in ("en", "vi"):
                key = (qid or obj[lang], lang)
                if key in seen:
                    continue
                seen[key] = {
                    "qid": qid, "lang": lang, "label": obj[lang],
                    "english": obj["en"], "aliases": list(aliases.get(lang, [])),
                }
    return seen


def cmd_list(args) -> None:
    facts = read_facts(ROOT / args.facts)
    seen = collect(facts)

    rows, auto = [], 0
    for (_, lang), rec in seen.items():
        entries = []
        top = 0
        for a in rec["aliases"]:
            # Two things are kept without asking: the canonical label, and the
            # English label in a Vietnamese list. The second is deliberate -
            # small models code-switch, and dropping it would score a correct
            # 'London' as OTHER.
            if a == rec["label"] or (lang != "en" and a == rec["english"]):
                auto += 1
                continue
            score, why = risk(rec["label"], a)
            entries.append({"alias": a, "keep": True, "why": why, "_score": score})
            top = max(top, score)
        if entries:
            entries.sort(key=lambda e: -e["_score"])
            rows.append({"_score": top, "qid": rec["qid"], "lang": lang,
                         "label": rec["label"], "aliases": entries})

    rows.sort(key=lambda r: -r["_score"])
    for r in rows:
        r.pop("_score")
        for e in r["aliases"]:
            e.pop("_score")

    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW, "w", encoding="utf-8") as fh:
        fh.write("# Alias review - riskiest first.\n"
                 "#\n"
                 "# Everything here defaults to keep: true. Set keep: false on any\n"
                 "# alias that is not a name a model would emit as THE ANSWER:\n"
                 "#   - descriptions rather than names ('Kinh do anh sang' for Paris)\n"
                 "#   - names of a different thing ('tieng My' under English)\n"
                 "#   - disambiguated titles ('Vienna, Ao')\n"
                 "#\n"
                 "# `why` is a sorting hint, not a verdict - string comparison cannot\n"
                 "# tell another name for X from the name of something else, so the\n"
                 "# judgement is yours. Canonical labels and the English label in a\n"
                 "# Vietnamese list are kept automatically and are not listed.\n"
                 "#\n"
                 "# Then: python scripts/review_aliases.py apply\n\n")
        yaml.safe_dump({"aliases": rows}, fh, allow_unicode=True,
                       sort_keys=False, width=100, default_flow_style=False)

    n_alias = sum(len(r["aliases"]) for r in rows)
    print(f"unique objects       : {len(seen)}")
    print(f"kept automatically   : {auto}  (canonical + English label)")
    print(f"for you to decide    : {n_alias} aliases across {len(rows)} objects")
    print(f"\nwrote {REVIEW.relative_to(ROOT)}  (riskiest first)")

    print("\ntop of the list:")
    for row in rows[:8]:
        for a in row["aliases"][:1]:
            print(f"  {row['lang']}  {row['label']:<20} {a['alias']!r:<26} {a['why']}")
    print("\nEdit, then: python scripts/review_aliases.py apply")


def cmd_apply(args) -> None:
    if not REVIEW.exists():
        sys.exit(f"{REVIEW} not found; run `list` first")
    doc = yaml.safe_load(REVIEW.read_text(encoding="utf-8"))
    decisions: dict[tuple[str, str], list[str]] = {}
    for section in ("needs_review", "accepted"):
        for row in doc.get(section) or []:
            key = (row["qid"] or row["label"], row["lang"])
            decisions[key] = [a["alias"] for a in row["aliases"] if a.get("keep")]

    facts = read_facts(ROOT / args.facts)
    removed = 0
    out: list[Fact] = []
    for f in facts:
        new_true, new_fake = {}, {}
        for lang in ("en", "vi"):
            for which, qid, target in (("true", f.object_true_qid, new_true),
                                       ("fake", f.object_fake_qid, new_fake)):
                obj = f.object_true if which == "true" else f.object_fake
                old = (f.aliases_true if which == "true" else f.aliases_fake).get(lang, [])
                kept = decisions.get((qid or obj[lang], lang), old)
                # The canonical label must survive whatever the review said.
                if obj[lang] not in kept:
                    kept = [obj[lang], *kept]
                removed += len(old) - len([a for a in old if a in kept])
                target[lang] = kept
        out.append(__import__("dataclasses").replace(
            f, aliases_true=new_true, aliases_fake=new_fake))

    write_facts(ROOT / args.facts, out)
    print(f"removed {removed} alias entries across {len(out)} facts")
    print(f"wrote {args.facts}")
    print("\nNext: ./run.sh gate aliases")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="data/facts.jsonl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    sub.add_parser("apply").set_defaults(fn=cmd_apply)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
