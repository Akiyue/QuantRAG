"""Print rendered prompts for eyeball inspection.

Run this before every full grid. Prompt bugs are invisible in aggregate numbers
and expensive to discover afterwards.

    python scripts/inspect_prompts.py [facts.jsonl] [fact_id]
"""

from __future__ import annotations

import sys

from quantrag.prompts import CONDITIONS, MODES, build_prompt, candidates_for
from quantrag.schema import read_facts


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/facts.sample.jsonl"
    facts = read_facts(path)
    fact = next((f for f in facts if f.fact_id == sys.argv[2]), facts[0]) \
        if len(sys.argv) > 2 else facts[0]

    for lang in ("en", "vi"):
        for cond in CONDITIONS:
            if cond == "C2_para" and not fact.evidence_fake_para.get(lang):
                continue
            for mode in MODES:
                if cond == "C0" and mode != MODES[0]:
                    continue
                cands = candidates_for(fact, cond, lang)
                print("=" * 72)
                print(f"{fact.fact_id} | {lang} | {cond} | {mode}")
                print(f"  scored answers : {cands.as_continuations()}")
                print(f"  document says  : {cands.context_answer or '(nothing)'}")
                print("-" * 72)
                print(build_prompt(fact, lang=lang, mode=mode, condition=cond))


if __name__ == "__main__":
    main()
