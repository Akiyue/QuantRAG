"""Span-level causal ablation on a stratified subset.

    python scripts/run_ablation.py --n 100

For each sampled item and each (model, precision, language) we fix y as the
answer the model prefers under the full prompt, then re-score y with one span
removed:

    A(s) = log P(y | x) - log P(y | x \\ s)

and correct the evidence term against a length-matched, information-free
control document:

    A*(evidence) = A(evidence) - A(control)

The correction is not optional. Deleting any tokens lowers the log-probability,
so a bare A(evidence) mostly measures how much text was removed. Only the
excess over the control is attributable to the evidence carrying the answer.

The same C2_para condition is scored here so SemanticGap can be computed:
reliance that survives a full rewording is semantic, reliance that collapses was
surface copying.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantrag.analysis import load  # noqa: E402
from quantrag.backends import load_backend  # noqa: E402
from quantrag.backends.base import BoundaryError  # noqa: E402
from quantrag.prompts import build_prompt, candidates_for  # noqa: E402
from quantrag.schema import Fact, append_jsonl, iter_jsonl, read_facts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFLICT = "C2"


def stratified_sample(facts: list[Fact], n: int, runs: Path,
                      seed: int) -> list[Fact]:
    """Balance across relations and, where known, across flip behaviour.

    Sampling only stable items would make the attribution analysis describe the
    cases where nothing happened. Flips are the phenomenon, so they have to be
    represented even though they are the minority.
    """
    rng = random.Random(seed)
    by_rel: dict[str, list[Fact]] = defaultdict(list)
    for f in facts:
        by_rel[f.relation_id or f.relation].append(f)

    flipped: set[str] = set()
    try:
        records = load(runs, "main")
        by_item: dict[tuple[str, str], set] = defaultdict(set)
        for r in records:
            if r.kind == "generate" and r.key.condition == CONFLICT and r.label:
                by_item[(r.key.fact_id, r.precision)].add(r.label)
        seen: dict[str, set] = defaultdict(set)
        for (fid, _prec), labels in by_item.items():
            seen[fid] |= labels
        flipped = {fid for fid, labels in seen.items() if len(labels) > 1}
        if flipped:
            print(f"  {len(flipped)} facts flipped somewhere in the main pass")
    except Exception:  # noqa: BLE001 - stratification is best-effort
        print("  no main-pass results; stratifying by relation only")

    per_rel = max(1, n // max(1, len(by_rel)))
    chosen: list[Fact] = []
    for rel, group in sorted(by_rel.items()):
        flip_group = [f for f in group if f.fact_id in flipped]
        stable = [f for f in group if f.fact_id not in flipped]
        rng.shuffle(flip_group)
        rng.shuffle(stable)
        # Half the quota to flips where they exist, so both behaviours appear.
        take_flip = min(len(flip_group), per_rel // 2)
        picked = flip_group[:take_flip]
        picked += stable[: per_rel - len(picked)]
        chosen.extend(picked[:per_rel])

    rng.shuffle(chosen)
    return chosen[:n]


def ablate(backend, fact: Fact, lang: str, mode: str) -> dict:
    """All prompts needed for one item, scored against a fixed y."""
    cands = candidates_for(fact, CONFLICT, lang)
    conts = cands.as_continuations()

    full = build_prompt(fact, lang=lang, mode=mode, condition=CONFLICT)
    s_fake, s_true = backend.score(full, conts)

    # y is what the model actually prefers here. Attribution asks how much each
    # span supports *this* answer, so it must be held fixed while spans change.
    y = cands.fake if s_fake.mean_logprob >= s_true.mean_logprob else cands.true
    y_full = max(s_fake, s_true, key=lambda s: s.mean_logprob).mean_logprob

    out: dict[str, float] = {}
    for span in ("instruction", "evidence", "question"):
        p = build_prompt(fact, lang=lang, mode=mode, condition=CONFLICT,
                         drop_spans=(span,))
        out[f"A_{span}"] = y_full - backend.score(p, [y])[0].mean_logprob

    # Control: same subject, same template family, no answer in it. Better
    # length-matched than a generic filler would be.
    ctrl_doc = fact.evidence_irrelevant[lang]
    p_ctrl = build_prompt(fact, lang=lang, mode=mode, documents=[ctrl_doc])
    out["A_control"] = y_full - backend.score(p_ctrl, [y])[0].mean_logprob
    out["A_star_evidence"] = out["A_evidence"] - out["A_control"]

    # Token-length delta between the real and the control document, so a reader
    # can judge how well matched the control actually was.
    ev_len = backend.score(full, [fact.evidence_fake[lang]])[0].n_tokens
    ct_len = backend.score(full, [ctrl_doc])[0].n_tokens
    out["control_len_delta"] = ct_len - ev_len

    result = {
        "fact_id": fact.fact_id, "relation": fact.relation, "lang": lang,
        "mode": mode, "y": y, "y_is_context": y == cands.fake,
        "r_full": s_fake.mean_logprob - s_true.mean_logprob,
        **out,
    }

    # SemanticGap needs the paraphrased counterfactual scored the same way.
    if fact.evidence_fake_para.get(lang):
        p_para = build_prompt(fact, lang=lang, mode=mode, condition="C2_para")
        pf, pt = backend.score(p_para, conts)
        r_para = pf.mean_logprob - pt.mean_logprob
        result["r_para"] = r_para
        result["semantic_gap"] = result["r_full"] - r_para
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="data/facts.jsonl")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--tier", default="A")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    facts = read_facts(ROOT / args.facts)
    exp = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    models_cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))

    print(f"1. sampling {args.n} of {len(facts)} facts")
    sample = stratified_sample(facts, args.n, ROOT / args.runs, args.seed)
    ids = [f.fact_id for f in sample]
    (ROOT / "data" / "xai_sample.json").write_text(
        __import__("json").dumps(ids, indent=1), encoding="utf-8")
    print(f"  {len(sample)} facts -> data/xai_sample.json")
    print("  the SAME items are used for every model, precision and language")

    specs = [{"backend": "mock", "model_id": "mock", "precision": "MOCK"}] if args.mock else [
        {**v, "model_id": m["id"], **{k: models_cfg["runtime"][k]
                                      for k in ("n_ctx", "seed", "n_gpu_layers")
                                      if k in models_cfg["runtime"]}}
        for m in models_cfg["models"] if not m.get("optional")
        if not args.models or m["id"] in args.models
        for v in m["variants"] if v.get("tier", "A") == args.tier
    ]

    langs = exp["grid"]["languages"]
    modes = exp["grid"]["modes"]
    out_dir = ROOT / exp["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        backend = load_backend(spec)
        path = out_dir / f"xai__{backend.model_id}__{backend.precision}.jsonl"
        done = {r["key"] for r in iter_jsonl(path) if "key" in r}
        print(f"\n-> {path.name} ({len(done)} already done)")

        for fact in sample:
            for lang in langs:
                for mode in modes:
                    key = f"{backend.model_id}|{backend.precision}|{fact.fact_id}|{lang}|{mode}"
                    if key in done:
                        continue
                    try:
                        rec = ablate(backend, fact, lang, mode)
                    except BoundaryError as exc:
                        append_jsonl(path, {"key": key, "error": "boundary",
                                            "detail": str(exc)})
                        continue
                    rec |= {"key": key, "model_id": backend.model_id,
                            "precision": backend.precision}
                    append_jsonl(path, rec)
        print("   done")

    print("\nNext: scripts/analyze_xai.py")


if __name__ == "__main__":
    main()
