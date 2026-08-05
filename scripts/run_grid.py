"""CLI driver for the experimental grid.

    python scripts/run_grid.py --pass main
    python scripts/run_grid.py --pass filter --limit 50
    python scripts/run_grid.py --pass dose --models qwen2.5-1.5b

Each (model, precision) variant writes its own resumable JSONL under runs/.
Re-running is safe and cheap: completed cells are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from quantrag.backends import load_backend
from quantrag.runner import run
from quantrag.schema import read_facts

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def variants(models_cfg: dict, only: list[str] | None, tier: str) -> list[dict]:
    out: list[dict] = []
    for model in models_cfg["models"]:
        if only and model["id"] not in only:
            continue
        if model.get("optional") and not only:
            continue  # upper reference point is opt-in
        for var in model["variants"]:
            v_tier = var.get("tier", "A")
            if tier != "all" and v_tier != tier:
                continue
            out.append({**var, "model_id": model["id"],
                        **{k: v for k, v in models_cfg["runtime"].items()
                           if k in ("n_ctx", "seed", "n_gpu_layers")}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_name", required=True,
                    choices=["filter", "main", "dose"])
    ap.add_argument("--facts", default="data/facts.jsonl")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--tier", default="A", choices=["A", "B", "all"])
    ap.add_argument("--limit", type=int, default=0, help="first N facts (pilot)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--mock", action="store_true",
                    help="MockBackend: exercises the plumbing, produces no science")
    args = ap.parse_args()

    models_cfg = load_yaml(ROOT / "configs" / "models.yaml")
    exp = load_yaml(ROOT / "configs" / "experiment.yaml")

    facts_path = ROOT / args.facts
    if not facts_path.exists():
        sys.exit(f"missing {facts_path}; run `./run.sh dataset` first")
    facts = read_facts(facts_path)
    if args.limit:
        facts = facts[: args.limit]

    bad = [(f.fact_id, f.validate()) for f in facts if f.validate()]
    if bad:
        for fid, problems in bad[:10]:
            print(f"  {fid}: {'; '.join(problems)}", file=sys.stderr)
        sys.exit(f"{len(bad)} invalid facts; fix the dataset before spending GPU time")

    out_dir = Path(args.out_dir or exp["output"]["dir"])
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pass_name == "filter":
        section, kwargs = exp["filter_pass"], {}
    elif args.pass_name == "main":
        section, kwargs = exp["grid"], {}
    else:
        d = exp["dose_response"]
        section = {"languages": d["languages"], "modes": d["modes"],
                   "conditions": [], "kinds": ["score"], "question_kind": "q_eval"}
        kwargs = {"doses": d["doses"]}

    specs = [{"backend": "mock", "model_id": "mock", "precision": "MOCK"}] if args.mock \
        else variants(models_cfg, args.models, args.tier)
    if not specs:
        sys.exit("no model variants selected")

    max_tokens = models_cfg["runtime"].get("max_tokens", 32)
    strip = exp["evaluation"].get("strip_diacritics", False)

    summary: list[dict] = []
    for spec in specs:
        backend = load_backend(spec)
        out = out_dir / f"{args.pass_name}__{backend.model_id}__{backend.precision}.jsonl"
        print(f"\n-> {out.name}")
        stats = run(
            backend, facts, out,
            languages=section["languages"],
            conditions=section["conditions"],
            modes=section["modes"],
            kinds=section["kinds"],
            question_kind=section.get("question_kind", "q_eval"),
            max_tokens=max_tokens,
            strip_diacritics=strip,
            **kwargs,
        )
        print(f"   {stats}")
        summary.append({"file": out.name, "model": backend.model_id,
                        "precision": backend.precision, **stats})

    manifest = out_dir / f"{args.pass_name}__manifest.json"
    manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    total_err = sum(s["errors"] for s in summary)
    print(f"\nwrote {manifest.name}; {total_err} boundary errors")
    if total_err:
        print("boundary errors invalidate those cells - inspect before analysing",
              file=sys.stderr)


if __name__ == "__main__":
    main()
