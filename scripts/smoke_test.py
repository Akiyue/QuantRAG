"""First contact between the real weights and the pipeline.

    python scripts/smoke_test.py

Runs a handful of facts through one model at every precision and prints what
came back. Everything up to now has been exercised with a mock backend, so this
is the first time a real tokenizer, real log-probabilities and the answer
classifier meet each other.

Three things can only fail here:

  * the tokenizer merges across the prompt/answer boundary, which invalidates
    scores rather than merely degrading them
  * the model answers in a form the alias lists do not cover, so correct
    answers land in OTHER and every rate is quietly wrong
  * the scored preference and the generated answer disagree, which would mean
    the two measurements are not tracking the same thing

Cheap to run, and each of these is expensive to discover on a full grid.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantrag.backends import load_backend  # noqa: E402
from quantrag.normalize import Label  # noqa: E402
from quantrag.runner import run  # noqa: E402
from quantrag.schema import read_facts  # noqa: E402
from quantrag.analysis import index, load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

GREEN, RED, YELLOW, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-0.5b")
    ap.add_argument("--facts", default="data/facts.sample.jsonl")
    ap.add_argument("--out", default="runs/smoke")
    ap.add_argument("--keep", action="store_true", help="do not delete the output")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    model = next((m for m in cfg["models"] if m["id"] == args.model), None)
    if model is None:
        sys.exit(f"{args.model} not in configs/models.yaml")

    facts = read_facts(ROOT / args.facts)
    out_dir = ROOT / args.out
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    variants = [v for v in model["variants"] if v.get("tier", "A") == "A"]
    missing = [v["path"] for v in variants if not (ROOT / v["path"]).exists()]
    if missing:
        sys.exit("missing GGUF files - run ./run.sh models first:\n  "
                 + "\n  ".join(missing))

    print(f"{len(facts)} facts x {len(variants)} precisions\n")
    total_errors = 0
    for v in variants:
        backend = load_backend({**v, "model_id": model["id"],
                                **{k: cfg["runtime"][k]
                                   for k in ("n_ctx", "seed", "n_gpu_layers")
                                   if k in cfg["runtime"]}})
        stats = run(backend, facts, out_dir / f"main__{model['id']}__{v['precision']}.jsonl",
                    conditions=("C0", "C1", "C2"), progress=False)
        total_errors += stats["errors"]
        print(f"  {v['precision']:<8} {stats['written']:>4} cells, "
              f"{stats['errors']} errors")

    records = load(out_dir, "main")
    print("\n--- what the model actually said (C2, counterfactual document) ---")
    for r in records:
        if r.kind == "generate" and r.key.condition == "C2":
            text = r.text.replace("\n", " ")[:44]
            print(f"  {r.precision:<8} {r.key.lang}  {r.key.mode[:6]:<6} "
                  f"{r.label.value if r.label else '?':<8} {text!r}")

    # -- verdicts -------------------------------------------------------
    print("\n--- checks ---")
    ok = True

    if total_errors:
        print(f"{RED}FAIL{OFF} {total_errors} boundary errors. Those cells are "
              f"unusable, not merely imprecise - the prompt template needs fixing.")
        ok = False
    else:
        print(f"{GREEN}PASS{OFF} no tokenizer boundary errors")

    labels = Counter(r.label.value for r in records
                     if r.kind == "generate" and r.label)
    other = labels.get("OTHER", 0)
    named = sum(labels.get(k, 0) for k in ("TRUE", "FAKE"))
    print(f"     labels: {dict(labels)}")
    if named == 0:
        print(f"{RED}FAIL{OFF} nothing classified as TRUE or FAKE. Either the "
              f"model is answering in an unexpected form or the alias lists "
              f"do not cover it - check the transcript above.")
        ok = False
    elif other > named:
        print(f"{YELLOW}WARN{OFF} more OTHER than recognised answers. Usually "
              f"incomplete aliases rather than a bad model.")
    else:
        print(f"{GREEN}PASS{OFF} answers land on the candidates")

    gen, sco = index(records, "generate"), index(records, "score")
    agree = tot = 0
    for k, per_prec in gen.items():
        for prec, g in per_prec.items():
            s = sco.get(k, {}).get(prec)
            if not s or s.lp_fake is None or not g.label:
                continue
            if g.label not in (Label.TRUE, Label.FAKE):
                continue
            tot += 1
            agree += (Label.FAKE if s.lp_fake > s.lp_true else Label.TRUE) is g.label
    if tot:
        rate = agree / tot
        tag = GREEN + "PASS" if rate >= 0.9 else YELLOW + "WARN"
        print(f"{tag}{OFF} generation vs scored argmax: {rate:.2f} ({agree}/{tot})")
        if rate < 0.9:
            print("     below 0.90 - the two measurements are not tracking each "
                  "other; explain why before trusting either")

    if not args.keep:
        shutil.rmtree(out_dir)

    print()
    if ok:
        print(f"{GREEN}Pipeline works with real weights.{OFF} Next: ./run.sh dataset")
    else:
        print(f"{RED}Fix the above before building the dataset.{OFF}")
        sys.exit(1)


if __name__ == "__main__":
    main()
