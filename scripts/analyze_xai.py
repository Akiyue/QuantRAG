"""Tables for the causal attribution results (RQ4).

    python scripts/analyze_xai.py

Answers two questions the behavioural tables cannot:

  * Does the evidence span causally support the answer, over and above the
    effect of simply deleting text?
  * Does attribution predict which items flip - and does it still do so once
    baseline margin is controlled for? If A* only predicts flips because both
    track how close the item was to the boundary, it adds nothing.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantrag.analysis import BASELINE_PRECISION, load  # noqa: E402
from quantrag.metrics import auc, paired_bootstrap_ci  # noqa: E402
from quantrag.normalize import Label  # noqa: E402
from quantrag.schema import iter_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_xai(run_dir: Path) -> list[dict]:
    out = []
    for path in sorted(run_dir.glob("xai__*.jsonl")):
        out.extend(r for r in iter_jsonl(path) if "error" not in r)
    return out


def show(path: Path) -> str:
    """Path for display; --out may legitimately point outside the repo."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_md(path: Path, title: str, note: str, header: list[str],
             rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", note, "",
             "| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {show(path)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/xai")
    args = ap.parse_args()

    recs = load_xai(ROOT / args.runs)
    if not recs:
        sys.exit(f"no xai records in {args.runs}; run `./run.sh xai` first")
    out = ROOT / args.out
    print(f"{len(recs)} ablation records")

    # -- attribution by precision --------------------------------------
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        groups[(r["model_id"], r["lang"], r["mode"], r["precision"])].append(r)

    rows = []
    for k in sorted(groups):
        g = groups[k]
        a_ev = paired_bootstrap_ci([r["A_evidence"] for r in g])
        a_st = paired_bootstrap_ci([r["A_star_evidence"] for r in g])
        a_q = statistics.fmean(r["A_question"] for r in g)
        a_i = statistics.fmean(r["A_instruction"] for r in g)
        dlen = statistics.fmean(r.get("control_len_delta", 0) for r in g)
        rows.append([*k, str(len(g)),
                     f"{a_ev[0]:.3f} [{a_ev[1]:.3f}, {a_ev[2]:.3f}]",
                     f"{a_st[0]:.3f} [{a_st[1]:.3f}, {a_st[2]:.3f}]",
                     f"{a_q:.3f}", f"{a_i:.3f}", f"{dlen:+.1f}"])

    write_md(
        out / "10_attribution.md",
        "Span-level causal attribution",
        "A(span) is the drop in log-probability of the model's own answer when "
        "that span is removed. A\\*(evidence) subtracts a length-matched, "
        "information-free control document.\n\n"
        "**Read A\\*, not A.** Deleting any tokens lowers the log-probability, so "
        "A(evidence) largely reflects how much text disappeared. Only the excess "
        "over the control is attributable to the evidence carrying the answer. "
        "'ctrl Δlen' is the mean token-length difference between the real and the "
        "control document - the closer to zero, the better matched the control.",
        ["model", "lang", "mode", "precision", "n", "A(evidence)",
         "A*(evidence)", "A(question)", "A(instruction)", "ctrl Δlen"], rows)

    # -- semantic gap ---------------------------------------------------
    rows = []
    for k in sorted(groups):
        g = [r for r in groups[k] if "semantic_gap" in r]
        if not g:
            continue
        sg = paired_bootstrap_ci([r["semantic_gap"] for r in g])
        rows.append([*k, str(len(g)), f"{sg[0]:.3f} [{sg[1]:.3f}, {sg[2]:.3f}]"])

    if rows:
        write_md(
            out / "11_semantic_gap.md",
            "Semantic grounding versus surface copying",
            "SemanticGap = R(evidence) - R(paraphrased evidence). The paraphrase "
            "keeps the proposition and rewrites the surface form entirely.\n\n"
            "Near zero: the model is grounding on meaning. Large and positive: it "
            "was leaning on the specific wording, i.e. copying a string. If this "
            "grows as precision drops, context grounding is degrading from "
            "semantic to surface - a mechanism-flavoured claim reachable with "
            "input-level interventions alone (PROPOSAL H6).",
            ["model", "lang", "mode", "precision", "n", "SemanticGap"], rows)

    # -- does attribution predict flips? --------------------------------
    behaviour = load(ROOT / args.runs, "main")
    labels: dict[tuple, Label] = {}
    for r in behaviour:
        if r.kind == "generate" and r.key.condition == "C2" and r.label:
            labels[(r.model_id, r.precision, r.key.fact_id, r.key.lang,
                    r.key.mode)] = r.label

    rows = []
    by_arm: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        by_arm[(r["model_id"], r["precision"])].append(r)

    for (model, prec), g in sorted(by_arm.items()):
        if prec == BASELINE_PRECISION:
            continue
        a_star, margins, flips = [], [], []
        for r in g:
            base = labels.get((model, BASELINE_PRECISION, r["fact_id"],
                               r["lang"], r["mode"]))
            quant = labels.get((model, prec, r["fact_id"], r["lang"], r["mode"]))
            if base is None or quant is None:
                continue
            a_star.append(r["A_star_evidence"])
            margins.append(abs(r["r_full"]))
            flips.append(base is not quant)
        if len(flips) < 20 or not any(flips) or all(flips):
            continue
        rows.append([model, prec, str(len(flips)),
                     f"{auc(a_star, flips):.3f}",
                     f"{auc([-m for m in margins], flips):.3f}"])

    if rows:
        write_md(
            out / "12_attribution_vs_flips.md",
            "Does attribution predict instability?",
            "AUC for predicting a flip from A\\*(evidence), alongside the AUC for "
            "predicting it from baseline margin alone.\n\n"
            "**Compare the two columns.** If margin alone predicts flips just as "
            "well, attribution is carrying no independent information and RQ4 has "
            "a negative answer - which is a reportable result, not a failure. "
            "Attribution earns its place only by beating the margin column.",
            ["model", "precision", "n", "AUC from A*", "AUC from margin"], rows)

    print("\nCompare 12_attribution_vs_flips.md against results/*/00_margin_control.md")


if __name__ == "__main__":
    main()
