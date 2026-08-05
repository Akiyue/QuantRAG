"""Produce the paper's tables from runs/.

    python scripts/analyze.py

Writes results/*.md and results/*.csv. Nothing here invents a number that the
protocol did not define; where a quantity is undefined for a condition it is
omitted rather than filled in.

Read results/00_margin_control.md first. It is the falsification test: if flips
are fully explained by how close the baseline already was to the decision
boundary, the central claim is not about quantization at all.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantrag.analysis import (  # noqa: E402
    BASELINE_PRECISION, index, labelled, load, load_scope, models, paired,
    precisions, scored,
)
from quantrag.metrics import (  # noqa: E402
    ccr, holm, margin_control, mcnemar, paired_bootstrap_ci, prr,
    quantization_flip_rate, wilcoxon_paired,
)
from quantrag.normalize import Label  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFLICT = "C2"

# Success means different things under the two instructions, which is the whole
# point of running both. Reporting one number for "accuracy" across them would
# average two opposite behaviours together.
SUCCESS = {"strict": Label.FAKE, "truth_seeking": Label.TRUE}


def fmt_ci(point: float, lo: float, hi: float) -> str:
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


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


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------------------------ sections

def section_margin(records, scope, out: Path) -> None:
    """The falsification test. Run and read this before anything else."""
    rows, csv_rows = [], []
    for model in models(records):
        for prec in precisions(records):
            if prec == BASELINE_PRECISION:
                continue
            items = paired(records, model, prec, condition=CONFLICT, scope=scope)
            usable = [i for i in items
                      if i.baseline_r is not None
                      and i.baseline_label is not None and i.quant_label is not None]
            if len(usable) < 20:
                continue
            flips = [i.baseline_label is not i.quant_label for i in usable]
            mc = margin_control([i.baseline_r for i in usable], flips)
            auc = mc["auc_margin_predicts_flip"]
            verdict = ("boundary noise - CLAIM FAILS" if auc >= 0.85
                       else "mostly margin" if auc >= 0.70
                       else "not reducible to margin")
            rows.append([model, prec, str(len(usable)),
                         f"{sum(flips) / len(flips):.3f}", f"{auc:.3f}", verdict])
            csv_rows.append([model, prec, len(usable), sum(flips), auc])

    write_md(
        out / "00_margin_control.md",
        "Margin control (falsification test)",
        "Can flips be explained away as items that already sat near the decision "
        "boundary at F16? AUC is for predicting a flip from the baseline margin "
        "|R_F16| under the conflict condition.\n\n"
        "**AUC near 1.0**: quantization did nothing structural - items merely "
        "jittered across a boundary they were already on, and the central claim "
        "collapses into a study of margin sensitivity.\n"
        "**AUC near 0.5**: flips are not reducible to baseline margin.",
        ["model", "precision", "n", "flip rate", "AUC", "verdict"], rows)
    write_csv(out / "00_margin_control.csv",
              ["model", "precision", "n", "flips", "auc"], csv_rows)


def section_behaviour(records, scope, out: Path) -> dict[str, float]:
    """CCR, PRR and mode-appropriate success, with paired CIs and McNemar."""
    rows, csv_rows = [], []
    pvals: dict[str, float] = {}

    for model in models(records):
        for lang in ("en", "vi"):
            for mode in ("strict", "truth_seeking"):
                for prec in precisions(records):
                    items = paired(records, model, prec, condition=CONFLICT,
                                   mode=mode, lang=lang, scope=scope)
                    base_lab, quant_lab = labelled(items)
                    if len(base_lab) < 20:
                        continue

                    target = SUCCESS[mode]
                    base_ok = [l is target for l in base_lab]
                    quant_ok = [l is target for l in quant_lab]

                    c = paired_bootstrap_ci([l is Label.FAKE for l in quant_lab])
                    p = paired_bootstrap_ci([l is Label.TRUE for l in quant_lab])
                    s = paired_bootstrap_ci(quant_ok)

                    if prec == BASELINE_PRECISION:
                        stat = "-"
                    else:
                        mn = mcnemar(base_ok, quant_ok)
                        tag = f"{model}|{lang}|{mode}|{prec}|mcnemar"
                        pvals[tag] = mn["p_value"]
                        stat = f"p={mn['p_value']:.3g} OR={mn['odds_ratio']:.2f}"

                    rows.append([model, lang, mode, prec, str(len(base_lab)),
                                 fmt_ci(*c), fmt_ci(*p), fmt_ci(*s), stat])
                    csv_rows.append([model, lang, mode, prec, len(base_lab),
                                     ccr(quant_lab), prr(quant_lab), s[0]])

    write_md(
        out / "01_behaviour.md",
        "Context compliance and parametric retention under conflict",
        "Condition C2: the retrieved document asserts the counterfactual object. "
        "Scope is the parametric-known subset. Brackets are paired bootstrap 95% CIs "
        "over items.\n\n"
        "**CCR and PRR must never be read without the instruction mode.** Under "
        "strict grounding, following the document is compliance with the "
        "instruction; under truth-seeking, the same behaviour is being steered by "
        "misinformation. 'success' applies the mode-appropriate criterion.\n\n"
        "McNemar compares each precision against F16 on the same items.",
        ["model", "lang", "mode", "precision", "n", "CCR", "PRR", "success",
         "vs F16"], rows)
    write_csv(out / "01_behaviour.csv",
              ["model", "lang", "mode", "precision", "n", "ccr", "prr", "success"],
              csv_rows)
    return pvals


def section_flips(records, scope, out: Path) -> None:
    rows, csv_rows = [], []
    for model in models(records):
        for lang in ("en", "vi"):
            for mode in ("strict", "truth_seeking"):
                for prec in precisions(records):
                    if prec == BASELINE_PRECISION:
                        continue
                    items = paired(records, model, prec, condition=CONFLICT,
                                   mode=mode, lang=lang, scope=scope)
                    base_lab, quant_lab = labelled(items)
                    if len(base_lab) < 20:
                        continue
                    fr = quantization_flip_rate(base_lab, quant_lab)
                    b = fr.breakdown
                    rows.append([
                        model, lang, mode, prec, str(fr.n), f"{fr.qfr:.3f}",
                        str(b.get("true_to_fake", 0)), str(b.get("fake_to_true", 0)),
                        str(b.get("answer_to_refusal", 0)),
                        str(b.get("answer_to_other", 0)),
                        f"{fr.asymmetry():+.3f}",
                    ])
                    csv_rows.append([model, lang, mode, prec, fr.n, fr.qfr,
                                     b.get("true_to_fake", 0), b.get("fake_to_true", 0),
                                     fr.asymmetry()])

    write_md(
        out / "02_flips.md",
        "Instance-level instability under quantization",
        "Reported as an extension of *correctness agreement* (arXiv 2607.08734) "
        "into the context-memory conflict setting - the bare observation that "
        "individual responses change while aggregates hold is prior work and is "
        "cited as such.\n\n"
        "**Asymmetry is the directional claim.** Balanced flips in both directions "
        "and a one-sided shift give the same QFR; only the asymmetry distinguishes "
        "them, and it survives a null net effect.",
        ["model", "lang", "mode", "precision", "n", "QFR", "true→fake",
         "fake→true", "→refusal", "→other", "asymmetry"], rows)
    write_csv(out / "02_flips.csv",
              ["model", "lang", "mode", "precision", "n", "qfr", "true_to_fake",
               "fake_to_true", "asymmetry"], csv_rows)


def section_reliance(records, scope, out: Path) -> dict[str, float]:
    rows, csv_rows = [], []
    pvals: dict[str, float] = {}

    for model in models(records):
        for lang in ("en", "vi"):
            for mode in ("strict", "truth_seeking"):
                for prec in precisions(records):
                    if prec == BASELINE_PRECISION:
                        continue
                    items = paired(records, model, prec, condition=CONFLICT,
                                   mode=mode, lang=lang, scope=scope)
                    base_r, quant_r = scored(items)
                    if len(base_r) < 20:
                        continue

                    deltas = [q - b for b, q in zip(base_r, quant_r)]
                    d = paired_bootstrap_ci(deltas)
                    w = wilcoxon_paired(quant_r, base_r)
                    pvals[f"{model}|{lang}|{mode}|{prec}|wilcoxon_R"] = w["p_value"]

                    pc = [(i.baseline_p_ctx, i.quant_p_ctx) for i in items
                          if i.baseline_p_ctx is not None and i.quant_p_ctx is not None]
                    dp = paired_bootstrap_ci([q - b for b, q in pc]) if pc else (
                        float("nan"),) * 3

                    rows.append([model, lang, mode, prec, str(len(base_r)),
                                 fmt_ci(*d), fmt_ci(*dp),
                                 f"p={w['p_value']:.3g} r={w['rank_biserial']:+.3f}"])
                    csv_rows.append([model, lang, mode, prec, len(base_r), d[0],
                                     dp[0], w["p_value"], w["rank_biserial"]])

    write_md(
        out / "03_reliance.md",
        "Shift in context-memory arbitration",
        "R = s(context answer) - s(parametric answer) on length-normalised "
        "log-probabilities; Delta R is the paired change from F16.\n\n"
        "R > 0 leans toward the retrieved context, R < 0 toward parametric memory.\n\n"
        "**Delta P_ctx is the metric of record.** Quantization changes how peaked "
        "the output distribution is, so a raw Delta R can be moved by a global "
        "sharpening rather than by any change in which answer is preferred; the "
        "renormalised two-way form is far less sensitive to that. Check "
        "05_diagnostics.md before reading either.",
        ["model", "lang", "mode", "precision", "n", "ΔR", "ΔP_ctx", "Wilcoxon"],
        rows)
    write_csv(out / "03_reliance.csv",
              ["model", "lang", "mode", "precision", "n", "delta_r", "delta_p_ctx",
               "p_value", "rank_biserial"], csv_rows)
    return pvals


def section_interactions(records, scope, out: Path) -> None:
    """Where the signal most likely is if the net effect is null."""
    rows = []
    for model in models(records):
        for prec in precisions(records):
            if prec == BASELINE_PRECISION:
                continue
            # precision x mode: does the gap between instructions narrow?
            gaps = {}
            for mode in ("strict", "truth_seeking"):
                items = paired(records, model, prec, condition=CONFLICT, mode=mode,
                               scope=scope)
                base_lab, quant_lab = labelled(items)
                if len(base_lab) >= 20:
                    gaps[mode] = (ccr(base_lab), ccr(quant_lab))
            if len(gaps) == 2:
                b = gaps["strict"][0] - gaps["truth_seeking"][0]
                q = gaps["strict"][1] - gaps["truth_seeking"][1]
                rows.append([model, prec, "mode gap (CCR strict - truth)",
                             f"{b:.3f}", f"{q:.3f}", f"{q - b:+.3f}"])

            # precision x language
            langs = {}
            for lang in ("en", "vi"):
                items = paired(records, model, prec, condition=CONFLICT, lang=lang,
                               scope=scope)
                br, qr = scored(items)
                if len(br) >= 20:
                    langs[lang] = sum(q - b for b, q in zip(br, qr)) / len(br)
            if len(langs) == 2:
                rows.append([model, prec, "ΔR (en) vs ΔR (vi)",
                             f"{langs['en']:+.3f}", f"{langs['vi']:+.3f}",
                             f"{langs['vi'] - langs['en']:+.3f}"])

    write_md(
        out / "04_interactions.md",
        "Interaction effects",
        "If the net shift is null - which the prior literature makes likely - this "
        "is where the result lives. Both rows are pre-registered hypotheses "
        "(PROPOSAL H3, H4).\n\n"
        "The language row tests a live disagreement: arXiv 2407.03211 reports "
        "quantization hurting non-Latin scripts disproportionately, while arXiv "
        "2503.03592 reports k-quantization *not* disproportionately harming "
        "multilingual performance. Vietnamese is Latin-script but heavily "
        "diacritised, sitting between the two.",
        ["model", "precision", "quantity", "F16", "quantized", "change"], rows)


def section_diagnostics(records, scope, out: Path) -> None:
    """Checks that decide whether the other tables mean anything."""
    lines = ["# Diagnostics", "",
             "Read before interpreting any result table.", ""]

    # Generation vs teacher-forced argmax. The plan requires this: if the two
    # disagree often they are measuring different things.
    gen, sco = index(records, "generate"), index(records, "score")
    agree = total = 0
    for k, per_prec in gen.items():
        for prec, g in per_prec.items():
            s = sco.get(k, {}).get(prec)
            if s is None or s.lp_fake is None or g.label is None:
                continue
            if g.label not in (Label.TRUE, Label.FAKE):
                continue
            predicted = Label.FAKE if s.lp_fake > s.lp_true else Label.TRUE
            agree += predicted is g.label
            total += 1
    rate = agree / total if total else float("nan")
    lines += ["## Generation vs teacher-forced argmax", "",
              f"Agreement: **{rate:.3f}** ({agree}/{total})", ""]
    if total and rate < 0.90:
        lines += ["> Below 0.90. The scored preference and the generated answer are "
                  "not tracking each other, so they are measuring different things. "
                  "Explain why before interpreting either.", ""]

    # Ambiguous generations, which are the manual-review target.
    amb = sum(1 for r in records if r.kind == "generate" and r.both_present)
    ngen = sum(1 for r in records if r.kind == "generate")
    lines += ["## Answers mentioning both candidates", "",
              f"{amb}/{ngen} generations contain both the true and the "
              f"counterfactual object. These are where the evaluator is most "
              f"likely to be wrong; the 200-item manual validation should "
              f"oversample them.", ""]

    # Label distribution, to catch a collapsed arm early.
    lines += ["## Label distribution by precision", "",
              "| precision | TRUE | FAKE | REFUSAL | OTHER |",
              "|---|---|---|---|---|"]
    by_prec: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        if r.kind == "generate" and r.label is not None:
            by_prec[r.precision][r.label.value] += 1
    for prec in precisions(records):
        c = by_prec.get(prec, Counter())
        lines.append(f"| {prec} | {c['TRUE']} | {c['FAKE']} | {c['REFUSAL']} "
                     f"| {c['OTHER']} |")
    lines += ["", "> A large OTHER count usually means the alias lists are "
              "incomplete rather than that the model failed - check before "
              "reporting anything.", ""]

    (out / "05_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {show(out / '05_diagnostics.md')}")


def section_holm(pvals: dict[str, float], out: Path) -> None:
    if not pvals:
        return
    adjusted = holm(pvals)
    rows = [[k, f"{v['p_raw']:.3g}", f"{v['p_holm']:.3g}",
             "yes" if v["reject"] else "no"]
            for k, v in sorted(adjusted.items(), key=lambda kv: kv[1]["p_holm"])]
    write_md(
        out / "06_multiple_comparisons.md",
        "Holm-Bonferroni across the test family",
        "Every model x language x mode x precision test is in one family. "
        "Reporting raw p-values across this many comparisons would manufacture "
        "significance.",
        ["test", "p raw", "p Holm", "reject at 0.05"], rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--split", default="known_all",
                    choices=["known_all", "known_fp16", "unknown"])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    records = load(ROOT / args.runs, "main")
    if not records:
        sys.exit(f"no main-pass records in {args.runs}; run `./run.sh main` first")

    splits_path = ROOT / args.splits
    scope = None
    if splits_path.exists():
        scope = load_scope(splits_path, args.split)
        print(f"scope: {args.split}, {len(scope)} (fact, lang) items")
    else:
        print(f"WARNING: {args.splits} missing - analysing ALL items unscoped.")
        print("  Primary results must be scoped to the parametric-known subset:")
        print("  a model that never knew the fact cannot be said to have chosen")
        print("  between context and memory. Run ./run.sh filter first.")

    out = ROOT / args.out / args.split
    print(f"\nwriting to {out.relative_to(ROOT)}")

    section_margin(records, scope, out)
    pvals = section_behaviour(records, scope, out)
    section_flips(records, scope, out)
    pvals |= section_reliance(records, scope, out)
    section_interactions(records, scope, out)
    section_diagnostics(records, scope, out)
    section_holm(pvals, out)

    print("\nRead 00_margin_control.md and 05_diagnostics.md before the rest.")


if __name__ == "__main__":
    main()
