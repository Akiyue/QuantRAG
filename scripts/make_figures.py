"""Generate the paper's figures from runs/.

    python scripts/make_figures.py

Writes results/<split>/figures/*.pdf (for LaTeX) and *.png (for looking at).

Reads the run records directly rather than the summary CSVs, because a couple of
the figures need per-item detail - the margin deciles especially - that the
tables aggregate away.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from quantrag.analysis import (  # noqa: E402
    BASELINE_PRECISION, load, load_scope, labelled, models, paired, precisions,
    scored,
)
from quantrag.figures import (  # noqa: E402
    LINESTYLES, MARKERS, NEG, POS, SERIES, apply_style, ladder_color, save,
    zero_line,
)
from quantrag.metrics import margin_control, paired_bootstrap_ci, quantization_flip_rate  # noqa: E402
from quantrag.normalize import Label  # noqa: E402
from quantrag.schema import iter_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFLICT = "C2"
MIN_N = 20


# --------------------------------------------------------------- figure 1

def fig_reliance_ladder(recs, scope, out: Path) -> None:
    """Arbitration shift along the precision ladder.

    Precision is ordered, so it belongs on the x-axis as a ladder rather than
    as coloured categories. Identity is model (colour + marker) crossed with
    language (line style), which keeps the series distinguishable in greyscale.
    """
    import matplotlib.pyplot as plt

    precs = [p for p in precisions(recs) if p != BASELINE_PRECISION]
    if not precs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)

    for ax, mode in zip(axes, ("strict", "truth_seeking")):
        plotted = False
        for mi, model in enumerate(models(recs)):
            for lang in ("en", "vi"):
                xs, ys, los, his = [], [], [], []
                for pi, prec in enumerate(precs):
                    items = paired(recs, model, prec, condition=CONFLICT,
                                   mode=mode, lang=lang, scope=scope)
                    pc = [(i.baseline_p_ctx, i.quant_p_ctx) for i in items
                          if i.baseline_p_ctx is not None
                          and i.quant_p_ctx is not None]
                    if len(pc) < MIN_N:
                        continue
                    m, lo, hi = paired_bootstrap_ci([q - b for b, q in pc])
                    xs.append(pi); ys.append(m); los.append(m - lo); his.append(hi - m)
                if not xs:
                    continue
                plotted = True
                ax.errorbar(
                    xs, ys, yerr=[los, his],
                    color=SERIES[mi % len(SERIES)],
                    linestyle=LINESTYLES[lang],
                    marker=MARKERS[mi % len(MARKERS)],
                    markerfacecolor=SERIES[mi % len(SERIES)] if lang == "en" else "white",
                    markeredgecolor=SERIES[mi % len(SERIES)],
                    capsize=2, elinewidth=1.0,
                    label=f"{model} · {lang}",
                )
        zero_line(ax)
        ax.set_xticks(range(len(precs)))
        ax.set_xticklabels(precs)
        ax.set_title(f"{mode.replace('_', '-')}")
        if not plotted:
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                    transform=ax.transAxes)

    axes[0].set_ylabel(r"$\Delta P_{\mathrm{ctx}}$ vs F16")
    # One shared axis label, so the legend below it has room and does not
    # collide with per-panel labels.
    fig.supxlabel("precision", y=-0.02)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)),
                   frameon=False, bbox_to_anchor=(0.5, -0.22))
    fig.suptitle("Shift toward context under quantization", y=1.02)
    save(fig, out, "fig1_reliance_ladder")
    plt.close(fig)


# --------------------------------------------------------------- figure 2

def fig_flip_asymmetry(recs, scope, out: Path) -> None:
    """Signed flips: toward the counterfactual above zero, back to truth below.

    A stacked bar would hide the thing that matters. Equal flow in both
    directions and a one-sided shift give the same total; only the signed form
    separates them, and the asymmetry is the directional claim that survives a
    null net effect.
    """
    import matplotlib.pyplot as plt

    precs = [p for p in precisions(recs) if p != BASELINE_PRECISION]
    if not precs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)

    for ax, mode in zip(axes, ("strict", "truth_seeking")):
        labels_x, up, down = [], [], []
        for model in models(recs):
            for prec in precs:
                items = paired(recs, model, prec, condition=CONFLICT,
                               mode=mode, scope=scope)
                base, quant = labelled(items)
                if len(base) < MIN_N:
                    continue
                fr = quantization_flip_rate(base, quant)
                labels_x.append(f"{model.split('-')[-1]}\n{prec}")
                up.append(fr.breakdown.get("true_to_fake", 0) / fr.n)
                down.append(-fr.breakdown.get("fake_to_true", 0) / fr.n)

        if not labels_x:
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        x = np.arange(len(labels_x))
        # 2px surface gap between opposing fills so they never touch.
        ax.bar(x, up, color=POS, width=0.62, label="true → counterfactual",
               edgecolor="white", linewidth=0.8)
        ax.bar(x, down, color=NEG, width=0.62, label="counterfactual → true",
               edgecolor="white", linewidth=0.8)
        zero_line(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_x, fontsize=7)
        ax.set_title(mode.replace("_", "-"))

    axes[0].set_ylabel("share of items")
    handles, lbls = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, lbls, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Direction of quantization-induced flips", y=1.02)
    save(fig, out, "fig2_flip_asymmetry")
    plt.close(fig)


# --------------------------------------------------------------- figure 3

def fig_margin_control(recs, scope, out: Path) -> None:
    """The falsification test, drawn.

    If flip rate falls off steeply with baseline margin, the flips were items
    sitting on the decision boundary and quantization contributed nothing
    structural. A flat line is the result the paper needs.
    """
    import matplotlib.pyplot as plt

    precs = [p for p in precisions(recs) if p != BASELINE_PRECISION]
    mdls = models(recs)
    if not precs or not mdls:
        return

    # One panel per model. Colour encodes precision, so two models on one axis
    # would put the same colour on different entities - faceting is what keeps
    # colour meaning one thing.
    fig, axes = plt.subplots(1, len(mdls), figsize=(3.6 * len(mdls), 3.0),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, model in zip(axes, mdls):
        drew = False
        for prec in precs:
            items = paired(recs, model, prec, condition=CONFLICT, scope=scope)
            usable = [i for i in items if i.baseline_r is not None
                      and i.baseline_label is not None and i.quant_label is not None]
            if len(usable) < MIN_N:
                continue
            mc = margin_control([i.baseline_r for i in usable],
                                [i.baseline_label is not i.quant_label for i in usable])
            ax.plot([d["mean_margin"] for d in mc["by_decile"]],
                    [d["flip_rate"] for d in mc["by_decile"]],
                    color=ladder_color(prec),
                    marker=MARKERS[precs.index(prec) % len(MARKERS)],
                    label=f"{prec}  (AUC {mc['auc_margin_predicts_flip']:.2f})")
            drew = True
        ax.set_title(model)
        if drew:
            ax.legend(frameon=False, fontsize=7)
        else:
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                    transform=ax.transAxes)

    axes[0].set_ylabel("flip rate")
    fig.supxlabel(r"baseline margin $|R_{\mathrm{F16}}|$", y=-0.02)
    fig.suptitle("Are flips just boundary noise?", y=1.02)
    save(fig, out, "fig3_margin_control")
    plt.close(fig)


# --------------------------------------------------------------- figure 4

def fig_language_interaction(recs, scope, out: Path) -> None:
    """English against Vietnamese on identical facts.

    This tests a live disagreement in the literature: one line of work reports
    quantization disproportionately harming non-Latin scripts, another reports
    k-quantization not harming multilingual performance disproportionately.
    Vietnamese is Latin-script but heavily diacritised, sitting between them.
    """
    import matplotlib.pyplot as plt

    precs = [p for p in precisions(recs) if p != BASELINE_PRECISION]
    if not precs:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    width, drew = 0.36, False
    xs = np.arange(len(precs))

    for li, lang in enumerate(("en", "vi")):
        vals = []
        for prec in precs:
            deltas = []
            for model in models(recs):
                items = paired(recs, model, prec, condition=CONFLICT,
                               lang=lang, scope=scope)
                b, q = scored(items)
                deltas += [qq - bb for bb, qq in zip(b, q)]
            vals.append(statistics.fmean(deltas) if len(deltas) >= MIN_N else 0.0)
            drew |= len(deltas) >= MIN_N
        ax.bar(xs + (li - 0.5) * width, vals, width * 0.92,
               color=SERIES[li], label=lang, edgecolor="white", linewidth=0.8)

    zero_line(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels(precs)
    ax.set_xlabel("precision")
    ax.set_ylabel(r"mean $\Delta R$ vs F16")
    ax.set_title("Language × precision")
    if drew:
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                transform=ax.transAxes)
    save(fig, out, "fig4_language_interaction")
    plt.close(fig)


# --------------------------------------------------------------- figure 5

def fig_attribution(run_dir: Path, out: Path) -> None:
    """Corrected causal attribution of the evidence span, by precision."""
    import matplotlib.pyplot as plt

    recs = [r for p in sorted(run_dir.glob("xai__*.jsonl"))
            for r in iter_jsonl(p) if "error" not in r]
    if not recs:
        return

    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in recs:
        groups[(r["precision"], r["lang"])].append(r["A_star_evidence"])

    precs = [p for p in ["F16", "Q8_0", "Q4_K_M", "Q3_K_M"]
             if any(k[0] == p for k in groups)]
    if not precs:
        return

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    xs = np.arange(len(precs))
    for li, lang in enumerate(("en", "vi")):
        means, errs = [], []
        for prec in precs:
            vals = groups.get((prec, lang), [])
            if len(vals) < 5:
                means.append(0.0); errs.append(0.0); continue
            m, lo, hi = paired_bootstrap_ci(vals)
            means.append(m); errs.append((hi - lo) / 2)
        ax.errorbar(xs + (li - 0.5) * 0.06, means, yerr=errs, color=SERIES[li],
                    linestyle=LINESTYLES[lang], marker=MARKERS[li],
                    capsize=2, label=lang)

    zero_line(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels(precs)
    ax.set_xlabel("precision")
    ax.set_ylabel(r"$A^{*}(\mathrm{evidence})$")
    ax.set_title("Causal weight of the evidence span")
    ax.legend(frameon=False)
    save(fig, out, "fig5_attribution")
    plt.close(fig)


# --------------------------------------------------------------- figure 6

def fig_dose_response(run_dir: Path, out: Path) -> None:
    """P_ctx against evidence pressure, with the 0.5 crossing marked.

    The crossing is the override threshold - how much corroboration a false
    document needs before the model abandons what it knows. A shift in that
    threshold across precisions is easier to read, and to act on, than a shift
    in mean reliance.
    """
    import matplotlib.pyplot as plt

    recs = load(run_dir, "dose")
    if not recs:
        return
    by: dict[tuple, list[float]] = defaultdict(list)
    for r in recs:
        if r.kind == "score" and r.p_ctx is not None and r.key.condition.startswith("DOSE"):
            by[(r.precision, int(r.key.condition[4:]))].append(r.p_ctx)
    if not by:
        return

    precs = [p for p in ["F16", "Q8_0", "Q4_K_M", "Q3_K_M"]
             if any(k[0] == p for k in by)]
    doses = sorted({k[1] for k in by})

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for prec in precs:
        ys = [statistics.fmean(by[(prec, d)]) if by.get((prec, d)) else np.nan
              for d in doses]
        ax.plot(doses, ys, color=ladder_color(prec),
                marker=MARKERS[precs.index(prec) % len(MARKERS)], label=prec)

    ax.axhline(0.5, color="#898781", linestyle=":", linewidth=1.0)
    # Right-aligned, where the curves have already climbed away from 0.5.
    ax.text(doses[-1], 0.47, "override threshold", fontsize=7, color="#52514e",
            ha="right", va="top")
    ax.set_xticks(doses)
    ax.set_xlabel("evidence pressure (dose)")
    ax.set_ylabel(r"$P_{\mathrm{ctx}}$")
    ax.set_title("How much false evidence does it take?")
    ax.legend(frameon=False)
    save(fig, out, "fig6_dose_response")
    plt.close(fig)


# --------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--split", default="known_all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    apply_style()
    run_dir = ROOT / args.runs
    recs = load(run_dir, "main")
    if not recs:
        sys.exit(f"no main-pass records in {args.runs}")

    splits = ROOT / args.splits
    scope = load_scope(splits, args.split) if splits.exists() else None
    if scope is None:
        print("WARNING: no splits file; figures are unscoped and not publishable")

    out = Path(args.out) if args.out else ROOT / "results" / args.split / "figures"
    print(f"writing figures to {out}")

    fig_reliance_ladder(recs, scope, out)
    fig_flip_asymmetry(recs, scope, out)
    fig_margin_control(recs, scope, out)
    fig_language_interaction(recs, scope, out)
    fig_attribution(run_dir, out)
    fig_dose_response(run_dir, out)

    print("\nOpen the PNGs and look at them. The palette is validated; the "
          "layout is not - check for collisions and clipping before submitting.")


if __name__ == "__main__":
    main()
