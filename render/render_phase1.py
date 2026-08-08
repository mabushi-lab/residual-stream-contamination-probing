"""
render_phase1.py
Turn the Phase 1 audit outputs into the paper's real-model section.

Reads runs/phase1/*.json and writes:

  phase1_macros.tex    every scalar quoted in Section 7
  phase1_table.tex     the results table
  figures/p1_*.pdf     baseline profiles, and span against BA_nuisance

    python3 render_phase1.py [--dir runs/phase1]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))
from paths import VAL_RESULTS, GENERATED, FIGURES, RUNS, DATA, CACHE, EXPERIMENTS, s


OUT = s(FIGURES)
INK, GREY, FAINT = "#1a1a1a", "#8a8a8a", "#d8d8d8"
BLUE, RED, GREEN, ORANGE = "#1f6feb", "#c0392b", "#2e7d4f", "#e07b39"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.7,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

MAC: list[str] = []
WORD = {160: "OneSixty", 410: "FourTen", 1400: "OneFour"}


def mac(name, val, fmt=".3f"):
    MAC.append("\\newcommand{\\" + name + "}{" + format(val, fmt) + "}")


def load(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        o = json.load(open(f))
        if o.get("dry_run"):
            continue
        stem = os.path.basename(f)[:-5]
        model, _, split = stem.partition("__")
        model = model.replace("EleutherAI_", "")
        base = np.array(o["baseline_profile"], dtype=float)
        obs = np.array(o["ba_by_layer"], dtype=float)
        rows.append(dict(
            model=model, split=split, ba_nuis=o["ba_nuisance"],
            exch=o.get("exchangeability", "ok"), base=base, obs=obs,
            span=100 * (base.max() - base.min()), peak=int(base.argmax()),
            L=len(base) - 1, n_suspect=o["n_suspect"],
            T_raw=o["contrast"]["T_raw"], baseline=o["contrast"]["baseline"],
            T_adj=o["contrast"]["T_adj"], p=o["contrast"]["p_value"],
            p_raw=o["contrast_uncorrected"]["p_value"],
            blocked=bool(o.get("verdict_blocked", False))))
    return rows


def figures(rows):
    os.makedirs(OUT, exist_ok=True)
    wk = [r for r in rows if r["split"] == "wikimia"]
    pl = [r for r in rows if r["split"].startswith("pile")]

    # 1. Real baseline depth profiles, the paper's central empirical claim.
    for paper in (False, True):
        suf = "_paper" if paper else ""
        k = 0.8 if paper else 1.0
        fig, axes = plt.subplots(
            1, 2, figsize=(6.9, 2.5) if paper else (7.6, 3.0))
        for ax, group, title in [
                (axes[0], wk, "temporal split (WikiMIA)"),
                (axes[1], pl, "matched split (Pile train vs.\\ val)")]:
            for r in group:
                x = np.arange(len(r["base"])) / max(r["L"], 1)
                ax.plot(x, r["base"], "-", lw=2.0,
                        color=RED if r["split"] == "wikimia" else BLUE,
                        alpha=0.55 if "410m" in r["model"] else 1.0,
                        label=f"{r['model']}, {r['split'].replace('pile_','')}")
            ax.axhline(0.5, color=GREY, ls=":", lw=1.4)
            ax.set_ylim(0.44, 0.83)
            ax.set_xlabel("relative depth", fontsize=10 * k)
            ax.set_title(title, fontsize=9.5 * k, loc="left")
            ax.legend(fontsize=7.4 * k, frameon=False, loc="upper right")
        axes[0].set_ylabel("baseline separability", fontsize=10 * k)
        fig.savefig(f"{OUT}/p1_baseline_profiles{suf}.pdf",
                    bbox_inches="tight", pad_inches=0.04)
        if not paper:
            fig.savefig(f"{OUT}/p1_baseline_profiles.png", dpi=300,
                        bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        print(f"  {OUT}/p1_baseline_profiles{suf}")

    # 2. Span against BA_nuisance: the correction scales with the need for it.
    for paper in (False, True):
        suf = "_paper" if paper else ""
        k = 0.8 if paper else 1.0
        fig, ax = plt.subplots(figsize=(3.35, 2.3) if paper else (4.6, 3.0))
        for r in rows:
            c = RED if r["split"] == "wikimia" else BLUE
            m = "o" if "160m" in r["model"] else "s"
            ax.plot(r["ba_nuis"], r["span"], m, color=c, ms=8, mec="white",
                    mew=1.2)
        b = np.array([r["ba_nuis"] for r in rows])
        s = np.array([r["span"] for r in rows])
        ax.set_xlabel("$\\mathrm{BA}_{\\mathrm{nuis}}$ (blind separability)",
                      fontsize=10 * k)
        ax.set_ylabel("baseline span (accuracy points)", fontsize=10 * k)
        ax.set_xlim(0.44, 0.92)
        ax.text(0.50, 25, "matched", color=BLUE, fontsize=9 * k)
        ax.text(0.72, 25, "temporal", color=RED, fontsize=9 * k)
        fig.savefig(f"{OUT}/p1_span_vs_nuisance{suf}.pdf",
                    bbox_inches="tight", pad_inches=0.04)
        if not paper:
            fig.savefig(f"{OUT}/p1_span_vs_nuisance.png", dpi=300,
                        bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        print(f"  {OUT}/p1_span_vs_nuisance{suf}")
    return float(np.corrcoef(b, s)[0, 1])


def table(rows):
    L = [r"\begin{table*}[t]", r"\small", r"\centering",
         r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{@{}llrrrrrrl@{}}", r"\toprule",
         r"Model & Split & $n_S$ & $\BA_{\nuis}$ & baseline span & "
         r"$T_{\mathrm{raw}}$ & baseline & $T_{\mathrm{adj}}$ & Outcome \\",
         r"\midrule"]
    for r in sorted(rows, key=lambda x: (x["split"], x["model"])):
        outcome = ("no verdict, Req.\\ E fails" if r["blocked"]
                   else f"null, $p={r['p']:.2f}$")
        L.append(
            f"{r['model']} & {r['split'].replace('_',' ')} & {r['n_suspect']} & "
            f"{r['ba_nuis']:.3f} & {r['span']:.1f} pts & {r['T_raw']:+.4f} & "
            f"{r['baseline']:+.4f} & {r['T_adj']:+.4f} & {outcome} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Phase 1, measured. $\BA_{\nuis}$ is what a classifier "
          r"with no access to the model achieves; near $0.5$ the two sets are "
          r"exchangeable, near $0.86$ they are not and no contamination "
          r"verdict is admissible. The baseline span is the range of the "
          r"placebo depth profile, and it is the quantity the flat-profile "
          r"assumption gets wrong. Every Pile arm is null with a negative "
          r"adjusted statistic, which is what the literature predicts for a "
          r"corpus seen approximately once.}",
          r"\label{tab:phase1}", r"\end{table*}"]
    open(GENERATED / "phase1_table.tex", "w").write("\n".join(L) + "\n")
    print(f"  {GENERATED}/phase1_table.tex")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=s(RUNS / "phase1"))
    a = ap.parse_args()
    rows = load(a.dir)
    if not rows:
        raise SystemExit(f"no audits in {a.dir}")
    print(f"rendering Phase 1 from {len(rows)} audits:")
    corr = figures(rows)
    table(rows)

    wk = [r for r in rows if r["split"] == "wikimia"]
    pl = [r for r in rows if r["split"].startswith("pile")]
    mac("PoneN", float(len(rows)), ".0f")
    mac("PoneModels", float(len({r["model"] for r in rows})), ".0f")
    mac("PoneCorr", corr, ".2f")
    mac("PoneWkNuisLo", min(r["ba_nuis"] for r in wk))
    mac("PoneWkNuisHi", max(r["ba_nuis"] for r in wk))
    mac("PonePileNuisLo", min(r["ba_nuis"] for r in pl))
    mac("PonePileNuisHi", max(r["ba_nuis"] for r in pl))
    mac("PoneWkSpanLo", min(r["span"] for r in wk), ".1f")
    mac("PoneWkSpanHi", max(r["span"] for r in wk), ".1f")
    mac("PonePileSpanLo", min(r["span"] for r in pl), ".1f")
    mac("PonePileSpanHi", max(r["span"] for r in pl), ".1f")
    mac("PonePileNulls", float(sum(1 for r in pl if r["p"] > 0.05)), ".0f")
    mac("PonePileTotal", float(len(pl)), ".0f")
    mac("PoneBlocked", float(sum(1 for r in rows if r["blocked"])), ".0f")
    mac("PoneShareLo", 100*min(abs(r["baseline"] / r["T_raw"]) for r in wk), ".0f")
    mac("PoneShareHi", 100*max(abs(r["baseline"] / r["T_raw"]) for r in wk), ".0f")
    mac("PoneNsuspect", float(max(r["n_suspect"] for r in rows)), ".0f")
    open(GENERATED / "phase1_macros.tex", "w").write(
        "% Auto-generated by render_phase1.py from runs/phase1/*.json\n"
        + "\n".join(MAC) + "\n")
    print(f"  {GENERATED}/phase1_macros.tex ({len(MAC)} macros)")


if __name__ == "__main__":
    main()
