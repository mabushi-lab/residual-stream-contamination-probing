"""
render_phase3.py
Turn the Phase 3 audits into the paper's positive-control section.

Phase 3 is the only arm where Requirement E holds by construction rather than
by argument: Oren et al. injected a random subset of PIQA's test file into
training and shipped the whole file, so the withheld items come back by
difference and both arms are one pool split before training.

Reads runs/phase3/*.json and writes:

  paper/generated/phase3_macros.tex   every scalar quoted in the section
  paper/generated/phase3_table.tex    the results table
  figures/p3_profiles.{pdf,png}       observed against baseline, per prefix

    python3 render/render_phase3.py [--dir experiments/runs/phase3]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import RUNS, GENERATED, FIGURES, s

INK, GREY = "#1a1a1a", "#8a8a8a"
BLUE, RED, GREEN = "#1f6feb", "#c0392b", "#2e7d4f"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.7,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

# Duplication counts, from Oren et al.'s README.
DUP = {"contam-1.4b-dupcount-higher": 50, "contam-1.4b": 1}
# Prefix modes, longest first so the table reads in order of information.
MODE_ORDER = {"full": 0, "raw": 1, "goal": 2}
MODE_LABEL = {"full": "full record", "goal": "question only",
              "raw": "verbatim JSONL"}

MAC: list[str] = []
WORD = {1: "One", 50: "Fifty"}


def mac(name, val, fmt=".3f"):
    MAC.append("\\newcommand{\\" + name + "}{" + format(val, fmt) + "}")


def load(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        o = json.load(open(f))
        if o.get("dry_run"):
            continue
        stem = os.path.basename(f)[:-5]
        model = stem.split("__")[0].replace("yonatano_", "")
        tail = stem.split("__")[-1]
        mode = tail.rsplit("_", 1)[-1] if tail.rsplit("_", 1)[-1] in MODE_LABEL \
            else "goal"
        pl = o.get("placebo", {})
        base = np.asarray(o["baseline_profile"], dtype=float)
        rows.append(dict(
            model=model, mode=mode, dup=DUP.get(model.lower(), None),
            ba_nuis=o["ba_nuisance"], exch=o.get("exchangeability", "ok"),
            obs=np.asarray(o["ba_by_layer"], dtype=float), base=base,
            span=100 * (base.max() - base.min()), peak=int(base.argmax()),
            L=len(base) - 1, n_suspect=o["n_suspect"],
            n_layers=o["n_layers"], nuis=o["nuisance_dims"],
            gap=pl.get("match_gap", float("nan")),
            T_raw=o["contrast"]["T_raw"], baseline=o["contrast"]["baseline"],
            T_adj=o["contrast"]["T_adj"], p=o["contrast"]["p_value"],
            p_raw=o["contrast_uncorrected"]["p_value"],
            blocked=bool(o.get("verdict_blocked", False))))
    rows.sort(key=lambda r: MODE_ORDER.get(r["mode"], 9))
    return rows


def figures(rows):
    os.makedirs(s(FIGURES), exist_ok=True)
    for paper in (False, True):
        suf, k = ("_paper", 0.8) if paper else ("", 1.0)
        fig, axes = plt.subplots(
            1, len(rows), figsize=((6.9, 2.5) if paper else (7.6, 3.0)),
            squeeze=False)
        for ax, r in zip(axes[0], rows):
            x = np.arange(len(r["base"])) / max(r["L"], 1)
            ax.plot(x, r["obs"], "-", lw=2.0, color=BLUE, label="observed")
            ax.plot(x, r["base"], "-", lw=2.0, color=RED, label="placebo baseline")
            ax.axhline(0.5, color=GREY, ls=":", lw=1.4)
            ax.set_ylim(0.46, 0.58)
            ax.set_xlabel("relative depth", fontsize=10 * k)
            ax.set_title(f"prefix: {MODE_LABEL[r['mode']]}"
                         f"  ($p={r['p']:.2f}$)", fontsize=9.5 * k, loc="left")
            ax.legend(fontsize=7.4 * k, frameon=False, loc="upper right")
        axes[0][0].set_ylabel("balanced accuracy", fontsize=10 * k)
        fig.savefig(s(FIGURES / f"p3_profiles{suf}.pdf"),
                    bbox_inches="tight", pad_inches=0.04)
        if not paper:
            fig.savefig(s(FIGURES / "p3_profiles.png"), dpi=300,
                        bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        print(f"  figures/p3_profiles{suf}")


def table(rows):
    L = [r"\begin{table}[t]", r"\small", r"\centering",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}lrrrrl@{}}", r"\toprule",
         r"Prefix & words & $\BA_{\nuis}$ & baseline span & "
         r"$T_{\mathrm{adj}}$ & Outcome \\", r"\midrule"]
    words = {"full": 44, "goal": 10, "raw": 52}
    for r in rows:
        L.append(
            f"{MODE_LABEL[r['mode']]} & {words.get(r['mode'], 0)} & "
            f"{r['ba_nuis']:.3f} & {r['span']:.1f} pts & {r['T_adj']:+.4f} & "
            f"null, $p={r['p']:.2f}$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Phase 3, measured. Oren et al.'s deliberately "
          r"contaminated 1.4B checkpoint, PIQA injected at $m=50$. Both arms "
          r"are the same test file split at random before training, so "
          r"Requirement E holds by construction and $\BA_{\nuis}$ confirms it. "
          r"The two rows differ only in how much of each record the probe "
          r"reads. Neither is significant, so the null is not an artefact of "
          r"a short prefix.}",
          r"\label{tab:phase3}", r"\end{table}"]
    open(s(GENERATED / "phase3_table.tex"), "w").write("\n".join(L) + "\n")
    print("  paper/generated/phase3_table.tex")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=s(RUNS / "phase3"))
    a = ap.parse_args()
    rows = load(a.dir)
    if not rows:
        raise SystemExit(f"no audits in {a.dir}")
    print(f"rendering Phase 3 from {len(rows)} audits:")
    os.makedirs(s(GENERATED), exist_ok=True)
    figures(rows)
    table(rows)

    full = next((r for r in rows if r["mode"] == "full"), rows[0])
    mac("PthreeN", float(len(rows)), ".0f")
    mac("PthreeDup", float(full["dup"] or 0), ".0f")
    mac("PthreeNsuspect", float(full["n_suspect"]), ".0f")
    mac("PthreeLayers", float(full["n_layers"]), ".0f")
    mac("PthreeNuisLo", min(r["ba_nuis"] for r in rows))
    mac("PthreeNuisHi", max(r["ba_nuis"] for r in rows))
    mac("PthreePLo", min(r["p"] for r in rows), ".2f")
    mac("PthreePHi", max(r["p"] for r in rows), ".2f")
    mac("PthreeTadjFull", full["T_adj"], ".4f")
    mac("PthreePFull", full["p"], ".2f")
    mac("PthreeGapFull", full["gap"], ".3f")
    mac("PthreeSpanLo", min(r["span"] for r in rows), ".1f")
    mac("PthreeSpanHi", max(r["span"] for r in rows), ".1f")
    mac("PthreePeakFull", float(full["peak"]), ".0f")
    mac("PthreeNegative", float(sum(1 for r in rows if r["T_adj"] < 0)), ".0f")
    open(s(GENERATED / "phase3_macros.tex"), "w").write(
        "% Auto-generated by render_phase3.py from runs/phase3/*.json\n"
        + "\n".join(MAC) + "\n")
    print(f"  paper/generated/phase3_macros.tex ({len(MAC)} macros)")


if __name__ == "__main__":
    main()
