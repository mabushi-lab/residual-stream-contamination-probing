"""
export_tables.py
Non-graph displays of the same data: matrices, a heatmap, a decision grid and
a pipeline schematic, plus LaTeX tables generated from phase0_results.json.

Writes to figures/ :
  matrix_methods            capability matrix, RSCP against the alternatives
  heatmap_null_calibration  V1 as a labelled matrix rather than four curves
  matrix_decision_gsm       what each combination of outcomes licenses
  schematic_protocol        the eleven protocol steps and the placebo gate

and to the working directory:
  phase0_tables.tex         booktabs tables, \\input by thesis.tex

    python3 export_tables.py
"""

from __future__ import annotations

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

from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = s(FIGURES)
RES = s(VAL_RESULTS / "phase0_results.json")

# Standalone renders carry their own title and footnote, for slides and for
# repository thumbnails. The paper variants omit both, because there the
# LaTeX caption says the same thing and repeating it looks careless.
TITLES = True
SUFFIX = ""


def fs(standalone, paper):
    """Paper variants are drawn at close to their printed size, so the text
    is not downscaled into illegibility inside a two-column layout."""
    return paper if SUFFIX else standalone

INK = "#1a1a1a"
GREY = "#8a8a8a"
FAINT = "#d8d8d8"
BLUE = "#1f6feb"
RED = "#c0392b"
GREEN = "#2e7d4f"
AMBER = "#d99017"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def save(fig, name, dpi=300):
    os.makedirs(OUT, exist_ok=True)
    exts = [("pdf", {})] if SUFFIX else [("png", {"dpi": dpi}), ("pdf", {})]
    for ext, kw in exts:
        fig.savefig(f"{OUT}/{name}{SUFFIX}.{ext}", bbox_inches="tight",
                    pad_inches=0.05, **kw)
    plt.close(fig)
    print(f"  {OUT}/{name}{SUFFIX}")


def heading(ax_or_fig, *args, **kwargs):
    """Draw a heading only in the standalone renders."""
    if TITLES:
        ax_or_fig.text(*args, **kwargs)


def _blank(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# --------------------------------------------------------------------------
# 1. Capability matrix
# --------------------------------------------------------------------------

METHODS = ["n-gram\noverlap", "Likelihood\nMIA", "Black-box\norder test",
           "Activation\nprobe (prior)", "RSCP\n(this paper)"]
CRITERIA = [
    "Works without the training corpus",
    "Works without model weights",
    "Needs no foresight at dataset release",
    "Detects paraphrased contamination",
    "Detects item-level, not just file-level",
    "Reports a calibrated uncertainty",
    "Declares a nuisance control",
    "Reports an exposure level, not a verdict",
]
# 2 = yes, 1 = partial, 0 = no
GRID = np.array([
    [0, 1, 1, 1, 1],
    [2, 0, 2, 0, 0],
    [2, 2, 2, 2, 2],
    [0, 1, 0, 1, 1],
    [2, 2, 0, 2, 1],
    [0, 1, 2, 0, 2],
    [0, 0, 0, 0, 2],
    [0, 0, 0, 0, 2],
])


def matrix_methods():
    nr, nc = GRID.shape
    fig, ax = plt.subplots(figsize=fs((7.8, 4.4), (7.1, 3.5)))
    _blank(ax)
    ax.set_xlim(-0.02, nc); ax.set_ylim(-1.15, nr + (0.55 if TITLES else 0.12))

    def dot(x, y, v):
        # ax.plot markers stay circular regardless of the axes aspect ratio.
        if v == 2:
            ax.plot(x, y, "o", ms=12, mfc=GREEN, mec="white", mew=1.2, zorder=3)
        elif v == 1:
            ax.plot(x, y, "o", ms=12, mfc=AMBER, mfcalt="white", mec=AMBER,
                    mew=1.4, fillstyle="left", zorder=3)
        else:
            ax.plot(x, y, "o", ms=12, mfc=FAINT, mec="white", mew=1.2, zorder=3)

    for i in range(nr):
        yy = nr - 1 - i
        if i % 2 == 0:
            ax.add_patch(Rectangle((0, yy - 0.5), nc, 1.0,
                                   color="#f7f7f7", zorder=0))
        ax.text(-0.12, yy, CRITERIA[i], ha="right", va="center", fontsize=9)
        for j in range(nc):
            dot(j + 0.5, yy, GRID[i, j])

    for j, mname in enumerate(METHODS):
        ax.text(j + 0.5, nr - 0.32, mname, ha="center", va="bottom",
                fontsize=9, fontweight="bold" if j == nc - 1 else "normal",
                color=BLUE if j == nc - 1 else INK)
    ax.add_patch(Rectangle((nc - 1, -0.62), 1.0, nr + 0.06, fill=False,
                           ec=BLUE, lw=1.8, zorder=6))

    for lab, v, xx in [("yes", 2, 0.35), ("partial", 1, 1.55), ("no", 0, 2.85)]:
        dot(xx, -0.92, v)
        ax.text(xx + 0.14, -0.92, lab, va="center", fontsize=8.5, color=GREY)

    heading(ax, -0.12, nr + 0.30, "What each contamination detector can and "
          "cannot do", fontsize=11, ha="right", va="bottom")
    save(fig, "matrix_methods")


# --------------------------------------------------------------------------
# 2. Null-calibration heatmap
# --------------------------------------------------------------------------

def heatmap_null(r):
    v1 = sorted(r["v1_null_capacity"], key=lambda x: x["p_nuis"])
    cols = [v["p_nuis"] for v in v1]
    rows = [("p_naive", "raw probe accuracy\n(no control at all)"),
            ("p_maxlevel", "$\\max_\\ell\\,\\Delta_\\ell$\n(level, as first specified)"),
            ("p_meanlevel", "mean$_\\ell\\,\\Delta_\\ell$\n(level)"),
            ("p_contrast", "depth-profile contrast\n(adopted)")]
    M = np.array([[v[k]["reject_05"] for v in v1] for k, _ in rows])

    fig, ax = plt.subplots(figsize=fs((6.4, 3.4), (6.4, 3.4)))
    _blank(ax)
    nr, nc = M.shape
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "fp", ["#ffffff", "#fbe3df", "#e8a297", "#c0392b"])
    for i in range(nr):
        for j in range(nc):
            val = M[i, j]
            ax.add_patch(Rectangle((j, nr - 1 - i), 1, 1,
                                   color=cmap(min(val / 1.0, 1.0)),
                                   ec="white", lw=2.5))
            ax.text(j + 0.5, nr - 1 - i + 0.5, f"{val:.3f}",
                    ha="center", va="center", fontsize=11,
                    color="white" if val > 0.55 else INK,
                    fontweight="bold" if i == nr - 1 else "normal")
    for i, (_, lab) in enumerate(rows):
        ax.text(-0.12, nr - 1 - i + 0.5, lab, ha="right", va="center",
                fontsize=8.8, linespacing=1.35,
                color=BLUE if i == nr - 1 else INK,
                fontweight="bold" if i == nr - 1 else "normal")
    for j, c in enumerate(cols):
        ax.text(j + 0.5, nr + 0.12, str(c), ha="center", va="bottom",
                fontsize=10)
    ax.text(nc / 2, nr + 0.60, "nuisance control set dimension",
            ha="center", va="bottom", fontsize=9.0, color=GREY)
    ax.add_patch(Rectangle((0, 0), nc, 1, fill=False, ec=BLUE, lw=2.2,
                           zorder=5))
    ax.set_xlim(-0.05, nc + 0.05)
    ax.set_ylim(-0.75 if TITLES else -0.10, nr + 1.02)
    heading(ax, 0, -0.42, "False positive rate under a true null. Nominal is "
          "0.05; 150 replicates per cell.", fontsize=8.5, color=GREY, va="top")
    save(fig, "heatmap_null_calibration")


# --------------------------------------------------------------------------
# 3. Decision grid
# --------------------------------------------------------------------------

CELLS = {
    (1, 1): (GREEN, "Contamination confirmed",
             "The straightforward reading.\n"
             "Report the exposure interval\nand the adjusted score."),
    (1, 0): (AMBER, "Seen, not exploited",
             "Items are stored but not used.\n"
             "Memorisation without\nexploitation."),
    (0, 1): (AMBER, "Inflation, not memory",
             "Exposure below the detection\n"
             "floor, or format overfitting,\nor difficulty mismatch."),
    (0, 0): (GREY, "No evidence",
             "At this sensitivity. Report the\n"
             "detection floor so the null\nreads at the right strength."),
}


def matrix_decision():
    fig, ax = plt.subplots(figsize=fs((8.4, 4.3), (7.1, 3.3)))
    k = 1.0 if not SUFFIX else 0.80   # scale type with the canvas
    _blank(ax)
    ax.set_xlim(-1.25, 2.06)
    ax.set_ylim(-0.55 if TITLES else -0.02, 2.62 if TITLES else 2.42)
    for (row, col), (c, title, body) in CELLS.items():
        x, y = col, row
        ax.add_patch(FancyBboxPatch((x + 0.03, y + 0.05), 0.94, 0.90,
                                    boxstyle="round,pad=0.012,rounding_size=0.03",
                                    fc="white", ec=c, lw=2.0, zorder=2))
        ax.add_patch(Rectangle((x + 0.03, y + 0.76), 0.94, 0.19, color=c,
                               alpha=0.15, zorder=3))
        ax.text(x + 0.5, y + 0.855, title, ha="center", va="center",
                fontsize=9.2 * k, fontweight="bold", color=c, zorder=4)
        ax.text(x + 0.5, y + 0.40, body, ha="center", va="center",
                fontsize=8.6 * k, linespacing=1.6, color=INK, zorder=4)

    ax.text(1.0, 2.30, "accuracy gap on the matched twin", ha="center",
            va="bottom", fontsize=9.5 * k, color=GREY)
    ax.plot([0.03, 1.97], [2.24, 2.24], color=FAINT, lw=1.2)
    ax.text(0.5, 2.06, "absent", ha="center", va="bottom", fontsize=10.5 * k)
    ax.text(1.5, 2.06, "present", ha="center", va="bottom", fontsize=10.5 * k)
    ax.text(-0.08, 1.5, "depth contrast\nsignificant", ha="right",
            va="center", fontsize=10.5 * k, linespacing=1.5)
    ax.text(-0.08, 0.5, "depth contrast\nnull", ha="right", va="center",
            fontsize=10.5 * k, linespacing=1.5)

    heading(ax, -1.22, 2.50, "Two lines of evidence, four conclusions "
          "(GSM8K against GSM1k)", fontsize=11.5, va="bottom")
    heading(ax, -1.22, -0.30, "Behavioural evidence alone cannot separate the "
          "right-hand column. That is what the representational test adds.",
          fontsize=8.8, color=GREY, va="top")
    save(fig, "matrix_decision_gsm")


# --------------------------------------------------------------------------
# 4. Pipeline schematic
# --------------------------------------------------------------------------

def schematic():
    fig, ax = plt.subplots(figsize=fs((8.2, 4.2), (7.3, 3.5)))
    _blank(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.7 if TITLES else 5.25)

    def box(x, y, w, h, text, ec=INK, fc="white", fs=8.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.09",
                                    fc=fc, ec=ec, lw=1.6, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, linespacing=1.45, zorder=3,
                fontweight="bold" if bold else "normal")

    def arr(x1, y1, x2, y2, c=GREY):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), lw=1.5, color=c,
                                     arrowstyle="-|>", mutation_scale=12,
                                     zorder=1, shrinkA=0, shrinkB=0))

    R1, R2, R3, H = 4.05, 2.20, 0.35, 1.00
    W = 2.05
    xs = [0.15, 2.60, 5.05, 7.50]

    box(xs[0], R1, W, H, "suspect set $S$\nreference set $R$", ec=BLUE)
    box(xs[1], R1, W, H, "Requirement E\nexchangeable\nby construction?", ec=BLUE)
    box(xs[2], R1, W, H, "prefix activations\nevery layer, never\npast the answer")
    box(xs[3], R1, W + 0.30, H, "nuisance family\nembeddings, $n$-grams,\nlength, reference LM")
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        arr(xs[a] + W, R1 + H / 2, xs[b], R1 + H / 2)
    ax.text(xs[1] + W / 2, R1 - 0.10, "if no, the method does not apply",
            fontsize=7.4, color=RED, ha="center", va="top")

    # serpentine wrap, right of row 1 down and back to the left of row 2
    yw = R1 - 0.62
    ax.plot([xs[3] + (W + 0.30) / 2, xs[3] + (W + 0.30) / 2], [R1, yw],
            color=GREY, lw=1.5, zorder=1)
    ax.plot([xs[3] + (W + 0.30) / 2, xs[0] + W / 2], [yw, yw],
            color=GREY, lw=1.5, zorder=1)
    arr(xs[0] + W / 2, yw, xs[0] + W / 2, R2 + H)

    box(xs[0], R2, W, H, "cross-fitted\nridge probes\n$\\Pi_\\ell$ free of labels")
    box(xs[1], R2, W, H, "$BA_\\ell$ profile and\n$BA_{\\mathrm{nuis}}$\n(descriptive only)",
        ec=GREY)
    box(xs[2], R2, W, H,
        "depth contrast\n$T_c=\\sum_\\ell w_\\ell BA_\\ell$", ec=BLUE, bold=True)
    box(xs[3], R2, W + 0.30, H,
        "PLACEBO GATE\nsplit $R$ on a surface\nvariable, rerun $T_c$", ec=RED)
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        arr(xs[a] + W, R2 + H / 2, xs[b], R2 + H / 2)

    box(xs[1], R3, W + 0.30, H, "no claim can\nbe made", ec=RED,
        fc="#fdf1ef", bold=True)
    box(xs[3], R3, W + 0.30, H,
        "exposure interval\n$\\widehat{\\mathcal{M}}$ and floor $m^{*}$",
        ec=GREEN, fc="#f2f9f5", bold=True)
    arr(xs[3] + 0.35, R2, xs[1] + W + 0.10, R3 + H, c=RED)
    ax.text(5.25, R3 + H + 0.30, "placebo significant", fontsize=7.6,
            color=RED, ha="center")
    arr(xs[3] + (W + 0.30) / 2, R2, xs[3] + (W + 0.30) / 2, R3 + H, c=GREEN)
    ax.text(xs[3] + (W + 0.30) / 2 + 0.12, R3 + H + 0.32, "null", fontsize=7.6,
            color=GREEN)

    heading(ax, 0.15, 5.40, "The protocol, with the placebo as a gate rather "
          "than a footnote", fontsize=11, va="bottom")
    save(fig, "schematic_protocol")


# --------------------------------------------------------------------------
# 5. LaTeX tables
# --------------------------------------------------------------------------

def latex_tables(r):
    v1 = sorted(r["v1_null_capacity"], key=lambda x: x["p_nuis"])
    L = ["% Auto-generated by export_tables.py. Do not edit by hand.", ""]

    L += [r"\begin{table}[t]", r"\small", r"\setlength{\tabcolsep}{3.2pt}",
          r"\centering",
          r"\begin{tabular}{@{}l" + "c" * len(v1) + r"@{}}", r"\toprule",
          r"Statistic & \multicolumn{%d}{c}{nuisance block dimension $p$} \\"
          % len(v1),
          r"\cmidrule(l){2-%d}" % (len(v1) + 1),
          " & " + " & ".join(str(v["p_nuis"]) for v in v1) + r" \\",
          r"\midrule"]
    for key, lab in [("p_naive", r"raw $\max_\ell \BA_\ell$, no control"),
                     ("p_maxlevel", r"$\max_\ell \Delta_\ell$ (level)"),
                     ("p_meanlevel", r"$\overline{\Delta_\ell}$ (level)")]:
        L.append(lab + " & " + " & ".join(f"{v[key]['reject_05']:.3f}"
                                          for v in v1) + r" \\")
    L += [r"\addlinespace",
          r"\textbf{depth contrast} & " + " & ".join(
              f"\\textbf{{{v['p_contrast']['reject_05']:.3f}}}" for v in v1)
          + r" \\",
          r"\quad 95\% CI & " + " & ".join(
              f"\\tiny[{v['p_contrast']['ci'][0]:.2f},{v['p_contrast']['ci'][1]:.2f}]"
              for v in v1) + r" \\",
          r"\midrule",
          r"$\BA_{\nuis}$ & " + " & ".join(f"{v['ba_nuis']:.3f}" for v in v1)
          + r" \\",
          r"mean $\Delta_\ell$ & " + " & ".join(f"{v['mean_delta']:+.3f}"
                                                for v in v1) + r" \\",
          r"\bottomrule", r"\end{tabular}",
          r"\caption{Measured false positive rate under a true null "
          r"($\varepsilon=0$, $\rho=1$), nominal $0.05$, 150 replicates per "
          r"cell. The activations are 96-dimensional, so $p=96$ is the "
          r"dimension-matched column. Only the contrast is stable across the "
          r"row.}",
          r"\label{tab:phase0-null}", r"\end{table}", ""]

    e = r["v5_exposure"]
    t = r["v6_timing"][-1]
    p3 = {n: {round(x["eps"], 2): x["reject_05"] for x in rec}
          for n, rec in r["v3_power"].items()}
    n_big = max(p3, key=lambda k: int(k))
    prof = r["v4_profile"]
    L += [r"\begin{table}[t]", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
          r"\centering",
          r"\begin{tabular}{@{}llp{0.40\columnwidth}@{}}", r"\toprule",
          r"& Question & Measured outcome \\", r"\midrule",
          r"V1 & Null calibration vs.\ control size & "
          r"level statistic %.3f to %.3f; contrast %.3f to %.3f \\"
          % (min(v["p_maxlevel"]["reject_05"] for v in v1),
             max(v["p_maxlevel"]["reject_05"] for v in v1),
             min(v["p_contrast"]["reject_05"] for v in v1),
             max(v["p_contrast"]["reject_05"] for v in v1)),
          r"V2 & Nuisance growing with depth & false positives %.2f at "
          r"10\%% growth, %.2f at 50\%% \\"
          % (r["v2_depth_nuisance"][2]["false_positive_rate"]["reject_05"],
             r["v2_depth_nuisance"][-1]["false_positive_rate"]["reject_05"]),
          r"V3 & Power vs.\ effect and sample size & %.2f at "
          r"$\varepsilon{=}0.8$, $n{=}%s$; miscalibrated at $n{=}250$ \\"
          % (p3[n_big].get(0.8, float("nan")), n_big),
          r"V4 & Is the planted profile recovered & argmax at layer %d, "
          r"planted peak %d \\"
          % (int(np.argmax(prof["ba_mean"])), prof["planted_peak_layer"]),
          r"V5 & Does exposure calibration invert & coverage %.2f to %.2f; "
          r"floor $m^{*}{=}%.1f$ \\"
          % (min(c["coverage"] for c in e["inversion_coverage"]),
             max(c["coverage"] for c in e["inversion_coverage"]),
             e["detection_floor_m"]),
          r"V6 & Cost of the smoother shortcut & %.2f\,s setup, then "
          r"%.1f\,s vs.\ %.1f\,h for 2000 resamples \\"
          % (t["setup_s"], t["projected_B2000_shortcut_s"],
             t["projected_B2000_refit_s"] / 3600.0),
          r"\bottomrule", r"\end{tabular}",
          r"\caption{The six Phase 0 experiments. All values measured on "
          r"synthetic activations; none is a claim about a language model.}",
          r"\label{tab:phase0-summary}", r"\end{table}", ""]

    open(GENERATED / "phase0_tables.tex", "w").write("\n".join(L) + "\n")
    print(f"  {GENERATED}/phase0_tables.tex")


def main():
    global TITLES, SUFFIX
    r = json.load(open(RES))
    print("standalone renders (titled):")
    matrix_methods(); heatmap_null(r); matrix_decision(); schematic()
    latex_tables(r)
    print("paper variants (caption carries the heading):")
    TITLES, SUFFIX = False, "_paper"
    matrix_methods(); heatmap_null(r); matrix_decision(); schematic()


if __name__ == "__main__":
    main()
