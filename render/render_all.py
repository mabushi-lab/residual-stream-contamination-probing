"""
render_all.py
Turn the results files into everything the paper needs.

Reads phase0c_results.json (the final protocol, both simulators) and
phase0b_results.json (the capacity sweep that killed the level statistic),
and writes:

  results_macros.tex   every scalar and every coordinate quoted in the paper
  results_tables.tex   booktabs tables
  figures/*.pdf|png    standalone renders

Nothing in the paper is transcribed by hand. Re-run the validation, re-run
this, recompile, and the paper updates.

    python3 render_all.py
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

PAPER = False          # set by main(); paper variants are drawn at print size
SUF = ""


def fs(standalone, paper):
    return paper if PAPER else standalone


def sz(base):
    """Font size, shrunk for the single-column paper variants."""
    return base * (0.78 if PAPER else 1.0)


WORD = {0: "Zero", 5: "Five", 10: "Ten", 15: "Fifteen", 20: "Twenty",
        25: "TwentyFive", 30: "Thirty", 50: "Fifty", 100: "Hundred"}


def wd(x, scale=10):
    """LaTeX control sequences may contain letters only, so numeric suffixes
    are spelled out."""
    k = int(round(x * scale))
    return WORD.get(k, "N" + str(k).replace("-", "M"))


MAC: list[str] = []


def mac(name, val, fmt=".3f"):
    MAC.append("\\newcommand{\\" + name + "}{" + format(val, fmt) + "}")


def coords(name, xs, ys, prec=4):
    body = " ".join(f"({x:.{prec}g},{y:.{prec}g})" for x, y in zip(xs, ys))
    MAC.append("\\newcommand{\\" + name + "}{" + body + "}")


def save(fig, name, dpi=300):
    os.makedirs(OUT, exist_ok=True)
    exts = [("pdf", {})] if PAPER else [("png", {"dpi": dpi}), ("pdf", {})]
    for ext, kw in exts:
        fig.savefig(f"{OUT}/{name}{SUF}.{ext}", bbox_inches="tight",
                    pad_inches=0.04, **kw)
    plt.close(fig)
    print(f"  {OUT}/{name}{SUF}")


SIMNAME = {"A": "Sim-A (independent layers)", "B": "Sim-B (residual stream)"}


# --------------------------------------------------------------------------

def do_properties(c):
    p = c["sim_properties"]
    for s in "AB":
        mac(f"Prop{s}Growth", p[s]["rms_growth"], ".2f")
        mac(f"Prop{s}Corr", p[s]["corr_first_last"], ".2f")
        mac(f"Prop{s}Kurt", p[s]["kurtosis"], ".1f")
        mac(f"Prop{s}Eig", p[s]["top_eig_share"], ".3f")


def do_capacity(b):
    """C1: the level statistic tracks the size of the control set."""
    v = sorted([x for x in b["null_calibration"] if x["sim"] == "A"],
               key=lambda x: x["p_nuis"])
    xs = [x["p_nuis"] for x in v]
    for key, nm in [("p_perm", "Contrast"), ("p_maxlevel", "MaxLevel"),
                    ("p_meanlevel", "MeanLevel"), ("p_naive", "Naive")]:
        coords("CapA" + nm, xs, [x[key]["reject_05"] for x in v])
    tag = {96: "Match", 400: "Large", 1200: "Huge"}
    for x in v:
        t = tag[x["p_nuis"]]
        mac("CapContrast" + t, x["p_perm"]["reject_05"])
        mac("CapMax" + t, x["p_maxlevel"]["reject_05"])
        mac("CapNaive" + t, x["p_naive"]["reject_05"])
        mac("CapBAn" + t, x["ba_nuis"])
    bo = [x["p_boot"]["reject_05"] for x in v]
    mac("BootMin", min(bo))
    mac("BootMax", max(bo))
    for x in v:
        mac("CapBoot" + tag[x["p_nuis"]], x["p_boot"]["reject_05"])
    mac("CapPdimLo", float(min(xs)), ".0f")
    mac("CapPdimHi", float(max(xs)), ".0f")
    mac("CapRatio", max(xs) / min(xs), ".0f")

    fig, ax = plt.subplots(figsize=fs((5.2, 3.0), (3.35, 2.25)))
    for key, lab, col, mk in [
            ("p_naive", "raw accuracy, no control", "#7d5ba6", "d"),
            ("p_maxlevel", r"$\max_\ell\,\Delta_\ell$ (level)", RED, "s"),
            ("p_meanlevel", r"mean$_\ell\,\Delta_\ell$ (level)", ORANGE, "^"),
            ("p_perm", "depth contrast", BLUE, "o")]:
        ax.plot(xs, [x[key]["reject_05"] for x in v], "-", marker=mk,
                color=col, lw=2.2, ms=6, label=lab, zorder=3)
    ax.axhline(0.05, color=GREY, ls=":", lw=1.6)
    ax.set_xscale("log"); ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("dimension of the nuisance control set $p$", fontsize=sz(10))
    ax.set_ylabel("false positive rate", fontsize=sz(10))
    ax.set_ylim(-0.06, 1.35); ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.legend(fontsize=sz(8.2), frameon=False, ncol=2, loc="upper left",
              bbox_to_anchor=(-0.015, 1.04), columnspacing=1.0)
    save(fig, "c1_capacity")


def do_slope(c):
    """C2: recentring on the placebo removes the depth-slope failure."""
    d = c["depth_slope"]
    for s in "AB":
        v = sorted([x for x in d if x["sim"] == s], key=lambda x: x["slope"])
        coords(f"Slope{s}Raw", [x["slope"] for x in v],
               [x["raw"]["reject_05"] for x in v])
        coords(f"Slope{s}Adj", [x["slope"] for x in v],
               [x["adj"]["reject_05"] for x in v])
        for x in v:
            t = wd(x["slope"], 100)
            mac(f"Slope{s}Raw{t}", x["raw"]["reject_05"])
            mac(f"Slope{s}Adj{t}", x["adj"]["reject_05"])

    fig, ax = plt.subplots(figsize=fs((5.2, 3.0), (3.35, 2.25)))
    for s, col, mk in [("A", RED, "s"), ("B", ORANGE, "^")]:
        v = sorted([x for x in d if x["sim"] == s], key=lambda x: x["slope"])
        ax.plot([x["slope"] for x in v], [x["raw"]["reject_05"] for x in v],
                "--", marker=mk, color=col, lw=2.0, ms=6,
                label=f"uncorrected, Sim-{s}")
    for s, col, mk in [("A", BLUE, "o"), ("B", GREEN, "v")]:
        v = sorted([x for x in d if x["sim"] == s], key=lambda x: x["slope"])
        ax.plot([x["slope"] for x in v], [x["adj"]["reject_05"] for x in v],
                "-", marker=mk, color=col, lw=2.4, ms=6,
                label=f"recentred, Sim-{s}")
    ax.axhline(0.05, color=GREY, ls=":", lw=1.6)
    ax.set_xlabel("growth of surface decodability across depth", fontsize=sz(10))
    ax.set_ylabel("false positive rate", fontsize=sz(10))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0, .25, .5]); ax.set_xticklabels(["0%", "25%", "50%"])
    ax.legend(fontsize=sz(8.2), frameon=False, ncol=2, loc="upper left")
    save(fig, "c2_depth_slope")


def do_power(c):
    fig, ax = plt.subplots(figsize=fs((5.2, 3.0), (3.35, 2.25)))
    for s, col, mk in [("A", BLUE, "o"), ("B", GREEN, "s")]:
        v = sorted([x for x in c["power"] if x["sim"] == s],
                   key=lambda x: x["eps"])
        coords(f"Power{s}", [x["eps"] for x in v],
               [x["reject_05"] for x in v])
        for x in v:
            mac(f"Pw{s}{wd(x['eps'], 10)}", x["reject_05"])
        ax.plot([x["eps"] for x in v], [x["reject_05"] for x in v],
                "-", marker=mk, color=col, lw=2.4, ms=6, label=SIMNAME[s])
    ax.axhline(0.8, color=GREY, ls=":", lw=1.6)
    ax.axhline(0.05, color=FAINT, ls=":", lw=1.4)
    ax.set_xlabel("planted effect size $\\varepsilon$", fontsize=sz(10))
    ax.set_ylabel("power", fontsize=sz(10))
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=sz(8.6), frameon=False, loc="lower right")
    save(fig, "c3_power")


def do_profiles(c):
    fig, axes = plt.subplots(1, 2, figsize=fs((6.6, 2.7), (6.9, 2.5)), sharey=True)
    for ax, s in zip(axes, "AB"):
        p = c["profiles"][s]
        ell = np.arange(len(p["null_ba"]))
        ax.plot(ell, p["null_ba"], "-o", color=RED, lw=2.0, ms=4,
                label="observed, no signal")
        ax.plot(ell, p["null_placebo"], "--", color=GREY, lw=1.8,
                label="placebo baseline")
        ax.plot(ell, p["signal_ba"], "-o", color=BLUE, lw=2.0, ms=4,
                label="observed, signal planted")
        ax.set_title(SIMNAME[s], fontsize=sz(9), loc="left")
        ax.set_xlabel("layer $\\ell$", fontsize=sz(9))
        coords(f"ProfNull{s}", ell, p["null_ba"])
        coords(f"ProfPlc{s}", ell, p["null_placebo"])
        coords(f"ProfSig{s}", ell, p["signal_ba"])
        mac(f"ProfNullFirst{s}", p["null_ba"][0])
        mac(f"ProfNullLast{s}", p["null_ba"][-1])
        mac(f"ProfArgmax{s}", float(int(np.argmax(p["signal_ba"]))), ".0f")
    axes[0].set_ylabel("balanced accuracy", fontsize=sz(9))
    axes[1].legend(fontsize=sz(7.6), frameon=False, loc="lower left")
    save(fig, "c4_profiles")


def do_size_and_key(c):
    for s, o in c["baseline_size"].items():
        mac(f"SizeMatched{s}", o["matched"]["reject_05"])
        mac(f"SizeHalf{s}", o["half"]["reject_05"])
    v = sorted(c["key_sensitivity"], key=lambda x: x["key_noise"])
    coords("KeyFPR", [x["key_noise"] for x in v],
           [x["fpr"]["reject_05"] for x in v])
    coords("KeyPower", [x["key_noise"] for x in v],
           [x["power"]["reject_05"] for x in v])
    for x in v:
        t = wd(x["key_noise"], 10)
        mac(f"KeyFpr{t}", x["fpr"]["reject_05"])
        mac(f"KeyPw{t}", x["power"]["reject_05"])

    fig, ax = plt.subplots(figsize=fs((5.0, 2.8), (3.35, 2.1)))
    ax.plot([x["key_noise"] for x in v], [x["power"]["reject_05"] for x in v],
            "-o", color=BLUE, lw=2.4, ms=6, label="power ($\\varepsilon=1.5$)")
    ax.plot([x["key_noise"] for x in v], [x["fpr"]["reject_05"] for x in v],
            "-s", color=RED, lw=2.4, ms=6, label="false positive rate")
    ax.axhline(0.05, color=GREY, ls=":", lw=1.6)
    ax.set_xlabel("noise in the analyst's surface key", fontsize=sz(10))
    ax.set_ylabel("rate", fontsize=sz(10)); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=sz(8.6), frameon=False, loc="center right")
    save(fig, "c5_key_sensitivity")


def do_timing(c):
    t = c["timing"][-1]
    mac("TimeSetup", t["setup_s"], ".2f")
    mac("TimePerPerm", t["per_perm_ms"], ".2f")
    mac("TimeB", t["proj_B2000_s"], ".1f")
    mac("TimeRefitH", t["proj_B2000_refit_h"], ".1f")
    mac("TimeSpeedup", t["refit_per_perm_ms"] / t["per_perm_ms"], ".0f")
    mac("TimeN", float(t["n"] // 2), ".0f")
    mac("TimeD", float(t["d"]), ".0f")


def do_tables(c, b):
    L = ["% Auto-generated by render_all.py. Do not edit by hand.", ""]
    v = sorted([x for x in b["null_calibration"] if x["sim"] == "A"],
               key=lambda x: x["p_nuis"])
    L += [r"\begin{table}[t]", r"\small", r"\centering",
          r"\setlength{\tabcolsep}{4pt}",
          r"\begin{tabular}{@{}l" + "c" * len(v) + r"@{}}", r"\toprule",
          r"Statistic & \multicolumn{%d}{c}{control set dimension $p$} \\"
          % len(v), r"\cmidrule(l){2-%d}" % (len(v) + 1),
          " & " + " & ".join(str(x["p_nuis"]) for x in v) + r" \\", r"\midrule"]
    for k, lab in [("p_naive", "raw accuracy, no control"),
                   ("p_maxlevel", r"$\max_\ell \Delta_\ell$ (level)"),
                   ("p_meanlevel", r"$\overline{\Delta_\ell}$ (level)")]:
        L.append(lab + " & " + " & ".join(f"{x[k]['reject_05']:.3f}"
                                          for x in v) + r" \\")
    L += [r"\addlinespace",
          r"\textbf{depth contrast} & " + " & ".join(
              f"\\textbf{{{x['p_perm']['reject_05']:.3f}}}" for x in v)
          + r" \\", r"\midrule",
          r"$\BA_{\nuis}$ & " + " & ".join(f"{x['ba_nuis']:.3f}" for x in v)
          + r" \\", r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{C1.} False positive rate under a true null as the "
          r"nuisance control set grows, Sim-A, nominal $0.05$. The level of "
          r"excess separability inherits the control set's estimation "
          r"capacity; the zero-sum contrast cancels it algebraically.}",
          r"\label{tab:c1}", r"\end{table}", ""]

    d = c["depth_slope"]
    L += [r"\begin{table}[t]", r"\small", r"\centering",
          r"\setlength{\tabcolsep}{5pt}",
          r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
          r"Simulator & depth slope & uncorrected & recentred \\", r"\midrule"]
    for s in "AB":
        for i, x in enumerate(sorted([z for z in d if z["sim"] == s],
                                     key=lambda z: z["slope"])):
            nm = f"Sim-{s}" if i == 0 else ""
            L.append(f"{nm} & {x['slope']:.0%} & "
                     f"{x['raw']['reject_05']:.3f} & "
                     f"\\textbf{{{x['adj']['reject_05']:.3f}}} \\\\"
                     .replace("%", r"\%"))
        if s == "A":
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{C2.} False positive rate under a true null when "
          r"surface decodability grows with depth. Recentring on the "
          r"level-matched placebo holds the nominal rate where the "
          r"uncorrected contrast does not.}",
          r"\label{tab:c2}", r"\end{table}", ""]
    open(GENERATED / "results_tables.tex", "w").write("\n".join(L) + "\n")
    print(f"  {GENERATED}/results_tables.tex")


def main():
    global PAPER, SUF
    c = json.load(open(VAL_RESULTS / "phase0c_results.json"))
    b = json.load(open(VAL_RESULTS / "phase0b_results.json"))
    print("rendering:")
    do_properties(c)
    do_capacity(b)
    do_slope(c)
    do_power(c)
    do_profiles(c)
    do_size_and_key(c)
    do_timing(c)
    mac("RunMinutes", c.get("wallclock_s", 0) / 60.0 +
        b.get("wallclock_s", 0) / 60.0, ".0f")
    mac("NPerSet", float(c["config"]["n_per_set"]), ".0f")
    mac("BPerm", float(c["config"]["B"]), ".0f")
    do_tables(c, b)
    open(GENERATED / "results_macros.tex", "w").write(
        "% Auto-generated by render_all.py. Every value is measured.\n"
        + "\n".join(MAC) + "\n")
    print(f"  results_macros.tex ({len(MAC)} macros)")
    print("paper variants:")
    PAPER, SUF = True, "_paper"
    do_capacity(b); do_slope(c); do_power(c); do_profiles(c)
    do_size_and_key(c)


if __name__ == "__main__":
    main()
