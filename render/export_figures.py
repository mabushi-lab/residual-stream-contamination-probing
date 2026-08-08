"""
export_figures.py
Render the Phase 0 figures as standalone images from phase0_results.json.

Produces, in figures/:
  fig1_capacity_bias      the headline result
  fig2_depth_assumption   the limitation that constrains the method
  fig3_power              power against effect size
  fig4_depth_profile      recovered profile vs planted
  fig5_exposure           exposure calibration curve
  thumbnail_capacity_bias 1600x900, large type, for a repository thumbnail

Each as 300 dpi PNG and as vector PDF. Same data as the pgfplots figures in
the paper; matplotlib is used here only so the images stand alone.

    python3 export_figures.py
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
RES = s(VAL_RESULTS / "phase0_results.json")

INK = "#1a1a1a"
GREY = "#8a8a8a"
BLUE = "#1f6feb"      # the statistic we keep
RED = "#c0392b"       # the statistic we discarded
ORANGE = "#e07b39"
PURPLE = "#7d5ba6"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.7,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext, kw in [("png", {"dpi": 300}), ("pdf", {})]:
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight",
                    pad_inches=0.06, **kw)
    plt.close(fig)
    print(f"  {OUT}/{name}.png  {OUT}/{name}.pdf")


def fig1(r):
    v1 = sorted(r["v1_null_capacity"], key=lambda x: x["p_nuis"])
    x = [v["p_nuis"] for v in v1]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    series = [
        ("p_naive", "raw probe accuracy (no control)", PURPLE, "d", "--"),
        ("p_maxlevel", r"$\max_\ell\,\Delta_\ell$  (level)", RED, "s", "-"),
        ("p_meanlevel", r"mean$_\ell\,\Delta_\ell$  (level)", ORANGE, "^", "-"),
        ("p_contrast", "depth-profile contrast", BLUE, "o", "-"),
    ]
    for k, lab, c, m, ls in series:
        ax.plot(x, [v[k]["reject_05"] for v in v1], ls, color=c, marker=m,
                lw=2.2, ms=6, label=lab, zorder=3)
    ax.axhline(0.05, color=GREY, ls=":", lw=1.6, zorder=2)
    ax.text(x[-1] * 0.98, -0.045, "nominal 0.05", color=GREY, fontsize=8.5,
            ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel("dimension of the declared nuisance block $p$", fontsize=10)
    ax.set_ylabel("false positive rate", fontsize=10)
    ax.set_ylim(-0.10, 1.42)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axvspan(96, max(x) * 1.08, color="#c0392b", alpha=0.05, zorder=0)
    ax.annotate("activations are 96-dim here;\nreal nuisance families are larger",
                xy=(430, 0.62), xytext=(112, 0.30), fontsize=8.2, color=INK,
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
    ax.legend(fontsize=8.4, frameon=False, loc="upper left", ncol=2,
              bbox_to_anchor=(-0.015, 1.03), columnspacing=1.1,
              handlelength=1.9)
    ax.set_title("Under a true null, the level statistic tracks the size of "
                 "the control set", fontsize=9.5, loc="left", pad=8)
    save(fig, "fig1_capacity_bias")


def fig2(r):
    v2 = sorted(r["v2_depth_nuisance"], key=lambda x: x["nuis_slope"])
    x = [v["nuis_slope"] for v in v2]
    y = [v["false_positive_rate"]["reject_05"] for v in v2]
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    ax.plot(x, y, "-o", color=RED, lw=2.2, ms=6, zorder=3)
    ax.axhline(0.05, color=GREY, ls=":", lw=1.6)
    ax.text(0.005, 0.085, "nominal 0.05", color=GREY, fontsize=8.5)
    ax.set_xlabel("growth of nuisance decodability from layer 0 to layer $L$",
                  fontsize=10)
    ax.set_ylabel("false positive rate", fontsize=10)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.0%}" for v in x])
    ax.set_title("The contrast assumes surface separability is flat in depth",
                 fontsize=9.5, loc="left", pad=8)
    save(fig, "fig2_depth_assumption")


def fig3(r):
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    style = {"250": (ORANGE, "s"), "500": (BLUE, "o"), "1000": (PURPLE, "^")}
    for n, rec in sorted(r["v3_power"].items(), key=lambda kv: int(kv[0])):
        rec = sorted(rec, key=lambda v: v["eps"])
        c, m = style.get(n, (GREY, "o"))
        ax.plot([v["eps"] for v in rec], [v["reject_05"] for v in rec],
                "-", marker=m, color=c, lw=2.2, ms=5.5,
                label=f"$n={n}$ per set", zorder=3)
    ax.axhline(0.8, color=GREY, ls=":", lw=1.6)
    ax.text(0.01, 0.83, "power 0.8", color=GREY, fontsize=8.5)
    ax.set_xlabel("planted effect size $\\varepsilon$", fontsize=10)
    ax.set_ylabel("power", fontsize=10)
    ax.set_ylim(-0.04, 1.06)
    ax.legend(fontsize=8.6, frameon=False, loc="upper left")
    ax.set_title("Power of the depth contrast", fontsize=9.5, loc="left", pad=8)
    save(fig, "fig3_power")


def fig4(r):
    p = r["v4_profile"]
    ell = np.arange(len(p["ba_mean"]))
    mu, sd = np.array(p["ba_mean"]), np.array(p["ba_sd"])
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    ax.fill_between(ell, mu - sd, mu + sd, color=BLUE, alpha=0.15, lw=0)
    ax.plot(ell, mu, "-o", color=BLUE, lw=2.2, ms=5, label=r"$BA_\ell$  (mean $\pm$ 1 sd)")
    ax.axhline(p["ba_nuis_mean"], color=RED, ls="--", lw=1.8,
               label=r"$BA_{\mathrm{nuis}}$")
    ax.axvline(p["planted_peak_layer"], color=GREY, ls=":", lw=1.4)
    ax.text(p["planted_peak_layer"] + 0.15, mu.min(), "planted peak",
            color=GREY, fontsize=8.2, rotation=90, va="bottom")
    ax.set_xlabel("layer $\\ell$", fontsize=10)
    ax.set_ylabel("balanced accuracy", fontsize=10)
    ax.set_xlim(0, ell[-1])
    ax.legend(fontsize=8.6, frameon=False, loc="lower right")
    ax.set_title("Recovered depth profile: nuisance is flat, the signal rises",
                 fontsize=9.5, loc="left", pad=8)
    save(fig, "fig4_depth_profile")


def fig5(r):
    e = r["v5_exposure"]
    m = np.array(e["m_grid"])
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    ax.errorbar(m, e["T_mean"], yerr=np.array(e["T_sd"]) / np.sqrt(40),
                fmt="o", color=INK, ms=5, capsize=3, lw=1.2, label="measured",
                zorder=3)
    grid = np.linspace(0, m.max(), 200)
    ax.plot(grid, e["fit_gamma"] * (1 - np.exp(-e["fit_beta"] * grid)),
            "-", color=BLUE, lw=2.2,
            label=r"fit $\gamma(1-e^{-\beta m})$", zorder=2)
    ax.axvline(e["detection_floor_m"], color=RED, ls="--", lw=1.6)
    ax.text(e["detection_floor_m"] + 0.6, 0.001,
            f"detection floor\n$m^*={e['detection_floor_m']:.1f}$",
            color=RED, fontsize=8.2, va="bottom")
    ax.set_xlabel("duplication count $m$", fontsize=10)
    ax.set_ylabel("contrast $T_c$", fontsize=10)
    ax.legend(fontsize=8.6, frameon=False, loc="lower right")
    ax.set_title("Exposure calibration: the statistic inverts to an "
                 "exposure interval", fontsize=9.5, loc="left", pad=8)
    save(fig, "fig5_exposure")


def thumbnail(r):
    """1600 x 900, large type, readable at card size."""
    v1 = sorted(r["v1_null_capacity"], key=lambda x: x["p_nuis"])
    x = [v["p_nuis"] for v in v1]
    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_axes([0.105, 0.185, 0.735, 0.575])

    ax.plot(x, [v["p_maxlevel"]["reject_05"] for v in v1], "-s", color=RED,
            lw=6, ms=19, zorder=3,
            label="probe accuracy in excess of a control set")
    ax.plot(x, [v["p_contrast"]["reject_05"] for v in v1], "-o", color=BLUE,
            lw=6, ms=19, zorder=4,
            label="depth-profile contrast (this paper)")
    ax.axhline(0.05, color=GREY, ls=":", lw=3.5, zorder=2)

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x], fontsize=23)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=23)
    ax.set_xlim(x[0] * 0.88, x[-1] * 1.12)
    ax.set_ylim(-0.06, 1.14)
    ax.set_xlabel("size of the declared nuisance control set (dimensions)",
                  fontsize=25, labelpad=14)
    ax.set_ylabel("false positive rate", fontsize=25, labelpad=14)

    ax.text(x[0] * 0.90, 0.075, "nominal 0.05", color=GREY, fontsize=19,
            va="bottom")
    ax.annotate("always\nrejects", xy=(1.015, 1.00), xycoords="axes fraction",
                color=RED, fontsize=24, ha="left", va="center",
                fontweight="bold", linespacing=1.2)
    ax.annotate("holds\ncalibration", xy=(1.015, 0.10),
                xycoords="axes fraction", color=BLUE, fontsize=24, ha="left",
                va="center", fontweight="bold", linespacing=1.2)
    ax.legend(fontsize=23, frameon=False, loc="upper left",
              bbox_to_anchor=(0.02, 1.0), handlelength=2.4)

    fig.text(0.105, 0.955,
             "The obvious contamination statistic false-positives in",
             fontsize=31, color=INK, va="top")
    fig.text(0.105, 0.888,
             "proportion to how large you made the control",
             fontsize=31, color=INK, va="top")
    fig.text(0.105, 0.038,
             "Phase 0 validation on synthetic activations. True null, "
             "no planted signal, 150 replicates per point.",
             fontsize=17, color=GREY)

    for ext, kw in [("png", {"dpi": 100}), ("pdf", {})]:
        fig.savefig(f"{OUT}/thumbnail_capacity_bias.{ext}", **kw)
    plt.close(fig)
    print(f"  {OUT}/thumbnail_capacity_bias.png (1600x900)")

def main():
    r = json.load(open(RES))
    os.makedirs(OUT, exist_ok=True)
    print("writing figures:")
    fig1(r); fig2(r); fig3(r); fig4(r); fig5(r); thumbnail(r)


if __name__ == "__main__":
    main()
