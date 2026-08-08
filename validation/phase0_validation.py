"""
phase0_validation.py
Phase 0 of the RSCP validation programme: does the inference procedure work?

These experiments run on SYNTHETIC activations with a planted direction of
known size. They validate the statistical machinery. They say nothing about
whether any real language model carries a familiarity direction; that is
Phases 1 to 5 and needs a GPU.

Generative model
----------------
For item i with membership y_i, sign s_i = 2 y_i - 1, and layer l = 0..L:

    H_i^(l) = G_i^(l) + (s_i / 2) [ kappa_l * v_nuis + eps * a(l) * v_mem ]

with G iid N(0, I_d) and v_nuis, v_mem fixed orthonormal directions.
a(0) = 0 and a(l) is a Gaussian bump peaking at l = round(0.7 L): the planted
memorisation signal is absent from the embeddings and has to be computed.
kappa_l = kappa (1 + slope * l / L) controls whether nuisance decodability is
flat in depth (slope = 0, the assumption the primary test needs) or grows.

The declared nuisance family is

    Psi_i = G^psi_i + (s_i / 2) * kappa * rho * u_psi        (p_nuis dims)

rho in [0, 1] is the NUISANCE COVERAGE, the fraction of the nuisance
separation the analyst's declared family captures. p_nuis is varied to expose
the capacity bias.

Experiments
-----------
V1 null calibration and the capacity bias, across nuisance block dimension
V2 false positives when nuisance decodability grows with depth
V3 power of the contrast against effect size and sample size
V4 recovery of the planted depth profile
V5 exposure calibration, detection floor, inversion coverage
V6 timing of the smoother shortcut

    python3 phase0_validation.py --quick    # smoke test
    python3 phase0_validation.py            # full run

Writes phase0_results.json and phase0_pgfplots.tex.
"""

from __future__ import annotations

import argparse
import json
import sys
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))
from paths import VAL_RESULTS, GENERATED, FIGURES, RUNS, DATA, CACHE, EXPERIMENTS, s


from rscp import (
    _ba_from_c,
    _make_folds,
    correctness_vector,
    cross_fit_smoother,
    detection_floor,
    exposure_link,
    fit_exposure_link,
    invert_exposure,
    lam_for_df,
    profile_weights,
    time_permutation_strategies,
)

N_FOLDS = 5
N_SEEDS = 2
TARGET_DF = 24.0
D = 96
L = 12
PEAK = round(0.7 * L)
KAPPA = 1.0          # Bayes-optimal nuisance BA = Phi(kappa/2) = 0.691
P_NUIS_DEFAULT = 96  # dimension-matched unless a sweep says otherwise


def layer_profile(L: int = L, peak: int = PEAK, width: float = 3.0) -> np.ndarray:
    a = np.exp(-0.5 * ((np.arange(L + 1) - peak) / width) ** 2)
    a[0] = 0.0
    return a / a.max()


A_PROFILE = layer_profile()
W_RAMP = profile_weights(L + 1)                       # pre-registered
W_ORACLE = (lambda a: (a - a.mean()) / np.abs(a - a.mean()).sum())(A_PROFILE)


def simulate(n_per_set, eps, rho, seed, p_nuis=P_NUIS_DEFAULT,
             nuis_slope=0.0, kappa=KAPPA, d=D):
    rng = np.random.default_rng(seed)
    N = 2 * n_per_set
    y = np.zeros(N, dtype=np.int64)
    y[:n_per_set] = 1
    y = rng.permutation(y)
    s = (2.0 * y - 1.0)[:, None] / 2.0

    Q, _ = np.linalg.qr(rng.standard_normal((d, 2)))
    v_nuis, v_mem = Q[:, 0], Q[:, 1]

    layers = []
    for l in range(L + 1):
        k_l = kappa * (1.0 + nuis_slope * l / L)
        shift = k_l * v_nuis + eps * A_PROFILE[l] * v_mem
        layers.append(rng.standard_normal((N, d)) + s * shift[None, :])

    u = rng.standard_normal(p_nuis)
    u /= np.linalg.norm(u)
    nuis = rng.standard_normal((N, p_nuis)) + s * (kappa * rho) * u[None, :]
    return layers, nuis, y


@dataclass
class Rep:
    p_contrast: float      # PRIMARY: depth-profile contrast, ramp weights
    p_oracle: float        # contrast with oracle weights, upper bound
    p_maxlevel: float      # max_l (BA_l - BA_nuis), descriptive comparator
    p_meanlevel: float     # mean_l (BA_l - BA_nuis), descriptive comparator
    p_naive: float         # max_l BA_l against chance, no nuisance handling
    T_contrast: float
    ba_nuis: float
    ba_by_layer: list
    delta_mean: float


def _build(X, N):
    lam = lam_for_df(X, TARGET_DF)
    return [cross_fit_smoother(X, _make_folds(N, N_FOLDS,
            np.random.default_rng(s)), lam) for s in range(N_SEEDS)]


def one_rep(n_per_set, eps, rho, seed, B, p_nuis=P_NUIS_DEFAULT, nuis_slope=0.0):
    layers, nuis, y = simulate(n_per_set, eps, rho, seed,
                               p_nuis=p_nuis, nuis_slope=nuis_slope)
    N = len(y)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)

    c_nuis = correctness_vector(_build(nuis, N), y)
    C = [correctness_vector(_build(H, N), y) for H in layers]      # 0..L

    ba = np.array([_ba_from_c(c, pos, neg) for c in C])
    ba_n = _ba_from_c(c_nuis, pos, neg)
    delta = ba - ba_n

    rng = np.random.default_rng(seed + 11)
    ip = rng.integers(0, len(pos), (B, len(pos)))
    ine = rng.integers(0, len(neg), (B, len(neg)))
    P, Ng = pos[ip], neg[ine]
    bab = np.stack([0.5 * (c[P].mean(1) + c[Ng].mean(1)) for c in C])   # (L+1,B)
    ban_b = 0.5 * (c_nuis[P].mean(1) + c_nuis[Ng].mean(1))
    dlt_b = bab - ban_b[None, :]

    def boot_p(obs, bt):
        return float((1.0 + np.sum(bt - obs >= obs)) / (1.0 + B))

    p_c = boot_p(float(W_RAMP @ ba), W_RAMP @ bab)
    p_o = boot_p(float(W_ORACLE @ ba), W_ORACLE @ bab)

    # Level comparators, studentised max and mean over layers 1..L.
    d1, db1 = delta[1:], dlt_b[1:]
    se = db1.std(axis=1, ddof=1)
    se[se < 1e-12] = 1e-12
    tmax = float((d1 / se).max())
    tnull = ((db1 - d1[:, None]) / se[:, None]).max(axis=0)
    p_max = float((1.0 + np.sum(tnull >= tmax)) / (1.0 + B))
    p_mean = boot_p(float(d1.mean()), db1.mean(axis=0))

    # Naive: best raw layer accuracy against chance, nuisance ignored.
    m = float(ba[1:].max())
    p_naive = float((1.0 + np.sum(
        (bab[1:] - ba[1:, None] + 0.5).max(axis=0) >= m)) / (1.0 + B))

    return Rep(p_c, p_o, p_max, p_mean, p_naive, float(W_RAMP @ ba),
               float(ba_n), [float(v) for v in ba], float(d1.mean()))


def _job(a):
    return one_rep(*a)


def run_many(n, eps, rho, R, B, seed0=0, workers=4, **kw):
    jobs = [(n, eps, rho, seed0 + r, B, kw.get("p_nuis", P_NUIS_DEFAULT),
             kw.get("nuis_slope", 0.0)) for r in range(R)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_job, jobs, chunksize=max(1, R // (workers * 4))))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def rate(reps, attr, R):
    k = int(sum(getattr(r, attr) < 0.05 for r in reps))
    return {"reject_05": k / R, "ci": wilson(k, R)}


# --------------------------------------------------------------------------

def v1_null_capacity(p_grid, R, B, n, workers):
    out = []
    for p_nuis in p_grid:
        reps = run_many(n, 0.0, 1.0, R, B, seed0=1000 + p_nuis,
                        workers=workers, p_nuis=p_nuis)
        rec = {"p_nuis": p_nuis, "R": R,
               "ba_nuis": float(np.mean([r.ba_nuis for r in reps])),
               "mean_delta": float(np.mean([r.delta_mean for r in reps])),
               "mean_T_contrast": float(np.mean([r.T_contrast for r in reps]))}
        for a in ["p_contrast", "p_oracle", "p_maxlevel",
                  "p_meanlevel", "p_naive"]:
            rec[a] = rate(reps, a, R)
        out.append(rec)
    return out


def v2_depth_varying_nuisance(slope_grid, R, B, n, workers):
    out = []
    for sl in slope_grid:
        reps = run_many(n, 0.0, 1.0, R, B, seed0=2000 + int(100 * sl),
                        workers=workers, nuis_slope=sl)
        out.append({"nuis_slope": sl, "R": R,
                    "false_positive_rate": rate(reps, "p_contrast", R),
                    "mean_T_contrast": float(np.mean([r.T_contrast for r in reps]))})
    return out


def v3_power(eps_grid, n_grid, R, B, workers):
    out = {}
    for n in n_grid:
        rec = []
        for e in eps_grid:
            reps = run_many(n, e, 1.0, R, B, seed0=3000 + int(1000 * e),
                            workers=workers)
            rec.append({"eps": e, **rate(reps, "p_contrast", R),
                        "mean_T": float(np.mean([r.T_contrast for r in reps]))})
        out[str(n)] = rec
    return out


def v4_profile(eps, R, B, n, workers):
    reps = run_many(n, eps, 1.0, R, B, seed0=4000, workers=workers)
    M = np.array([r.ba_by_layer for r in reps])
    return {"eps": eps, "R": R, "n_per_set": n,
            "planted_profile": [float(v) for v in A_PROFILE],
            "planted_peak_layer": int(PEAK),
            "ba_mean": [float(v) for v in M.mean(0)],
            "ba_sd": [float(v) for v in M.std(0)],
            "ba_nuis_mean": float(np.mean([r.ba_nuis for r in reps])),
            "weights_ramp": [float(v) for v in W_RAMP]}


def v5_exposure(R, B, n, workers):
    eps_max, beta_true = 0.9, 0.15
    m_cal = np.array([0, 1, 2, 4, 8, 16, 32], dtype=float)
    T, sd, pw = [], [], []
    for m in m_cal:
        e = eps_max * (1 - math.exp(-beta_true * m))
        reps = run_many(n, e, 1.0, R, B, seed0=5000 + int(m), workers=workers)
        t = np.array([r.T_contrast for r in reps])
        T.append(float(t.mean()))
        sd.append(float(t.std()))
        pw.append(float(np.mean([r.p_contrast < 0.05 for r in reps])))
    gamma, beta = fit_exposure_link(m_cal, np.array(T))
    sigma = float(np.mean(sd))
    cov = []
    for mt in [1.0, 3.0, 6.0, 12.0, 24.0]:
        e = eps_max * (1 - math.exp(-beta_true * mt))
        reps = run_many(n, e, 1.0, max(20, R // 3), B, seed0=6000 + int(mt),
                        workers=workers)
        hit = 0
        for r in reps:
            lo, hi = invert_exposure(r.T_contrast, gamma, beta, sigma)
            if not math.isnan(lo) and lo <= mt <= hi:
                hit += 1
        cov.append({"m_true": mt, "coverage": hit / len(reps), "n_rep": len(reps)})
    return {"eps_max": eps_max, "beta_true": beta_true,
            "m_grid": [float(v) for v in m_cal], "T_mean": T, "T_sd": sd,
            "power": pw, "fit_gamma": gamma, "fit_beta": beta, "sigma": sigma,
            "fit_curve": [float(v) for v in exposure_link(m_cal, gamma, beta)],
            "detection_floor_m": detection_floor(m_cal, pw, 0.8),
            "inversion_coverage": cov}


def v6_timing():
    out = []
    for (n_set, d) in [(500, 512), (1000, 2048)]:
        rng = np.random.default_rng(0)
        y = np.zeros(2 * n_set, dtype=int)
        y[:n_set] = 1
        X = rng.standard_normal((2 * n_set, d))
        r = time_permutation_strategies(X, y, n_perm=(20 if d <= 512 else 4), lam=100.0)
        r["shortcut_per_perm_ms"] = 1e3 * r["shortcut_s"] / r["n_perm"]
        r["refit_per_perm_ms"] = 1e3 * r["refit_s"] / r["n_perm"]
        r["projected_B2000_shortcut_s"] = 2000 * r["shortcut_s"] / r["n_perm"]
        r["projected_B2000_refit_s"] = 2000 * r["refit_s"] / r["n_perm"]
        out.append(r)
    return out


# --------------------------------------------------------------------------

def coords(xs, ys, prec=4):
    return " ".join(f"({x:.{prec}g},{y:.{prec}g})" for x, y in zip(xs, ys))


def emit(res, path):
    L_ = ["% Auto-generated by phase0_validation.py. Every coordinate is a",
          "% measured value from that run. Do not edit by hand.", ""]
    v1 = res["v1_null_capacity"]
    xs = [r["p_nuis"] for r in v1]
    for key, cmd in [("p_contrast", "Contrast"), ("p_maxlevel", "MaxLevel"),
                     ("p_meanlevel", "MeanLevel"), ("p_naive", "Naive")]:
        L_.append(f"\\newcommand{{\\PzCap{cmd}}}{{"
                  + coords(xs, [r[key]["reject_05"] for r in v1]) + "}")
    L_.append("")
    v2 = res["v2_depth_nuisance"]
    L_ += ["\\newcommand{\\PzSlope}{" + coords(
        [r["nuis_slope"] for r in v2],
        [r["false_positive_rate"]["reject_05"] for r in v2]) + "}", ""]
    for n, rec in res["v3_power"].items():
        rec = sorted(rec, key=lambda r: r["eps"])
        L_.append(f"\\newcommand{{\\PzPowerN{n}}}{{"
                  + coords([r["eps"] for r in rec],
                           [r["reject_05"] for r in rec]) + "}")
    L_.append("")
    p = res["v4_profile"]
    ell = list(range(len(p["ba_mean"])))
    L_ += ["\\newcommand{\\PzBaMean}{" + coords(ell, p["ba_mean"]) + "}",
           "\\newcommand{\\PzBaUpper}{" + coords(
               ell, np.array(p["ba_mean"]) + np.array(p["ba_sd"])) + "}",
           "\\newcommand{\\PzBaLower}{" + coords(
               ell, np.array(p["ba_mean"]) - np.array(p["ba_sd"])) + "}",
           "\\newcommand{\\PzBaNuis}{" + coords(
               [ell[0], ell[-1]], [p["ba_nuis_mean"]] * 2) + "}", ""]
    e = res["v5_exposure"]
    L_ += ["\\newcommand{\\PzExpObs}{" + coords(e["m_grid"], e["T_mean"]) + "}",
           "\\newcommand{\\PzExpFit}{" + coords(e["m_grid"], e["fit_curve"]) + "}",
           "\\newcommand{\\PzExpPower}{" + coords(e["m_grid"], e["power"]) + "}", ""]
    # Scalar macros, so every number quoted in the paper traces to this run
    # rather than to a hand transcription.
    def mac(name, val, fmt=".3f"):
        L_.append(f"\\newcommand{{\\{name}}}{{{format(val, fmt)}}}")
    L_.append("% ---- scalars quoted in the text ----")
    by_p = {r["p_nuis"]: r for r in v1}
    for pn in by_p:
        tag = {32: "Small", 96: "Match", 400: "Large", 1200: "Huge"}.get(pn, str(pn))
        mac(f"PzFpContrast{tag}", by_p[pn]["p_contrast"]["reject_05"])
        mac(f"PzFpMax{tag}", by_p[pn]["p_maxlevel"]["reject_05"])
        mac(f"PzFpMean{tag}", by_p[pn]["p_meanlevel"]["reject_05"])
        mac(f"PzFpNaive{tag}", by_p[pn]["p_naive"]["reject_05"])
        mac(f"PzBAn{tag}", by_p[pn]["ba_nuis"], ".3f")
        mac(f"PzDelta{tag}", by_p[pn]["mean_delta"], "+.4f")
    for r in v2:
        tag = {0.0:"Zero",0.05:"Five",0.10:"Ten",0.25:"TwentyFive",0.5:"Fifty"}[round(r["nuis_slope"],2)]
        mac(f"PzSlopeFp{tag}", r["false_positive_rate"]["reject_05"])
    for n, rec in res["v3_power"].items():
        for r in rec:
            mac(f"PzPw{n}E{int(round(r['eps']*10)):02d}", r["reject_05"])
    mac("PzNullMin", min(r["p_contrast"]["reject_05"] for r in v1))
    mac("PzNullMax", max(r["p_contrast"]["reject_05"] for r in v1))
    mac("PzFloor", res["v5_exposure"]["detection_floor_m"], ".1f")
    mac("PzGamma", res["v5_exposure"]["fit_gamma"], ".4f")
    mac("PzBeta", res["v5_exposure"]["fit_beta"], ".3f")
    cv = res["v5_exposure"]["inversion_coverage"]
    mac("PzCovMin", min(c["coverage"] for c in cv))
    mac("PzCovMax", max(c["coverage"] for c in cv))
    t = res["v6_timing"][-1]
    mac("PzTimeSetup", t["setup_s"], ".2f")
    mac("PzTimeShort", t["projected_B2000_shortcut_s"], ".2f")
    mac("PzTimeRefit", t["projected_B2000_refit_s"], ".0f")
    mac("PzSpeedup", t["refit_per_perm_ms"] / t["shortcut_per_perm_ms"], ".0f")
    mac("PzWall", res.get("wallclock_s", 0.0) / 60.0, ".0f")
    mac("PzR", float(v1[0]["R"]), ".0f")
    mac("PzB", float(res["config"]["B_bootstrap"]), ".0f")
    open(path, "w").write("\n".join(L_) + "\n")


RES_PATH = s(VAL_RESULTS / "phase0_results.json")


def _load():
    if os.path.exists(RES_PATH):
        return json.load(open(RES_PATH))
    return {}


def _save(res):
    json.dump(res, open(RES_PATH, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    help="'all' runs every stage in order; individual stages "
                         "exist so the run can be checkpointed on a machine "
                         "with a short job limit")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    W = a.workers
    st = a.stage
    if st == "all":
        stages = (["init"]
                  + [f"v1{p}" for p in (32, 96, 400, 1200)]
                  + [f"v2{int(sl*100)}" for sl in (0, 0.05, 0.10, 0.25, 0.50)]
                  + [f"v3{n}_{c}" for n in (250, 500) for c in range(4)]
                  + ["v4"]
                  + [f"v5cal{m}" for m in (0, 1, 2, 4, 8, 16, 32)]
                  + ["v5fit", "v6", "emit"])
        t_all = time.time()
        for k, sub in enumerate(stages, 1):
            print(f"[{k}/{len(stages)}] {sub}", flush=True)
            sys.argv = [sys.argv[0], "--stage", sub, "--workers", str(W)]
            main()
        print(f"\nall stages done in {(time.time()-t_all)/60:.1f} min")
        return

    res = _load()
    B = 300
    t0 = time.time()

    if st == "init":
        res["config"] = {
            "target_df": TARGET_DF, "n_folds": N_FOLDS, "fold_seeds": N_SEEDS,
            "d": D, "L": L, "peak_layer": PEAK, "kappa": KAPPA,
            "bayes_optimal_ba_nuisance": float(
                0.5 * (1 + math.erf((KAPPA / 2) / math.sqrt(2)))),
            "B_bootstrap": B, "alpha": 0.05,
            "n_per_set_null": 300, "R_null": 150, "R_slope": 150,
            "R_power": 100, "R_profile": 200, "R_exposure": 40,
            "weights_ramp": [float(v) for v in W_RAMP]}
        res["wallclock_s"] = 0.0

    elif st.startswith("v1"):
        pn = int(st[2:])
        r = v1_null_capacity([pn], 150, B, 300, W)[0]
        res.setdefault("v1_null_capacity", [])
        res["v1_null_capacity"] = [x for x in res["v1_null_capacity"]
                                   if x["p_nuis"] != pn] + [r]
        res["v1_null_capacity"].sort(key=lambda x: x["p_nuis"])
        print(f"   p_nuis={pn} BAn={r['ba_nuis']:.4f} "
              f"contrast={r['p_contrast']['reject_05']:.3f} "
              f"max={r['p_maxlevel']['reject_05']:.3f} "
              f"mean={r['p_meanlevel']['reject_05']:.3f} "
              f"naive={r['p_naive']['reject_05']:.3f} "
              f"meanDelta={r['mean_delta']:+.4f}")

    elif st.startswith("v2"):
        sl = float(st[2:]) / 100.0
        r = v2_depth_varying_nuisance([sl], 150, B, 300, W)[0]
        res.setdefault("v2_depth_nuisance", [])
        res["v2_depth_nuisance"] = [x for x in res["v2_depth_nuisance"]
                                    if abs(x["nuis_slope"] - sl) > 1e-9] + [r]
        res["v2_depth_nuisance"].sort(key=lambda x: x["nuis_slope"])
        print(f"   slope={sl:.2f} FPR={r['false_positive_rate']['reject_05']:.3f}")

    elif st.startswith("v3"):
        n, chunk = (int(x) for x in st[2:].split("_"))
        chunks = {0: [0.0, 0.1], 1: [0.2, 0.3], 2: [0.4, 0.6], 3: [0.8]}
        R = 80 if n <= 500 else 50
        rec = v3_power(chunks[chunk], [n], R, B, W)[str(n)]
        res.setdefault("v3_power", {}).setdefault(str(n), [])
        keep = [x for x in res["v3_power"][str(n)]
                if all(abs(x["eps"] - e) > 1e-9 for e in chunks[chunk])]
        res["v3_power"][str(n)] = sorted(keep + rec, key=lambda x: x["eps"])
        print("   ", [(r["eps"], r["reject_05"]) for r in rec])

    elif st == "v4":
        res["v4_profile"] = v4_profile(0.5, 200, B, 400, W)
        print("   peak recovered at layer",
              int(np.argmax(res["v4_profile"]["ba_mean"])))

    elif st.startswith("v5cal"):
        m = float(st[5:])
        eps_max, beta_true = 0.9, 0.15
        e = eps_max * (1 - math.exp(-beta_true * m))
        reps = run_many(400, e, 1.0, 40, B, seed0=5000 + int(m), workers=W)
        t = np.array([r.T_contrast for r in reps])
        res.setdefault("v5_raw", {})
        res["v5_raw"][str(m)] = {
            "T_mean": float(t.mean()), "T_sd": float(t.std()),
            "power": float(np.mean([r.p_contrast < 0.05 for r in reps])),
            "T_all": [float(v) for v in t]}
        print(f"   m={m:.0f} T={t.mean():+.4f} power="
              f"{np.mean([r.p_contrast < 0.05 for r in reps]):.2f}")

    elif st == "v5fit":
        raw = res["v5_raw"]
        m_cal = np.array(sorted(float(k) for k in raw))
        T = np.array([raw[str(m)]["T_mean"] for m in m_cal])
        sd = np.array([raw[str(m)]["T_sd"] for m in m_cal])
        pw = [raw[str(m)]["power"] for m in m_cal]
        gamma, beta = fit_exposure_link(m_cal, T)
        sigma = float(sd.mean())
        cov = []
        for mt in m_cal[1:]:
            hits = 0
            for tv in raw[str(mt)]["T_all"]:
                lo, hi = invert_exposure(tv, gamma, beta, sigma)
                if not math.isnan(lo) and lo <= mt <= hi:
                    hits += 1
            cov.append({"m_true": float(mt),
                        "coverage": hits / len(raw[str(mt)]["T_all"]),
                        "n_rep": len(raw[str(mt)]["T_all"])})
        res["v5_exposure"] = {
            "eps_max": 0.9, "beta_true": 0.15,
            "m_grid": [float(v) for v in m_cal],
            "T_mean": [float(v) for v in T], "T_sd": [float(v) for v in sd],
            "power": pw, "fit_gamma": gamma, "fit_beta": beta, "sigma": sigma,
            "fit_curve": [float(v) for v in exposure_link(m_cal, gamma, beta)],
            "detection_floor_m": detection_floor(m_cal, pw, 0.8),
            "inversion_coverage": cov}
        print(f"   gamma={gamma:.4f} beta={beta:.3f} "
              f"floor={res['v5_exposure']['detection_floor_m']:.1f} "
              f"cov={[round(c['coverage'],2) for c in cov]}")

    elif st == "v6":
        res["v6_timing"] = v6_timing()
        t = res["v6_timing"][-1]
        print(f"   setup={t['setup_s']:.2f}s shortcut/perm="
              f"{t['shortcut_per_perm_ms']:.3f}ms refit/perm="
              f"{t['refit_per_perm_ms']:.1f}ms")

    elif st == "emit":
        emit(res, "phase0_pgfplots.tex")
        print("   wrote phase0_pgfplots.tex")

    res["wallclock_s"] = res.get("wallclock_s", 0.0) + (time.time() - t0)
    _save(res)
    print(f"[{st}] {time.time()-t0:.1f}s  (cumulative "
          f"{res['wallclock_s']/60:.1f} min)")


if __name__ == "__main__":
    main()
