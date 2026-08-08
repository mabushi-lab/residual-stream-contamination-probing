"""
phase0b_validation.py
Phase 0b: the validation that goes in the paper.

Supersedes phase0_validation.py in two ways. It uses the label-permutation
null rather than the item bootstrap, because the bootstrap holds the fitted
probe fixed and is measurably anti-conservative. And it runs everything under
two structurally different simulators (see simulators.py), so no conclusion
rests on one toy generative model.

Experiments
-----------
props            measured structural differences between Sim-A and Sim-B
n1<sim><p>       null calibration vs nuisance block dimension, all statistics
n2<sim><slope>   false positives when nuisance decodability grows with depth
n3<sim><n>_<c>   power against effect size
n4<sim>          recovery of the planted depth profile, and the nuisance
                 decodability profile under a true null
n5<m>            exposure calibration (Sim-A)
n6               timing
emit             write phase0b_results.json summaries

    python3 phase0b_validation.py --stage all
    python3 phase0b_validation.py --stage n1A96      # one cell

Everything is checkpointed into phase0b_results.json so a run can be resumed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
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
    _ba_perm_batch,
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
from simulators import SIMULATORS, layer_bump

RES_PATH = s(VAL_RESULTS / "phase0b_results.json")
N_FOLDS = 5
N_SEEDS = 2
TARGET_DF = 24.0
D = 96
L = 12
KAPPA = 1.0
B_DEFAULT = 400
W_RAMP = profile_weights(L + 1)


# --------------------------------------------------------------------------

@dataclass
class Rep:
    p_perm: float        # PRIMARY: contrast vs label-permutation null
    p_boot: float        # contrast vs item bootstrap, comparator
    p_maxlevel: float    # max_l Delta_l, the statistic we discarded
    p_meanlevel: float
    p_naive: float       # best raw layer accuracy vs chance
    T: float
    ba_nuis: float
    ba_by_layer: list


def _fit(X, y, N):
    lam = lam_for_df(X, TARGET_DF)
    return [cross_fit_smoother(
                X, _make_folds(N, N_FOLDS, np.random.default_rng(s)), lam)
            for s in range(N_SEEDS)]


def one_rep(sim, n_per_set, eps, seed, B, p_nuis, nuis_slope):
    layers, nuis, y, _ = SIMULATORS[sim](
        n_per_set, eps, seed, d=D, L=L, kappa=KAPPA,
        p_nuis=p_nuis, nuis_slope=nuis_slope)
    N = len(y)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)

    SM = [_fit(H, y, N) for H in layers]
    sm_n = _fit(nuis, y, N)
    C = [correctness_vector(sm, y) for sm in SM]
    c_n = correctness_vector(sm_n, y)

    ba = np.array([_ba_from_c(c, pos, neg) for c in C])
    ba_n = float(_ba_from_c(c_n, pos, neg))
    obs = float(W_RAMP @ ba)
    delta = ba - ba_n

    rng = np.random.default_rng(seed + 11)

    # permutation null: Pi is label-free, so Pi @ y_perm refits the direction
    Y = np.stack([rng.permutation(y) for _ in range(B)], 1).astype(float)
    nb = np.stack([_ba_perm_batch(sm, Y) for sm in SM])
    nn = _ba_perm_batch(sm_n, Y)
    p_perm = float((1 + np.sum((W_RAMP @ nb) >= obs)) / (1 + B))
    dn = nb[1:] - nn[None, :]
    se = dn.std(1, ddof=1); se[se < 1e-12] = 1e-12
    p_max = float((1 + np.sum((dn / se[:, None]).max(0)
                              >= (delta[1:] / se).max())) / (1 + B))
    p_mean = float((1 + np.sum(dn.mean(0) >= delta[1:].mean())) / (1 + B))
    p_naive = float((1 + np.sum(nb[1:].max(0) >= ba[1:].max())) / (1 + B))

    # bootstrap comparator
    ip = rng.integers(0, len(pos), (B, len(pos)))
    ine = rng.integers(0, len(neg), (B, len(neg)))
    P, Ng = pos[ip], neg[ine]
    bb = W_RAMP @ np.stack([0.5 * (c[P].mean(1) + c[Ng].mean(1)) for c in C])
    p_boot = float((1 + np.sum(bb - obs >= obs)) / (1 + B))

    return Rep(p_perm, p_boot, p_max, p_mean, p_naive, obs, ba_n,
               [float(v) for v in ba])


def _job(a):
    return one_rep(*a)


def run_many(sim, n, eps, R, B, seed0, workers, p_nuis=96, nuis_slope=0.0):
    jobs = [(sim, n, eps, seed0 + r, B, p_nuis, nuis_slope) for r in range(R)]
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


def rate(reps, attr):
    R = len(reps)
    k = int(sum(getattr(r, attr) < 0.05 for r in reps))
    return {"reject_05": k / R, "ci": wilson(k, R), "R": R}


# --------------------------------------------------------------------------

def _load():
    return json.load(open(RES_PATH)) if os.path.exists(RES_PATH) else {}


def _save(r):
    json.dump(r, open(RES_PATH, "w"), indent=2)


def _upsert(lst, rec, keys):
    out = [x for x in lst if any(x[k] != rec[k] for k in keys)]
    return sorted(out + [rec], key=lambda x: tuple(str(x[k]) for k in keys))


def sim_properties():
    out = {}
    for name, fn in SIMULATORS.items():
        layers, nuis, y, a = fn(400, 0.0, 0, d=D, L=L)
        H = np.stack(layers)
        rms = np.linalg.norm(H, axis=2).mean(axis=1)
        flat = np.concatenate([l.ravel() for l in layers])
        ev = np.linalg.svd(layers[-1] - layers[-1].mean(0),
                           compute_uv=False) ** 2
        out[name] = {
            "rms_growth": float(rms[-1] / rms[0]),
            "corr_first_last": float(np.corrcoef(layers[1].ravel(),
                                                 layers[-1].ravel())[0, 1]),
            "kurtosis": float(((flat - flat.mean()) ** 4).mean()
                              / flat.var() ** 2),
            "top_eig_share": float(ev[0] / ev.sum()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--R", type=int, default=0, help="override replicate count")
    a = ap.parse_args()
    W, st = a.workers, a.stage
    B = B_DEFAULT
    res = _load()
    t0 = time.time()

    if st == "all":
        stages = (["init", "props"]
                  + [f"n1{s}{p}" for s in "AB" for p in (96, 400, 1200)]
                  + [f"n2{s}{int(v*100)}" for s in "AB" for v in (0, .1, .25, .5)]
                  + [f"n3{s}500_{c}" for s in "AB" for c in range(4)]
                  + [f"n4{s}" for s in "AB"]
                  + [f"n5{m}" for m in (0, 1, 2, 4, 8, 16, 32)]
                  + ["n5fit", "n6", "emit"])
        t = time.time()
        for i, sub in enumerate(stages, 1):
            print(f"[{i}/{len(stages)}] {sub}", flush=True)
            sys.argv = [sys.argv[0], "--stage", sub, "--workers", str(W)]
            main()
        print(f"\nall stages in {(time.time()-t)/60:.1f} min")
        return

    if st == "init":
        res["config"] = {"d": D, "L": L, "kappa": KAPPA, "n_folds": N_FOLDS,
                         "fold_seeds": N_SEEDS, "target_df": TARGET_DF,
                         "B": B, "alpha": 0.05, "n_per_set_null": 400,
                         "R_null": 150, "R_slope": 120, "R_power": 100,
                         "R_profile": 150, "R_exposure": 40,
                         "weights": [float(v) for v in W_RAMP],
                         "primary_null": "label permutation"}
        res["wallclock_s"] = 0.0

    elif st == "props":
        res["sim_properties"] = sim_properties()
        for k, v in res["sim_properties"].items():
            print(f"   Sim-{k}: " + "  ".join(f"{a}={b:.3f}"
                                              for a, b in v.items()))

    elif st.startswith("n1"):
        sim, p = st[2], int(st[3:])
        R = a.R or 150
        reps = run_many(sim, 400, 0.0, R, B, 1000 + p, W, p_nuis=p)
        rec = {"sim": sim, "p_nuis": p,
               "ba_nuis": float(np.mean([r.ba_nuis for r in reps])),
               "mean_T": float(np.mean([r.T for r in reps]))}
        for att in ["p_perm", "p_boot", "p_maxlevel", "p_meanlevel", "p_naive"]:
            rec[att] = rate(reps, att)
        res["null_calibration"] = _upsert(
            res.get("null_calibration", []), rec, ["sim", "p_nuis"])
        print(f"   Sim-{sim} p={p:5d}  perm={rec['p_perm']['reject_05']:.3f}  "
              f"boot={rec['p_boot']['reject_05']:.3f}  "
              f"max={rec['p_maxlevel']['reject_05']:.3f}  "
              f"naive={rec['p_naive']['reject_05']:.3f}  "
              f"BAn={rec['ba_nuis']:.3f}")

    elif st.startswith("n2"):
        sim, sl = st[2], int(st[3:]) / 100.0
        R = a.R or 120
        reps = run_many(sim, 400, 0.0, R, B, 2000 + int(sl * 100), W,
                        nuis_slope=sl)
        rec = {"sim": sim, "slope": sl, "fpr": rate(reps, "p_perm"),
               "mean_T": float(np.mean([r.T for r in reps]))}
        res["depth_slope"] = _upsert(res.get("depth_slope", []), rec,
                                     ["sim", "slope"])
        print(f"   Sim-{sim} slope={sl:.2f}  FPR={rec['fpr']['reject_05']:.3f}")

    elif st.startswith("n3"):
        sim = st[2]
        n, chunk = (int(x) for x in st[3:].split("_"))
        grid = {0: [0.0, 0.2], 1: [0.4, 0.5], 2: [0.6, 0.7], 3: [0.8, 1.0]}
        R = a.R or 100
        for eps in grid[chunk]:
            reps = run_many(sim, n, eps, R, B, 3000 + int(1000 * eps), W)
            rec = {"sim": sim, "n": n, "eps": eps,
                   **rate(reps, "p_perm"),
                   "mean_T": float(np.mean([r.T for r in reps]))}
            res["power"] = _upsert(res.get("power", []), rec,
                                   ["sim", "n", "eps"])
            print(f"   Sim-{sim} n={n} eps={eps:.1f}  "
                  f"power={rec['reject_05']:.3f}")

    elif st.startswith("n4"):
        sim = st[2]
        R = a.R or 150
        sig = run_many(sim, 500, 0.5, R, B, 4000, W)
        nul = run_many(sim, 500, 0.0, R, B, 4500, W)
        M = np.array([r.ba_by_layer for r in sig])
        Mn = np.array([r.ba_by_layer for r in nul])
        res.setdefault("profile", {})[sim] = {
            "eps": 0.5, "R": R,
            "planted": [float(v) for v in layer_bump(L)],
            "ba_mean": [float(v) for v in M.mean(0)],
            "ba_sd": [float(v) for v in M.std(0)],
            "ba_nuis_mean": float(np.mean([r.ba_nuis for r in sig])),
            "null_ba_mean": [float(v) for v in Mn.mean(0)],
            "null_ba_nuis": float(np.mean([r.ba_nuis for r in nul])),
        }
        print(f"   Sim-{sim} argmax={int(np.argmax(M.mean(0)))} "
              f"(planted 8);  nuisance profile under null: "
              f"{Mn.mean(0)[0]:.3f} -> {Mn.mean(0)[-1]:.3f}")

    elif st.startswith("n5fit"):
        raw = res["exposure_raw"]
        m = np.array(sorted(float(k) for k in raw))
        T = np.array([raw[str(v)]["T_mean"] for v in m])
        sd = np.array([raw[str(v)]["T_sd"] for v in m])
        pw = [raw[str(v)]["power"] for v in m]
        g, b = fit_exposure_link(m, T)
        sig = float(sd.mean())
        cov = []
        for mt in m[1:]:
            hits = sum(1 for t in raw[str(mt)]["T_all"]
                       if (lambda lh: not math.isnan(lh[0]) and lh[0] <= mt <= lh[1])(
                           invert_exposure(t, g, b, sig)))
            cov.append({"m_true": float(mt),
                        "coverage": hits / len(raw[str(mt)]["T_all"])})
        res["exposure"] = {"m_grid": [float(v) for v in m],
                           "T_mean": [float(v) for v in T],
                           "T_sd": [float(v) for v in sd], "power": pw,
                           "fit_gamma": g, "fit_beta": b, "sigma": sig,
                           "fit_curve": [float(v) for v in exposure_link(m, g, b)],
                           "detection_floor_m": detection_floor(m, pw, 0.8),
                           "inversion_coverage": cov}
        print(f"   gamma={g:.4f} beta={b:.3f} "
              f"floor={res['exposure']['detection_floor_m']:.1f} "
              f"cov={[round(c['coverage'],2) for c in cov]}")

    elif st.startswith("n5"):
        m = float(st[2:])
        eps = 0.9 * (1 - math.exp(-0.15 * m))
        R = a.R or 40
        reps = run_many("A", 400, eps, R, B, 5000 + int(m), W)
        t = np.array([r.T for r in reps])
        res.setdefault("exposure_raw", {})[str(m)] = {
            "T_mean": float(t.mean()), "T_sd": float(t.std()),
            "power": float(np.mean([r.p_perm < 0.05 for r in reps])),
            "T_all": [float(v) for v in t]}
        print(f"   m={m:.0f} T={t.mean():+.5f} "
              f"power={np.mean([r.p_perm < 0.05 for r in reps]):.2f}")

    elif st == "n6":
        out = []
        for n_set, d in [(500, 512), (1000, 2048)]:
            rng = np.random.default_rng(0)
            y = np.zeros(2 * n_set, dtype=int); y[:n_set] = 1
            X = rng.standard_normal((2 * n_set, d))
            r = time_permutation_strategies(X, y, n_perm=(20 if d <= 512 else 4))
            r["per_perm_ms"] = 1e3 * r["shortcut_s"] / r["n_perm"]
            r["refit_per_perm_ms"] = 1e3 * r["refit_s"] / r["n_perm"]
            r["proj_B2000_s"] = 2000 * r["shortcut_s"] / r["n_perm"]
            r["proj_B2000_refit_s"] = 2000 * r["refit_s"] / r["n_perm"]
            out.append(r)
        res["timing"] = out
        print(f"   setup={out[-1]['setup_s']:.2f}s  "
              f"{out[-1]['per_perm_ms']:.3f} ms/perm vs "
              f"{out[-1]['refit_per_perm_ms']:.0f} ms refit")

    elif st == "emit":
        print("   (use make_figures_b.py to render)")

    res["wallclock_s"] = res.get("wallclock_s", 0.0) + (time.time() - t0)
    _save(res)
    print(f"[{st}] {time.time()-t0:.1f}s  cumulative "
          f"{res['wallclock_s']/60:.1f} min")


if __name__ == "__main__":
    main()
