"""
phase0c_validation.py
The validation that goes in the paper. Supersedes phase0 and phase0b.

Each of the four design choices in the final protocol was forced by a
measurement, and every measurement is reproduced here under two structurally
different simulators (simulators.py).

  C1  zero-sum depth contrast, not a level      the level is biased by the
                                                dimension of the control set
  C2  level-matched placebo baseline            depth profiles are not flat,
                                                and fail in both directions
  C3  label-permutation null                    the item bootstrap holds the
                                                fitted probe fixed
  C4  |R| = 2|S|                                the baseline must be measured
                                                at the same sample size

Stages
------
props                      structural differences between Sim-A and Sim-B
cap<sim><p>                C1: null vs nuisance control dimension
slope<sim><s>              C2: null when surface decodability varies with depth
size<sim>                  C4: half-size vs matched-size baseline
power<sim>_<c>             power against planted effect size
prof<sim>                  recovered profiles, observed and baseline
key<sim><noise>            sensitivity to the quality of the surface key
emit                       nothing to compute; see make_figures_c.py

    python3 phase0c_validation.py --stage all --workers 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))
from paths import VAL_RESULTS, GENERATED, FIGURES, RUNS, DATA, CACHE, EXPERIMENTS, s


from rscp import (
    RSCPConfig,
    _ba_from_c,
    _ba_perm_batch,
    correctness_vector,
    layer_profile,
    level_matched_placebo,
    profile_weights,
    recentred_contrast_test,
    time_permutation_strategies,
)
from simulators import SIMULATORS, layer_bump, placebo_key

RES = s(VAL_RESULTS / "phase0c_results.json")
D, L, N_PER = 96, 12, 200
CFG = RSCPConfig(seeds=(0, 1), n_folds=5, target_df=24.0, n_bootstrap=300)
B = 300
W = profile_weights(L + 1)


def one_rep(sim, seed, eps, slope, key_noise, matched, n=N_PER):
    layers, nuis, y, meta = SIMULATORS[sim](
        n, eps, seed, n_ref_extra=n, d=D, L=L, nuis_slope=slope)
    rng = np.random.default_rng(seed + 11)
    S = np.flatnonzero(y == 1)
    R = np.flatnonzero(y == 0)
    key = placebo_key(meta, noise=key_noise, seed=seed)

    # Comparison arm: S against one half of R, split on the surface key so the
    # two arms are drawn from comparable regions of the covariate.
    o = np.argsort(key[R])
    Ra, Rb = R[o[:n]], R[o[n:]]
    idx = np.concatenate([S, Ra])
    yo = np.concatenate([np.ones(n, int), np.zeros(n, int)])
    ba, sms = layer_profile(layers, idx, yo, CFG)

    # Baseline arm.
    pool = np.concatenate([Ra, Rb]) if matched else Ra
    plc = level_matched_placebo(layers, pool, key, float(ba[0]), CFG)
    prof = plc.get("profile")

    raw = recentred_contrast_test(ba, sms, yo, None, B=B,
                                  rng=np.random.default_rng(seed + 12))
    adj = recentred_contrast_test(ba, sms, yo, prof, B=B,
                                  rng=np.random.default_rng(seed + 12))
    pos, neg = np.flatnonzero(yo == 1), np.flatnonzero(yo == 0)
    ba_n, _ = layer_profile([nuis], idx, yo, CFG)
    return {"p_raw": raw["p_value"], "p_adj": adj["p_value"],
            "T_raw": raw["T_raw"], "T_adj": adj["T_adj"],
            "base": adj["baseline"], "ba": [float(v) for v in ba],
            "ba_plc": None if prof is None else [float(v) for v in prof],
            "ba_nuis": float(ba_n[0]), "ba0_gap": plc.get("match_gap")}


def placebo_spread_rep(sim, seed, L_, reps=8, n=N_PER):
    """Spread of the placebo baseline contrast across split seeds, at depth L_.

    The permutation null treats the baseline as a known constant, and C2
    measured a nominal false positive rate with the placebo re-estimated
    inside every replicate, so at these settings the treatment is adequate.
    On the real 49-layer model it is not: the baseline's standard deviation
    across split seeds was 0.0057, half the size of the statistic it corrects,
    and a significant result there survived only because the search happened
    to land on a low draw.

    This isolates depth as the candidate explanation. Everything else is held
    at the Phase 0c defaults, and only the number of layers moves. If the
    spread grows with depth, the mechanism is that the zero-sum contrast
    distributes weight over more layers and the placebo profile has
    correspondingly more room to wander.
    """
    import dataclasses
    layers, _, y, meta = SIMULATORS[sim](n, 0.0, seed, n_ref_extra=n, d=D, L=L_)
    w = profile_weights(L_ + 1)
    key = placebo_key(meta, noise=0.5, seed=seed)
    S, R = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    o = np.argsort(key[R])
    Ra, Rb = R[o[:n]], R[o[n:]]
    idx = np.concatenate([S, Ra])
    yo = np.concatenate([np.ones(n, int), np.zeros(n, int)])
    ba, _ = layer_profile(layers, idx, yo, CFG, keep_smoothers=False)
    pool = np.concatenate([Ra, Rb])

    bases = []
    for r in range(reps):
        cfg_r = dataclasses.replace(CFG, rng_seed=CFG.rng_seed + 1000 * (r + 1))
        p_r = level_matched_placebo(layers, pool, key, float(ba[0]), cfg_r)
        if p_r.get("profile") is not None:
            bases.append(float(w @ np.asarray(p_r["profile"], float)))
    b = np.asarray(bases)
    return {"L": L_, "seed": seed, "n_draws": len(b),
            "sd": float(b.std(ddof=1)) if len(b) > 1 else float("nan"),
            "mean": float(b.mean()) if len(b) else float("nan"),
            "range": float(b.max() - b.min()) if len(b) > 1 else float("nan")}


def _depth_job(a):
    return placebo_spread_rep(*a)


def _job(a):
    return one_rep(*a)


def run(sim, R_, eps=0.0, slope=0.0, key_noise=0.5, matched=True, seed0=0,
        workers=4):
    jobs = [(sim, seed0 + r, eps, slope, key_noise, matched)
            for r in range(R_)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_job, jobs, chunksize=max(1, R_ // (workers * 3))))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def rate(reps, key):
    k = int(sum(r[key] < 0.05 for r in reps))
    return {"reject_05": k / len(reps), "ci": wilson(k, len(reps)),
            "R": len(reps)}


def _load():
    return json.load(open(RES)) if os.path.exists(RES) else {}


def _save(r):
    json.dump(r, open(RES, "w"), indent=2)


def _up(lst, rec, keys):
    return sorted([x for x in lst if any(x[k] != rec[k] for k in keys)] + [rec],
                  key=lambda x: tuple(str(x[k]) for k in keys))


def summarise(reps):
    return {"mean_T_raw": float(np.mean([r["T_raw"] for r in reps])),
            "mean_T_adj": float(np.mean([r["T_adj"] for r in reps])),
            "mean_base": float(np.mean([r["base"] for r in reps])),
            "mean_ba_nuis": float(np.mean([r["ba_nuis"] for r in reps]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--R", type=int, default=0)
    a = ap.parse_args()
    W_, st = a.workers, a.stage
    res = _load()
    t0 = time.time()

    if st == "all":
        stages = (["init", "props"]
                  + [f"cap{s}{p}" for s in "AB" for p in (96, 1200)]
                  + [f"slope{s}{int(v*100)}" for s in "AB" for v in (0, .25, .5)]
                  + [f"size{s}" for s in "AB"]
                  + [f"power{s}_{c}" for s in "AB" for c in range(3)]
                  + [f"prof{s}" for s in "AB"]
                  + [f"key{s}{int(v*10)}" for s in "B" for v in (0.5, 2.0, 5.0)]
                  + ["depth", "timing"])
        t = time.time()
        for i, sub in enumerate(stages, 1):
            print(f"[{i}/{len(stages)}] {sub}", flush=True)
            sys.argv = [sys.argv[0], "--stage", sub, "--workers", str(W_)]
            main()
        print(f"\nall stages in {(time.time()-t)/60:.1f} min")
        return

    if st == "init":
        res["config"] = {"d": D, "L": L, "n_per_set": N_PER,
                         "reference_multiplier": 2, "B": B, "alpha": 0.05,
                         "n_folds": CFG.n_folds, "fold_seeds": len(CFG.seeds),
                         "target_df": CFG.target_df,
                         "weights": [float(v) for v in W],
                         "primary": "level-matched recentred depth contrast, "
                                    "label-permutation null"}
        res["wallclock_s"] = 0.0

    elif st == "props":
        out = {}
        for name, fn in SIMULATORS.items():
            layers, _, _, _ = fn(400, 0.0, 0, d=D, L=L)
            H = np.stack(layers)
            rms = np.linalg.norm(H, axis=2).mean(axis=1)
            flat = np.concatenate([l.ravel() for l in layers])
            ev = np.linalg.svd(layers[-1] - layers[-1].mean(0),
                               compute_uv=False) ** 2
            out[name] = {"rms_growth": float(rms[-1] / rms[0]),
                         "corr_first_last": float(np.corrcoef(
                             layers[1].ravel(), layers[-1].ravel())[0, 1]),
                         "kurtosis": float(((flat - flat.mean()) ** 4).mean()
                                           / flat.var() ** 2),
                         "top_eig_share": float(ev[0] / ev.sum())}
        res["sim_properties"] = out
        for k, v in out.items():
            print("   Sim-" + k + " " + " ".join(f"{a}={b:.3f}"
                                                 for a, b in v.items()))

    elif st == "depth":
        # Does the placebo baseline become unstable as the model gets deeper?
        depths = (12, 24, 36, 49)
        n_sim = a.R or 4
        jobs = [("B", 7000 + s, Ld) for Ld in depths for s in range(n_sim)]
        with ProcessPoolExecutor(max_workers=W_) as ex:
            out = list(ex.map(_depth_job, jobs))
        rec = []
        for Ld in depths:
            g = [o for o in out if o["L"] == Ld]
            rec.append({"L": Ld, "n_sims": len(g),
                        "mean_sd": float(np.mean([o["sd"] for o in g])),
                        "max_sd": float(np.max([o["sd"] for o in g])),
                        "mean_range": float(np.mean([o["range"] for o in g]))})
            print(f"   L={Ld:>3}: sd(base) mean={rec[-1]['mean_sd']:.5f} "
                  f"max={rec[-1]['max_sd']:.5f} "
                  f"range={rec[-1]['mean_range']:.5f}")
        res["placebo_depth"] = rec
        ratio = rec[-1]["mean_sd"] / max(rec[0]["mean_sd"], 1e-12)
        print(f"   sd at L={depths[-1]} is {ratio:.1f}x that at L={depths[0]}")
        print("   real 49-layer model measured sd(base) = 0.00566 "
              "against T_adj = 0.01079")

    elif st.startswith("cap"):
        sim, p = st[3], int(st[4:])
        reps = run(sim, a.R or 60, seed0=1000 + p, workers=W_)
        rec = {"sim": sim, "p_nuis": p, "raw": rate(reps, "p_raw"),
               "adj": rate(reps, "p_adj"), **summarise(reps)}
        res["capacity"] = _up(res.get("capacity", []), rec, ["sim", "p_nuis"])
        print(f"   Sim-{sim} p={p}: raw={rec['raw']['reject_05']:.3f} "
              f"adj={rec['adj']['reject_05']:.3f}")

    elif st.startswith("slope"):
        sim, sl = st[5], int(st[6:]) / 100.0
        reps = run(sim, a.R or 60, slope=sl, seed0=2000 + int(sl * 100),
                   workers=W_)
        rec = {"sim": sim, "slope": sl, "raw": rate(reps, "p_raw"),
               "adj": rate(reps, "p_adj"), **summarise(reps)}
        res["depth_slope"] = _up(res.get("depth_slope", []), rec,
                                 ["sim", "slope"])
        print(f"   Sim-{sim} slope={sl:.2f}: raw={rec['raw']['reject_05']:.3f} "
              f"adj={rec['adj']['reject_05']:.3f}")

    elif st.startswith("size"):
        sim = st[4]
        out = {}
        for matched in (True, False):
            reps = run(sim, a.R or 50, matched=matched, seed0=8000,
                       workers=W_)
            out["matched" if matched else "half"] = {
                **rate(reps, "p_adj"), **summarise(reps)}
        res.setdefault("baseline_size", {})[sim] = out
        print(f"   Sim-{sim}: matched={out['matched']['reject_05']:.3f}  "
              f"half-size={out['half']['reject_05']:.3f}")

    elif st.startswith("power"):
        sim = st[5]
        chunk = int(st.split("_")[1])
        grid = {0: [0.0, 0.5], 1: [1.0, 1.5], 2: [2.0, 3.0]}
        for eps in grid[chunk]:
            reps = run(sim, a.R or 50, eps=eps, seed0=3000 + int(100 * eps),
                       workers=W_)
            rec = {"sim": sim, "eps": eps, **rate(reps, "p_adj"),
                   "raw_reject": rate(reps, "p_raw")["reject_05"],
                   **summarise(reps)}
            res["power"] = _up(res.get("power", []), rec, ["sim", "eps"])
            print(f"   Sim-{sim} eps={eps:.1f}: power={rec['reject_05']:.3f}")

    elif st.startswith("prof"):
        sim = st[4]
        sig = run(sim, a.R or 40, eps=1.5, seed0=4000, workers=W_)
        nul = run(sim, a.R or 40, eps=0.0, seed0=4500, workers=W_)
        res.setdefault("profiles", {})[sim] = {
            "planted": [float(v) for v in layer_bump(L)],
            "signal_ba": np.array([r["ba"] for r in sig]).mean(0).tolist(),
            "signal_sd": np.array([r["ba"] for r in sig]).std(0).tolist(),
            "signal_placebo": np.array([r["ba_plc"] for r in sig]).mean(0).tolist(),
            "null_ba": np.array([r["ba"] for r in nul]).mean(0).tolist(),
            "null_placebo": np.array([r["ba_plc"] for r in nul]).mean(0).tolist(),
        }
        nb = res["profiles"][sim]["null_ba"]
        print(f"   Sim-{sim} null profile {nb[0]:.3f} -> {nb[-1]:.3f}; "
              f"signal argmax {int(np.argmax(res['profiles'][sim]['signal_ba']))}")

    elif st.startswith("key"):
        sim, nz = st[3], int(st[4:]) / 10.0
        reps0 = run(sim, a.R or 40, eps=0.0, key_noise=nz, seed0=6000,
                    workers=W_)
        reps1 = run(sim, a.R or 40, eps=1.5, key_noise=nz, seed0=6500,
                    workers=W_)
        rec = {"sim": sim, "key_noise": nz,
               "fpr": rate(reps0, "p_adj"), "power": rate(reps1, "p_adj"),
               "mean_gap": float(np.mean([r["ba0_gap"] for r in reps0]))}
        res["key_sensitivity"] = _up(res.get("key_sensitivity", []), rec,
                                     ["sim", "key_noise"])
        print(f"   Sim-{sim} key noise={nz}: FPR={rec['fpr']['reject_05']:.3f} "
              f"power={rec['power']['reject_05']:.3f}")

    elif st == "timing":
        out = []
        for n_set, d in [(500, 512), (1000, 2048)]:
            rng = np.random.default_rng(0)
            y = np.zeros(2 * n_set, dtype=int)
            y[:n_set] = 1
            X = rng.standard_normal((2 * n_set, d))
            r = time_permutation_strategies(X, y,
                                            n_perm=(20 if d <= 512 else 4))
            r["per_perm_ms"] = 1e3 * r["shortcut_s"] / r["n_perm"]
            r["refit_per_perm_ms"] = 1e3 * r["refit_s"] / r["n_perm"]
            r["proj_B2000_s"] = 2000 * r["shortcut_s"] / r["n_perm"]
            r["proj_B2000_refit_h"] = 2000 * r["refit_s"] / r["n_perm"] / 3600
            out.append(r)
        res["timing"] = out
        print(f"   setup={out[-1]['setup_s']:.2f}s  "
              f"{out[-1]['per_perm_ms']:.3f} ms/perm  vs refit "
              f"{out[-1]['proj_B2000_refit_h']:.1f} h for B=2000")

    res["wallclock_s"] = res.get("wallclock_s", 0.0) + (time.time() - t0)
    _save(res)
    print(f"[{st}] {time.time()-t0:.1f}s  cumulative "
          f"{res['wallclock_s']/60:.1f} min")


if __name__ == "__main__":
    main()
