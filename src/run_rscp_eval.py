"""
run_rscp_eval.py
End-to-end RSCP audit of one model against one pair of item sets.

This is the runner for Phases 1, 3 and 4 of the validation programme. Given a
model and two JSONL item files, it extracts prefix activations, builds the
declared nuisance family, runs the placebo gate, computes the depth contrast,
and writes a report.

    # 1. plumbing check, no torch and no model needed
    python3 run_rscp_eval.py --dry-run --out runs/dryrun

    # 2. a real audit
    python3 run_rscp_eval.py \\
        --model EleutherAI/pythia-410m \\
        --suspect data/gsm8k.jsonl \\
        --reference data/gsm1k.jsonl \\
        --out runs/pythia410m_gsm

Item file format, one JSON object per line:
    {"prefix": "everything the model sees before it must answer",
     "answer": "the gold continuation",
     "correct": true}
`answer` is optional and is never fed to the probe; it is used only for the
contamination-adjusted score. `correct` is optional; supply it if you already
have per-item evaluation results and want the inflation estimate.

THE PREFIX RULE IS ENFORCED HERE. Only the `prefix` field reaches the model
during extraction. If you put the answer in the prefix you defeat the method,
and no amount of statistics downstream will tell you that you did.

Status: the statistical core is validated (see phase0_validation.py). The
activation-extraction path is written against the transformers API but has
not been run end to end in this repository, because the environment it was
written in has no GPU and no model access. Run --dry-run first, then a small
model, before trusting a large one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))
from paths import VAL_RESULTS, GENERATED, FIGURES, RUNS, DATA, CACHE, EXPERIMENTS, s


import rscp
from rscp import (
    RSCPConfig,
    _ba_from_c,
    _make_folds,
    char_ngram_features,
    correctness_vector,
    cross_fit_smoother,
    lam_for_df,
    finalise_contrast,
    layer_profile,
    level_matched_placebo,
    profile_weights,
    recentred_contrast_test,
    streaming_profile_and_null,
    surface_features,
)


# --------------------------------------------------------------------------
# Item loading
# --------------------------------------------------------------------------

def load_items(path: str) -> list[dict]:
    items = []
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{ln}: not valid JSON ({e})")
            if "prefix" not in o:
                raise SystemExit(f"{path}:{ln}: missing required key 'prefix'")
            items.append(o)
    if not items:
        raise SystemExit(f"{path}: no items")
    return items


def balance(a: list[dict], b: list[dict], seed: int = 0):
    """Trim to a common size so the design stays balanced."""
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b))
    ia = rng.choice(len(a), n, replace=False)
    ib = rng.choice(len(b), n, replace=False)
    return [a[i] for i in ia], [b[i] for i in ib], n


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------

def build_activations(model, prefixes, pooling, device, batch_size, dry_run,
                      n_layers=13, d=96, seed=0, n_items=None,
                      max_length=512, cache=True):
    if dry_run:
        # Structured noise so the plumbing can be exercised without a model:
        # a nuisance direction present at every layer, no memorisation signal.
        rng = np.random.default_rng(seed)
        N = len(prefixes)
        v = rng.standard_normal(d)
        v /= np.linalg.norm(v)
        cov = rng.standard_normal((N, 1))
        return [rng.standard_normal((N, d)) + cov * (0.5 * v)[None, :]
                for _ in range(n_layers)]
    print(f"  extracting activations from {model} "
          f"({len(prefixes)} prefixes, pooling={pooling}) ...", flush=True)
    return rscp.extract_prefix_activations(
        model, prefixes, pooling=pooling, device=device,
        batch_size=batch_size, max_length=max_length,
        cache_dir=s(CACHE / "activations") if cache else None)


def build_nuisance(prefixes, layer0, ref_model, device, batch_size, dry_run,
                   use_ngrams=True, seed=0, max_length=512, cache_dir=None):
    blocks = [np.asarray(layer0, dtype=np.float64), surface_features(prefixes)]
    if use_ngrams:
        blocks.append(char_ngram_features(prefixes, max_features=4000))
    if ref_model and not dry_run:
        print(f"  reference-model likelihood features from {ref_model} ...",
              flush=True)
        blocks.append(rscp.reference_likelihood_features(
            ref_model, prefixes, device=device, batch_size=batch_size,
            max_length=max_length, cache_dir=cache_dir))
    return np.concatenate(blocks, axis=1)


def probe_set(X, y, cfg):
    lam = lam_for_df(X, cfg.target_df)
    sm = [cross_fit_smoother(
              X, _make_folds(len(y), cfg.n_folds, np.random.default_rng(s)), lam)
          for s in cfg.seeds]
    return correctness_vector(sm, y)


# --------------------------------------------------------------------------
# The audit
# --------------------------------------------------------------------------

def audit(args):
    cfg = RSCPConfig(seeds=tuple(range(args.fold_seeds)),
                     n_bootstrap=args.bootstrap, target_df=args.target_df)
    t0 = time.time()

    if args.dry_run:
        # Both sets are drawn from the same word pool with the same template
        # and a spread of lengths. If the dry run showed BA_nuisance near 1 it
        # would only be telling us the fake data was trivially separable.
        rng = np.random.default_rng(args.seed)
        pool = ("model data corpus token layer probe signal value item set "
                "score depth margin sample weight").split()
        def fake():
            k = int(rng.integers(18, 46))
            return "Q: " + " ".join(rng.choice(pool, k)) + " ? A:"
        n = args.dry_n
        S = [{"prefix": fake()} for _ in range(n)]
        R = [{"prefix": fake()} for _ in range(2 * n)]   # protocol wants 2x
    else:
        S = load_items(args.suspect)
        R = load_items(args.reference)
        print(f"  {len(S)} suspect, {len(R)} reference "
              f"({len(R)/max(len(S),1):.1f}x)")

    items = S + R
    prefixes = [it["prefix"] for it in items]
    y = np.array([1] * len(S) + [0] * len(R), dtype=int)

    layers = build_activations(args.model, prefixes, args.pooling, args.device,
                               args.batch_size, args.dry_run, seed=args.seed,
                               n_items=len(items), max_length=args.max_length,
                               cache=not args.no_cache)
    nuis = build_nuisance(prefixes, layers[0], args.reference_model,
                          args.device, args.batch_size, args.dry_run,
                          use_ngrams=not args.no_ngrams, seed=args.seed,
                          max_length=args.max_length,
                          cache_dir=None if args.no_cache
                          else s(CACHE / "activations"))
    print(f"  {len(layers)} layers x {layers[0].shape[1]} dims; "
          f"nuisance block {nuis.shape[1]} dims")

    # --- observed depth profile ---------------------------------------
    # One pass. Retaining a dense (N, N) smoother per layer per seed costs
    # 17.6 GB at 49 layers and N = 3000, which is what killed the first Phase 3
    # attempt. The null consumes them a layer at a time regardless, so they are
    # accumulated and released rather than held.
    idx = np.arange(len(y))
    ba, tnull = streaming_profile_and_null(
        layers, idx, y, cfg, B=cfg.n_bootstrap,
        rng=np.random.default_rng(args.seed))
    ba_n, _ = layer_profile([nuis], idx, y, cfg, keep_smoothers=False)
    ba_nuis = float(ba_n[0])

    # --- level-matched placebo baseline --------------------------------
    # Split the reference set on a surface variable and match its
    # embedding-layer separability to the observed split. Every item on both
    # sides is a non-member, so the resulting depth profile is the profile of
    # surface separability in this model, which is what the observed profile
    # must be compared against.
    ref_idx = np.flatnonzero(y == 0)
    key = np.array([len(p.split()) for p in prefixes], dtype=float)
    plc = level_matched_placebo(layers, ref_idx, key, float(ba[0]), cfg)
    prof = plc.get("profile")

    # How stable is the baseline itself? The null permutes labels in the
    # observed comparison and treats `base` as known, but `base` comes from a
    # random split of the reference set at a searched noise level. If its
    # spread is comparable to T_adj, the p-value understates the uncertainty.
    placebo_spread = None
    if args.placebo_reps > 1:
        import dataclasses
        import time as _time
        w_ = profile_weights(len(layers))
        bases = []
        print(f"  placebo stability: {args.placebo_reps} re-estimates, each "
              f"rebuilding the full {len(layers)}-layer profile")
        t0 = _time.time()
        for r in range(args.placebo_reps):
            cfg_r = dataclasses.replace(cfg, rng_seed=cfg.rng_seed + 1000 * (r + 1))
            p_r = level_matched_placebo(layers, ref_idx, key, float(ba[0]), cfg_r)
            if p_r.get("profile") is not None:
                bases.append(float(w_ @ np.asarray(p_r["profile"], float)))
                el = _time.time() - t0
                print(f"    rep {r + 1}/{args.placebo_reps}: "
                      f"base={bases[-1]:+.5f}  ({el:.0f}s elapsed, "
                      f"~{el / (r + 1) * (args.placebo_reps - r - 1):.0f}s left)",
                      flush=True)
        if len(bases) > 1:
            b = np.asarray(bases)
            placebo_spread = {"reps": len(bases), "mean": float(b.mean()),
                              "sd": float(b.std(ddof=1)),
                              "min": float(b.min()), "max": float(b.max())}
            print(f"  placebo baseline over {len(bases)} split seeds: "
                  f"mean {b.mean():+.5f}, sd {b.std(ddof=1):.5f}, "
                  f"range [{b.min():+.5f}, {b.max():+.5f}]")

    # The placebo enters the observed statistic, never the null, so one null
    # serves both. The previous code recomputed the whole permutation batch to
    # produce the uncorrected comparison.
    base_sd = None if placebo_spread is None else placebo_spread["sd"]
    ct = finalise_contrast(ba, tnull, prof, base_sd=base_sd,
                           rng=np.random.default_rng(args.seed + 5))
    raw = finalise_contrast(ba, tnull, None)

    # The decisive ratio. The null treats the baseline as known, so when the
    # placebo's own spread approaches the null's the p-value is not usable.
    if placebo_spread is not None and ct.get("null_sd"):
        ratio = placebo_spread["sd"] / ct["null_sd"]
        placebo_spread["null_sd"] = ct["null_sd"]
        placebo_spread["sd_ratio"] = float(ratio)
        note = ("negligible against the null" if ratio < 0.25 else
                "a substantial share of the null" if ratio < 0.75 else
                "LARGER THAN THE NULL ITSELF")
        print(f"  sd(baseline)={placebo_spread['sd']:.5f} vs "
              f"sd(null)={ct['null_sd']:.5f}  ratio={ratio:.2f}, {note}")
        if ct.get("p_value_baseline_inflated") is not None:
            print(f"  p with the baseline's own variance propagated: "
                  f"{ct['p_value_baseline_inflated']:.4f}  "
                  f"(vs {ct['p_value']:.4f} holding it fixed)")

    recentred = prof is not None

    # Requirement E, enforced rather than documented. BA_nuisance is its
    # operationalisation: it is what a classifier with no access to the model
    # achieves on this split. Far above 0.5 means the two sets differ so much
    # as text that the baseline correction is being asked to absorb a
    # first-order difference, and second-order mismatch is then easily larger
    # than any memorisation signal. WikiMIA-style temporal splits sit near
    # 0.86. The thresholds below are a judgement call, not a measured
    # quantity, and are deliberately conservative.
    exch = ("ok" if ba_nuis < args.nuis_warn
            else "marginal" if ba_nuis < args.nuis_block else "failed")
    out = {
        "model": args.model, "dry_run": bool(args.dry_run),
        "protocol": "level-matched recentred depth contrast, "
                    "label-permutation null",
        "n_suspect": int(len(S)), "n_reference": int(len(R)),
        "reference_multiplier": float(len(R) / max(len(S), 1)),
        "n_layers": len(layers), "d": int(layers[0].shape[1]),
        "nuisance_dims": int(nuis.shape[1]),
        "pooling": args.pooling, "alpha": args.alpha, "config": cfg.to_dict(),
        "ba_by_layer": [float(v) for v in ba], "ba_nuisance": ba_nuis,
        "exchangeability": exch,
        "baseline_profile": None if prof is None else [float(v) for v in prof],
        "placebo_spread": placebo_spread,
        "placebo": {"status": plc.get("status"),
                    "ba0": plc.get("ba0"), "target_ba0": plc.get("target_ba0"),
                    "match_gap": plc.get("match_gap"),
                    "n_per_side": plc.get("n_per_side")},
        "contrast": {"T_adj": ct["T_adj"], "T_raw": ct["T_raw"],
                     "baseline": ct["baseline"], "p_value": ct["p_value"],
                     "recentred": recentred},
        "contrast_uncorrected": {"T": raw["T_raw"], "p_value": raw["p_value"]},
        "verdict": None,
        "warnings": [],
        "wallclock_s": time.time() - t0,
    }

    if exch == "marginal":
        out["warnings"].append(
            f"BA_nuisance = {ba_nuis:.3f}: a blind classifier separates these "
            "sets well above chance, so Requirement E holds only "
            "approximately. Treat the contrast as suggestive.")
    if len(R) < 1.8 * len(S):
        out["warnings"].append(
            f"reference set is only {len(R)/max(len(S),1):.1f}x the suspect "
            "set; the protocol wants 2x so the baseline is measured at the "
            "same sample size. A half-size baseline tripled the false "
            "positive rate in Phase 0c.")

    if exch == "failed" and not args.acknowledge_non_exchangeable:
        out["verdict_blocked"] = True
        out["verdict"] = (
            f"NO VERDICT. BA_nuisance = {ba_nuis:.3f} exceeds "
            f"{args.nuis_block:.2f}: a classifier with no access to the model "
            "separates these two sets almost perfectly, so they are not "
            "exchangeable and Requirement E fails. This is the defining "
            "property of a temporal split, and it is why published "
            "membership-inference numbers on such splits are not "
            "interpretable. The contrast is reported above as a measurement, "
            "not as evidence of contamination. Pass "
            "--acknowledge-non-exchangeable to override, and say so in any "
            "writeup.")
    elif not recentred:
        out["verdict"] = (f"NO BASELINE ({plc.get('status')}). Without a "
                          "level-matched placebo the contrast assumes surface "
                          "decodability is flat in depth, which Phase 0c shows "
                          "fails in both directions. No claim admissible. "
                          "Supply a surface key with spread, or more reference "
                          "items.")
    elif ct["p_value"] < args.alpha:
        # Second gate, alongside Requirement E. The p-value above holds the
        # placebo baseline fixed, but it is estimated, and when its own spread
        # rivals the null's the significance can be an artefact of which split
        # the search happened to draw. Measured on a real audit: sd 0.0057
        # against a null sd of 0.0043, turning p = 0.0075 into about 0.065.
        p_i = ct.get("p_value_baseline_inflated")
        ratio = (placebo_spread or {}).get("sd_ratio")
        pre = ("[REQUIREMENT E OVERRIDDEN] " if exch == "failed" else
               "[EXCHANGEABILITY MARGINAL] " if exch == "marginal" else "")
        if placebo_spread is None:
            out["verdict"] = (
                pre + f"PROVISIONAL. Contrast significant (p="
                f"{ct['p_value']:.4f}) holding the placebo baseline fixed, but "
                "the baseline is an estimate and its variance has not been "
                "measured. Re-run with --placebo-reps 8 before reporting "
                "this: on a real audit that correction turned p = 0.0075 into "
                "0.065.")
        elif p_i is not None and p_i >= args.alpha:
            out["verdict_blocked"] = True
            out["verdict"] = (
                pre + f"NO VERDICT. The contrast is significant (p="
                f"{ct['p_value']:.4f}) only while the placebo baseline is "
                f"treated as known. Its measured standard deviation is "
                f"{placebo_spread['sd']:.5f}, {ratio:.2f} times the null's "
                f"own, and propagating it gives p = {p_i:.4f}. The result is "
                "inside the noise of the correction used to produce it.")
        else:
            out["verdict"] = (
                pre + f"Contrast significant (p={ct['p_value']:.4f}, "
                f"p={p_i:.4f} with the baseline's variance propagated) after "
                "recentring on the model's own null depth profile. Evidence "
                "of depth-dependent familiarity beyond surface separability. "
                "Report an exposure interval only if a same-family "
                "calibration exists.")
    else:
        out["verdict"] = (f"Contrast null (p={ct['p_value']:.4f}). No evidence "
                          "at this sensitivity. Report the detection floor so "
                          "the null can be read at the right strength.")

    # --- contamination-adjusted score, if labels were supplied ------------
    corr = [it.get("correct") for it in S]
    if all(c is not None for c in corr) and len(corr) == len(S):
        peak = int(np.argmax(ba[1:])) + 1
        c_peak = correctness_vector(sms[peak], y)
        margins = c_peak[:len(S)]
        order = np.argsort(-margins)
        q = max(1, len(order) // 4)
        hi = [bool(corr[i]) for i in order[:q]]
        lo = [bool(corr[i]) for i in order[-q:]]
        out["inflation_raw"] = float(np.mean(hi) - np.mean(lo))
        out["inflation_note"] = ("Raw, not difficulty-matched. See Section 4.9 "
                                 "of the paper before quoting this.")
    return out


def report(o, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(o, open(path + ".json", "w"), indent=2)
    L = ["RSCP audit", "=" * 66,
         f"model            {o['model']}",
         f"protocol         {o['protocol']}",
         f"items            {o['n_suspect']} suspect / {o['n_reference']} "
         f"reference  ({o['reference_multiplier']:.1f}x)",
         f"layers x dims    {o['n_layers']} x {o['d']}",
         f"nuisance dims    {o['nuisance_dims']}", ""]
    ex = o.get("exchangeability", "ok")
    tagline = {"ok": "sets are well matched",
               "marginal": "MARGINAL: blind separation well above chance",
               "failed": "REQUIREMENT E FAILS: sets are separable blind"}[ex]
    L.append(f"BA_nuisance      {o['ba_nuisance']:.4f}   <- read first; {tagline}")
    L.append("observed profile " + " ".join(f"{b:.3f}"
                                            for b in o["ba_by_layer"]))
    if o["baseline_profile"]:
        bp = o["baseline_profile"]
        L.append("baseline profile " + " ".join(f"{b:.3f}" for b in bp))
        import numpy as _np
        arr = _np.asarray(bp)
        pk = int(arr.argmax())
        shape = ("rising" if pk == len(arr) - 1 else
                 "falling" if pk == 0 else f"hump, peak at layer {pk}")
        L.append(f"                 span {100*(arr.max()-arr.min()):.1f} "
                 f"accuracy points, {shape}")
    p = o["placebo"]
    if p["ba0"] is None:
        L.append(f"BASELINE         unavailable ({p['status']})")
    else:
        L.append(f"BASELINE         matched at layer 0: {p['ba0']:.3f} vs "
                 f"observed {p['target_ba0']:.3f}  "
                 f"(gap {p['match_gap']:.3f}, n/side {p['n_per_side']})")
    L.append("")
    c = o["contrast"]
    L.append(f"depth contrast   T_raw={c['T_raw']:+.5f}  "
             f"baseline={c['baseline']:+.5f}  T_adj={c['T_adj']:+.5f}")
    L.append(f"                 p={c['p_value']:.4f}  "
             f"(recentred={c['recentred']})")
    u = o["contrast_uncorrected"]
    L.append(f"uncorrected      T={u['T']:+.5f}  p={u['p_value']:.4f}  "
             "(shown only to expose the baseline's effect)")
    if "inflation_raw" in o:
        L.append(f"raw inflation    {o['inflation_raw']:+.4f}  "
                 "(not difficulty-matched)")
    for w in o["warnings"]:
        L += ["", "WARNING", "  " + w]
    L += ["", "VERDICT", "  " + o["verdict"], "",
          f"ran in {o['wallclock_s']:.1f}s"]
    if o["dry_run"]:
        L += ["", "DRY RUN: activations are synthetic noise. Pipeline check "
              "only."]
    txt = "\n".join(L)
    open(path + ".txt", "w").write(txt + "\n")
    print()
    print(txt)
    print()
    print(f"wrote {path}.json and {path}.txt")


def main():
    ap = argparse.ArgumentParser(
        description="Run an RSCP contamination audit.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="(dry run)",
                    help="HF model id, e.g. EleutherAI/pythia-410m")
    ap.add_argument("--suspect", help="JSONL of items under audit")
    ap.add_argument("--reference", help="JSONL of exchangeable reference items")
    ap.add_argument("--reference-model", default=None,
                    help="independent model for likelihood nuisance features")
    ap.add_argument("--out", default=s(RUNS / "audit"))
    ap.add_argument("--pooling", choices=["last", "mean"], default="last")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--fold-seeds", type=int, default=5)
    ap.add_argument("--target-df", type=float, default=32.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ngrams", action="store_true",
                    help="skip the character n-gram nuisance block")
    ap.add_argument("--dry-run", action="store_true",
                    help="synthetic activations; no torch or model required")
    ap.add_argument("--dry-n", type=int, default=300)
    ap.add_argument("--nuis-warn", type=float, default=0.60,
                    help="BA_nuisance above this earns a warning")
    ap.add_argument("--nuis-block", type=float, default=0.75,
                    help="BA_nuisance above this blocks a contamination "
                         "verdict; Requirement E has failed")
    ap.add_argument("--acknowledge-non-exchangeable", action="store_true",
                    help="report a verdict anyway on a non-exchangeable split")
    ap.add_argument("--max-length", type=int, default=512,
                    help="truncate prefixes to this many tokens")
    ap.add_argument("--no-cache", action="store_true",
                    help="do not reuse cached activations")
    ap.add_argument("--placebo-reps", type=int, default=1,
                    help="re-estimate the placebo baseline this many times "
                         "with different split seeds and report the spread. "
                         "The permutation null treats the baseline as a known "
                         "constant, so if this spread is comparable to T_adj "
                         "the reported p-value is understated.")
    ap.add_argument("--placebo-min", type=int, default=50,
                    help="minimum reference items per placebo side")
    a = ap.parse_args()

    if not a.dry_run and not (a.suspect and a.reference):
        ap.error("--suspect and --reference are required unless --dry-run")
    if not a.dry_run and a.model == "(dry run)":
        ap.error("--model is required unless --dry-run")

    print("RSCP audit")
    if a.dry_run:
        print("  DRY RUN: synthetic activations, pipeline check only")
    report(audit(a), a.out)


if __name__ == "__main__":
    sys.exit(main())
