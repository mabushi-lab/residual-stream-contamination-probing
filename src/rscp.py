"""
rscp.py
Residual-Stream Contamination Probing (RSCP).

Reference implementation of the protocol in
"Excess Separability: Nuisance-Controlled Residual-Stream Probing for
Benchmark Contamination Detection".

The core object is the cross-fitted ridge smoother Pi. Because ridge fitted
values are linear in the labels, Pi does not depend on y, so a permutation
test over the label vector costs one matrix-vector product per permutation
after a single O(K(n d^2 + d^3)) setup. That is what makes the full battery
of controls affordable.

Dependencies
------------
Required : numpy
Optional : scikit-learn (character n-gram TF-IDF nuisance features)
           torch + transformers (activation extraction from a real model)

The library imports and runs without torch. Activation extraction raises a
clear error if torch is absent, so the statistical core can be validated on
synthetic data with no GPU.

Author's note on scope: the statistical core is exercised by
phase0_validation.py and the numbers in Section 6 of the paper come from
that run. Nothing here has been run against a real language model.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "RSCPConfig",
    "LayerResult",
    "RSCPResult",
    "cross_fit_smoother",
    "balanced_accuracy",
    "ba_from_smoother",
    "run_rscp",
    "permutation_null",
    "paired_bootstrap_test",
    "correctness_vector",
    "nuisance_strata",
    "stratified_permutations",
    "control_task",
    "random_direction_baseline",
    "internal_null",
    "length_matched_indices",
    "surface_features",
    "char_ngram_features",
    "bag_of_embeddings",
    "profile_weights",
    "profile_contrast_test",
    "recentred_contrast_test",
    "level_matched_placebo",
    "layer_profile",
    "_ba_perm_batch",
    "placebo_contrast",
    "lam_for_df",
    "fit_exposure_link",
    "invert_exposure",
    "detection_floor",
    "extract_prefix_activations",
    "reference_likelihood_features",
]


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class RSCPConfig:
    """Protocol settings. Defaults match the paper."""

    n_folds: int = 5
    seeds: Sequence[int] = (0, 1, 2, 3, 4)
    n_permutations: int = 1000
    ridge_lambdas: Sequence[float] = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)
    standardise: bool = True
    # Layers included in the max statistic. Layer 0 is excluded by default:
    # it is a bag of token embeddings and therefore belongs to the nuisance
    # family, not to the set of layers that can carry evidence.
    skip_layer_zero: bool = True
    rng_seed: int = 0
    # Permute labels within strata of the cross-fitted nuisance score. This
    # tests exchangeability CONDITIONAL on the nuisance family, which is the
    # hypothesis of interest. Set False to recover the unrestricted
    # permutation, retained only as a comparator.
    conditional_permutation: bool = False
    n_strata: int = 10
    # Primary inference. The paired bootstrap over items is the default;
    # the label permutation is retained as a (valid but conservative)
    # comparator and is off by default because it costs more.
    n_bootstrap: int = 2000
    studentise: bool = True
    # Effective degrees of freedom matched across feature blocks.
    target_df: float = 32.0
    also_run_permutation: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seeds"] = list(self.seeds)
        d["ridge_lambdas"] = list(self.ridge_lambdas)
        return d


@dataclass
class LayerResult:
    layer: int
    ba_mean: float
    ba_sd: float


@dataclass
class RSCPResult:
    ba_nuisance: float
    ba_nuisance_sd: float
    layers: list[LayerResult] = field(default_factory=list)
    delta: list[float] = field(default_factory=list)
    T: float = float("nan")
    layer_star: int = -1
    p_value: float = float("nan")
    null_quantiles: dict = field(default_factory=dict)
    n_per_set: int = 0
    contrast: float = float("nan")
    contrast_p: float = float("nan")
    contrast_se: float = float("nan")
    notes: str = ""

    def summary(self) -> str:
        return (
            f"BA_nuis = {self.ba_nuisance:.4f} (sd {self.ba_nuisance_sd:.4f})\n"
            f"T_c     = {self.contrast:+.5f}  p = {self.contrast_p:.4f}"
            f"  (depth-profile contrast, primary)\n"
            f"level   = {self.T:+.4f} at layer {self.layer_star}"
            f"  p = {self.p_value:.4f}  (descriptive only)\n"
            f"n/set   = {self.n_per_set}\n{self.notes}"
        )


# --------------------------------------------------------------------------
# Core: cross-fitted ridge smoother
# --------------------------------------------------------------------------

def _make_folds(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.permutation(n)
    fold = np.empty(n, dtype=np.int64)
    for f, chunk in enumerate(np.array_split(idx, k)):
        fold[chunk] = f
    return fold


def cross_fit_smoother(
    X: np.ndarray,
    fold: np.ndarray,
    lam: float,
    standardise: bool = True,
) -> np.ndarray:
    """Cross-fitted ridge smoother Pi with Pi[i, j] = 0 when fold[i] == fold[j].

    Out-of-fold predictions for any label vector y are simply Pi @ y, and Pi
    is independent of y. This is Eq. (5) of the paper.

    Parameters
    ----------
    X   : (N, d) design matrix.
    fold: (N,) integer fold assignment.
    lam : ridge penalty.

    Returns
    -------
    Pi : (N, N) smoother.
    """
    X = np.asarray(X, dtype=np.float64)
    N, d = X.shape
    Pi = np.zeros((N, N), dtype=np.float64)

    for f in np.unique(fold):
        te = np.flatnonzero(fold == f)
        tr = np.flatnonzero(fold != f)
        Xtr, Xte = X[tr], X[te]

        if standardise:
            mu = Xtr.mean(axis=0)
            sd = Xtr.std(axis=0)
            sd[sd < 1e-12] = 1.0
            Xtr = (Xtr - mu) / sd
            Xte = (Xte - mu) / sd
        # Intercept handled by centring the labels at prediction time; adding
        # an explicit unpenalised intercept column would break the linearity
        # of Pi in y, so we rely on the median threshold instead.

        if d <= Xtr.shape[0]:
            A = Xtr.T @ Xtr + lam * np.eye(d)
            W = np.linalg.solve(A, Xtr.T)          # (d, n_tr)
        else:
            # Dual form, cheaper when d > n_tr.
            G = Xtr @ Xtr.T + lam * np.eye(Xtr.shape[0])
            W = Xtr.T @ np.linalg.inv(G)           # (d, n_tr)
        Pi[np.ix_(te, tr)] = Xte @ W

    return Pi


def balanced_accuracy(scores: np.ndarray, y: np.ndarray) -> float:
    """Balanced accuracy at the median score threshold (Eq. 6)."""
    tau = np.median(scores)
    pred = scores > tau
    pos = y == 1
    neg = ~pos
    n1, n0 = pos.sum(), neg.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return 0.5 * (pred[pos].sum() / n1 + (~pred[neg]).sum() / n0)


def ba_from_smoother(Pi: np.ndarray, y: np.ndarray) -> float:
    """Balanced accuracy of the cross-fitted probe for label vector y."""
    ytilde = 2.0 * y - 1.0
    return balanced_accuracy(Pi @ ytilde, y)


def lam_for_df(X: np.ndarray, target_df: float) -> float:
    """Ridge penalty giving a target effective degrees of freedom.

    df(lambda) = sum_j s_j^2 / (s_j^2 + lambda) for singular values s_j of the
    standardised design. Matching df across feature blocks is not cosmetic.
    Phase 0 shows that at a shared penalty the block with fewer dimensions
    estimates its discriminative direction more efficiently, so BA is biased
    by block dimensionality and any statistic built on the LEVEL of
    BA_l - BA_nuis inherits that bias. With a 20,000-feature character n-gram
    block against 2,048-dimensional activations the bias runs towards false
    positives.
    """
    X = np.asarray(X, dtype=np.float64)
    Xs = (X - X.mean(0)) / np.maximum(X.std(0), 1e-12)
    s2 = np.linalg.svd(Xs, compute_uv=False) ** 2
    lo, hi = 1e-8, 1e12
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if (s2 / (s2 + mid)).sum() > target_df:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def profile_weights(n_layers: int, kind: str = "ramp") -> np.ndarray:
    """Pre-registered depth weights summing to zero.

    The primary statistic is a contrast on the DEPTH PROFILE of separability,
    not on its level:  T_c = sum_l w_l BA_l  with  sum_l w_l = 0.

    Two things follow from sum w = 0. The nuisance term cancels exactly, since
    sum_l w_l (BA_l - BA_nuis) = sum_l w_l BA_l, so the statistic is immune to
    the capacity bias above. And the statistic tests the hypothesis that
    actually distinguishes memorisation from surface difference: nuisance
    information is present already in a bag of token embeddings and is roughly
    flat in depth, whereas a familiarity signal has to be computed and should
    therefore rise with depth.

    "ramp" is the default because it assumes only monotone rise and does not
    require guessing where the profile peaks.
    """
    l = np.arange(n_layers, dtype=float)
    a = l / max(l.max(), 1.0) if kind == "ramp" else np.asarray(kind, float)
    w = a - a.mean()
    return w / np.abs(w).sum()


def _ba_perm_batch(smoothers: Sequence[np.ndarray], Y: np.ndarray) -> np.ndarray:
    """Balanced accuracy of each fold-seed probe under a batch of label vectors.

    Y is (N, B) in {0, 1}. Because Pi is label-independent, ``Pi @ y`` is the
    out-of-fold prediction of a probe *refitted* to that label vector, so
    permuting labels here re-estimates the discriminative direction. That is
    the variance component an item bootstrap cannot see, and it is why the
    permutation null is calibrated where the bootstrap is not.
    """
    out = np.zeros(Y.shape[1])
    for Pi in smoothers:
        U = Pi @ (2.0 * Y - 1.0)
        pred = U > np.median(U, axis=0, keepdims=True)
        n1 = Y.sum(axis=0)
        n0 = Y.shape[0] - n1
        out += 0.5 * ((pred & (Y == 1)).sum(0) / np.maximum(n1, 1)
                      + ((~pred) & (Y == 0)).sum(0) / np.maximum(n0, 1))
    return out / len(smoothers)


def layer_profile(layer_activations, idx, y, cfg, keep_smoothers: bool = True):
    """Per-layer balanced accuracy and the smoothers that produced it.

    Each smoother is dense and (N, N), and there is one per layer per fold
    seed, so retaining them costs ``layers x seeds x N^2 x 8`` bytes: 17.6 GB
    at 49 layers, 5 seeds and N = 3000. Callers that only want the profile
    should pass ``keep_smoothers=False``, which returns ``(ba, None)`` and
    holds one layer at a time. ``streaming_profile_and_null`` avoids the cost
    even when the null is needed.
    """
    idx = np.asarray(idx)
    y = np.asarray(y).astype(int)
    N = idx.size
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    ba, sms = [], ([] if keep_smoothers else None)
    for H in layer_activations:
        X = np.asarray(H)[idx]
        lam = lam_for_df(X, cfg.target_df)
        sm = [cross_fit_smoother(
                  X, _make_folds(N, cfg.n_folds, np.random.default_rng(s)), lam)
              for s in cfg.seeds]
        if keep_smoothers:
            sms.append(sm)
        ba.append(_ba_from_c(correctness_vector(sm, y), pos, neg))
    return np.asarray(ba), sms


def level_matched_placebo(
    layer_activations,
    reference_index: np.ndarray,
    surface_key: np.ndarray,
    target_ba0: float,
    cfg: "RSCPConfig | None" = None,
    noise_grid: Sequence[float] = (0.0, 0.5, 1.0, 1.8, 3.0, 5.0, 8.0),
) -> dict:
    """Baseline depth profile from a within-reference split, matched at layer 0.

    Every item on both sides is a non-member, so whatever depth profile this
    produces is the profile of *surface* separability in this model. That is
    the baseline the observed profile has to be compared against.

    Two matching conditions have to hold or the baseline is not comparable.

    Sample size. The split must have the same number of items per side as the
    observed contrast, which is why the protocol requires |R| = 2|S|: half the
    reference set is the comparison arm, half is the baseline arm. With a
    half-size baseline the false positive rate measured in Phase 0b was 0.21
    rather than 0.05.

    Separability magnitude. Splitting the reference set at the extremes of a
    surface key separates the halves more strongly than the suspect and
    reference sets actually differ, and a stronger separation bends the depth
    profile differently. We therefore search over how coarsely to split, and
    take the split whose embedding-layer separability is closest to the
    observed one. Layer 0 is the right anchor because it is a bag of token
    embeddings and so carries surface information only.
    """
    cfg = cfg or RSCPConfig()
    idx = np.asarray(reference_index)
    k0 = np.asarray(surface_key, dtype=float)[idx]
    n = idx.size // 2
    if n < 2:
        return {"status": "not computable: reference set too small",
                "profile": None, "n_per_side": int(n)}
    if np.unique(k0).size < 2:
        return {"status": "not computable: surface key is constant",
                "profile": None, "n_per_side": int(n)}

    rs = np.random.default_rng(cfg.rng_seed + 77)
    y = np.concatenate([np.ones(n, int), np.zeros(n, int)])
    best = None
    for nz in noise_grid:
        kk = k0 + nz * k0.std() * rs.standard_normal(k0.size)
        o = np.argsort(kk)
        cand = idx[np.concatenate([o[len(o) - n:], o[:n]])]
        ba0, _ = layer_profile(layer_activations[:1], cand, y, cfg,
                               keep_smoothers=False)
        gap = abs(float(ba0[0]) - target_ba0)
        if best is None or gap < best[0]:
            best = (gap, cand, float(ba0[0]), nz)

    gap, cand, ba0, nz = best
    # The placebo needs the profile, never the smoothers.
    ba, _ = layer_profile(layer_activations, cand, y, cfg,
                          keep_smoothers=False)
    return {"status": "ok", "profile": ba, "ba0": ba0, "target_ba0": target_ba0,
            "match_gap": gap, "chosen_noise": nz, "n_per_side": int(n)}


def recentred_contrast_test(
    ba_observed: np.ndarray,
    smoothers: Sequence[Sequence[np.ndarray]],
    y: np.ndarray,
    placebo_profile: np.ndarray | None,
    weights: np.ndarray | None = None,
    B: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict:
    """PRIMARY test: depth contrast recentred on a level-matched baseline.

    Three measured facts determine this design.

    The level of separability cannot be used, because it depends on the
    dimension of the nuisance control set: under a true null the max-level
    statistic rejects 0.03 of the time at p = 96 and 0.99 at p = 1200. A
    zero-sum contrast on the depth profile cancels that term algebraically.

    The contrast cannot be compared against a flat profile, because real
    depth profiles are not flat and fail in both directions. If surface
    decodability rises with depth the uncorrected contrast rejects a true null
    essentially always; if it falls, which is what a residual stream with
    accumulating noise produces, the contrast loses all power. Recentring on
    the placebo baseline removes both failures: measured false positive rates
    fall from 1.000 to 0.020 in the rising case while power is restored in the
    falling case.

    The null cannot be an item bootstrap, which holds the fitted probe fixed
    and rejects a true null about 0.13 of the time. Permuting labels against
    the label-free smoother refits the direction and is calibrated.
    """
    rng = rng or np.random.default_rng(0)
    y = np.asarray(y).astype(int)
    ba = np.asarray(ba_observed, dtype=float)
    w = profile_weights(len(ba)) if weights is None else np.asarray(weights)
    if abs(w.sum()) > 1e-9:
        raise ValueError("contrast weights must sum to zero")

    raw = float(w @ ba)
    base = 0.0 if placebo_profile is None else float(
        w @ np.asarray(placebo_profile, dtype=float))
    obs = raw - base

    Y = np.stack([rng.permutation(y) for _ in range(B)], axis=1).astype(float)
    tnull = w @ np.stack([_ba_perm_batch(sm, Y) for sm in smoothers])
    p = float((1.0 + np.sum(tnull >= obs)) / (1.0 + B))
    return {"T_adj": obs, "T_raw": raw, "baseline": base, "p_value": p,
            "ba_by_layer": [float(v) for v in ba],
            "null_q95": float(np.quantile(tnull, 0.95)),
            "recentred": placebo_profile is not None}


def streaming_profile_and_null(
    layer_activations,
    idx: np.ndarray,
    y: np.ndarray,
    cfg: "RSCPConfig | None" = None,
    weights: np.ndarray | None = None,
    B: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Observed depth profile and the contrast's permutation null, one pass.

    ``layer_profile`` followed by ``recentred_contrast_test`` computes exactly
    this, but retains every smoother until the null is evaluated. Since the
    null consumes them one layer at a time anyway, nothing requires holding
    them: this accumulates the weighted contribution per layer and releases
    the layer's smoothers immediately. Peak memory goes from
    ``layers x seeds x N^2`` to ``seeds x N^2``, which at 49 layers, 5 seeds
    and N = 3000 is 17.6 GB against 0.36 GB.

    The arithmetic is the same operations in the same order per layer, on the
    same permutations, so results agree to floating-point tolerance. They are
    not guaranteed bit-identical: the retained path forms the weighted sum as
    one BLAS dot product over layers, this one accumulates sequentially, and
    the two associate the additions differently. ``test_rscp.py`` asserts the
    two paths agree.

    The placebo baseline does not enter the null, only the observed statistic,
    so recentring happens afterwards in ``finalise_contrast``. That is what
    lets this be a single pass: the placebo needs ``ba[0]``, which is not
    known until the pass is done.
    """
    cfg = cfg or RSCPConfig()
    rng = rng or np.random.default_rng(0)
    idx = np.asarray(idx)
    y = np.asarray(y).astype(int)
    N = idx.size
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    L = len(layer_activations)
    w = profile_weights(L) if weights is None else np.asarray(weights, float)
    if abs(w.sum()) > 1e-9:
        raise ValueError("contrast weights must sum to zero")

    # Drawn once, before the loop, so every layer sees the same permutations.
    Y = np.stack([rng.permutation(y) for _ in range(B)], axis=1).astype(float)

    ba = np.empty(L, dtype=float)
    tnull = np.zeros(B, dtype=float)
    for i, H in enumerate(layer_activations):
        X = np.asarray(H)[idx]
        lam = lam_for_df(X, cfg.target_df)
        sm = [cross_fit_smoother(
                  X, _make_folds(N, cfg.n_folds, np.random.default_rng(s)), lam)
              for s in cfg.seeds]
        ba[i] = _ba_from_c(correctness_vector(sm, y), pos, neg)
        tnull += w[i] * _ba_perm_batch(sm, Y)
        del sm                      # the point of the exercise
    return ba, tnull


def finalise_contrast(
    ba_observed: np.ndarray,
    tnull: np.ndarray,
    placebo_profile: np.ndarray | None,
    weights: np.ndarray | None = None,
    base_sd: float | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Recentre and test, given a null from ``streaming_profile_and_null``.

    Output matches ``recentred_contrast_test`` field for field.

    ``base_sd`` is the sampling standard deviation of the placebo baseline,
    measured by re-estimating it across split seeds. Supply it and the null is
    widened to account for it, which is the honest test.

    The permutation null describes the variability of the observed contrast
    with the baseline held fixed. But the baseline is itself an estimate, from
    a random split of the reference set at a searched noise level, and its
    spread depends on how well the surface key tracks what separates the two
    sets. Measured on a real 49-layer audit it was 0.0057 against a null
    standard deviation of 0.0043, so the unmodelled component was the larger
    of the two and an apparent p of 0.0075 was really about 0.065.

    Widening assumes the baseline's error is independent of the permutation
    null and roughly normal. Both are approximations, and both are far better
    than treating an estimated quantity as known.
    """
    ba = np.asarray(ba_observed, dtype=float)
    w = profile_weights(len(ba)) if weights is None else np.asarray(weights)
    if abs(w.sum()) > 1e-9:
        raise ValueError("contrast weights must sum to zero")
    raw = float(w @ ba)
    base = 0.0 if placebo_profile is None else float(
        w @ np.asarray(placebo_profile, dtype=float))
    obs = raw - base
    B = len(tnull)
    p = float((1.0 + np.sum(tnull >= obs)) / (1.0 + B))

    p_infl = None
    if base_sd is not None and base_sd > 0 and placebo_profile is not None:
        r = rng or np.random.default_rng(20260815)
        widened = np.asarray(tnull) + r.normal(0.0, float(base_sd), B)
        p_infl = float((1.0 + np.sum(widened >= obs)) / (1.0 + B))

    return {"T_adj": obs, "T_raw": raw, "baseline": base, "p_value": p,
            "p_value_baseline_inflated": p_infl,
            "base_sd": None if base_sd is None else float(base_sd),
            "ba_by_layer": [float(v) for v in ba],
            "null_q95": float(np.quantile(tnull, 0.95)),
            # Reported so it can be compared against the placebo's own spread.
            # The null is built by permuting labels in the observed comparison
            # and treats `base` as a known constant. When the baseline's
            # sampling standard deviation approaches this one, the p-value
            # understates the uncertainty and the verdict is not trustworthy.
            "null_sd": float(np.std(tnull, ddof=1)),
            "recentred": placebo_profile is not None}


def contrast_verdict(
    *,
    exchangeability: str,
    ba_nuisance: float,
    nuis_block: float,
    acknowledged: bool,
    recentred: bool,
    placebo_status: str | None,
    p_value: float,
    p_value_inflated: float | None = None,
    base_sd: float | None = None,
    sd_ratio: float | None = None,
    alpha: float = 0.05,
) -> tuple[str, bool]:
    """Decide the verdict. Pure, so the gates can be tested without an audit.

    Returns ``(verdict, blocked)``. Two preconditions can veto a positive
    finding, and both exist because a measurement said they had to.

    Requirement E. When a classifier with no access to the model separates the
    two sets, any statistic computed on them confounds memorisation with
    distribution shift. This is the defining property of a temporal split.

    Baseline variance. The permutation null describes the contrast with the
    placebo baseline held fixed, but the baseline is estimated from a random
    split of the reference set, and its spread depends on how well the surface
    key tracks what separates the sets. Measured on a real audit it was 0.0057
    against a null sd of 0.0043, turning p = 0.0075 into roughly 0.065. A
    result that survives only while an estimated quantity is treated as known
    is not a result.

    This function exists as a separate unit because both gates were, at
    different times, believed to be wired when they were not. Logic that can
    veto a finding should be testable without running the finding.
    """
    if exchangeability == "failed" and not acknowledged:
        return (f"NO VERDICT. BA_nuisance = {ba_nuisance:.3f} exceeds "
                f"{nuis_block:.2f}: a classifier with no access to the model "
                "separates these two sets almost perfectly, so they are not "
                "exchangeable and Requirement E fails. This is the defining "
                "property of a temporal split, and it is why published "
                "membership-inference numbers on such splits are not "
                "interpretable. The contrast is reported above as a "
                "measurement, not as evidence of contamination. Pass "
                "--acknowledge-non-exchangeable to override, and say so in "
                "any writeup.", True)

    if not recentred:
        return (f"NO BASELINE ({placebo_status}). Without a level-matched "
                "placebo the contrast assumes surface decodability is flat in "
                "depth, which Phase 0c shows fails in both directions. No "
                "claim admissible. Supply a surface key with spread, or more "
                "reference items.", False)

    pre = ("[REQUIREMENT E OVERRIDDEN] " if exchangeability == "failed" else
           "[EXCHANGEABILITY MARGINAL] " if exchangeability == "marginal"
           else "")

    if p_value >= alpha:
        return (f"Contrast null (p={p_value:.4f}). No evidence at this "
                "sensitivity. Report the detection floor so the null can be "
                "read at the right strength.", False)

    if base_sd is None:
        return (pre + f"PROVISIONAL. Contrast significant (p={p_value:.4f}) "
                "holding the placebo baseline fixed, but the baseline is an "
                "estimate and its variance has not been measured. Re-run with "
                "--placebo-reps 8 before reporting this: on a real audit that "
                "correction turned p = 0.0075 into 0.065.", False)

    if p_value_inflated is not None and p_value_inflated >= alpha:
        return (pre + f"NO VERDICT. The contrast is significant "
                f"(p={p_value:.4f}) only while the placebo baseline is treated "
                f"as known. Its measured standard deviation is {base_sd:.5f}, "
                f"{sd_ratio:.2f} times the null's own, and propagating it "
                f"gives p = {p_value_inflated:.4f}. The result is inside the "
                "noise of the correction used to produce it.", True)

    return (pre + f"Contrast significant (p={p_value:.4f}, "
            f"p={p_value_inflated:.4f} with the baseline's variance "
            "propagated) after recentring on the model's own null depth "
            "profile. Evidence of depth-dependent familiarity beyond surface "
            "separability. Report an exposure interval only if a same-family "
            "calibration exists.", False)


def profile_contrast_test(
    c_layers: Sequence[np.ndarray],
    y: np.ndarray,
    weights: np.ndarray | None = None,
    B: int = 2000,
    rng: np.random.Generator | None = None,
    smoothers: Sequence[Sequence[np.ndarray]] | None = None,
    null: str = "permutation",
) -> dict:
    """PRIMARY test: depth-profile contrast against a label-permutation null.

    ``c_layers`` must be indexed from layer 0 upward, since layer 0 anchors the
    profile at the level a bag of token embeddings can reach.

    null="permutation" (default, and the calibrated choice) needs ``smoothers``,
    a list over layers of the per-fold-seed cross-fitted smoothers. Phase 0
    measures a false positive rate of about 0.05 for this null against about
    0.13 for the item bootstrap, across nuisance block dimensions from 96 to
    1200. The bootstrap holds the fitted probe fixed and so understates the
    variance of the statistic; permutation refits it.

    null="bootstrap" resamples items with the fit held fixed. It is retained
    as a comparator and for confidence intervals on the effect size, but it
    should not be used for the headline p-value.

    Validity rests on one assumption, that nuisance decodability is flat in
    depth. Phase 0 measures what happens when it is not: with nuisance
    separability growing 50 percent from embeddings to final layer the test
    rejects essentially always under a true null. Run placebo_contrast before
    trusting any positive result.
    """
    rng = rng or np.random.default_rng(0)
    y = np.asarray(y).astype(int)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("profile_contrast_test needs both classes present")
    w = profile_weights(len(c_layers)) if weights is None else np.asarray(weights)
    if abs(w.sum()) > 1e-9:
        raise ValueError("contrast weights must sum to zero")

    ba = np.array([_ba_from_c(c, pos, neg) for c in c_layers])
    obs = float(w @ ba)
    if not np.isfinite(obs):
        raise ValueError("contrast statistic is not finite; check the inputs")

    # Bootstrap over items: used for the confidence interval on the effect,
    # and as the comparator null.
    ip = rng.integers(0, len(pos), size=(B, len(pos)))
    ine = rng.integers(0, len(neg), size=(B, len(neg)))
    P, Ng = pos[ip], neg[ine]
    bab = np.stack([0.5 * (c[P].mean(1) + c[Ng].mean(1)) for c in c_layers])
    boot = w @ bab
    se = float(boot.std(ddof=1))
    p_boot = float((1.0 + np.sum(boot - obs >= obs)) / (1.0 + B))

    p_perm = None
    if null == "permutation":
        if smoothers is None:
            raise ValueError("null='permutation' requires the smoothers; pass "
                             "smoothers=[[Pi_seed0, ...], ...] per layer, or "
                             "use null='bootstrap' and accept the miscalibration")
        Y = np.stack([rng.permutation(y) for _ in range(B)], axis=1).astype(float)
        null_ba = np.stack([_ba_perm_batch(sm, Y) for sm in smoothers])
        tnull = w @ null_ba
        p_perm = float((1.0 + np.sum(tnull >= obs)) / (1.0 + B))

    return {
        "T_contrast": obs, "se": se, "weights": w, "ba_by_layer": ba,
        "p_value": p_perm if p_perm is not None else p_boot,
        "null": null,
        "p_permutation": p_perm, "p_bootstrap": p_boot,
        "ci95": (float(np.quantile(boot, 0.025)),
                 float(np.quantile(boot, 0.975))),
    }


def placebo_contrast(
    layer_activations: Sequence[np.ndarray],
    reference_index: np.ndarray,
    surface_key: np.ndarray,
    cfg: "RSCPConfig | None" = None,
    min_per_side: int = 50,
) -> dict:
    """Required validity check for the flat-in-depth assumption.

    Split the REFERENCE set in two on a purely surface variable (prefix
    length, source subdomain, anything that cannot correlate with membership
    because every item here is a non-member), then run the contrast. Both
    halves are non-members, so a significant contrast can only mean that
    surface separability grows with depth in this model, which invalidates the
    contrast as evidence of memorisation. Report the placebo p-value beside
    every contrast result.

    Returns ``status`` alongside the usual fields. A degenerate surface key
    (all items tied, or too few per side) cannot produce a placebo, and the
    caller must treat that as "unknown", never as "passed". Returning a NaN
    statistic here silently blocks or waves through real audits, so we refuse
    instead.
    """
    cfg = cfg or RSCPConfig()
    idx = np.asarray(reference_index)
    k = np.asarray(surface_key, dtype=float)[idx]

    if np.unique(k).size < 2:
        return {"status": "not computable: surface key is constant",
                "p_value": None, "T_contrast": None, "n_per_side": 0}

    # Rank with a deterministic random tiebreak so ties split evenly rather
    # than piling on one side of the median.
    rng = np.random.default_rng(cfg.rng_seed + 31337)
    order = np.lexsort((rng.random(k.size), k))
    half = k.size // 2
    lo_pos, hi_pos = order[:half], order[k.size - half:]
    n = min(lo_pos.size, hi_pos.size)
    if n < min_per_side:
        return {"status": f"not computable: only {n} items per side "
                          f"(need {min_per_side})",
                "p_value": None, "T_contrast": None, "n_per_side": int(n)}

    sel = np.concatenate([hi_pos[:n], lo_pos[:n]])
    y = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    sub = idx[sel]

    C, smoothers = [], []
    for H in layer_activations:
        X = np.asarray(H)[sub]
        lam = lam_for_df(X, cfg.target_df)
        sm = [cross_fit_smoother(
                  X, _make_folds(len(sub), cfg.n_folds,
                                 np.random.default_rng(s)), lam)
              for s in cfg.seeds]
        smoothers.append(sm)
        C.append(correctness_vector(sm, y))

    out = profile_contrast_test(C, y, B=cfg.n_bootstrap,
                                rng=np.random.default_rng(cfg.rng_seed + 31337),
                                smoothers=smoothers, null="permutation")
    out["status"] = "ok"
    out["n_per_side"] = int(n)
    return out


def _select_lambda(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: Sequence[float],
    n_folds: int,
    rng: np.random.Generator,
    standardise: bool,
) -> float:
    """Pick lambda on an inner split of the data only.

    Note: lambda selection uses labels, so strictly it must be redone inside
    each permutation for exact validity. In practice the selected value is
    stable, so we select once on the observed labels and hold it fixed across
    permutations; this is recorded as a deliberate approximation and is one of
    the things the null-calibration experiment checks empirically.
    """
    N = X.shape[0]
    inner = _make_folds(N, n_folds, rng)
    best, best_ba = lambdas[0], -np.inf
    for lam in lambdas:
        Pi = cross_fit_smoother(X, inner, lam, standardise)
        ba = ba_from_smoother(Pi, y)
        if ba > best_ba:
            best, best_ba = lam, ba
    return best


# --------------------------------------------------------------------------
# The protocol
# --------------------------------------------------------------------------

def _mean_sd_ba(
    X: np.ndarray,
    y: np.ndarray,
    cfg: RSCPConfig,
    lam: float | None = None,
) -> tuple[float, float, list[np.ndarray], float]:
    """Balanced accuracy averaged over fold-seeds; also returns the smoothers."""
    bas, smoothers = [], []
    if lam is None:
        # Degrees of freedom are matched across blocks rather than the penalty.
        lam = lam_for_df(X, cfg.target_df)
    for s in cfg.seeds:
        fold = _make_folds(X.shape[0], cfg.n_folds, np.random.default_rng(s))
        Pi = cross_fit_smoother(X, fold, lam, cfg.standardise)
        smoothers.append(Pi)
        bas.append(ba_from_smoother(Pi, y))
    return float(np.mean(bas)), float(np.std(bas)), smoothers, lam


def nuisance_strata(
    nuisance_score: np.ndarray, n_strata: int = 10
) -> np.ndarray:
    """Bin items by their cross-fitted nuisance score.

    Used by the conditional permutation test. Under the null of no signal
    beyond nuisance, labels are exchangeable WITHIN a stratum of items that
    the nuisance family scores alike, but not across strata. Permuting
    globally instead would destroy the nuisance structure and test the wrong
    null, which is both a validity problem and, as Phase 0 shows, a severe
    power problem: the global null sits at BA = 0.5 where sampling variance
    is maximal while the observed statistic sits well above it.
    """
    s = np.asarray(nuisance_score, dtype=float)
    edges = np.quantile(s, np.linspace(0, 1, n_strata + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return np.clip(np.digitize(s, edges[1:-1]), 0, n_strata - 1)


def stratified_permutations(
    y: np.ndarray, strata: np.ndarray, B: int, rng: np.random.Generator
) -> np.ndarray:
    """(N, B) matrix of labels permuted within strata."""
    y = np.asarray(y)
    N = y.shape[0]
    Y = np.empty((N, B), dtype=np.float64)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    for b in range(B):
        col = np.empty(N, dtype=np.float64)
        for g in groups:
            col[g] = y[rng.permutation(g)]
        Y[:, b] = col
    return Y


def correctness_vector(
    smoothers: Sequence[np.ndarray], y: np.ndarray
) -> np.ndarray:
    """Per-item out-of-fold correctness, averaged over fold-seeds.

    c_i in [0, 1] is the fraction of fold-seeds under which the cross-fitted
    probe places item i on the correct side of the median threshold. This is
    the object the primary bootstrap resamples.
    """
    y = np.asarray(y).astype(int)
    ytilde = 2.0 * y - 1.0
    acc = np.zeros(y.shape[0], dtype=np.float64)
    for Pi in smoothers:
        u = Pi @ ytilde
        pred = (u > np.median(u)).astype(int)
        acc += (pred == y).astype(float)
    return acc / len(smoothers)


def _ba_from_c(c: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> float:
    return 0.5 * (c[pos].mean() + c[neg].mean())


def paired_bootstrap_test(
    c_layers: Sequence[np.ndarray],
    c_nuis: np.ndarray,
    y: np.ndarray,
    B: int = 2000,
    rng: np.random.Generator | None = None,
    studentise: bool = True,
) -> dict:
    """Primary test: studentised paired bootstrap with a max over layers.

    Delta_l = BA_l - BA_nuis is a paired comparison of two accuracies measured
    on the same items, so its null distribution comes from resampling items,
    not from permuting labels. Label permutation drives both terms to 0.5,
    where the sampling variance of balanced accuracy is maximal, which makes
    the resulting test valid but badly underpowered (see Phase 0). Stratifying
    the permutation on the fitted nuisance score to preserve that structure is
    circular and anti-conservative. The bootstrap avoids both problems.

    Family-wise error over layers is handled Westfall-Young style: the null
    is the distribution of the maximum centred statistic, evaluated at the
    least favourable configuration Delta_l = 0 for every l.

    Items are resampled within class so the balanced design is preserved. The
    probe fit is held fixed; cross-fitting makes every score out of fold, so
    the evaluation variance this captures is the dominant term, but this is an
    approximation and Phase 0 is what checks that it calibrates.
    """
    rng = rng or np.random.default_rng(0)
    y = np.asarray(y).astype(int)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    L = len(c_layers)

    ba_n = _ba_from_c(c_nuis, pos, neg)
    delta = np.array([_ba_from_c(c, pos, neg) - ba_n for c in c_layers])

    ip = rng.integers(0, len(pos), size=(B, len(pos)))
    ineg = rng.integers(0, len(neg), size=(B, len(neg)))
    P, Ng = pos[ip], neg[ineg]

    ban_b = 0.5 * (c_nuis[P].mean(axis=1) + c_nuis[Ng].mean(axis=1))
    d_b = np.empty((L, B))
    for l, c in enumerate(c_layers):
        d_b[l] = 0.5 * (c[P].mean(axis=1) + c[Ng].mean(axis=1)) - ban_b

    se = d_b.std(axis=1, ddof=1)
    se[se < 1e-12] = 1e-12
    if studentise:
        t_obs = delta / se
        t_null = ((d_b - delta[:, None]) / se[:, None]).max(axis=0)
    else:
        t_obs = delta
        t_null = (d_b - delta[:, None]).max(axis=0)

    l_star = int(np.argmax(t_obs))
    T = float(t_obs[l_star])
    p = float((1.0 + np.sum(t_null >= T)) / (1.0 + B))
    return {
        "delta": delta, "se": se, "t": t_obs, "T": T, "l_star": l_star,
        "p_value": p, "ba_nuisance": ba_n, "t_null": t_null,
        "delta_at_lstar": float(delta[l_star]),
        "ci95_at_lstar": (
            float(np.quantile(d_b[l_star], 0.025)),
            float(np.quantile(d_b[l_star], 0.975)),
        ),
    }


def _ba_batch(Pi: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Balanced accuracy for a batch of label vectors.

    Y is (N, B) with entries in {0, 1}. Returns (B,).
    """
    U = Pi @ (2.0 * Y - 1.0)                      # (N, B)
    tau = np.median(U, axis=0, keepdims=True)
    pred = U > tau
    n1 = Y.sum(axis=0)
    n0 = Y.shape[0] - n1
    tp = (pred & (Y == 1)).sum(axis=0)
    tn = ((~pred) & (Y == 0)).sum(axis=0)
    return 0.5 * (tp / np.maximum(n1, 1) + tn / np.maximum(n0, 1))


def run_rscp(
    layer_activations: Sequence[np.ndarray],
    nuisance_features: np.ndarray,
    y: np.ndarray,
    cfg: RSCPConfig | None = None,
    return_null: bool = False,
) -> RSCPResult:
    """Run the full protocol.

    Parameters
    ----------
    layer_activations : list of (N, d_l) arrays, index = layer. Element 0 is
        the embedding layer; it is excluded from the max statistic when
        cfg.skip_layer_zero is set, because a bag of token embeddings is a
        nuisance feature rather than model computation.
    nuisance_features : (N, p) concatenated nuisance family.
    y : (N,) binary membership labels, 1 = suspect set.
    """
    cfg = cfg or RSCPConfig()
    y = np.asarray(y).astype(np.int64)
    N = y.shape[0]
    rng = np.random.default_rng(cfg.rng_seed)

    ba_n, sd_n, sm_n, lam_n = _mean_sd_ba(nuisance_features, y, cfg)

    layer_res, layer_smoothers = [], []
    for li, H in enumerate(layer_activations):
        ba, sd, sm, _ = _mean_sd_ba(H, y, cfg)
        layer_res.append(LayerResult(li, ba, sd))
        layer_smoothers.append(sm)

    lo = 1 if cfg.skip_layer_zero else 0
    scan = list(range(lo, len(layer_res)))

    # Per-item out-of-fold correctness, the object the bootstrap resamples.
    c_nuis = correctness_vector(sm_n, y)
    c_layers = [correctness_vector(layer_smoothers[i], y) for i in scan]

    # Descriptive level statistics (biased by relative block capacity, so
    # reported but not used for inference).
    bt = paired_bootstrap_test(
        c_layers, c_nuis, y, B=cfg.n_bootstrap, rng=rng,
        studentise=cfg.studentise,
    )
    # Primary inference: the depth-profile contrast over layers 0..L.
    c_all = [correctness_vector(sm, y) for sm in layer_smoothers]
    ct = profile_contrast_test(c_all, y, B=cfg.n_bootstrap,
                               rng=np.random.default_rng(cfg.rng_seed + 5),
                               smoothers=layer_smoothers, null="permutation")
    deltas = [layer_res[i].ba_mean - ba_n for i in range(len(layer_res))]
    l_star = scan[bt["l_star"]]
    T = float(bt["delta_at_lstar"])
    p = bt["p_value"]

    res = RSCPResult(
        ba_nuisance=ba_n,
        ba_nuisance_sd=sd_n,
        layers=layer_res,
        delta=deltas,
        T=T,
        layer_star=int(l_star),
        p_value=float(p),
        null_quantiles={
            "t_null_q95": float(np.quantile(bt["t_null"], 0.95)),
            "t_null_q99": float(np.quantile(bt["t_null"], 0.99)),
            "ci95_lo": bt["ci95_at_lstar"][0],
            "ci95_hi": bt["ci95_at_lstar"][1],
        },
        n_per_set=int(min((y == 1).sum(), (y == 0).sum())),
        notes=("primary test: depth-profile contrast, paired bootstrap. "
               "Level statistics are descriptive only."),
    )
    res.contrast = float(ct["T_contrast"])
    res.contrast_p = float(ct["p_value"])
    res.contrast_se = float(ct["se"])
    res.null_quantiles["level_p_descriptive"] = float(bt["p_value"])

    if cfg.also_run_permutation:
        B = cfg.n_permutations
        Yp = np.empty((N, B), dtype=np.float64)
        for b in range(B):
            Yp[:, b] = rng.permutation(y)
        ba_null_nuis = np.mean([_ba_batch(Pi, Yp) for Pi in sm_n], axis=0)
        T_null = np.full(B, -np.inf)
        for i in scan:
            ba_l = np.mean([_ba_batch(Pi, Yp) for Pi in layer_smoothers[i]], axis=0)
            T_null = np.maximum(T_null, ba_l - ba_null_nuis)
        res.null_quantiles["perm_p"] = float(
            (1.0 + np.sum(T_null >= T)) / (1.0 + B))
        res.null_quantiles["perm_q95"] = float(np.quantile(T_null, 0.95))
        if return_null:
            res.__dict__["_T_null_perm"] = T_null

    if return_null:
        res.__dict__["_t_null"] = bt["t_null"]
    return res


def permutation_null(res: RSCPResult) -> np.ndarray | None:
    """Retrieve the attached bootstrap null, if run_rscp was given return_null."""
    return res.__dict__.get("_t_null")


# --------------------------------------------------------------------------
# Required controls
# --------------------------------------------------------------------------

def control_task(
    layer_activations, nuisance_features, y, cfg: RSCPConfig | None = None
) -> RSCPResult:
    """Hewitt and Liang control: labels randomised independently of membership."""
    cfg = cfg or RSCPConfig()
    rng = np.random.default_rng(cfg.rng_seed + 9001)
    y_rand = rng.permutation(np.asarray(y))
    return run_rscp(layer_activations, nuisance_features, y_rand, cfg)


def random_direction_baseline(
    layer_activations, nuisance_features, y, r: int = 64,
    cfg: RSCPConfig | None = None,
) -> RSCPResult:
    """Project activations onto a random r-dimensional subspace and rerun."""
    cfg = cfg or RSCPConfig()
    rng = np.random.default_rng(cfg.rng_seed + 4242)
    proj = []
    for H in layer_activations:
        d = H.shape[1]
        P = rng.standard_normal((d, min(r, d))) / math.sqrt(d)
        proj.append(H @ P)
    return run_rscp(proj, nuisance_features, y, cfg)


def internal_null(
    layer_activations, nuisance_features, y, cfg: RSCPConfig | None = None
) -> RSCPResult:
    """Split the reference set against itself. An exact instance of E1."""
    cfg = cfg or RSCPConfig()
    rng = np.random.default_rng(cfg.rng_seed + 777)
    ref = np.flatnonzero(np.asarray(y) == 0)
    rng.shuffle(ref)
    half = len(ref) // 2
    sub = np.concatenate([ref[:half], ref[half: 2 * half]])
    y_new = np.zeros(len(sub), dtype=np.int64)
    y_new[:half] = 1
    return run_rscp(
        [H[sub] for H in layer_activations], nuisance_features[sub], y_new, cfg
    )


def length_matched_indices(
    lengths: np.ndarray, y: np.ndarray, n_bins: int = 10, seed: int = 0
) -> np.ndarray:
    """Indices of a subsample whose length distribution matches across sets."""
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths)
    y = np.asarray(y)
    edges = np.quantile(lengths, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    keep = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (lengths >= lo) & (lengths < hi)
        a = np.flatnonzero(m & (y == 1))
        b = np.flatnonzero(m & (y == 0))
        k = min(len(a), len(b))
        if k:
            keep.append(rng.choice(a, k, replace=False))
            keep.append(rng.choice(b, k, replace=False))
    return np.concatenate(keep) if keep else np.array([], dtype=np.int64)


# --------------------------------------------------------------------------
# Nuisance feature family
# --------------------------------------------------------------------------

def surface_features(texts: Sequence[str]) -> np.ndarray:
    """Length and orthography statistics. Model independent by construction."""
    out = np.zeros((len(texts), 7))
    for i, t in enumerate(texts):
        n = max(len(t), 1)
        words = t.split()
        nw = max(len(words), 1)
        out[i] = [
            len(t),
            nw,
            sum(c.isdigit() for c in t) / n,
            sum(not c.isalnum() and not c.isspace() for c in t) / n,
            len(set(w.lower() for w in words)) / nw,
            float(np.mean([len(w) for w in words])) if words else 0.0,
            sum(c.isupper() for c in t) / n,
        ]
    return out


def char_ngram_features(
    texts: Sequence[str], ngram_range=(3, 5), max_features: int = 20000
) -> np.ndarray:
    """L2-normalised character n-gram TF-IDF. Requires scikit-learn."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=ngram_range,
        max_features=max_features, norm="l2", sublinear_tf=True,
    )
    return vec.fit_transform(texts).toarray()


def bag_of_embeddings(layer0_activations: np.ndarray) -> np.ndarray:
    """Mean-pooled layer-0 activations: a bag of token embeddings.

    This is the natural floor for a representational claim, since it contains
    nothing the model computed. Pass the already-extracted layer 0.
    """
    return np.asarray(layer0_activations)


def build_nuisance(
    texts: Sequence[str] | None = None,
    layer0: np.ndarray | None = None,
    reference_stats: np.ndarray | None = None,
    use_char_ngrams: bool = True,
) -> np.ndarray:
    """Concatenate the declared nuisance family."""
    blocks = []
    if layer0 is not None:
        blocks.append(bag_of_embeddings(layer0))
    if texts is not None:
        blocks.append(surface_features(texts))
        if use_char_ngrams:
            blocks.append(char_ngram_features(texts))
    if reference_stats is not None:
        blocks.append(np.asarray(reference_stats))
    if not blocks:
        raise ValueError("nuisance family is empty")
    return np.concatenate([np.asarray(b, dtype=np.float64) for b in blocks], axis=1)


# --------------------------------------------------------------------------
# Exposure calibration
# --------------------------------------------------------------------------

def fit_exposure_link(m: np.ndarray, T: np.ndarray) -> tuple[float, float]:
    """Fit g(m) = gamma (1 - exp(-beta m)) by least squares (Eq. 9)."""
    from scipy.optimize import curve_fit

    m = np.asarray(m, dtype=float)
    T = np.asarray(T, dtype=float)

    def g(mm, gamma, beta):
        return gamma * (1.0 - np.exp(-beta * mm))

    p0 = [max(T.max(), 1e-3), 0.2]
    popt, _ = curve_fit(g, m, T, p0=p0, maxfev=20000,
                        bounds=([0.0, 1e-4], [1.0, 10.0]))
    return float(popt[0]), float(popt[1])


def exposure_link(m, gamma: float, beta: float):
    return gamma * (1.0 - np.exp(-beta * np.asarray(m, dtype=float)))


def invert_exposure(
    T_obs: float, gamma: float, beta: float, sigma: float,
    grid: Iterable[float] = tuple(np.arange(0, 64.01, 0.25)),
    z: float = 1.96,
) -> tuple[float, float]:
    """Effective-exposure interval: all m whose predicted T covers T_obs."""
    grid = np.asarray(list(grid), dtype=float)
    pred = exposure_link(grid, gamma, beta)
    ok = np.abs(pred - T_obs) <= z * sigma
    if not ok.any():
        return (float("nan"), float("nan"))
    return (float(grid[ok].min()), float(grid[ok].max()))


def detection_floor(
    m_grid: Sequence[float], power: Sequence[float], target: float = 0.8
) -> float:
    """Smallest exposure count reaching the target power. Linear interpolation."""
    m_grid = np.asarray(m_grid, dtype=float)
    power = np.asarray(power, dtype=float)
    hit = np.flatnonzero(power >= target)
    if hit.size == 0:
        return float("inf")
    i = hit[0]
    if i == 0:
        return float(m_grid[0])
    x0, x1 = m_grid[i - 1], m_grid[i]
    y0, y1 = power[i - 1], power[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


# --------------------------------------------------------------------------
# Activation extraction (requires torch + transformers)
# --------------------------------------------------------------------------

def _require_torch():
    """Import torch and transformers, reporting *why* if it fails.

    An earlier version reported "needs torch and transformers" for every
    failure, which is wrong and wastes the user's time when both are installed
    and one of them is refusing to load for another reason. The commonest such
    reason is a version clash between transformers and huggingface-hub.
    """
    import importlib.util
    missing = [m for m in ("torch", "transformers")
               if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError(
            "Activation extraction needs " + " and ".join(missing) + ". The "
            "statistical core runs without them; install with "
            "`pip install torch transformers`.")
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        msg = str(exc)
        hint = ""
        if "huggingface-hub" in msg or "huggingface_hub" in msg:
            hint = ("\n\nThis is a version clash, not a missing package. "
                    "Either upgrade transformers to a release that accepts "
                    "your hub version:\n"
                    "    pip install -U transformers\n"
                    "or pin the hub back to the range transformers wants:\n"
                    "    pip install 'huggingface-hub>=0.34,<1.0'")
        raise RuntimeError(
            "torch and transformers are installed but transformers failed to "
            "import:\n\n    " + msg + hint) from exc


def extract_prefix_activations(
    model_name: str,
    prefixes: Sequence[str],
    pooling: str = "last",
    device: str | None = None,
    batch_size: int = 8,
    max_length: int = 512,
    dtype: str = "float16",
    cache_dir: str | None = None,
    progress: bool = True,
):
    """Residual-stream activations at every layer, read at the item PREFIX.

    The prefix rule (Section 4.3) is enforced here rather than left to the
    caller: pass only the text the model sees before it must answer. Passing
    a string that contains the gold answer defeats the protocol.

    pooling : "last" for the final prefix token, "mean" for mean over the
              prefix. "last" is the primary extraction point.

    Three things matter for making this run on a laptop.

    We load AutoModel, not AutoModelForCausalLM. The language-model head
    projects every position onto the vocabulary, which for a 50k vocabulary
    and a batch of long sequences is gigabytes of logits we never read. The
    hidden states are all the protocol uses.

    On out-of-memory we halve the batch and retry rather than dying, which
    matters on Apple silicon where the memory ceiling is shared with the rest
    of the machine.

    Results are cached by (model, pooling, max_length, items), so re-running an
    audit with different statistics costs nothing.
    """
    _require_torch()
    import hashlib
    import torch
    from transformers import AutoModel, AutoTokenizer

    if pooling not in {"last", "mean"}:
        raise ValueError("pooling must be 'last' or 'mean'")

    if cache_dir is None:
        from paths import CACHE
        cache_dir = str(CACHE / "activations")
    key = hashlib.sha256(
        ("\x00".join(prefixes) + f"|{model_name}|{pooling}|{max_length}")
        .encode("utf-8")).hexdigest()[:16]
    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{key}.npz")
        if os.path.exists(path):
            with np.load(path) as z:
                out = [z[f"l{i}"] for i in range(len(z.files))]
            if progress:
                print(f"    cached activations: {path}", flush=True)
            return out

    device = device or ("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(
        model_name, dtype=getattr(torch, dtype)).to(device).eval()

    def _empty_cache():
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    chunks: list[list[np.ndarray]] = []
    bs = batch_size
    s_i = 0
    n = len(prefixes)
    with torch.no_grad():
        while s_i < n:
            batch = list(prefixes[s_i: s_i + bs])
            try:
                enc = tok(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length).to(device)
                hs = model(**enc, output_hidden_states=True).hidden_states
                mask = enc["attention_mask"]
                if pooling == "last":
                    last = mask.sum(dim=1) - 1
                    pooled = [h[torch.arange(h.shape[0]), last] for h in hs]
                else:
                    m = mask.unsqueeze(-1).to(hs[0].dtype)
                    pooled = [(h * m).sum(1) / m.sum(1) for h in hs]
                chunks.append([p.float().cpu().numpy() for p in pooled])
                del hs, pooled, enc
                _empty_cache()
            except RuntimeError as e:
                if "out of memory" not in str(e).lower() or bs == 1:
                    raise
                _empty_cache()
                bs = max(1, bs // 2)
                if progress:
                    print(f"    out of memory; retrying with batch {bs}",
                          flush=True)
                continue
            s_i += len(batch)
            if progress and (s_i % (bs * 20) == 0 or s_i >= n):
                print(f"    {s_i}/{n} prefixes", flush=True)

    n_layers = len(chunks[0])
    out = [np.concatenate([c[i] for c in chunks], axis=0)
           for i in range(n_layers)]
    if path:
        np.savez_compressed(path, **{f"l{i}": a for i, a in enumerate(out)})
        if progress:
            print(f"    cached to {path}", flush=True)
    return out


def reference_likelihood_features(
    model_name: str,
    texts: Sequence[str],
    ks: Sequence[int] = (10, 20),
    device: str | None = None,
    batch_size: int = 8,
    max_length: int = 384,
    cache_dir: str | None = None,
    progress: bool = True,
) -> np.ndarray:
    """Likelihood summaries under an INDEPENDENT reference model.

    Includes the statistics that likelihood-based membership inference uses
    (mean and variance of token log-probability, min-k% mean, zlib ratio).
    Computed on a model that cannot have memorised the items, so whatever
    separability they achieve is attributable to how the two sets differ as
    text. Belongs in the nuisance family, not the evidence.

    Unlike activation extraction this genuinely needs the language-model head,
    so it is the expensive half of an audit. It gets the same treatment:
    a cache, a progress line, out-of-memory backoff, and a shorter default
    truncation. Without those it looks like a hang.
    """
    _require_torch()
    import hashlib
    import zlib
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if cache_dir is None:
        from paths import CACHE
        cache_dir = str(CACHE / "activations")
    key = hashlib.sha256(
        ("\x00".join(texts) + f"|{model_name}|{max_length}|{tuple(ks)}")
        .encode("utf-8")).hexdigest()[:16]
    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"ref_{key}.npy")
        if os.path.exists(path):
            if progress:
                print(f"    cached reference features: {path}", flush=True)
            return np.load(path)

    device = device or ("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    def _empty():
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    feats: list[list[float]] = []
    bs = batch_size
    s_i, n = 0, len(texts)
    with torch.no_grad():
        while s_i < n:
            batch = list(texts[s_i: s_i + bs])
            try:
                enc = tok(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length).to(device)
                logits = model(**enc).logits[:, :-1]
                tgt = enc["input_ids"][:, 1:]
                lp = torch.log_softmax(logits.float(), dim=-1)
                tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                m = enc["attention_mask"][:, 1:].bool()
                rows = []
                for j2, txt in enumerate(batch):
                    v = tok_lp[j2][m[j2]].cpu().numpy()
                    if v.size == 0:
                        v = np.array([0.0])
                    row = [v.mean(), v.var(), v.min()]
                    for k in ks:
                        kk = max(1, int(len(v) * k / 100))
                        row.append(np.sort(v)[:kk].mean())
                    z = len(zlib.compress(txt.encode("utf-8"))) or 1
                    row.append(v.sum() / z)
                    rows.append(row)
                del logits, lp, tok_lp, enc
                _empty()
            except RuntimeError as e:
                if "out of memory" not in str(e).lower() or bs == 1:
                    raise
                _empty()
                bs = max(1, bs // 2)
                if progress:
                    print(f"    out of memory; retrying with batch {bs}",
                          flush=True)
                continue
            feats.extend(rows)
            s_i += len(batch)
            if progress and (s_i % (bs * 20) == 0 or s_i >= n):
                print(f"    reference model {s_i}/{n}", flush=True)

    out = np.asarray(feats, dtype=np.float64)
    if path:
        np.save(path, out)
        if progress:
            print(f"    cached to {path}", flush=True)
    return out


# --------------------------------------------------------------------------
# Timing helper, used for the cost claim in the paper
# --------------------------------------------------------------------------

def time_permutation_strategies(
    X: np.ndarray, y: np.ndarray, n_perm: int = 100,
    lam: float = 1.0, n_folds: int = 5,
) -> dict:
    """Compare the smoother shortcut against refitting per permutation."""
    N = X.shape[0]
    rng = np.random.default_rng(0)
    fold = _make_folds(N, n_folds, rng)

    t0 = time.perf_counter()
    Pi = cross_fit_smoother(X, fold, lam)
    t_setup = time.perf_counter() - t0

    Y = np.stack([rng.permutation(y) for _ in range(n_perm)], axis=1).astype(float)
    t0 = time.perf_counter()
    _ = _ba_batch(Pi, Y)
    t_fast = time.perf_counter() - t0

    t0 = time.perf_counter()
    for b in range(n_perm):
        Pib = cross_fit_smoother(X, fold, lam)   # refit, as a non-linear probe forces
        _ = ba_from_smoother(Pib, Y[:, b].astype(int))
    t_slow = time.perf_counter() - t0

    return {
        "n": int(N), "d": int(X.shape[1]), "n_perm": int(n_perm),
        "setup_s": t_setup, "shortcut_s": t_fast, "refit_s": t_slow,
        "speedup": t_slow / max(t_fast, 1e-9),
    }


def save_result(res: RSCPResult, path: str) -> None:
    d = asdict(res)
    d["layers"] = [asdict(l) for l in res.layers]
    with open(path, "w") as fh:
        json.dump(d, fh, indent=2)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="RSCP reference implementation")
    ap.add_argument("--selftest", action="store_true",
                    help="Run a small synthetic sanity check.")
    a = ap.parse_args()
    if a.selftest:
        rng = np.random.default_rng(0)
        N, d, L = 400, 64, 8
        y = np.zeros(N, dtype=int); y[: N // 2] = 1
        s = (2 * y - 1)[:, None]
        v = rng.standard_normal(d); v /= np.linalg.norm(v)
        layers = [rng.standard_normal((N, d)) + 0.30 * s * v for _ in range(L + 1)]
        layers[0] = rng.standard_normal((N, d)) + 0.30 * s * v  # nuisance only
        nuis = rng.standard_normal((N, 16)) + 0.30 * s * rng.standard_normal(16)
        r = run_rscp(layers, nuis, y, RSCPConfig(n_permutations=200, seeds=(0, 1)))
        print(r.summary())
