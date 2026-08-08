"""
test_rscp.py
Property tests for the RSCP implementation.

These check the mathematical claims the method rests on, not just that the
code runs. Each test corresponds to a statement made in the paper, and if the
statement is false the test fails.

    pip install pytest
    pytest -q test_rscp.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import rscp
from rscp import (
    RSCPConfig,
    _ba_from_c,
    _ba_perm_batch,
    _make_folds,
    balanced_accuracy,
    correctness_vector,
    cross_fit_smoother,
    detection_floor,
    fit_exposure_link,
    invert_exposure,
    lam_for_df,
    layer_profile,
    level_matched_placebo,
    profile_weights,
    recentred_contrast_test,
    surface_features,
)
from simulators import SIMULATORS, layer_bump, placebo_key

CFG = RSCPConfig(seeds=(0, 1), n_folds=5, target_df=16.0, n_bootstrap=200)


def _toy(n=60, d=12, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(2 * n, dtype=int)
    y[:n] = 1
    y = rng.permutation(y)
    X = rng.standard_normal((2 * n, d)) + (2.0 * y - 1.0)[:, None] * 0.3
    return X, y


# --------------------------------------------------------------------------
# The smoother
# --------------------------------------------------------------------------

def test_smoother_is_label_independent():
    """Pi must not depend on y. The whole permutation argument rests on this."""
    X, y = _toy()
    fold = _make_folds(len(y), 5, np.random.default_rng(0))
    Pi = cross_fit_smoother(X, fold, 1.0)
    y2 = np.random.default_rng(1).permutation(y)
    Pi2 = cross_fit_smoother(X, fold, 1.0)
    assert np.allclose(Pi, Pi2)
    assert not np.allclose(Pi @ (2 * y - 1), Pi @ (2 * y2 - 1))


def test_smoother_is_strictly_out_of_fold():
    """Pi[i, j] must be zero whenever i and j share a fold."""
    X, y = _toy()
    fold = _make_folds(len(y), 5, np.random.default_rng(0))
    Pi = cross_fit_smoother(X, fold, 1.0)
    same = fold[:, None] == fold[None, :]
    assert np.abs(Pi[same]).max() == 0.0
    assert np.abs(Pi[~same]).max() > 0.0


def test_smoother_predictions_are_linear_in_labels():
    """Pi(a y1 + b y2) == a Pi y1 + b Pi y2. This is what makes the
    permutation test cost one matrix-vector product."""
    X, _ = _toy()
    fold = _make_folds(X.shape[0], 5, np.random.default_rng(0))
    Pi = cross_fit_smoother(X, fold, 1.0)
    rng = np.random.default_rng(3)
    y1, y2 = rng.standard_normal(X.shape[0]), rng.standard_normal(X.shape[0])
    assert np.allclose(Pi @ (2.5 * y1 - 1.7 * y2),
                       2.5 * (Pi @ y1) - 1.7 * (Pi @ y2))


def test_dual_and_primal_forms_agree():
    """d <= n and d > n take different code paths and must agree."""
    rng = np.random.default_rng(0)
    n = 40
    X = rng.standard_normal((n, 8))
    fold = _make_folds(n, 4, np.random.default_rng(0))
    Pi_primal = cross_fit_smoother(X, fold, 2.0)
    Xw = np.concatenate([X, np.zeros((n, 200))], axis=1)  # forces the dual form
    Pi_dual = cross_fit_smoother(Xw, fold, 2.0)
    assert np.allclose(Pi_primal, Pi_dual, atol=1e-8)


# --------------------------------------------------------------------------
# Degrees of freedom
# --------------------------------------------------------------------------

@pytest.mark.parametrize("d,target", [(20, 5.0), (50, 12.0), (200, 30.0)])
def test_lam_for_df_hits_its_target(d, target):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, d))
    lam = lam_for_df(X, target)
    Xs = (X - X.mean(0)) / X.std(0)
    s2 = np.linalg.svd(Xs, compute_uv=False) ** 2
    assert abs((s2 / (s2 + lam)).sum() - target) < 0.05 * target


def test_df_matching_equalises_capacity_across_block_sizes():
    """Two blocks of different width, matched on df, must not differ wildly
    in achieved accuracy at equal population separation. This is the bias
    that killed the level statistic."""
    rng = np.random.default_rng(0)
    n = 400
    y = np.zeros(n, dtype=int)
    y[: n // 2] = 1
    s = (2.0 * y - 1.0)[:, None] / 2.0
    accs = []
    for d in (32, 512):
        v = rng.standard_normal(d)
        v /= np.linalg.norm(v)
        X = rng.standard_normal((n, d)) + s * v[None, :]
        fold = _make_folds(n, 5, np.random.default_rng(0))
        Pi = cross_fit_smoother(X, fold, lam_for_df(X, 16.0))
        accs.append(balanced_accuracy(Pi @ (2.0 * y - 1.0), y))
    assert abs(accs[0] - accs[1]) < 0.12, accs


# --------------------------------------------------------------------------
# The contrast
# --------------------------------------------------------------------------

def test_weights_sum_to_zero():
    for n in (3, 8, 13, 40):
        assert abs(profile_weights(n).sum()) < 1e-12


def test_weights_are_increasing_in_depth():
    w = profile_weights(13)
    assert np.all(np.diff(w) > 0)
    assert w[0] < 0 < w[-1]


def test_contrast_cancels_any_constant_added_to_every_layer():
    """Zero-sum weights mean the level of separability drops out exactly.
    That is the algebraic reason the contrast is immune to the capacity bias."""
    w = profile_weights(13)
    ba = np.linspace(0.55, 0.72, 13)
    for shift in (-0.2, 0.0, 0.13, 0.4):
        assert abs(w @ (ba + shift) - w @ ba) < 1e-12


def test_contrast_rejects_nonzero_sum_weights():
    ba = np.linspace(0.5, 0.6, 13)
    bad = np.ones(13) / 13
    rng = np.random.default_rng(0)
    y = np.array([1, 0] * 20)
    sms = [[np.zeros((40, 40))] for _ in range(13)]
    with pytest.raises(ValueError):
        recentred_contrast_test(ba, sms, y, None, weights=bad, B=10, rng=rng)


def test_recentring_subtracts_the_baseline_contrast():
    rng = np.random.default_rng(0)
    L = 13
    ba = np.linspace(0.60, 0.70, L)
    plc = np.linspace(0.60, 0.66, L)
    y = np.array([1] * 20 + [0] * 20)
    sms = [[cross_fit_smoother(rng.standard_normal((40, 5)),
                               _make_folds(40, 5, np.random.default_rng(0)),
                               1.0)] for _ in range(L)]
    w = profile_weights(L)
    out = recentred_contrast_test(ba, sms, y, plc, B=50, rng=rng)
    assert abs(out["T_raw"] - w @ ba) < 1e-12
    assert abs(out["baseline"] - w @ plc) < 1e-12
    assert abs(out["T_adj"] - (w @ ba - w @ plc)) < 1e-12


# --------------------------------------------------------------------------
# The permutation null
# --------------------------------------------------------------------------

def test_perm_batch_matches_the_scalar_path():
    X, y = _toy()
    fold = _make_folds(len(y), 5, np.random.default_rng(0))
    Pi = cross_fit_smoother(X, fold, 1.0)
    Y = np.stack([y, 1 - y], axis=1).astype(float)
    batch = _ba_perm_batch([Pi], Y)
    for j in range(2):
        yj = Y[:, j].astype(int)
        scalar = balanced_accuracy(Pi @ (2.0 * Y[:, j] - 1.0), yj)
        assert abs(batch[j] - scalar) < 1e-12


def test_pvalue_is_bounded_and_never_zero():
    """The (1 + count) / (1 + B) form must never return exactly zero."""
    rng = np.random.default_rng(0)
    L, n = 13, 30
    y = np.array([1] * n + [0] * n)
    sms = [[cross_fit_smoother(rng.standard_normal((2 * n, 6)),
                               _make_folds(2 * n, 5, np.random.default_rng(0)),
                               1.0)] for _ in range(L)]
    out = recentred_contrast_test(np.full(L, 0.99), sms, y, None, B=99, rng=rng)
    assert 0.0 < out["p_value"] <= 1.0
    assert out["p_value"] >= 1.0 / 100.0


# --------------------------------------------------------------------------
# End to end behaviour
# --------------------------------------------------------------------------

def test_null_is_not_wildly_miscalibrated():
    """A coarse guard, not a calibration study: over 40 replicates of a true
    null the rejection rate must not exceed 0.25."""
    rej = 0
    R = 40
    for r in range(R):
        layers, _, y, meta = SIMULATORS["A"](80, 0.0, 5000 + r,
                                             n_ref_extra=80, d=24, L=6)
        S = np.flatnonzero(y == 1)
        Rf = np.flatnonzero(y == 0)
        key = placebo_key(meta, noise=0.5, seed=r)
        o = np.argsort(key[Rf])
        idx = np.concatenate([S, Rf[o[:80]]])
        yo = np.concatenate([np.ones(80, int), np.zeros(80, int)])
        ba, sms = layer_profile(layers, idx, yo, CFG)
        plc = level_matched_placebo(layers, Rf, key, float(ba[0]), CFG)
        out = recentred_contrast_test(ba, sms, yo, plc["profile"], B=200,
                                      rng=np.random.default_rng(r))
        rej += out["p_value"] < 0.05
    assert rej / R <= 0.25, f"rejection rate {rej / R}"


def test_large_planted_effect_is_detected():
    layers, _, y, meta = SIMULATORS["A"](150, 4.0, 11, n_ref_extra=150,
                                         d=48, L=8)
    S = np.flatnonzero(y == 1)
    Rf = np.flatnonzero(y == 0)
    key = placebo_key(meta, noise=0.5, seed=11)
    o = np.argsort(key[Rf])
    idx = np.concatenate([S, Rf[o[:150]]])
    yo = np.concatenate([np.ones(150, int), np.zeros(150, int)])
    ba, sms = layer_profile(layers, idx, yo, CFG)
    plc = level_matched_placebo(layers, Rf, key, float(ba[0]), CFG)
    out = recentred_contrast_test(ba, sms, yo, plc["profile"], B=300,
                                  rng=np.random.default_rng(0))
    assert out["p_value"] < 0.05, out


# --------------------------------------------------------------------------
# The placebo
# --------------------------------------------------------------------------

def test_placebo_refuses_a_constant_surface_key():
    """A degenerate key must return a refusal, never a NaN that downstream
    code reads as a significant result."""
    layers = [np.random.default_rng(0).standard_normal((100, 8))
              for _ in range(4)]
    out = level_matched_placebo(layers, np.arange(100), np.ones(100), 0.6, CFG)
    assert out["profile"] is None
    assert "not computable" in out["status"]


def test_placebo_refuses_a_tiny_reference_set():
    layers = [np.random.default_rng(0).standard_normal((3, 8))
              for _ in range(4)]
    out = level_matched_placebo(layers, np.arange(3), np.arange(3.0), 0.6, CFG)
    assert out["profile"] is None


def test_placebo_matches_the_embedding_layer_level():
    layers, _, y, meta = SIMULATORS["B"](120, 0.0, 3, n_ref_extra=120,
                                         d=32, L=6)
    S, Rf = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    key = placebo_key(meta, noise=0.5, seed=3)
    o = np.argsort(key[Rf])
    idx = np.concatenate([S, Rf[o[:120]]])
    yo = np.concatenate([np.ones(120, int), np.zeros(120, int)])
    ba, _ = layer_profile(layers, idx, yo, CFG)
    plc = level_matched_placebo(layers, Rf, key, float(ba[0]), CFG)
    assert plc["status"] == "ok"
    assert abs(plc["ba0"] - ba[0]) < 0.10, (plc["ba0"], ba[0])


# --------------------------------------------------------------------------
# Simulators
# --------------------------------------------------------------------------

def test_simulators_are_structurally_different():
    """The claim that Sim-B is a different regime, made checkable."""
    stats = {}
    for name, fn in SIMULATORS.items():
        layers, _, _, _ = fn(200, 0.0, 0, d=48, L=8)
        rms = [np.linalg.norm(h, axis=1).mean() for h in layers]
        stats[name] = (rms[-1] / rms[0],
                       np.corrcoef(layers[1].ravel(), layers[-1].ravel())[0, 1])
    assert stats["A"][0] < 1.1 and stats["B"][0] > 1.3
    assert stats["A"][1] < 0.2 and stats["B"][1] > 0.4


def test_covariate_overlaps_between_sets():
    """The nuisance must be a covariate with overlapping distributions, not a
    label proxy. If it separated perfectly, the placebo would be meaningless."""
    _, _, y, meta = SIMULATORS["A"](300, 0.0, 0)
    u = meta["covariate"]
    lo, hi = u[y == 0], u[y == 1]
    assert hi.mean() > lo.mean()
    assert lo.max() > hi.min()          # genuine overlap


def test_reference_multiplier():
    _, _, y, _ = SIMULATORS["A"](100, 0.0, 0, n_ref_extra=100)
    assert (y == 1).sum() == 100 and (y == 0).sum() == 200


# --------------------------------------------------------------------------
# Exposure calibration
# --------------------------------------------------------------------------

def test_exposure_link_recovers_planted_parameters():
    m = np.array([0, 1, 2, 4, 8, 16, 32], dtype=float)
    T = 0.05 * (1 - np.exp(-0.2 * m))
    g, b = fit_exposure_link(m, T)
    assert abs(g - 0.05) < 0.01 and abs(b - 0.2) < 0.05


def test_exposure_inversion_brackets_the_truth():
    g, b, sigma = 0.05, 0.2, 0.002
    for m_true in (1.0, 4.0, 12.0):
        t = g * (1 - np.exp(-b * m_true))
        lo, hi = invert_exposure(t, g, b, sigma)
        assert lo <= m_true <= hi


def test_detection_floor_interpolates():
    assert abs(detection_floor([0, 1, 2, 4], [0.0, 0.4, 0.8, 1.0], 0.8) - 2) < 1e-9
    assert detection_floor([0, 1], [0.0, 0.1], 0.8) == float("inf")


# --------------------------------------------------------------------------
# Determinism and hygiene
# --------------------------------------------------------------------------

def test_pipeline_is_deterministic_under_a_fixed_seed():
    def once():
        layers, _, y, meta = SIMULATORS["A"](80, 1.0, 7, n_ref_extra=80,
                                             d=24, L=6)
        S, Rf = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        key = placebo_key(meta, noise=0.5, seed=7)
        o = np.argsort(key[Rf])
        idx = np.concatenate([S, Rf[o[:80]]])
        yo = np.concatenate([np.ones(80, int), np.zeros(80, int)])
        ba, sms = layer_profile(layers, idx, yo, CFG)
        plc = level_matched_placebo(layers, Rf, key, float(ba[0]), CFG)
        return recentred_contrast_test(ba, sms, yo, plc["profile"], B=100,
                                       rng=np.random.default_rng(0))
    a, b = once(), once()
    assert a["p_value"] == b["p_value"] and a["T_adj"] == b["T_adj"]


def test_surface_features_are_finite_on_awkward_input():
    X = surface_features(["", "a", "  ", "123 !!! ???", "x" * 500])
    assert np.isfinite(X).all()


def test_extraction_raises_a_clear_error_without_torch():
    pytest.importorskip  # noqa
    try:
        import torch  # noqa: F401
        pytest.skip("torch present; the missing-dependency path is not taken")
    except ImportError:
        with pytest.raises(RuntimeError, match="torch"):
            rscp.extract_prefix_activations("gpt2", ["hello"])
