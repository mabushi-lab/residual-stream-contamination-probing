"""
simulators.py
Two generative models for synthetic residual streams.

Every Phase 0 conclusion in the paper is a statement about a statistical
procedure, and a statement of that kind is only as good as the range of data
it was checked against. Sim-A is the deliberately simple model: isotropic
Gaussian layers, one nuisance direction, layers independent given the label.
Sim-B breaks every one of those assumptions in a direction real transformers
are known to go. If a conclusion holds under both, it is a property of the
procedure. If it holds only under Sim-A, it is a property of Sim-A.

Nuisance is a CONTINUOUS ITEM COVARIATE, not a label effect
-----------------------------------------------------------
Both simulators give every item a latent surface covariate u_i, and make the
suspect and reference sets differ in its *distribution* rather than tying it
to membership:

    u_i ~ N( mu (2 y_i - 1) / 2 , 1 )        overlapping, not separating

The activation carries kappa_l u_i v_nuis. This matters. If the nuisance were
tied to the label, depth-dependent nuisance would exist only along the
suspect/reference axis and no within-reference split could ever see it, so a
placebo baseline would be useless by construction. Real surface features are
properties of items, shared by both sets, differing in distribution. Modelling
them that way is what makes the placebo meaningful, and an earlier version of
these simulators got it wrong.

Sim-A
-----
    h^(l) = G^(l) + kappa_l u v_nuis + (s/2) eps a(l) v_mem

with G iid standard normal, v_nuis orthogonal to v_mem, a(0) = 0 and a(l) a
Gaussian bump peaking at 0.7 L. Layers are conditionally independent.

Sim-B
-----
A residual stream, not a stack of independent draws:

    h^(0) = e                                   (embedding)
    h^(l) = h^(l-1) + delta_l                   (accumulation)

with

  * anisotropic covariance, power-law eigenspectrum lambda_k ~ k^-alpha,
    matching the heavy-tailed spectra reported for real activations;
  * multivariate-t innovations (nu = 5), so the noise is heavy-tailed rather
    than Gaussian;
  * per-layer innovation scale rising with depth, so the residual-stream norm
    grows, as it does in trained transformers;
  * THREE correlated nuisance directions rather than one, written into the
    embedding and carried forward unchanged;
  * the memorisation direction written incrementally from 0.3 L, peaking at
    0.7 L, so it is computed rather than present at the embedding.

The consequences are not cosmetic. Layers become strongly correlated, which
changes the variance of any statistic aggregated over depth. And because the
nuisance signal is written once while noise accumulates, nuisance decodability
*declines* with depth in Sim-B, the opposite of the adversarial case in the
V2 experiment. Whether the contrast survives both directions is the question.
"""

from __future__ import annotations

import numpy as np

__all__ = ["layer_bump", "sim_a", "sim_b", "SIMULATORS", "placebo_key"]


def layer_bump(L: int, peak_frac: float = 0.7, width: float = 3.0) -> np.ndarray:
    """Planted memorisation profile: zero at the embedding, peaked in depth."""
    peak = round(peak_frac * L)
    a = np.exp(-0.5 * ((np.arange(L + 1) - peak) / width) ** 2)
    a[0] = 0.0
    return a / a.max()


def _labels(n_per_set: int, rng: np.random.Generator, n_ref_extra: int = 0):
    """Labels for n suspect and (n + n_ref_extra) reference items.

    The extra reference items come from the same latent structure, which is
    what makes a size-matched placebo baseline possible: with |R| = 2|S| the
    baseline contrast can be computed on n-per-side halves of R, exactly the
    sample size of the observed contrast.
    """
    N = 2 * n_per_set + n_ref_extra
    y = np.zeros(N, dtype=np.int64)
    y[:n_per_set] = 1
    y = rng.permutation(y)
    return y, (2.0 * y - 1.0)[:, None] / 2.0


def _covariate(y, mu, rng):
    """Latent surface covariate: overlapping distributions, not a label."""
    return mu * (2.0 * y - 1.0) / 2.0 + rng.standard_normal(y.size)


# --------------------------------------------------------------------------
# Sim-A: the simple model
# --------------------------------------------------------------------------

def sim_a(n_per_set, eps, seed, *, n_ref_extra=0, d=96, L=12, kappa=1.0, rho=1.0,
          p_nuis=96, nuis_slope=0.0, peak_frac=0.7, mu=1.0):
    rng = np.random.default_rng(seed)
    y, s = _labels(n_per_set, rng, n_ref_extra)
    N = y.size
    a = layer_bump(L, peak_frac)

    Q, _ = np.linalg.qr(rng.standard_normal((d, 2)))
    v_nuis, v_mem = Q[:, 0], Q[:, 1]

    cov = _covariate(y, mu, rng)[:, None]
    layers = []
    for l in range(L + 1):
        k_l = kappa * (1.0 + nuis_slope * l / L)
        layers.append(rng.standard_normal((N, d))
                      + cov * (k_l * v_nuis)[None, :]
                      + s * (eps * a[l] * v_mem)[None, :])

    un = rng.standard_normal(p_nuis)
    un /= np.linalg.norm(un)
    nuis = rng.standard_normal((N, p_nuis)) + cov * (kappa * rho * un)[None, :]
    return layers, nuis, y, {"profile": a, "covariate": cov[:, 0]}


# --------------------------------------------------------------------------
# Sim-B: a residual stream with realistic pathologies
# --------------------------------------------------------------------------

def _power_law_cov_factor(d, alpha, rng):
    """Cholesky-like factor of an anisotropic covariance with a power-law
    eigenspectrum. Returns A with A A' = Sigma, trace normalised to d."""
    lam = (np.arange(1, d + 1, dtype=float)) ** (-alpha)
    lam *= d / lam.sum()
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sqrt(lam)[None, :]


def _mvt(n, d, A, nu, rng):
    """Multivariate-t innovations with covariance factor A and nu dof."""
    z = rng.standard_normal((n, d)) @ A.T
    g = rng.chisquare(nu, size=(n, 1)) / nu
    return z / np.sqrt(g) * np.sqrt((nu - 2) / nu)


def sim_b(n_per_set, eps, seed, *, n_ref_extra=0, d=96, L=12, kappa=1.0, rho=1.0,
          p_nuis=96, nuis_slope=0.0, peak_frac=0.7, mu=1.0,
          alpha=1.0, nu=5.0, growth=0.6, n_nuis_dirs=3, write_start=0.3):
    """Residual-stream simulator. See the module docstring for the rationale.

    nuis_slope is retained for interface compatibility with sim_a and adds a
    depth-dependent *rewrite* of the nuisance direction, which is how a real
    model could make surface information more decodable with depth.
    """
    rng = np.random.default_rng(seed)
    y, s = _labels(n_per_set, rng, n_ref_extra)
    N = y.size
    a = layer_bump(L, peak_frac)

    A = _power_law_cov_factor(d, alpha, rng)

    # Nuisance subspace: several correlated directions, not one.
    Vn = rng.standard_normal((d, n_nuis_dirs))
    Vn += 0.6 * Vn[:, [0]]                      # deliberately correlated
    Vn /= np.linalg.norm(Vn, axis=0, keepdims=True)
    wn = rng.dirichlet(np.ones(n_nuis_dirs) * 3.0)
    v_nuis = Vn @ wn
    v_nuis /= np.linalg.norm(v_nuis)

    # Memorisation direction, orthogonalised against the nuisance subspace.
    v_mem = rng.standard_normal(d)
    v_mem -= Vn @ np.linalg.lstsq(Vn, v_mem, rcond=None)[0]
    v_mem /= np.linalg.norm(v_mem)

    # Embedding carries the nuisance covariate and nothing the model computed.
    cov = _covariate(y, mu, rng)[:, None]
    h = _mvt(N, d, A, nu, rng) + cov * (kappa * v_nuis)[None, :]
    layers = [h.copy()]

    start = write_start * L
    for l in range(1, L + 1):
        scale = 1.0 + growth * (l / L)
        delta = _mvt(N, d, A, nu, rng) * scale / np.sqrt(L)
        # memorisation is written incrementally once the model has depth to
        # compute it; the cumulative sum reproduces the planted bump
        inc = a[l] - a[l - 1] if l > 1 else a[1]
        if l >= start:
            delta = delta + s * (eps * max(inc, 0.0) * v_mem)[None, :]
        if nuis_slope:
            delta = delta + cov * (kappa * nuis_slope / L * v_nuis)[None, :]
        h = h + delta
        layers.append(h.copy())

    An = _power_law_cov_factor(p_nuis, alpha, rng)
    un = rng.standard_normal(p_nuis)
    un /= np.linalg.norm(un)
    nuis = _mvt(N, p_nuis, An, nu, rng) + cov * (kappa * rho * un)[None, :]
    return layers, nuis, y, {"profile": a, "covariate": cov[:, 0]}


def placebo_key(meta, noise=0.5, seed=0):
    """The analyst's surface variable for the placebo split.

    In a real audit this is prefix length, source subdomain, or similar. It is
    useful only insofar as it tracks the covariate that actually differs
    between the two sets. We model it as a noisy observation of u:
    ``noise`` = 0 is an oracle key, larger values a key that tracks the
    nuisance geometry only loosely. Sensitivity to this is measured, not
    assumed.
    """
    rng = np.random.default_rng(seed + 991)
    u = np.asarray(meta["covariate"], dtype=float)
    return u + noise * rng.standard_normal(u.size)


SIMULATORS = {"A": sim_a, "B": sim_b}


if __name__ == "__main__":  # pragma: no cover
    # Report the properties that distinguish the two simulators, so the claim
    # "Sim-B is structurally different" is checkable rather than asserted.
    for name, fn in SIMULATORS.items():
        layers, nuis, y, meta = fn(400, 0.0, 0)
        H = np.stack(layers)
        rms = np.linalg.norm(H, axis=2).mean(axis=1)
        cor = np.corrcoef(layers[1].ravel(), layers[-1].ravel())[0, 1]
        flat = np.concatenate([l.ravel() for l in layers])
        kurt = float(((flat - flat.mean()) ** 4).mean() / flat.var() ** 2)
        ev = np.linalg.svd(layers[-1] - layers[-1].mean(0),
                           compute_uv=False) ** 2
        print(f"Sim-{name}:  RMS layer0={rms[0]:.2f} layerL={rms[-1]:.2f} "
              f"(x{rms[-1]/rms[0]:.2f})   corr(h1,hL)={cor:+.3f}   "
              f"kurtosis={kurt:.2f}   top-eig share={ev[0]/ev.sum():.3f}")
