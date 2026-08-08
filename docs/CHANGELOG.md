# What changed, and why

The protocol went through three substantive revisions, each forced by a
measurement rather than a preference. This file is the short version; the
paper's Section 6 is the long one.

## v1, excess separability (discarded)
Reported `max_l (BA_l - BA_nuisance)` against a label-permutation null.
**Failed:** the level of that statistic depends on the dimension of the
nuisance control set. Under a true null it rejected 0.03 of the time at p=96
and 0.99 at p=1200. Since the recommended nuisance family contains a
20,000-feature character n-gram block, the protocol as written would have
manufactured contamination on clean models.

## v2, depth-profile contrast (partially correct)
Replaced the level with a zero-sum contrast on the depth profile, which cancels
the nuisance term algebraically. **Failed differently:** it assumed surface
decodability is flat in depth. With decodability rising it rejected 0.72 of
true nulls; on a residual stream where noise accumulates the profile *declines*
and the test lost all power.

## v3, level-matched placebo baseline (current)
The placebo split already estimates the null depth profile, since both its
halves are non-members. Recentring on it replaces a global assumption with a
per-model measurement. Requires |R| = 2|S| so the baseline is measured at the
same sample size, and a search over split coarseness so it is measured at the
same embedding-layer separability.

## Inference
The item bootstrap was replaced by a label permutation on the cross-fitted
smoother. The bootstrap holds the fitted probe fixed and misses its variance.

## Simulators
An early version tied the nuisance to the membership label, which made
depth-dependent surface separability invisible to any within-reference split
and produced a misleading negative. Nuisance is now a continuous item covariate
whose distribution differs between the sets.

## Guards added after real runs
- `BA_nuisance` above 0.75 blocks a contamination verdict (Requirement E).
- Item sets below 250 per side are refused.
- Degenerate surface keys return a refusal, never a number.
