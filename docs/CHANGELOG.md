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

## v1.2.1

Housekeeping after both preprints announced. The paper's page-1 footnote cited
zenodo.21855510, the v1.0.0 version DOI. That was correct when written and
stopped being correct when the reported numbers changed, twice. It now cites
21969647, the snapshot the current results actually came from. The README
badge keeps the concept DOI, which is what a badge should track, and now says
so explicitly. `CITATION.cff` identifiers and page count updated to match.

The companion paper is at arXiv:2608.14896.

## v1.1.0

Streaming smoothers, and Phase 3 re-run at full sample size.

`layer_profile` retained a dense N x N smoother per layer per seed, which is
17.6 GB at 49 layers and N = 3000 and capped Phase 3 at half the available
items. The permutation null consumes those smoothers one layer at a time
anyway, so `streaming_profile_and_null` accumulates the weighted contribution
and releases each layer immediately: 0.36 GB at the same settings.
`layer_profile` also takes `keep_smoothers=False`, which the placebo now uses.
A test asserts the streaming and retained paths give the same profile, null
and p-value, and the Phase 3 audit reproduced to every reported digit before
the sample size changed. The statistical stage also roughly halved, because
the uncorrected comparison no longer recomputes the entire permutation batch.

Phase 3 now runs at n = 1000, the whole injected arm, against 2000 withheld.
Both prefix arms are null: T_adj +0.0049 at p = 0.081 on the question-only
prefix, +0.0035 at p = 0.210 on the full record. The layer-0 match improves to
0.001 on the full-record arm.

Two corrections came out of matching the sample sizes. The claim that the
baseline profile changes shape with prefix length did not survive: at n = 500
the short prefix looked falling and the full record humped, but at matched
n = 1000 both peak in the first few layers. What scales with prefix length is
the span, 1.4 against 3.0 accuracy points. And the placebo limitation is
broader than first recorded: it cannot match below chance, but it also
undershot an observed 0.515 by reaching only 0.497, so the general statement
is that the surface key bounds the attainable range in both directions. That
is now L11 in the paper.

## v1.0.1

Preprint live at arXiv:2608.12652. Adds Phase 3, the positive control, and
its renderer; pins upstream dataset revisions in the item-set builders; tracks
the item sets in the manifest, taking it from 62 artefacts to 80; adds
`make arxiv` to assemble a submission tarball with the graphics path rewritten
and a verification compile; and fixes the pre-reorganisation paths that had
left `make phase1` broken.

## Phase 3, the positive control

Run against Oren et al.'s deliberately contaminated 1.4B checkpoint, PIQA
injected at 50x. Two findings and one correction.

The construction is E1, not E3 as first assumed. They inject a random subset
of the public test file and ship the whole file, so the withheld arm comes
back by difference and both arms are one pool split before training. All 1000
injected items are present in the public pool, which the builder now verifies
rather than assumes.

The contrast is null under both prefix lengths: BA_nuisance 0.4955 with the
question alone, 0.5174 with the full record, T_adj negative in both, p = 0.38
and 0.77. Running it twice was necessary because a null on a ten-word prefix
says more about the prefix than about the instrument.

Two limitations surfaced. The placebo cannot match below chance, since a
surface-key split always produces some separability, so the layer-0 match is
censored from below when the observed profile dips under 0.5. And
`layer_profile` retains a dense N x N smoother per layer per seed, which is
17.6 GB at 49 layers and N = 3000; this is why the first run was killed.

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
