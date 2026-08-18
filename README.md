# Residual-Stream Contamination Probing

[![arXiv](https://img.shields.io/badge/arXiv-2608.12652-b31b1b.svg)](https://arxiv.org/abs/2608.12652)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21855509.svg)](https://doi.org/10.5281/zenodo.21855509)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Detecting benchmark contamination by probing a language model's internal
activations, with the controls that make the result mean something.

**Paper:** [arXiv:2608.12652](https://arxiv.org/abs/2608.12652), also built
from source as `paper/thesis.pdf` (23 pp) · **Extended abstract:**
`paper/abstract_of_thesis.pdf` (2 pp) · **Where things live:**
`PROJECT_STRUCTURE.md` · **How to run:** `docs/RUNNING.md`

The concept DOI badge above always resolves to the newest release. The paper's
own footnote cites the version DOI of the snapshot its numbers came from,
which is [10.5281/zenodo.21969647](https://doi.org/10.5281/zenodo.21969647).

---

## What this is

Benchmark scores are claims about capability. Contamination breaks the claim.
The obvious way to detect it with a probe on internal activations does not
work, and this project is mostly an account of *why not*, plus a protocol that
survives measurement.

Four design choices, each adopted because a simpler alternative was measured
and failed:

1. **Contrast the depth profile, not its level.** The level of a probe's
   advantage over a control depends on how large you made the control: under a
   true null the max-level statistic rejects 0.03 of the time against a
   96-dimensional control and 0.99 against a 1200-dimensional one. A zero-sum
   contrast cancels the control term algebraically.
2. **Recentre on a level-matched placebo baseline.** Contrasting against a flat
   depth profile fails in both directions: 0.72 false positives when surface
   decodability rises with depth, and no power at all when it falls. The
   placebo split already estimates the baseline, so use it. This is the main
   contribution.
3. **Permute labels, don't bootstrap items.** A bootstrap holds the fitted
   probe fixed and misses its variance.
4. **|R| = 2|S|.** The baseline must be measured at the same sample size as
   the comparison. A half-size baseline triples the error rate.

---

## Quick start

```bash
pip install -r requirements.txt
make test        # 29 property tests, ~2 s
make all         # validate, render, build the paper. ~10 min, no GPU
```

To audit a real model you also need `torch transformers datasets zstandard`:

```bash
make dryrun      # exercises the whole audit path with no model
make phase1      # build item sets, audit Pythia models, collect results
```

---

## What has been established

Baseline depth profiles on real transformers are **not flat**, spanning up to
29.1 accuracy points, and their non-flatness scales with how different the two
item sets are as text (correlation 0.87 over six audits). The correction is
therefore largest exactly where it is needed and negligible where it is not.
All four well-matched Pile arms return null at both model scales, which is what
the literature predicts for a corpus seen approximately once.

## What has not

Whether a transformer carries a linearly decodable familiarity signal at all.
The only significant contrast sits on WikiMIA, a temporal split where a
classifier with no access to the model already reaches 0.856, so the protocol
refuses a verdict rather than reporting one. Settling the question needs the
exposure calibration described in the paper's §9 and a benchmark with a
reference set that satisfies exchangeability.

---

## A note on the failure modes

Two bugs found during development are worth knowing about, because both would
have produced confident wrong answers rather than errors:

- A degenerate surface key made the placebo return a non-finite statistic,
  which was then read as a *significant* placebo and would have silently
  blocked legitimate audits.
- The exchangeability check computed its verdict correctly, printed
  "Requirement E fails", and then issued a contamination verdict anyway,
  because the gate was never wired into the verdict branch.

Both are covered by tests now. A contamination detector that fails closed
without saying so is worse than no detector.

---

## Citation

See `paper/thesis.tex` for the bibliography. Every reference is to a paper
verified to exist; every number in the paper is generated from a results file
rather than transcribed.
