# Running the evaluations

Two tiers. Tier 1 needs a laptop and about ten minutes. Tier 2 needs model
weights and, for one phase, a GPU.

---

## 0. Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                          # tier 1
pip install torch transformers datasets zstandard        # tier 2 as well
```

Everything below is run from the directory containing `rscp.py`.

---

## Tier 1: reproduce Phase 0 (no GPU, no model)

This is the run reported in Section 6 of the paper: the statistical machinery
on synthetic activations with a planted direction of known size. It validates
the procedure and says nothing about any language model.

```bash
make test        # 29 property tests, ~2 s
make validate    # the Phase 0c study, ~10 min on 4 cores
make figures     # every figure, table and macro
make paper       # thesis.pdf and abstract_of_thesis.pdf
```

or `make all` for the lot. `make verify` re-runs the tests and checks the
regenerated artefacts against `MANIFEST.sha256`.

The study is checkpointed per cell, so one cell can be re-run on its own:

```bash
python3 phase0c_validation.py --stage slopeB50   # Sim-B, 50% depth slope
python3 phase0c_validation.py --stage powerA_1   # Sim-A, mid effect sizes
python3 simulators.py                            # the Sim-A/Sim-B contrast
```

Every number in the validation section flows from `phase0c_results.json`
through `render_all.py` into `results_macros.tex`, so changing the run changes
the paper and no value is hand-copied.

---

## Tier 2: audit a real model

### Step 1, check the plumbing

```bash
python3 run_rscp_eval.py --dry-run --out runs/dryrun
```

Synthetic activations, no torch required. Confirms loading, the nuisance
family, the placebo gate, the contrast and the report all work. Do this
before spending GPU time.

### Step 2, prepare the two item sets

One JSON object per line:

```json
{"prefix": "Q: Natalia sold clips to 48 friends ... How many? A:", "answer": "72", "correct": true}
```

`prefix` is everything the model sees **before** it must answer. It must not
contain the answer. This is the one rule the protocol cannot check for you:
put the answer in the prefix and the detector becomes a competence probe, the
statistics will look fine, and the result will be meaningless.

`answer` and `correct` are optional and are used only for the inflation
estimate, never by the probe.

**The reference set is the hard part, not the code.** It has to be
exchangeable with the suspect set (Requirement E), and it has to be **twice
the size** of the suspect set: half is the comparison arm, half is the
baseline arm. A half-size baseline tripled the false positive rate in Phase 0c.
`build_itemsets.py` enforces both. Three constructions qualify:

| Construction | Example | Exchangeability |
|---|---|---|
| Randomised injection before training | Oren et al. contaminated checkpoints | exact |
| Commissioned twin | GSM1k against GSM8K | approximate, checkable |
| Within-corpus held-out split | Pile train vs. validation for Pythia | approximate, fails on drifting subdomains |

A temporal split (reference postdates the model cutoff) does **not** qualify,
and is what makes most published membership-inference numbers uninterpretable.

Or let the builders do it:

```bash
python3 build_itemsets.py --list
python3 build_itemsets.py --set gsm8k_gsm1k --n 500
```

### Step 3, run the audit

```bash
python3 run_rscp_eval.py \
    --model EleutherAI/pythia-410m \
    --suspect data/gsm8k.jsonl \
    --reference data/gsm1k.jsonl \
    --reference-model EleutherAI/pythia-160m \
    --out runs/pythia410m_gsm
```

Reads, in order: `BA_nuisance` (how much a blind classifier can do), the
level-matched baseline, then the recentred contrast. Writes a JSON and a
plain-text report.

Useful flags: `--pooling mean` for the robustness check, `--no-ngrams` to
drop the character n-gram block, `--bootstrap 5000` for tighter p-values,
`--placebo-min` for the minimum reference items per placebo side.

### How to read the output

1. **`BA_nuisance` first, and it can veto everything below it.** Near 0.5 means
   your two sets are well matched. Above 0.60 the report warns; above 0.75 it
   refuses to issue a contamination verdict at all, because Requirement E has
   failed and the baseline correction is being asked to absorb a first-order
   difference. WikiMIA sits near 0.86. `--acknowledge-non-exchangeable`
   overrides, and obliges you to say so in any writeup.
2. **The baseline second.** The report shows the observed and baseline depth
   profiles and how closely they match at layer 0. If no baseline could be
   built, no claim is admissible: supply a surface key with spread, or more
   reference items.
3. **The recentred contrast third.** `T_raw` minus `baseline` is `T_adj`, and
   the report shows all three so you can see how much work the baseline did.
   Significant means depth-dependent familiarity beyond surface separability.
   Null means no evidence *at this sensitivity*, which is why the detection
   floor matters.

---

## The programme, and what each phase costs

| Phase | What | Needs | Cost |
|---|---|---|---|
| 0 | Statistical validation, two simulators | laptop | ~10 min **(done)** |
| 1 | Nuisance audit of existing splits | forward passes | ~25 min on an M-series laptop **(done)** |
| 2 | Exposure calibration by injection | training | 60–80 GPU-hours |
| 3 | Public contaminated checkpoints | forward passes | ~1 GPU-hour |
| 4 | GSM8K vs. GSM1k application | forward passes | ~2 GPU-hours |
| 5 | Robustness | forward passes | ~4 GPU-hours |

Phases 1, 3, 4 and 5 need forward passes only and run on Apple silicon.
Phase 2 continues pretraining Pythia-160M and 410M at seven duplication
counts across four injection formats, and wants a discrete GPU; it is the
only phase `run_rscp_eval.py` does not cover, because it is a training loop
rather than an audit.

**Do Phase 1 first.** `bash run_phase1.sh` builds the item sets, audits three
Pythia scales and collects the output into `phase1_table.tex`. It is cheap, it
needs no reference set of your own, and it produces a standalone result: how
much of each published membership-inference split a classifier with no access
to the model can already solve. It also measures the depth profile of surface
separability on real models, which is the assumption Phase 0c had to remove
rather than defend, and nothing downstream is interpretable without it.

---

## Troubleshooting

**`RuntimeError: torch_shm_manager ... execl failed: Permission denied`**
(macOS). `datasets` calls torch's shared-memory allocator when it builds a
streaming dataset, and that helper binary ships without the execute bit in
some wheels. The builders now disable the code path, so this should not
recur, but the direct fix is:

```bash
chmod +x "$(python3 -c 'import torch,os; print(os.path.dirname(torch.__file__))')/bin/torch_shm_manager"
```

**`huggingface-hub>=0.34.0,<1.0 is required ... found huggingface-hub==1.x`.**
A version clash, not a missing package: your `huggingface_hub` is ahead of
what your `transformers` accepts. Either

```bash
pip install -U transformers                  # newer transformers accepts hub 1.x
pip install 'huggingface-hub>=0.34,<1.0'     # or pin the hub back
```

The preflight in `run_phase1.sh` now catches this before anything downloads.

**`Compression type zstd not supported`.** The Pile ships as `.jsonl.zst`
and `datasets` needs a codec: `pip install zstandard`. The preflight checks
for it now.

**Extraction looks hung after "reference-model likelihood features".** It was
not hung, it was silent. That pass needs the language-model head, so it is the
expensive half of an audit, and it now prints progress, caches its output, and
truncates at 384 tokens by default.

**`Bad split: validation. Available splits: ['train']`** (the Pile). The
uncopyrighted mirror exposes only `train` through its loader, with the
held-out partitions as separate files. The builder now names them explicitly
and tries several known layouts. If all of them fail, check the repo's file
list and add the correct path to `SPECS` in `build_itemsets.py`. Do not
substitute a train/train split: it would measure nothing.

**`Invalid HF URI 'hf://datasets/gsm8k@...'`.** Recent `huggingface_hub`
requires `namespace/name`, so the legacy bare `gsm8k` id no longer resolves.
The builder now tries `openai/gsm8k` first.

**`only NN items per side after balancing (need >= 250)`.** Working as
intended. A single WikiMIA length split has 250 rows in total, which leaves
about 55 per side once the 2:1 reference requirement is applied, and Phase 0c
measured calibration degrading well before that. The wikimia builder now pools
all four length splits. If a source genuinely has too few items, either use a
different one or pass `--allow-small` and label the result indicative.

**`GSM1k could not be loaded`.** GSM1k is released under access conditions.
Without it there is no exchangeable reference set for GSM8K, so the builder
refuses rather than substituting a temporal split, which would silently
reintroduce the exact confound this protocol exists to remove.

**Every arm skipped, no audits ran.** No item sets were built. Run
`python3 build_itemsets.py --list` and build one by hand to see the error.

**`MPS backend out of memory`** (Apple silicon). Three things changed to make
this unlikely: activations come from `AutoModel` rather than
`AutoModelForCausalLM`, so the language-model head never runs and gigabytes of
unused logits are never allocated; prefixes truncate at `--max-length 512` by
default; and an out-of-memory error halves the batch and retries instead of
dying. If it still happens, drop `--batch-size` to 2, or set
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` and accept the risk.

**Re-running is slow.** It should not be: activations are cached under
`activation_cache/`, keyed by model, pooling, truncation length and the exact
item list. Changing a statistic and re-running costs seconds. `--no-cache`
disables it.

---

## Status and honesty

- The statistical core is validated end to end (`phase0c_validation.py`, two
  simulators), covered by 29 property tests, and the dry run exercises the
  full audit path.
- The activation-extraction path (`rscp.extract_prefix_activations`,
  `reference_likelihood_features`) is written against the transformers API but
  has **not** been run against a real model here, because the environment had
  no GPU and no model access. Expect to fix something on first contact. Start
  with `pythia-70m` and 200 items.
- Phase 1 has been run: 6 audits over Pythia-160M and 410M against WikiMIA
  and two Pile subdomains, reported in Section 7. Activations are cached, so
  re-running is minutes rather than an hour.
- No result yet shows that a transformer carries a familiarity direction. The
  only significant contrast is on WikiMIA, where a blind classifier already
  reaches 0.86 and the protocol therefore refuses a verdict. Any writeup should
  say so as plainly as Section 7 does.
