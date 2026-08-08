# Project structure

*Where everything lives and why. If you come back to this in six months, read
this file and then `docs/RUNNING.md`.*

---

## The one-paragraph version

`src/` is the method. `validation/` is the synthetic study that shaped the
method. `experiments/` is the real-model study. `render/` turns results into
figures, tables and LaTeX macros. `paper/` consumes those. Nothing in `paper/`
is hand-typed from a result: every number flows from a JSON file through a
generated macro, so re-running an experiment and rebuilding updates the paper.
`make all` does the whole chain in about ten minutes on a laptop.

---

## Tree

```
NLP Evaluation Methodology/
├── README.md                     start here
├── PROJECT_STRUCTURE.md          this file
├── Makefile                      every workflow; `make all` runs the lot
├── requirements.txt              pinned versions the results were produced with
├── MANIFEST.sha256               hash of every artefact + the environment
│
├── src/                          THE METHOD
│   ├── paths.py                  single source of truth for locations
│   ├── rscp.py                   the protocol: probes, contrast, placebo, null
│   ├── simulators.py             Sim-A and Sim-B synthetic residual streams
│   ├── run_rscp_eval.py          audit one model against one pair of item sets
│   ├── build_itemsets.py         public datasets -> JSONL item files
│   ├── collect_phase1.py         audit JSONs -> a table
│   └── manifest.py               write/check the artefact hashes
│
├── validation/                   SYNTHETIC STUDY (no GPU, no model)
│   ├── phase0c_validation.py     CURRENT. The study reported in the paper
│   ├── phase0b_validation.py     superseded; kept for the capacity sweep
│   ├── phase0_validation.py      superseded; the first, bootstrap-based run
│   └── results/*.json            checkpointed outputs, one per stage
│
├── experiments/                  REAL-MODEL STUDY (needs torch + weights)
│   ├── run_phase1.sh             build item sets, audit, collect
│   ├── data/*.jsonl              item sets: <name>_suspect / _reference
│   ├── runs/phase1/*.json|.txt   one audit each, machine- and human-readable
│   └── phase1_summary.json       flattened summary of all audits
│
├── render/                       RESULTS -> PAPER ASSETS
│   ├── render_all.py             Phase 0 -> macros, tables, figures c1..c5
│   ├── render_phase1.py          Phase 1 -> macros, table, figures p1_*
│   ├── export_figures.py         standalone figure renders + RG thumbnail
│   └── export_tables.py          capability matrix, decision grid, schematic
│
├── paper/
│   ├── thesis.tex                the paper (21 pp)
│   ├── abstract_of_thesis.tex    2-page extended abstract
│   ├── thesis.pdf                built output
│   ├── abstract_of_thesis.pdf    built output
│   └── generated/*.tex           AUTO-GENERATED. Do not edit by hand
│
├── figures/                      all figures, .pdf + .png
├── tests/test_rscp.py            29 property tests of the method's claims
├── docs/RUNNING.md               how to run everything, and troubleshooting
└── cache/activations/            cached model activations. Regenerable, large
```

---

## What to open for a given question

| Question | File |
|---|---|
| What is the method? | `paper/thesis.pdf` §4, or `src/rscp.py` |
| How do I run it? | `docs/RUNNING.md` |
| Why is the statistic a *contrast* and not a level? | paper §6, C1 |
| Why recentre on a placebo? | paper §6, C2. This is the main contribution |
| What did real models show? | paper §7, and `experiments/runs/phase1/*.txt` |
| Is the implementation trustworthy? | `tests/test_rscp.py`, `make test` |
| Where did number X in the paper come from? | `paper/generated/*_macros.tex`, then the JSON it was rendered from |
| What changed and why? | paper §6 documents three of our own design choices being falsified |

---

## The data flow

```
validation/phase0c_validation.py ──> validation/results/phase0c_results.json ─┐
validation/phase0b_validation.py ──> validation/results/phase0b_results.json ─┤
                                                                              ├─> render/render_all.py ──> paper/generated/results_macros.tex
experiments/run_phase1.sh ──> experiments/runs/phase1/*.json ─────────────────┴─> render/render_phase1.py ──> paper/generated/phase1_macros.tex
                                                                                                                          │
                                                                                                        paper/thesis.tex ─┘ \input{}s them
```

The rule the project is built on: **no number is typed into the paper.** If a
value appears in `thesis.tex` as a literal decimal rather than a macro, it is
either a nominal constant like $0.05$ or a mistake.

---

## Naming conventions

- `*_suspect.jsonl` / `*_reference.jsonl`: the two item sets of an audit. The
  reference set is always **twice** the size of the suspect set; half is the
  comparison arm, half the baseline arm.
- `runs/<study>/<model>__<split>.json`: one audit. The `.txt` beside it is the
  same thing formatted for a human.
- `figures/<name>.png` is for slides and repositories and carries its own title.
  `figures/<name>_paper.pdf` is for the paper: no title, drawn at print size.
- `paper/generated/`: anything here is overwritten by `make figures`.
- Phase numbering follows the paper's programme: Phase 0 synthetic, Phase 1
  nuisance audit, Phase 2 exposure calibration (not yet run), Phases 3–5 in
  paper §9.

---

## Things that are safe to delete

- `cache/` holds cached activations, several GB. Deleting costs re-extraction time
  and nothing else.
- `paper/*.aux`, `*.log`, `*.out`, cleared by `make clean`.
- `figures/` and `paper/generated/`, regenerated by `make figures`.
- `__pycache__/`, `.pytest_cache/`.

## Things that are not

- `validation/results/*.json`. Re-running takes about ten minutes and the numbers
  will differ slightly (different seeds are not fixed across stages).
- `experiments/runs/phase1/*.json`. These are the real-model results, roughly
  25 minutes of laptop compute, and the paper's §7 is rendered from them.
- `experiments/data/*.jsonl`. The exact item sets the audits used. Rebuilding
  them from the Hub may not reproduce the same sample.

---

## State of the work, as of this snapshot

**Done.** The protocol, validated on synthetic data under two dissimilar
simulators. Phase 1 on real models: 6 audits, Pythia-160M and 410M, WikiMIA
and two Pile subdomains.

**Established.** Baseline depth profiles on real transformers are not flat
(up to 29.1 accuracy points), and their non-flatness scales with the surface
difference between item sets (correlation 0.87). All four well-matched Pile
arms return null, as the literature predicts. The protocol refuses a verdict
on the temporal split rather than reporting one.

**Not established.** Whether transformers carry a familiarity direction at
all. The only significant contrast is on WikiMIA, where a blind classifier
reaches 0.856 and Requirement E fails, so nothing is admissible from it.

**Next, in order of value.** Phase 2, the exposure calibration by randomised
injection, which converts the four nulls into a stated detection floor
(60–80 GPU-hours). Then any benchmark with a genuine matched twin; GSM1k's
access control is currently the binding constraint on saying anything about a
benchmark people actually use.
