# NLP Evaluation Methodology / RSCP
# Regenerate every number in the paper from scratch.
#
#   make test        property tests, ~2 s
#   make validate    the Phase 0c study, ~10 min on 4 cores
#   make figures     figures, tables and macros from the results files
#   make paper       thesis.pdf and abstract_of_thesis.pdf
#   make all         everything, in order
#   make verify      tests, then check artefacts against MANIFEST.sha256
#   make phase1      audit real models (needs torch + model weights)
#   make arxiv       build/arxiv.tar.gz, ready to upload as TeX source
#
# Only `phase1` needs a GPU or model weights. Everything else runs on a laptop.
# Run from this directory; scripts resolve paths through src/paths.py, so an
# absolute invocation from anywhere also works.

PY      ?= python3
WORKERS ?= 4
LATEX   ?= pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all test validate validate-legacy figures paper manifest verify \
        phase1 dryrun arxiv clean distclean tree

all: test validate figures paper manifest

test:
	$(PY) -m pytest -q tests/test_rscp.py

validate:
	$(PY) validation/phase0c_validation.py --stage all --workers $(WORKERS)

validate-legacy:
	$(PY) validation/phase0b_validation.py --stage all --workers $(WORKERS)

figures:
	$(PY) render/render_all.py
	$(PY) render/export_figures.py
	$(PY) render/export_tables.py
	@[ -n "$$(ls experiments/runs/phase1/*.json 2>/dev/null)" ] \
	  && $(PY) render/render_phase1.py \
	  || echo "  (no Phase 1 audits yet; skipping)"

paper:
	cd paper && $(LATEX) thesis.tex >/dev/null && $(LATEX) thesis.tex >/dev/null
	cd paper && $(LATEX) abstract_of_thesis.tex >/dev/null \
	         && $(LATEX) abstract_of_thesis.tex >/dev/null
	@echo "  paper/thesis.pdf and paper/abstract_of_thesis.pdf built"

manifest:
	$(PY) src/manifest.py

verify: test
	$(PY) src/manifest.py --check

phase1:
	bash experiments/run_phase1.sh

dryrun:
	$(PY) src/run_rscp_eval.py --dry-run --out experiments/runs/dryrun

arxiv:
	$(PY) src/make_arxiv.py

tree:
	@find . -type d -not -path '*/.*' -not -path './cache/*' | sort

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc

distclean: clean
	rm -f paper/thesis.pdf paper/abstract_of_thesis.pdf MANIFEST.sha256
	rm -rf figures cache
