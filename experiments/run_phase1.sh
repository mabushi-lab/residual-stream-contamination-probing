#!/usr/bin/env bash
# Phase 1: the nuisance audit.
#
# About one GPU-hour. Needs no reference set of your own and produces a
# standalone result: how much of each published membership-inference split a
# classifier with no access to the model can already solve, and what the depth
# profile of surface separability looks like on real models. That second one is
# the assumption Phase 0c had to remove rather than defend, so nothing
# downstream is interpretable without it.
#
#   bash run_phase1.sh                    small models, quick
#   MODELS="EleutherAI/pythia-1.4b" bash run_phase1.sh
#   SKIP_BUILD=1 bash run_phase1.sh       reuse data/ from a previous run
#
# Individual steps are allowed to fail. A missing dataset should cost you that
# arm, not the run.
set -uo pipefail

MODELS="${MODELS:-EleutherAI/pythia-160m EleutherAI/pythia-410m}"
REF_MODEL="${REF_MODEL:-EleutherAI/pythia-70m}"
N="${N:-500}"
MAXLEN="${MAXLEN:-512}"
BATCH="${BATCH:-8}"
OUT="${OUT:-runs/phase1}"
SKIP_BUILD="${SKIP_BUILD:-0}"

hr() { printf '%s\n' "------------------------------------------------------------"; }

hr; echo "0. preflight"
python3 - <<'PY' || exit 1
import importlib.util, os, sys

missing = [m for m in ("numpy", "sklearn", "scipy")
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("  missing: " + ", ".join(missing)
             + "\n  fix: pip install -r requirements.txt")

for m in ("torch", "transformers", "datasets", "zstandard"):
    if importlib.util.find_spec(m) is None:
        sys.exit(f"  {m} is not installed."
                 "\n  fix: pip install torch transformers datasets zstandard")

# Import them for real. Presence on disk is not the same as importable, and
# the commonest failure here is a version clash that only shows up on import.
try:
    import torch, transformers, datasets           # noqa: F401
except ImportError as e:
    msg = str(e)
    fix = "  fix: pip install -U transformers"
    if "huggingface" in msg:
        fix = ("  fix, pick one:\n"
               "      pip install -U transformers\n"
               "      pip install 'huggingface-hub>=0.34,<1.0'")
    sys.exit(f"  a dependency is installed but will not import:\n\n"
             f"    {msg}\n\n{fix}")

b = os.path.join(os.path.dirname(torch.__file__), "bin", "torch_shm_manager")
if os.path.exists(b) and not os.access(b, os.X_OK):
    print(f"  note: {b} is not executable; the builders work around it, but "
          "chmod +x on that path is the real fix.")
print(f"  torch {torch.__version__}, transformers {transformers.__version__}, "
      f"datasets {datasets.__version__}")
print("  preflight ok")
PY

hr; echo "1. plumbing check (no model needed)"
python3 run_rscp_eval.py --dry-run --out "$OUT/_dryrun" >/dev/null && echo "  ok" \
  || { echo "  dry run FAILED; stopping"; exit 1; }

if [ "$SKIP_BUILD" = "0" ]; then
  hr; echo "2. build item sets"
  python3 build_itemsets.py --set wikimia --n "$N"                        || echo "  (wikimia arm unavailable)"
  python3 build_itemsets.py --set pile --subdomain "Wikipedia (en)" --n "$N" || echo "  (pile/wikipedia arm unavailable)"
  python3 build_itemsets.py --set pile --subdomain "Github" --n "$N"      || echo "  (pile/github arm unavailable)"
  python3 build_itemsets.py --set gsm8k_gsm1k --n "$N"                    || echo "  (GSM arm unavailable; GSM1k is access-controlled)"
else
  hr; echo "2. reusing data/"
fi

hr; echo "3. audit"
ran=0
for m in $MODELS; do
  tag="$(echo "$m" | tr '/' '_')"
  for pair in "wikimia:wikimia" "pile_wikipedia:pile_wikipedia" \
              "pile_github:pile_github" "gsm8k:gsm1k"; do
    sus="data/${pair%%:*}_suspect.jsonl"
    ref="data/${pair##*:}_reference.jsonl"
    [ -f "$sus" ] && [ -f "$ref" ] || continue
    echo "-- $m  ${pair%%:*}"
    if python3 run_rscp_eval.py --model "$m" --reference-model "$REF_MODEL" \
         --suspect "$sus" --reference "$ref" --max-length "$MAXLEN" \
         --batch-size "$BATCH" --out "$OUT/${tag}__${pair%%:*}"; then
      ran=$((ran+1))
    else
      echo "  FAILED: $m ${pair%%:*}"
    fi
  done
done

hr
if [ "$ran" -eq 0 ]; then
  echo "no audits completed. If every arm was skipped, no item sets were built:"
  echo "  python3 build_itemsets.py --list"
  exit 1
fi
echo "4. collect ($ran audits)"
python3 collect_phase1.py --dir "$OUT"
