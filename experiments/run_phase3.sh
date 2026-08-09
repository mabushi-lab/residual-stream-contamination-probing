#!/usr/bin/env bash
# Phase 3: the positive control.
#
# Every real-model result so far is null (Pile) or refused (WikiMIA). A
# detector that has only ever said "nothing here" and "I decline to answer"
# has not been shown to detect anything. This run is the first opportunity for
# the protocol to fire true, on checkpoints where membership is known.
#
# Oren et al. trained GPT-2 models on RedPajama Wikitext with whole benchmark
# test sets injected at known duplication counts. Suspect items are the exact
# files they injected; reference items come from the same benchmark's training
# split, which was not injected.
#
# Read this before interpreting anything: because they inject entire test sets
# rather than a random half of a pool, there is no exactly-exchangeable
# withheld arm. This is construction E3, not E1. Membership is exact, which is
# why the run is worth doing; exchangeability is approximate, and BA_nuisance
# is what tells you whether the comparison is admissible. If BA_nuisance clears
# the block threshold the protocol will refuse a verdict here too, and that is
# a real outcome rather than a failure of the script.
#
#   bash experiments/run_phase3.sh
#   SETS="piqa" bash experiments/run_phase3.sh
#   MODEL=yonatano/Contam-1.4b-dupcount-lower bash experiments/run_phase3.sh
#   SKIP_BUILD=1 bash experiments/run_phase3.sh
#
# Run from the repository root. About one GPU-hour for both arms.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# dupcount-higher carries piqa at 50x and mnli at 10x. If the protocol has any
# sensitivity at all, 50x duplication in a 1.4B model is where it shows.
MODEL="${MODEL:-yonatano/Contam-1.4b-dupcount-higher}"
REF_MODEL="${REF_MODEL:-gpt2}"
SETS="${SETS:-piqa mnli}"
N="${N:-500}"
MAXLEN="${MAXLEN:-512}"
BATCH="${BATCH:-4}"
OUT="${OUT:-experiments/runs/phase3}"
DATA="${DATA:-experiments/data}"
SKIP_BUILD="${SKIP_BUILD:-0}"

hr() { printf '%s\n' "------------------------------------------------------------"; }

hr; echo "0. preflight"
python3 - <<'PY' || exit 1
import importlib.util, sys
for m in ("numpy", "sklearn", "scipy"):
    if importlib.util.find_spec(m) is None:
        sys.exit(f"  missing {m}\n  fix: pip install -r requirements.txt")
for m in ("torch", "transformers", "datasets"):
    if importlib.util.find_spec(m) is None:
        sys.exit(f"  {m} is not installed."
                 "\n  fix: pip install torch transformers datasets")
try:
    import torch, transformers, datasets           # noqa: F401
except ImportError as e:
    fix = ("  fix, pick one:\n      pip install -U transformers\n"
           "      pip install 'huggingface-hub>=0.34,<1.0'"
           if "huggingface" in str(e) else "  fix: pip install -U transformers")
    sys.exit(f"  installed but will not import:\n\n    {e}\n\n{fix}")
print(f"  torch {torch.__version__}, transformers {transformers.__version__}")
print("  preflight ok")
PY

hr; echo "1. plumbing check (no model needed)"
python3 src/run_rscp_eval.py --dry-run --out "$OUT/_dryrun" >/dev/null \
  && echo "  ok" || { echo "  dry run FAILED; stopping"; exit 1; }

if [ "$SKIP_BUILD" = "0" ]; then
  hr; echo "2. build item sets"
  for s in $SETS; do
    python3 src/build_itemsets.py --set contam --contam-set "$s" \
      --n "$N" --out "$DATA" || echo "  ($s arm unavailable)"
  done
else
  hr; echo "2. reusing $DATA"
fi

hr; echo "3. audit  ($MODEL)"
tag="$(echo "$MODEL" | tr '/' '_')"
ran=0
for s in $SETS; do
  sus="$DATA/contam_${s}_suspect.jsonl"
  ref="$DATA/contam_${s}_reference.jsonl"
  [ -f "$sus" ] && [ -f "$ref" ] || { echo "  (skipping $s, no item sets)"; continue; }
  echo "-- $s"
  if python3 src/run_rscp_eval.py --model "$MODEL" --reference-model "$REF_MODEL" \
       --suspect "$sus" --reference "$ref" --max-length "$MAXLEN" \
       --batch-size "$BATCH" --out "$OUT/${tag}__contam_${s}"; then
    ran=$((ran+1))
  else
    echo "  FAILED: $s"
  fi
done

hr
if [ "$ran" -eq 0 ]; then
  echo "no audits completed. Check the builders first:"
  echo "  python3 src/build_itemsets.py --set contam --contam-set piqa --n 500"
  exit 1
fi
echo "4. done ($ran audits) -> $OUT"
echo
echo "How to read it:"
echo "  BA_nuisance first. Above the block threshold, no verdict is admissible"
echo "  and the E3 construction has failed, which is itself worth reporting."
echo "  If it is near 0.5 and T_adj is significant, the protocol has fired true"
echo "  on a model with known contamination. That is the positive control."
echo "  If it is near 0.5 and T_adj is null at 50x duplication, the honest"
echo "  reading is a sensitivity ceiling, not a bug. Report it as one."
