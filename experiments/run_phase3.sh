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
VARIANT="${VARIANT:-contam-1.4b-dupcount-higher}"
CONTAM_DIR="${CONTAM_DIR:-test_set_contamination/detection_challenge_benchmarks}"
REF_MODEL="${REF_MODEL:-gpt2}"
SETS="${SETS:-piqa}"
# How much of each record the probe reads. The first Phase 3 run used 'goal',
# about 15 tokens, and returned null; that is too short to attribute the null
# to the instrument rather than to the prefix. 'full' is the whole record,
# which is what the model was trained on.
PREFIX="${PREFIX:-full}"
# 1000 injected items and 2084 withheld, so n=1000 uses the whole injected
# arm and still satisfies |R| = 2|S| exactly. No reason to leave power unused.
N="${N:-1000}"
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

# The 1.4B checkpoint is 5.8 GB in fp32 and the reference model adds ~0.5 GB.
# Without this check the download dies partway and transformers reports it as
# a missing model, which sends you looking in the wrong place entirely.
import os, shutil
cache = os.environ.get("HF_HUB_CACHE") or os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
os.makedirs(cache, exist_ok=True)
free = shutil.disk_usage(cache).free / 1e9
print(f"  model cache: {cache}  ({free:.1f} GB free)")
if free < 8.0:
    sys.exit(
        f"  only {free:.1f} GB free, and this run needs about 8.\n\n"
        "  Point the cache at a volume with room, ideally the same one this\n"
        "  repository is on:\n\n"
        "      export HF_HUB_CACHE=\"$(pwd)/cache/hf\"\n"
        "      make phase3\n\n"
        "  A failed download also leaves a partial blob behind. Remove it:\n"
        f"      rm -rf {cache}/models--yonatano--*")
print("  preflight ok")
PY

hr; echo "1. plumbing check (no model needed)"
python3 src/run_rscp_eval.py --dry-run --out "$OUT/_dryrun" >/dev/null \
  && echo "  ok" || { echo "  dry run FAILED; stopping"; exit 1; }

if [ "$SKIP_BUILD" = "0" ]; then
  hr; echo "2. build item sets"
  # Their injected sets ship as one zip per trained variant. Cloning is more
  # reliable than the raw URL and costs 25 MB once.
  if [ ! -d "$CONTAM_DIR" ]; then
    echo "  cloning Oren et al.'s release"
    git clone --depth 1 https://github.com/tatsu-lab/test_set_contamination \
      >/dev/null 2>&1 || echo "  (clone failed; will try the raw URL)"
  fi
  for s in $SETS; do
    # Remove first. The builder refuses when a source cannot support the
    # construction, and without this the audit below would silently proceed on
    # whatever a previous run left behind: different items, possibly a
    # different model, and a reference arm that is not withheld at all.
    rm -f "$DATA/contam_${s}_${PREFIX}_suspect.jsonl" \
          "$DATA/contam_${s}_${PREFIX}_reference.jsonl"
    python3 src/build_itemsets.py --set contam --contam-set "$s" \
      --contam-model "$VARIANT" --contam-prefix "$PREFIX" \
      ${CONTAM_DIR:+--contam-dir "$CONTAM_DIR"} \
      --n "$N" --out "$DATA" || echo "  ($s arm unavailable)"
  done
else
  hr; echo "2. reusing $DATA"
fi

hr; echo "3. audit  ($MODEL)"
tag="$(echo "$MODEL" | tr '/' '_')"
ran=0
for s in $SETS; do
  sus="$DATA/contam_${s}_${PREFIX}_suspect.jsonl"
  ref="$DATA/contam_${s}_${PREFIX}_reference.jsonl"
  [ -f "$sus" ] && [ -f "$ref" ] || { echo "  (skipping $s, no item sets)"; continue; }
  echo "-- $s  (prefix=$PREFIX)"
  if python3 src/run_rscp_eval.py --model "$MODEL" --reference-model "$REF_MODEL" \
       --suspect "$sus" --reference "$ref" --max-length "$MAXLEN" \
       --batch-size "$BATCH" --out "$OUT/${tag}__contam_${s}_${PREFIX}"; then
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
