#!/usr/bin/env bash
set -euo pipefail

# Re-fit the READ-KG linear scorer and validation selector for an arbitrary
# subset of the already-generated structured evidence views.  This script does
# not invoke the reader.  Each subset is trained and calibrated independently,
# then evaluated with the official dataset evaluator.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../code" && pwd)}"
REASON="$ROOT/reason"
EXP_ROOT="${EXP_ROOT:-$REASON/logs/budgetmatch_legacy_b8_v3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$EXP_ROOT/view_subset_ablation}"
FEATURE_PROFILE="${FEATURE_PROFILE:-full}"
DEVICE="${DEVICE:-cpu}"
EPOCHS="${EPOCHS:-50}"
DATASET="${1:-}"
SUBSET="${2:-}"

export ROG_LOCAL_DATA_ONLY="${ROG_LOCAL_DATA_ONLY:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

case "$DATASET" in
  cwq)
    RET_DIR="$ROOT/retrieve/cwq_entityref_gate_soft_May18-21:48:31"
    ;;
  webqsp)
    RET_DIR="$ROOT/retrieve/webqsp_entityref_gate_soft_May18-15:43:26"
    ;;
  *)
    echo "Usage: bash $0 {cwq|webqsp} {path|minimal|candidate|path_minimal|path_candidate|minimal_candidate|all}" >&2
    exit 2
    ;;
esac

case "$SUBSET" in
  path)              VIEWS="pathfirst" ;;
  minimal)           VIEWS="minimal" ;;
  candidate)         VIEWS="candidate" ;;
  path_minimal)      VIEWS="pathfirst minimal" ;;
  path_candidate)    VIEWS="pathfirst candidate" ;;
  minimal_candidate) VIEWS="minimal candidate" ;;
  all)               VIEWS="pathfirst minimal candidate" ;;
  *)
    echo "Usage: bash $0 {cwq|webqsp} {path|minimal|candidate|path_minimal|path_candidate|minimal_candidate|all}" >&2
    exit 2
    ;;
esac

TRAIN_RET="$RET_DIR/retrieval_result_train_top200.pth"
VAL_RET="$RET_DIR/retrieval_result_val_top200.pth"
TEST_RET="$RET_DIR/retrieval_result_top500.pth"
OUT="$OUTPUT_ROOT/$DATASET/$SUBSET"

mkdir -p "$OUT"
if [[ -s "$OUT/DONE" ]]; then
  echo "SKIP completed: dataset=$DATASET subset=$SUBSET"
  exit 0
fi

cd "$REASON"

build_view_cli() {
  local split="$1"
  local prediction_file="$EXP_ROOT/runs/$DATASET/$split/structured/predictions.jsonl"
  local view
  VIEW_CLI=()
  if [[ ! -s "$prediction_file" ]]; then
    echo "ERROR: missing structured predictions: $prediction_file" >&2
    exit 1
  fi
  for view in $VIEWS; do
    VIEW_CLI+=(--view "$view=$prediction_file:1.0#view=$view")
  done
}

build_view_cli train
python train_multiview_aggregator.py \
  --dataset "$DATASET" \
  --split train \
  "${VIEW_CLI[@]}" \
  --retrieval "$TRAIN_RET" \
  --save_data "$OUT/train_rows.pth" \
  --output "$OUT/scorer_seed42.pth" \
  --feature_profile "$FEATURE_PROFILE" \
  --val_ratio 0.2 \
  --epochs "$EPOCHS" \
  --lr 5e-3 \
  --aux_weight 0.4 \
  --seed 42 \
  --device "$DEVICE" \
  2>&1 | tee "$OUT/scorer_train.log"

build_view_cli val
python search_set_selector.py \
  --dataset "$DATASET" \
  --split val \
  "${VIEW_CLI[@]}" \
  --retrieval "$VAL_RET" \
  --model "$OUT/scorer_seed42.pth" \
  --feature_profile "$FEATURE_PROFILE" \
  --grid_csv "$OUT/validation_grid.csv" \
  --best_json "$OUT/selector.json" \
  --output_path "$OUT/validation_predictions.jsonl" \
  --device "$DEVICE" \
  2>&1 | tee "$OUT/selector_search.log"

build_view_cli test
python apply_set_selector.py \
  --dataset "$DATASET" \
  --split test \
  "${VIEW_CLI[@]}" \
  --retrieval "$TEST_RET" \
  --model "$OUT/scorer_seed42.pth" \
  --feature_profile "$FEATURE_PROFILE" \
  --selection_json "$OUT/selector.json" \
  --output_path "$OUT/test_predictions.jsonl" \
  --device "$DEVICE" \
  2>&1 | tee "$OUT/test_apply.log"

python evaluate_readkg_official.py \
  --dataset "$DATASET" \
  --predictions "$OUT/test_predictions.jsonl" \
  --scored_triples "$TEST_RET" \
  --alias_dir "$OUT/eval_alias" \
  --alias_tag "view_ablation_${SUBSET}" \
  --output_json "$OUT/official_eval.json" \
  2>&1 | tee "$OUT/official_eval.log"

date -Iseconds > "$OUT/DONE"
echo "VIEW_SUBSET_ABLATION_COMPLETE dataset=$DATASET subset=$SUBSET views=$VIEWS"
