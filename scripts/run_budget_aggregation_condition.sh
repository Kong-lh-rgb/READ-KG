#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../code" && pwd)}"
REASON="$ROOT/reason"
EXP_ROOT="${EXP_ROOT:-$REASON/logs/budgetmatch_legacy_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$EXP_ROOT/aggregation_full}"
DATASET="${1:-}"
CONDITION="${2:-}"
DEVICE="${DEVICE:-cpu}"
FEATURE_PROFILE="${FEATURE_PROFILE:-full}"
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
    echo "Usage: bash run_budget_aggregation_condition.sh {cwq|webqsp} {structured|random|self_consistency}" >&2
    exit 2
    ;;
esac

case "$CONDITION" in
  structured|random|self_consistency)
    ;;
  *)
    echo "Usage: bash run_budget_aggregation_condition.sh {cwq|webqsp} {structured|random|self_consistency}" >&2
    exit 2
    ;;
esac

TRAIN_RET="$RET_DIR/retrieval_result_train_top200.pth"
VAL_RET="$RET_DIR/retrieval_result_val_top200.pth"
TEST_RET="$RET_DIR/retrieval_result_top500.pth"
OUT="$OUTPUT_ROOT/$DATASET/$CONDITION"

mkdir -p "$OUT"
cd "$REASON"

view_args() {
  local split="$1"
  local base="$EXP_ROOT/runs/$DATASET/$split"
  case "$CONDITION" in
    structured)
      printf '%s\n' \
        "pathfirst=$base/structured/predictions.jsonl:1.0#view=pathfirst" \
        "minimal=$base/structured/predictions.jsonl:1.0#view=minimal" \
        "candidate=$base/structured/predictions.jsonl:1.0#view=candidate"
      ;;
    random)
      printf '%s\n' \
        "random_101=$base/random/predictions.jsonl:1.0#view=random_101" \
        "random_102=$base/random/predictions.jsonl:1.0#view=random_102" \
        "random_103=$base/random/predictions.jsonl:1.0#view=random_103"
      ;;
    self_consistency)
      printf '%s\n' \
        "sc101=$base/self_consistency_seed101/predictions.jsonl:1.0#view=flat" \
        "sc102=$base/self_consistency_seed102/predictions.jsonl:1.0#view=flat" \
        "sc103=$base/self_consistency_seed103/predictions.jsonl:1.0#view=flat"
      ;;
  esac
}

build_view_cli() {
  local split="$1"
  VIEW_CLI=()
  while IFS= read -r spec; do
    VIEW_CLI+=(--view "$spec")
  done < <(view_args "$split")
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
  --epochs 50 \
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
  --alias_tag "budgetmatch_${CONDITION}" \
  --output_json "$OUT/official_eval.json" \
  2>&1 | tee "$OUT/official_eval.log"

date -Iseconds > "$OUT/DONE"
echo "AGGREGATION_COMPLETE dataset=$DATASET condition=$CONDITION"
