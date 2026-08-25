#!/usr/bin/env bash
set -euo pipefail

# Ablation: replace the learned answer scorer with non-learned candidate
# ordering/scoring, then run the same set-selector search/apply/eval path.
#
# Usage:
#   bash run_readkg_base_selector_ablation.sh {cwq|webqsp} [order|vote|base] {preflight|val_selector|test_apply|eval_official|all}
# Backward compatible:
#   bash run_readkg_base_selector_ablation.sh {cwq|webqsp} {preflight|val_selector_base|test_apply_base|eval_official_base|all}

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../code" && pwd)}"
REASON="$ROOT/reason"
DATASET="${1:-}"
MODE="${SCORER_MODE:-base}"
ACTION="${2:-}"

if [[ "${2:-}" == "order" || "${2:-}" == "vote" || "${2:-}" == "base" ]]; then
  MODE="$2"
  ACTION="${3:-}"
fi

if [[ "$MODE" != "order" && "$MODE" != "vote" && "$MODE" != "base" ]]; then
  echo "Invalid mode: $MODE. Expected one of: order, vote, base." >&2
  exit 2
fi

action_suffix() {
  case "$1" in
    val_selector_base) echo "val_selector" ;;
    test_apply_base) echo "test_apply" ;;
    eval_official_base) echo "eval_official" ;;
    *) echo "$1" ;;
  esac
}

ACTION="$(action_suffix "$ACTION")"

case "$DATASET" in
  cwq)
    RET_DIR="${RET_DIR:-$ROOT/retrieve/cwq_entityref_gate_soft_May18-21:48:31}"
    VAL_RET="$RET_DIR/retrieval_result_val_top200.pth"
    TEST_RET="$RET_DIR/retrieval_result_top500.pth"
    VAL_PRED="logs/cwq_view_probe_val800_3view_real/predictions.jsonl"
    TEST_PRED="logs/cwq_view_probe_testfull_3view/predictions.jsonl"
    BASE_SELECTOR="${BASE_SELECTOR:-logs/cwq_val800_selector_family_${MODE}_best.json}"
    BASE_GRID="${BASE_GRID:-logs/cwq_val800_selector_family_${MODE}_grid.csv}"
    BASE_VAL_PRED="${BASE_VAL_PRED:-logs/cwq_val800_selector_family_${MODE}_best.jsonl}"
    BASE_FINAL="${BASE_FINAL:-logs/cwq_testfull_readkg_selector_family_${MODE}_set.jsonl}"
    BASE_EVAL_JSON="${BASE_EVAL_JSON:-logs/cwq_testfull_readkg_selector_family_${MODE}_official_eval.json}"
    LEARNED_FINAL="logs/cwq_testfull_readkg_selector_family_set.jsonl"
    ;;
  webqsp)
    RET_DIR="${RET_DIR:-$ROOT/retrieve/webqsp_entityref_gate_soft_May18-15:43:26}"
    VAL_RET="$RET_DIR/retrieval_result_val_top200.pth"
    TEST_RET="$RET_DIR/retrieval_result_top500.pth"
    VAL_PRED="logs/webqsp_view_probe_val_3view/predictions.jsonl"
    TEST_PRED="logs/webqsp_view_probe_testfull_3view/predictions.jsonl"
    BASE_SELECTOR="${BASE_SELECTOR:-logs/webqsp_val_selector_family_${MODE}_best.json}"
    BASE_GRID="${BASE_GRID:-logs/webqsp_val_selector_family_${MODE}_grid.csv}"
    BASE_VAL_PRED="${BASE_VAL_PRED:-logs/webqsp_val_selector_family_${MODE}_best.jsonl}"
    BASE_FINAL="${BASE_FINAL:-logs/webqsp_testfull_readkg_selector_family_${MODE}_set.jsonl}"
    BASE_EVAL_JSON="${BASE_EVAL_JSON:-logs/webqsp_testfull_readkg_selector_family_${MODE}_official_eval.json}"
    LEARNED_FINAL="logs/webqsp_testfull_readkg_selector_family_set.jsonl"
    ;;
  *)
    echo "Usage: bash run_readkg_base_selector_ablation.sh {cwq|webqsp} [order|vote|base] {preflight|val_selector|test_apply|eval_official|all}" >&2
    exit 2
    ;;
esac

cd "$REASON"
mkdir -p logs

check_file() {
  local path="$1"
  if [[ -s "$path" ]]; then
    echo "OK   $path"
  else
    echo "MISS $path"
  fi
}

run_val_selector() {
  python search_set_selector.py \
    --scorer_mode "$MODE" \
    --dataset "$DATASET" \
    --split val \
    --view "pathfirst=$VAL_PRED:1.0#view=pathfirst" \
    --view "minimal=$VAL_PRED:1.0#view=minimal" \
    --view "candidate=$VAL_PRED:1.0#view=candidate" \
    --retrieval "$VAL_RET" \
    --grid_csv "$BASE_GRID" \
    --best_json "$BASE_SELECTOR" \
    --output_path "$BASE_VAL_PRED" \
    --device cuda:0 \
    2>&1 | tee "logs/${DATASET}_val_selector_family_${MODE}_search.log"
}

run_test_apply() {
  python apply_set_selector.py \
    --scorer_mode "$MODE" \
    --dataset "$DATASET" \
    --split test \
    --view "pathfirst=$TEST_PRED:1.0#view=pathfirst" \
    --view "minimal=$TEST_PRED:1.0#view=minimal" \
    --view "candidate=$TEST_PRED:1.0#view=candidate" \
    --retrieval "$TEST_RET" \
    --selection_json "$BASE_SELECTOR" \
    --output_path "$BASE_FINAL" \
    --device cuda:0 \
    2>&1 | tee "logs/${DATASET}_testfull_readkg_selector_family_${MODE}_set.log"
}

run_eval_official() {
  python evaluate_readkg_official.py \
    --dataset "$DATASET" \
    --predictions "$BASE_FINAL" \
    --scored_triples "$TEST_RET" \
    --output_json "$BASE_EVAL_JSON" \
    2>&1 | tee "logs/${DATASET}_testfull_readkg_selector_family_${MODE}_official_eval.log"
}

case "$ACTION" in
  preflight)
    echo "=== selector ablation inputs: $DATASET mode=$MODE ==="
    check_file "$VAL_RET"
    check_file "$TEST_RET"
    check_file "$VAL_PRED"
    check_file "$TEST_PRED"
    echo
    echo "=== learned output for comparison ==="
    check_file "$LEARNED_FINAL"
    echo
    echo "=== existing $MODE ablation outputs ==="
    check_file "$BASE_SELECTOR"
    check_file "$BASE_VAL_PRED"
    check_file "$BASE_FINAL"
    check_file "$BASE_EVAL_JSON"
    ;;
  val_selector)
    run_val_selector
    ;;
  test_apply)
    run_test_apply
    ;;
  eval_official)
    run_eval_official
    ;;
  all)
    run_val_selector
    run_test_apply
    run_eval_official
    ;;
  *)
    echo "Usage: bash run_readkg_base_selector_ablation.sh {cwq|webqsp} [order|vote|base] {preflight|val_selector|test_apply|eval_official|all}" >&2
    exit 2
    ;;
esac
