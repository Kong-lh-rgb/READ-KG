#!/usr/bin/env bash
set -euo pipefail

# Ablation: keep the learned candidate scorer, remove calibrated set selection,
# and output a fixed global top-k answer set for every question. The global k is
# selected on validation data from {1,2,3,5,8}, then applied unchanged to test.
#
# Usage:
#   bash run_readkg_scorer_only_topk_ablation.sh {cwq|webqsp} {preflight|val_topk|test_apply|eval_official|all}

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../code" && pwd)}"
REASON="$ROOT/reason"
DATASET="${1:-}"
ACTION="${2:-}"

case "$DATASET" in
  cwq)
    RET_DIR="${RET_DIR:-$ROOT/retrieve/cwq_entityref_gate_soft_May18-21:48:31}"
    VAL_RET="$RET_DIR/retrieval_result_val_top200.pth"
    TEST_RET="$RET_DIR/retrieval_result_top500.pth"
    VAL_PRED="logs/cwq_view_probe_val800_3view_real/predictions.jsonl"
    TEST_PRED="logs/cwq_view_probe_testfull_3view/predictions.jsonl"
    ANSWER_SCORER="${ANSWER_SCORER:-mscav_ckpt/cwq_train3000_learned_aggregator.pth}"
    TOPK_SELECTOR="${TOPK_SELECTOR:-logs/cwq_val800_scorer_only_topk_best.json}"
    TOPK_GRID="${TOPK_GRID:-logs/cwq_val800_scorer_only_topk_grid.csv}"
    TOPK_VAL_PRED="${TOPK_VAL_PRED:-logs/cwq_val800_scorer_only_topk_best.jsonl}"
    TOPK_FINAL="${TOPK_FINAL:-logs/cwq_testfull_readkg_scorer_only_topk_set.jsonl}"
    TOPK_EVAL_JSON="${TOPK_EVAL_JSON:-logs/cwq_testfull_readkg_scorer_only_topk_official_eval.json}"
    FULL_FINAL="logs/cwq_testfull_readkg_selector_family_set.jsonl"
    ;;
  webqsp)
    RET_DIR="${RET_DIR:-$ROOT/retrieve/webqsp_entityref_gate_soft_May18-15:43:26}"
    VAL_RET="$RET_DIR/retrieval_result_val_top200.pth"
    TEST_RET="$RET_DIR/retrieval_result_top500.pth"
    VAL_PRED="logs/webqsp_view_probe_val_3view/predictions.jsonl"
    TEST_PRED="logs/webqsp_view_probe_testfull_3view/predictions.jsonl"
    ANSWER_SCORER="${ANSWER_SCORER:-mscav_ckpt/webqsp_learned_aggregator.pth}"
    TOPK_SELECTOR="${TOPK_SELECTOR:-logs/webqsp_val_scorer_only_topk_best.json}"
    TOPK_GRID="${TOPK_GRID:-logs/webqsp_val_scorer_only_topk_grid.csv}"
    TOPK_VAL_PRED="${TOPK_VAL_PRED:-logs/webqsp_val_scorer_only_topk_best.jsonl}"
    TOPK_FINAL="${TOPK_FINAL:-logs/webqsp_testfull_readkg_scorer_only_topk_set.jsonl}"
    TOPK_EVAL_JSON="${TOPK_EVAL_JSON:-logs/webqsp_testfull_readkg_scorer_only_topk_official_eval.json}"
    FULL_FINAL="logs/webqsp_testfull_readkg_selector_family_set.jsonl"
    ;;
  *)
    echo "Usage: bash run_readkg_scorer_only_topk_ablation.sh {cwq|webqsp} {preflight|val_topk|test_apply|eval_official|all}" >&2
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

run_val_topk() {
  python search_set_selector.py \
    --scorer_mode learned \
    --dataset "$DATASET" \
    --split val \
    --view "pathfirst=$VAL_PRED:1.0#view=pathfirst" \
    --view "minimal=$VAL_PRED:1.0#view=minimal" \
    --view "candidate=$VAL_PRED:1.0#view=candidate" \
    --retrieval "$VAL_RET" \
    --model "$ANSWER_SCORER" \
    --min_scores -100 \
    --score_gaps 0 \
    --top1_min_scores -100 \
    --target_modes fixed \
    --max_answers_list 1,2,3,5,8 \
    --grid_csv "$TOPK_GRID" \
    --best_json "$TOPK_SELECTOR" \
    --output_path "$TOPK_VAL_PRED" \
    --device cuda:0 \
    2>&1 | tee "logs/${DATASET}_val_scorer_only_topk_search.log"
}

run_test_apply() {
  python apply_set_selector.py \
    --scorer_mode learned \
    --dataset "$DATASET" \
    --split test \
    --view "pathfirst=$TEST_PRED:1.0#view=pathfirst" \
    --view "minimal=$TEST_PRED:1.0#view=minimal" \
    --view "candidate=$TEST_PRED:1.0#view=candidate" \
    --retrieval "$TEST_RET" \
    --model "$ANSWER_SCORER" \
    --selection_json "$TOPK_SELECTOR" \
    --output_path "$TOPK_FINAL" \
    --device cuda:0 \
    2>&1 | tee "logs/${DATASET}_testfull_readkg_scorer_only_topk_set.log"
}

run_eval_official() {
  python evaluate_readkg_official.py \
    --dataset "$DATASET" \
    --predictions "$TOPK_FINAL" \
    --scored_triples "$TEST_RET" \
    --output_json "$TOPK_EVAL_JSON" \
    2>&1 | tee "logs/${DATASET}_testfull_readkg_scorer_only_topk_official_eval.log"
}

case "$ACTION" in
  preflight)
    echo "=== scorer-only top-k ablation inputs: $DATASET ==="
    check_file "$VAL_RET"
    check_file "$TEST_RET"
    check_file "$VAL_PRED"
    check_file "$TEST_PRED"
    check_file "$ANSWER_SCORER"
    echo
    echo "=== full scorer+selector output for comparison ==="
    check_file "$FULL_FINAL"
    echo
    echo "=== existing scorer-only top-k outputs ==="
    check_file "$TOPK_SELECTOR"
    check_file "$TOPK_VAL_PRED"
    check_file "$TOPK_FINAL"
    check_file "$TOPK_EVAL_JSON"
    ;;
  val_topk)
    run_val_topk
    ;;
  test_apply)
    run_test_apply
    ;;
  eval_official)
    run_eval_official
    ;;
  all)
    run_val_topk
    run_test_apply
    run_eval_official
    ;;
  *)
    echo "Usage: bash run_readkg_scorer_only_topk_ablation.sh {cwq|webqsp} {preflight|val_topk|test_apply|eval_official|all}" >&2
    exit 2
    ;;
esac
