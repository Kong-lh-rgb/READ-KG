#!/usr/bin/env bash
set -euo pipefail

# Official CWQ Stage-B reproduction path for the best three-view READ-KG result.
# Run from any directory after `conda activate reasoner`.
#
# Pipeline:
#   1. preflight
#   2. train3000_reader   # builds pathfirst,minimal,candidate reader outputs
#   3. train3000_scorer   # trains answer-level multi-view scorer
#   4. val_reader         # builds val800 pathfirst,minimal,candidate outputs
#   5. val_selector       # searches selector thresholds on val800
#   6. test_reader
#   7. test_apply
#   8. eval_official      # computes full/sub paper metrics if deps exist
#
# Efficient verification:
#   - If reader outputs, scorer, and selector already exist, run `verify_cached`.
#   - This reruns only the answer scorer + set selector and compares against the
#     cached best full-test output; it does not call the Reader.

if [[ "${OMP_NUM_THREADS:-}" =~ ^[0-9]+$ ]]; then
  export OMP_NUM_THREADS
else
  export OMP_NUM_THREADS=8
fi

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../code" && pwd)}"
REASON="$ROOT/reason"
: "${MODEL:?Set MODEL to the local reader-model path}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SEQ_LEN_TO_CAPTURE="${MAX_SEQ_LEN_TO_CAPTURE:-8192}"
MAX_TOKENS="${MAX_TOKENS:-4000}"
RET_DIR="${RET_DIR:-$ROOT/retrieve/cwq_entityref_gate_soft_May18-21:48:31}"

TRAIN_RET="$RET_DIR/retrieval_result_train_top200.pth"
VAL_RET="$RET_DIR/retrieval_result_val_top200.pth"
TEST_RET="$RET_DIR/retrieval_result_top500.pth"

TRAIN_OUT="${TRAIN_OUT:-cwq_view_probe_train3000_3view}"
VAL_OUT="${VAL_OUT:-cwq_view_probe_val800_3view_real}"
TEST_OUT="${TEST_OUT:-cwq_view_probe_testfull_3view}"

TRAIN_PRED="logs/$TRAIN_OUT/predictions.jsonl"
VAL_PRED="logs/$VAL_OUT/predictions.jsonl"
TEST_PRED="logs/$TEST_OUT/predictions.jsonl"

ANSWER_SCORER="${ANSWER_SCORER:-mscav_ckpt/cwq_train3000_learned_aggregator.pth}"
BEST_SELECTOR="${BEST_SELECTOR:-logs/cwq_val800_selector_family_best.json}"
VAL_SELECTOR_GRID="${VAL_SELECTOR_GRID:-logs/cwq_val800_selector_family_grid.csv}"
VAL_SELECTOR_PRED="${VAL_SELECTOR_PRED:-logs/cwq_val800_selector_family_best.jsonl}"
FINAL_PRED="${FINAL_PRED:-logs/cwq_testfull_readkg_selector_family_set.jsonl}"
VERIFY_PRED="${VERIFY_PRED:-logs/cwq_testfull_readkg_selector_family_set.verify.jsonl}"
SCORED_TRIPLES="${SCORED_TRIPLES:-$TEST_RET}"
OFFICIAL_EVAL_JSON="${OFFICIAL_EVAL_JSON:-logs/cwq_testfull_readkg_selector_family_official_eval.json}"

cd "$REASON"
mkdir -p logs mscav_data/cwq mscav_ckpt

check_file() {
  local path="$1"
  if [[ -s "$path" ]]; then
    echo "OK   $path"
  else
    echo "MISS $path"
  fi
}

case "${1:-}" in
  preflight)
    echo "=== official three-view READ-KG inputs ==="
    check_file "$MODEL"
    check_file "$TRAIN_RET"
    check_file "$VAL_RET"
    check_file "$TEST_RET"
    echo
    echo "=== existing three-view outputs ==="
    check_file "$TRAIN_PRED"
    check_file "$VAL_PRED"
    check_file "$TEST_PRED"
    check_file "$ANSWER_SCORER"
    check_file "$BEST_SELECTOR"
    check_file "$FINAL_PRED"
    echo
    echo "=== official metric retrieval/sample file ==="
    check_file "$SCORED_TRIPLES"
    ;;

  train3000_reader)
    python probe_evidence_views.py \
      -d cwq \
      --split train \
      -p "$TRAIN_RET" \
      --sample_size 3000 \
      --seed 42 \
      --views pathfirst,minimal,candidate \
      -m "$MODEL" \
      --llm_mode sys_icl_dc \
      --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
      --max_seq_len_to_capture "$MAX_SEQ_LEN_TO_CAPTURE" \
      --max_tokens "$MAX_TOKENS" \
      --frequency_penalty 0.16 \
      --batch_size "$BATCH_SIZE" \
      --output_name "$TRAIN_OUT" \
      2>&1 | tee "logs/${TRAIN_OUT}.log"
    ;;

  train3000_scorer)
    python train_multiview_aggregator.py \
      --dataset cwq \
      --split train \
      --view "pathfirst=$TRAIN_PRED:1.0#view=pathfirst" \
      --view "minimal=$TRAIN_PRED:1.0#view=minimal" \
      --view "candidate=$TRAIN_PRED:1.0#view=candidate" \
      --retrieval "$TRAIN_RET" \
      --save_data mscav_data/cwq/train3000_learned_rows.pth \
      --output "$ANSWER_SCORER" \
      --val_ratio 0.2 \
      --epochs 50 \
      --lr 5e-3 \
      --aux_weight 0.4 \
      --seed 42 \
      2>&1 | tee "logs/${TRAIN_OUT}_scorer_train.log"
    ;;

  val_reader)
    python probe_evidence_views.py \
      -d cwq \
      --split val \
      -p "$VAL_RET" \
      --sample_size 800 \
      --seed 42 \
      --views pathfirst,minimal,candidate \
      -m "$MODEL" \
      --llm_mode sys_icl_dc \
      --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
      --max_seq_len_to_capture "$MAX_SEQ_LEN_TO_CAPTURE" \
      --max_tokens "$MAX_TOKENS" \
      --frequency_penalty 0.16 \
      --batch_size "$BATCH_SIZE" \
      --output_name "$VAL_OUT" \
      2>&1 | tee "logs/${VAL_OUT}.log"
    ;;

  val_selector)
    python search_set_selector.py \
      --dataset cwq \
      --split val \
      --view "pathfirst=$VAL_PRED:1.0#view=pathfirst" \
      --view "minimal=$VAL_PRED:1.0#view=minimal" \
      --view "candidate=$VAL_PRED:1.0#view=candidate" \
      --retrieval "$VAL_RET" \
      --model "$ANSWER_SCORER" \
      --grid_csv "$VAL_SELECTOR_GRID" \
      --best_json "$BEST_SELECTOR" \
      --output_path "$VAL_SELECTOR_PRED" \
      --device cuda:0 \
      2>&1 | tee "logs/${VAL_OUT}_selector_search.log"
    ;;

  test_reader)
    python probe_evidence_views.py \
      -d cwq \
      --split test \
      -p "$TEST_RET" \
      --sample_size 0 \
      --seed 42 \
      --views pathfirst,minimal,candidate \
      -m "$MODEL" \
      --llm_mode sys_icl_dc \
      --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
      --max_seq_len_to_capture "$MAX_SEQ_LEN_TO_CAPTURE" \
      --max_tokens "$MAX_TOKENS" \
      --frequency_penalty 0.16 \
      --batch_size "$BATCH_SIZE" \
      --output_name "$TEST_OUT" \
      2>&1 | tee "logs/${TEST_OUT}.log"
    ;;

  test_apply)
    python apply_set_selector.py \
      --dataset cwq \
      --split test \
      --view "pathfirst=$TEST_PRED:1.0#view=pathfirst" \
      --view "minimal=$TEST_PRED:1.0#view=minimal" \
      --view "candidate=$TEST_PRED:1.0#view=candidate" \
      --retrieval "$TEST_RET" \
      --model "$ANSWER_SCORER" \
      --selection_json "$BEST_SELECTOR" \
      --output_path "$FINAL_PRED" \
      --device cuda:0 \
      2>&1 | tee "logs/${TEST_OUT}_selector_apply.log"
    ;;

  verify_cached)
    python apply_set_selector.py \
      --dataset cwq \
      --split test \
      --view "pathfirst=$TEST_PRED:1.0#view=pathfirst" \
      --view "minimal=$TEST_PRED:1.0#view=minimal" \
      --view "candidate=$TEST_PRED:1.0#view=candidate" \
      --retrieval "$TEST_RET" \
      --model "$ANSWER_SCORER" \
      --selection_json "$BEST_SELECTOR" \
      --output_path "$VERIFY_PRED" \
      --device cuda:0 \
      2>&1 | tee "logs/${TEST_OUT}_selector_apply.verify.log"

    if [[ -s "$FINAL_PRED" ]] && cmp -s "$FINAL_PRED" "$VERIFY_PRED"; then
      echo "EXACT_MATCH $VERIFY_PRED == $FINAL_PRED"
    elif [[ -s "$FINAL_PRED" ]]; then
      echo "DIFF $VERIFY_PRED != $FINAL_PRED" >&2
      exit 1
    else
      echo "MISS cached final output: $FINAL_PRED" >&2
      exit 1
    fi
    ;;

  eval_official)
    python evaluate_readkg_official.py \
      --dataset cwq \
      --predictions "$FINAL_PRED" \
      --scored_triples "$SCORED_TRIPLES" \
      --output_json "$OFFICIAL_EVAL_JSON" \
      2>&1 | tee "logs/${TEST_OUT}_official_eval.log"
    ;;

  *)
    echo "Usage: bash run_readkg_stage_b.sh {preflight|train3000_reader|train3000_scorer|val_reader|val_selector|test_reader|test_apply|verify_cached|eval_official}" >&2
    exit 2
    ;;
esac
