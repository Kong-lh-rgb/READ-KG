#!/usr/bin/env bash
set -euo pipefail

# Official WebQSP Stage-B reproduction path for the best three-view READ-KG result.
# Run from any directory after `conda activate reasoner`.
# WebQSP Stage-B pipeline:
#   1. preflight
#   2. make_trainval_retrieval   # only if train/val retrieval files are missing
#   3. train_reader
#   4. val_reader
#   5. train_scorer
#   6. val_selector
#   7. test_reader
#   8. test_apply
#   9. eval_official      # computes full/sub paper metrics if deps exist
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
RETRIEVE="$ROOT/retrieve"
: "${MODEL:?Set MODEL to the local reader-model path}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SEQ_LEN_TO_CAPTURE="${MAX_SEQ_LEN_TO_CAPTURE:-8192}"
MAX_TOKENS="${MAX_TOKENS:-4000}"
RET_DIR="${RET_DIR:-$ROOT/retrieve/webqsp_entityref_gate_soft_May18-15:43:26}"
CKPT="$RET_DIR/cpt.pth"

TRAIN_RET="$RET_DIR/retrieval_result_train_top200.pth"
VAL_RET="$RET_DIR/retrieval_result_val_top200.pth"
TEST_RET="$RET_DIR/retrieval_result_top500.pth"

TRAIN_OUT="${TRAIN_OUT:-webqsp_view_probe_train_3view}"
VAL_OUT="${VAL_OUT:-webqsp_view_probe_val_3view}"
TEST_OUT="${TEST_OUT:-webqsp_view_probe_testfull_3view}"

TRAIN_PRED="logs/$TRAIN_OUT/predictions.jsonl"
VAL_PRED="logs/$VAL_OUT/predictions.jsonl"
TEST_PRED="logs/$TEST_OUT/predictions.jsonl"

ANSWER_SCORER="${ANSWER_SCORER:-mscav_ckpt/webqsp_learned_aggregator.pth}"
BEST_SELECTOR="${BEST_SELECTOR:-logs/webqsp_val_selector_family_best.json}"
VAL_SELECTOR_GRID="${VAL_SELECTOR_GRID:-logs/webqsp_val_selector_family_grid.csv}"
VAL_SELECTOR_PRED="${VAL_SELECTOR_PRED:-logs/webqsp_val_selector_family_best.jsonl}"
FINAL_PRED="${FINAL_PRED:-logs/webqsp_testfull_readkg_selector_family_set.jsonl}"
VERIFY_PRED="${VERIFY_PRED:-logs/webqsp_testfull_readkg_selector_family_set.verify.jsonl}"
SCORED_TRIPLES="${SCORED_TRIPLES:-$TEST_RET}"
OFFICIAL_EVAL_JSON="${OFFICIAL_EVAL_JSON:-logs/webqsp_testfull_readkg_selector_family_official_eval.json}"

mkdir -p "$REASON/logs" "$REASON/mscav_data/webqsp" "$REASON/mscav_ckpt"

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
    echo "=== required retrieval/model files ==="
    check_file "$CKPT"
    check_file "$TRAIN_RET"
    check_file "$VAL_RET"
    check_file "$TEST_RET"
    echo
    echo "=== existing reader/scorer/selector outputs ==="
    cd "$REASON"
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

  make_trainval_retrieval)
    cd "$RETRIEVE"
    python inference.py \
      -p "$CKPT" \
      --split train \
      --max_K 200 \
      --output_path "$TRAIN_RET" \
      2>&1 | tee "$REASON/logs/webqsp_make_train_retrieval_top200.log"

    python inference.py \
      -p "$CKPT" \
      --split val \
      --max_K 200 \
      --output_path "$VAL_RET" \
      2>&1 | tee "$REASON/logs/webqsp_make_val_retrieval_top200.log"
    ;;

  train_reader)
    cd "$REASON"
    python probe_evidence_views.py \
      -d webqsp \
      --split train \
      -p "$TRAIN_RET" \
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
      --output_name "$TRAIN_OUT" \
      2>&1 | tee "logs/${TRAIN_OUT}.log"
    ;;

  val_reader)
    cd "$REASON"
    python probe_evidence_views.py \
      -d webqsp \
      --split val \
      -p "$VAL_RET" \
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
      --output_name "$VAL_OUT" \
      2>&1 | tee "logs/${VAL_OUT}.log"
    ;;

  train_scorer)
    cd "$REASON"
    python train_multiview_aggregator.py \
      --dataset webqsp \
      --split train \
      --view "pathfirst=$TRAIN_PRED:1.0#view=pathfirst" \
      --view "minimal=$TRAIN_PRED:1.0#view=minimal" \
      --view "candidate=$TRAIN_PRED:1.0#view=candidate" \
      --retrieval "$TRAIN_RET" \
      --save_data mscav_data/webqsp/train_learned_rows.pth \
      --output "$ANSWER_SCORER" \
      --val_ratio 0.2 \
      --epochs 50 \
      --lr 5e-3 \
      --aux_weight 0.4 \
      --seed 42 \
      2>&1 | tee logs/webqsp_learned_train.log
    ;;

  val_selector)
    cd "$REASON"
    python search_set_selector.py \
      --dataset webqsp \
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
      2>&1 | tee logs/webqsp_val_selector_family_search.log
    ;;

  test_reader)
    cd "$REASON"
    python probe_evidence_views.py \
      -d webqsp \
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
    cd "$REASON"
    python apply_set_selector.py \
      --dataset webqsp \
      --split test \
      --view "pathfirst=$TEST_PRED:1.0#view=pathfirst" \
      --view "minimal=$TEST_PRED:1.0#view=minimal" \
      --view "candidate=$TEST_PRED:1.0#view=candidate" \
      --retrieval "$TEST_RET" \
      --model "$ANSWER_SCORER" \
      --selection_json "$BEST_SELECTOR" \
      --output_path "$FINAL_PRED" \
      --device cuda:0 \
      2>&1 | tee logs/webqsp_testfull_readkg_selector_family_set.log
    ;;

  verify_cached)
    cd "$REASON"
    python apply_set_selector.py \
      --dataset webqsp \
      --split test \
      --view "pathfirst=$TEST_PRED:1.0#view=pathfirst" \
      --view "minimal=$TEST_PRED:1.0#view=minimal" \
      --view "candidate=$TEST_PRED:1.0#view=candidate" \
      --retrieval "$TEST_RET" \
      --model "$ANSWER_SCORER" \
      --selection_json "$BEST_SELECTOR" \
      --output_path "$VERIFY_PRED" \
      --device cuda:0 \
      2>&1 | tee logs/webqsp_testfull_readkg_selector_family_set.verify.log

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
    cd "$REASON"
    python evaluate_readkg_official.py \
      --dataset webqsp \
      --predictions "$FINAL_PRED" \
      --scored_triples "$SCORED_TRIPLES" \
      --output_json "$OFFICIAL_EVAL_JSON" \
      2>&1 | tee logs/webqsp_testfull_readkg_selector_family_official_eval.log
    ;;

  *)
    echo "Usage: bash run_readkg_webqsp_stage_b.sh {preflight|make_trainval_retrieval|train_reader|val_reader|train_scorer|val_selector|test_reader|test_apply|verify_cached|eval_official}" >&2
    exit 2
    ;;
esac
