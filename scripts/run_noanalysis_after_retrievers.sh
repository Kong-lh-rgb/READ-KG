#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../code" && pwd)}"
RETRIEVE="$ROOT/retrieve"
REASON="$ROOT/reason"
: "${MODEL:?Set MODEL to the local reader-model path}"
EXP_ROOT="${EXP_ROOT:-$REASON/logs/noanalysis_b8}"

CWQ_RET="$RETRIEVE/cwq_noanalysis_Aug08-08:34:11"
WEB_RET="$RETRIEVE/webqsp_noanalysis_Aug08-08:34:11"
CWQ_CPT="$CWQ_RET/cpt.pth"
WEB_CPT="$WEB_RET/cpt.pth"

export READKG_DISABLE_QUESTION_ANALYSIS=1
export ROG_LOCAL_DATA_ONLY=1
export OMP_NUM_THREADS=8
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HOME/.cache/huggingface/transformers}"
export HF_HUB_OFFLINE=1

mkdir -p "$EXP_ROOT/environment" "$EXP_ROOT/prompts" "$EXP_ROOT/runs"
date -Iseconds > "$EXP_ROOT/environment/orchestrator_start.txt"

while screen -ls 2>/dev/null | grep -qE 'noanalysis_ret_(cwq|webqsp)'; do
  date -Iseconds > "$EXP_ROOT/environment/waiting_for_retrievers.txt"
  sleep 60
done

if grep -q 'Traceback' "$RETRIEVE/logs/noanalysis_b8/cwq_train.log"; then
  echo "CWQ retriever training failed; see its log." >&2
  exit 1
fi
if grep -q 'Traceback' "$RETRIEVE/logs/noanalysis_b8/webqsp_train.log"; then
  echo "WebQSP retriever training failed; see its log." >&2
  exit 1
fi
test -s "$CWQ_CPT"
test -s "$WEB_CPT"

run_retrieval_exports() {
  local dataset="$1"
  local cpt="$2"
  local ret_dir="$3"
  local gpu="$4"
  export CUDA_VISIBLE_DEVICES="$gpu"
  cd "$RETRIEVE"
  python inference.py -p "$cpt" --split train --max_K 200 \
    --skip_no_path \
    --output_path "$ret_dir/retrieval_result_train_top200.pth" \
    2>&1 | tee "$EXP_ROOT/${dataset}_retrieval_train.log"
  python inference.py -p "$cpt" --split val --max_K 200 \
    --skip_no_path \
    --output_path "$ret_dir/retrieval_result_val_top200.pth" \
    2>&1 | tee "$EXP_ROOT/${dataset}_retrieval_val.log"
  if [[ "$dataset" == "webqsp" ]]; then
    python inference_fill_missing_noanalysis.py \
      --dataset "$dataset" --split train --checkpoint "$cpt" \
      --output_path "$ret_dir/retrieval_result_train_top200.pth" --max_k 200 \
      2>&1 | tee "$EXP_ROOT/${dataset}_retrieval_train_fill.log"
    python inference_fill_missing_noanalysis.py \
      --dataset "$dataset" --split val --checkpoint "$cpt" \
      --output_path "$ret_dir/retrieval_result_val_top200.pth" --max_k 200 \
      2>&1 | tee "$EXP_ROOT/${dataset}_retrieval_val_fill.log"
  fi
  python inference.py -p "$cpt" --split test --max_K 500 \
    --output_path "$ret_dir/retrieval_result_top500.pth" \
    2>&1 | tee "$EXP_ROOT/${dataset}_retrieval_test.log"
}

if [[ -s "$CWQ_RET/retrieval_result_train_top200.pth" \
   && -s "$CWQ_RET/retrieval_result_val_top200.pth" \
   && -s "$CWQ_RET/retrieval_result_top500.pth" \
   && -s "$WEB_RET/retrieval_result_train_top200.pth" \
   && -s "$WEB_RET/retrieval_result_val_top200.pth" \
   && -s "$WEB_RET/retrieval_result_top500.pth" ]]; then
  echo "Reusing completed no-analysis retrieval exports."
else
  run_retrieval_exports cwq "$CWQ_CPT" "$CWQ_RET" 0 &
  PID_CWQ_RET=$!
  run_retrieval_exports webqsp "$WEB_CPT" "$WEB_RET" 1 &
  PID_WEB_RET=$!
  wait "$PID_CWQ_RET"
  wait "$PID_WEB_RET"
fi

prepare_prompts() {
  local dataset="$1"
  local split="$2"
  local sample_size="$3"
  local ret_dir="$4"
  local retrieval
  case "$split" in
    train) retrieval="$ret_dir/retrieval_result_train_top200.pth" ;;
    val) retrieval="$ret_dir/retrieval_result_val_top200.pth" ;;
    test) retrieval="$ret_dir/retrieval_result_top500.pth" ;;
    *) return 2 ;;
  esac
  if [[ -s "$EXP_ROOT/prompts/$dataset/$split/manifest.json" \
     && -s "$EXP_ROOT/prompts/$dataset/$split/structured_prompts.jsonl" ]]; then
    echo "Reusing completed prompts for dataset=$dataset split=$split."
    return 0
  fi
  cd "$REASON"
  python prepare_budget_matched_prompts.py \
    --dataset "$dataset" \
    --split "$split" \
    --retrieval "$retrieval" \
    --output_dir "$EXP_ROOT/prompts/$dataset/$split" \
    --model_name "$MODEL" \
    --k 100 \
    --sample_size "$sample_size" \
    --sample_seed 42 \
    --random_seeds 101,102,103 \
    --llm_mode sys_icl_dc \
    2>&1 | tee "$EXP_ROOT/${dataset}_${split}_prompt_prep.log"
}

prepare_prompts cwq train 3000 "$CWQ_RET"
prepare_prompts cwq val 800 "$CWQ_RET"
prepare_prompts cwq test 0 "$CWQ_RET"
prepare_prompts webqsp train 0 "$WEB_RET"
prepare_prompts webqsp val 0 "$WEB_RET"
prepare_prompts webqsp test 0 "$WEB_RET"

python - "$EXP_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "prompts"
expected = {
    ("cwq", "train"): (3000, 9000),
    ("cwq", "val"): (800, 2400),
    ("cwq", "test"): (3531, 10593),
    ("webqsp", "train"): (2848, 8544),
    ("webqsp", "val"): (250, 750),
    ("webqsp", "test"): (1628, 4884),
}
for key, counts in expected.items():
    dataset, split = key
    manifest = json.loads((root / dataset / split / "manifest.json").read_text())
    actual = (
        manifest["question_count"],
        manifest["condition_prompt_counts"]["structured"],
    )
    if actual != counts:
        raise SystemExit(f"count mismatch {key}: expected={counts}, actual={actual}")
print("PROMPT_COUNTS_OK")
PY

run_pilot() {
  local dataset="$1"
  local gpu="$2"
  local out="$EXP_ROOT/pilot/$dataset"
  mkdir -p "$out"
  export CUDA_VISIBLE_DEVICES="$gpu"
  cd "$REASON"
  python run_budget_reader.py \
    --prompts "$EXP_ROOT/prompts/$dataset/test/structured_prompts.jsonl" \
    --output_path "$out/predictions.jsonl" \
    --summary_csv "$out/summary.csv" \
    --run_stats_json "$out/run_stats.json" \
    --model_name "$MODEL" \
    --llm_mode sys_icl_dc \
    --tensor_parallel_size 1 \
    --max_seq_len_to_capture 8192 \
    --max_tokens 4000 \
    --temperature 0 \
    --frequency_penalty 0.16 \
    --seed 42 \
    --batch_size 8 \
    --legacy_double_check \
    --max_prompts 50 \
    2>&1 | tee "$out/run.log"
  test "$(wc -l < "$out/predictions.jsonl")" -eq 50
}

run_pilot cwq 0 &
PID_CWQ_PILOT=$!
run_pilot webqsp 1 &
PID_WEB_PILOT=$!
wait "$PID_CWQ_PILOT"
wait "$PID_WEB_PILOT"

run_reader_split() {
  local dataset="$1"
  local split="$2"
  local gpu="$3"
  local prompt="$EXP_ROOT/prompts/$dataset/$split/structured_prompts.jsonl"
  local out="$EXP_ROOT/runs/$dataset/$split/structured"
  mkdir -p "$out"
  export CUDA_VISIBLE_DEVICES="$gpu"
  cd "$REASON"
  python run_budget_reader.py \
    --prompts "$prompt" \
    --output_path "$out/predictions.jsonl" \
    --summary_csv "$out/summary.csv" \
    --run_stats_json "$out/run_stats.json" \
    --model_name "$MODEL" \
    --llm_mode sys_icl_dc \
    --tensor_parallel_size 1 \
    --max_seq_len_to_capture 8192 \
    --max_tokens 4000 \
    --temperature 0 \
    --frequency_penalty 0.16 \
    --seed 42 \
    --batch_size 8 \
    --legacy_double_check \
    --resume \
    2>&1 | tee "$out/run.log"
  date -Iseconds > "$out/DONE"
}

gpu0_queue() {
  run_reader_split cwq test 0
  run_reader_split webqsp train 0
}

gpu1_queue() {
  run_reader_split cwq train 1
  run_reader_split cwq val 1
  run_reader_split webqsp test 1
  run_reader_split webqsp val 1
}

gpu0_queue &
PID_GPU0=$!
gpu1_queue &
PID_GPU1=$!
wait "$PID_GPU0"
wait "$PID_GPU1"

run_aggregation() {
  local dataset="$1"
  local gpu="$2"
  local ret_dir="$3"
  local train_action
  local train_out="noanalysis_b8/runs/$dataset/train/structured"
  local val_out="noanalysis_b8/runs/$dataset/val/structured"
  local test_out="noanalysis_b8/runs/$dataset/test/structured"
  local out_root="logs/noanalysis_b8/aggregation/$dataset"
  export CUDA_VISIBLE_DEVICES="$gpu"
  export READKG_DISABLE_QUESTION_ANALYSIS=1
  cd "$REASON"
  if [[ "$dataset" == "cwq" ]]; then
    train_action=train3000_scorer
    ROOT="$ROOT" RET_DIR="$ret_dir" TRAIN_OUT="$train_out" VAL_OUT="$val_out" TEST_OUT="$test_out" \
      ANSWER_SCORER="$out_root/scorer_seed42.pth" \
      BEST_SELECTOR="$out_root/selector.json" \
      VAL_SELECTOR_GRID="$out_root/validation_grid.csv" \
      VAL_SELECTOR_PRED="$out_root/validation_predictions.jsonl" \
      FINAL_PRED="$out_root/test_predictions.jsonl" \
      OFFICIAL_EVAL_JSON="$out_root/official_eval.json" \
      bash run_readkg_stage_b.sh "$train_action"
    ROOT="$ROOT" RET_DIR="$ret_dir" TRAIN_OUT="$train_out" VAL_OUT="$val_out" TEST_OUT="$test_out" \
      ANSWER_SCORER="$out_root/scorer_seed42.pth" BEST_SELECTOR="$out_root/selector.json" \
      VAL_SELECTOR_GRID="$out_root/validation_grid.csv" VAL_SELECTOR_PRED="$out_root/validation_predictions.jsonl" \
      FINAL_PRED="$out_root/test_predictions.jsonl" OFFICIAL_EVAL_JSON="$out_root/official_eval.json" \
      bash run_readkg_stage_b.sh val_selector
    ROOT="$ROOT" RET_DIR="$ret_dir" TRAIN_OUT="$train_out" VAL_OUT="$val_out" TEST_OUT="$test_out" \
      ANSWER_SCORER="$out_root/scorer_seed42.pth" BEST_SELECTOR="$out_root/selector.json" \
      FINAL_PRED="$out_root/test_predictions.jsonl" OFFICIAL_EVAL_JSON="$out_root/official_eval.json" \
      bash run_readkg_stage_b.sh test_apply
    ROOT="$ROOT" RET_DIR="$ret_dir" FINAL_PRED="$out_root/test_predictions.jsonl" \
      OFFICIAL_EVAL_JSON="$out_root/official_eval.json" \
      bash run_readkg_stage_b.sh eval_official
  else
    ROOT="$ROOT" RET_DIR="$ret_dir" TRAIN_OUT="$train_out" VAL_OUT="$val_out" TEST_OUT="$test_out" \
      ANSWER_SCORER="$out_root/scorer_seed42.pth" BEST_SELECTOR="$out_root/selector.json" \
      VAL_SELECTOR_GRID="$out_root/validation_grid.csv" VAL_SELECTOR_PRED="$out_root/validation_predictions.jsonl" \
      FINAL_PRED="$out_root/test_predictions.jsonl" OFFICIAL_EVAL_JSON="$out_root/official_eval.json" \
      bash run_readkg_webqsp_stage_b.sh train_scorer
    ROOT="$ROOT" RET_DIR="$ret_dir" TRAIN_OUT="$train_out" VAL_OUT="$val_out" TEST_OUT="$test_out" \
      ANSWER_SCORER="$out_root/scorer_seed42.pth" BEST_SELECTOR="$out_root/selector.json" \
      VAL_SELECTOR_GRID="$out_root/validation_grid.csv" VAL_SELECTOR_PRED="$out_root/validation_predictions.jsonl" \
      FINAL_PRED="$out_root/test_predictions.jsonl" OFFICIAL_EVAL_JSON="$out_root/official_eval.json" \
      bash run_readkg_webqsp_stage_b.sh val_selector
    ROOT="$ROOT" RET_DIR="$ret_dir" TRAIN_OUT="$train_out" VAL_OUT="$val_out" TEST_OUT="$test_out" \
      ANSWER_SCORER="$out_root/scorer_seed42.pth" BEST_SELECTOR="$out_root/selector.json" \
      FINAL_PRED="$out_root/test_predictions.jsonl" OFFICIAL_EVAL_JSON="$out_root/official_eval.json" \
      bash run_readkg_webqsp_stage_b.sh test_apply
    ROOT="$ROOT" RET_DIR="$ret_dir" FINAL_PRED="$out_root/test_predictions.jsonl" \
      OFFICIAL_EVAL_JSON="$out_root/official_eval.json" \
      bash run_readkg_webqsp_stage_b.sh eval_official
  fi
  date -Iseconds > "$EXP_ROOT/aggregation/$dataset/DONE"
}

run_aggregation cwq 0 "$CWQ_RET" &
PID_CWQ_AGG=$!
run_aggregation webqsp 1 "$WEB_RET" &
PID_WEB_AGG=$!
wait "$PID_CWQ_AGG"
wait "$PID_WEB_AGG"

date -Iseconds > "$EXP_ROOT/DONE"
echo "NO_ANALYSIS_ABLATION_COMPLETE root=$EXP_ROOT"
