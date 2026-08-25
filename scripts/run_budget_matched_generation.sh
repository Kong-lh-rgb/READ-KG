#!/usr/bin/env bash
set -euo pipefail

# Reader generation aligned to the historical SubgraphRAG/READ-KG protocol.
# Prompts must already exist under logs/budgetmatch_v1/prompts.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../code" && pwd)}"
REASON="$ROOT/reason"
: "${MODEL:?Set MODEL to the local reader-model path}"
EXP_ROOT="${EXP_ROOT:-$REASON/logs/budgetmatch_legacy_v2}"
PROMPT_ROOT="${PROMPT_ROOT:-$EXP_ROOT/prompts}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-4000}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
LLM_MODE="${LLM_MODE:-sys_icl_dc}"
FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-0.16}"
DATASETS="${DATASETS:-cwq webqsp}"
CONDITIONS="${CONDITIONS:-structured random self_consistency_seed101 self_consistency_seed102 self_consistency_seed103 flat}"
LEGACY_DOUBLE_CHECK="${LEGACY_DOUBLE_CHECK:-1}"

if [[ "$LLM_MODE" != "sys_icl_dc" ]]; then
  echo "ERROR: legacy-aligned generation requires sys_icl_dc." >&2
  exit 2
fi

cd "$REASON"
mkdir -p "$EXP_ROOT/environment" "$EXP_ROOT/runs"

date -Iseconds > "$EXP_ROOT/environment/start_time.txt"
hostname > "$EXP_ROOT/environment/hostname.txt"
nvidia-smi > "$EXP_ROOT/environment/nvidia_smi.txt"
python - <<'PY' > "$EXP_ROOT/environment/python_packages.txt"
import platform
import torch
import transformers
import vllm
print("python", platform.python_version())
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
PY
git rev-parse HEAD > "$EXP_ROOT/environment/git_commit.txt"
git status --short > "$EXP_ROOT/environment/git_status.txt"
git diff > "$EXP_ROOT/environment/git_diff.patch"

run_one() {
  local dataset="$1"
  local split="$2"
  local condition="$3"
  local prompt_file="$4"
  local temperature="$5"
  local seed="$6"
  local run_dir="$EXP_ROOT/runs/$dataset/$split/$condition"

  mkdir -p "$run_dir"
  if [[ -s "$run_dir/DONE" ]]; then
    echo "SKIP completed: $dataset $split $condition"
    return
  fi

  local retry_arg=()
  if [[ "$LEGACY_DOUBLE_CHECK" == "1" ]]; then
    retry_arg=(--legacy_double_check)
  fi

  python run_budget_reader.py \
    --prompts "$prompt_file" \
    --output_path "$run_dir/predictions.jsonl" \
    --summary_csv "$run_dir/summary.csv" \
    --run_stats_json "$run_dir/run_stats.json" \
    --model_name "$MODEL" \
    --llm_mode "$LLM_MODE" \
    --tensor_parallel_size 1 \
    --max_seq_len_to_capture "$MAX_SEQ_LEN" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$temperature" \
    --frequency_penalty "$FREQUENCY_PENALTY" \
    --seed "$seed" \
    --batch_size "$BATCH_SIZE" \
    "${retry_arg[@]}" \
    ${RESUME:+--resume} \
    2>&1 | tee "$run_dir/run.log"

  date -Iseconds > "$run_dir/DONE"
}

run_dataset() {
  local dataset="$1"
  local split

  # Produce the full READ-KG outputs first.
  if condition_enabled structured; then
    for split in train val test; do
      run_one \
        "$dataset" "$split" structured \
        "$PROMPT_ROOT/$dataset/$split/structured_prompts.jsonl" \
        0 42
    done
  fi

  # Fixed random permutations of the identical flat top-k evidence set.
  if condition_enabled random; then
    for split in train val test; do
      run_one \
        "$dataset" "$split" random \
        "$PROMPT_ROOT/$dataset/$split/random_prompts.jsonl" \
        0 42
    done
  fi

  # Self-consistency: the exact same flat prompt with three fixed sampling seeds.
  local seed
  for seed in 101 102 103; do
    if condition_enabled "self_consistency_seed${seed}"; then
      for split in train val test; do
        run_one \
          "$dataset" "$split" "self_consistency_seed${seed}" \
          "$PROMPT_ROOT/$dataset/$split/flat_prompts.jsonl" \
          0.7 "$seed"
      done
    fi
  done

  # One-call deterministic flat baseline.
  if condition_enabled flat; then
    for split in train val test; do
      run_one \
        "$dataset" "$split" flat \
        "$PROMPT_ROOT/$dataset/$split/flat_prompts.jsonl" \
        0 42
    done
  fi
}

condition_enabled() {
  local requested="$1"
  local condition
  for condition in $CONDITIONS; do
    if [[ "$condition" == "$requested" ]]; then
      return 0
    fi
  done
  return 1
}

# A second worker can set DATASETS=webqsp while the first worker runs CWQ.
# Dataset names are validated here to avoid silently launching the wrong job.
for dataset in $DATASETS; do
  case "$dataset" in
    cwq|webqsp)
      run_dataset "$dataset"
      ;;
    *)
      echo "ERROR: unsupported dataset partition: $dataset" >&2
      exit 2
      ;;
  esac
done

date -Iseconds > "$EXP_ROOT/environment/end_time.txt"
echo "GENERATION_RUNS_COMPLETE datasets=$DATASETS conditions=$CONDITIONS"
