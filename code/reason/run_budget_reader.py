#!/usr/bin/env python3
"""Run one-pass reader generation with auditable call, token, and time counts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from apply_answer_verifier import eval_record
from llm_utils import (
    build_conversation,
    llm_init,
    run_chat_batch,
    should_double_check,
)
from main import get_defined_prompts


def load_jsonl(path: str) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def write_jsonl_atomic(path: str, rows: list[dict]) -> None:
    """Rewrite a checkpoint atomically after retroactive format recovery."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary_path = f"{path}.rewrite.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def has_ans_marker(text: str) -> bool:
    return "ans:" in str(text or "").lower()


def requires_retry(text: str, args: argparse.Namespace) -> bool:
    if args.legacy_double_check:
        return should_double_check(text)
    if args.conditional_retry_missing_ans:
        return not has_ans_marker(text)
    return False


def summarize(records: list[dict], views: list[str]) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["view"]].append(eval_record(record))
    rows = []
    for view in views:
        metrics = grouped.get(view, [])
        if not metrics:
            continue
        rows.append({
            "view": view,
            "count": len(metrics),
            "hit": sum(item["hit"] for item in metrics) / len(metrics),
            "f1": sum(item["f1"] for item in metrics) / len(metrics),
            "precision": sum(item["precision"] for item in metrics) / len(metrics),
            "recall": sum(item["recall"] for item in metrics) / len(metrics),
            "no_answer": sum(item["no_answer"] for item in metrics) / len(metrics),
        })
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gpu_info() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ]
    try:
        return subprocess.check_output(command, text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc!r}"


def token_summary(values: list[int]) -> dict:
    if not values:
        return {"total": 0, "mean": 0.0, "max": 0}
    return {
        "total": int(sum(values)),
        "mean": float(sum(values) / len(values)),
        "max": int(max(values)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run strict one-pass reader inference.")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--run_stats_json", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--llm_mode", default="sys_icl_dc")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_seq_len_to_capture", type=int, default=8192)
    parser.add_argument("--max_tokens", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--frequency_penalty", type=float, default=0.16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=0,
        help="Optional auditable prefix limit for protocol-equivalence pilots.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--conditional_retry_missing_ans",
        action="store_true",
        help=(
            "Allow exactly one format-recovery generation only when the first "
            "output lacks the literal ans: marker. Answer quality and "
            "'ans: not available' never trigger a retry."
        ),
    )
    parser.add_argument(
        "--legacy_double_check",
        action="store_true",
        help=(
            "Match the historical SubgraphRAG/READ-KG sys_icl_dc protocol: "
            "retry exactly once when the first output lacks ans:, says "
            "'ans: not available', or says 'ans: no information available'."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "cot" in args.llm_mode:
        raise ValueError(
            "The auditable reader runner does not support a CoT mode."
        )
    if args.legacy_double_check and "dc" not in args.llm_mode:
        raise ValueError("--legacy_double_check requires a dc llm_mode.")
    if args.conditional_retry_missing_ans and "dc" in args.llm_mode:
        raise ValueError(
            "Use --legacy_double_check with sys_icl_dc; the narrower "
            "missing-ans-only policy is only valid with sys_icl."
        )
    prompts = load_jsonl(args.prompts)
    if not prompts:
        raise ValueError(f"No prompts found: {args.prompts}")
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    keys = [(row["id"], row["view"]) for row in prompts]
    if len(keys) != len(set(keys)):
        raise ValueError("Prompt file contains duplicate (id, view) keys.")

    if os.path.exists(args.output_path) and os.path.getsize(args.output_path) > 0:
        if not args.resume:
            raise FileExistsError(
                f"Refusing to overwrite existing output: {args.output_path}"
            )
        completed_rows = load_jsonl(args.output_path)
    else:
        completed_rows = []
    completed = {(row["id"], row["view"]) for row in completed_rows}
    pending = [row for row in prompts if (row["id"], row["view"]) not in completed]
    views = list(dict.fromkeys(row["view"] for row in prompts))

    sys_prompt, cot_prompt = get_defined_prompts(
        "scored_100",
        args.model_name,
        args.llm_mode,
    )
    process_start = time.perf_counter()
    model_start = time.perf_counter()
    llm = llm_init(
        args.model_name,
        args.tensor_parallel_size,
        args.max_seq_len_to_capture,
        args.max_tokens,
        args.seed,
        args.temperature,
        args.frequency_penalty,
    )
    model_load_seconds = time.perf_counter() - model_start
    tokenizer = getattr(llm, "_tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("The auditable runner currently requires a local vLLM tokenizer.")

    inference_start = time.perf_counter()
    batch_size = max(1, args.batch_size)
    first_pass_calls_this_run = 0
    conditional_retry_calls_this_run = 0

    completed_by_key = {
        (row["id"], row["view"]): row for row in completed_rows
    }
    prompt_by_key = {
        (row["id"], row["view"]): row for row in prompts
    }

    # A strict one-pass checkpoint from an interrupted earlier run can be
    # upgraded without regenerating valid outputs. Each missing-marker record
    # receives at most one recovery call and is then marked permanently.
    recovery_keys = []
    if args.conditional_retry_missing_ans or args.legacy_double_check:
        recovery_keys = [
            key
            for key, row in completed_by_key.items()
            if key in prompt_by_key
            and requires_retry(row.get("prediction", ""), args)
            and not row.get("conditional_retry_attempted", False)
        ]
    for batch_start in range(0, len(recovery_keys), batch_size):
        batch_keys = recovery_keys[batch_start:batch_start + batch_size]
        recovery_conversations = []
        for key in batch_keys:
            prompt = prompt_by_key[key]
            llm_record = {
                "sys_query": sys_prompt,
                "user_query": prompt["user_query"],
                "cot_query": cot_prompt,
            }
            conversation = build_conversation(llm_record, args.llm_mode)
            recovery_conversations.append(
                list(conversation)
                + [{"role": "user", "content": cot_prompt}]
            )
        recovery_outputs = run_chat_batch(
            llm,
            recovery_conversations,
            args.model_name,
        )
        for key, recovery_output in zip(batch_keys, recovery_outputs):
            row = completed_by_key[key]
            row["first_prediction"] = row.get("prediction", "")
            row["prediction"] = recovery_output
            row["conditional_retry_attempted"] = True
            row["conditional_retry_reason"] = "missing_ans_marker"
        conditional_retry_calls_this_run += len(batch_keys)
    if recovery_keys:
        write_jsonl_atomic(args.output_path, completed_rows)

    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start:batch_start + batch_size]
        llm_records = [
            {
                "sys_query": sys_prompt,
                "user_query": prompt["user_query"],
                "cot_query": cot_prompt,
            }
            for prompt in batch
        ]
        conversations = [
            build_conversation(record, args.llm_mode)
            for record in llm_records
        ]
        first_outputs = run_chat_batch(
            llm,
            conversations,
            args.model_name,
        )
        if len(first_outputs) != len(batch):
            raise RuntimeError("Reader output count does not match prompt count.")
        first_pass_calls_this_run += len(batch)

        retry_outputs_by_index = {}
        if args.conditional_retry_missing_ans or args.legacy_double_check:
            retry_indices = [
                index
                for index, output in enumerate(first_outputs)
                if requires_retry(output, args)
            ]
            recovery_conversations = [
                list(conversations[index])
                + [{"role": "user", "content": llm_records[index]["cot_query"]}]
                for index in retry_indices
            ]
            retry_outputs = run_chat_batch(
                llm,
                recovery_conversations,
                args.model_name,
            )
            retry_outputs_by_index = dict(zip(retry_indices, retry_outputs))
            conditional_retry_calls_this_run += len(retry_indices)

        for index, (prompt, first_output) in enumerate(zip(batch, first_outputs)):
            retry_attempted = index in retry_outputs_by_index
            prediction = retry_outputs_by_index.get(index, first_output)
            output = {
                "id": prompt["id"],
                "view": prompt["view"],
                "condition": prompt.get("condition", ""),
                "question": prompt["question"],
                "ground_truth": prompt["ground_truth"],
                "prediction": prediction,
                "reader_seed": args.seed,
                "reader_temperature": args.temperature,
                "llm_mode": args.llm_mode,
                "conditional_retry_attempted": retry_attempted,
            }
            if retry_attempted:
                output["first_prediction"] = first_output
                output["conditional_retry_reason"] = (
                    "legacy_sys_icl_dc"
                    if args.legacy_double_check
                    else "missing_ans_marker"
                )
            append_jsonl(args.output_path, output)
            completed_rows.append(output)
    inference_seconds = time.perf_counter() - inference_start
    total_seconds = time.perf_counter() - process_start

    completed_by_key = {
        (row["id"], row["view"]): row for row in completed_rows
    }
    missing = [key for key in keys if key not in completed_by_key]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} outputs after inference.")

    first_input_tokens = []
    retry_input_tokens = []
    first_output_tokens = []
    retry_output_tokens = []
    ans_format = 0
    first_ans_format = 0
    total_retry_attempts = 0
    for prompt in prompts:
        record = {
            "sys_query": sys_prompt,
            "user_query": prompt["user_query"],
            "cot_query": cot_prompt,
        }
        conversation = build_conversation(record, args.llm_mode)
        token_ids = tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
        )
        first_input_tokens.append(len(token_ids))
        output_row = completed_by_key[(prompt["id"], prompt["view"])]
        output = output_row["prediction"]
        first_output = output_row.get("first_prediction", output)
        first_output_tokens.append(
            len(tokenizer.encode(first_output, add_special_tokens=False))
        )
        first_ans_format += int(has_ans_marker(first_output))
        if output_row.get("conditional_retry_attempted", False):
            total_retry_attempts += 1
            recovery_conversation = (
                list(conversation)
                + [{"role": "user", "content": cot_prompt}]
            )
            retry_token_ids = tokenizer.apply_chat_template(
                recovery_conversation,
                tokenize=True,
                add_generation_prompt=True,
            )
            retry_input_tokens.append(len(retry_token_ids))
            retry_output_tokens.append(
                len(tokenizer.encode(output, add_special_tokens=False))
            )
        ans_format += int("ans:" in str(output).lower())

    summary_rows = summarize(completed_rows, views)
    write_csv(args.summary_csv, summary_rows)
    stats = {
        "prompts": os.path.abspath(args.prompts),
        "output_path": os.path.abspath(args.output_path),
        "model_name": args.model_name,
        "llm_mode": args.llm_mode,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "batch_size": batch_size,
        "prompt_count": len(prompts),
        "completed_before_run": len(completed),
        "first_pass_calls_this_run": first_pass_calls_this_run,
        "conditional_retry_calls_this_run": conditional_retry_calls_this_run,
        "generation_calls_this_run": (
            first_pass_calls_this_run + conditional_retry_calls_this_run
        ),
        "generation_calls_for_current_outputs": (
            len(completed_rows) + total_retry_attempts
        ),
        "total_output_records": len(completed_rows),
        "strict_one_pass": not (
            args.conditional_retry_missing_ans or args.legacy_double_check
        ),
        "conditional_retry_missing_ans": args.conditional_retry_missing_ans,
        "legacy_double_check": args.legacy_double_check,
        "retry_policy": (
            "legacy_sys_icl_dc"
            if args.legacy_double_check
            else (
                "missing_ans_only"
                if args.conditional_retry_missing_ans
                else "none"
            )
        ),
        "max_generations_per_prompt": (
            2
            if args.conditional_retry_missing_ans or args.legacy_double_check
            else 1
        ),
        "conditional_retry_count": total_retry_attempts,
        "conditional_retry_rate": total_retry_attempts / len(prompts),
        "first_pass_ans_format_rate": first_ans_format / len(prompts),
        "ans_format_rate": ans_format / len(prompts),
        "first_pass_input_tokens": token_summary(first_input_tokens),
        "retry_input_tokens": token_summary(retry_input_tokens),
        "input_tokens": token_summary(first_input_tokens + retry_input_tokens),
        "first_pass_output_tokens": token_summary(first_output_tokens),
        "retry_output_tokens": token_summary(retry_output_tokens),
        "output_tokens": token_summary(first_output_tokens + retry_output_tokens),
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": total_seconds,
        "seconds_per_generation": (
            inference_seconds
            / (first_pass_calls_this_run + conditional_retry_calls_this_run)
            if (first_pass_calls_this_run + conditional_retry_calls_this_run)
            else 0.0
        ),
        "gpu": gpu_info(),
        "host": platform.node(),
        "python": platform.python_version(),
    }
    os.makedirs(os.path.dirname(args.run_stats_json) or ".", exist_ok=True)
    with open(args.run_stats_json, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    print(f"Saved predictions to: {args.output_path}")
    print(f"Saved summary to: {args.summary_csv}")
    print(f"Saved run stats to: {args.run_stats_json}")


if __name__ == "__main__":
    main()
