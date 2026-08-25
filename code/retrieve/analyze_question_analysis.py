import json
import os

import pandas as pd

from argparse import ArgumentParser
from collections import Counter


def load_jsonl(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def main(args):
    records = load_jsonl(args.path)
    errors = load_jsonl(args.error_path) if args.error_path else []

    total_outputs = len(records)
    total_errors = len(errors)
    total_seen = total_outputs + total_errors

    relation_counts = [len(record.get("relation_intents", [])) for record in records]
    step_counts = [len(record.get("reasoning_steps", [])) for record in records]
    constraint_counts = [len(record.get("constraints", [])) for record in records]
    hop_counts = [record.get("hop_count_estimate", 0) or 0 for record in records]
    unknown_answer = [
        record.get("answer_type", {}).get("coarse", "unknown") == "unknown"
        for record in records
    ]
    multi_hop = [bool(record.get("requires_multi_hop", False)) for record in records]

    summary = {
        "path": args.path,
        "total_seen": total_seen,
        "valid_outputs": total_outputs,
        "errors": total_errors,
        "success_rate": safe_div(total_outputs, total_seen),
        "empty_relation_rate": safe_mean([count == 0 for count in relation_counts]),
        "empty_step_rate": safe_mean([count == 0 for count in step_counts]),
        "unknown_answer_type_rate": safe_mean(unknown_answer),
        "multi_hop_rate": safe_mean(multi_hop),
        "avg_relation_intents": safe_mean(relation_counts),
        "avg_reasoning_steps": safe_mean(step_counts),
        "avg_constraints": safe_mean(constraint_counts),
        "avg_hop_count": safe_mean(hop_counts),
        "long_chain_rate_gt4": safe_mean([count > 4 for count in step_counts]),
        "max_reasoning_steps": max(step_counts) if step_counts else 0,
    }

    print(pd.DataFrame([summary]).to_string(index=False))

    if errors:
        error_counter = Counter(record.get("error", "unknown").split(":", 1)[0] for record in errors)
        print()
        print("Top errors:")
        for error, count in error_counter.most_common(args.top_errors):
            print(f"{count}\t{error}")

    if args.show_bad:
        print()
        print("Bad/low-information examples:")
        shown = 0
        for record in records:
            if len(record.get("relation_intents", [])) > 0 and len(record.get("reasoning_steps", [])) > 0:
                continue
            print(json.dumps({
                "id": record.get("id"),
                "question": record.get("question"),
                "answer_type": record.get("answer_type"),
                "relation_intents": record.get("relation_intents", []),
                "reasoning_steps": record.get("reasoning_steps", []),
            }, ensure_ascii=False))
            shown += 1
            if shown >= args.show_bad:
                break


def safe_div(num, den):
    return 0.0 if den == 0 else num / den


def safe_mean(values):
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


if __name__ == "__main__":
    parser = ArgumentParser("Analyze question analysis extraction quality")
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("--error_path", default=None)
    parser.add_argument("--show_bad", type=int, default=0)
    parser.add_argument("--top_errors", type=int, default=5)
    args = parser.parse_args()
    main(args)
