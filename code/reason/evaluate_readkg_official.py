#!/usr/bin/env python3
"""Evaluate READ-KG final predictions with the paper metric scripts.

The original metric scripts infer dataset and prompt mode from the prediction
filename. This wrapper creates a stable scored_100 alias first, then reports
both full and subset metrics in one JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os

from metrics.evaluate_results import eval_results as eval_hit
from metrics.evaluate_results_corrected import eval_results as eval_corrected


DEFAULT_SCORE_FILES = {
    "cwq": "scored_triples/cwq_240907_unidir_test.pth",
    "webqsp": "scored_triples/webqsp_240912_unidir_test.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate READ-KG final predictions with official metrics."
    )
    parser.add_argument("--dataset", required=True, choices=["cwq", "webqsp"])
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--scored_triples",
        default=None,
        help=(
            "Retrieval/scored_triples .pth used for full/subset evaluation. "
            "Stage-B runners pass the current test retrieval by default."
        ),
    )
    parser.add_argument("--alias_dir", default="logs/eval_alias")
    parser.add_argument(
        "--alias_tag",
        default="readkg_selector_family",
        help="Name embedded in the scored_100 alias filename.",
    )
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def ensure_file(path: str, description: str) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(
            f"Missing {description}: {path}\n"
            "Restore this file or pass SCORED_TRIPLES=/path/to/retrieval.pth."
        )


def alias_predictions(args: argparse.Namespace) -> tuple[str, dict]:
    os.makedirs(args.alias_dir, exist_ok=True)
    alias = os.path.join(
        args.alias_dir,
        (
            f"scored_100-{args.dataset}_{args.alias_tag}"
            "-0.16-thres_0.0-test-predictions.jsonl"
        ),
    )
    total = 0
    kept = 0
    skipped_empty_ground_truth = 0
    with open(args.predictions, "r", encoding="utf-8") as src, open(
        alias, "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            ground_truth = row.get("ground_truth", [])
            if ground_truth is None:
                ground_truth = []
            if not isinstance(ground_truth, list):
                ground_truth = [ground_truth]
            if not ground_truth:
                skipped_empty_ground_truth += 1
                continue
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    return alias, {
        "source_rows": total,
        "alias_rows": kept,
        "skipped_empty_ground_truth": skipped_empty_ground_truth,
    }


def evaluate_split(path: str, scored_triples: str, subset: bool) -> dict:
    hit, _, _, _ = eval_hit(
        path,
        cal_f1=True,
        subset=subset,
        samples_to_eval_path=scored_triples,
    )
    corrected = eval_corrected(
        path,
        cal_f1=True,
        subset=subset,
        samples_to_eval_path=scored_triples,
    )
    return {
        "hit": hit,
        "f1": corrected[1],
        "hit1": corrected[0],
        "precision": corrected[2],
        "recall": corrected[3],
        "exact_match": corrected[4],
        "totally_wrong": corrected[5],
        "micro_f1": corrected[6],
        "micro_precision": corrected[7],
        "micro_recall": corrected[8],
        "total": corrected[9],
        "no_answer": corrected[10],
        "no_answer_ratio": corrected[11],
        "hal_score": corrected[12],
    }


def main() -> None:
    args = parse_args()
    scored_triples = args.scored_triples or DEFAULT_SCORE_FILES[args.dataset]
    ensure_file(args.predictions, "prediction file")
    ensure_file(scored_triples, "official scored_triples dependency")

    alias, alias_stats = alias_predictions(args)
    summary = {
        "dataset": args.dataset,
        "predictions": args.predictions,
        "alias_predictions": alias,
        "alias_stats": alias_stats,
        "scored_triples": scored_triples,
        "full": evaluate_split(alias, scored_triples, subset=False),
        "sub": evaluate_split(alias, scored_triples, subset=True),
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print("=== READ-KG Official Metrics ===")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print(f"Saved summary to: {args.output_json}")


if __name__ == "__main__":
    main()
