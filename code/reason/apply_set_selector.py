"""Apply a fixed set-level selector for a learned multi-view scorer.

Use this after selecting set-level hyperparameters on validation data. Unlike
search_set_selector.py, this script does not optimize anything on the target
split; it only applies the provided selection config.
"""

import argparse
import json
import os

from apply_answer_verifier import metric_summary, write_jsonl
from multiview_learning_utils import FEATURE_PROFILES
from search_set_selector import materialize_records, precompute


def load_selection(args):
    if args.selection_json:
        with open(args.selection_json, "r", encoding="utf-8") as f:
            selection = json.load(f)
    else:
        selection = {
            "top1_min_score": args.top1_min_score,
            "min_score": args.min_score,
            "score_gap": args.score_gap,
            "max_answers": args.max_answers,
            "min_answers": args.min_answers,
            "target_mode": args.target_mode,
            "budget_high": args.budget_high,
            "budget_multi": args.budget_multi,
        }

    required = ["top1_min_score", "min_score", "score_gap", "max_answers", "min_answers", "target_mode"]
    missing = [name for name in required if name not in selection]
    if missing:
        raise ValueError(f"Selection config missing fields: {missing}")
    if selection.get("target_mode") == "cardinality":
        for name in ("budget_high", "budget_multi"):
            if name not in selection or selection[name] in ("", None):
                raise ValueError(f"Cardinality selector requires {name}.")

    selection["top1_min_score"] = float(selection["top1_min_score"])
    selection["min_score"] = float(selection["min_score"])
    selection["score_gap"] = float(selection["score_gap"])
    selection["max_answers"] = int(selection["max_answers"])
    selection["min_answers"] = int(selection["min_answers"])
    selection["target_mode"] = str(selection["target_mode"])
    if "budget_high" in selection and selection["budget_high"] not in ("", None):
        selection["budget_high"] = int(selection["budget_high"])
    if "budget_multi" in selection and selection["budget_multi"] not in ("", None):
        selection["budget_multi"] = int(selection["budget_multi"])
    return selection


def main(args):
    selection = load_selection(args)
    precomputed, view_specs, view_records = precompute(args)

    print("\n=== Fixed set selector config ===")
    print(json.dumps(selection, ensure_ascii=False, sort_keys=True))

    print("\n=== Per-view standalone metrics ===")
    for view in view_specs:
        records = list(view_records[view["name"]].values())
        ms = metric_summary(records)
        print(
            f"  {view['name']:<12} hit={ms.get('hit', 0):.4f} f1={ms.get('f1', 0):.4f} "
            f"prec={ms.get('precision', 0):.4f} recall={ms.get('recall', 0):.4f} "
            f"no_ans={ms.get('no_answer', 0):.4f}"
        )

    records = materialize_records(precomputed, selection)
    write_jsonl(args.output_path, records)
    print("\n=== Fixed set selector output ===")
    ms = metric_summary(records)
    print(
        f"  selected    hit={ms.get('hit', 0):.4f} f1={ms.get('f1', 0):.4f} "
        f"prec={ms.get('precision', 0):.4f} recall={ms.get('recall', 0):.4f} "
        f"no_ans={ms.get('no_answer', 0):.4f}"
    )
    print(f"\nSaved selected predictions to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Apply a fixed READ-KG set selector.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--scorer_mode", choices=["learned", "order", "vote", "base"], default="learned")
    parser.add_argument("--view", action="append", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--dataset", required=True, choices=["cwq", "webqsp"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--analysis_path", default=None)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--max_hops", type=int, default=4)
    parser.add_argument("--feature_profile", choices=FEATURE_PROFILES, default="full")
    parser.add_argument("--keep_topics", action="store_true")
    parser.add_argument("--include_no_answer_candidate", action="store_true")
    parser.add_argument("--default_multi_n", type=int, default=2)
    parser.add_argument("--multi_min_n", type=int, default=2)
    parser.add_argument("--multi_max_n", type=int, default=8)
    parser.add_argument("--selection_json", default=None)
    parser.add_argument("--top1_min_score", type=float, default=-100.0)
    parser.add_argument("--min_score", type=float, default=-1.0)
    parser.add_argument("--score_gap", type=float, default=1.0)
    parser.add_argument("--max_answers", type=int, default=8)
    parser.add_argument("--budget_high", type=int, default=8)
    parser.add_argument("--budget_multi", type=int, default=8)
    parser.add_argument("--min_answers", type=int, default=1)
    parser.add_argument("--target_mode", choices=["expected", "fixed", "cardinality"], default="fixed")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args)
