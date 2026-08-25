"""Search set-level selection parameters for a learned multi-view scorer.

This script does not train the scorer. It loads an existing learned answer
scorer, precomputes per-sample candidate scores once, then grid-searches output
set selection parameters against sample-level QA F1.
"""

import argparse
import csv
import itertools
import json
import os
import re
import sys

from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from apply_answer_verifier import (
    dedupe_answers,
    eval_record,
    extract_pred_values,
    format_prediction,
    likely_multi_answer,
    metric_summary,
    write_jsonl,
)
from apply_multiview_aggregator import load_model, score_entries
from multiview_learning_utils import (
    FEATURE_NAMES,
    FEATURE_PROFILES,
    collect_sample_entries,
    expected_answer_count,
    load_multiview_context,
    vote_kg_base_score_from_features,
)


def parse_float_list(text):
    return [float(item) for item in str(text).split(",") if item.strip()]


def parse_int_list(text):
    return [int(item) for item in str(text).split(",") if item.strip()]


LIST_STYLE_MARKERS = {
    "songs", "books", "movies", "films", "languages", "teams",
    "works", "awards", "places", "albums", "products", "children",
    "members", "cities", "countries", "actors", "characters",
    "representatives",
}


def question_cardinality_tier(question):
    """Dataset-agnostic answer-cardinality heuristic for set-size budgeting."""
    q = str(question or "").lower()
    tokens = set(re.findall(r"[a-z]+", q))
    if tokens & LIST_STYLE_MARKERS:
        return "high"
    if likely_multi_answer(q):
        return "multi"
    return "single"


def select_set(scored, selection):
    real = [item for item in scored if not item.get("is_no_answer")]
    if not real:
        return []
    top_score = real[0]["score"]
    if top_score < selection["top1_min_score"]:
        return []

    limit = selection["target_n"]
    selected = []
    for item in real:
        if len(selected) >= limit:
            break
        if len(selected) > 0 and item["score"] < selection["min_score"]:
            break
        if len(selected) > 0 and selection["score_gap"] > 0 and (top_score - item["score"]) > selection["score_gap"]:
            break
        selected.append(item["answer"])
    return dedupe_answers(selected)


def score_entries_nonlearned(entries, scorer_mode, score_feature="vote_kg_base_score"):
    """Score candidates with fixed ablation scorers.

    Modes:
      order: strict no-scorer baseline, preserve first occurrence order across
        pathfirst -> minimal -> candidate and each view's output order.
      vote: cross-view vote count only, without KG support.
      base: hand-designed vote+KG support score.
    """
    if not entries:
        return []

    score_idx = FEATURE_NAMES.index(score_feature) if score_feature in FEATURE_NAMES else None
    output = []
    for order_idx, entry in enumerate(entries):
        enriched = dict(entry)
        if entry.get("is_no_answer"):
            score = -100.0
        elif scorer_mode == "order":
            score = -float(order_idx)
        elif scorer_mode == "vote":
            score = (
                float(entry.get("votes", 0))
                + 1e-3 * float(entry.get("view_weight_sum", 0.0))
                - 1e-6 * float(order_idx)
            )
        elif scorer_mode == "base":
            if score_idx is not None and len(entry.get("features", [])) > score_idx:
                score = entry["features"][score_idx]
            else:
                score = vote_kg_base_score_from_features(entry["features"])
        else:
            raise ValueError(f"Unsupported non-learned scorer_mode: {scorer_mode}")
        enriched["score"] = float(score)
        output.append(enriched)
    output.sort(key=lambda item: item["score"], reverse=True)
    return output


def materialize_records(precomputed, selection):
    records = []
    for item in precomputed:
        if selection["target_mode"] == "expected":
            target_n = item["expected_n"]
        elif selection["target_mode"] == "cardinality":
            tier = question_cardinality_tier(item["ref_record"].get("question", ""))
            if tier == "high":
                target_n = selection.get("budget_high", selection["max_answers"])
            elif tier == "multi":
                target_n = selection.get("budget_multi", selection["max_answers"])
            else:
                target_n = 1
        else:
            target_n = selection["max_answers"]
        target_n = max(selection["min_answers"], target_n)
        target_n = min(target_n, selection["max_answers"])
        local_selection = dict(selection)
        local_selection["target_n"] = target_n
        answers = select_set(item["scored"], local_selection)
        record = dict(item["ref_record"])
        record["prediction"] = format_prediction(answers) if answers else "ans: not available"
        records.append(record)
    return records


def precompute(args):
    device = args.device
    model = None
    if args.scorer_mode == "learned":
        if not args.model:
            raise ValueError("--model is required when --scorer_mode learned.")
        model, _, payload = load_model(args.model, device)
        print(f"Loaded model from: {args.model}")
        print("Train info:", json.dumps(payload.get("train_info", {}), ensure_ascii=False))
    elif args.scorer_mode in {"order", "vote", "base"}:
        scorer_descriptions = {
            "order": "strict no-scorer order baseline",
            "vote": "vote-only scorer without KG support",
            "base": "vote+KG support scorer",
        }
        print(f"Using non-learned scorer: {scorer_descriptions[args.scorer_mode]}")
    else:
        raise ValueError(f"Unknown scorer_mode: {args.scorer_mode}")

    view_specs, view_weights, view_records, primary_records, retrieval, analysis_by_id = load_multiview_context(args)
    precomputed = []
    missing_retrieval = 0
    no_candidate = 0

    for sample_id, ref_record in tqdm(primary_records.items(), desc="precompute scores"):
        _, record_per_view, entries = collect_sample_entries(
            sample_id, ref_record, view_specs, view_weights, view_records, retrieval, analysis_by_id, args
        )
        if entries is None:
            missing_retrieval += 1
            continue
        if not entries:
            no_candidate += 1
            scored = []
        elif args.scorer_mode in {"order", "vote", "base"}:
            scored = score_entries_nonlearned(entries, args.scorer_mode)
        else:
            scored = score_entries(model, entries, device)
        views_pred_values = [
            extract_pred_values(record.get("prediction", ""))
            for record in (record_per_view or {}).values()
        ]
        precomputed.append({
            "id": sample_id,
            "ref_record": ref_record,
            "scored": scored,
            "expected_n": expected_answer_count(views_pred_values),
        })

    print(f"precomputed={len(precomputed)} missing_retrieval={missing_retrieval} no_candidate={no_candidate}")
    return precomputed, view_specs, view_records


def main(args):
    precomputed, view_specs, view_records = precompute(args)

    print("\n=== Per-view standalone metrics ===")
    for view in view_specs:
        records = list(view_records[view["name"]].values())
        ms = metric_summary(records)
        print(f"  {view['name']:<12} hit={ms.get('hit', 0):.4f} f1={ms.get('f1', 0):.4f} "
              f"prec={ms.get('precision', 0):.4f} recall={ms.get('recall', 0):.4f} "
              f"no_ans={ms.get('no_answer', 0):.4f}")

    grid_rows = []
    best = None
    thresholds = parse_float_list(args.min_scores)
    gaps = parse_float_list(args.score_gaps)
    max_answers_list = parse_int_list(args.max_answers_list)
    budget_high_list = parse_int_list(args.budget_high_list)
    budget_multi_list = parse_int_list(args.budget_multi_list)
    top1_thresholds = parse_float_list(args.top1_min_scores)
    target_modes = [item for item in args.target_modes.split(",") if item.strip()]

    for target_mode in target_modes:
        if target_mode == "cardinality":
            budget_configs = [
                (max(budget_high, budget_multi), budget_high, budget_multi)
                for budget_high, budget_multi in itertools.product(budget_high_list, budget_multi_list)
            ]
        else:
            budget_configs = [(max_answers, "", "") for max_answers in max_answers_list]

        for top1_min_score, min_score, score_gap, (max_answers, budget_high, budget_multi) in itertools.product(
            top1_thresholds, thresholds, gaps, budget_configs
        ):
            selection = {
                "top1_min_score": top1_min_score,
                "min_score": min_score,
                "score_gap": score_gap,
                "max_answers": max_answers,
                "min_answers": args.min_answers,
                "target_mode": target_mode,
                "budget_high": budget_high,
                "budget_multi": budget_multi,
            }
            records = materialize_records(precomputed, selection)
            ms = metric_summary(records)
            row = {
                **selection,
                "hit": ms.get("hit", 0.0),
                "f1": ms.get("f1", 0.0),
                "precision": ms.get("precision", 0.0),
                "recall": ms.get("recall", 0.0),
                "no_answer": ms.get("no_answer", 0.0),
            }
            grid_rows.append(row)
            if best is None or (row["f1"], row["hit"], row["precision"], row["recall"]) > (
                best["f1"], best["hit"], best["precision"], best["recall"]
            ):
                best = row

    os.makedirs(os.path.dirname(args.grid_csv), exist_ok=True)
    with open(args.grid_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        writer.writeheader()
        writer.writerows(grid_rows)
    print(f"Saved grid results to: {args.grid_csv}")

    best_records = materialize_records(precomputed, best)
    write_jsonl(args.output_path, best_records)
    if args.best_json:
        os.makedirs(os.path.dirname(args.best_json) or ".", exist_ok=True)
        with open(args.best_json, "w", encoding="utf-8") as f:
            json.dump(best, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Saved best selection config to: {args.best_json}")
    print("\n=== Best set selector ===")
    print(json.dumps(best, ensure_ascii=False, sort_keys=True))
    print(f"Saved best predictions to: {args.output_path}")

    print("\nTop 10 configs by F1:")
    for row in sorted(grid_rows, key=lambda x: (x["f1"], x["hit"], x["precision"]), reverse=True)[:10]:
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Search set-level selection for learned multi-view aggregation.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--scorer_mode", choices=["learned", "order", "vote", "base"], default="learned")
    parser.add_argument("--view", action="append", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--dataset", required=True, choices=["cwq", "webqsp"])
    parser.add_argument("--split", default="val")
    parser.add_argument("--analysis_path", default=None)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--max_hops", type=int, default=4)
    parser.add_argument("--feature_profile", choices=FEATURE_PROFILES, default="full")
    parser.add_argument("--keep_topics", action="store_true")
    parser.add_argument("--include_no_answer_candidate", action="store_true")
    parser.add_argument("--min_scores", default="-100,-2,-1,-0.5,0,0.5,1,1.5,2")
    parser.add_argument("--score_gaps", default="0,0.5,1,1.5,2,3")
    parser.add_argument("--top1_min_scores", default="-100")
    parser.add_argument(
        "--max_answers_list",
        default="1,2,3,5,8,12,16,24,32,64,96",
        help="Validation search space for the maximum answer-set size.",
    )
    parser.add_argument("--budget_high_list", default="8,12,16,24,32")
    parser.add_argument("--budget_multi_list", default="4,8,12")
    parser.add_argument("--min_answers", type=int, default=1)
    parser.add_argument("--target_modes", default="expected,fixed")
    parser.add_argument("--grid_csv", required=True)
    parser.add_argument("--best_json", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args)
