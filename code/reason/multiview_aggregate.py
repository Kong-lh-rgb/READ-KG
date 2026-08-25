"""READ-KG: Test-Time Reader-Aware Multi-View Answer Aggregation.

Pipeline assumed already done (offline, expensive):
  - For each sample, the Llama reader has been run independently on N evidence
    views (flat / pathfirst / candidate / constraint / minimal). Each run was
    saved as a separate predictions.jsonl with the standard schema:
        {"id": ..., "question": ..., "ground_truth": [...], "prediction": "..."}

This script does the cheap part: read those N jsonl files, extract candidate
answers across all views, score each candidate by (cross-view consistency +
weighted view origin + KG support + analysis alignment + no-answer penalty),
and emit a single fused predictions.jsonl that downstream evaluation (and the
existing fill-only verifier, optionally) can consume unchanged.

Nothing is trained. All fusion weights are CLI flags so you can grid-search
them on a small dev split. A learned-weight v2 (logistic regression on
val outcome labels) is a 30-line follow-up if v1 shows real headroom.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from apply_answer_verifier import (
    collect_candidates,
    dedupe_answers,
    eval_record,
    extract_pred_values,
    format_prediction,
    likely_multi_answer,
    load_analysis_by_id,
    map_prediction_to_candidate,
    metric_summary,
    normalize_text,
    prediction_has_no_answer,
    resolve_analysis_path,
    torch_load,
    write_detail,
    write_jsonl,
)


def parse_view_specs(view_specs):
    """Parse "name1=path1[:weight1],name2=path2[:weight2],..." into a list."""
    parsed = []
    for spec in view_specs:
        items = spec.split(",") if "," in spec else [spec]
        for item in items:
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"Bad --view spec '{item}', expected name=path[:weight]")
            name, rest = item.split("=", 1)
            if ":" in rest:
                path, weight = rest.rsplit(":", 1)
                weight = float(weight)
            else:
                path = rest
                weight = 1.0
            filter_view = None
            if "#" in path:
                path, filter_view = path.rsplit("#", 1)
                if filter_view.startswith("view="):
                    filter_view = filter_view.split("=", 1)[1]
            parsed.append({
                "name": name.strip(),
                "path": path.strip(),
                "weight": weight,
                "filter_view": (filter_view or name.strip()),
            })
    return parsed


def load_predictions(path, filter_view=None):
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            # Our probe script writes all views into one jsonl with a "view"
            # field. If present, keep only the requested view. Plain per-view
            # prediction files without this field are still accepted unchanged.
            if filter_view is not None and record.get("view") is not None:
                if record.get("view") != filter_view:
                    continue
            sid = record.get("id")
            if sid is not None:
                records[sid] = record
    return records


def candidate_to_kg_obj(pred_value, kg_candidates):
    """Map a free-text reader answer to a KG candidate object, if possible."""
    mapped = map_prediction_to_candidate(pred_value, kg_candidates)
    return mapped


def merge_keys(norm_text):
    return norm_text.strip()


def score_fused(args, n_views, votes, weight_sum, kg_obj, is_no_answer):
    """Combine all signals into a single scalar score."""
    if is_no_answer:
        return -args.w_noans  # negative bias; we will only output if nothing else survives
    consist = votes / max(n_views, 1)
    weighted = weight_sum
    kg_score = float(kg_obj["score"]) if kg_obj is not None else 0.0
    rel = float(kg_obj["relation_match"]) if kg_obj is not None else 0.0
    cstr = float(kg_obj["constraint_match"]) if kg_obj is not None else 0.0
    atype = float(kg_obj["answer_type_match"]) if kg_obj is not None else 0.0
    path = float(kg_obj["path_score"]) if kg_obj is not None else 0.0
    return (
        args.w_consist * consist
        + args.w_weighted * weighted
        + args.w_path * path
        + args.w_kg * kg_score
        + args.w_relation * rel
        + args.w_constraint * cstr
        + args.w_type * atype
    )


def answer_disagreement(record_per_view):
    n_views = len(record_per_view)
    distinct = set()
    active_views = 0
    for record in record_per_view.values():
        pred_values = extract_pred_values(record.get("prediction", "")) if record else []
        if not pred_values:
            continue
        active_views += 1
        for pred in pred_values:
            norm = normalize_text(pred)
            if norm:
                distinct.add(norm)
    if active_views == 0:
        return 1.0
    # 0 means all active views produced the same answer; values near 1 mean
    # strong cross-view disagreement and a higher risk that the default view is
    # confidently wrong.
    return min(max((len(distinct) - 1) / max(n_views - 1, 1), 0.0), 1.0)


def effective_view_weights(args, view_weights, record_per_view):
    if args.pathfirst_decay <= 0:
        return dict(view_weights), answer_disagreement(record_per_view)
    disagreement = answer_disagreement(record_per_view)
    output = {}
    for name, weight in view_weights.items():
        if name == args.pathfirst_view_name:
            output[name] = weight * max(0.0, 1.0 - args.pathfirst_decay * disagreement)
        else:
            output[name] = weight
    return output, disagreement


def expected_answer_count(views_pred_values):
    counts = [len(p) for p in views_pred_values if p]
    if not counts:
        return 1
    return max(1, int(round(float(np.median(counts)))))


def aggregate_sample(record_per_view, view_weights, kg_candidates, args, multi_answer_hint):
    """Aggregate one sample across N views. Returns final answers + per-cand debug info."""
    n_views = len(record_per_view)
    effective_weights, disagreement = effective_view_weights(args, view_weights, record_per_view)
    by_key = {}  # norm -> dict of {original, votes, weight_sum, view_origins, is_no_answer}
    views_pred_values = []
    for view_name, record in record_per_view.items():
        pred_values = extract_pred_values(record.get("prediction", "")) if record else []
        views_pred_values.append(pred_values)
        if (not pred_values) or prediction_has_no_answer(record.get("prediction", "") if record else ""):
            key = "__NO_ANSWER__"
            entry = by_key.setdefault(key, {
                "original": None, "votes": 0, "weight_sum": 0.0,
                "view_origins": [], "is_no_answer": True,
            })
            entry["votes"] += 1
            entry["weight_sum"] += effective_weights.get(view_name, 1.0)
            entry["view_origins"].append(view_name)
            continue
        for pv in pred_values:
            norm = normalize_text(pv)
            if not norm:
                continue
            entry = by_key.setdefault(norm, {
                "original": pv, "votes": 0, "weight_sum": 0.0,
                "view_origins": [], "is_no_answer": False,
            })
            entry["votes"] += 1
            entry["weight_sum"] += effective_weights.get(view_name, 1.0)
            entry["view_origins"].append(view_name)
            if entry["original"] is None:
                entry["original"] = pv

    scored = []
    for key, info in by_key.items():
        kg_obj = None
        if not info["is_no_answer"]:
            kg_obj = candidate_to_kg_obj(info["original"], kg_candidates)
        score = score_fused(args, n_views, info["votes"], info["weight_sum"], kg_obj, info["is_no_answer"])
        scored.append({
            "key": key,
            "answer": info["original"],
            "score": score,
            "votes": info["votes"],
            "weight_sum": info["weight_sum"],
            "view_origins": info["view_origins"],
            "kg_score": float(kg_obj["score"]) if kg_obj is not None else 0.0,
            "kg_path": float(kg_obj["path_score"]) if kg_obj is not None else 0.0,
            "kg_rel": float(kg_obj["relation_match"]) if kg_obj is not None else 0.0,
            "kg_cstr": float(kg_obj["constraint_match"]) if kg_obj is not None else 0.0,
            "kg_type": float(kg_obj["answer_type_match"]) if kg_obj is not None else 0.0,
            "is_no_answer": info["is_no_answer"],
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Decide output size.
    multi = multi_answer_hint
    if multi:
        target_n = max(args.multi_min_n, expected_answer_count(views_pred_values))
        target_n = min(target_n, args.multi_max_n)
    else:
        target_n = 1

    # Filter no_answer placeholder unless it's the only option.
    real_scored = [s for s in scored if not s["is_no_answer"]]
    if not real_scored:
        return [], scored[: args.debug_top_n], disagreement

    # Apply single-answer confidence threshold.
    final = []
    if multi:
        for s in real_scored:
            if len(final) >= target_n:
                break
            if s["score"] < args.multi_min_score and final:
                break
            final.append(s["answer"])
    else:
        top = real_scored[0]
        if top["score"] >= args.single_min_score:
            final.append(top["answer"])

    return final, scored[: args.debug_top_n], disagreement


def main(args):
    view_specs = parse_view_specs(args.view)
    if not view_specs:
        raise ValueError("No --view specs provided.")
    view_weights = {v["name"]: v["weight"] for v in view_specs}
    print(f"View specs:")
    for v in view_specs:
        print(f"  {v['name']:<12} weight={v['weight']:.3f} path={v['path']}")

    # Load all views.
    view_records = {}
    for v in view_specs:
        view_records[v["name"]] = load_predictions(v["path"], filter_view=v.get("filter_view"))
        print(f"  loaded {v['name']}: {len(view_records[v['name']])} records")

    # Use the first view as the reference set of sample_ids and ground_truth.
    primary_view_name = view_specs[0]["name"]
    primary_records = view_records[primary_view_name]

    retrieval = torch_load(args.retrieval)
    analysis_path = resolve_analysis_path(args.dataset, args.split, args.analysis_path)
    analysis_by_id = load_analysis_by_id(analysis_path)
    if analysis_path:
        print(f"Using question analysis: {analysis_path}")

    corrected = []
    detail_rows = []
    action_counts = defaultdict(int)
    missing_retrieval = 0

    for sample_id, ref_record in tqdm(primary_records.items(), desc="aggregate"):
        sample = retrieval.get(sample_id)
        if sample is None:
            missing_retrieval += 1
            corrected.append(ref_record)
            continue
        sample_for_scoring = dict(sample)
        sample_for_scoring.setdefault("question", ref_record.get("question", ""))
        analysis = analysis_by_id.get(sample_id, {})
        kg_candidates = collect_candidates(
            sample_for_scoring,
            k=args.k,
            analysis=analysis,
            max_hops=args.max_hops,
            keep_topics=args.keep_topics,
        )

        record_per_view = {}
        for v in view_specs:
            r = view_records.get(v["name"], {}).get(sample_id)
            if r is not None:
                record_per_view[v["name"]] = r

        multi_hint = likely_multi_answer(ref_record.get("question", ""))
        old_eval = eval_record(ref_record)
        answers, top_debug, disagreement = aggregate_sample(
            record_per_view, view_weights, kg_candidates, args, multi_hint
        )
        answers = dedupe_answers(answers)
        action = "aggregated" if answers else "no_answer"

        # Fallback safety: if aggregator produced nothing but primary view had something
        # (e.g., everything was filtered by single_min_score), keep the primary.
        if not answers:
            primary_pred = extract_pred_values(ref_record.get("prediction", ""))
            if primary_pred:
                answers = primary_pred
                action = "fallback_primary"

        new_record = dict(ref_record)
        new_record["raw_prediction_before_aggregator"] = ref_record.get("prediction", "")
        new_record["prediction"] = format_prediction(answers) if answers else "ans: not available"
        new_eval = eval_record(new_record)
        corrected.append(new_record)
        action_counts[action] += 1
        detail_rows.append({
            "id": sample_id,
            "question": ref_record.get("question", ""),
            "action": action,
            "multi_hint": int(multi_hint),
            "disagreement": disagreement,
            "old_prediction": " || ".join(extract_pred_values(ref_record.get("prediction", ""))),
            "new_prediction": " || ".join(answers),
            "old_hit": old_eval["hit"],
            "new_hit": new_eval["hit"],
            "old_f1": old_eval["f1"],
            "new_f1": new_eval["f1"],
            "old_no_answer": old_eval["no_answer"],
            "new_no_answer": new_eval["no_answer"],
            "top_candidates": " || ".join(
                f"{s['answer'] or '<NA>'}[score={s['score']:.3f};votes={s['votes']};views={'+'.join(s['view_origins'])};kg={s['kg_score']:.2f}]"
                for s in top_debug
            ),
        })

    write_jsonl(args.output_path, corrected)
    detail_path = args.detail_path or args.output_path.replace(".jsonl", "_multiview_detail.csv")
    write_detail(detail_path, detail_rows)

    print(f"Loaded primary view: {len(primary_records)} records")
    print(f"Missing retrieval samples: {missing_retrieval}")
    print("Action counts:", dict(sorted(action_counts.items())))

    # Per-view standalone baseline + aggregated comparison.
    print("\n=== Per-view standalone metrics (using each view's raw predictions) ===")
    for v in view_specs:
        recs = list(view_records[v["name"]].values())
        ms = metric_summary(recs)
        print(f"  {v['name']:<12} hit={ms.get('hit', 0):.4f} f1={ms.get('f1', 0):.4f} "
              f"prec={ms.get('precision', 0):.4f} recall={ms.get('recall', 0):.4f} "
              f"no_ans={ms.get('no_answer', 0):.4f}")

    oracle_records = []
    for sample_id in primary_records:
        candidates = []
        for v in view_specs:
            record = view_records[v["name"]].get(sample_id)
            if record is None:
                continue
            ev = eval_record(record)
            candidates.append((ev["hit"], ev["f1"], ev["precision"], ev["recall"], record))
        if candidates:
            candidates.sort(key=lambda item: item[:4], reverse=True)
            oracle_records.append(candidates[0][-1])
    oracle = metric_summary(oracle_records)
    print("\n=== Oracle best-view upper bound ===")
    print(f"  oracle      hit={oracle.get('hit', 0):.4f} f1={oracle.get('f1', 0):.4f} "
          f"prec={oracle.get('precision', 0):.4f} recall={oracle.get('recall', 0):.4f} "
          f"no_ans={oracle.get('no_answer', 0):.4f}")

    print("\n=== Aggregator output ===")
    after = metric_summary(corrected)
    print(f"  fused        hit={after.get('hit', 0):.4f} f1={after.get('f1', 0):.4f} "
          f"prec={after.get('precision', 0):.4f} recall={after.get('recall', 0):.4f} "
          f"no_ans={after.get('no_answer', 0):.4f}")
    print(f"\nSaved fused predictions to: {args.output_path}")
    print(f"Saved aggregator detail to: {detail_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("READ-KG: test-time multi-view answer aggregator.")
    parser.add_argument("--view", action="append", required=True,
                        help="Repeatable: --view name=path[:weight]. Example: "
                             "--view pathfirst=pred_pathfirst.jsonl:0.40 "
                             "--view minimal=pred_minimal.jsonl:0.20")
    parser.add_argument("--retrieval", required=True, help="Retrieval result .pth for KG support.")
    parser.add_argument("--dataset", required=True, choices=["cwq", "webqsp"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--analysis_path", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--detail_path", default=None)
    # KG support knobs.
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--max_hops", type=int, default=4)
    parser.add_argument("--keep_topics", action="store_true")
    # Fusion weights (start with these, then grid-search on a dev split).
    parser.add_argument("--w_consist", type=float, default=1.2)
    parser.add_argument("--w_weighted", type=float, default=0.8)
    parser.add_argument("--w_path", type=float, default=0.5)
    parser.add_argument("--w_kg", type=float, default=0.6)
    parser.add_argument("--w_relation", type=float, default=0.3)
    parser.add_argument("--w_constraint", type=float, default=0.3)
    parser.add_argument("--w_type", type=float, default=0.2)
    parser.add_argument("--w_noans", type=float, default=1.5)
    parser.add_argument("--pathfirst_decay", type=float, default=0.0,
                        help="Discount the pathfirst view weight when views disagree. 0 disables it.")
    parser.add_argument("--pathfirst_view_name", default="pathfirst")
    # Output-shape knobs.
    parser.add_argument("--single_min_score", type=float, default=0.30)
    parser.add_argument("--multi_min_score", type=float, default=0.35)
    parser.add_argument("--multi_min_n", type=int, default=2)
    parser.add_argument("--multi_max_n", type=int, default=8)
    parser.add_argument("--debug_top_n", type=int, default=6)
    args = parser.parse_args()
    main(args)
