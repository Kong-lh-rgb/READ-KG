"""Shared utilities for learned multi-view answer aggregation."""

import os
import sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from apply_answer_verifier import (
    collect_candidates,
    dedupe_answers,
    double_check_enabled,
    eval_record,
    extract_pred_values,
    format_prediction,
    likely_multi_answer,
    load_analysis_by_id,
    map_prediction_to_candidate,
    match_text,
    metric_summary,
    normalize_answers,
    normalize_text,
    prediction_has_no_answer,
    resolve_analysis_path,
    torch_load,
    write_detail,
    write_jsonl,
)
from multiview_aggregate import load_predictions
from rule_score_utils import load_candidate_score_config


FEATURE_NAMES = [
    "bias",
    "votes_frac",
    "view_weight_sum",
    "appears_pathfirst",
    "appears_minimal",
    "appears_candidate",
    "appears_flat",
    "appears_constraint",
    "kg_score",
    "kg_path",
    "kg_rank",
    "kg_triple",
    "kg_mean_triple",
    "kg_relation",
    "kg_constraint",
    "kg_type",
    "kg_step",
    "kg_question",
    "support_count",
    "answer_len_norm",
    "is_no_answer",
    "agreement_2plus",
    "agreement_all",
    "pf_and_minimal",
    "pf_and_candidate",
    "minimal_and_candidate",
    "votes_x_kg_score",
    "vote_kg_base_score",
    "vote_kg_rank_score",
    "vote_kg_gap_to_top",
    "vote_kg_gap_to_next",
    "view_rank_min_score",
    "view_rank_mean_score",
    "appears_first_in_any_view",
    "view_rank_spread_norm",
]

FEATURE_PROFILES = ("full", "generic")
VIEW_IDENTITY_FEATURES = {
    "appears_pathfirst",
    "appears_minimal",
    "appears_candidate",
    "appears_flat",
    "appears_constraint",
    "pf_and_minimal",
    "pf_and_candidate",
    "minimal_and_candidate",
}


def apply_feature_profile(entries, profile):
    if profile == "full":
        return entries
    if profile != "generic":
        raise ValueError(f"Unknown feature profile: {profile}")
    masked_indices = [FEATURE_NAMES.index(name) for name in VIEW_IDENTITY_FEATURES]
    for entry in entries:
        features = list(entry.get("features", []))
        for index in masked_indices:
            if index < len(features):
                features[index] = 0.0
        entry["features"] = features
    return entries

def match_answer_value(pred_value, answers, question=""):
    double_check = double_check_enabled(question)
    for answer in answers:
        if match_text(pred_value, answer) or (double_check and match_text(answer, pred_value)):
            return True
    return False


def expected_answer_count(views_pred_values):
    counts = [len(values) for values in views_pred_values if values]
    if not counts:
        return 1
    return max(1, int(round(float(np.median(counts)))))


def sample_answer_limit(ref_record, args):
    if likely_multi_answer(ref_record.get("question", "")):
        target_n = max(args.multi_min_n, int(args.default_multi_n))
        return min(target_n, args.multi_max_n)
    return 1


def kg_feature_value(kg_obj, key):
    if kg_obj is None:
        return 0.0
    return float(kg_obj.get(key, 0.0) or 0.0)


def vote_kg_base_score_from_features(features):
    """Strong non-learned ranking score used as residual baseline.

    This fixed vote+KG recipe lets the learned scorer focus on residual
    corrections instead of rediscovering basic cross-view/KG support signals.
    """
    votes_frac = float(features[1])
    view_weight_sum = float(features[2])
    kg_score = float(features[8])
    kg_triple = float(features[11])
    kg_mean_triple = float(features[12])
    support = float(features[18])
    return (
        votes_frac
        + 0.75 * kg_score
        + 0.35 * kg_triple
        + 0.20 * kg_mean_triple
        + 0.20 * support
        + 0.10 * view_weight_sum
    )


def parse_view_specs(view_specs):
    """Parse view specs and support both path:weight#view=x and path#view=x:weight."""
    parsed = []
    for spec in view_specs:
        items = spec.split(",") if "," in spec else [spec]
        for item in items:
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"Bad --view spec '{item}', expected name=path[:weight][#view=name]")
            name, rest = item.split("=", 1)
            name = name.strip()
            rest = rest.strip()
            filter_view = None
            weight = 1.0

            if "#view=" in rest:
                rest, filter_view = rest.split("#view=", 1)
                rest = rest.strip()
                filter_view = filter_view.strip()
                if ":" in filter_view:
                    maybe_filter, maybe_weight = filter_view.rsplit(":", 1)
                    try:
                        weight = float(maybe_weight)
                        filter_view = maybe_filter
                    except ValueError:
                        pass

            if ":" in rest:
                path, maybe_weight = rest.rsplit(":", 1)
                try:
                    weight = float(maybe_weight)
                    rest = path
                except ValueError:
                    pass

            parsed.append({
                "name": name,
                "path": rest.strip(),
                "weight": weight,
                "filter_view": filter_view or name,
            })
    return parsed


def build_candidate_entries(
    record_per_view,
    view_weights,
    kg_candidates,
    include_no_answer=False,
):
    n_views = len(record_per_view)
    canonicalize_to_kg = os.environ.get("READKG_CANONICALIZE_TO_KG", "").strip().lower() in {"1", "true", "yes", "on"}
    by_key = {}
    views_pred_values = []

    for view_name, record in record_per_view.items():
        pred_values = extract_pred_values(record.get("prediction", "")) if record else []
        if canonicalize_to_kg and pred_values:
            canonical_values = []
            canonical_seen = set()
            for pred_value in pred_values:
                pred_lower = str(pred_value).strip().lower()
                if any(marker in pred_lower for marker in (
                    "not available", "no information", "not provided",
                    "cannot determine", "unable to determine", "insufficient information",
                    "no answer", "does not contain", "do not contain",
                )):
                    continue
                kg_match = map_prediction_to_candidate(pred_value, kg_candidates)
                canonical_value = kg_match["entity"] if kg_match is not None else pred_value
                canonical_norm = normalize_text(canonical_value)
                if not canonical_norm or canonical_norm in canonical_seen:
                    continue
                canonical_seen.add(canonical_norm)
                canonical_values.append(canonical_value)
            pred_values = canonical_values
        views_pred_values.append(pred_values)
        if include_no_answer and ((not pred_values) or prediction_has_no_answer(record.get("prediction", "") if record else "")):
            entry = by_key.setdefault("__NO_ANSWER__", {
                "answer": "not available",
                "votes": 0,
                "view_weight_sum": 0.0,
                "view_origins": [],
                "view_ranks": {},
                "is_no_answer": True,
            })
            entry["votes"] += 1
            entry["view_weight_sum"] += view_weights.get(view_name, 1.0)
            entry["view_origins"].append(view_name)
            continue

        for rank, pred in enumerate(pred_values, start=1):
            norm = normalize_text(pred)
            if not norm:
                continue
            entry = by_key.setdefault(norm, {
                "answer": pred,
                "votes": 0,
                "view_weight_sum": 0.0,
                "view_origins": [],
                "view_ranks": {},
                "is_no_answer": False,
            })
            entry["votes"] += 1
            entry["view_weight_sum"] += view_weights.get(view_name, 1.0)
            entry["view_origins"].append(view_name)
            entry["view_ranks"][view_name] = min(entry["view_ranks"].get(view_name, rank), rank)

    entries = []
    for key, info in by_key.items():
        kg_obj = None
        if not info["is_no_answer"]:
            kg_obj = map_prediction_to_candidate(info["answer"], kg_candidates)
        origins = set(info["view_origins"])
        answer_len = len(str(info["answer"] or "").split())
        votes_frac = float(info["votes"]) / max(n_views, 1)
        kg_score = kg_feature_value(kg_obj, "score")
        features = [
            1.0,
            votes_frac,
            float(info["view_weight_sum"]),
            float("pathfirst" in origins),
            float("minimal" in origins),
            float("candidate" in origins),
            float("flat" in origins),
            float("constraint" in origins),
            kg_score,
            kg_feature_value(kg_obj, "path_score"),
            kg_feature_value(kg_obj, "max_rank_score"),
            kg_feature_value(kg_obj, "max_triple_score"),
            kg_feature_value(kg_obj, "mean_triple_score"),
            kg_feature_value(kg_obj, "relation_match"),
            kg_feature_value(kg_obj, "constraint_match"),
            kg_feature_value(kg_obj, "answer_type_match"),
            kg_feature_value(kg_obj, "step_match"),
            kg_feature_value(kg_obj, "question_match"),
            kg_feature_value(kg_obj, "support_count_score"),
            min(answer_len, 8) / 8.0,
            float(info["is_no_answer"]),
            float(info["votes"] >= 2),
            float(info["votes"] >= max(n_views, 1) and n_views > 0),
            float("pathfirst" in origins and "minimal" in origins),
            float("pathfirst" in origins and "candidate" in origins),
            float("minimal" in origins and "candidate" in origins),
            votes_frac * kg_score,
        ]
        entries.append({
            "key": key,
            "answer": info["answer"],
            "features": features,
            "votes": info["votes"],
            "view_weight_sum": info["view_weight_sum"],
            "view_origins": sorted(origins),
            "view_ranks": dict(info.get("view_ranks", {})),
            "kg_obj": kg_obj,
            "is_no_answer": info["is_no_answer"],
        })

    real_entries = [entry for entry in entries if not entry.get("is_no_answer")]
    base_scores = {
        entry["key"]: vote_kg_base_score_from_features(entry["features"])
        for entry in real_entries
    }
    ranked_keys = [
        key for key, _ in sorted(base_scores.items(), key=lambda item: item[1], reverse=True)
    ]
    rank_by_key = {key: rank for rank, key in enumerate(ranked_keys, start=1)}
    sorted_scores = [base_scores[key] for key in ranked_keys]
    top_score = sorted_scores[0] if sorted_scores else 0.0

    for entry in entries:
        if entry.get("is_no_answer"):
            competition_features = [0.0] * 8
        else:
            key = entry["key"]
            rank = rank_by_key.get(key, len(ranked_keys) + 1)
            score = base_scores.get(key, 0.0)
            next_score = sorted_scores[rank] if rank < len(sorted_scores) else 0.0
            ranks = list(entry.get("view_ranks", {}).values())
            min_rank = min(ranks) if ranks else 999.0
            mean_rank = float(np.mean(ranks)) if ranks else 999.0
            spread = (max(ranks) - min(ranks)) if len(ranks) >= 2 else 0.0
            competition_features = [
                score,
                1.0 / max(rank, 1),
                top_score - score,
                score - next_score,
                1.0 / max(min_rank, 1.0),
                1.0 / max(mean_rank, 1.0),
                float(min_rank <= 1.0),
                min(float(spread), 16.0) / 16.0,
            ]
        entry["features"].extend(competition_features)
    return entries, views_pred_values


def load_multiview_context(args):
    view_specs = parse_view_specs(args.view)
    if not view_specs:
        raise ValueError("No --view specs provided.")
    view_weights = {item["name"]: item["weight"] for item in view_specs}

    view_records = {}
    for item in view_specs:
        view_records[item["name"]] = load_predictions(item["path"], filter_view=item.get("filter_view"))
        print(f"loaded {item['name']}: {len(view_records[item['name']])} records")

    primary_name = view_specs[0]["name"]
    primary_records = view_records[primary_name]
    retrieval = torch_load(args.retrieval)
    analysis_path = resolve_analysis_path(args.dataset, args.split, args.analysis_path)
    analysis_by_id = load_analysis_by_id(analysis_path)
    if analysis_path:
        print(f"Using question analysis: {analysis_path}")

    return view_specs, view_weights, view_records, primary_records, retrieval, analysis_by_id


def collect_sample_entries(sample_id, ref_record, view_specs, view_weights, view_records, retrieval, analysis_by_id, args):
    sample = retrieval.get(sample_id)
    if sample is None:
        return None, None, None

    sample_for_scoring = dict(sample)
    sample_for_scoring.setdefault("question", ref_record.get("question", ""))
    analysis = analysis_by_id.get(sample_id, {})
    if not hasattr(args, "_candidate_score_config"):
        args._candidate_score_config = load_candidate_score_config(
            getattr(args, "candidate_score_config", None)
        )
        if args._candidate_score_config:
            print(
                "Using frozen candidate KG score config: "
                f"{getattr(args, 'candidate_score_config', None)}"
            )
    kg_candidates = collect_candidates(
        sample_for_scoring,
        k=args.k,
        analysis=analysis,
        max_hops=args.max_hops,
        keep_topics=args.keep_topics,
        score_config=args._candidate_score_config,
    )

    record_per_view = {}
    for view in view_specs:
        record = view_records.get(view["name"], {}).get(sample_id)
        if record is not None:
            record_per_view[view["name"]] = record

    entries, _ = build_candidate_entries(
        record_per_view,
        view_weights,
        kg_candidates,
        include_no_answer=args.include_no_answer_candidate,
    )
    entries = apply_feature_profile(
        entries,
        getattr(args, "feature_profile", "full"),
    )
    return sample_for_scoring, record_per_view, entries


def make_labeled_rows(args):
    view_specs, view_weights, view_records, primary_records, retrieval, analysis_by_id = load_multiview_context(args)
    rows = []
    sample_count = 0
    skipped_no_candidate = 0
    skipped_no_retrieval = 0
    pos_count = 0
    neg_count = 0

    for sample_id, ref_record in primary_records.items():
        _, _, entries = collect_sample_entries(
            sample_id, ref_record, view_specs, view_weights, view_records, retrieval, analysis_by_id, args
        )
        if entries is None:
            skipped_no_retrieval += 1
            continue
        if not entries:
            skipped_no_candidate += 1
            continue

        answers = normalize_answers(ref_record)
        question = ref_record.get("question", "")
        sample_rows = []
        for entry in entries:
            label = 0
            if not entry["is_no_answer"] and match_answer_value(entry["answer"], answers, question=question):
                label = 1
            sample_rows.append({
                "id": sample_id,
                "answer": entry["answer"],
                "features": entry["features"],
                "label": label,
                "view_origins": entry["view_origins"],
            })
            pos_count += label
            neg_count += int(label == 0)

        if sample_rows:
            rows.extend(sample_rows)
            sample_count += 1

    stats = {
        "samples": sample_count,
        "rows": len(rows),
        "positives": pos_count,
        "negatives": neg_count,
        "skipped_no_candidate": skipped_no_candidate,
        "skipped_no_retrieval": skipped_no_retrieval,
    }
    return rows, stats
