"""Apply a learned answer-level multi-view aggregator."""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn as nn
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
    torch_load,
    write_detail,
    write_jsonl,
)
from multiview_learning_utils import (
    FEATURE_NAMES,
    collect_sample_entries,
    expected_answer_count,
    load_multiview_context,
)


class LinearAggregator(nn.Module):
    def __init__(self, input_dim, residual_base_idx=None):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.residual_base_idx = residual_base_idx

    def forward(self, features):
        if self.residual_base_idx is not None:
            delta_features = features.clone()
            delta_features[:, self.residual_base_idx] = 0.0
            logits = self.linear(delta_features).squeeze(-1) + features[:, self.residual_base_idx]
        else:
            logits = self.linear(features).squeeze(-1)
        return logits


def load_model(path, device):
    payload = torch_load(path)
    feature_names = payload.get("feature_names", FEATURE_NAMES)
    residual_base_feature = payload.get("residual_base_feature")
    residual_base_idx = None
    if residual_base_feature:
        if residual_base_feature not in feature_names:
            raise ValueError(
                f"Checkpoint residual_base_feature={residual_base_feature!r} "
                f"not found in feature_names."
            )
        residual_base_idx = feature_names.index(residual_base_feature)
    model = LinearAggregator(len(feature_names), residual_base_idx=residual_base_idx).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, feature_names, payload


def align_features(features, input_dim):
    if len(features) == input_dim:
        return features
    if len(features) > input_dim:
        return features[:input_dim]
    return list(features) + [0.0] * (input_dim - len(features))


def score_entries(model, entries, device):
    if not entries:
        return []
    input_dim = model.linear.in_features
    features = torch.tensor(
        [align_features(entry["features"], input_dim) for entry in entries],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        logits = model(features).detach().cpu().numpy().tolist()
    output = []
    for logit, entry in zip(logits, entries):
        enriched = dict(entry)
        enriched["score"] = float(logit)
        output.append(enriched)
    output.sort(key=lambda item: item["score"], reverse=True)
    return output


def select_answers(scored, ref_record, args, views_pred_values=None):
    real = [item for item in scored if not item.get("is_no_answer")]
    if not real:
        return []

    if likely_multi_answer(ref_record.get("question", "")):
        limit = max(args.multi_min_n, expected_answer_count(views_pred_values or []))
        limit = min(limit, args.multi_max_n)
        continuation_min_score = args.multi_min_score
    else:
        limit = 1
        continuation_min_score = args.min_score

    selected = []
    top_score = real[0]["score"]
    if top_score < args.top1_min_score:
        return []
    for item in real:
        if len(selected) >= limit:
            break
        if len(selected) > 0 and item["score"] < continuation_min_score:
            break
        if len(selected) > 0 and args.score_gap > 0 and (top_score - item["score"]) > args.score_gap:
            break
        selected.append(item["answer"])
    return dedupe_answers(selected)


def main(args):
    device = torch.device(args.device)
    model, feature_names, payload = load_model(args.model, device)
    print(f"Loaded model from: {args.model}")
    print("Train info:", json.dumps(payload.get("train_info", {}), ensure_ascii=False))

    view_specs, view_weights, view_records, primary_records, retrieval, analysis_by_id = load_multiview_context(args)
    corrected = []
    detail_rows = []
    action_counts = defaultdict(int)
    missing_retrieval = 0

    for sample_id, ref_record in tqdm(primary_records.items(), desc="learned aggregate"):
        _, record_per_view, entries = collect_sample_entries(
            sample_id, ref_record, view_specs, view_weights, view_records, retrieval, analysis_by_id, args
        )
        if entries is None:
            missing_retrieval += 1
            corrected.append(ref_record)
            continue

        scored = score_entries(model, entries, device)
        views_pred_values = [
            extract_pred_values(record.get("prediction", ""))
            for record in (record_per_view or {}).values()
        ]
        answers = select_answers(scored, ref_record, args, views_pred_values)
        action = "learned_aggregated" if answers else "no_answer"
        if not answers and args.fallback_primary:
            primary_pred = extract_pred_values(ref_record.get("prediction", ""))
            if primary_pred:
                answers = primary_pred
                action = "fallback_primary"

        new_record = dict(ref_record)
        new_record["raw_prediction_before_learned_aggregator"] = ref_record.get("prediction", "")
        new_record["prediction"] = format_prediction(answers) if answers else "ans: not available"
        corrected.append(new_record)
        action_counts[action] += 1

        old_eval = eval_record(ref_record)
        new_eval = eval_record(new_record)
        detail_rows.append({
            "id": sample_id,
            "question": ref_record.get("question", ""),
            "action": action,
            "old_prediction": " || ".join(extract_pred_values(ref_record.get("prediction", ""))),
            "new_prediction": " || ".join(answers),
            "old_hit": old_eval["hit"],
            "new_hit": new_eval["hit"],
            "old_f1": old_eval["f1"],
            "new_f1": new_eval["f1"],
            "top_candidates": " || ".join(
                f"{item['answer']}[score={item['score']:.3f};views={'+'.join(item['view_origins'])};kg={item['features'][8]:.2f}]"
                for item in scored[:args.debug_top_n]
            ),
        })

    write_jsonl(args.output_path, corrected)
    detail_path = args.detail_path or args.output_path.replace(".jsonl", "_learned_detail.csv")
    write_detail(detail_path, detail_rows)

    print(f"Loaded primary view: {len(primary_records)} records")
    print(f"Missing retrieval samples: {missing_retrieval}")
    print("Action counts:", dict(sorted(action_counts.items())))

    print("\n=== Per-view standalone metrics ===")
    for view in view_specs:
        records = list(view_records[view["name"]].values())
        ms = metric_summary(records)
        print(f"  {view['name']:<12} hit={ms.get('hit', 0):.4f} f1={ms.get('f1', 0):.4f} "
              f"prec={ms.get('precision', 0):.4f} recall={ms.get('recall', 0):.4f} "
              f"no_ans={ms.get('no_answer', 0):.4f}")

    print("\n=== Learned Aggregator output ===")
    after = metric_summary(corrected)
    print(f"  learned     hit={after.get('hit', 0):.4f} f1={after.get('f1', 0):.4f} "
          f"prec={after.get('precision', 0):.4f} recall={after.get('recall', 0):.4f} "
          f"no_ans={after.get('no_answer', 0):.4f}")
    print(f"\nSaved learned predictions to: {args.output_path}")
    print(f"Saved detail to: {detail_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Apply a learned READ-KG answer aggregator.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--view", action="append", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--dataset", required=True, choices=["cwq", "webqsp"])
    parser.add_argument("--split", default="val")
    parser.add_argument("--analysis_path", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--detail_path", default=None)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--max_hops", type=int, default=4)
    parser.add_argument("--keep_topics", action="store_true")
    parser.add_argument("--include_no_answer_candidate", action="store_true")
    parser.add_argument("--default_multi_n", type=int, default=2)
    parser.add_argument("--multi_min_n", type=int, default=2)
    parser.add_argument("--multi_max_n", type=int, default=8)
    parser.add_argument("--min_score", type=float, default=-100.0)
    parser.add_argument("--multi_min_score", type=float, default=-100.0,
                        help="Minimum score for additional answers in multi-answer questions.")
    parser.add_argument("--top1_min_score", type=float, default=-100.0,
                        help="Abstain if top-1 score is below this threshold.")
    parser.add_argument("--score_gap", type=float, default=0.0)
    parser.add_argument("--fallback_primary", action="store_true")
    parser.add_argument("--debug_top_n", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args)
