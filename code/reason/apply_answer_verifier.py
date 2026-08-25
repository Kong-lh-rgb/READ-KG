import argparse
import csv
import json
import math
import os
import re
import string
from collections import defaultdict, deque

import numpy as np
import torch
from tqdm import tqdm

try:
    from rule_score_utils import candidate_score
except ModuleNotFoundError:
    from .rule_score_utils import candidate_score


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "was", "were", "what", "when", "where", "which", "who",
    "with", "used", "uses", "use", "type", "name", "named", "did", "does",
    "had", "was", "were", "also",
}

NO_ANSWER_MARKERS = (
    "ans: not available",
    "ans: no information available",
)


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_text(text):
    text = str(text or "").lower().replace("_", " ").replace(".", " ")
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\b(<pad>)\b", " ", text)
    return " ".join(text.split())


def tokenize_text(text):
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower().replace("_", " "))
    return {token for token in tokens if token not in STOPWORDS and len(token) > 1}


def flatten_values(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        output = []
        for item in value.values():
            output.extend(flatten_values(item))
        return output
    if isinstance(value, list):
        output = []
        for item in value:
            output.extend(flatten_values(item))
        return output
    return [str(value)]


def overlap_score(left, right):
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap == 0:
        return 0.0
    return overlap / max(1, min(len(left), len(right)))


def relation_tokens(relation):
    return tokenize_text(str(relation).replace(".", " ").replace("_", " "))


def resolve_analysis_path(dataset, split, analysis_path=None):
    disabled = os.environ.get("READKG_DISABLE_QUESTION_ANALYSIS", "")
    if disabled.strip().lower() in {"1", "true", "yes", "on"}:
        return None
    if analysis_path:
        return analysis_path

    analysis_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "retrieve",
        "data_files",
        dataset,
        "question_analysis",
    ))
    candidates = [
        os.path.join(analysis_dir, f"{split}.hybrid_qwen_llama.jsonl"),
        os.path.join(analysis_dir, f"{split}.llama31_8b.jsonl"),
        os.path.join(analysis_dir, f"{split}.qwen35flash.jsonl"),
        os.path.join(analysis_dir, f"{split}.jsonl"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def load_analysis_by_id(path):
    if not path:
        return {}
    by_id = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record.get("id")
            if sample_id:
                by_id[sample_id] = record
    return by_id


def analysis_terms(analysis):
    if not analysis:
        return {
            "answer_type": set(),
            "steps": set(),
            "relations": set(),
            "constraints": set(),
            "all": set(),
        }
    answer_type = tokenize_text(" ".join(flatten_values(analysis.get("answer_type", {}))))
    steps = set()
    relations = set()
    constraints = set()
    for step in analysis.get("reasoning_steps", []) or []:
        if not isinstance(step, dict):
            continue
        steps |= tokenize_text(" ".join(flatten_values(step)))
        relations |= tokenize_text(" ".join(flatten_values([
            step.get("relation"),
            step.get("predicate"),
            step.get("intent"),
            step.get("target_type"),
        ])))
    for relation in analysis.get("relation_intents", []) or []:
        relations |= tokenize_text(" ".join(flatten_values(relation)))
    for constraint in analysis.get("constraints", []) or []:
        constraints |= tokenize_text(" ".join(flatten_values(constraint)))
    all_terms = answer_type | steps | relations | constraints
    return {
        "answer_type": answer_type,
        "steps": steps,
        "relations": relations,
        "constraints": constraints,
        "all": all_terms,
    }


def strip_triple(triple):
    return tuple(triple[:3])


def normalize_score(score):
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 0.0
    if 0.0 <= score <= 1.0:
        return score
    if score >= 20:
        return 1.0
    if score <= -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-score))


def triple_score(triple):
    if len(triple) < 4:
        return 0.0
    return normalize_score(triple[3])


def ranked_triples(sample, k):
    output = []
    seen = set()
    for triple in sample.get("scored_triples", [])[:k]:
        if len(triple) < 3:
            continue
        bare = strip_triple(triple)
        if bare in seen:
            continue
        seen.add(bare)
        output.append((bare, triple_score(triple)))
    return output


def build_graph(triples):
    adjacency = defaultdict(list)
    for rank, ((head, relation, tail), score) in enumerate(triples, start=1):
        edge = {
            "head": head,
            "relation": relation,
            "tail": tail,
            "score": score,
            "rank": rank,
        }
        adjacency[head].append((tail, edge))
        adjacency[tail].append((head, edge))
    return adjacency


def shortest_distances(adjacency, sources, max_hops=4):
    sources = [source for source in sources if source in adjacency]
    if not sources:
        return {}
    distances = {source: 0 for source in sources}
    queue = deque(sources)
    while queue:
        node = queue.popleft()
        if distances[node] >= max_hops:
            continue
        for nxt, _ in adjacency[node]:
            if nxt in distances:
                continue
            distances[nxt] = distances[node] + 1
            queue.append(nxt)
    return distances


def collect_candidates(sample, k, analysis=None, max_hops=4, keep_topics=False, score_config=None):
    triples = ranked_triples(sample, k)
    topic_entities = set(sample.get("q_entity_in_graph", []))
    adjacency = build_graph(triples)
    distances = shortest_distances(adjacency, topic_entities, max_hops=max_hops)
    terms = analysis_terms(analysis)
    question_terms = tokenize_text(sample.get("question", ""))

    candidate_stats = {}
    for rank, ((head, relation, tail), score) in enumerate(triples, start=1):
        rank_score = 1.0 - ((rank - 1) / max(len(triples) - 1, 1))
        rel_terms = relation_tokens(relation)
        for entity, neighbor in ((head, tail), (tail, head)):
            if not keep_topics and entity in topic_entities:
                continue
            stat = candidate_stats.setdefault(entity, {
                "entity": entity,
                "first_triple_rank": rank,
                "incident_count": 0,
                "max_triple_score": 0.0,
                "mean_triple_score_sum": 0.0,
                "max_rank_score": 0.0,
                "topic_distance": math.inf,
                "relation_terms": set(),
                "neighbor_terms": set(),
            })
            stat["first_triple_rank"] = min(stat["first_triple_rank"], rank)
            stat["incident_count"] += 1
            stat["max_triple_score"] = max(stat["max_triple_score"], score)
            stat["mean_triple_score_sum"] += score
            stat["max_rank_score"] = max(stat["max_rank_score"], rank_score)
            stat["topic_distance"] = min(stat["topic_distance"], distances.get(entity, math.inf))
            stat["relation_terms"] |= rel_terms
            stat["neighbor_terms"] |= tokenize_text(neighbor)

    for stat in candidate_stats.values():
        entity_terms = tokenize_text(stat["entity"])
        support_terms = stat["relation_terms"] | stat["neighbor_terms"] | entity_terms
        stat["mean_triple_score"] = stat["mean_triple_score_sum"] / max(stat["incident_count"], 1)
        stat["path_score"] = 0.0 if math.isinf(stat["topic_distance"]) else 1.0 / max(stat["topic_distance"], 1)
        stat["answer_type_match"] = overlap_score(support_terms, terms["answer_type"])
        stat["relation_match"] = overlap_score(stat["relation_terms"], terms["relations"])
        stat["constraint_match"] = overlap_score(support_terms, terms["constraints"])
        stat["step_match"] = overlap_score(support_terms, terms["steps"])
        stat["question_match"] = overlap_score(support_terms, question_terms)
        stat["support_count_score"] = min(stat["incident_count"], 8) / 8.0
        stat["score"] = candidate_score(stat, score_config)
        stat["debug"] = (
            f"rank={stat['first_triple_rank']};score={stat['score']:.3f};"
            f"path={stat['path_score']:.3f};rel={stat['relation_match']:.3f};"
            f"constraint={stat['constraint_match']:.3f};type={stat['answer_type_match']:.3f}"
        )
        del stat["mean_triple_score_sum"]
        del stat["relation_terms"]
        del stat["neighbor_terms"]

    return sorted(
        candidate_stats.values(),
        key=lambda item: (item["score"], item["max_rank_score"], item["max_triple_score"]),
        reverse=True,
    )


def prediction_has_no_answer(prediction):
    text = str(prediction or "").lower()
    if "ans:" not in text:
        return True
    answer_lines = [line.lower() for line in str(prediction or "").splitlines() if "ans:" in line.lower()]
    return not any(not any(marker in line for marker in NO_ANSWER_MARKERS) for line in answer_lines)


def extract_pred_values(prediction):
    values = []
    seen = set()
    for line in str(prediction or "").splitlines():
        if "ans:" not in line:
            continue
        lower = line.lower()
        if any(marker in lower for marker in NO_ANSWER_MARKERS):
            continue
        after = line.split("ans:", 1)[-1]
        parts = re.split(r"\s*\|\|\s*|\s*;\s*", after)
        for part in parts:
            value = part.strip(" -:\t")
            if not value:
                continue
            norm = normalize_text(value)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            values.append(value)
    return values


def match_text(left, right):
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return False
    return left_norm in right_norm or right_norm in left_norm


def map_prediction_to_candidate(pred_value, candidates):
    best = None
    best_score = -1.0
    pred_norm = normalize_text(pred_value)
    if not pred_norm:
        return None
    for candidate in candidates:
        entity_norm = normalize_text(candidate["entity"])
        if not entity_norm:
            continue
        if pred_norm in entity_norm or entity_norm in pred_norm:
            score = min(len(pred_norm), len(entity_norm)) / max(len(pred_norm), len(entity_norm))
            score += candidate["score"] * 0.1
            if score > best_score:
                best = candidate
                best_score = score
    return best


def likely_multi_answer(question):
    q = str(question or "").lower()
    plural_markers = [
        "countries", "movies", "films", "teams", "people", "states", "languages",
        "cities", "books", "songs", "albums", "actors", "players", "places",
        "rivers", "mountains", "events", "schools", "organizations",
    ]
    return (
        " all " in f" {q} " or
        " list " in f" {q} " or
        (("which" in q or "what" in q) and any(marker in q for marker in plural_markers))
    )


def double_check_enabled(question):
    keywords = [
        "when",
        "what year",
        "which year",
        "where",
        "sport",
        "what countr",
        "language",
        "nba finals",
        "world series",
    ]
    question = str(question or "").lower()
    return any(keyword in question for keyword in keywords)


def normalize_answers(record):
    answers = record.get("ground_truth", [])
    if not isinstance(answers, list):
        answers = [answers]
    answers = sorted(set(answers), key=len, reverse=True)
    question = record.get("question", "")
    if "when" in question.lower() or "what year" in question.lower():
        answers = [
            answer.split("-")[0]
            if "-" in answer and answer.split("-")[0].isdigit()
            else answer
            for answer in answers
        ]
    return answers


def eval_record(record):
    pred_values = extract_pred_values(record.get("prediction", ""))
    answers = normalize_answers(record)
    double_check = double_check_enabled(record.get("question", ""))
    remaining_preds = sorted(pred_values, key=len, reverse=True)
    matched_answers = set()
    matched_count = 0
    for answer_idx, answer in enumerate(answers):
        for pred in list(remaining_preds):
            if match_text(pred, answer) or (double_check and match_text(answer, pred)):
                matched_answers.add(answer_idx)
                matched_count += 1
                remaining_preds.remove(pred)
                break
    precision = matched_count / len(pred_values) if pred_values else 0.0
    recall = len(matched_answers) / len(answers) if answers else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    hit = 0
    if pred_values:
        hit = int(any(match_text(pred_values[0], answer) or (double_check and match_text(answer, pred_values[0])) for answer in answers))
    return {
        "hit": hit,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "num_preds": len(pred_values),
        "num_answers": len(answers),
        "no_answer": int(len(pred_values) == 0 or prediction_has_no_answer(record.get("prediction", ""))),
    }


def metric_summary(records):
    if not records:
        return {}
    evals = [eval_record(record) for record in records]
    return {
        "hit": float(np.mean([item["hit"] for item in evals])),
        "f1": float(np.mean([item["f1"] for item in evals])),
        "precision": float(np.mean([item["precision"] for item in evals])),
        "recall": float(np.mean([item["recall"] for item in evals])),
        "no_answer": float(np.mean([item["no_answer"] for item in evals])),
    }


def dedupe_answers(answers):
    output = []
    seen = set()
    for answer in answers:
        norm = normalize_text(answer)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        output.append(answer)
    return output


def choose_verified_answers(record, candidates, args):
    question = record.get("question", "")
    pred_values = extract_pred_values(record.get("prediction", ""))
    no_answer = prediction_has_no_answer(record.get("prediction", ""))
    pred_matches = [map_prediction_to_candidate(pred, candidates) for pred in pred_values]
    top_candidates = [
        candidate for candidate in candidates
        if candidate["first_triple_rank"] <= args.max_candidate_rank
        and candidate["path_score"] >= args.min_path_score
    ]
    top_candidate = top_candidates[0] if top_candidates else None

    action = "unchanged"
    verified = list(pred_values)

    if args.mode in {"fill", "full", "conservative"} and no_answer and top_candidate:
        if top_candidate["score"] >= args.fill_threshold:
            verified = [top_candidate["entity"]]
            action = "fill_no_answer"
            if likely_multi_answer(question):
                for candidate in top_candidates[1:args.fill_top_n]:
                    if candidate["score"] < max(args.expand_threshold, top_candidate["score"] - args.fill_margin):
                        continue
                    verified.append(candidate["entity"])

    if args.mode in {"expand", "full", "conservative"} and verified and likely_multi_answer(question):
        existing_norm = {normalize_text(answer) for answer in verified}
        anchor_score = 0.0
        for match in pred_matches:
            if match is not None:
                anchor_score = max(anchor_score, match["score"])
        if anchor_score == 0.0 and top_candidate:
            anchor_score = top_candidate["score"]
        added = 0
        for candidate in top_candidates:
            if added >= args.max_expand_add:
                break
            if candidate["score"] < args.expand_threshold:
                continue
            if anchor_score and candidate["score"] < anchor_score - args.expand_margin:
                continue
            norm = normalize_text(candidate["entity"])
            if norm in existing_norm:
                continue
            verified.append(candidate["entity"])
            existing_norm.add(norm)
            added += 1
        if added > 0 and action == "unchanged":
            action = "expand_multi_answer"

    if args.mode == "full" and verified and top_candidate:
        weak_predictions = []
        strong_predictions = []
        for pred, match in zip(pred_values, pred_matches):
            if match is None:
                strong_predictions.append(pred)
            elif match["score"] < args.keep_threshold:
                weak_predictions.append(pred)
            else:
                strong_predictions.append(pred)
        if weak_predictions and top_candidate["score"] >= args.replace_threshold:
            existing_norm = {normalize_text(answer) for answer in strong_predictions}
            if normalize_text(top_candidate["entity"]) not in existing_norm:
                strong_predictions.insert(0, top_candidate["entity"])
            verified = strong_predictions
            action = "replace_weak"

    verified = dedupe_answers(verified)
    if not verified and not no_answer:
        verified = pred_values
    if not verified:
        return [], "unchanged_no_answer", top_candidates[: args.debug_top_n]
    return verified, action, top_candidates[: args.debug_top_n]


def format_prediction(answers):
    if not answers:
        return "ans: not available"
    return "\n".join(f"ans: {answer}" for answer in answers)


def load_predictions(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_detail(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    predictions = load_predictions(args.pred)
    retrieval = torch_load(args.retrieval)
    analysis_path = resolve_analysis_path(args.dataset, args.split, args.analysis_path)
    analysis_by_id = load_analysis_by_id(analysis_path)
    if analysis_path:
        print(f"Using question analysis: {analysis_path}")
    else:
        print("Question analysis not found; analysis-based verifier features are disabled.")

    corrected = []
    detail_rows = []
    action_counts = defaultdict(int)
    missing_retrieval = 0

    for record in tqdm(predictions, desc="verify answers"):
        sample_id = record.get("id")
        sample = retrieval.get(sample_id)
        if sample is None:
            missing_retrieval += 1
            corrected.append(record)
            continue
        analysis = analysis_by_id.get(sample_id, {})
        sample_for_scoring = dict(sample)
        sample_for_scoring.setdefault("question", record.get("question", ""))
        candidates = collect_candidates(
            sample_for_scoring,
            k=args.k,
            analysis=analysis,
            max_hops=args.max_hops,
            keep_topics=args.keep_topics,
        )
        old_eval = eval_record(record)
        answers, action, debug_candidates = choose_verified_answers(record, candidates, args)
        new_record = dict(record)
        new_record["raw_prediction_before_verifier"] = record.get("prediction", "")
        new_record["prediction"] = format_prediction(answers)
        new_eval = eval_record(new_record)
        corrected.append(new_record)
        action_counts[action] += 1
        detail_rows.append({
            "id": sample_id,
            "question": record.get("question", ""),
            "action": action,
            "old_prediction": " || ".join(extract_pred_values(record.get("prediction", ""))),
            "new_prediction": " || ".join(answers),
            "old_hit": old_eval["hit"],
            "new_hit": new_eval["hit"],
            "old_f1": old_eval["f1"],
            "new_f1": new_eval["f1"],
            "old_no_answer": old_eval["no_answer"],
            "new_no_answer": new_eval["no_answer"],
            "top_candidates": " || ".join(
                f"{candidate['entity']}[{candidate['debug']}]"
                for candidate in debug_candidates
            ),
        })

    write_jsonl(args.output_path, corrected)
    detail_path = args.detail_path
    if detail_path is None:
        detail_path = args.output_path.replace(".jsonl", "_verifier_detail.csv")
    write_detail(detail_path, detail_rows)

    before = metric_summary(predictions)
    after = metric_summary(corrected)
    print(f"Loaded predictions: {len(predictions)}")
    print(f"Missing retrieval samples: {missing_retrieval}")
    print("Action counts:", dict(sorted(action_counts.items())))
    print("Before:", json.dumps(before, ensure_ascii=False))
    print("After: ", json.dumps(after, ensure_ascii=False))
    print(f"Saved corrected predictions to: {args.output_path}")
    print(f"Saved verifier detail to: {detail_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Apply evidence-grounded answer verification to KGQA predictions.")
    parser.add_argument("--pred", required=True, help="Input prediction jsonl.")
    parser.add_argument("--retrieval", required=True, help="Retrieval result .pth.")
    parser.add_argument("--dataset", required=True, choices=["cwq", "webqsp"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--analysis_path", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--detail_path", default=None)
    parser.add_argument("--mode", default="conservative", choices=["fill", "expand", "conservative", "full"])
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--max_hops", type=int, default=4)
    parser.add_argument("--keep_topics", action="store_true")
    parser.add_argument("--max_candidate_rank", type=int, default=60)
    parser.add_argument("--min_path_score", type=float, default=0.25)
    parser.add_argument("--fill_threshold", type=float, default=0.62)
    parser.add_argument("--fill_top_n", type=int, default=5)
    parser.add_argument("--fill_margin", type=float, default=0.06)
    parser.add_argument("--expand_threshold", type=float, default=0.70)
    parser.add_argument("--expand_margin", type=float, default=0.08)
    parser.add_argument("--max_expand_add", type=int, default=5)
    parser.add_argument("--keep_threshold", type=float, default=0.45)
    parser.add_argument("--replace_threshold", type=float, default=0.68)
    parser.add_argument("--debug_top_n", type=int, default=8)
    args = parser.parse_args()
    main(args)
