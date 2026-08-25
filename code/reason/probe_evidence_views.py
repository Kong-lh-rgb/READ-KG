#!/usr/bin/env python3
"""Build and optionally run the stable READ-KG three-view Reader prompts.

This is the paper reproduction path for Stage-B evidence layouts. It keeps the
three views used by the answer-level aggregator:

    pathfirst, minimal, candidate
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

import torch
from tqdm import tqdm

from apply_answer_verifier import collect_candidates, eval_record
from llm_utils import llm_init, llm_inf_all_batch
from main import get_defined_prompts
from preprocess.evidence_organizer import ReasoningPathEvidenceOrganizer, strip_score
from preprocess.prepare_data import get_data
from preprocess.prepare_prompts import triplet_to_str


SUPPORTED_VIEWS = {"pathfirst", "minimal", "candidate", "flat"}


def torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_answers(record: dict) -> list[str]:
    answers = record.get("ground_truth", [])
    if not isinstance(answers, list):
        answers = [answers]
    return [str(answer).strip() for answer in answers if str(answer).strip()]


def scored_to_bare(scored_triplets: list, k: int) -> list[tuple]:
    output = []
    seen = set()
    for triplet in scored_triplets[:k]:
        if len(triplet) < 3:
            continue
        bare = strip_score(triplet)
        if bare in seen:
            continue
        seen.add(bare)
        output.append(bare)
    return output


def pathfirst_prompt(scored_triplets: list, k: int, analysis: dict | None) -> str:
    return ReasoningPathEvidenceOrganizer.from_mode(f"pathfirst_{k}").format_prompt(
        scored_triplets,
        analysis=analysis or {},
    )


def triples_for_candidate(
    scored_triplets: list,
    candidate: dict,
    max_support: int,
) -> list[tuple]:
    entity = candidate["entity"]
    support = []
    seen = set()
    for triplet in scored_triplets:
        if len(triplet) < 3:
            continue
        bare = strip_score(triplet)
        if bare in seen:
            continue
        if bare[0] == entity or bare[2] == entity:
            support.append(bare)
            seen.add(bare)
        if len(support) >= max_support:
            break
    return support


def _scoring_sample(sample: dict, retrieval_sample: dict | None) -> dict:
    scoring_sample = dict(retrieval_sample or {})
    scoring_sample.setdefault("question", sample.get("question", ""))
    scoring_sample.setdefault("scored_triples", sample.get("scored_triplets", []))
    scoring_sample.setdefault("q_entity_in_graph", sample.get("q_entity_in_graph", []))
    return scoring_sample


def candidate_prompt(
    sample: dict,
    retrieval_sample: dict | None,
    k: int,
    analysis: dict | None,
    top_n: int,
    support_per_candidate: int,
) -> str:
    candidates = collect_candidates(
        _scoring_sample(sample, retrieval_sample),
        k=k,
        analysis=analysis or {},
        max_hops=4,
        keep_topics=False,
    )[:top_n]

    lines = ["Triplets:", "Candidate-centered evidence:"]
    selected = []
    selected_set = set()
    for idx, candidate in enumerate(candidates, start=1):
        lines.append(
            f"Candidate {idx}: {candidate['entity']} "
            f"(utility={candidate['score']:.3f}, path={candidate['path_score']:.3f}, "
            f"constraint={candidate['constraint_match']:.3f})"
        )
        support = triples_for_candidate(
            sample.get("scored_triplets", []),
            candidate,
            support_per_candidate,
        )
        if not support:
            continue
        lines.append("Supporting triplets:")
        for triplet in support:
            lines.append(triplet_to_str(triplet))
            if triplet not in selected_set:
                selected.append(triplet)
                selected_set.add(triplet)

    residual = []
    for triplet in scored_to_bare(sample.get("scored_triplets", []), k):
        if len(selected) + len(residual) >= k:
            break
        if triplet not in selected_set:
            residual.append(triplet)
            selected_set.add(triplet)
    if residual:
        lines.append("Other retrieved triplets:")
        for triplet in residual:
            lines.append(triplet_to_str(triplet))
    return "\n".join(lines)


def minimal_prompt(
    sample: dict,
    retrieval_sample: dict | None,
    k: int,
    analysis: dict | None,
    top_n: int,
    support_per_candidate: int,
) -> str:
    candidates = collect_candidates(
        _scoring_sample(sample, retrieval_sample),
        k=k,
        analysis=analysis or {},
        max_hops=4,
        keep_topics=False,
    )[:top_n]
    organized = ReasoningPathEvidenceOrganizer.from_mode(f"pathfirst_{k}").organize(
        sample.get("scored_triplets", []),
        analysis=analysis or {},
    )

    selected = []
    selected_set = set()
    for block in organized.path_blocks[:8]:
        for triplet in block:
            if triplet not in selected_set:
                selected.append(triplet)
                selected_set.add(triplet)
    for candidate in candidates:
        support = triples_for_candidate(
            sample.get("scored_triplets", []),
            candidate,
            support_per_candidate,
        )
        for triplet in support:
            if triplet not in selected_set:
                selected.append(triplet)
                selected_set.add(triplet)
    for triplet in organized.constraint_triplets:
        if triplet not in selected_set:
            selected.append(triplet)
            selected_set.add(triplet)
    selected = selected[: min(k, 48)]

    lines = ["Triplets:", "Minimal evidence set:"]
    for triplet in selected:
        lines.append(triplet_to_str(triplet))
    return "\n".join(lines)


def build_triplet_prompt(
    view: str,
    sample: dict,
    retrieval_sample: dict | None,
    args: argparse.Namespace,
) -> str:
    analysis = sample.get("question_analysis", {}) or {}
    if view == "flat":
        lines = ["Triplets:"]
        seen = set()
        for triplet in sample.get("scored_triplets", []):
            if len(triplet) < 3:
                continue
            bare = strip_score(triplet)
            if bare in seen:
                continue
            seen.add(bare)
            lines.append(triplet_to_str(bare))
            if len(seen) >= args.k:
                break
        return "\n".join(lines)
    if view == "pathfirst":
        return pathfirst_prompt(sample.get("scored_triplets", []), args.k, analysis)
    if view == "candidate":
        return candidate_prompt(
            sample,
            retrieval_sample,
            args.k,
            analysis,
            args.candidate_top_n,
            args.support_per_candidate,
        )
    if view == "minimal":
        return minimal_prompt(
            sample,
            retrieval_sample,
            args.k,
            analysis,
            args.candidate_top_n,
            args.support_per_candidate,
        )
    raise ValueError(
        f"Unknown view: {view}. Supported views: {','.join(sorted(SUPPORTED_VIEWS))}"
    )


def build_prompt_record(
    sample: dict,
    retrieval_sample: dict | None,
    view: str,
    sys_prompt: str,
    cot_prompt: str,
    args: argparse.Namespace,
) -> dict:
    question_prompt = "Question:\n" + sample["question"]
    if question_prompt[-1] != "?":
        question_prompt += "?"
    triplet_prompt = build_triplet_prompt(view, sample, retrieval_sample, args)
    return {
        "sys_query": sys_prompt,
        "user_query": "\n\n".join([triplet_prompt, question_prompt]),
        "all_query": "\n\n".join([sys_prompt, triplet_prompt, question_prompt]),
        "cot_query": cot_prompt,
    }


def load_focus_ids(path: str | None, focus: str) -> list[str] | None:
    if not path:
        return None
    ids = []
    for record in load_jsonl(path):
        metrics = eval_record(record)
        if focus == "wrong" and metrics["hit"] == 1:
            continue
        if focus == "no_answer" and metrics["no_answer"] == 0:
            continue
        if focus == "partial" and not (metrics["hit"] == 1 and metrics["f1"] < 0.999):
            continue
        ids.append(record["id"])
    return ids


def select_samples(data: list[dict], args: argparse.Namespace) -> list[dict]:
    focus_ids = load_focus_ids(args.focus_pred, args.focus)
    if focus_ids is not None:
        focus_set = set(focus_ids)
        data = [record for record in data if record["id"] in focus_set]
    sample_seed = args.seed if args.sample_seed is None else args.sample_seed
    rng = random.Random(sample_seed)
    if args.sample_size and len(data) > args.sample_size:
        data = rng.sample(data, args.sample_size)
    return data


def summarize(records: list[dict], views: list[str]) -> list[dict]:
    by_view = defaultdict(list)
    by_id = defaultdict(list)
    for record in records:
        by_view[record["view"]].append(record)
        by_id[record["id"]].append(record)

    rows = []
    for view in views:
        evals = [eval_record(record) for record in by_view.get(view, [])]
        if not evals:
            continue
        rows.append({
            "view": view,
            "count": len(evals),
            "hit": sum(item["hit"] for item in evals) / len(evals),
            "f1": sum(item["f1"] for item in evals) / len(evals),
            "precision": sum(item["precision"] for item in evals) / len(evals),
            "recall": sum(item["recall"] for item in evals) / len(evals),
            "no_answer": sum(item["no_answer"] for item in evals) / len(evals),
        })

    base_view = views[0]
    oracle_items = []
    improve_over_base = 0
    disagreement = 0
    for sample_records in by_id.values():
        evals = [(record["view"], eval_record(record)) for record in sample_records]
        if len({item["hit"] for _, item in evals}) > 1:
            disagreement += 1
        base = next((item for view, item in evals if view == base_view), None)
        best_hit = max(item["hit"] for _, item in evals)
        best_f1 = max(item["f1"] for _, item in evals)
        if base is not None and (best_hit > base["hit"] or best_f1 > base["f1"]):
            improve_over_base += 1
        oracle_items.append({"hit": best_hit, "f1": best_f1})

    if oracle_items:
        rows.append({
            "view": "oracle_best_view",
            "count": len(oracle_items),
            "hit": sum(item["hit"] for item in oracle_items) / len(oracle_items),
            "f1": sum(item["f1"] for item in oracle_items) / len(oracle_items),
            "precision": "",
            "recall": "",
            "no_answer": "",
        })
        rows.append({
            "view": "view_diagnostics",
            "count": len(oracle_items),
            "hit": "",
            "f1": "",
            "precision": f"disagreement={disagreement / len(oracle_items):.4f}",
            "recall": f"improve_over_{base_view}={improve_over_base / len(oracle_items):.4f}",
            "no_answer": "",
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stable READ-KG three-view evidence prompts."
    )
    parser.add_argument("-d", "--dataset", default="cwq")
    parser.add_argument("--split", default="test")
    parser.add_argument("-p", "--retrieval", required=True)
    parser.add_argument("--views", default="pathfirst,minimal,candidate")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=None,
        help="Seed for selecting a subset. Defaults to --seed.",
    )
    parser.add_argument(
        "--focus_pred",
        default=None,
        help="Existing prediction jsonl used to select hard samples.",
    )
    parser.add_argument(
        "--focus",
        choices=["all", "wrong", "no_answer", "partial"],
        default="all",
    )
    parser.add_argument("--candidate_top_n", type=int, default=8)
    parser.add_argument("--support_per_candidate", type=int, default=5)
    parser.add_argument("--llm_mode", default="sys_icl_dc")
    parser.add_argument(
        "-m",
        "--model_name",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_seq_len_to_capture", type=int, default=8192)
    parser.add_argument("--max_tokens", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--frequency_penalty", type=float, default=0.16)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--strict_entity_output", action="store_true")
    parser.add_argument("--conditional_retry_missing_ans", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--output_name", default="evidence_view_probe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    views = [view.strip() for view in args.views.split(",") if view.strip()]
    unsupported = sorted(set(views) - SUPPORTED_VIEWS)
    if unsupported:
        raise ValueError(
            f"Unsupported views after cleanup: {unsupported}. "
            f"Supported views: {sorted(SUPPORTED_VIEWS)}"
        )

    pred_scaffold = (
        f"./results/KGQA/{args.dataset}/RoG/{args.split}/"
        f"results_gen_rule_path_RoG-{args.dataset}_RoG_{args.split}_predictions_3_False_jsonl/"
        "predictions.jsonl"
    )
    data = get_data(
        args.dataset,
        pred_scaffold,
        args.retrieval,
        args.split,
        f"pathfirst_{args.k}",
    )
    data = select_samples(data, args)
    print(f"Selected {len(data)} samples; views={views}")

    sys_prompt, cot_prompt = get_defined_prompts("scored_100", args.model_name, args.llm_mode)
    if args.strict_entity_output:
        sys_prompt += (
            "\n\nOutput only the final answers. Follow these rules strictly: "
            "write exactly one distinct answer per line as ans: <answer>; "
            "use concise entity names or literal values; preserve the surface form when an answer appears in the triples; "
            "answers may be inferred when the question requires calculation or normalization; "
            "do not output explanations, complete sentences, identifiers, numbering, or duplicate answers; "
            "stop immediately after the last distinct answer."
        )
    output_records = []
    prompt_records = []
    reader_items = []

    for sample in tqdm(data, desc="probe evidence views"):
        for view in views:
            prompt_record = build_prompt_record(
                sample,
                None,
                view,
                sys_prompt,
                cot_prompt,
                args,
            )
            if args.strict_entity_output:
                prompt_record["user_query"] += (
                    "\n\nReturn only final answers: exactly one distinct ans: <answer> per line. "
                    "Use concise answer names or values; no explanation, numbering, or duplicates."
                )
            prompt_records.append({
                "id": sample["id"],
                "view": view,
                "question": sample["question"],
                "ground_truth": normalize_answers(sample),
                "user_query": prompt_record["user_query"],
            })
            if args.dry_run:
                continue
            reader_items.append(({
                "id": sample["id"],
                "view": view,
                "question": sample["question"],
                "ground_truth": normalize_answers(sample),
            }, {
                "sys_query": prompt_record["sys_query"],
                "user_query": prompt_record["user_query"],
                "cot_query": prompt_record["cot_query"],
            }))

    output_dir = Path("logs") / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "prompts.jsonl", prompt_records)
    if args.dry_run:
        print(f"Saved prompts to: {output_dir / 'prompts.jsonl'}")
        return

    del data
    llm = llm_init(
        args.model_name,
        args.tensor_parallel_size,
        args.max_seq_len_to_capture,
        args.max_tokens,
        args.seed,
        args.temperature,
        args.frequency_penalty,
        args.top_p,
        args.top_k,
        args.presence_penalty,
    )

    batch_size = max(1, args.batch_size)
    progress = tqdm(total=len(reader_items), desc="reader inference")
    for batch_start in range(0, len(reader_items), batch_size):
        batch = reader_items[batch_start: batch_start + batch_size]
        metadata_batch = [item[0] for item in batch]
        llm_batch = [item[1] for item in batch]
        predictions = llm_inf_all_batch(
            llm,
            llm_batch,
            args.llm_mode,
            args.model_name,
        )
        if args.conditional_retry_missing_ans:
            retry_indices = []
            for idx, prediction in enumerate(predictions):
                text = str(prediction[0] if prediction else "")
                has_answer = any(
                    line.strip().lower().startswith("ans:") and line.split(":", 1)[1].strip()
                    for line in text.splitlines()
                )
                if not has_answer:
                    retry_indices.append(idx)
            if retry_indices:
                retry_batch = []
                for idx in retry_indices:
                    retry_item = dict(llm_batch[idx])
                    retry_item["user_query"] += (
                        "\n\nThe answer format was missing. Return only one or more lines of "
                        "ans: <concise answer>, with no explanation or duplicates. "
                        "If the evidence does not support an answer, return ans: not available."
                    )
                    retry_batch.append(retry_item)
                retry_predictions = llm_inf_all_batch(
                    llm, retry_batch, args.llm_mode, args.model_name
                )
                for idx, retry_prediction in zip(retry_indices, retry_predictions):
                    predictions[idx] = retry_prediction
        for metadata, prediction in zip(metadata_batch, predictions):
            output_records.append({
                **metadata,
                "prediction": prediction[0],
            })
        progress.update(len(batch))
    progress.close()

    prediction_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.csv"
    write_jsonl(prediction_path, output_records)
    summary_rows = summarize(output_records, views)
    write_csv(summary_path, summary_rows)
    print(f"Saved predictions to: {prediction_path}")
    print(f"Saved summary to: {summary_path}")
    for row in summary_rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
