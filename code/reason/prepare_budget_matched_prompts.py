#!/usr/bin/env python3
"""Prepare auditable prompts for the READ-KG budget-matched experiment.

The controlled conditions share the same structure-guided retrieval result:

* flat: one score-ordered top-k evidence prompt;
* random: three fixed permutations of that exact top-k evidence set;
* structured: the existing pathfirst, minimal, and candidate READ-KG views.

This script only prepares prompts. It never calls the reader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from main import get_defined_prompts
from preprocess.prepare_data import get_data
from preprocess.prepare_prompts import triplet_to_str
from probe_evidence_views import (
    build_prompt_record,
    normalize_answers,
    strip_score,
    write_jsonl,
)


STRUCTURED_VIEWS = ("pathfirst", "minimal", "candidate")


def stable_seed(sample_id: str, permutation_seed: int) -> int:
    payload = f"{permutation_seed}\0{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def digest_lines(lines: list[str], sort_first: bool = False) -> str:
    values = sorted(lines) if sort_first else lines
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def question_prompt(question: str) -> str:
    text = "Question:\n" + str(question)
    if not text.endswith("?"):
        text += "?"
    return text


def evidence_prompt(triplets: list[tuple]) -> str:
    return "Triplets:\n" + "\n".join(triplet_to_str(row) for row in triplets)


def legacy_scored_topk(scored_triplets: list, k: int) -> list[tuple]:
    """Match SubgraphRAG scored_100: de-duplicate first, then truncate."""
    output = []
    seen = set()
    for triplet in scored_triplets:
        if len(triplet) < 3:
            continue
        bare = strip_score(triplet)
        if bare in seen:
            continue
        seen.add(bare)
        output.append(bare)
        if len(output) >= k:
            break
    return output


def prompt_row(
    sample: dict,
    view: str,
    condition: str,
    user_query: str,
    triplets: list[tuple] | None = None,
    permutation_seed: int | None = None,
) -> dict:
    row = {
        "id": sample["id"],
        "view": view,
        "condition": condition,
        "question": sample["question"],
        "ground_truth": normalize_answers(sample),
        "user_query": user_query,
    }
    if triplets is not None:
        serialized = [triplet_to_str(item) for item in triplets]
        row.update({
            "triplet_count": len(serialized),
            "evidence_set_sha256": digest_lines(serialized, sort_first=True),
            "evidence_order_sha256": digest_lines(serialized, sort_first=False),
        })
    if permutation_seed is not None:
        row["permutation_seed"] = permutation_seed
    return row


def select_samples(data: list[dict], sample_size: int, sample_seed: int) -> list[dict]:
    if sample_size and len(data) > sample_size:
        return random.Random(sample_seed).sample(data, sample_size)
    return data


def build_rows(data: list[dict], args: argparse.Namespace) -> dict[str, list[dict]]:
    sys_prompt, cot_prompt = get_defined_prompts(
        "scored_100",
        args.model_name,
        args.llm_mode,
    )
    random_seeds = [
        int(item) for item in args.random_seeds.split(",") if item.strip()
    ]
    if len(random_seeds) != 3 or len(set(random_seeds)) != 3:
        raise ValueError("--random_seeds must contain exactly three distinct integers.")

    flat_rows: list[dict] = []
    random_rows: list[dict] = []
    structured_rows: list[dict] = []
    invariant_failures = []

    for sample in data:
        # The original SubgraphRAG `scored_100` path calls
        # unique_preserve_order before slicing to K.  Keep that exact order so
        # the controlled flat prompt is a faithful reader-protocol baseline.
        topk = legacy_scored_topk(sample.get("scored_triplets", []), args.k)
        flat_user = "\n\n".join([
            evidence_prompt(topk),
            question_prompt(sample["question"]),
        ])
        flat_row = prompt_row(
            sample,
            view="flat",
            condition="flat",
            user_query=flat_user,
            triplets=topk,
        )
        flat_rows.append(flat_row)

        for permutation_seed in random_seeds:
            permuted = list(topk)
            random.Random(stable_seed(sample["id"], permutation_seed)).shuffle(permuted)
            random_user = "\n\n".join([
                evidence_prompt(permuted),
                question_prompt(sample["question"]),
            ])
            row = prompt_row(
                sample,
                view=f"random_{permutation_seed}",
                condition="random_order",
                user_query=random_user,
                triplets=permuted,
                permutation_seed=permutation_seed,
            )
            if row["evidence_set_sha256"] != flat_row["evidence_set_sha256"]:
                invariant_failures.append({
                    "id": sample["id"],
                    "view": row["view"],
                    "reason": "random evidence set differs from flat top-k",
                })
            random_rows.append(row)

        for view in STRUCTURED_VIEWS:
            built = build_prompt_record(
                sample,
                None,
                view,
                sys_prompt,
                cot_prompt,
                args,
            )
            structured_rows.append(prompt_row(
                sample,
                view=view,
                condition="structured",
                user_query=built["user_query"],
            ))

    if invariant_failures:
        raise AssertionError(json.dumps(invariant_failures[:10], ensure_ascii=False))
    return {
        "flat": flat_rows,
        "random": random_rows,
        "structured": structured_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare READ-KG budget-matched prompts.")
    parser.add_argument("--dataset", choices=["cwq", "webqsp"], required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--random_seeds", default="101,102,103")
    parser.add_argument("--candidate_top_n", type=int, default=8)
    parser.add_argument("--support_per_candidate", type=int, default=5)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--llm_mode", default="sys_icl_dc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "cot" in args.llm_mode:
        raise ValueError("Prompt preparation does not support a CoT mode.")

    scaffold = (
        f"./results/KGQA/{args.dataset}/RoG/{args.split}/"
        f"results_gen_rule_path_RoG-{args.dataset}_RoG_{args.split}"
        "_predictions_3_False_jsonl/predictions.jsonl"
    )
    data = get_data(
        args.dataset,
        scaffold,
        args.retrieval,
        args.split,
        f"pathfirst_{args.k}",
    )
    data = select_samples(data, args.sample_size, args.sample_seed)
    rows = build_rows(data, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for condition, condition_rows in rows.items():
        path = output_dir / f"{condition}_prompts.jsonl"
        write_jsonl(path, condition_rows)
        output_paths[condition] = str(path)

    ids = [str(row["id"]) for row in data]
    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "retrieval": os.path.abspath(args.retrieval),
        "k": args.k,
        "sample_size_argument": args.sample_size,
        "sample_seed": args.sample_seed,
        "question_count": len(data),
        "question_ids_sha256": digest_lines(ids, sort_first=False),
        "random_seeds": [
            int(item) for item in args.random_seeds.split(",") if item.strip()
        ],
        "llm_mode": args.llm_mode,
        "condition_prompt_counts": {
            name: len(condition_rows) for name, condition_rows in rows.items()
        },
        "output_paths": output_paths,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
