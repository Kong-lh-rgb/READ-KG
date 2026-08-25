import json
import os
import torch

from argparse import ArgumentParser
from tqdm import tqdm

from src.model.text_encoders import GTELargeEN
from src.question_analysis import (
    render_question_analysis_text,
    render_reasoning_step_text,
)


def resolve_analysis_path(dataset, split, analysis_path):
    if analysis_path is not None:
        return analysis_path

    analysis_dir = os.path.join("data_files", dataset, "question_analysis")
    candidates = [
        os.path.join(analysis_dir, f"{split}.hybrid_qwen_llama.jsonl"),
        os.path.join(analysis_dir, f"{split}.llama31_8b.jsonl"),
        os.path.join(analysis_dir, f"{split}.qwen35flash.jsonl"),
        os.path.join(analysis_dir, f"{split}.jsonl"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[-1]


def load_jsonl(path, limit=None):
    records = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def embed_in_batches(text_encoder, texts, batch_size):
    outputs = []
    for start in tqdm(range(0, len(texts), batch_size), desc="embed analysis text"):
        batch = texts[start:start + batch_size]
        outputs.append(text_encoder.embed(batch))
    return torch.cat(outputs, dim=0) if outputs else torch.zeros(0, 1024)


def main(args):
    analysis_path = resolve_analysis_path(args.dataset, args.split, args.analysis_path)
    if not os.path.exists(analysis_path):
        raise FileNotFoundError(f"Question analysis jsonl not found: {analysis_path}")

    records = load_jsonl(analysis_path, args.limit)
    if not records:
        raise ValueError(f"No records found in {analysis_path}")

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if args.text_encoder_name != "gte-large-en-v1.5":
        raise NotImplementedError(args.text_encoder_name)
    text_encoder = GTELargeEN(device)

    full_texts = [render_question_analysis_text(record) for record in records]
    full_embs = embed_in_batches(text_encoder, full_texts, args.batch_size)

    step_texts = []
    step_counts = []
    for record in records:
        answer_type = record.get("answer_type", {})
        constraints = record.get("constraints", [])
        steps = record.get("reasoning_steps", [])[:args.max_steps]
        step_counts.append(len(steps))
        for step in steps:
            step_texts.append(render_reasoning_step_text(step, answer_type, constraints))

    step_embs_flat = embed_in_batches(text_encoder, step_texts, args.batch_size)
    emb_size = full_embs.shape[-1]
    output = {}
    flat_offset = 0
    for idx, record in enumerate(records):
        sample_id = record.get("id")
        if not sample_id:
            raise ValueError(f"Question analysis record at index {idx} has no id.")

        num_steps = step_counts[idx]
        step_embs = torch.zeros(args.max_steps, emb_size)
        step_mask = torch.zeros(args.max_steps)
        if num_steps > 0:
            step_embs[:num_steps] = step_embs_flat[flat_offset:flat_offset + num_steps]
            step_mask[:num_steps] = 1.0
        flat_offset += num_steps

        output[sample_id] = {
            "analysis_emb": full_embs[idx].float(),
            "analysis_step_embs": step_embs.float(),
            "analysis_step_mask": step_mask.float(),
            "question": record.get("question", ""),
            "llm_model": record.get("llm_model", ""),
        }

    output_path = args.output_path
    if output_path is None:
        output_dir = os.path.join(
            "data_files",
            args.dataset,
            "question_analysis_emb",
            args.text_encoder_name,
        )
        output_path = os.path.join(output_dir, f"{args.split}.pth")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    torch.save(output, output_path)
    print(f"Saved {len(output)} question analysis embeddings to: {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser("Embed extracted question analysis for relation-aware DDE")
    parser.add_argument("-d", "--dataset", required=True, choices=["cwq", "webqsp"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--analysis_path", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--text_encoder_name", default="gte-large-en-v1.5")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args)
