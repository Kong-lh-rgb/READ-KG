"""Fill retrieval records that lack precomputed GTE embeddings.

This utility is intentionally narrow: it completes a previously exported
retrieval dictionary by encoding only the missing processed samples on the
fly. It uses the raw question as the retriever condition and supplies no
structured-question-analysis embeddings or reasoning steps.
"""

import argparse
import pickle

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.dataset.retriever import collate_retriever
from src.model.retriever import Retriever
from src.model.text_encoders import GTELargeEN
from src.setup import prepare_sample, set_seed


def embed_in_chunks(encoder, texts, chunk_size):
    if not texts:
        return torch.zeros(0, 1024)
    chunks = []
    for start in range(0, len(texts), chunk_size):
        chunks.append(encoder.embed(texts[start : start + chunk_size]))
    return torch.cat(chunks, dim=0)


def prepare_raw_sample(raw, triple_score_item, encoder, chunk_size):
    sample = dict(raw)
    q_emb = encoder.embed([sample["question"]])
    entity_embs = embed_in_chunks(encoder, sample["text_entity_list"], chunk_size)
    relation_embs = embed_in_chunks(encoder, sample["relation_list"], chunk_size)

    num_entities = len(sample["text_entity_list"]) + len(sample["non_text_entity_list"])
    topic_mask = torch.zeros(num_entities)
    topic_mask[sample["q_entity_id_list"]] = 1.0

    sample.update(
        {
            "q_emb": q_emb,
            "entity_embs": entity_embs,
            "relation_embs": relation_embs,
            "topic_entity_one_hot": F.one_hot(topic_mask.long(), num_classes=2).float(),
            "target_triple_probs": triple_score_item["triple_scores"],
            "max_path_length": triple_score_item["max_path_length"],
            "analysis_emb": q_emb.reshape(1, -1),
            "analysis_step_embs": torch.zeros(0, q_emb.shape[-1]),
            "analysis_step_mask": torch.zeros(0),
            "a_entity": list(set(sample["a_entity"])),
            "a_entity_id_list": list(set(sample["a_entity_id_list"])),
        }
    )
    return sample


@torch.no_grad()
def score_sample(model, raw, device, max_k, split):
    sample = collate_retriever([raw])
    (
        h_ids,
        r_ids,
        t_ids,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot,
        _,
        _,
        analysis_emb,
        analysis_step_embs,
        analysis_step_mask,
    ) = prepare_sample(device, sample)

    entity_list = raw["text_entity_list"] + raw["non_text_entity_list"]
    relation_list = raw["relation_list"]
    scored_triples = []
    scored_triple_ids = []
    if len(h_ids):
        logits = model(
            h_ids,
            r_ids,
            t_ids,
            q_emb,
            entity_embs,
            num_non_text_entities,
            relation_embs,
            topic_entity_one_hot,
            analysis_emb,
            analysis_step_embs,
            analysis_step_mask,
        )
        scores = torch.sigmoid(logits).reshape(-1)
        top = torch.topk(scores, min(max_k, len(scores)))
        scored_triple_ids = top.indices.cpu().tolist()
        for triple_id, score in zip(scored_triple_ids, top.values.cpu().tolist()):
            scored_triples.append(
                (
                    entity_list[h_ids[triple_id].item()],
                    relation_list[r_ids[triple_id].item()],
                    entity_list[t_ids[triple_id].item()],
                    score,
                )
            )

    target_ids = raw["target_triple_probs"].nonzero().reshape(-1).tolist()
    target_triples = [
        (
            entity_list[raw["h_id_list"][idx]],
            relation_list[raw["r_id_list"][idx]],
            entity_list[raw["t_id_list"][idx]],
        )
        for idx in target_ids
    ]
    return {
        "question": raw["question"],
        "scored_triples": scored_triples,
        "scored_triple_ids": scored_triple_ids,
        "q_entity": raw["q_entity"],
        "q_entity_in_graph": [entity_list[idx] for idx in raw["q_entity_id_list"]],
        "a_entity": raw["a_entity"],
        "a_entity_in_graph": [entity_list[idx] for idx in raw["a_entity_id_list"]],
        "max_path_length": raw["max_path_length"],
        "target_relevant_triples": target_triples,
        "target_relevant_triple_ids": target_ids,
        "split": split,
    }


def main(args):
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    if config["dataset"].get("question_analysis_emb_dir"):
        raise RuntimeError("Checkpoint is not a no-analysis checkpoint")
    if config["dataset"]["name"] != args.dataset:
        raise RuntimeError("Dataset/checkpoint mismatch")

    set_seed(config["env"]["seed"])
    with open(f"data_files/{args.dataset}/processed/{args.split}.pkl", "rb") as handle:
        processed = pickle.load(handle)
    triple_scores = torch.load(
        f"data_files/{args.dataset}/triple_scores/{args.split}.pth", map_location="cpu"
    )
    predictions = torch.load(args.output_path, map_location="cpu")
    missing = [sample for sample in processed if sample["id"] not in predictions]
    print(
        f"dataset={args.dataset} split={args.split} existing={len(predictions)} "
        f"missing={len(missing)} total={len(processed)}"
    )
    if not missing:
        return

    model = Retriever(1024, **config["retriever"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    encoder = GTELargeEN(device)

    for index, raw in enumerate(tqdm(missing), 1):
        sample = prepare_raw_sample(raw, triple_scores[raw["id"]], encoder, args.embed_chunk_size)
        predictions[raw["id"]] = score_sample(model, sample, device, args.max_k, args.split)
        if index % args.save_every == 0:
            torch.save(predictions, args.output_path)
    torch.save(predictions, args.output_path)
    print(f"Saved {len(predictions)} retrieval records to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cwq", "webqsp"])
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_k", type=int, default=200)
    parser.add_argument("--embed_chunk_size", type=int, default=128)
    parser.add_argument("--save_every", type=int, default=10)
    main(parser.parse_args())
