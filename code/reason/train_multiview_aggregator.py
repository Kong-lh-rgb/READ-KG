"""Train a lightweight answer-level multi-view aggregator."""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from apply_answer_verifier import torch_load
from multiview_learning_utils import (
    FEATURE_NAMES,
    FEATURE_PROFILES,
    make_labeled_rows,
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


def split_by_sample(rows, val_ratio, seed):
    by_id = defaultdict(list)
    for row in rows:
        by_id[row["id"]].append(row)
    sample_ids = sorted(by_id)
    rng = random.Random(seed)
    rng.shuffle(sample_ids)
    n_val = max(1, int(round(len(sample_ids) * val_ratio))) if val_ratio > 0 else 0
    val_ids = set(sample_ids[:n_val])
    train_rows = []
    val_rows = []
    for sample_id in sample_ids:
        target = val_rows if sample_id in val_ids else train_rows
        target.extend(by_id[sample_id])
    return train_rows, val_rows


def align_features(features, input_dim):
    if len(features) == input_dim:
        return features
    if len(features) > input_dim:
        return features[:input_dim]
    return list(features) + [0.0] * (input_dim - len(features))


def rows_to_tensors(rows, device, input_dim=None):
    input_dim = input_dim or len(FEATURE_NAMES)
    x = torch.tensor(
        [align_features(row["features"], input_dim) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    y = torch.tensor([row["label"] for row in rows], dtype=torch.float32, device=device)
    return x, y


def rank_stats(model, rows, device):
    by_id = defaultdict(list)
    for row in rows:
        by_id[row["id"]].append(row)
    top1_pos = 0
    any_pos = 0
    total = 0
    with torch.no_grad():
        for sample_rows in by_id.values():
            x, _ = rows_to_tensors(sample_rows, device, model.linear.in_features)
            scores = model(x).detach().cpu().numpy().tolist()
            ranked = sorted(zip(scores, sample_rows), key=lambda item: item[0], reverse=True)
            total += 1
            top1_pos += int(ranked[0][1]["label"] == 1)
            any_pos += int(any(row["label"] == 1 for row in sample_rows))
    return {
        "samples": total,
        "top1_pos": top1_pos / max(total, 1),
        "candidate_oracle": any_pos / max(total, 1),
    }


def train_model(train_rows, val_rows, args):
    import torch.nn.functional as F

    device = torch.device(args.device)
    residual_base_idx = None
    if args.residual_base_feature:
        if args.residual_base_feature not in FEATURE_NAMES:
            raise ValueError(f"Unknown --residual_base_feature: {args.residual_base_feature}")
        residual_base_idx = FEATURE_NAMES.index(args.residual_base_feature)
    model = LinearAggregator(len(FEATURE_NAMES), residual_base_idx=residual_base_idx).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.pos_weight > 0:
        pos_weight = args.pos_weight
    else:
        pos = sum(row["label"] for row in train_rows)
        neg = len(train_rows) - pos
        pos_weight = neg / max(pos, 1)
    aux_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )

    by_id_train = defaultdict(list)
    for row in train_rows:
        by_id_train[row["id"]].append(row)
    train_sample_ids = list(by_id_train.keys())

    best_metric = -1.0
    best_state = None
    best_epoch = -1
    rng = random.Random(args.seed)

    for epoch in range(args.epochs):
        model.train()
        rng.shuffle(train_sample_ids)
        listwise_losses = []
        aux_losses = []
        for sample_id in tqdm(train_sample_ids, desc=f"epoch {epoch}"):
            sample_rows = by_id_train[sample_id]
            if len(sample_rows) < 2:
                continue
            x, y = rows_to_tensors(sample_rows, device, len(FEATURE_NAMES))
            n_pos = y.sum()
            logits = model(x)
            if float(n_pos.item()) <= 0.0 or float(n_pos.item()) >= float(y.numel()):
                aux_loss = aux_criterion(logits, y)
                optimizer.zero_grad()
                aux_loss.backward()
                optimizer.step()
                aux_losses.append(float(aux_loss.item()))
                continue

            target = y / n_pos.clamp(min=1.0)
            log_probs = F.log_softmax(logits, dim=0)
            listwise_loss = -(target * log_probs).sum()
            aux_loss = aux_criterion(logits, y)
            margin_loss = logits.new_zeros(())
            if args.margin_weight > 0:
                pos_logits = logits[y > 0.5]
                neg_mask = y <= 0.5
                neg_logits_all = logits[neg_mask]
                if pos_logits.numel() > 0 and neg_logits_all.numel() > 0:
                    if residual_base_idx is not None and args.hard_negative_top_k > 0:
                        neg_base = x[neg_mask, residual_base_idx]
                        k = min(args.hard_negative_top_k, neg_base.numel())
                        hard_indices = torch.topk(neg_base, k=k).indices
                        neg_logits = neg_logits_all[hard_indices]
                    else:
                        neg_logits = neg_logits_all
                    pairwise = args.margin - pos_logits.unsqueeze(1) + neg_logits.unsqueeze(0)
                    margin_loss = F.relu(pairwise).mean()
            loss = listwise_loss + args.aux_weight * aux_loss + args.margin_weight * margin_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            listwise_losses.append(float(listwise_loss.item()))
            aux_losses.append(float(aux_loss.item()))

        model.eval()
        train_stats = rank_stats(model, train_rows, device)
        val_stats = rank_stats(model, val_rows, device) if val_rows else train_stats
        metric = val_stats["top1_pos"]
        avg_lw = float(np.mean(listwise_losses)) if listwise_losses else 0.0
        avg_aux = float(np.mean(aux_losses)) if aux_losses else 0.0
        print(
            f"epoch={epoch} listwise={avg_lw:.6f} aux={avg_aux:.6f} "
            f"train_top1={train_stats['top1_pos']:.4f} "
            f"train_oracle={train_stats['candidate_oracle']:.4f} "
            f"val_top1={val_stats['top1_pos']:.4f} "
            f"val_oracle={val_stats['candidate_oracle']:.4f}"
        )
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_metric": best_metric, "pos_weight": pos_weight}


def print_weights(model):
    weights = model.linear.weight.detach().cpu().numpy()[0]
    bias = float(model.linear.bias.detach().cpu().numpy()[0])
    pairs = sorted(zip(FEATURE_NAMES, weights), key=lambda item: abs(item[1]), reverse=True)
    print("Learned feature weights:")
    print(f"  linear_bias_param: {bias:.6f}")
    if model.residual_base_idx is not None:
        print(f"  residual_base_feature: {FEATURE_NAMES[model.residual_base_idx]}")
    for name, weight in pairs:
        print(f"  {name:<18} {float(weight): .6f}")


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.data:
        payload = torch_load(args.data)
        rows = payload["rows"]
        stats = payload.get("stats", {})
    else:
        rows, stats = make_labeled_rows(args)
        if args.save_data:
            os.makedirs(os.path.dirname(args.save_data), exist_ok=True)
            torch.save({"rows": rows, "stats": stats, "feature_names": FEATURE_NAMES}, args.save_data)
            print(f"Saved training data to: {args.save_data}")

    print("Data stats:", json.dumps(stats, ensure_ascii=False))
    train_rows, val_rows = split_by_sample(rows, args.val_ratio, args.seed)
    print(f"train_rows={len(train_rows)} val_rows={len(val_rows)}")
    model, train_info = train_model(train_rows, val_rows, args)
    print_weights(model)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "feature_names": FEATURE_NAMES,
        "residual_base_feature": args.residual_base_feature or None,
        "train_info": train_info,
        "data_stats": stats,
        "args": vars(args),
    }, args.output)
    print(f"Saved learned aggregator to: {args.output}")
    print("Train info:", json.dumps(train_info, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a learned READ-KG answer aggregator.")
    parser.add_argument("--data", default=None, help="Optional prebuilt .pth with rows.")
    parser.add_argument("--save_data", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", choices=["cwq", "webqsp"], default="cwq")
    parser.add_argument("--split", default="val")
    parser.add_argument("--view", action="append", default=[])
    parser.add_argument("--retrieval", default=None)
    parser.add_argument("--analysis_path", default=None)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--max_hops", type=int, default=4)
    parser.add_argument("--feature_profile", choices=FEATURE_PROFILES, default="full")
    parser.add_argument("--keep_topics", action="store_true")
    parser.add_argument("--include_no_answer_candidate", action="store_true")
    parser.add_argument("--default_multi_n", type=int, default=2)
    parser.add_argument("--multi_min_n", type=int, default=2)
    parser.add_argument("--multi_max_n", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pos_weight", type=float, default=0.0, help="<=0 uses neg/pos.")
    parser.add_argument("--aux_weight", type=float, default=0.4,
                        help="Weight of auxiliary BCE loss on top of listwise CE.")
    parser.add_argument("--residual_base_feature", default="",
                        help="Optional feature added to the learned logit as a fixed residual baseline.")
    parser.add_argument("--margin_weight", type=float, default=0.0,
                        help="Weight for hard-negative pairwise margin loss.")
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--hard_negative_top_k", type=int, default=16,
                        help="Use top-k negatives by residual base feature for margin loss. <=0 uses all negatives.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size != 512:
        print("Note: --batch_size is ignored by listwise training; each optimizer step uses one sample bag.")
    main(args)
