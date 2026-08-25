"""Validation-calibrated KG candidate scores for answer aggregation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


CANDIDATE_SCORE_FEATURES = [
    "max_rank_score",
    "max_triple_score",
    "path_score",
    "relation_match",
    "constraint_match",
    "answer_type_match",
    "step_match",
    "support_count_score",
]

REDUCED_CANDIDATE_SCORE_FEATURES = [
    "max_rank_score",
    "max_triple_score",
    "path_score",
]

LEGACY_CANDIDATE_SCORE_WEIGHTS = [
    0.28,
    0.20,
    0.18,
    0.12,
    0.08,
    0.06,
    0.05,
    0.03,
]


def validate_candidate_score_config(config):
    if not isinstance(config, dict):
        raise ValueError("Candidate score config must be a JSON object.")
    feature_names = config.get("feature_names", [])
    if not feature_names:
        raise ValueError("Candidate score config is missing feature_names.")
    unknown = sorted(set(feature_names) - set(CANDIDATE_SCORE_FEATURES))
    if unknown:
        raise ValueError(f"Unknown candidate-score features: {unknown}")

    scorer_type = config.get("scorer_type")
    if scorer_type == "logistic":
        required = ["weights", "mean", "scale", "bias"]
    elif scorer_type == "weighted_mean":
        required = ["weights"]
    else:
        raise ValueError(f"Unknown scorer_type: {scorer_type!r}")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"Candidate score config is missing fields: {missing}")
    for name in ("weights", "mean", "scale"):
        if name in config and len(config[name]) != len(feature_names):
            raise ValueError(
                f"{name} has length {len(config[name])}, expected {len(feature_names)}."
            )


def load_candidate_score_config(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_candidate_score_config(config)
    config["_path"] = str(path)
    return config


def save_candidate_score_config(path, config):
    validate_candidate_score_config(config)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def candidate_score(stat, config=None):
    if config is None:
        return float(
            np.dot(
                [float(stat.get(name, 0.0) or 0.0) for name in CANDIDATE_SCORE_FEATURES],
                LEGACY_CANDIDATE_SCORE_WEIGHTS,
            )
        )

    validate_candidate_score_config(config)
    features = np.asarray(
        [float(stat.get(name, 0.0) or 0.0) for name in config["feature_names"]],
        dtype=np.float64,
    )
    weights = np.asarray(config["weights"], dtype=np.float64)
    if config["scorer_type"] == "weighted_mean":
        denominator = float(np.abs(weights).sum())
        return float(np.dot(features, weights) / max(denominator, 1e-12))

    mean = np.asarray(config["mean"], dtype=np.float64)
    scale = np.asarray(config["scale"], dtype=np.float64)
    normalized = (features - mean) / np.maximum(scale, 1e-8)
    logit = float(np.dot(normalized, weights) + float(config["bias"]))
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1.0 + exp_logit)


def equal_candidate_weight_config(name, feature_names, metadata=None):
    feature_names = list(feature_names)
    return {
        "version": 1,
        "name": name,
        "scope": "candidate_kg_score",
        "scorer_type": "weighted_mean",
        "feature_names": feature_names,
        "weights": [1.0] * len(feature_names),
        "metadata": metadata or {},
    }


def legacy_candidate_weight_config(metadata=None):
    return {
        "version": 1,
        "name": "legacy_manual",
        "scope": "candidate_kg_score",
        "scorer_type": "weighted_mean",
        "feature_names": list(CANDIDATE_SCORE_FEATURES),
        "weights": list(LEGACY_CANDIDATE_SCORE_WEIGHTS),
        "metadata": metadata or {},
    }


def perturb_logistic_config(config, name, seed, fraction=0.5):
    if config.get("scorer_type") != "logistic":
        raise ValueError("Only logistic configs can be perturbed.")
    rng = np.random.default_rng(seed)
    factors = rng.uniform(1.0 - fraction, 1.0 + fraction, len(config["weights"]))
    output = {
        key: value
        for key, value in config.items()
        if not key.startswith("_")
    }
    output["name"] = name
    output["weights"] = (
        np.asarray(config["weights"], dtype=np.float64) * factors
    ).tolist()
    output["metadata"] = {
        **dict(config.get("metadata", {})),
        "variant": "random_weight_perturbation",
        "perturbation_fraction": fraction,
        "perturbation_seed": seed,
        "weight_factors": factors.tolist(),
    }
    return output
