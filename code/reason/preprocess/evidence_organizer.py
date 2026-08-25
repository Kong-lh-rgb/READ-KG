import re
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "was", "were", "what", "when", "where", "which", "who",
    "with", "used", "uses", "use", "type", "name", "named"
}

CONSTRAINT_KEYWORDS = {
    "after", "above", "army", "area", "before", "below", "capacity",
    "date", "deflator", "earliest", "emissions", "first", "founded",
    "gdp", "greater", "height", "largest", "last", "latest", "least",
    "less", "most", "number", "opened", "over", "percent", "percentage",
    "population", "rate", "released", "score", "smallest", "surface",
    "time", "under", "weight", "year"
}


@dataclass(frozen=True)
class OrganizerConfig:
    target_k: int
    path_budget: int
    use_constraint_block: bool = False


@dataclass
class OrganizedEvidence:
    triplets: list
    path_blocks: list
    support_triplets: list
    constraint_triplets: list
    topic_nodes: list

    def as_dict(self):
        return {
            "triplets": self.triplets,
            "path_blocks": self.path_blocks,
            "support_triplets": self.support_triplets,
            "constraint_triplets": self.constraint_triplets,
            "topic_nodes": self.topic_nodes,
        }


PATH_FEATURE_NAMES = [
    "path_len_norm",
    "is_single_hop",
    "candidate_score",
    "mean_retriever_score",
    "max_retriever_score",
    "min_retriever_score",
    "std_retriever_score",
    "mean_rank_score",
    "max_rank_score",
    "min_rank_score",
    "mean_combined",
    "max_combined",
    "mean_best_step_score",
    "max_best_step_score",
    "mean_answer_anchor",
    "max_answer_anchor",
    "mean_constraint_anchor",
    "max_constraint_anchor",
    "mean_topic_anchor",
    "max_topic_anchor",
    "step_coverage",
    "answer_edge_fraction",
    "constraint_edge_fraction",
    "topic_edge_fraction",
]


def triplet_to_str(triplet):
    return f"({triplet[0]},{triplet[1]},{triplet[2]})"


def unique_preserve_order(input_list):
    seen = set()
    unique_list = []
    for item in input_list:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)
    return unique_list


def normalize_text(text):
    return str(text or "").lower().replace("_", " ").replace(".", " ")


def tokenize_text(text):
    tokens = re.findall(r"[a-z0-9]+", normalize_text(text))
    return {token for token in tokens if token not in STOPWORDS and len(token) > 1}


def extract_numeric_tokens(text):
    return set(re.findall(r"\d+(?:\.\d+)?", str(text or "")))


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


def get_answer_type_terms(analysis):
    answer_type = analysis.get("answer_type", {}) if analysis else {}
    return tokenize_text(" ".join(flatten_values(answer_type)))


def get_constraint_terms(analysis):
    constraints = analysis.get("constraints", []) if analysis else []
    return tokenize_text(" ".join(flatten_values(constraints)))


def get_topic_terms(analysis):
    topic_entities = analysis.get("topic_entities", []) if analysis else []
    return tokenize_text(" ".join(flatten_values(topic_entities)))


def get_step_term_sets(analysis):
    steps = analysis.get("reasoning_steps", []) if analysis else []
    relation_intents = analysis.get("relation_intents", []) if analysis else []
    global_relation_terms = tokenize_text(" ".join(flatten_values(relation_intents)))
    step_sets = []
    for step in steps:
        relation_terms = tokenize_text(" ".join(flatten_values([
            step.get("relation_intent"),
            step.get("evidence_text"),
            step.get("source"),
            step.get("target"),
            step.get("target_type"),
        ]))) | global_relation_terms
        endpoint_terms = tokenize_text(" ".join(flatten_values([
            step.get("source"),
            step.get("target"),
            step.get("target_type"),
        ])))
        step_sets.append({
            "relation": relation_terms,
            "endpoint": endpoint_terms,
            "all": relation_terms | endpoint_terms,
        })
    return step_sets


def overlap_score(left, right):
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap == 0:
        return 0.0
    return overlap / max(1, min(len(left), len(right)))


def jaccard_score(left, right):
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def strip_score(triplet):
    return (triplet[0], triplet[1], triplet[2])


def get_retriever_score(triplet):
    if len(triplet) < 4:
        return 0.0
    try:
        return float(triplet[3])
    except (TypeError, ValueError):
        return 0.0


def add_unique(selected, selected_set, triplet):
    bare = strip_score(triplet)
    if bare in selected_set:
        return False
    selected.append(triplet)
    selected_set.add(bare)
    return True


def parse_pack_mode(mode):
    match = re.search(r"pack(?:order)?(?:p(\d+))?_(\d+)(?:_from(\d+))?", mode)
    if not match:
        raise ValueError(f"Invalid packing mode: {mode}")
    protect_k = int(match.group(1)) if match.group(1) else None
    target_k = int(match.group(2))
    pool_k = int(match.group(3)) if match.group(3) else target_k
    return target_k, max(pool_k, target_k), mode.startswith("packorder"), protect_k


def parse_pathfirst_mode(mode):
    match = re.search(r"(?:pathfirst|lrpeo)(?:_constraint)?(?:p(\d+))?_(\d+)", mode)
    if not match:
        raise ValueError(f"Invalid path-first mode: {mode}")
    path_budget = int(match.group(1)) if match.group(1) else None
    target_k = int(match.group(2))
    if path_budget is None:
        path_budget = min(40, max(12, target_k // 2))
    return OrganizerConfig(
        target_k=target_k,
        path_budget=min(path_budget, target_k),
        use_constraint_block="constraint" in mode,
    )


def score_candidate(triplet, rank_idx, pool_size, analysis):
    head, relation, tail = strip_score(triplet)
    head_terms = tokenize_text(head)
    relation_terms = tokenize_text(relation)
    tail_terms = tokenize_text(tail)
    endpoint_terms = head_terms | tail_terms
    triple_terms = endpoint_terms | relation_terms

    answer_terms = get_answer_type_terms(analysis)
    constraint_terms = get_constraint_terms(analysis)
    topic_terms = get_topic_terms(analysis)
    step_sets = get_step_term_sets(analysis)

    step_scores = []
    for step_terms in step_sets:
        relation_score = overlap_score(relation_terms, step_terms["relation"])
        endpoint_score = overlap_score(endpoint_terms, step_terms["endpoint"])
        triple_score = jaccard_score(triple_terms, step_terms["all"])
        step_scores.append(
            0.70 * relation_score + 0.15 * endpoint_score + 0.15 * triple_score
        )

    best_step_score = max(step_scores) if step_scores else 0.0
    best_step_idx = int(np.argmax(step_scores)) if step_scores else None
    answer_anchor = overlap_score(endpoint_terms, answer_terms)
    constraint_anchor = overlap_score(triple_terms, constraint_terms)
    topic_anchor = overlap_score(endpoint_terms, topic_terms)
    rank_score = 1.0 - (rank_idx / max(pool_size - 1, 1))

    combined = (
        0.50 * rank_score +
        0.22 * best_step_score +
        0.14 * answer_anchor +
        0.09 * constraint_anchor +
        0.05 * topic_anchor
    )
    return {
        "combined": combined,
        "rank_score": rank_score,
        "step_scores": step_scores,
        "best_step_score": best_step_score,
        "best_step_idx": best_step_idx,
        "answer_anchor": answer_anchor,
        "constraint_anchor": constraint_anchor,
        "topic_anchor": topic_anchor,
        "entities": {head, tail},
    }


def get_constraint_focus(analysis):
    if not analysis:
        return set(), set()

    all_text = " ".join(flatten_values(analysis))
    all_terms = tokenize_text(all_text)
    terms = set(get_constraint_terms(analysis))
    terms.update(all_terms & CONSTRAINT_KEYWORDS)

    for step in analysis.get("reasoning_steps", []) or []:
        if not isinstance(step, dict):
            continue
        step_text = " ".join(flatten_values([
            step.get("relation_intent"),
            step.get("evidence_text"),
            step.get("source"),
            step.get("target"),
            step.get("target_type"),
        ]))
        step_terms = tokenize_text(step_text)
        if step_terms & CONSTRAINT_KEYWORDS or extract_numeric_tokens(step_text):
            terms.update(step_terms)

    terms.update(get_answer_type_terms(analysis) & CONSTRAINT_KEYWORDS)
    return terms, extract_numeric_tokens(all_text)


def score_constraint_row(row, analysis):
    constraint_terms, constraint_numbers = get_constraint_focus(analysis)
    if not constraint_terms and not constraint_numbers:
        return 0.0

    head, relation, tail = row["triplet"]
    triple_text = " ".join([head, relation, tail])
    triple_terms = tokenize_text(triple_text)
    relation_terms = tokenize_text(relation)
    triple_numbers = extract_numeric_tokens(triple_text)
    features = row["features"]

    term_score = overlap_score(triple_terms, constraint_terms)
    relation_score = overlap_score(relation_terms, constraint_terms)
    number_match = 1.0 if constraint_numbers & triple_numbers else 0.0
    numeric_bonus = 0.35 if triple_numbers else 0.0

    return (
        0.30 * features["constraint_anchor"] +
        0.24 * term_score +
        0.18 * relation_score +
        0.13 * number_match +
        0.07 * numeric_bonus +
        0.05 * features["best_step_score"] +
        0.03 * features["rank_score"]
    )


def resolve_topic_nodes(candidates, analysis):
    entities = set()
    for head, _, tail in candidates:
        entities.add(head)
        entities.add(tail)

    topic_strings = flatten_values((analysis or {}).get("topic_entities", []))
    for step in (analysis or {}).get("reasoning_steps", []) or []:
        if isinstance(step, dict):
            topic_strings.extend(flatten_values(step.get("source")))
            if not step.get("depends_on"):
                topic_strings.extend(flatten_values(step.get("target")))

    topic_terms = [tokenize_text(item) for item in topic_strings]
    topic_nodes = set()
    entity_norm = {entity: normalize_text(entity) for entity in entities}
    topic_norms = {normalize_text(item) for item in topic_strings if item}

    for entity, norm in entity_norm.items():
        if norm in topic_norms:
            topic_nodes.add(entity)
            continue
        entity_terms = tokenize_text(entity)
        if any(overlap_score(entity_terms, terms) >= 0.75 for terms in topic_terms):
            topic_nodes.add(entity)

    return topic_nodes


def build_graph(rows):
    adjacency = defaultdict(list)
    for row in rows:
        head, _, tail = row["triplet"]
        adjacency[head].append((tail, row))
        adjacency[tail].append((head, row))
    return adjacency


def shortest_path_to_candidate(adjacency, topic_nodes, candidate_node, max_hops=4):
    sources = [node for node in topic_nodes if node in adjacency]
    if not sources or candidate_node not in adjacency:
        return []

    queue = deque(sources)
    parent = {source: (None, None) for source in sources}
    hit = None
    while queue:
        node = queue.popleft()
        depth = 0
        cur = node
        while parent[cur][0] is not None:
            depth += 1
            cur = parent[cur][0]
        if depth >= max_hops:
            continue
        for nxt, edge in adjacency[node]:
            if nxt in parent:
                continue
            parent[nxt] = (node, edge)
            if nxt == candidate_node:
                hit = nxt
                queue.clear()
                break
            queue.append(nxt)

    if hit is None:
        return []

    path = []
    cur = hit
    while parent[cur][0] is not None:
        prev, edge = parent[cur]
        path.append(edge)
        cur = prev
    path.reverse()
    return path


def row_step_idx(row):
    idx = row["features"].get("best_step_idx")
    return -1 if idx is None else idx


def score_path(path_rows, candidate_score):
    if not path_rows:
        return -1.0
    avg_rank_score = sum(row["features"]["rank_score"] for row in path_rows) / len(path_rows)
    avg_step_score = sum(row["features"]["best_step_score"] for row in path_rows) / len(path_rows)
    covered_steps = {row_step_idx(row) for row in path_rows if row_step_idx(row) >= 0}
    coverage_bonus = min(len(covered_steps), 3) * 0.08
    multi_hop_bonus = min(max(len(path_rows) - 1, 0), 2) * 0.10
    length_penalty = max(0, len(path_rows) - 3) * 0.08
    return candidate_score + 0.20 * avg_rank_score + 0.25 * avg_step_score + coverage_bonus + multi_hop_bonus - length_penalty


def path_feature_vector(path_rows, candidate_node=None, candidate_score=0.0):
    if not path_rows:
        return np.zeros(len(PATH_FEATURE_NAMES), dtype=np.float32)

    length = len(path_rows)
    retriever_scores = np.array([row.get("retriever_score", 0.0) for row in path_rows], dtype=np.float32)
    rank_scores = np.array([row["features"]["rank_score"] for row in path_rows], dtype=np.float32)
    combined_scores = np.array([row["features"]["combined"] for row in path_rows], dtype=np.float32)
    step_scores = np.array([row["features"]["best_step_score"] for row in path_rows], dtype=np.float32)
    answer_scores = np.array([row["features"]["answer_anchor"] for row in path_rows], dtype=np.float32)
    constraint_scores = np.array([row["features"]["constraint_anchor"] for row in path_rows], dtype=np.float32)
    topic_scores = np.array([row["features"]["topic_anchor"] for row in path_rows], dtype=np.float32)
    covered_steps = {row_step_idx(row) for row in path_rows if row_step_idx(row) >= 0}

    values = [
        min(length, 4) / 4.0,
        1.0 if length == 1 else 0.0,
        float(candidate_score),
        float(retriever_scores.mean()),
        float(retriever_scores.max()),
        float(retriever_scores.min()),
        float(retriever_scores.std()),
        float(rank_scores.mean()),
        float(rank_scores.max()),
        float(rank_scores.min()),
        float(combined_scores.mean()),
        float(combined_scores.max()),
        float(step_scores.mean()),
        float(step_scores.max()),
        float(answer_scores.mean()),
        float(answer_scores.max()),
        float(constraint_scores.mean()),
        float(constraint_scores.max()),
        float(topic_scores.mean()),
        float(topic_scores.max()),
        min(len(covered_steps), 4) / 4.0,
        float((answer_scores > 0).mean()),
        float((constraint_scores > 0).mean()),
        float((topic_scores > 0).mean()),
    ]
    return np.array(values, dtype=np.float32)


def pack_scored_triplets(scored_triplets, mode, analysis=None):
    target_k, pool_k, step_order, configured_protect_k = parse_pack_mode(mode)
    candidates = unique_preserve_order([strip_score(triplet) for triplet in scored_triplets])
    candidates = candidates[:pool_k]
    scored = [
        (triplet, score_candidate(triplet, idx, len(candidates), analysis or {}))
        for idx, triplet in enumerate(candidates)
    ]

    selected = []
    selected_set = set()
    step_match_threshold = 0.12

    if configured_protect_k is None:
        protect_k = min(target_k // 2, 50, len(candidates))
    else:
        protect_k = min(configured_protect_k, target_k, len(candidates))
    for triplet in candidates[:protect_k]:
        add_unique(selected, selected_set, triplet)

    step_sets = get_step_term_sets(analysis or {})
    per_step_quota = max(3, min(10, target_k // max(4, 2 * max(len(step_sets), 1))))
    for step_idx in range(len(step_sets)):
        step_ranked = sorted(
            scored,
            key=lambda item: (
                item[1]["step_scores"][step_idx] if step_idx < len(item[1]["step_scores"]) else 0.0,
                item[1]["rank_score"],
            ),
            reverse=True,
        )
        added = 0
        for triplet, features in step_ranked:
            if added >= per_step_quota or len(selected) >= target_k:
                break
            step_score = features["step_scores"][step_idx] if step_idx < len(features["step_scores"]) else 0.0
            if step_score <= step_match_threshold:
                continue
            if add_unique(selected, selected_set, triplet):
                added += 1

    anchor_ranked = sorted(
        scored,
        key=lambda item: (
            item[1]["answer_anchor"],
            item[1]["constraint_anchor"],
            item[1]["combined"],
        ),
        reverse=True,
    )
    anchor_quota = min(20, target_k // 5)
    added = 0
    for triplet, features in anchor_ranked:
        if added >= anchor_quota or len(selected) >= target_k:
            break
        if features["answer_anchor"] <= 0 and features["constraint_anchor"] <= 0:
            continue
        if add_unique(selected, selected_set, triplet):
            added += 1

    selected_entities = set()
    for triplet in selected:
        selected_entities.update([triplet[0], triplet[2]])
    bridge_ranked = sorted(
        scored,
        key=lambda item: (
            int(item[0][0] in selected_entities or item[0][2] in selected_entities),
            item[1]["combined"],
        ),
        reverse=True,
    )
    for triplet, _ in bridge_ranked:
        if len(selected) >= int(0.8 * target_k):
            break
        if triplet[0] not in selected_entities and triplet[2] not in selected_entities:
            continue
        add_unique(selected, selected_set, triplet)

    for triplet, features in sorted(scored, key=lambda item: item[1]["combined"], reverse=True):
        if len(selected) >= target_k:
            break
        add_unique(selected, selected_set, triplet)

    selected = selected[:target_k]
    if not step_order:
        selected_rank = {triplet: idx for idx, triplet in enumerate(candidates)}
        return sorted(selected, key=lambda triplet: selected_rank.get(triplet, len(candidates)))

    selected_features = {triplet: features for triplet, features in scored}
    ordered = []
    ordered_set = set()
    for step_idx in range(len(step_sets)):
        step_items = [
            triplet for triplet in selected
            if (
                selected_features[triplet]["best_step_idx"] == step_idx and
                selected_features[triplet]["best_step_score"] > step_match_threshold
            )
        ]
        step_items.sort(
            key=lambda triplet: (
                selected_features[triplet]["best_step_score"],
                selected_features[triplet]["rank_score"],
            ),
            reverse=True,
        )
        for triplet in step_items:
            add_unique(ordered, ordered_set, triplet)

    anchor_items = [
        triplet for triplet in selected
        if triplet not in ordered_set and (
            selected_features[triplet]["answer_anchor"] > 0 or
            selected_features[triplet]["constraint_anchor"] > 0
        )
    ]
    anchor_items.sort(
        key=lambda triplet: (
            selected_features[triplet]["answer_anchor"],
            selected_features[triplet]["constraint_anchor"],
            selected_features[triplet]["rank_score"],
        ),
        reverse=True,
    )
    for triplet in anchor_items:
        add_unique(ordered, ordered_set, triplet)

    selected_rank = {triplet: idx for idx, triplet in enumerate(candidates)}
    remaining = [triplet for triplet in selected if triplet not in ordered_set]
    remaining.sort(key=lambda triplet: selected_rank.get(triplet, len(candidates)))
    for triplet in remaining:
        add_unique(ordered, ordered_set, triplet)
    return ordered[:target_k]


class ReasoningPathEvidenceOrganizer:
    """Post-retrieval evidence organizer for LLM KGQA prompts.

    The organizer keeps the same top-K evidence set and changes only the
    presentation structure: priority reasoning paths, optional constraint
    evidence, supporting triples, then the remaining retrieved triples.
    """

    def __init__(self, config, path_scorer=None, constraint_scorer=None):
        self.config = config
        self.path_scorer = path_scorer
        self.constraint_scorer = constraint_scorer

    @classmethod
    def from_mode(cls, mode, path_scorer=None, constraint_scorer=None):
        return cls(
            parse_pathfirst_mode(mode),
            path_scorer=path_scorer,
            constraint_scorer=constraint_scorer,
        )

    def organize(self, scored_triplets, analysis=None):
        config = self.config
        analysis = analysis or {}
        candidates, rows = self._build_rows(scored_triplets, analysis)

        topic_nodes = resolve_topic_nodes(candidates, analysis)
        if not topic_nodes:
            selected = self._fallback_order(rows, candidates)
            return OrganizedEvidence(
                triplets=selected[:config.target_k],
                path_blocks=[],
                support_triplets=[],
                constraint_triplets=[],
                topic_nodes=sorted(topic_nodes),
            )

        path_items = self._rank_reasoning_paths(rows, topic_nodes, analysis=analysis)
        selected = []
        selected_set = set()
        path_blocks = []
        for item in path_items:
            block = []
            added_any = False
            for row in item["path"]:
                block.append(row["triplet"])
                if len(selected) < config.path_budget and add_unique(selected, selected_set, row["triplet"]):
                    added_any = True
            if added_any:
                path_blocks.append(block)
            if len(selected) >= config.path_budget:
                break

        constraint_triplets = []
        if config.use_constraint_block:
            constraint_triplets = self._add_constraint_triplets(
                rows,
                analysis,
                selected,
                selected_set,
            )

        support_triplets = self._add_support_triplets(
            rows,
            selected,
            selected_set,
            config.path_budget,
            config.target_k if config.use_constraint_block else config.path_budget,
        )

        for triplet in candidates:
            if len(selected) >= config.target_k:
                break
            add_unique(selected, selected_set, triplet)

        if len(selected) < config.target_k:
            for row in sorted(rows, key=lambda row: row["features"]["combined"], reverse=True):
                if len(selected) >= config.target_k:
                    break
                add_unique(selected, selected_set, row["triplet"])

        return OrganizedEvidence(
            triplets=selected[:config.target_k],
            path_blocks=path_blocks,
            support_triplets=support_triplets,
            constraint_triplets=constraint_triplets,
            topic_nodes=sorted(topic_nodes),
        )

    def format_prompt(self, scored_triplets, analysis=None):
        result = self.organize(scored_triplets, analysis=analysis)
        selected = result.triplets
        path_set = set()
        constraint_set = set(result.constraint_triplets)
        support_set = set(result.support_triplets)
        lines = ["Triplets:"]

        if result.path_blocks:
            lines.append("Priority reasoning paths:")
            for idx, block in enumerate(result.path_blocks, start=1):
                lines.append(f"Path {idx}:")
                for triplet in block:
                    lines.append(triplet_to_str(triplet))
                    path_set.add(triplet)

        constraint = [
            triplet for triplet in selected
            if triplet in constraint_set and triplet not in path_set
        ]
        if constraint:
            lines.append("Constraint-related triplets:")
            for triplet in constraint:
                lines.append(triplet_to_str(triplet))

        support = [
            triplet for triplet in selected
            if triplet in support_set and triplet not in path_set and triplet not in constraint_set
        ]
        if support:
            lines.append("Supporting triplets:")
            for triplet in support:
                lines.append(triplet_to_str(triplet))

        residual = [
            triplet for triplet in selected
            if (
                triplet not in path_set and
                triplet not in constraint_set and
                triplet not in support_set
            )
        ]
        if residual:
            lines.append("Other retrieved triplets:")
            for triplet in residual:
                lines.append(triplet_to_str(triplet))

        return "\n".join(lines)

    def _fallback_order(self, rows, candidates):
        ordered_rows = sorted(
            rows,
            key=lambda row: (
                row["features"]["best_step_score"],
                row["features"]["answer_anchor"],
                row["features"]["constraint_anchor"],
                row["features"]["rank_score"],
            ),
            reverse=True,
        )
        selected = [row["triplet"] for row in ordered_rows]
        selected_set = set(selected)
        for triplet in candidates:
            if triplet not in selected_set:
                selected.append(triplet)
        return selected

    def _build_rows(self, scored_triplets, analysis):
        score_by_triplet = {}
        candidates = []
        seen = set()
        for triplet in scored_triplets:
            bare = strip_score(triplet)
            if bare in seen:
                continue
            seen.add(bare)
            candidates.append(bare)
            score_by_triplet[bare] = get_retriever_score(triplet)
            if len(candidates) >= self.config.target_k:
                break

        rows = [
            {
                "triplet": triplet,
                "rank": idx,
                "retriever_score": score_by_triplet.get(triplet, 0.0),
                "features": score_candidate(triplet, idx, len(candidates), analysis),
            }
            for idx, triplet in enumerate(candidates)
        ]
        return candidates, rows

    def enumerate_candidate_paths(self, scored_triplets, analysis=None):
        analysis = analysis or {}
        candidates, rows = self._build_rows(scored_triplets, analysis)
        topic_nodes = resolve_topic_nodes(candidates, analysis)
        if not topic_nodes:
            return []
        return self._rank_reasoning_paths(rows, topic_nodes, analysis=analysis)

    def _rank_reasoning_paths(self, rows, topic_nodes, analysis=None):
        adjacency = build_graph(rows)
        candidate_node_scores = defaultdict(float)
        for row in rows:
            head, _, tail = row["triplet"]
            features = row["features"]
            base_score = (
                0.45 * features["combined"] +
                0.25 * features["best_step_score"] +
                0.15 * features["answer_anchor"] +
                0.10 * features["constraint_anchor"] +
                0.05 * features["rank_score"]
            )
            for node in (head, tail):
                if node in topic_nodes:
                    continue
                candidate_node_scores[node] = max(candidate_node_scores[node], base_score)

        candidate_nodes = sorted(
            candidate_node_scores,
            key=lambda node: candidate_node_scores[node],
            reverse=True,
        )[:40]

        path_items = []
        seen_path_signatures = set()
        for node in candidate_nodes:
            path = shortest_path_to_candidate(adjacency, topic_nodes, node, max_hops=4)
            if not path:
                continue
            signature = tuple(row["triplet"] for row in path)
            if signature in seen_path_signatures:
                continue
            seen_path_signatures.add(signature)
            if self.path_scorer is None:
                path_score = score_path(path, candidate_node_scores[node])
            else:
                path_score = self.path_scorer(path, node, candidate_node_scores[node])
            path_items.append({
                "node": node,
                "path": path,
                "score": path_score,
                "rule_score": score_path(path, candidate_node_scores[node]),
                "feature_vector": path_feature_vector(path, node, candidate_node_scores[node]),
            })

        path_sets = [set(row["triplet"] for row in item["path"]) for item in path_items]
        filtered_path_items = []
        for idx, item in enumerate(path_items):
            item_set = path_sets[idx]
            is_subpath = False
            for other_idx, other in enumerate(path_items):
                if idx == other_idx or len(other["path"]) <= len(item["path"]):
                    continue
                if item_set and item_set.issubset(path_sets[other_idx]):
                    is_subpath = True
                    break
            if not is_subpath:
                filtered_path_items.append(item)
        filtered_path_items.sort(key=lambda item: item["score"], reverse=True)
        return filtered_path_items

    def _add_constraint_triplets(self, rows, analysis, selected, selected_set):
        constraint_budget = min(12, max(4, self.config.target_k // 10))
        constraint_rows = sorted(
            rows,
            key=lambda row: (
                self._score_constraint(row, analysis),
                row["features"]["constraint_anchor"],
                row["features"]["best_step_score"],
                row["features"]["rank_score"],
            ),
            reverse=True,
        )
        constraint_triplets = []
        for row in constraint_rows:
            if len(constraint_triplets) >= constraint_budget or len(selected) >= self.config.target_k:
                break
            if self._score_constraint(row, analysis) <= 0.08:
                continue
            if add_unique(selected, selected_set, row["triplet"]):
                constraint_triplets.append(row["triplet"])
        return constraint_triplets

    def _score_constraint(self, row, analysis):
        if self.constraint_scorer is not None:
            return self.constraint_scorer(row, analysis)
        return score_constraint_row(row, analysis)

    def _add_support_triplets(self, rows, selected, selected_set, path_budget, support_limit):
        selected_entities = set()
        for triplet in selected:
            selected_entities.update([triplet[0], triplet[2]])

        support_rows = sorted(
            rows,
            key=lambda row: (
                int(row["triplet"][0] in selected_entities or row["triplet"][2] in selected_entities),
                row["features"]["answer_anchor"],
                row["features"]["constraint_anchor"],
                row["features"]["best_step_score"],
                row["features"]["rank_score"],
            ),
            reverse=True,
        )
        support_triplets = []
        support_budget = min(12, max(4, path_budget // 4))
        for row in support_rows:
            if len(support_triplets) >= support_budget or len(selected) >= support_limit:
                break
            features = row["features"]
            is_useful = (
                features["answer_anchor"] > 0 or
                features["constraint_anchor"] > 0 or
                features["best_step_score"] > 0.12 or
                row["triplet"][0] in selected_entities or
                row["triplet"][2] in selected_entities
            )
            if not is_useful:
                continue
            if add_unique(selected, selected_set, row["triplet"]):
                support_triplets.append(row["triplet"])
        return support_triplets


def pathfirst_select(scored_triplets, mode, analysis=None):
    return ReasoningPathEvidenceOrganizer.from_mode(mode).organize(
        scored_triplets,
        analysis=analysis,
    ).as_dict()


def pathfirst_scored_triplets(scored_triplets, mode, analysis=None):
    return ReasoningPathEvidenceOrganizer.from_mode(mode).organize(
        scored_triplets,
        analysis=analysis,
    ).triplets


def pathfirst_triplet_prompt(scored_triplets, mode, analysis=None):
    return ReasoningPathEvidenceOrganizer.from_mode(mode).format_prompt(
        scored_triplets,
        analysis=analysis,
    )
