import numpy as np
import re
from collections import defaultdict, deque

try:
    from .evidence_organizer import (
        pack_scored_triplets as organizer_pack_scored_triplets,
        pathfirst_select as organizer_pathfirst_select,
        pathfirst_scored_triplets as organizer_pathfirst_scored_triplets,
        pathfirst_triplet_prompt as organizer_pathfirst_triplet_prompt,
        ReasoningPathEvidenceOrganizer,
    )
except ImportError:
    from preprocess.evidence_organizer import (
        pack_scored_triplets as organizer_pack_scored_triplets,
        pathfirst_select as organizer_pathfirst_select,
        pathfirst_scored_triplets as organizer_pathfirst_scored_triplets,
        pathfirst_triplet_prompt as organizer_pathfirst_triplet_prompt,
        ReasoningPathEvidenceOrganizer,
    )


def get_lrpeo_path_scorer():
    raise ValueError(
        "lrpeo is an archived experimental prompt mode. "
        "Use pathfirst/minimal/candidate for the READ-KG reproduction path."
    )


def get_structured_evidence_organizer(mode):
    raise ValueError(
        f"{mode} is an archived experimental prompt mode. "
        "Use pathfirst/minimal/candidate for the READ-KG reproduction path."
    )


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


def remove_same_head_tail(triplets, mode):
    if 'rmht' not in mode:
        return triplets

    new_triplets = []
    seen = set()
    for triplet in triplets:
        item_1 = ','.join([str(triplet[0]), str(triplet[2])])
        item_2 = ','.join([str(triplet[2]), str(triplet[0])])
        if item_1 not in seen and item_2 not in seen:
            seen.add(item_1)
            seen.add(item_2)
            new_triplets.append(triplet)
    return new_triplets


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


def parse_pack_mode(mode):
    match = re.search(r"pack(?:order)?(?:p(\d+))?_(\d+)(?:_from(\d+))?", mode)
    if not match:
        raise ValueError(f"Invalid packing mode: {mode}")
    protect_k = int(match.group(1)) if match.group(1) else None
    target_k = int(match.group(2))
    pool_k = int(match.group(3)) if match.group(3) else target_k
    return target_k, max(pool_k, target_k), mode.startswith("packorder"), protect_k


def parse_pathfirst_mode(mode):
    match = re.search(r"pathfirst(?:_constraint)?(?:p(\d+))?_(\d+)", mode)
    if not match:
        raise ValueError(f"Invalid path-first mode: {mode}")
    path_budget = int(match.group(1)) if match.group(1) else None
    target_k = int(match.group(2))
    if path_budget is None:
        path_budget = min(40, max(12, target_k // 2))
    return target_k, min(path_budget, target_k)


def is_constraint_pathfirst_mode(mode):
    return "constraint" in mode


def strip_score(triplet):
    return (triplet[0], triplet[1], triplet[2])


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

    # These are only used as constraint anchors when the analysis itself contains
    # constraint-like language, so generic answer types do not dominate.
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


def add_unique(selected, selected_set, triplet):
    bare = strip_score(triplet)
    if bare in selected_set:
        return False
    selected.append(triplet)
    selected_set.add(bare)
    return True


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

    # Keep a stable high-confidence prefix so packing cannot destroy the retriever's
    # strongest signal.
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

    # Add local bridge evidence around already selected entities.
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
    features = row["features"]
    idx = features.get("best_step_idx")
    if idx is None:
        return -1
    return idx


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


def pathfirst_select(scored_triplets, mode, analysis=None):
    target_k, path_budget = parse_pathfirst_mode(mode)
    use_constraint_block = is_constraint_pathfirst_mode(mode)
    candidates = unique_preserve_order([strip_score(triplet) for triplet in scored_triplets])
    candidates = candidates[:target_k]
    analysis = analysis or {}
    rows = []
    for idx, triplet in enumerate(candidates):
        rows.append({
            "triplet": triplet,
            "rank": idx,
            "features": score_candidate(triplet, idx, len(candidates), analysis),
        })

    topic_nodes = resolve_topic_nodes(candidates, analysis)
    adjacency = build_graph(rows)
    row_by_triplet = {row["triplet"]: row for row in rows}

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

    if not topic_nodes:
        # Without reliable topic anchors, fall back to a stable semantic-first order
        # but keep the same top-k evidence set.
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
        return {
            "triplets": selected[:target_k],
            "path_blocks": [],
            "constraint_triplets": [],
            "support_triplets": [],
            "topic_nodes": sorted(topic_nodes),
        }

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
        path_items.append({
            "node": node,
            "path": path,
            "score": score_path(path, candidate_node_scores[node]),
        })

    # Prefer complete reasoning chains over their intermediate prefixes.
    filtered_path_items = []
    path_sets = [set(row["triplet"] for row in item["path"]) for item in path_items]
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
    path_items = filtered_path_items
    path_items.sort(key=lambda item: item["score"], reverse=True)

    selected = []
    selected_set = set()
    path_blocks = []
    for item in path_items:
        block = []
        added_any = False
        for row in item["path"]:
            # Display complete paths even when multiple paths share prefix edges;
            # only the underlying evidence budget is de-duplicated.
            block.append(row["triplet"])
            if len(selected) < path_budget and add_unique(selected, selected_set, row["triplet"]):
                added_any = True
        if added_any:
            path_blocks.append(block)
        if len(selected) >= path_budget:
            break

    constraint_triplets = []
    if use_constraint_block:
        constraint_budget = min(12, max(4, target_k // 10))
        constraint_rows = sorted(
            rows,
            key=lambda row: (
                score_constraint_row(row, analysis),
                row["features"]["constraint_anchor"],
                row["features"]["best_step_score"],
                row["features"]["rank_score"],
            ),
            reverse=True,
        )
        for row in constraint_rows:
            if len(constraint_triplets) >= constraint_budget or len(selected) >= target_k:
                break
            if score_constraint_row(row, analysis) <= 0.08:
                continue
            if add_unique(selected, selected_set, row["triplet"]):
                constraint_triplets.append(row["triplet"])

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
    support_limit = target_k if use_constraint_block else path_budget
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

    # Preserve the retriever's original top-k evidence set after the organized prefix.
    for triplet in candidates:
        if len(selected) >= target_k:
            break
        add_unique(selected, selected_set, triplet)

    # If duplicate removal left room, fill by semantic score from the same pool.
    if len(selected) < target_k:
        for row in sorted(rows, key=lambda row: row["features"]["combined"], reverse=True):
            if len(selected) >= target_k:
                break
            add_unique(selected, selected_set, row["triplet"])

    return {
        "triplets": selected[:target_k],
        "path_blocks": path_blocks,
        "constraint_triplets": constraint_triplets,
        "support_triplets": support_triplets,
        "topic_nodes": sorted(topic_nodes),
    }


def pathfirst_scored_triplets(scored_triplets, mode, analysis=None):
    return pathfirst_select(scored_triplets, mode, analysis=analysis)["triplets"]


def pathfirst_triplet_prompt(scored_triplets, mode, analysis=None):
    result = pathfirst_select(scored_triplets, mode, analysis=analysis)
    selected = result["triplets"]
    path_set = set()
    constraint_set = set(result.get("constraint_triplets", []))
    support_set = set(result["support_triplets"])
    lines = ["Triplets:"]

    if result["path_blocks"]:
        lines.append("Priority reasoning paths:")
        for idx, block in enumerate(result["path_blocks"], start=1):
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


def pack_scored_triplets(scored_triplets, mode, analysis=None):
    return organizer_pack_scored_triplets(scored_triplets, mode, analysis=analysis)


def pathfirst_select(scored_triplets, mode, analysis=None):
    return organizer_pathfirst_select(scored_triplets, mode, analysis=analysis)


def pathfirst_scored_triplets(scored_triplets, mode, analysis=None):
    return organizer_pathfirst_scored_triplets(scored_triplets, mode, analysis=analysis)


def pathfirst_triplet_prompt(scored_triplets, mode, analysis=None):
    return organizer_pathfirst_triplet_prompt(scored_triplets, mode, analysis=analysis)


def merge_tuples(tuple_list, mode=0):
    if mode == 0:
        merged_dict = defaultdict(lambda: [[], None, None])
        for t in tuple_list:
            key = (t[1], t[2])  # Group by the second and third elements
            merged_dict[key][0].append(t[0])  # Append the first element to the list
            merged_dict[key][1] = t[1]  # Set the second element
            merged_dict[key][2] = t[2]  # Set the third element

        # Convert the dictionary back to a list of merged tuples
        return [('[' + ','.join(v[0]) + ']', v[1], v[2]) for v in merged_dict.values()]
    else:
        assert mode == 2
        merged_dict = defaultdict(lambda: [None, None, []])
        for t in tuple_list:
            key = (t[0], t[1])
            merged_dict[key][2].append(t[2])
            merged_dict[key][0] = t[0]
            merged_dict[key][1] = t[1]
        return [(v[0], v[1], '[' + ','.join(v[2]) + ']') for v in merged_dict.values()]


def get_prompts(each_qa, mode, sys_prompt, cot_prompt, thres, seed=0):
    question_prompt = "Question:\n" + each_qa['question']
    if question_prompt[-1] != '?':
        question_prompt += '?'

    if mode.startswith('structpath'):
        organizer_mode = mode.replace('structpath', 'pathfirst', 1)
        triplet_prompt = get_structured_evidence_organizer(organizer_mode).format_prompt(
            each_qa['scored_triplets'],
            analysis=each_qa.get("question_analysis")
        )
    elif mode.startswith('lrpeo'):
        triplet_prompt = ReasoningPathEvidenceOrganizer.from_mode(
            mode,
            path_scorer=get_lrpeo_path_scorer(),
        ).format_prompt(
            each_qa['scored_triplets'],
            analysis=each_qa.get("question_analysis")
        )
    elif mode.startswith('pathfirst'):
        triplet_prompt = organizer_pathfirst_triplet_prompt(
            each_qa['scored_triplets'],
            mode,
            analysis=each_qa.get("question_analysis")
        )
    elif mode.startswith('pack'):
        input_triplets = organizer_pack_scored_triplets(
            each_qa['scored_triplets'],
            mode,
            analysis=each_qa.get("question_analysis")
        )
        input_triplets = [triplet_to_str(triplet) for triplet in input_triplets]
        triplet_prompt = "Triplets:\n" + "\n".join(input_triplets)
    elif 'rog' in mode:
        num_sampled_triplets = int(mode.split('_')[1])
        good_triplets_rog = each_qa['good_triplets_rog']
        input_triplets = remove_same_head_tail(good_triplets_rog, mode)
        # sampled_triplets = np.array(each_qa[f'sampled_triplets_{num_sampled_triplets}'])
        # input_triplets = np.concatenate([good_triplets_rog, sampled_triplets]) if len(good_triplets_rog) > 0 else sampled_triplets
        input_triplets = [triplet_to_str(triplet) for triplet in input_triplets]
        other_triplets = remove_same_head_tail(each_qa['scored_triplets'], mode)
        other_triplets = [triplet_to_str(triplet) for triplet in other_triplets]
        input_triplets = unique_preserve_order(input_triplets + other_triplets)
        input_triplets = input_triplets[:num_sampled_triplets]
        # input_triplets = np.random.permutation(input_triplets)
        triplet_prompt = "Triplets:\n" + "\n".join(input_triplets)
    elif 'scored' in mode:
        num_sampled_triplets = int(mode.split('_')[1])
        input_triplets = each_qa['scored_triplets']
        if thres:
            input_triplets = [(triplet[0], triplet[1], triplet[2]) for triplet in input_triplets if triplet[3] >= thres]
        else:
            input_triplets = [(triplet[0], triplet[1], triplet[2]) for triplet in input_triplets]

        input_triplets = unique_preserve_order(input_triplets)
        input_triplets = input_triplets[:num_sampled_triplets]
        input_triplets = [triplet_to_str(triplet) for triplet in input_triplets]
        if 'rev' in mode:
            input_triplets.reverse()
        triplet_prompt = "Triplets:\n" + "\n".join(input_triplets)

    elif 'rand' in mode:
        num_sampled_triplets = int(mode.split('_')[1])
        np.random.seed(seed)
        input_triplets = np.random.permutation(np.array(each_qa['graph']))
        if 'randNoA' in mode:
            for each_a in each_qa['a_entity']:
                input_triplets = [triplet for triplet in input_triplets if each_a not in triplet[0] and each_a not in triplet[2]]

        input_triplets = unique_preserve_order([triplet_to_str(triplet) for triplet in input_triplets])
        input_triplets = input_triplets[:num_sampled_triplets]
        triplet_prompt = "Triplets:\n" + "\n".join(input_triplets)
    elif 'noevi' in mode:
        triplet_prompt = ''
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if 'firstq' in mode:
        all_query = "\n\n".join([sys_prompt, question_prompt, triplet_prompt])
        user_query = "\n\n".join([question_prompt, triplet_prompt])
    else:
        all_query = "\n\n".join([sys_prompt, triplet_prompt, question_prompt])
        user_query = "\n\n".join([triplet_prompt, question_prompt])
        if triplet_prompt == '':
            user_query = question_prompt

    each_qa['sys_query'] = sys_prompt
    each_qa['user_query'] = user_query
    each_qa['all_query'] = all_query
    each_qa['cot_query'] = cot_prompt
    return each_qa


def get_prompts_for_data(data, mode, sys_prompt, cot_prompt, thres):
    new_data = []
    for each_qa in data:
        new_data.append(get_prompts(each_qa, mode, sys_prompt, cot_prompt, thres))
    return new_data
