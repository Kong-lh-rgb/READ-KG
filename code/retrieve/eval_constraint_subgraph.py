import os
import re
from collections import deque

import numpy as np
import pandas as pd
import torch


MONTH_TOKENS = {
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
    'september', 'october', 'november', 'december',
}

STOPWORDS = {
    'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'from', 'for', 'by',
    'with', 'and', 'or', 'is', 'was', 'were', 'be', 'been', 'being', 'are',
    'do', 'did', 'does', 'have', 'has', 'had', 'that', 'this', 'these',
    'those', 'which', 'what', 'who', 'whom', 'whose', 'where', 'when', 'why',
    'how', 'many', 'much', 'all', 'any', 'some', 'into', 'onto', 'than',
    'after', 'before', 'during', 'under', 'over', 'about', 'through',
}

CONDITION_TOKENS = {
    'first', 'last', 'latest', 'earliest', 'most', 'least', 'largest',
    'smallest', 'highest', 'lowest', 'longest', 'shortest', 'before', 'after',
    'between', 'during', 'since', 'until', 'current', 'former',
    'oldest', 'youngest', 'newest', 'older', 'younger', 'more', 'less',
}

ANSWER_TYPE_KEYWORDS = {
    'person': {'person', 'people', 'actor', 'actress', 'author', 'writer', 'artist', 'president', 'director', 'singer'},
    'location': {'location', 'place', 'city', 'country', 'state', 'province', 'river', 'mountain', 'airport'},
    'time': {'date', 'time', 'year', 'month', 'day', 'born', 'founded', 'released', 'opened', 'died'},
    'number': {'number', 'count', 'amount', 'population', 'age', 'height', 'length'},
}


def _simple_tokenize(text):
    if not text:
        return set()
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _extract_answer_type(question_tokens):
    token_set = set(question_tokens)
    joined = ' '.join(sorted(token_set))

    if 'who' in token_set or 'whom' in token_set:
        return 'person'
    if 'where' in token_set:
        return 'location'
    if 'when' in token_set:
        return 'time'
    if 'how many' in joined or 'how much' in joined or 'number' in token_set or 'count' in token_set:
        return 'number'
    return 'unknown'


def extract_question_constraints(question):
    question_tokens = _simple_tokenize(question)
    condition_tokens = {
        token for token in question_tokens
        if token in CONDITION_TOKENS or token in MONTH_TOKENS or token.isdigit()
    }
    relation_tokens = {
        token for token in question_tokens
        if token not in STOPWORDS and token not in condition_tokens and len(token) > 2
    }
    if len(relation_tokens) == 0:
        relation_tokens = {
            token for token in question_tokens
            if token not in STOPWORDS and len(token) > 1
        }

    return {
        'relation_tokens': sorted(relation_tokens),
        'condition_tokens': sorted(condition_tokens),
        'answer_type': _extract_answer_type(question_tokens),
    }


def _answer_type_match_score(text, answer_type):
    text_tokens = _simple_tokenize(text)
    if answer_type == 'unknown':
        return 0.0
    if answer_type == 'time':
        if re.search(r"\b(19|20)\d{2}\b", text):
            return 1.0
        if len(text_tokens & MONTH_TOKENS) > 0:
            return 0.85
    if answer_type == 'number' and re.search(r"\d", text):
        return 1.0
    overlap = text_tokens & ANSWER_TYPE_KEYWORDS.get(answer_type, set())
    if len(overlap) > 0:
        return min(1.0, 0.45 + 0.15 * len(overlap))
    return 0.0


def get_ranked_triples(sample_dict):
    ranked_triples = []
    for triple_entry in sample_dict.get('scored_triples', []):
        if len(triple_entry) < 3:
            continue
        ranked_triples.append((triple_entry[0], triple_entry[1], triple_entry[2]))
    return ranked_triples


def _build_undirected_graph(triples):
    adjacency = {}
    for head, _, tail in triples:
        adjacency.setdefault(head, set()).add(tail)
        adjacency.setdefault(tail, set()).add(head)
    return adjacency


def _count_connected_components(adjacency):
    visited = set()
    num_components = 0
    for node in adjacency:
        if node in visited:
            continue
        num_components += 1
        queue = deque([node])
        visited.add(node)
        while queue:
            cur = queue.popleft()
            for nxt in adjacency[cur]:
                if nxt in visited:
                    continue
                visited.add(nxt)
                queue.append(nxt)
    return num_components


def _has_reachability(adjacency, source_nodes, target_nodes):
    valid_sources = [node for node in source_nodes if node in adjacency]
    valid_targets = set(node for node in target_nodes if node in adjacency)
    if len(valid_sources) == 0 or len(valid_targets) == 0:
        return 0.0

    queue = deque(valid_sources)
    visited = set(valid_sources)
    while queue:
        cur = queue.popleft()
        if cur in valid_targets:
            return 1.0
        for nxt in adjacency[cur]:
            if nxt in visited:
                continue
            visited.add(nxt)
            queue.append(nxt)
    return 0.0


def compute_constraint_coverage(sample_dict, triples):
    metadata = sample_dict.get('constraint_metadata')
    if not metadata:
        metadata = extract_question_constraints(sample_dict.get('question', ''))
    relation_tokens = set(metadata.get('relation_tokens', []))
    condition_tokens = set(metadata.get('condition_tokens', []))
    answer_type = metadata.get('answer_type', 'unknown')
    topic_entities = set(sample_dict.get('q_entity_in_graph', []))

    covered_relation_tokens = set()
    covered_condition_tokens = set()
    topic_covered = 0.0
    answer_type_covered = 0.0

    for head, relation, tail in triples:
        text_tokens = _simple_tokenize(head) | _simple_tokenize(relation) | _simple_tokenize(tail)
        covered_relation_tokens |= relation_tokens & text_tokens
        covered_condition_tokens |= condition_tokens & text_tokens
        if head in topic_entities or tail in topic_entities:
            topic_covered = 1.0
        answer_type_covered = max(
            answer_type_covered,
            _answer_type_match_score(head, answer_type),
            _answer_type_match_score(relation, answer_type),
            _answer_type_match_score(tail, answer_type),
        )

    total_slots = len(relation_tokens) + len(condition_tokens) + 1
    covered_slots = len(covered_relation_tokens) + len(covered_condition_tokens) + topic_covered
    if answer_type != 'unknown':
        total_slots += 1
        covered_slots += 1.0 if answer_type_covered > 0.5 else 0.0

    return covered_slots / max(total_slots, 1)


def compute_graph_metrics(sample_dict, triples):
    if len(triples) == 0:
        return {
            'connectivity_rate': np.nan,
            'topic_to_answer_reachability': np.nan,
            'num_nodes': 0.0,
            'num_edges': 0.0,
            'density': np.nan,
        }

    adjacency = _build_undirected_graph(triples)
    num_nodes = len(adjacency)
    num_edges = len(triples)
    num_components = _count_connected_components(adjacency)
    connectivity_rate = 1.0 if num_components == 1 else 0.0
    density = 0.0 if num_nodes <= 1 else (2.0 * num_edges) / (num_nodes * (num_nodes - 1))
    reachability = _has_reachability(
        adjacency,
        sample_dict.get('q_entity_in_graph', []),
        sample_dict.get('a_entity_in_graph', []),
    )
    return {
        'connectivity_rate': connectivity_rate,
        'topic_to_answer_reachability': reachability,
        'num_nodes': float(num_nodes),
        'num_edges': float(num_edges),
        'density': density,
    }


def compute_compactness_score(num_nodes, k):
    if k <= 0:
        return np.nan
    return 1.0 - min(num_nodes / max(2.0 * k, 1.0), 1.0)


def compute_sqs(answer_recall, constraint_coverage, graph_metrics, compactness_score):
    answer_score = 0.0 if np.isnan(answer_recall) else answer_recall
    components = [
        answer_score,
        constraint_coverage,
        graph_metrics['connectivity_rate'],
        graph_metrics['topic_to_answer_reachability'],
        compactness_score,
    ]
    valid_components = [value for value in components if not np.isnan(value)]
    return np.mean(valid_components) if len(valid_components) > 0 else np.nan


def main(args):
    pred_dict = torch.load(args.path)
    gpt_triple_dict = torch.load(f'data_files/{args.dataset}/gpt_triples.pth')
    k_list = [int(k) for k in args.k_list.split(',')]

    metric_dict = {}
    for k in k_list:
        metric_dict[f'ans_recall@{k}'] = []
        metric_dict[f'answer_entity_recall@{k}'] = []
        metric_dict[f'shortest_path_triple_recall@{k}'] = []
        metric_dict[f'gpt_triple_recall@{k}'] = []
        metric_dict[f'constraint_coverage@{k}'] = []
        metric_dict[f'connectivity_rate@{k}'] = []
        metric_dict[f'topic_to_answer_reachability@{k}'] = []
        metric_dict[f'avg_nodes@{k}'] = []
        metric_dict[f'avg_edges@{k}'] = []
        metric_dict[f'density@{k}'] = []
        metric_dict[f'sqs@{k}'] = []

    for sample_id, sample_dict in pred_dict.items():
        ranked_triples = get_ranked_triples(sample_dict)
        if len(ranked_triples) == 0:
            continue

        a_entity_in_graph = set(sample_dict['a_entity_in_graph'])
        shortest_path_triples = set(sample_dict['target_relevant_triples'])
        gpt_triples = set(gpt_triple_dict.get(sample_id, []))

        for k in k_list:
            triples_k = ranked_triples[:k]
            triples_k_set = set(triples_k)
            entities_k = {triple[0] for triple in triples_k} | {triple[2] for triple in triples_k}
            ans_recall = np.nan

            if len(a_entity_in_graph) > 0:
                ans_recall = len(a_entity_in_graph & entities_k) / len(a_entity_in_graph)
                metric_dict[f'ans_recall@{k}'].append(ans_recall)
                metric_dict[f'answer_entity_recall@{k}'].append(ans_recall)

            if len(shortest_path_triples) > 0:
                metric_dict[f'shortest_path_triple_recall@{k}'].append(
                    len(shortest_path_triples & triples_k_set) / len(shortest_path_triples)
                )

            if len(gpt_triples) > 0:
                metric_dict[f'gpt_triple_recall@{k}'].append(
                    len(gpt_triples & triples_k_set) / len(gpt_triples)
                )

            constraint_coverage = compute_constraint_coverage(sample_dict, triples_k)
            metric_dict[f'constraint_coverage@{k}'].append(constraint_coverage)
            graph_metrics = compute_graph_metrics(sample_dict, triples_k)
            compactness_score = compute_compactness_score(graph_metrics['num_nodes'], k)
            metric_dict[f'connectivity_rate@{k}'].append(graph_metrics['connectivity_rate'])
            metric_dict[f'topic_to_answer_reachability@{k}'].append(graph_metrics['topic_to_answer_reachability'])
            metric_dict[f'avg_nodes@{k}'].append(graph_metrics['num_nodes'])
            metric_dict[f'avg_edges@{k}'].append(graph_metrics['num_edges'])
            metric_dict[f'density@{k}'].append(graph_metrics['density'])
            metric_dict[f'sqs@{k}'].append(
                compute_sqs(ans_recall, constraint_coverage, graph_metrics, compactness_score)
            )

    for metric_name, values in metric_dict.items():
        metric_dict[metric_name] = np.mean(values) if len(values) > 0 else np.nan

    table_rows = []
    for k in k_list:
        table_rows.append({
            'K': k,
            'ans_recall': round(metric_dict[f'ans_recall@{k}'], 3),
            'answer_entity_recall': round(metric_dict[f'answer_entity_recall@{k}'], 3),
            'shortest_path_triple_recall': round(metric_dict[f'shortest_path_triple_recall@{k}'], 3),
            'gpt_triple_recall': round(metric_dict[f'gpt_triple_recall@{k}'], 3),
            'constraint_coverage': round(metric_dict[f'constraint_coverage@{k}'], 3),
            'graph_connectivity_rate': round(metric_dict[f'connectivity_rate@{k}'], 3),
            'connectivity_rate': round(metric_dict[f'connectivity_rate@{k}'], 3),
            'topic_to_answer_reachability': round(metric_dict[f'topic_to_answer_reachability@{k}'], 3),
            'avg_nodes': round(metric_dict[f'avg_nodes@{k}'], 3),
            'avg_edges': round(metric_dict[f'avg_edges@{k}'], 3),
            'density': round(metric_dict[f'density@{k}'], 3),
            'sqs': round(metric_dict[f'sqs@{k}'], 3),
        })

    df = pd.DataFrame(table_rows)
    report_str = df.to_string(index=False)
    print(report_str)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_name is None:
        result_stem = os.path.splitext(os.path.basename(args.path))[0]
        output_stem = f'{args.dataset}_{result_stem}_constraint_eval'
    else:
        output_stem = args.output_name

    txt_path = os.path.join(args.output_dir, f'{output_stem}.txt')
    csv_path = os.path.join(args.output_dir, f'{output_stem}.csv')
    with open(txt_path, 'w') as f:
        f.write(report_str + '\n')
    df.to_csv(csv_path, index=False)

    print()
    print(f'Saved evaluation text to: {txt_path}')
    print(f'Saved evaluation csv to: {csv_path}')


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        choices=['webqsp', 'cwq'], help='Dataset name')
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to retrieval result')
    parser.add_argument('--k_list', type=str, default='50,100,200,400',
                        help='Comma-separated list of K values for top-K evaluation')
    parser.add_argument('--output_dir', type=str, default='logs',
                        help='Directory for saving evaluation outputs')
    parser.add_argument('--output_name', type=str, default=None,
                        help='Optional output file stem without extension')
    args = parser.parse_args()

    main(args)
