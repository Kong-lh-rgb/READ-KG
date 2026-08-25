import os
import re
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from .prepare_prompts import unique_preserve_order


def _load_local_processed_subgraphs(dataset_name, split):
    processed_path = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'retrieve', 'data_files', dataset_name, 'processed', f'{split}.pkl'
    )
    processed_path = os.path.abspath(processed_path)
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f'Local processed fallback not found: {processed_path}')

    print(f"Falling back to local processed data: {processed_path}")
    with open(processed_path, 'rb') as f:
        processed_dict_list = pickle.load(f)

    subgraphs = []
    for sample in tqdm(processed_dict_list):
        entity_list = sample['text_entity_list'] + sample['non_text_entity_list']
        relation_list = sample['relation_list']
        graph = []
        for h_id, r_id, t_id in zip(sample['h_id_list'], sample['r_id_list'], sample['t_id_list']):
            graph.append((
                entity_list[h_id],
                relation_list[r_id],
                entity_list[t_id],
            ))
        subgraphs.append({
            'id': sample['id'],
            'question': sample['question'],
            'graph': graph,
            'a_entity': sample.get('a_entity', []),
            'answer': sample.get('a_entity', []),
        })
    return subgraphs


def get_subgraphs(dataset_name, split):
    input_file = os.path.join("rmanluo", f"RoG-{dataset_name}")
    if os.environ.get("ROG_LOCAL_DATA_ONLY") == "1":
        return _load_local_processed_subgraphs(dataset_name, split)
    try:
        return load_dataset(input_file, split=split)
    except Exception as e:
        print(f"Remote dataset load failed for {input_file}: {repr(e)}")
        return _load_local_processed_subgraphs(dataset_name, split)


def extract_reasoning_paths(text):
    if text is None:
        return None
    pattern = r"Reasoning Paths:(.*?)\n\nQuestion:"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        reasoning_paths = match.group(1).strip()
        return reasoning_paths
    else:
        return None


def add_good_triplets_from_rog(data):
    print("Adding good triplets from ROG...")
    total_good_triplets = 0
    total_good_triplets_in_graph = 0
    total_good_triplets_not_in_graph = 0
    for idx, each_qa in enumerate(tqdm(data)):
        reasoning_paths = extract_reasoning_paths(each_qa.get("input"))
        if reasoning_paths is None or reasoning_paths == "":
            all_paths = []
        else:
            all_paths = reasoning_paths.split("\n")
        data[idx]["good_paths_rog"] = all_paths
        all_good_triplets = []
        for each_path in all_paths:
            each_path = each_path.split(" -> ")
            good_triplets = []
            i = 0
            while i < len(each_path):
                if i + 2 < len(each_path):
                    triplet = (each_path[i], each_path[i + 1], each_path[i + 2])
                    temp_triplet = (each_path[i + 2], each_path[i + 1], each_path[i])
                    total_good_triplets += 1
                    # if triplet in each_qa["graph"] or temp_triplet in each_qa["graph"]:
                    #     total_good_triplets_in_graph += 1
                    # else:
                    #     total_good_triplets_not_in_graph += 1
                    good_triplets.append(triplet)
                i += 2
            all_good_triplets.extend(good_triplets)
        data[idx]["good_triplets_rog"] = unique_preserve_order(all_good_triplets)
    return data


def add_gt_if_not_present(triple_score_dict):
    st = [','.join(list(each)[:3]) for each in triple_score_dict['scored_triples']]
    tt = [','.join(list(each)[:3]) for each in triple_score_dict['target_relevant_triples']]
    for each in tt:
        if each in st:
            continue
        else:
            # put at the beginning
            triple_score_dict["scored_triples"].insert(0, tuple(each.split(',')))
    return triple_score_dict["scored_triples"]


def add_scored_triplets(data, score_dict_path, prompt_mode):
    print("Adding scored triplets...")
    new_data = []
    cnt = 0
    triple_score_dict = torch.load(score_dict_path, weights_only=False)

    running_baselines = False
    if 'triples' in triple_score_dict[next(iter(triple_score_dict))]:
        running_baselines = True
        for k, v in tqdm(triple_score_dict.items()):
            triple_score_dict[k]['scored_triples'] = v['triples']

    for each_qa in tqdm(data):
        if each_qa["id"] in triple_score_dict:
            retrieval_sample = triple_score_dict[each_qa["id"]]
            if 'gt' in prompt_mode:
                scored_triples = add_gt_if_not_present(retrieval_sample)
            else:
                scored_triples = retrieval_sample["scored_triples"]
            each_qa['scored_triplets'] = scored_triples
            if "q_entity_in_graph" in retrieval_sample:
                each_qa["q_entity_in_graph"] = retrieval_sample["q_entity_in_graph"]
            new_data.append(each_qa)
        else:
            print(f"Triplets not found for {each_qa['id']}")
            if running_baselines:
                each_qa['scored_triplets'] = [('', '', '')]
                new_data.append(each_qa)
            else:
                cnt += 1
    print(f"Triplets not found for {cnt} questions")
    return new_data


def resolve_question_analysis_path(dataset_name, split):
    analysis_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'retrieve', 'data_files', dataset_name, 'question_analysis'
    ))
    candidates = [
        os.path.join(analysis_dir, f'{split}.hybrid_qwen_llama.jsonl'),
        os.path.join(analysis_dir, f'{split}.llama31_8b.jsonl'),
        os.path.join(analysis_dir, f'{split}.qwen35flash.jsonl'),
        os.path.join(analysis_dir, f'{split}.jsonl'),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def add_question_analysis(data, dataset_name, split):
    disabled = os.environ.get("READKG_DISABLE_QUESTION_ANALYSIS", "")
    if disabled.strip().lower() in {"1", "true", "yes", "on"}:
        print("Question analysis explicitly disabled; using rank/topology fallbacks.")
        return data
    analysis_path = resolve_question_analysis_path(dataset_name, split)
    if analysis_path is None:
        print("Question analysis file not found. Evidence packing will fall back to rank-based selection.")
        return data

    print(f"Adding question analysis from: {analysis_path}")
    analysis_by_id = {}
    with open(analysis_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record.get("id")
            if sample_id:
                analysis_by_id[sample_id] = record

    missing = 0
    for each_qa in data:
        analysis = analysis_by_id.get(each_qa["id"])
        if analysis is None:
            missing += 1
            continue
        each_qa["question_analysis"] = analysis
    print(f"Question analysis missing for {missing} questions")
    return data


def sample_random_triplets(data, num_triplets, seed=0):
    print(f"Sampling {num_triplets} random triplets...")
    np.random.seed(seed)
    for idx, each_qa in enumerate(tqdm(data)):
        all_triplets = np.array(each_qa["graph"])
        sampled_triplets = np.random.permutation(all_triplets)[:num_triplets]
        data[idx][f"sampled_triplets_{num_triplets}"] = sampled_triplets.tolist()
    return data


def get_data(dataset_name, pred_file_path, score_dict_path, split, prompt_mode, seed=0, triplets_to_sample=[50, 100, 200, 300]):
    print("Loading subgraphs...")
    subgraphs = get_subgraphs(dataset_name, split)
    score_dict_ids = set(torch.load(score_dict_path, weights_only=False).keys())

    if os.path.exists(pred_file_path):
        with open(pred_file_path, "r") as f:
            raw_data = [json.loads(line) for line in f]
    else:
        print(f"Prediction scaffold not found at {pred_file_path}. Building a minimal scaffold from subgraphs.")
        raw_data = []
        for each in tqdm(subgraphs):
            raw_data.append({
                "id": each["id"],
                "question": each["question"],
                "ground_truth": each.get("answer", each.get("a_entity", [])),
                "input": "Reasoning Paths:\n\nQuestion:",
            })
    raw_data = [each for each in raw_data if each["id"] in score_dict_ids]
    subgraphs = [each for each in subgraphs if each["id"] in score_dict_ids]

    print("Adding subgraphs to data...")
    data = []
    for i, each_qa in enumerate(tqdm(raw_data)):
        assert each_qa["id"] == subgraphs[i]["id"]
        each_qa["graph"] = [tuple(each) for each in subgraphs[i]["graph"]]
        each_qa['a_entity'] = subgraphs[i]['a_entity']
        data.append(each_qa)
    # data = raw_data

    data = add_good_triplets_from_rog(data)
    if 'pack' in prompt_mode or 'pathfirst' in prompt_mode or 'lrpeo' in prompt_mode or 'structpath' in prompt_mode:
        data = add_question_analysis(data, dataset_name, split)
    data = add_scored_triplets(data, score_dict_path, prompt_mode)
    # for num_triplets in triplets_to_sample:
    #     data = sample_random_triplets(data, num_triplets, seed)

    return data
