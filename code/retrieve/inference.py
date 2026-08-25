import os
import torch

from tqdm import tqdm

from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.model.retriever import Retriever
from src.setup import set_seed, prepare_sample

@torch.no_grad()
def main(args):
    if args.device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    cpt = torch.load(args.path, map_location='cpu')
    config = cpt['config']
    set_seed(config['env']['seed'])
    torch.set_num_threads(config['env']['num_threads'])
    
    infer_set = RetrieverDataset(
        config=config, split=args.split, skip_no_path=args.skip_no_path)
    start_idx = max(args.start_idx, 0)
    end_idx = len(infer_set) if args.limit is None else min(start_idx + args.limit, len(infer_set))
    num_samples = max(end_idx - start_idx, 0)
    
    emb_size = infer_set[0]['q_emb'].shape[-1]
    model = Retriever(emb_size, **config['retriever']).to(device)
    model.load_state_dict(cpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    pred_dict = dict()
    print(f'Running inference on split={args.split}, start_idx={start_idx}, end_idx={end_idx}, num_samples={num_samples}')
    for i in tqdm(range(start_idx, end_idx), total=num_samples):
        raw_sample = infer_set[i]
        sample = collate_retriever([raw_sample])
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
            num_non_text_entities, relation_embs, topic_entity_one_hot,\
            target_triple_probs, a_entity_id_list, analysis_emb,\
            analysis_step_embs, analysis_step_mask = prepare_sample(device, sample)

        entity_list = raw_sample['text_entity_list'] + raw_sample['non_text_entity_list']
        relation_list = raw_sample['relation_list']
        top_K_triples = []
        top_K_triple_ids = []
        top_K_scores = []
        target_relevant_triples = []
        target_relevant_triple_ids = []

        if len(h_id_tensor) != 0:
            pred_triple_logits = model(
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot,
                analysis_emb, analysis_step_embs, analysis_step_mask)
            pred_triple_scores = torch.sigmoid(pred_triple_logits).reshape(-1)
            top_K_results = torch.topk(pred_triple_scores, 
                                       min(args.max_K, len(pred_triple_scores)))
            top_K_scores = top_K_results.values.cpu().tolist()
            top_K_triple_IDs = top_K_results.indices.cpu().tolist()

            for j, triple_id in enumerate(top_K_triple_IDs):
                top_K_triple_ids.append(triple_id)
                top_K_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                    top_K_scores[j]
                ))

            target_relevant_triple_ids = raw_sample['target_triple_probs'].nonzero().reshape(-1).tolist()
            for triple_id in target_relevant_triple_ids:
                target_relevant_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                ))

        if args.candidate_pool_only:
            sample_dict = {
                'scored_triple_ids': torch.tensor(top_K_triple_ids, dtype=torch.int32),
                'scored_triple_scores': torch.tensor(top_K_scores, dtype=torch.float32),
                'target_relevant_triple_ids': torch.tensor(target_relevant_triple_ids, dtype=torch.int32),
                'split': args.split,
            }
        else:
            sample_dict = {
                'question': raw_sample['question'],
                'scored_triples': top_K_triples,
                'scored_triple_ids': top_K_triple_ids,
                'q_entity': raw_sample['q_entity'],
                'q_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['q_entity_id_list']],
                'a_entity': raw_sample['a_entity'],
                'a_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['a_entity_id_list']],
                'max_path_length': raw_sample['max_path_length'],
                'target_relevant_triples': target_relevant_triples,
                'target_relevant_triple_ids': target_relevant_triple_ids,
                'split': args.split,
            }
        
        pred_dict[raw_sample['id']] = sample_dict

    root_path = os.path.dirname(args.path)
    if args.output_path is not None:
        output_path = args.output_path
    elif args.split == 'test':
        output_path = os.path.join(root_path, 'retrieval_result.pth')
    else:
        output_path = os.path.join(root_path, f'retrieval_result_{args.split}.pth')
    torch.save(pred_dict, output_path)
    print(f'Saved retrieval result to: {output_path}')

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to a saved model checkpoint, e.g., webqsp_Nov08-01:14:47/cpt.pth')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split for retrieval inference')
    parser.add_argument('--max_K', type=int, default=500,
                        help='K in top-K triple retrieval')
    parser.add_argument('--limit', type=int, default=None,
                        help='Optional cap on the number of test samples to run; default is the full test split')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Optional starting sample index for chunked inference export')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Optional path to save retrieval result')
    parser.add_argument('--candidate_pool_only', action='store_true',
                        help='Only save scored_triple_ids for downstream selector training/inference')
    parser.add_argument('--skip_no_path', action='store_true',
                        help='Skip samples without shortest-path supervision; use this for train/val selector candidate export')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda:0', 'cpu'],
                        help='Device for retriever inference. Use cpu when a QA LLM process occupies GPU memory.')
    args = parser.parse_args()
    
    main(args)
