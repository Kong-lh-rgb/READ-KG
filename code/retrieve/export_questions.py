import json
import os
import pickle

from argparse import ArgumentParser

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def load_processed(dataset, split):
    path = os.path.join("data_files", dataset, "processed", f"{split}.pkl")
    print(f"Loading processed split: {path}", flush=True)
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_entity_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if str(x).strip()]
    return [str(value)]

def sample_to_question_record(sample, dataset, split):
    return {
        "id": sample["id"],
        "dataset": dataset,
        "split": split,
        "question": sample["question"],
        "q_entity": normalize_entity_list(sample.get("q_entity", [])),
        "q_entity_id_list": list(sample.get("q_entity_id_list", [])),
    }


def export_split(dataset, split, output_dir, overwrite=False):
    output_path = os.path.join(output_dir, f"{split}.questions.jsonl")
    if os.path.exists(output_path) and not overwrite:
        print(f"Skip existing file: {output_path}")
        return output_path

    samples = load_processed(dataset, split)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in tqdm(samples, desc=f"export {dataset}/{split}"):
            record = sample_to_question_record(sample, dataset, split)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(samples)} questions to: {output_path}")
    return output_path


def main(args):
    output_dir = args.output_dir or os.path.join(
        "data_files", args.dataset, "question_analysis"
    )
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]

    for split in splits:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        export_split(
            dataset=args.dataset,
            split=split,
            output_dir=output_dir,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True, choices=["cwq", "webqsp"])
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args)
