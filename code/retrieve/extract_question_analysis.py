import json
import os
import pickle

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from src.question_analysis import (
    build_compact_question_analysis_messages,
    build_question_analysis_messages,
    build_json_repair_messages,
    call_chat_with_retry,
    call_openai_compatible_http_with_retry,
    extract_json_object,
    normalize_question_analysis,
)


DEFAULT_QWEN_MODEL = "qwen3.5-flash"
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_API_KEY_ENV = "DASHSCOPE_API_KEY"


def load_processed(dataset, split):
    processed_path = os.path.join("data_files", dataset, "processed", f"{split}.pkl")
    print(f"Loading processed questions from: {processed_path}", flush=True)
    with open(processed_path, "rb") as f:
        return pickle.load(f)


def load_questions_jsonl(path):
    samples = []
    print(f"Loading lightweight questions from: {path}", flush=True)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            samples.append({
                "id": record["id"],
                "question": record["question"],
                "q_entity": record.get("q_entity", []),
                "dataset": record.get("dataset"),
                "split": record.get("split"),
            })
    return samples


def load_retrieval_result(path):
    print(f"Loading lightweight questions from retrieval result: {path}", flush=True)
    import torch

    result = torch.load(path, map_location="cpu")
    samples = []
    for sample_id, record in result.items():
        if "question" not in record:
            continue
        samples.append({
            "id": sample_id,
            "question": record["question"],
            "q_entity": record.get("q_entity", []),
            "dataset": record.get("dataset"),
            "split": record.get("split"),
        })
    return samples


def load_done_ids(output_path):
    done_ids = set()
    if not os.path.exists(output_path):
        return done_ids
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = record.get("id")
            if sample_id:
                done_ids.add(sample_id)
    return done_ids


def get_topic_entities(sample):
    q_entities = sample.get("q_entity", [])
    if isinstance(q_entities, str):
        q_entities = [q_entities]
    return [str(entity).strip() for entity in q_entities if str(entity).strip()]


def model_to_slug(model):
    model = os.path.basename(str(model).rstrip("/"))
    return model.replace(".", "").replace("-", "").replace("_", "")


def build_output_path(dataset, split, model, output_path):
    if output_path is not None:
        return output_path
    output_dir = os.path.join("data_files", dataset, "question_analysis")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{split}.{model_to_slug(model)}.jsonl")


def build_error_path(output_path, error_path):
    if error_path is not None:
        return error_path
    root, _ = os.path.splitext(output_path)
    return f"{root}.errors.jsonl"


def main(args):
    output_model_name = args.local_model_path if args.client_type == "vllm" and args.local_model_path else args.model
    output_path = build_output_path(args.dataset, args.split, output_model_name, args.output_path)
    error_path = build_error_path(output_path, args.error_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    error_dir = os.path.dirname(error_path)
    if error_dir:
        os.makedirs(error_dir, exist_ok=True)

    if args.question:
        samples = [{
            "id": args.sample_id or "manual-question",
            "question": args.question,
            "q_entity": args.topic_entity or [],
        }]
    elif args.questions_jsonl:
        samples = load_questions_jsonl(args.questions_jsonl)
        start_idx = max(args.start_idx, 0)
        end_idx = len(samples) if args.limit is None else min(start_idx + args.limit, len(samples))
        samples = samples[start_idx:end_idx]
    elif args.retrieval_result_path:
        samples = load_retrieval_result(args.retrieval_result_path)
        start_idx = max(args.start_idx, 0)
        end_idx = len(samples) if args.limit is None else min(start_idx + args.limit, len(samples))
        samples = samples[start_idx:end_idx]
    else:
        samples = load_processed(args.dataset, args.split)
        start_idx = max(args.start_idx, 0)
        end_idx = len(samples) if args.limit is None else min(start_idx + args.limit, len(samples))
        samples = samples[start_idx:end_idx]
    print(f"Prepared {len(samples)} samples for extraction.", flush=True)

    done_ids = set()
    if args.resume and not args.overwrite:
        done_ids = load_done_ids(output_path)
    write_mode = "w" if args.overwrite else "a"
    error_write_mode = "w" if args.overwrite else "a"
    samples = [sample for sample in samples if sample["id"] not in done_ids]
    if done_ids:
        print(f"Skipped {len(done_ids)} already completed samples.", flush=True)
    print(f"Remaining {len(samples)} samples to extract.", flush=True)
    if args.client_type == "vllm":
        print(f"Using local vLLM batch_size={args.batch_size}.", flush=True)
    else:
        print(f"Using num_workers={args.num_workers}.", flush=True)

    api_key = None
    client = None
    if not args.dry_run:
        if args.client_type in {"http", "openai"} and args.api_key_env:
            api_key = os.environ.get(args.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Missing API key. Set environment variable {args.api_key_env} "
                    "or pass --api_key_env with a variable that exists."
                )
        if args.client_type == "openai":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "Python package `openai` is not installed in this environment. "
                    "Use the default --client_type http, activate an environment that has it, "
                    "or install it with `pip install openai`."
                ) from exc

            client_kwargs = {}
            if args.base_url:
                client_kwargs["base_url"] = args.base_url
            if api_key:
                client_kwargs["api_key"] = api_key
            client = OpenAI(**client_kwargs)
        elif args.client_type == "vllm":
            client = init_vllm(args)

    if args.dry_run:
        if not samples:
            print("No samples left after resume filtering.")
            return
        sample = samples[0]
        messages = build_messages(args, sample)
        print(json.dumps(messages, ensure_ascii=False, indent=2))
        return

    num_written = 0
    num_errors = 0
    with open(output_path, write_mode) as out_f, open(error_path, error_write_mode) as err_f:
        if args.client_type == "vllm":
            num_written, num_errors = extract_with_vllm(
                args=args,
                llm=client,
                samples=samples,
                out_f=out_f,
                err_f=err_f,
            )
        elif args.num_workers <= 1:
            iterator = (
                extract_one_sample(args, api_key, client, sample)
                for sample in samples
            )
            progress = tqdm(iterator, total=len(samples), desc=f"extract {args.dataset}/{args.split}")
            for result in progress:
                written, errored = write_result(
                    args=args,
                    result=result,
                    out_f=out_f,
                    err_f=err_f,
                )
                num_written += written
                num_errors += errored
        else:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                future_to_id = {
                    executor.submit(extract_one_sample, args, api_key, client, sample): sample["id"]
                    for sample in samples
                }
                for future in tqdm(
                    as_completed(future_to_id),
                    total=len(future_to_id),
                    desc=f"extract {args.dataset}/{args.split}",
                ):
                    try:
                        result = future.result()
                    except Exception as exc:
                        sample_id = future_to_id[future]
                        result = {
                            "ok": False,
                            "error": {
                                "id": sample_id,
                                "question": "",
                                "error": repr(exc),
                            },
                        }
                    written, errored = write_result(
                        args=args,
                        result=result,
                        out_f=out_f,
                        err_f=err_f,
                    )
                    num_written += written
                    num_errors += errored

    print(f"Saved {num_written} analyses to: {output_path}")
    if num_errors:
        print(f"Saved {num_errors} errors to: {error_path}")


def build_messages(args, sample):
    prompt_style = args.prompt_style
    if prompt_style == "auto":
        prompt_style = "compact" if args.client_type == "vllm" else "schema"

    topic_entities = get_topic_entities(sample)
    if prompt_style == "compact":
        return build_compact_question_analysis_messages(
            question=sample["question"],
            topic_entities=topic_entities,
        )
    if prompt_style == "schema":
        return build_question_analysis_messages(
            question=sample["question"],
            topic_entities=topic_entities,
        )
    raise ValueError(f"Unknown prompt_style: {args.prompt_style}")


def extract_one_sample(args, api_key, client, sample):
    sample_id = sample["id"]
    question = sample["question"]
    topic_entities = get_topic_entities(sample)
    messages = build_messages(args, sample)

    raw_text = None
    try:
        if args.client_type == "http":
            raw_text = call_openai_compatible_http_with_retry(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                use_response_format=not args.no_response_format,
                timeout=args.timeout,
            )
        elif args.client_type == "openai":
            raw_text = call_chat_with_retry(
                client=client,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                use_response_format=not args.no_response_format,
            )
        else:
            raise ValueError(f"Unsupported client_type for single-sample extraction: {args.client_type}")
        try:
            raw_json = extract_json_object(raw_text)
        except Exception:
            if args.disable_json_repair:
                raise
            raw_json = repair_json_response(
                args=args,
                api_key=api_key,
                client=client,
                raw_text=raw_text,
            )
        analysis = normalize_question_analysis(
            raw=raw_json,
            question=question,
            sample_id=sample_id,
            known_topic_entities=topic_entities,
        )
        validate_analysis(analysis, args)
    except Exception as exc:
        error_record = {
            "id": sample_id,
            "question": question,
            "error": repr(exc),
        }
        if args.log_raw_errors and raw_text:
            error_record["raw_response"] = raw_text
        return {"ok": False, "error": error_record}

    analysis["split"] = args.split
    if sample.get("split"):
        analysis["split"] = sample["split"]
    if sample.get("dataset"):
        analysis["dataset"] = sample["dataset"]
    analysis["llm_model"] = args.model
    return {"ok": True, "analysis": analysis}


def write_result(args, result, out_f, err_f):
    if result["ok"]:
        out_f.write(json.dumps(result["analysis"], ensure_ascii=False) + "\n")
        out_f.flush()
        return 1, 0

    err_f.write(json.dumps(result["error"], ensure_ascii=False) + "\n")
    err_f.flush()
    if not args.continue_on_error:
        raise RuntimeError(result["error"]["error"])
    return 0, 1


def repair_json_response(args, api_key, client, raw_text):
    last_error = None
    repair_messages = build_json_repair_messages(raw_text)

    for _ in range(args.json_repair_retries):
        try:
            if args.client_type == "http":
                repaired_text = call_openai_compatible_http_with_retry(
                    api_key=api_key,
                    base_url=args.base_url,
                    model=args.model,
                    messages=repair_messages,
                    temperature=0.0,
                    max_tokens=max(args.max_tokens, args.repair_max_tokens),
                    max_retries=args.max_retries,
                    retry_sleep=args.retry_sleep,
                    use_response_format=not args.no_response_format,
                    timeout=args.timeout,
                )
            elif args.client_type == "openai":
                repaired_text = call_chat_with_retry(
                    client=client,
                    model=args.model,
                    messages=repair_messages,
                    temperature=0.0,
                    max_tokens=max(args.max_tokens, args.repair_max_tokens),
                    max_retries=args.max_retries,
                    retry_sleep=args.retry_sleep,
                    use_response_format=not args.no_response_format,
                )
            elif args.client_type == "vllm":
                repaired_text = run_vllm_chat_batch(
                    llm=client,
                    conversations=[repair_messages],
                    temperature=0.0,
                    max_tokens=max(args.max_tokens, args.repair_max_tokens),
                )[0]
            else:
                raise ValueError(f"Unsupported client_type: {args.client_type}")
            return extract_json_object(repaired_text)
        except Exception as exc:
            last_error = exc

    raise last_error


def init_vllm(args):
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError(
            "Python package `vllm` is not installed in this environment. "
            "Activate the reasoner environment or install vllm."
        ) from exc

    model_path = args.local_model_path or args.model
    print(f"Loading local vLLM model: {model_path}", flush=True)
    return LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_seq_len_to_capture=args.max_seq_len_to_capture,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
    )


def run_vllm_chat_batch(llm, conversations, temperature, max_tokens):
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    prompts = build_vllm_prompts(llm, conversations)
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    return [output.outputs[0].text for output in outputs]


def build_vllm_prompts(llm, conversations):
    tokenizer = llm.get_tokenizer()
    prompts = []
    for messages in conversations:
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = render_fallback_chat_prompt(messages)
        prompts.append(prompt)
    return prompts


def render_fallback_chat_prompt(messages):
    rendered = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        rendered.append(f"{role}:\n{content}")
    rendered.append("assistant:\n")
    return "\n\n".join(rendered)


def extract_with_vllm(args, llm, samples, out_f, err_f):
    num_written = 0
    num_errors = 0

    for start in tqdm(range(0, len(samples), args.batch_size), desc=f"extract {args.dataset}/{args.split}"):
        batch_samples = samples[start:start + args.batch_size]
        conversations = [
            build_messages(args, sample)
            for sample in batch_samples
        ]

        try:
            raw_texts = run_vllm_chat_batch(
                llm=llm,
                conversations=conversations,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:
            for sample in batch_samples:
                result = {
                    "ok": False,
                    "error": {
                        "id": sample["id"],
                        "question": sample["question"],
                        "error": repr(exc),
                    },
                }
                written, errored = write_result(args, result, out_f, err_f)
                num_written += written
                num_errors += errored
            continue

        for sample, raw_text in zip(batch_samples, raw_texts):
            result = parse_generated_analysis(args, sample, raw_text, llm)
            written, errored = write_result(args, result, out_f, err_f)
            num_written += written
            num_errors += errored

    return num_written, num_errors


def parse_generated_analysis(args, sample, raw_text, llm=None):
    sample_id = sample["id"]
    question = sample["question"]
    topic_entities = get_topic_entities(sample)

    try:
        try:
            raw_json = extract_json_object(raw_text)
        except Exception:
            if args.disable_json_repair:
                raise
            raw_json = repair_json_response(
                args=args,
                api_key=None,
                client=llm,
                raw_text=raw_text,
            )
        analysis = normalize_question_analysis(
            raw=raw_json,
            question=question,
            sample_id=sample_id,
            known_topic_entities=topic_entities,
        )
        validate_analysis(analysis, args)
    except Exception as exc:
        error_record = {
            "id": sample_id,
            "question": question,
            "error": repr(exc),
        }
        if args.log_raw_errors and raw_text:
            error_record["raw_response"] = raw_text
        return {"ok": False, "error": error_record}

    analysis["split"] = args.split
    if sample.get("split"):
        analysis["split"] = sample["split"]
    if sample.get("dataset"):
        analysis["dataset"] = sample["dataset"]
    analysis["llm_model"] = args.local_model_path or args.model
    return {"ok": True, "analysis": analysis}


def validate_analysis(analysis, args):
    if args.allow_empty_analysis:
        return
    if len(analysis.get("relation_intents", [])) == 0:
        raise ValueError("Low-information analysis: empty relation_intents")
    if len(analysis.get("reasoning_steps", [])) == 0:
        raise ValueError("Low-information analysis: empty reasoning_steps")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True, choices=["cwq", "webqsp"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--questions_jsonl", type=str, default=None)
    parser.add_argument("--retrieval_result_path", type=str, default=None)
    parser.add_argument("--sample_id", type=str, default=None)
    parser.add_argument("--topic_entity", action="append", default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--base_url", type=str, default=DEFAULT_QWEN_BASE_URL)
    parser.add_argument("--api_key_env", type=str, default=DEFAULT_QWEN_API_KEY_ENV)
    parser.add_argument("--client_type", type=str, default="http", choices=["http", "openai", "vllm"])
    parser.add_argument("--prompt_style", type=str, default="auto",
                        choices=["auto", "schema", "compact"])
    parser.add_argument("--local_model_path", type=str, default=None,
                        help="Local model path for --client_type vllm. If omitted, --model is used.")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--error_path", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1400)
    parser.add_argument("--repair_max_tokens", type=int, default=1800)
    parser.add_argument("--json_repair_retries", type=int, default=2)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of concurrent API requests. Start with 4 or 8 to avoid provider throttling.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for local vLLM generation.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_seq_len_to_capture", type=int, default=8192)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_response_format", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--disable_json_repair", action="store_true")
    parser.add_argument("--allow_empty_analysis", action="store_true")
    parser.add_argument("--log_raw_errors", action="store_true", default=True)
    args = parser.parse_args()
    main(args)
