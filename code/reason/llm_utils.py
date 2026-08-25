import os
import re
import time
import openai
from vllm import LLM, SamplingParams
from openai import OpenAI
from functools import partial
from prompts import icl_user_prompt, icl_ass_prompt


def llm_init(model_name, tensor_parallel_size=1, max_seq_len_to_capture=8192, max_tokens=4000, seed=0, temperature=0, frequency_penalty=0, top_p=1.0, top_k=-1, presence_penalty=0.0):
    if "gemma-4" in str(model_name).lower():
        import torch
        from types import SimpleNamespace
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name)
        processor.tokenizer.padding_side = "left"
        model = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="balanced",
            low_cpu_mem_usage=True,
        )
        model.eval()
        return SimpleNamespace(
            _gemma4_model=model,
            _gemma4_processor=processor,
            _gemma4_generation={
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "seed": seed,
                "max_input_tokens": max_seq_len_to_capture,
            },
        )

    if "gpt" not in model_name:
        llm_kwargs = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_seq_len_to_capture,
            "gpu_memory_utilization": 0.95,
        }
        if "gemma-4" in str(model_name).lower():
            llm_kwargs["hf_overrides"] = {
                "text_config": {
                    "allow_global_per_layer_attribute_access": True,
                }
            }
        client = LLM(**llm_kwargs)
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            seed=seed,
        )
        llm = partial(client.chat, sampling_params=sampling_params, use_tqdm=False)
        llm._vllm_client = client
        llm._sampling_params = sampling_params
        llm._tokenizer = client.get_tokenizer()
    else:
        client = OpenAI()
        llm = partial(client.chat.completions.create, model=model_name, seed=seed, temperature=temperature, max_tokens=max_tokens)
    return llm


def sanitize_model_output(text, model_name):
    """Remove model-specific reasoning channel wrappers before answer parsing."""
    text = str(text or "")
    if "gemma-4" in str(model_name).lower():
        text = re.sub(r"<\|channel\|?>thought.*?<channel\|>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<\|channel\|?>final", "", text, flags=re.IGNORECASE)
        text = text.replace("<channel|>", "")
    return text.strip()


def get_outputs(outputs, model_name):
    if "gpt" not in model_name:
        text = outputs[0].outputs[0].text
    else:
        text = outputs.choices[0].message.content
    return sanitize_model_output(text, model_name)


def get_batch_outputs(outputs, model_name):
    if "gpt" in model_name:
        return [get_outputs(output, model_name) for output in outputs]
    return [sanitize_model_output(output.outputs[0].text, model_name) for output in outputs]


def build_conversation(prompts, mode):
    conversation = []
    if 'sys' in mode:
        conversation.append({"role": "system", "content": prompts['sys_query']})

    if 'icl' in mode:
        conversation.append({"role": "user", "content": icl_user_prompt})
        conversation.append({"role": "assistant", "content": icl_ass_prompt})

    if 'sys' in mode:
        conversation.append({"role": "user", "content": prompts['user_query']})
    return conversation


def should_double_check(output):
    text = str(output or "").lower()
    return (
        'ans:' not in text
        or "ans: not available" in text
        or "ans: no information available" in text
    )


def run_gemma4_transformers_batch(llm, conversations, model_name):
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    model = llm._gemma4_model
    processor = llm._gemma4_processor
    settings = llm._gemma4_generation
    prompts = [
        processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for conversation in conversations
    ]
    inputs = processor.tokenizer(
        prompts, padding=True, return_tensors="pt", add_special_tokens=False
    )
    input_len = inputs["input_ids"].shape[1]
    if input_len > settings["max_input_tokens"]:
        raise ValueError(
            f"Gemma-4 input length {input_len} exceeds the configured "
            f"limit {settings['max_input_tokens']}"
        )
    if input_len > 6000 and len(conversations) > 1:
        del inputs
        midpoint = len(conversations) // 2
        return (
            run_gemma4_transformers_batch(llm, conversations[:midpoint], model_name)
            + run_gemma4_transformers_batch(llm, conversations[midpoint:], model_name)
        )
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    class GeneratedTokenPenalty(LogitsProcessor):
        def __init__(self, start_length, frequency_penalty, presence_penalty):
            self.start_length = start_length
            self.frequency_penalty = float(frequency_penalty)
            self.presence_penalty = float(presence_penalty)

        def __call__(self, input_ids, scores):
            generated = input_ids[:, self.start_length:]
            if generated.numel() == 0:
                return scores
            for row_idx in range(generated.shape[0]):
                token_ids, counts = torch.unique(
                    generated[row_idx], return_counts=True
                )
                token_ids = token_ids.to(scores.device)
                counts = counts.to(device=scores.device, dtype=scores.dtype)
                if self.frequency_penalty:
                    scores[row_idx, token_ids] -= self.frequency_penalty * counts
                if self.presence_penalty:
                    scores[row_idx, token_ids] -= self.presence_penalty
            return scores

    penalty = GeneratedTokenPenalty(
        input_len, settings["frequency_penalty"], settings["presence_penalty"]
    )
    generation_kwargs = {
        "max_new_tokens": settings["max_new_tokens"],
        "do_sample": settings["temperature"] > 0,
        "logits_processor": LogitsProcessorList([penalty]),
    }
    if generation_kwargs["do_sample"]:
        generation_kwargs.update(
            temperature=settings["temperature"],
            top_p=settings["top_p"],
            top_k=settings["top_k"],
        )
    torch.manual_seed(settings["seed"])
    try:
        with torch.inference_mode():
            outputs = model.generate(**inputs, **generation_kwargs)
    except torch.OutOfMemoryError:
        if len(conversations) <= 1:
            raise
        del inputs
        torch.cuda.empty_cache()
        midpoint = len(conversations) // 2
        return (
            run_gemma4_transformers_batch(llm, conversations[:midpoint], model_name)
            + run_gemma4_transformers_batch(llm, conversations[midpoint:], model_name)
        )
    generated = outputs[:, input_len:]
    texts = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return [sanitize_model_output(text, model_name) for text in texts]


def run_chat_batch(llm, conversations, model_name):
    if not conversations:
        return []
    if "gemma-4" in str(model_name).lower() and hasattr(llm, "_gemma4_model"):
        return run_gemma4_transformers_batch(llm, conversations, model_name)
    if "gpt" in model_name:
        # The OpenAI client path is kept sequential to avoid changing API rate
        # and retry semantics. vLLM uses real batching below.
        return [
            get_outputs(llm(messages=conversation), model_name)
            for conversation in conversations
        ]

    client = getattr(llm, "_vllm_client", None)
    sampling_params = getattr(llm, "_sampling_params", None)
    tokenizer = getattr(llm, "_tokenizer", None)
    if client is None or sampling_params is None or tokenizer is None:
        return [
            get_outputs(llm(messages=conversation), model_name)
            for conversation in conversations
        ]

    chat_template_kwargs = {}
    if "qwen3" in str(model_name).lower() or "gemma-4" in str(model_name).lower():
        chat_template_kwargs["enable_thinking"] = False

    prompts = [
        tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        for conversation in conversations
    ]
    return get_batch_outputs(
        client.generate(prompts, sampling_params=sampling_params, use_tqdm=False),
        model_name,
    )


def llm_inf(llm, prompts, mode, model_name):
    res = []
    if 'sys' in mode:
        conversation = build_conversation(prompts, mode)
        outputs = get_outputs(llm(messages=conversation), model_name)
        res.append(outputs)

    if 'sys_cot' in mode:
        if 'clear' in mode:
            conversation = []
        conversation.append({"role": "assistant", "content": outputs})
        conversation.append({"role": "user", "content": prompts['cot_query']})
        outputs = get_outputs(llm(messages=conversation), model_name)
        res.append(outputs)
    elif "dc" in mode:
        if should_double_check(res[0]):
            conversation.append({"role": "user", "content": prompts['cot_query']})
            outputs = get_outputs(llm(messages=conversation), model_name)
            res[0] = outputs
        res.append("")
    else:
        res.append("")

    return res


def llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries):
    retries = 0
    while retries < max_retries:
        try:
            return llm_inf(llm, each_qa, llm_mode, model_name)
        except openai.RateLimitError as e:
            wait_time = (2 ** retries) * 5  # Exponential backoff
            print(f"Rate limit error encountered. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    raise Exception("Max retries exceeded. Please check your rate limits or try again later.")


def llm_inf_all(llm, each_qa, llm_mode, model_name, max_retries=5):
    if 'gpt' in model_name:
        return llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries)
    else:
        return llm_inf(llm, each_qa, llm_mode, model_name)


def llm_inf_all_batch(llm, records, llm_mode, model_name):
    """Batch Reader inference for vLLM-backed local models.

    Returns the same shape as repeated llm_inf_all: a list of [answer, cot].
    """
    if not records:
        return []
    if "gpt" in model_name:
        return [llm_inf_all(llm, record, llm_mode, model_name) for record in records]

    conversations = [build_conversation(record, llm_mode) for record in records]
    first_outputs = run_chat_batch(llm, conversations, model_name)
    results = [[output, ""] for output in first_outputs]

    if 'sys_cot' in llm_mode:
        cot_conversations = []
        for record, conversation, output in zip(records, conversations, first_outputs):
            cot_conversation = [] if 'clear' in llm_mode else list(conversation)
            cot_conversation.append({"role": "assistant", "content": output})
            cot_conversation.append({"role": "user", "content": record["cot_query"]})
            cot_conversations.append(cot_conversation)
        cot_outputs = run_chat_batch(llm, cot_conversations, model_name)
        for result, cot_output in zip(results, cot_outputs):
            result[1] = cot_output
    elif "dc" in llm_mode:
        retry_items = [
            (idx, record, conversation)
            for idx, (record, conversation, output) in enumerate(zip(records, conversations, first_outputs))
            if should_double_check(output)
        ]
        retry_conversations = [
            list(conversation) + [{"role": "user", "content": record["cot_query"]}]
            for _, record, conversation in retry_items
        ]
        retry_outputs = run_chat_batch(llm, retry_conversations, model_name)
        for (idx, _, _), retry_output in zip(retry_items, retry_outputs):
            results[idx][0] = retry_output

    return results
