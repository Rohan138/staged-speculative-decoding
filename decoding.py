import time
from types import MethodType

import torch
from tqdm import tqdm

from ssd_generation import staged_assisted_decoding

INP_LENGTH = 256
GEN_LENGTH = 128
NUM_TOKENS = 5
TOPK = 3


def autoregressive_decoding(inputs, model, temperature=None):
    if temperature is not None:
        kwargs = {"temperature": temperature, "do_sample": True}
    else:
        kwargs = {"do_sample": False}
    return model.generate(
        **inputs,
        max_new_tokens=GEN_LENGTH,
        pad_token_id=model.generation_config.eos_token_id,
        use_cache=True,
        **kwargs,
    )


def speculative_decoding(
    inputs, model, temperature=None, draft_model=None, num_tokens=NUM_TOKENS
):
    if temperature is not None:
        kwargs = {"temperature": temperature, "do_sample": True}
    else:
        kwargs = {"do_sample": False}
    draft_model.generation_config.num_assistant_tokens = num_tokens
    # See https://huggingface.co/blog/assisted-generation
    draft_model.generation_config.num_assistant_tokens_schedule = None # "heuristic"
    return model.generate(
        **inputs,
        max_new_tokens=GEN_LENGTH,
        pad_token_id=model.generation_config.eos_token_id,
        assistant_model=draft_model,
        **kwargs,
    )


def staged_speculative_decoding(
    inputs, model, temperature=None, draft_model=None, topk=TOPK, num_tokens=NUM_TOKENS
):
    if temperature is not None:
        kwargs = {"temperature": temperature, "do_sample": True}
    else:
        kwargs = {"do_sample": False}
    draft_model.generation_config.num_assistant_tokens = num_tokens
    draft_model.generation_config.topk_tokens = topk
    # See https://huggingface.co/blog/assisted-generation
    draft_model.generation_config.num_assistant_tokens_schedule = None # "heuristic"
    model.assisted_decoding = MethodType(staged_assisted_decoding, model)
    return model.generate(
        **inputs,
        max_new_tokens=GEN_LENGTH,
        pad_token_id=model.generation_config.eos_token_id,
        assistant_model=draft_model,
        **kwargs,
    )


def generate(
    model,
    tokenizer,
    dataset,
    temperature,
    decoding_method,
    **kwargs,
):
    assert decoding_method in [
        autoregressive_decoding,
        speculative_decoding,
        staged_speculative_decoding,
    ], f"Unknown decoding method: {decoding_method}"

    gen_time = []
    num_tokens = []

    for data in tqdm(dataset, desc=decoding_method.__name__):
        inputs = tokenizer(
            data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=INP_LENGTH,
        )
        inputs = inputs.to(model.device)

        torch.cuda.synchronize()
        start = time.monotonic()
        gen_out = decoding_method(inputs, model, temperature, **kwargs)
        torch.cuda.synchronize()
        end = time.monotonic()

        gen_time.append(end - start)
        num_tokens.append(gen_out.shape[1] - inputs.input_ids.shape[1])

    avg_time = sum(gen_time) / sum(num_tokens) * 1000
    return avg_time
