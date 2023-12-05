import time

import torch
from tqdm import tqdm

INP_LENGTH = 256
GEN_LENGTH = 128


def autoregressive_decoding(inputs, model, **kwargs):
    return model.generate(
        **inputs,
        max_new_tokens=GEN_LENGTH,
        pad_token_id=model.generation_config.eos_token_id,
        **kwargs,
    )


def speculative_decoding(inputs, model, **kwargs):
    kwargs["assistant_model"] = kwargs.pop("draft_model")
    return model.generate(
        **inputs,
        max_new_tokens=GEN_LENGTH,
        pad_token_id=model.generation_config.eos_token_id,
        **kwargs,
    )


def staged_speculative_decoding(inputs, model, **kwargs):
    pass


def generate(
    model,
    tokenizer,
    dataset,
    batch_size,
    temperature,
    decoding,
    draft_model=None,
):
    assert decoding in [
        autoregressive_decoding,
        speculative_decoding,
        staged_speculative_decoding,
    ], f"Unknown decoding method: {decoding}"
    assert (
        decoding is not autoregressive_decoding or draft_model is None
    ), "Cannot use draft model with autoregressive decoding"

    if temperature is not None:
        kwargs = {"temperature": temperature, "do_sample": True}
    else:
        kwargs = {"do_sample": False}

    if draft_model is not None:
        kwargs["draft_model"] = draft_model

    gen_time = []
    num_tokens = []

    num_samples = len(dataset) - len(dataset) % batch_size
    pbar = tqdm(range(num_samples), desc=decoding.__name__)

    for idx in range(0, len(dataset), batch_size):
        content = dataset[idx : idx + batch_size]
        inputs = tokenizer(
            content,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=INP_LENGTH,
        )
        inputs = inputs.to(model.device)

        torch.cuda.synchronize()
        start = time.monotonic()
        gen_out = decoding(inputs, model, **kwargs)
        torch.cuda.synchronize()
        end = time.monotonic()

        gen_time.append(end - start)
        num_tokens.append(gen_out.shape[1] - inputs.input_ids.shape[1])

        pbar.update(batch_size)

    avg_time = sum(gen_time) / sum(num_tokens) * 1000
    return avg_time
