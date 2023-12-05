import time

import torch
from tqdm import tqdm

INP_LENGTH = 256
GEN_LENGTH = 128
NUM_TOKENS = 5


def autoregressive_decoding(inputs, model, draft_model, temperature):
    if temperature is not None:
        kwargs = {"temperature": temperature, "do_sample": True}
    else:
        kwargs = {"do_sample": False}
    return model.generate(
        **inputs,
        max_new_tokens=GEN_LENGTH,
        pad_token_id=model.generation_config.eos_token_id,
        **kwargs,
    )


def speculative_decoding(inputs, model, draft_model, temperature):
    if temperature is not None:
        kwargs = {"temperature": temperature, "do_sample": True}
    else:
        kwargs = {"do_sample": False}
    return model.generate(
        **inputs,
        max_new_tokens=GEN_LENGTH,
        pad_token_id=model.generation_config.eos_token_id,
        assistant_model=draft_model,
        **kwargs,
    )

    input_ids: torch.LongTensor = inputs["input_ids"]
    attention_mask: torch.LongTensor = inputs["attention_mask"]

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1).bool()
    eos_token_id = model.generation_config.eos_token_id
    max_len = input_ids.shape[-1] + GEN_LENGTH
    num_tokens = NUM_TOKENS

    while True:
        candidate_input_ids = input_ids.clone()
        for _ in range(int(num_tokens)):
            outputs = draft_model(
                input_ids=candidate_input_ids,
                attention_mask=attention_mask,
            )
            next_token_logits = outputs.logits[:, -1, :]
            if temperature is None:
                next_token_ids = next_token_logits.argmax(dim=-1)
            else:
                next_token_logits = next_token_logits / temperature
                next_token_ids = torch.multinomial(
                    torch.softmax(next_token_logits, dim=-1), num_samples=1
                )
            next_token_ids.masked_fill_(~unfinished_sequences, eos_token_id)
            candidate_input_ids = torch.cat(
                (candidate_input_ids, next_token_ids[:, None]), dim=-1
            )
            is_eos_token = next_token_ids == eos_token_id
            unfinished_sequences = unfinished_sequences.mul(~is_eos_token)
            attention_mask = torch.cat(
                (attention_mask, is_eos_token[:, None].long()), dim=-1
            )

        outputs = model(
            input_ids=candidate_input_ids,
            attention_mask=attention_mask,
        )
        new_token_logits = outputs.logits[:, -num_tokens - 1 :]
        if temperature is None:
            new_token_ids = new_token_logits.argmax(dim=-1)
        else:
            new_token_logits = new_token_logits / temperature
            new_token_ids = torch.multinomial(
                torch.softmax(new_token_logits, dim=-1), num_samples=1
            )
        candidate_new_tokens = candidate_input_ids[:, -num_tokens:]
        n_matches = (candidate_new_tokens == new_token_ids[::-1]).cumsum(dim=-1)
        breakpoint()
        


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
        gen_out = decoding(inputs, model, draft_model, temperature)
        torch.cuda.synchronize()
        end = time.monotonic()

        gen_time.append(end - start)
        num_tokens.append(gen_out.shape[1] - inputs.input_ids.shape[1])

        pbar.update(batch_size)

    avg_time = sum(gen_time) / sum(num_tokens) * 1000
    return avg_time
