import copy
import time

import einops
import torch
from torch.nn import functional as F
from tqdm import tqdm

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
    # if n_matches == int(num_assistant_tokens):
    #     num_assistant_tokens += 2.0
    # else:
    #     num_assistant_tokens = max(1.0, num_assistant_tokens - 1.0)
    draft_model.generation_config.num_assistant_tokens_schedule = None  # "heuristic"
    return model.generate(
        **inputs,
        max_new_tokens=GEN_LENGTH,
        pad_token_id=model.generation_config.eos_token_id,
        assistant_model=draft_model,
        **kwargs,
    )


def _expand_past_key_values(past_key_values, batch_size):
    expanded_past_key_values = []
    for layer in past_key_values:
        expanded_layer = []
        for item in layer:
            expanded_item = einops.repeat(item, "b ... -> (b k) ...", k=batch_size)
            expanded_layer.append(expanded_item)
        expanded_past_key_values.append(expanded_layer)
    return expanded_past_key_values


@torch.no_grad()
def staged_speculative_decoding(
    inputs, model, temperature=None, draft_model=None, topk=TOPK, num_tokens=NUM_TOKENS
):
    inputs = copy.copy(inputs)
    output_sequence = inputs["input_ids"]
    total_length = output_sequence.shape[1]
    eos_token_id = model.generation_config.eos_token_id

    # Generate initial kv cache for model
    model_inputs = {
        "input_ids": inputs["input_ids"][:, :-1],
        "attention_mask": inputs["attention_mask"][:, :-1],
        "past_key_values": None,
    }
    model_outputs = model(**model_inputs, use_cache=True)
    model_past_key_values = model_outputs.past_key_values

    while True:
        # Generate initial kv cache for draft model
        # currently need to rerun this every iteration because last token
        # is generated by model and does not have a corresponding kv cache
        # TODO (Rohan138): This is inefficient; find a better way!
        prev_inputs = {}
        prev_inputs["input_ids"] = inputs["input_ids"][:, :-1]
        prev_inputs["past_key_values"] = inputs.get("past_key_values", None)
        draft_outputs = draft_model(**prev_inputs, use_cache=True)

        draft_inputs = {
            "input_ids": inputs["input_ids"][:, -1:],
            "past_key_values": draft_outputs.past_key_values,
        }

        # Build staged speculative decoding tree
        tree_input_ids = []
        tree_past_key_values = []

        tree_input_ids.append(
            einops.repeat(
                draft_inputs["input_ids"], "b ... -> (b k) ...", k=topk**num_tokens
            )
        )
        tree_past_key_values.append(draft_inputs["past_key_values"])

        for depth in range(num_tokens):
            draft_outputs = draft_model(**draft_inputs, use_cache=True)
            next_token_logits = draft_outputs.logits[:, -1, :]
            if temperature is not None:
                next_token_logits = next_token_logits / temperature

            _, next_token_ids = next_token_logits.topk(topk, dim=-1)

            # Update draft inputs for next iteration
            # Fold leaf tokens into batch dimension
            next_token_ids = next_token_ids.view(-1, 1)
            draft_inputs["input_ids"] = next_token_ids

            draft_inputs["past_key_values"] = _expand_past_key_values(
                draft_outputs.past_key_values, topk
            )

            tree_input_ids.append(
                einops.repeat(
                    draft_inputs["input_ids"],
                    "b ... -> (b k) ...",
                    k=topk ** (num_tokens - depth - 1),
                )
            )
            tree_past_key_values.append(draft_inputs["past_key_values"])

            if (next_token_ids == eos_token_id).any():
                break

        # TODO (Rohan138): Handle eos_token better; right now we early stop
        # the depthwise generation of the whole tree if eos_token is found,
        # then pad with zeros
        input_ids = torch.cat(tree_input_ids, dim=1)
        input_ids = F.pad(input_ids, (0, num_tokens + 1 - input_ids.shape[1]))

        model_inputs["input_ids"] = input_ids
        mask = inputs["attention_mask"]
        attention_mask = torch.cat(
            [mask, mask.new_ones((mask.shape[0], num_tokens + 1))], dim=-1
        )
        model_inputs["attention_mask"] = einops.repeat(
            attention_mask[:, :-1], "b ... -> (b k) ...", k=input_ids.shape[0]
        )
        model_inputs["past_key_values"] = _expand_past_key_values(
            model_past_key_values, input_ids.shape[0]
        )

        position_ids = torch.arange(
            total_length, total_length + input_ids.shape[1], device=input_ids.device
        )
        position_ids = einops.repeat(position_ids, "... -> k ...", k=input_ids.shape[0])
        model_inputs["position_ids"] = position_ids

        model_outputs = model(**model_inputs, use_cache=True)
        new_logits = model_outputs.logits

        if temperature is not None:
            new_logits = new_logits / temperature
            dist = torch.distributions.Categorical(logits=new_logits)
            selected_tokens = dist.sample()
        else:
            selected_tokens = new_logits.argmax(dim=-1)

        # find valid indices where input token predicted by
        # draft model is equal to previous output of model
        check_all = input_ids[:, 1:] == selected_tokens[:, :-1]
        # hacky way to ensure that we "stop early" if False is found in check_all
        check_cum = torch.cumsum(check_all, dim=1) * torch.cumprod(check_all, dim=1)
        valid_indices = torch.max(check_cum, dim=1).values

        # choose the best valid index and find number of tokens generated
        chosen_index = valid_indices.argmax().item()
        num_generated = valid_indices[chosen_index].item() + 1
        chosen_tokens = selected_tokens[None, chosen_index, :num_generated]

        # update output sequence
        output_sequence = torch.cat([output_sequence, chosen_tokens], dim=1)
        last_token = chosen_tokens[:, -1:]
        total_length += num_generated

        is_eos_token = output_sequence == eos_token_id
        if is_eos_token.any():
            # Find the index of the first eos token in output_sequence
            eos_index = torch.nonzero(is_eos_token)[0, 1].item()
            output_sequence = output_sequence[:, : eos_index + 1]
            break
        if total_length >= INP_LENGTH + GEN_LENGTH:
            output_sequence = output_sequence[
                :, : min(total_length, INP_LENGTH + GEN_LENGTH)
            ]
            break

        # update draft model inputs for next iteration
        last_input_token = input_ids[chosen_index, num_generated - 1]
        last_input_token = last_input_token.view(1, 1)
        inputs["input_ids"] = torch.cat([last_input_token, last_token], dim=1)
        inputs["attention_mask"] = attention_mask[:, :total_length]

        tree_index = chosen_index // (topk ** (num_tokens - num_generated + 1))
        chosen_past_key_values = []
        for layer in tree_past_key_values[num_generated - 1]:
            chosen_layer = []
            for item in layer:
                chosen_item = item[None, tree_index, :, :]
                chosen_layer.append(chosen_item)
            chosen_past_key_values.append(chosen_layer)
        inputs["past_key_values"] = chosen_past_key_values

        # update model past_key_values for next iteration
        chosen_model_past_key_values = []
        for layer in model_outputs.past_key_values:
            chosen_layer = []
            for item in layer:
                chosen_item = item[None, chosen_index, :, : total_length - 1]
                chosen_layer.append(chosen_item)
            chosen_model_past_key_values.append(chosen_layer)
        model_past_key_values = chosen_model_past_key_values

        # explicitly free up unused memory before next iteration
        torch.cuda.empty_cache()

    return output_sequence


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
        # print(tokenizer.batch_decode(gen_out)[0])

    avg_time = sum(gen_time) / sum(num_tokens) * 1000
    return avg_time
