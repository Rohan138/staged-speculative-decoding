import time

import torch
from tqdm import tqdm

INP_LENGTH = 256
GEN_LENGTH = 128
NUM_TOKENS = 5
DEPTH = 3


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


def speculative_decoding(
    inputs, model, draft_model, temperature=None, num_tokens=NUM_TOKENS
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


class DecodingNode:
    def __init__(self, inputs, draft_model, temperature=None, num_tokens=NUM_TOKENS):
        self.inputs = inputs
        self.draft_model = draft_model
        self.temperature = temperature
        self.num_tokens = num_tokens
        self.children = []
        self.last_token_eos = False
        self.build()

    def build(self):
        input_ids: torch.LongTensor = self.inputs["input_ids"]
        attention_mask: torch.LongTensor = self.inputs["attention_mask"]
        eos_token_id = self.draft_model.generation_config.eos_token_id

        # Check if the last token is an eos token
        if torch.any(input_ids[:, -1] == eos_token_id):
            self.last_token_eos = True
            return

        candidate_input_ids = input_ids
        outputs = self.draft_model(
            input_ids=candidate_input_ids,
            attention_mask=attention_mask,
        )
        next_token_logits = outputs.logits[:, -1, :]
        if self.temperature is not None:
            next_token_logits = next_token_logits / self.temperature

        next_token_logits, next_token_ids = next_token_logits.topk(
            self.num_tokens, dim=-1
        )
        self.next_token_logits = next_token_logits
        self.next_input_ids = torch.cat(
            (
                candidate_input_ids.expand(1, self.num_tokens, -1),
                next_token_ids.unsqueeze(-1),
            ),
            dim=-1,
        )
        is_eos_token = next_token_ids == eos_token_id
        self.attention_masks = torch.cat(
            (
                attention_mask.expand(1, self.num_tokens, -1),
                (~is_eos_token).unsqueeze(-1).long(),
            ),
            dim=-1,
        )

    def get_next_inputs(self):
        if self.last_token_eos:
            return []
        return [
            {
                "input_ids": self.next_input_ids[:, idx],
                "attention_mask": self.attention_masks[:, idx],
            }
            for idx in range(self.num_tokens)
        ]


class DecodingTree:
    def __init__(
        self, inputs, draft_model, temperature=None, num_tokens=NUM_TOKENS, depth=DEPTH
    ):
        self.inputs = inputs
        self.draft_model = draft_model
        self.temperature = temperature
        self.num_tokens = num_tokens
        self.depth = depth
        self.root = self.build()

    def build(self, depth=None):
        if depth is None:
            depth = self.depth
        root = DecodingNode(
            self.inputs, self.draft_model, self.temperature, self.num_tokens
        )
        if depth > 1:
            for next_input in root.get_next_inputs():
                tree = DecodingTree(
                    next_input,
                    self.draft_model,
                    self.temperature,
                    self.num_tokens,
                    depth - 1,
                )
                root.children.append(tree.root)
        return root


def _count_nodes(tree):
    return 1 + sum(_count_nodes(child) for child in tree.children)


def staged_speculative_decoding(
    inputs, model, draft_model, temperature=None, num_tokens=NUM_TOKENS, depth=DEPTH
):
    tree = DecodingTree(inputs, draft_model, temperature, num_tokens, depth)
    breakpoint()
    pass


def generate(
    model,
    tokenizer,
    dataset,
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

    for data in tqdm(dataset, desc=decoding.__name__):
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
        gen_out = decoding(inputs, model, draft_model, temperature)
        torch.cuda.synchronize()
        end = time.monotonic()

        gen_time.append(end - start)
        num_tokens.append(gen_out.shape[1] - inputs.input_ids.shape[1])

    avg_time = sum(gen_time) / sum(num_tokens) * 1000
    return avg_time
