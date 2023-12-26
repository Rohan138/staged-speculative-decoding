import copy
import inspect
import warnings
from typing import TYPE_CHECKING, List, Optional, Union

import torch
import torch.distributed as dist
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import (
    GreedySearchDecoderOnlyOutput,
    GreedySearchEncoderDecoderOutput,
    _crop_past_key_values,
    _split_model_outputs,
)
from transformers.utils import logging

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer
    from transformers.generation.utils import GenerationMixin
    from transformers.modeling_utils import PreTrainedModel

logger = logging.get_logger(__name__)


def staged_assisted_decoding(
    self: "GenerationMixin",
    input_ids: torch.LongTensor,
    assistant_model: "PreTrainedModel",
    do_sample: bool = False,
    logits_processor: Optional[LogitsProcessorList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
):
    """Override `transformers.generation.utils.GenerationMixin.assisted_decoding`
    to enable staged speculative decoding"""
    # Assistant: initialize assistant-related variables
    num_assistant_tokens = assistant_model.generation_config.num_assistant_tokens
    topk_tokens = assistant_model.generation_config.topk_tokens

    # check if assistant model accepts encoder_outputs
    assistant_accepts_encoder_outputs = "encoder_outputs" in set(
        inspect.signature(assistant_model.forward).parameters.keys()
    )

    # init values
    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    logits_warper = (
        logits_warper if logits_warper is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    pad_token_id = (
        pad_token_id
        if pad_token_id is not None
        else self.generation_config.pad_token_id
    )
    eos_token_id = (
        eos_token_id
        if eos_token_id is not None
        else self.generation_config.eos_token_id
    )
    if eos_token_id is not None and pad_token_id is None:
        raise ValueError(
            "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
        )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = (
        torch.tensor(eos_token_id).to(input_ids.device)
        if eos_token_id is not None
        else None
    )
    output_scores = (
        output_scores
        if output_scores is not None
        else self.generation_config.output_scores
    )
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

    # other auxiliary variables
    max_len = stopping_criteria[0].max_length
    assistant_kv_indexing = (
        1
        if "bloom" in assistant_model.__class__.__name__.lower()
        or (
            assistant_model.config.architectures is not None
            and "bloom" in assistant_model.config.architectures[0].lower()
        )
        else 0
    )

    this_peer_finished = False  # used by synced_gpus only
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(
                0.0 if this_peer_finished else 1.0
            ).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # Assistant: main logic start
        cur_len = input_ids.shape[-1]

        #  1. Forecast next N tokens using the assistant model. This `for` block can be replaced with a
        # `.generate()` call if we decide to add `past_key_values` as a possible output of generate, as we
        # need access to the assistant cache to secure strong speedups.
        candidate_input_ids = input_ids
        for _ in range(int(num_assistant_tokens)):
            # 1.1. use the assistant model to obtain the next candidate logits
            if "assistant_past_key_values" in model_kwargs:
                prev_seq_len = model_kwargs["assistant_past_key_values"][0][
                    assistant_kv_indexing
                ].shape[-2]
                # `new_token_len` can be 1 or 2 (next token in assistant + last token picked by the larger model)
                new_token_len = candidate_input_ids.shape[1] - prev_seq_len
                assist_inputs = candidate_input_ids[:, -new_token_len:]
                # TODO (joao): make it compatible with models that use unconventional fwd pass logic, like blip2
                if assistant_model.config.is_encoder_decoder:
                    assistant_model_outputs = assistant_model(
                        decoder_input_ids=assist_inputs,
                        past_key_values=model_kwargs["assistant_past_key_values"],
                        encoder_outputs=model_kwargs["assistant_encoder_outputs"],
                    )
                else:
                    encoder_kwargs = {}

                    if (
                        assistant_accepts_encoder_outputs
                        and "assistant_encoder_outputs" in model_kwargs
                    ):
                        encoder_kwargs["encoder_outputs"] = model_kwargs[
                            "assistant_encoder_outputs"
                        ]

                    assistant_model_outputs = assistant_model(
                        assist_inputs,
                        past_key_values=model_kwargs["assistant_past_key_values"],
                        **encoder_kwargs,
                    )
            else:
                if assistant_model.config.is_encoder_decoder:
                    assistant_model_outputs = assistant_model(
                        decoder_input_ids=candidate_input_ids,
                        encoder_outputs=model_kwargs["assistant_encoder_outputs"],
                    )
                else:
                    encoder_kwargs = {}

                    if (
                        assistant_accepts_encoder_outputs
                        and "assistant_encoder_outputs" in model_kwargs
                    ):
                        encoder_kwargs["encoder_outputs"] = model_kwargs[
                            "assistant_encoder_outputs"
                        ]

                    assistant_model_outputs = assistant_model(
                        candidate_input_ids, **encoder_kwargs
                    )

            # 1.2. greedily select the next candidate token
            model_kwargs[
                "assistant_past_key_values"
            ] = assistant_model_outputs.past_key_values
            if len(logits_processor) > 0:
                assistant_model_outputs.logits[:, -1, :] = logits_processor(
                    candidate_input_ids, assistant_model_outputs.logits[:, -1, :]
                )
            new_token = assistant_model_outputs.logits[:, -1, :].argmax(dim=-1)
            candidate_input_ids = torch.cat(
                (candidate_input_ids, new_token[:, None]), dim=-1
            )

            # 1.3. stop assistant generation on EOS
            if eos_token_id_tensor is not None:
                last_assistant_token_is_eos = new_token.tile(
                    eos_token_id_tensor.shape[0], 1
                )
                last_assistant_token_is_eos = (
                    ~last_assistant_token_is_eos.ne(eos_token_id_tensor.unsqueeze(1))
                    .prod(dim=0)
                    .bool()
                )
                if last_assistant_token_is_eos:
                    break
            else:
                last_assistant_token_is_eos = False

        candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]

        # 2. Use the original model to obtain the next token logits given the candidate sequence. We obtain
        # `candidate_length + 1` relevant logits from this process: in the event that all candidates are correct,
        # we use this forward pass to also pick the subsequent logits in the original model.

        # 2.1. Prepare the model inputs
        candidate_kwargs = copy.copy(model_kwargs)
        candidate_kwargs = self._extend_attention_mask(
            candidate_kwargs, candidate_input_ids.shape[1]
        )
        candidate_kwargs = self._extend_token_type_ids(
            candidate_kwargs, candidate_input_ids.shape[1]
        )

        model_inputs = self.prepare_inputs_for_generation(
            candidate_input_ids, **candidate_kwargs
        )

        # 2.2. Run a forward pass on the candidate sequence
        outputs = self(
            **model_inputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # 2.3. Process the new logits
        new_logits = outputs.logits[
            :, -candidate_length - 1 :
        ]  # excludes the input prompt if present
        if len(logits_processor) > 0:
            for i in range(candidate_length + 1):
                new_logits[:, i, :] = logits_processor(
                    candidate_input_ids[:, : cur_len + i], new_logits[:, i, :]
                )
        if len(logits_warper) > 0:
            for i in range(candidate_length + 1):
                new_logits[:, i, :] = logits_warper(
                    candidate_input_ids[:, : cur_len + i], new_logits[:, i, :]
                )

        # 3. Obtain the next tokens from the original model logits.
        if do_sample:
            probs = new_logits.softmax(dim=-1)
            selected_tokens = torch.multinomial(probs[0, :, :], num_samples=1).squeeze(
                1
            )[None, :]
        else:
            selected_tokens = new_logits.argmax(dim=-1)

        # 4. Compare the argmax from the original model logits with the assistant forecasted tokens. We can keep
        # the assistant forecasted tokens until the first mismatch, or until the max length is reached.
        candidate_new_tokens = candidate_input_ids[:, -candidate_length:]
        n_matches = (
            (~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1
        ).sum()

        # 5. Update variables according to the number of matching assistant tokens. Remember: the token generated
        # by the model after the last candidate match is also valid, as it is generated from a correct sequence.
        # Because of this last token, assisted generation search reduces to a normal greedy search/sample if there
        # is no match.

        # 5.1. Ensure we don't generate beyond max_len or an EOS token
        if last_assistant_token_is_eos and n_matches == candidate_length:
            n_matches -= 1
        n_matches = min(n_matches, max_len - cur_len - 1)

        # 5.2. Get the valid continuation, after the matching tokens
        valid_tokens = selected_tokens[:, : n_matches + 1]
        input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
        if streamer is not None:
            streamer.put(valid_tokens.cpu())
        new_cur_len = input_ids.shape[-1]

        # 5.3. Discard past key values relative to unused assistant tokens
        new_cache_size = new_cur_len - 1
        outputs.past_key_values = _crop_past_key_values(
            self, outputs.past_key_values, new_cache_size
        )
        model_kwargs["assistant_past_key_values"] = _crop_past_key_values(
            assistant_model,
            model_kwargs["assistant_past_key_values"],
            new_cache_size - 1,
        )  # the assistant does not have the token after the last match, hence the -1

        # 6. Adjust the max number of assistant tokens to use in the next iteration. This is a simple heuristic,
        # probably can be improved -- we want to balance the benefits of getting assistant tokens correct with the
        # cost of forecasting incorrect assistant tokens.
        if (
            assistant_model.generation_config.num_assistant_tokens_schedule
            == "heuristic"
        ):
            if n_matches == int(num_assistant_tokens):
                num_assistant_tokens += 2.0
            else:
                num_assistant_tokens = max(1.0, num_assistant_tokens - 1.0)

        # Assistant: main logic end
        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        # Store scores, attentions and hidden_states when required
        # Assistant: modified to append one tuple element per token, as in the other generation methods.
        if return_dict_in_generate:
            if output_scores:
                scores += tuple(new_logits[:, i, :] for i in range(n_matches + 1))

            if "past_key_values" not in model_kwargs:
                added_len = new_cur_len
            else:
                added_len = n_matches + 1

            if output_attentions:
                if self.config.is_encoder_decoder:
                    cross_attentions = _split_model_outputs(
                        cross_attentions, outputs.cross_attentions, cur_len, added_len
                    )
                    decoder_attentions = _split_model_outputs(
                        decoder_attentions,
                        outputs.decoder_attentions,
                        cur_len,
                        added_len,
                        is_decoder_attention=True,
                    )
                else:
                    decoder_attentions = _split_model_outputs(
                        decoder_attentions,
                        outputs.attentions,
                        cur_len,
                        added_len,
                        is_decoder_attention=True,
                    )
            if output_hidden_states:
                if self.config.is_encoder_decoder:
                    decoder_hidden_states = _split_model_outputs(
                        decoder_hidden_states,
                        outputs.decoder_hidden_states,
                        cur_len,
                        added_len,
                    )
                else:
                    decoder_hidden_states = _split_model_outputs(
                        decoder_hidden_states, outputs.hidden_states, cur_len, added_len
                    )

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                input_ids[:, -1]
                .tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GreedySearchEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return GreedySearchDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids
