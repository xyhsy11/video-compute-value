"""Explicit greedy policy, insulated from checkpoint generation defaults."""

from __future__ import annotations

from .common import Blocked

GENERATION_POLICY_VERSION = "explicit-greedy-v2"


def greedy_call_kwargs(generation, model_defaults):
    if generation.do_sample or generation.num_beams != 1 or generation.max_new_tokens != 8:
        raise Blocked("INPUT_INTEGRITY_FAILED", "Experiment 0 requires greedy, one beam, eight new tokens")
    if generation.eos_token_id is None or generation.pad_token_id is None:
        raise Blocked("INPUT_INTEGRITY_FAILED", "EOS and padding must resolve explicitly")
    return {"use_model_defaults": False, "do_sample": False, "num_beams": 1,
            "max_new_tokens": generation.max_new_tokens,
            "eos_token_id": generation.eos_token_id, "pad_token_id": generation.pad_token_id,
            "bos_token_id": generation.bos_token_id if generation.bos_token_id is not None else model_defaults.bos_token_id}


def resolve_greedy_policy(generation, model_defaults):
    from types import SimpleNamespace
    from transformers.generation.utils import GenerationMixin
    kwargs = greedy_call_kwargs(generation, model_defaults)
    holder = SimpleNamespace(generation_config=model_defaults)
    effective, unused = GenerationMixin._prepare_generation_config(holder, generation, **kwargs)
    if unused or effective.get_generation_mode().value != "greedy_search":
        raise Blocked("INPUT_INTEGRITY_FAILED", "Effective generation mode is not greedy")
    return {"version": GENERATION_POLICY_VERSION, "call_overrides": kwargs,
            "effective_generation": effective.to_dict(), "effective_mode": effective.get_generation_mode().value}
