"""Regression tests for the production generation boundary (CPU fixtures only)."""
from types import SimpleNamespace

import pytest

from video_compute_value.common import Blocked
from video_compute_value.generation import greedy_call_kwargs, resolve_greedy_policy


def configs():
    from transformers import GenerationConfig
    requested = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=8,
                                 eos_token_id=151645, pad_token_id=151643)
    checkpoint = GenerationConfig(do_sample=True, top_k=20, top_p=0.8, temperature=0.7,
                                  bos_token_id=151643, transformers_version="4.56.0")
    return requested, checkpoint


def test_effective_production_policy_is_greedy_despite_checkpoint_sampling():
    requested, checkpoint = configs()
    policy = resolve_greedy_policy(requested, checkpoint)
    assert policy["effective_mode"] == "greedy_search"
    assert policy["effective_generation"]["do_sample"] is False
    assert policy["call_overrides"]["use_model_defaults"] is False
    assert policy["effective_generation"]["bos_token_id"] == 151643
    assert checkpoint.do_sample is True and requested.bos_token_id is None


@pytest.mark.parametrize("field,value", [("do_sample", True), ("num_beams", 2), ("max_new_tokens", 9),
                                         ("eos_token_id", None), ("pad_token_id", None)])
def test_invalid_generation_policy_is_rejected(field, value):
    requested, checkpoint = configs()
    setattr(requested, field, value)
    with pytest.raises(Blocked):
        greedy_call_kwargs(requested, checkpoint)


def test_adapter_calls_generate_with_explicit_effective_policy():
    import torch
    from transformers.generation.utils import GenerationMixin
    from video_compute_value.model import QwenAdapter
    requested, checkpoint = configs()
    calls = []
    class Model:
        generation_config = checkpoint
        def generate(self, **kwargs):
            effective, unused = GenerationMixin._prepare_generation_config(
                self, kwargs.pop("generation_config"), **{k: v for k, v in kwargs.items() if k != "input_ids"})
            calls.append((effective, unused, kwargs))
            return torch.tensor([[123, 456, 65]])
    adapter = QwenAdapter.__new__(QwenAdapter)
    adapter.torch, adapter.model, adapter.generation = torch, Model(), requested
    adapter.generation_policy = resolve_greedy_policy(requested, checkpoint)
    adapter.model_call_count = 0
    adapter.processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *a, **k: "A"))
    result = adapter.generate({"input_ids": torch.tensor([[123, 456]])})
    assert calls[0][0].do_sample is False and calls[0][0].get_generation_mode().value == "greedy_search"
    assert calls[0][1] == {} and calls[0][2]["use_model_defaults"] is False
    assert result["parsed_answer"] == "A" and adapter.model_call_count == 1

def test_frozen_policy_roundtrip_matches_runtime_resolution():
    from transformers import GenerationConfig
    requested, checkpoint = configs()
    policy = resolve_greedy_policy(requested, checkpoint)
    frozen_generation = GenerationConfig(**policy["effective_generation"])
    assert resolve_greedy_policy(frozen_generation, checkpoint) == policy
