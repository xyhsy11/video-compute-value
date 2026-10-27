"""Diagnostic logging must preserve masked/infinite values explicitly."""

import importlib.util
import json
from pathlib import Path


def diagnostic_module():
    path = Path(__file__).parents[1] / "scripts" / "diagnose_e000.py"
    spec = importlib.util.spec_from_file_location("diagnose_e000_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nonfinite_diagnostic_scores_remain_distinguishable():
    safe = diagnostic_module().json_safe({"scores": [1.25, float("-inf"), float("inf"), float("nan")], "missing": None})
    assert json.loads(json.dumps(safe, allow_nan=False)) == {
        "scores": [1.25, "-Infinity", "+Infinity", "NaN"], "missing": None}


def test_checkpoint_sampling_defaults_cannot_override_fixed_greedy_call():
    import sys
    from types import SimpleNamespace
    from transformers import GenerationConfig
    from transformers.generation.utils import GenerationMixin
    scripts = str(Path(__file__).parents[1] / "scripts")
    sys.path.insert(0, scripts)
    from diagnostic_generation import fixed_greedy_kwargs
    requested = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=8, eos_token_id=151645, pad_token_id=151643)
    checkpoint = GenerationConfig(do_sample=True, top_k=20, top_p=0.8, temperature=0.7,
                                  bos_token_id=151643, transformers_version="4.56.0")
    holder = SimpleNamespace(generation_config=checkpoint)
    legacy, _ = GenerationMixin._prepare_generation_config(holder, requested, use_model_defaults=None)
    assert legacy.do_sample is True  # Reproduces the actual version-specific failure without a GPU.
    adapter = SimpleNamespace(generation=requested, model=holder)
    fixed, _ = GenerationMixin._prepare_generation_config(holder, requested, **fixed_greedy_kwargs(adapter))
    assert fixed.do_sample is False and fixed.num_beams == 1 and fixed.max_new_tokens == 8
    assert fixed.get_generation_mode().value == "greedy_search"
    assert requested.do_sample is False  # The declared config itself was not the bug.
