"""Explicit generation controls: prevent checkpoint defaults from enabling sampling."""

from __future__ import annotations

from types import SimpleNamespace

from video_compute_value.common import file_hash, write_json


def fixed_greedy_kwargs(adapter):
    generation = adapter.generation
    model_defaults = adapter.model.generation_config
    return {"use_model_defaults": False, "do_sample": False, "num_beams": 1,
            "max_new_tokens": generation.max_new_tokens,
            "eos_token_id": generation.eos_token_id, "pad_token_id": generation.pad_token_id,
            "bos_token_id": generation.bos_token_id if generation.bos_token_id is not None else model_defaults.bos_token_id}


def audit_generation_defaults(parent, output):
    import inspect
    import yaml
    from transformers import GenerationConfig
    from transformers.generation.utils import GenerationMixin
    frozen = yaml.safe_load((parent / "resolved_config.yaml").read_text(encoding="utf-8"))
    requested = GenerationConfig(**frozen["generation"])
    defaults = GenerationConfig.from_pretrained(frozen["model_path"], local_files_only=True)
    holder = SimpleNamespace(generation_config=defaults)
    inherited, _ = GenerationMixin._prepare_generation_config(holder, requested, use_model_defaults=None)
    adapter = SimpleNamespace(generation=requested, model=holder)
    kwargs = fixed_greedy_kwargs(adapter)
    fixed, _ = GenerationMixin._prepare_generation_config(holder, requested, **kwargs)
    data = {"requested": requested.to_dict(), "checkpoint_defaults": defaults.to_dict(),
            "legacy_effective": inherited.to_dict(), "fixed_call_kwargs": kwargs,
            "fixed_effective": fixed.to_dict(), "legacy_generation_mode": inherited.get_generation_mode().value,
            "fixed_generation_mode": fixed.get_generation_mode().value,
            "installed_generation_source_sha256": file_hash(inspect.getfile(GenerationMixin)),
            "reference": "https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/generation/utils.py",
            "cause": "Supplied global-default-valued fields were replaced by checkpoint defaults when use_model_defaults was omitted"}
    assert requested.do_sample is False and inherited.do_sample is True and fixed.do_sample is False
    write_json(output, data)
    return data
