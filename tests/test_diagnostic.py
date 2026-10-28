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


def test_timing_plan_is_matched_and_preserves_all_reference_questions(tmp_path, monkeypatch):
    import sys
    from video_compute_value.common import write_json, digest
    from video_compute_value.protocol import make_order
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    import diagnose_e000_timing as timing
    monkeypatch.setattr(timing, "source_manifest", lambda root: {})
    parent, run = tmp_path / "parent", tmp_path / "diagnostic"
    parent.mkdir()
    run.mkdir()
    cohort = {"videos": [{"source_id": f"v{i}", "questions": [{"question_id": f"q{i}-{j}"} for j in range(3)]}
                         for i in range(8)]}
    base = {"seed": 20261027, "frame_budgets": [8, 16, 32, 48, 64], "repetitions": 3}
    write_json(parent / "cohort_manifest.json", cohort)
    write_json(parent / "execution_order.json", make_order(base, cohort))
    for budget in (8, 64):
        for repetition in (1, 2, 3):
            for v in cohort["videos"]:
                for q in v["questions"]:
                    row = {"source_id": v["source_id"], "question_id": q["question_id"], "budget": budget,
                           "repetition": repetition, "latency_seconds": 1.0}
                    write_json(parent / "records" / (digest(row) + ".json"), row)
    write_json(parent / "probes.json", {"probes": [{"budget": b, "worker_wall_seconds": 10,
        "timed_calls_seconds": 5, "component_profile": {"latency_seconds": 1.0}} for b in (8, 64)]})
    warmups = [{"budget": b, "latency_seconds": 1.0} for b in (8, 64)]
    (parent / "diagnostics").mkdir()
    (parent / "diagnostics" / "warmups.jsonl").write_text("\n".join(json.dumps(r) for r in warmups))
    write_json(run / "generation_defaults_audit.json", {"fixed_effective": {"do_sample": False}})
    plan = timing.create_timing_plan(tmp_path, run, {"parent_run": "parent", "timing_budgets": [8, 64]})
    assert (plan["timed_requests"], plan["warmups"], plan["component_profiles"]) == (192, 24, 4)
    assert len(plan["blocks"]) == 12
    for i in range(0, len(plan["blocks"]), 2):
        left, right = plan["blocks"][i:i + 2]
        assert left["questions"] == right["questions"] and left["arm"] != right["arm"]
        assert len(left["questions"]) == (24 if left["budget"] == 64 else 8)
    assert plan["projected_seconds_with_safety"] == 585


def test_strict_adapter_enforces_call_overrides(monkeypatch):
    import sys
    from types import SimpleNamespace
    from transformers import GenerationConfig
    from transformers.generation.utils import GenerationMixin
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    import diagnose_e000_timing as timing
    class FakeModel:
        generation_config = GenerationConfig(do_sample=True, top_p=0.8, temperature=0.7,
                                               bos_token_id=151643, transformers_version="4.56.0")
        def _prepare_generation_config(self, requested, **kwargs):
            return GenerationMixin._prepare_generation_config(self, requested, **kwargs)
        def generate(self, **kwargs):
            return kwargs
    def initialize(adapter, frozen):
        adapter.generation = GenerationConfig(do_sample=False, max_new_tokens=8, eos_token_id=151645, pad_token_id=151643)
        adapter.model = FakeModel()
    monkeypatch.setattr(timing.QwenAdapter, "__init__", initialize)
    adapter = timing.StrictGreedyAdapter(SimpleNamespace())
    actual = adapter.model.generate(do_sample=True)
    assert actual["do_sample"] is False and actual["use_model_defaults"] is False
    assert adapter.effective_generation["do_sample"] is False
