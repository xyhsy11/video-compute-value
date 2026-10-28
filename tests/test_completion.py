"""Regression fixtures for completion gates, execution policies and persistence."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from video_compute_value.common import Blocked, atomic_text, digest, read_json, write_json
from video_compute_value.analysis import generation_receipts_valid, repeatability_gates


def test_repeatability_requires_tokens_and_raw_outputs():
    condition = {"prepared_input_agreement": True, "parsed_answer_agreement": True,
                 "raw_token_agreement": True, "raw_output_agreement": True}
    assert all(repeatability_gates(True, [condition]).values())
    for field in ("raw_token_agreement", "raw_output_agreement"):
        changed = {**condition, field: False}
        assert not all(repeatability_gates(True, [changed]).values())
    assert not any(repeatability_gates(False, [condition]).values())


def test_generation_gate_requires_effective_policy_and_worker_receipt(tmp_path):
    from transformers import GenerationConfig
    from video_compute_value.generation import resolve_greedy_policy
    checkpoint = GenerationConfig(do_sample=True, temperature=.7, bos_token_id=151643, transformers_version="4.56.0")
    model = tmp_path / "model"
    checkpoint.save_pretrained(model)
    generation = GenerationConfig(do_sample=False, max_new_tokens=8, eos_token_id=151645, pad_token_id=151643)
    policy = resolve_greedy_policy(generation, checkpoint)
    frozen = {"model_path": str(model), "generation": generation.to_dict(), "generation_policy": policy}
    row = {"worker_key": "collect_0_123", "generation_policy_sha256": digest(policy)}
    receipt = tmp_path / "worker_collect_0_123.json"
    write_json(receipt, {"generation_policy": policy})
    assert generation_receipts_valid(frozen, [row], tmp_path)
    assert not generation_receipts_valid({}, [row], tmp_path)
    assert not generation_receipts_valid(frozen, [{**row, "generation_policy_sha256": "bad"}], tmp_path)
    write_json(receipt, {"generation_policy": {**policy, "effective_mode": "sample"}})
    assert not generation_receipts_valid(frozen, [row], tmp_path)


def test_atomic_replace_retries_only_permission_errors(tmp_path, monkeypatch):
    import video_compute_value.common as common
    native = common.os.replace
    attempts, sleeps = [], []
    def replace(a, b):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError("transient reader lock")
        return native(a, b)
    monkeypatch.setattr(common.os, "replace", replace)
    monkeypatch.setattr(common.time, "sleep", sleeps.append)
    target = tmp_path / "result.json"
    atomic_text(target, "valid")
    assert target.read_text() == "valid" and len(attempts) == 3 and len(sleeps) == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_storage_scan_runtime_is_charged(tmp_path, monkeypatch):
    import video_compute_value.resources as resources
    clock = [10.0]
    monkeypatch.setattr(resources.time, "monotonic", lambda: clock[0])
    def scan(root):
        clock[0] += 5
        return 100
    monkeypatch.setattr(resources, "tree_bytes", scan)
    ledger = resources.Ledger(tmp_path / "ledger.json", tmp_path,
        {"cumulative_execution_limit_hours": 4, "additional_disk_limit_gib": 20, "video_acquisition_limit_gib": 1})
    with ledger.track("synthetic-accounting-fixture"):
        clock[0] += 3
    assert read_json(ledger.path)["active_seconds"] == 13


def test_versioned_execution_policy_does_not_relax_core_protocol():
    import yaml
    from video_compute_value.protocol import validate_config
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/e000.yaml").read_text(encoding="utf-8"))
    changed = {**config, "decoder_threads": 4, "decoder_thread_type": "FRAME", "cuda_trim_budgets": [64]}
    with pytest.raises(Blocked):
        validate_config(changed)
    changed["execution_policy_version"] = "threaded-decode-trim64-v2"
    assert validate_config(changed) == changed
    with pytest.raises(Blocked):
        validate_config({**changed, "num_videos": 7})


def test_frame_threading_preserves_real_decoder_pixels_on_synthetic_clip(tmp_path):
    import av
    from video_compute_value.video import inspect_video, decode_selected
    path = tmp_path / "threading fixture.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=16)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for i in range(96):
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.full((64, 64, 3), i, dtype=np.uint8), format="rgb24")):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    video = inspect_video(path, [8, 64], 65536)
    for budget in (8, 64):
        baseline, a = decode_selected(video, budget)
        candidate, b = decode_selected(video, budget, decoder_threads=4, decoder_thread_type="FRAME")
        assert np.array_equal(baseline, candidate)
        assert a["source_indices"] == b["source_indices"] and a["actual_timestamps"] == b["actual_timestamps"]


def test_uniform64_trim_runs_inside_generate_before_model(monkeypatch):
    import torch
    from transformers import GenerationConfig
    from video_compute_value.model import QwenAdapter
    from video_compute_value.generation import resolve_greedy_policy
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("trim"))
    generation = GenerationConfig(max_new_tokens=8, eos_token_id=151645, pad_token_id=151643)
    adapter = QwenAdapter.__new__(QwenAdapter)
    adapter.torch, adapter.generation, adapter.model_call_count = torch, generation, 0
    adapter.frozen = {"cuda_trim_budgets": [64]}
    adapter.generation_policy = resolve_greedy_policy(generation, GenerationConfig(bos_token_id=151643))
    def generate(**kwargs):
        calls.append("generate")
        return torch.tensor([[1, 2, 65]])
    adapter.model = SimpleNamespace(generate=generate)
    adapter.processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *a, **k: "A"))
    for budget, expected in ((8, ["generate"]), (64, ["trim", "generate"])):
        calls.clear()
        adapter._prepared_budget = budget
        result = adapter.generate({"input_ids": torch.tensor([[1, 2]])})
        assert calls == expected and result["cuda_cache_trimmed"] is (budget == 64)
        assert result["generation_policy_sha256"] == digest(adapter.generation_policy)


def test_full_projection_includes_probes_warmups_startups_retries_and_margin():
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    from forecast_e000_completion import forecast_collection
    result = forecast_collection({b: 2 for b in (8, 16, 32, 48, 64)},
        {b: 3 for b in (8, 16, 32, 48, 64)}, 10, 1, 8, 5000,
        verification_seconds=100, retry_reserve_seconds=600, setup_seconds=50)
    assert result["planned_model_invocations_before_retries"] == 441
    assert result["video_call_seconds"] == 415 * 5
    assert result["text_call_seconds"] == 26
    assert result["worker_startup_seconds"] == 210
    assert result["raw_projected_seconds"] == 3069
    assert result["projected_seconds"] == 4603.5 and result["fits_remaining_budget"]
    assert not forecast_collection({b: 2 for b in (8, 16, 32, 48, 64)},
        {b: 3 for b in (8, 16, 32, 48, 64)}, 10, 1, 8, 100)["fits_remaining_budget"]


def test_atomic_persistent_failure_preserves_previous_result(tmp_path, monkeypatch):
    import video_compute_value.common as common
    target = tmp_path / "result.json"
    target.write_text("prior evidence", encoding="utf-8")
    attempts = []
    def denied(*args):
        attempts.append(1)
        raise PermissionError("persistent access failure")
    monkeypatch.setattr(common.os, "replace", denied)
    monkeypatch.setattr(common.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError):
        atomic_text(target, "replacement")
    assert len(attempts) == 6 and target.read_text(encoding="utf-8") == "prior evidence"
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_verify_returns_failure_for_unpassed_operational_gates(tmp_path, monkeypatch):
    import video_compute_value.e000 as cli
    import video_compute_value.analysis as analysis
    config = tmp_path / "config.yaml"
    config.write_text((Path(__file__).parents[1] / "configs/e000.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(cli, "set_local_paths", lambda root: None)
    monkeypatch.setattr(analysis, "analyze", lambda *args: None)
    monkeypatch.setattr(analysis, "verify", lambda *args: {"records_valid": True, "all_operational_gates_passed": False})
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    assert cli.main(["verify", "--config", str(config), "--run-dir", str(tmp_path / "run"), "--root", str(tmp_path)]) == 3


@pytest.mark.parametrize("case", ["missing", "older", "active_other", "inherited"])
def test_cli_refuses_fresh_or_forked_resource_allowance(tmp_path, case):
    from video_compute_value.e000 import require_cumulative_ledger
    old = tmp_path / "results/old/resource_ledger.json"
    write_json(old, {"active_seconds": 11605, "sessions": []})
    run = tmp_path / "results/new"
    run.mkdir()
    if case != "missing":
        write_json(run / "resource_ledger.json", {"active_seconds": 100 if case == "older" else 11605})
    if case == "active_other":
        write_json(old, {"active_seconds": 11605, "sessions": [{"finished_at": None}]})
    if case == "inherited":
        require_cumulative_ledger(run, tmp_path)
    else:
        with pytest.raises(Blocked):
            require_cumulative_ledger(run, tmp_path)


def test_cpu_audit_refuses_to_overwrite_existing_frozen_evidence(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    import benchmark_e000_cpu as audit
    monkeypatch.setattr(audit, "RUN", tmp_path)
    write_json(tmp_path / "cpu_plan.json", {"frozen": True})
    with pytest.raises(Blocked, match="immutable"):
        audit.main()
    assert read_json(tmp_path / "cpu_plan.json") == {"frozen": True}


def test_standalone_audit_helper_preserves_and_charges_cumulative_ledger(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    import benchmark_e000_cpu as audit
    from video_compute_value.resources import Ledger
    monkeypatch.setattr(audit, "RUN", tmp_path)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    path = tmp_path / "resource_ledger.json"
    ledger = Ledger(path, tmp_path, {"cumulative_execution_limit_hours": 4, "additional_disk_limit_gib": 20, "video_acquisition_limit_gib": 1})
    ledger.value["active_seconds"] = 100
    ledger.save()
    assert audit.accounted_call(lambda: 42, "synthetic-helper-accounting") == 42
    assert read_json(path)["active_seconds"] >= 100
    assert read_json(path)["sessions"][-1]["outcome"] == "completed"


@pytest.mark.parametrize("hours,authorization,accepted", [(16 / 3, "user_20261028_5h20m", True), (16 / 3, None, False), (6, "user_20261028_5h20m", False)])
def test_authorized_time_limit_is_explicit_and_bounded(hours, authorization, accepted):
    import yaml
    from video_compute_value.protocol import validate_config
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/e000.yaml").read_text(encoding="utf-8"))
    config.update(cumulative_execution_limit_hours=hours, resource_authorization_id=authorization)
    if accepted:
        assert validate_config(config) == config
    else:
        with pytest.raises(Blocked):
            validate_config(config)


def test_probe_requires_exact_tokens_even_when_answer_matches():
    from video_compute_value.runner import validate_probe_pair
    first = {"input_tensor_hashes": {"x": "same"}, "input_metadata_hash": "same",
             "generated_token_ids": [1], "decoded_output": "A", "parsed_answer": "A", "generation_policy_sha256": "greedy"}
    validate_probe_pair(first, dict(first))
    with pytest.raises(Blocked):
        validate_probe_pair(first, {**first, "generated_token_ids": [1, 2]})


def test_profile_projection_counts_remaining_collection_only():
    from video_compute_value.runner import collection_projection
    probes = [{"budget": b, "worker_wall_seconds": 20, "timed_calls_seconds": 10,
               "component_profile": {"latency_seconds": 3, "components_seconds": {"cpu_prepare": 2}}}
              for b in (8, 16, 32, 48, 64)]
    cpu = {f"v{i}": {"elapsed_seconds": 2} for i in range(8)}
    result = collection_projection(probes, cpu, "v0")
    assert result["estimated_video_call_seconds"] == 5 * 78 * 3
    assert result["raw_estimate_seconds"] == 1170 + 26 + 160 + 104 + 180 + 600
    assert result["projection_seconds"] == 1.5 * result["raw_estimate_seconds"]


def test_finalizer_cannot_reopen_older_allowance_after_followup(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    import finalize_e000_v3 as finalizer
    from video_compute_value.common import read_json
    run = tmp_path / "results/collection"
    run.mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/e000_threaded_authorized_v3.yaml").write_text("{}", encoding="utf-8")
    write_json(run / "runtime_resource_grant.json", {"authorized_limit_seconds": 20400, "human_reply": "Authorize 5h40m cumulative time"})
    write_json(run / "resource_ledger.json", {"active_seconds": 18000, "sessions": []})
    write_json(tmp_path / "results/followup/resource_ledger.json", {"active_seconds": 18600, "sessions": []})
    monkeypatch.setattr(finalizer, "ROOT", tmp_path)
    monkeypatch.setattr(finalizer, "RUN", run)
    with pytest.raises(Blocked, match="Inherit the latest"):
        finalizer.main()
    assert read_json(run / "resource_ledger.json")["active_seconds"] == 18000
