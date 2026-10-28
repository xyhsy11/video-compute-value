"""CPU-only real-cohort equivalence and throughput audit; never loads model weights."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time
from types import SimpleNamespace

import yaml

from video_compute_value.common import Blocked, digest, file_hash, read_json, write_json, utc_now
from video_compute_value.e000 import set_local_paths
from video_compute_value.protocol import question_input, video_input

ROOT = Path.cwd()
RUN = ROOT / "results/e000_20261028_completion_v1"
PARENT = ROOT / "results/e000_20261027_collection_v2"


def remaining_seconds():
    ledger = read_json(RUN / "resource_ledger.json")
    live = sum((datetime.now(timezone.utc) - datetime.fromisoformat(s["started_at"])).total_seconds()
               for s in ledger["sessions"] if s.get("finished_at") is None)
    return 14400 - ledger["active_seconds"] - live


def inventory(path):
    return {p.relative_to(path).as_posix(): file_hash(p) for p in sorted(path.rglob("*"))
            if p.is_file() and p.name not in (".lock", ".worker.lock")}


def main():
    if (RUN / "cpu_plan.json").exists():
        raise Blocked("INPUT_INTEGRITY_FAILED", "Existing CPU audit is immutable; use a new named run rather than overwrite it")
    if remaining_seconds() < 900:
        raise Blocked("BLOCKED_RESOURCE", "CPU audit requires a 600-second cap plus 300-second closeout reserve")
    started = time.monotonic()
    deadline = started + 600
    set_local_paths(ROOT)
    import torch
    import numpy as np
    from transformers import AutoConfig, AutoProcessor
    from video_compute_value.model import QwenAdapter, adapter_source_hashes, tensor_hashes
    from video_compute_value.video import decode_selected
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    parent = yaml.safe_load((PARENT / "resolved_config.yaml").read_text(encoding="utf-8"))
    cohort = read_json(PARENT / "cohort_manifest.json")
    sources = {str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p)
               for p in [*sorted((ROOT / "src/video_compute_value").glob("*.py")), Path(__file__).resolve()]}
    history = {name: inventory(ROOT / "results" / name) for name in
               ("e000_20261027_collection_v2", "e000_20261027_diagnostic_v1", "e000_20261028_diagnostic_v2")}
    write_json(RUN / "private/historical_inventory_before.json", history)
    write_json(RUN / "cpu_plan.json", {"frozen_at": utc_now(), "sources": sources,
        "parent_cohort_sha256": digest(cohort), "candidate_config_sha256": file_hash(ROOT / "configs/e000_threaded_v2.yaml"),
        "budgets": [8, 16, 32, 48, 64], "raw_rgb_pairs": 16, "candidate_preparations": 40,
        "direct_tensor_pairs": 4, "model_calls": 0, "gpu_context": False,
        "maximum_cpu_audit_seconds": 600, "remaining_before_seconds": remaining_seconds(),
        "selection": "All 8 original videos, first official question per source, all 5 budgets; direct raw RGB equality at 8 and 64",
        "boundary": "CPU-only diagnostic; not primary end-to-end latency. Every prepare/decode starts from the compressed video; no cross-request decoded reuse"})
    def guard():
        if time.monotonic() > deadline or remaining_seconds() < 300:
            raise Blocked("BLOCKED_RESOURCE", "Bounded CPU audit stopped before the closeout reserve")
    raw_rows = []
    for index, labelled in enumerate(cohort["videos"]):
        video = video_input(labelled)
        for budget in (8, 64):
            order = [(1, "NONE"), (4, "FRAME")]
            if (index + budget) % 2:
                order.reverse()
            outputs = []
            for threads, kind in order:
                guard()
                tick = time.perf_counter_ns()
                frames, details = decode_selected(video, budget, decoder_threads=threads, decoder_thread_type=kind)
                elapsed = (time.perf_counter_ns() - tick) / 1e9
                outputs.append((frames, details))
                raw_rows.append({"source_id": labelled["source_id"], "budget": budget, "decoder_threads": threads,
                    "thread_type": kind, "seconds": elapsed, "pixels_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
                    "frames_decoded": details["frames_decoded"], "source_indices": details["source_indices"],
                    "timestamps": details["actual_timestamps"]})
            assert np.array_equal(outputs[0][0], outputs[1][0])
            assert outputs[0][1]["source_indices"] == outputs[1][1]["source_indices"]
            assert outputs[0][1]["actual_timestamps"] == outputs[1][1]["actual_timestamps"]
            del outputs, frames
            write_json(RUN / "private/cpu_decode_records.json", raw_rows)
        print(f"Raw RGB equivalence: source {index + 1}/8", flush=True)
    adapter = QwenAdapter.__new__(QwenAdapter)
    adapter.torch = torch
    adapter.processor = AutoProcessor.from_pretrained(parent["model_path"], local_files_only=True, trust_remote_code=False, use_fast=True)
    adapter.model = SimpleNamespace(config=AutoConfig.from_pretrained(parent["model_path"], local_files_only=True, trust_remote_code=False))
    assert adapter_source_hashes() == parent["processor_source_hashes"]
    assert digest(adapter.processor.chat_template) == parent["chat_template_hash"]
    references = {(r["source_id"], r["question_id"], r["budget"]): r["input_tensor_hashes"]
                  for r in (read_json(p) for p in (PARENT / "records").glob("*.json"))}
    prepare_rows = []
    for index, labelled in enumerate(cohort["videos"]):
        question = question_input(labelled["questions"][0])
        video = video_input(labelled)
        for budget in (8, 16, 32, 48, 64):
            guard()
            adapter.frozen = {**parent, "decoder_threads": 4, "decoder_thread_type": "FRAME"}
            tick = time.perf_counter_ns()
            inputs, metadata = adapter.prepare(question, video, budget)
            elapsed = (time.perf_counter_ns() - tick) / 1e9
            hashes = tensor_hashes(inputs)
            assert hashes == references[(labelled["source_id"], question.question_id, budget)]
            direct = index in (0, 7) and budget in (8, 64)
            if direct:
                adapter.frozen = parent
                baseline, baseline_metadata = adapter.prepare(question, video, budget)
                assert set(baseline) == set(inputs) and all(torch.equal(inputs[k], baseline[k]) for k in inputs)
                assert metadata["source_indices"] == baseline_metadata["source_indices"]
                assert metadata["model_timestamps"] == baseline_metadata["model_timestamps"]
                del baseline
            prepare_rows.append({"source_id": labelled["source_id"], "question_id": question.question_id,
                "budget": budget, "seconds": elapsed, "input_tensor_hashes": hashes,
                "source_indices": metadata["source_indices"], "model_timestamps": metadata["model_timestamps"],
                "direct_tensor_comparison": direct})
            del inputs
            write_json(RUN / "private/cpu_prepare_records.json", prepare_rows)
        print(f"Native processor input equivalence: source {index + 1}/8", flush=True)
    assert not torch.cuda.is_initialized()
    write_json(RUN / "cpu_verification.json", {"passed": True, "raw_rgb_equal_pairs": 16,
        "exact_parent_input_hash_matches": len(prepare_rows), "direct_equal_tensor_pairs": 4,
        "all_original_videos": 8, "all_frame_budgets": [8, 16, 32, 48, 64], "model_invocations": 0,
        "cuda_initialized": False, "elapsed_seconds": time.monotonic() - started,
        "raw_evidence_sha256": digest(raw_rows), "prepare_evidence_sha256": digest(prepare_rows)})
    print("CPU audit passed; no model was loaded and no CUDA context was created.", flush=True)


def accounted_call(function, name):
    from video_compute_value.common import exclusive_lock
    from video_compute_value.resources import Ledger
    config = {"cumulative_execution_limit_hours": 4, "additional_disk_limit_gib": 20, "video_acquisition_limit_gib": 1}
    if not (RUN / "resource_ledger.json").exists():
        raise Blocked("BLOCKED_RESOURCE", "Inherit the latest cumulative ledger first")
    if any(s.get("finished_at") is None for s in read_json(RUN / "resource_ledger.json")["sessions"]):
        raise Blocked("BLOCKED_RESOURCE", "Active accounting must close or be reconciled before standalone execution")
    with exclusive_lock(RUN / ".lock"):
        ledger = Ledger(RUN / "resource_ledger.json", ROOT, config)
        with ledger.track(name):
            return function()


if __name__ == "__main__":
    accounted_call(main, "standalone-cpu-equivalence-audit")
