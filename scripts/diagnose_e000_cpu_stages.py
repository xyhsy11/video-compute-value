"""Bounded CPU-only stage diagnosis after the completed, timing-unstable collection."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import yaml

from video_compute_value.common import Blocked, atomic_text, digest, file_hash, read_json, utc_now, write_json
from video_compute_value.e000 import require_cumulative_ledger, set_local_paths
from video_compute_value.resources import Ledger

ROOT = Path.cwd()
PARENT = ROOT / "results/e000_20261028_greedy_collection_v3"
RUN = ROOT / "results/e000_20261028_cpu_stage_v4"


def diagnose(ledger):
    import torch
    from transformers import AutoConfig, AutoProcessor
    import video_compute_value.model as model_module
    import video_compute_value.video as video_module
    from video_compute_value.model import QwenAdapter, adapter_source_hashes, tensor_hashes
    from video_compute_value.protocol import question_input, video_input
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    frozen = yaml.safe_load((PARENT / "resolved_config.yaml").read_text(encoding="utf-8"))
    native = AutoProcessor.from_pretrained(frozen["model_path"], local_files_only=True, trust_remote_code=False, use_fast=True)
    assert adapter_source_hashes() == frozen["processor_source_hashes"]
    adapter = QwenAdapter.__new__(QwenAdapter)
    adapter.torch, adapter.frozen = torch, frozen
    adapter.model = SimpleNamespace(config=AutoConfig.from_pretrained(frozen["model_path"], local_files_only=True, trust_remote_code=False))
    spans = {}

    class ProfiledProcessor:
        def __getattr__(self, name):
            return getattr(native, name)

        def apply_chat_template(self, *args, **kwargs):
            start = time.perf_counter()
            value = native.apply_chat_template(*args, **kwargs)
            spans["chat_template"] = time.perf_counter() - start
            return value

        def __call__(self, *args, **kwargs):
            start = time.perf_counter()
            value = native(*args, **kwargs)
            spans["native_processor"] = time.perf_counter() - start
            return value

    adapter.processor = ProfiledProcessor()
    decode, letterbox = model_module.decode_selected, video_module.letterbox

    def timed_decode(*args, **kwargs):
        start = time.perf_counter()
        value = decode(*args, **kwargs)
        spans["decode_selection_rgb_resize_stack"] = time.perf_counter() - start
        return value

    def timed_letterbox(*args, **kwargs):
        start = time.perf_counter()
        value = letterbox(*args, **kwargs)
        spans["letterbox_subset"] += time.perf_counter() - start
        return value

    model_module.decode_selected, video_module.letterbox = timed_decode, timed_letterbox
    cohort = read_json(PARENT / "cohort_manifest.json")
    lookup = {(v["source_id"], v["questions"][0]["question_id"]): (v, v["questions"][0]) for v in cohort["videos"]}
    reference = {(r["source_id"], r["question_id"], r["budget"]): r["input_tensor_hashes"] for r in
                 (read_json(p) for p in (PARENT / "records").glob("*.json"))}
    plan = []
    for block in read_json(PARENT / "execution_order.json"):
        if block["budget"] in (8, 64):
            plan.extend({"pair": pair, "budget": block["budget"], "repetition": block["repetition"]}
                        for pair in block["questions"] if tuple(pair) in lookup)
    assert len(plan) == 48
    import psutil
    process = psutil.Process()
    write_json(RUN / "plan.json", {"at": utc_now(), "parent_run": PARENT.name, "planned_cpu_preparations": 48,
        "selection": "First official question per original source; 8/64 frames; three repetitions in parent order; no correctness selection",
        "scope": "One persistent CPU process, no resident model or GPU. Stage clocks are intrusive diagnostic spans, not primary E2E measurements.",
        "cpu_loop_deadline_seconds": 300, "model_calls": 0, "cpu_affinity": process.cpu_affinity(),
        "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in (ROOT / "src/video_compute_value").glob("*.py")},
        "script_sha256": file_hash(Path(__file__)), "no_frame_or_feature_cache": True})
    rows, started = [], time.monotonic()
    try:
        for item in plan:
            if time.monotonic() - started > 300:
                raise Blocked("BLOCKED_RESOURCE", "CPU diagnostic loop reached its 300-second bound")
            ledger.check()
            video, question = lookup[tuple(item["pair"])]
            spans.clear()
            spans["letterbox_subset"] = 0
            before = sum(process.cpu_times()[:2])
            tick = time.perf_counter()
            inputs, _ = adapter.prepare(question_input(question), video_input(video), item["budget"])
            elapsed = time.perf_counter() - tick
            cpu_seconds = sum(process.cpu_times()[:2]) - before
            hashes = tensor_hashes(inputs)
            assert hashes == reference[video["source_id"], question["question_id"], item["budget"]]
            residual = elapsed - sum(spans[key] for key in ("decode_selection_rgb_resize_stack", "chat_template", "native_processor"))
            rows.append({**item, "total_cpu_prepare_seconds": elapsed, **spans,
                "copies_metadata_validation_other_seconds": residual, "process_cpu_seconds": cpu_seconds,
                "input_tensor_hashes": hashes})
            del inputs
            write_json(RUN / "private/cpu_stage_records.json", rows)
            if len(rows) % 8 == 0:
                print(f"CPU stage preparation {len(rows)}/48; hashes match", flush=True)
    finally:
        model_module.decode_selected, video_module.letterbox = decode, letterbox
    assert not torch.cuda.is_initialized()
    conditions, lines = [], []
    for budget in (8, 64):
        group = [r for r in rows if r["budget"] == budget]
        cells = defaultdict(list)
        for row in group:
            cells[tuple(row["pair"])].append(row)
        ranges = [(max(r["total_cpu_prepare_seconds"] for r in cell) - min(r["total_cpu_prepare_seconds"] for r in cell)) /
                  statistics.median(r["total_cpu_prepare_seconds"] for r in cell) for cell in cells.values()]
        keys = ("total_cpu_prepare_seconds", "decode_selection_rgb_resize_stack", "letterbox_subset", "native_processor", "chat_template", "copies_metadata_validation_other_seconds")
        values = {key: statistics.median(statistics.median(r[key] for r in cell) for cell in cells.values()) for key in keys}
        conditions.append({"budget": budget, "successful_cpu_preparations": len(group), **values,
            "median_repetition_relative_range": statistics.median(ranges)})
        lines.append(f"| {budget} | {values['total_cpu_prepare_seconds']:.3f} | {values['decode_selection_rgb_resize_stack']:.3f} | {values['letterbox_subset']:.3f} | {values['native_processor']:.4f} | {values['copies_metadata_validation_other_seconds']:.4f} | {100*statistics.median(ranges):.2f}% |")
    write_json(RUN / "verification.json", {"passed": True, "cpu_preparations": 48,
        "parent_input_hash_matches": 48, "model_calls": 0, "cuda_initialized": False,
        "private_record_sha256": digest(rows), "conditions": conditions, "elapsed_loop_seconds": time.monotonic() - started})
    atomic_text(RUN / "report.md", "# CPU preparation stage diagnostic v4\n\n"
        "This is a separate CPU-only diagnostic after the completed v3 collection. Its timings do not replace any primary measurement or change the MEASUREMENT_UNSTABLE verdict.\n\n"
        "One persistent process prepared the first official question of each source at 8/64 frames, three times, in the parent block order. No model was instantiated, no CUDA context was initialized, and no frames/features were reused. All 48 input tensor hashes match the saved collection.\n\n"
        "| Frames | CPU preparation s | Decode/selection/RGB/resize/stack s | Letterbox subset s | Native processor s | Copies/metadata/checks/other s | Median CPU repetition range |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(lines) +
        "\n\nEach stage uses per-source repetition medians, then the median over eight sources. Stage medians need not add to the total median. Letterboxing is nested within the decode span. The remaining decode span includes I/O, FFmpeg decoding, source-PTS checks, selection, RGB conversion and stacking; it is not isolated decoder-kernel time.\n\n"
        "These measurements localize CPU costs but do not establish a unique OS scheduling, clock, I/O, or thread-pool cause. A persistent CPU-only process differs from fresh GPU workers; it cannot validate end-to-end timing stability. A changed execution policy requires separately named real preflight and collection, not replacement of v3 repetitions.\n")


def main():
    if RUN.exists():
        raise Blocked("INPUT_INTEGRITY_FAILED", "Existing named CPU diagnostic is immutable")
    set_local_paths(ROOT)
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "4"
    prior = read_json(PARENT / "resource_ledger.json")
    assert all(s.get("finished_at") for s in prior["sessions"])
    start = datetime.fromisoformat(prior["sessions"][-1]["finished_at"])
    now = datetime.now().astimezone()
    gap = (now - start).total_seconds()
    prior["active_seconds"] += gap
    prior["sessions"].append({"name": "Postcollection timing review and CPU-stage diagnostic preparation",
        "started_at": start.isoformat(), "finished_at": now.isoformat(), "active_seconds": gap, "outcome": "completed"})
    (RUN / "private").mkdir(parents=True)
    write_json(RUN / "resource_ledger.json", prior)
    write_json(RUN / "lineage.json", {"parent_resource_ledger": str((PARENT / "resource_ledger.json").relative_to(ROOT)),
        "inherited_seconds_before_new_preparation": read_json(PARENT / "resource_ledger.json")["active_seconds"],
        "cumulative_limit_seconds": 20400, "model_calls_planned": 0, "projection_seconds": 600})
    require_cumulative_ledger(RUN, ROOT)
    config = {"cumulative_execution_limit_hours": 20400 / 3600, "additional_disk_limit_gib": 20, "video_acquisition_limit_gib": 1}
    ledger = Ledger(RUN / "resource_ledger.json", ROOT, config)
    with ledger.track("bounded-postcollection-cpu-stage-diagnostic"):
        ledger.check(projected_seconds=600)
        diagnose(ledger)
    print({"cumulative_seconds": ledger.value["active_seconds"], "remaining_seconds": 20400 - ledger.value["active_seconds"]})


if __name__ == "__main__":
    main()
