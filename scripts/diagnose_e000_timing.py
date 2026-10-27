"""Matched greedy-generation and allocator diagnostics, charged to the inherited ledger."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import yaml

import diagnose_e000 as forensic
from diagnostic_generation import fixed_greedy_kwargs
from video_compute_value.common import (Blocked, append_json, digest, exclusive_lock, file_hash,
    read_json, utc_now, write_json)
from video_compute_value.e000 import set_local_paths
from video_compute_value.measurement import MemorySampler, timed_request
from video_compute_value.model import QwenAdapter, tensor_hashes
from video_compute_value.protocol import question_input, video_input
from video_compute_value.resources import Ledger, environment

ALLOCATOR = "garbage_collection_threshold:0.75,max_split_size_mb:128"


class StrictGreedyAdapter(QwenAdapter):
    def __init__(self, frozen):
        super().__init__(frozen)
        self.call_overrides = fixed_greedy_kwargs(self)
        effective, _ = self.model._prepare_generation_config(self.generation, **self.call_overrides)
        if effective.do_sample or effective.get_generation_mode().value != "greedy_search":
            raise Blocked("INPUT_INTEGRITY_FAILED", "Effective generation is not greedy")
        self.effective_generation = effective.to_dict()
        native_generate = self.model.generate
        overrides = self.call_overrides
        def generate(*args, **kwargs):
            kwargs.update(overrides)
            return native_generate(*args, **kwargs)
        self.model.generate = generate

    def prepare(self, *args, **kwargs):
        started = time.perf_counter_ns()
        inputs, details = super().prepare(*args, **kwargs)
        details["cpu_prepare_host_seconds"] = (time.perf_counter_ns() - started) / 1e9
        return inputs, details


def source_manifest(root):
    names = ["scripts/diagnose_e000_timing.py", "scripts/diagnostic_generation.py", "scripts/diagnose_e000.py",
             "src/video_compute_value/model.py", "src/video_compute_value/measurement.py",
             "src/video_compute_value/video.py"]
    return {name: file_hash(root / name) for name in names}


def telemetry(torch):
    import psutil
    result = {"process_cpu_seconds": sum(psutil.Process().cpu_times()[:2]),
              "system_cpu_percent": psutil.cpu_percent(), "system_ram_available_bytes": psutil.virtual_memory().available}
    stats = torch.cuda.memory_stats()
    for key in ("allocated_bytes.all.current", "reserved_bytes.all.current", "inactive_split_bytes.all.current",
                "num_alloc_retries", "num_ooms", "max_split_size"):
        result[key] = stats.get(key)
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        functions = {
            "gpu_temperature_c": lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
            "gpu_sm_clock_mhz": lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
            "gpu_memory_clock_mhz": lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM),
            "gpu_power_mw": lambda: pynvml.nvmlDeviceGetPowerUsage(handle),
            "gpu_throttle_reasons": lambda: pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle),
            "device_used_bytes": lambda: pynvml.nvmlDeviceGetMemoryInfo(handle).used,
        }
        for key, function in functions.items():
            try:
                result[key] = function()
            except Exception:
                result[key] = None
        pynvml.nvmlShutdown()
    except Exception:
        result["nvml_telemetry_unavailable"] = True
    return result


def create_timing_plan(root, run, config):
    parent = root / config["parent_run"]
    cohort = read_json(parent / "cohort_manifest.json")
    first_questions = {(v["source_id"], v["questions"][0]["question_id"]) for v in cohort["videos"]}
    parent_order = read_json(parent / "execution_order.json")
    blocks = []
    for i, original in enumerate(parent_order):
        if original["budget"] not in config["timing_budgets"]:
            continue
        questions = [pair for pair in original["questions"]
                     if original["budget"] == 64 or tuple(pair) in first_questions]
        arms = ["native_allocator", "bounded_allocator"]
        if (i + original["repetition"]) % 2:
            arms.reverse()
        for arm in arms:
            blocks.append({"arm": arm, "budget": original["budget"], "repetition": original["repetition"],
                           "questions": questions, "profile": original["repetition"] == 1})
    value = {"phase": "matched_timing_v1", "frozen_at": utc_now(), "parent_run": config["parent_run"],
        "sources": source_manifest(root), "generation": read_json(run / "generation_defaults_audit.json")["fixed_effective"],
        "allocator": {"native_allocator": None, "bounded_allocator": ALLOCATOR}, "blocks": blocks,
        "timed_requests": sum(len(b["questions"]) for b in blocks), "warmups": 2 * len(blocks),
        "component_profiles": sum(b["profile"] for b in blocks),
        "selection": "All 24 parent questions at Uniform-64; first official question per video at Uniform-8; no correctness-based selection",
        "boundary": "Original synchronized resident-model local-file request; allocator maintenance occurs within CUDA allocation calls and is included",
        "arms_crossover": "Each budget/repetition is paired; arm order alternates by original block index and repetition",
        "no_decoded_frame_or_feature_reuse": True}
    parent_rows = [read_json(p) for p in (parent / "records").glob("*.json")]
    lookup = {(r["source_id"], r["question_id"], r["budget"], r["repetition"]): r["latency_seconds"] for r in parent_rows}
    probes = read_json(parent / "probes.json")["probes"]
    worst_startup = max(p["worker_wall_seconds"] - p["timed_calls_seconds"] for p in probes)
    warmups = [r for r in read_jsonl_compatible(parent / "diagnostics" / "warmups.jsonl")]
    import statistics
    request_estimate = sum(lookup[(p[0], p[1], b["budget"], b["repetition"])] for b in blocks for p in b["questions"])
    auxiliary_estimate = sum(2 * statistics.median(r["latency_seconds"] for r in warmups if r["budget"] == b["budget"]) +
        (next(p["component_profile"]["latency_seconds"] for p in probes if p["budget"] == b["budget"]) if b["profile"] else 0)
        for b in blocks)
    value["projected_seconds_with_safety"] = 1.5 * (request_estimate + auxiliary_estimate + worst_startup * len(blocks) +
        0.5 * (value["timed_requests"] + value["warmups"] + value["component_profiles"]))
    value["projection_basis"] = "Sum exact matched parent request costs, per-budget warmups/profiles, worst whole-worker startup/overhead, 0.5s extra telemetry/hash overhead per call; multiply by 1.5"
    value["phase_hash"] = digest(value)
    return value


def read_jsonl_compatible(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def greedy_worker(root, phase, config):
    plan = read_json(phase / "forensic_plan.json")
    forensic.QwenAdapter = StrictGreedyAdapter
    forensic.forensic_worker(root, phase, config, "fixed_greedy")
    write_json(phase / "effective_generation.json", plan["effective_generation"])


def timing_worker(root, run, config, block_index):
    plan = read_json(run / "timing_plan.json")
    if plan["sources"] != source_manifest(root):
        raise Blocked("INPUT_INTEGRITY_FAILED", "Timing diagnostic implementation changed")
    block = plan["blocks"][block_index]
    if block["arm"] == "bounded_allocator":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ALLOCATOR
    else:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    import torch
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    parent = root / config["parent_run"]
    frozen = yaml.safe_load((parent / "resolved_config.yaml").read_text(encoding="utf-8"))
    cohort = read_json(parent / "cohort_manifest.json")
    lookup = {(v["source_id"], q["question_id"]): (video_input(v), question_input(q))
              for v in cohort["videos"] for q in v["questions"]}
    parent_rows = [read_json(p) for p in (parent / "records").glob("*.json")]
    references = {(r["source_id"], r["question_id"], r["budget"]): r["input_tensor_hashes"] for r in parent_rows}
    with exclusive_lock(root / ".cache" / "cuda_worker.lock"):
        adapter = StrictGreedyAdapter(frozen)
        stats = torch.cuda.memory_stats()
        if block["arm"] == "bounded_allocator" and stats.get("max_split_size") != 128 * 1024 ** 2:
            raise Blocked("INPUT_INTEGRITY_FAILED", "Requested allocator setting was not applied")
        worker_id = f"timing_{block_index}_{time.time_ns()}"
        write_json(run / "private" / f"{worker_id}.json", {"block": block, "load_seconds": adapter.load_seconds,
            "effective_generation": adapter.effective_generation, "generation_call_overrides": adapter.call_overrides,
            "allocator_env": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "allocator_backend": torch.cuda.memory.get_allocator_backend(), "baseline": adapter.baseline})
        def execute(pair, kind, sequence):
            video, question = lookup[tuple(pair)]
            identity = {"phase_hash": plan["phase_hash"], "block": block_index, "kind": kind,
                        "source_id": pair[0], "question_id": pair[1], "sequence": sequence}
            stable_id = digest(identity)
            path = run / "private" / "timing_records" / f"{stable_id}.json"
            if path.exists() and kind == "request":
                return
            call_id = stable_id + "_" + str(time.time_ns())
            append_json(run / "private" / "timing_events.jsonl", {"call_id": call_id, "event": "started", **identity, "at": utc_now()})
            before = telemetry(torch)
            try:
                row, cpu = timed_request(adapter, question, video, block["budget"], profile=kind == "profile",
                    sampler=MemorySampler(0.05))
                after = telemetry(torch)
                hashes = tensor_hashes(cpu)
                if hashes != references[(pair[0], pair[1], block["budget"])]:
                    raise Blocked("INPUT_INTEGRITY_FAILED", "Prepared tensors differ from immutable parent")
                row.update(identity, call_id=call_id, arm=block["arm"], budget=block["budget"], repetition=block["repetition"],
                           input_tensor_hashes=hashes, telemetry_before=before, telemetry_after=after,
                           process_cpu_seconds=after["process_cpu_seconds"] - before["process_cpu_seconds"])
                del cpu
                gc.collect()
                row["live_allocated_after_bytes"] = torch.cuda.memory_allocated()
                output_path = path if kind == "request" else run / "private" / "timing_auxiliary" / (call_id + ".json")
                write_json(output_path, row)
                append_json(run / "private" / "timing_events.jsonl", {"call_id": call_id, "event": "completed",
                    "kind": kind, "model_invoked": True, "at": utc_now()})
            except BaseException as exc:
                append_json(run / "private" / "timing_events.jsonl", {"call_id": call_id, "event": "failed",
                    "kind": kind, "error": type(exc).__name__, "message": str(exc), "at": utc_now()})
                raise
        for i, pair in enumerate(frozen["two_warmups"]):
            execute(pair, "warmup", i)
        for i, pair in enumerate(block["questions"]):
            execute(pair, "request", i)
        if block["profile"]:
            execute(frozen["profile_question"], "profile", 0)
        write_json(run / f"timing_block_{block_index}.json", {"complete": True, "block": block,
                   "model_load_seconds": adapter.load_seconds, "timed_requests": len(block["questions"])})


def timing_summary(run):
    import statistics
    rows = [read_json(p) for p in (run / "private" / "timing_records").glob("*.json")]
    result = []
    for arm in ("native_allocator", "bounded_allocator"):
        for budget in (8, 64):
            group = [r for r in rows if r["arm"] == arm and r["budget"] == budget]
            cells = defaultdict(list)
            for row in group:
                cells[(row["source_id"], row["question_id"])].append(row)
            full = [g for g in cells.values() if len(g) == 3]
            medians = [statistics.median(r["latency_seconds"] for r in g) for g in full]
            ranges = [(max(r["latency_seconds"] for r in g) - min(r["latency_seconds"] for r in g)) /
                      statistics.median(r["latency_seconds"] for r in g) for g in full]
            result.append({"arm": arm, "budget": budget, "completed_requests": len(group), "complete_cells": len(full),
                "planned_requests": 72 if budget == 64 else 24,
                "median_request_seconds": statistics.median(medians) if medians else None,
                "median_relative_timing_range": statistics.median(ranges) if ranges else None,
                "answer_disagreement_cells": sum(len({r["parsed_answer"] for r in g}) > 1 for g in full),
                "peak_allocated_bytes": max((r["peak_allocated_bytes"] for r in group), default=None),
                "peak_reserved_bytes": max((r["peak_reserved_bytes"] for r in group), default=None),
                "sampled_device_peak_bytes": max((r["sampled_device_gpu_peak_bytes"] for r in group if r["sampled_device_gpu_peak_bytes"] is not None), default=None),
                "max_request_seconds": max((r["latency_seconds"] for r in group), default=None),
                "median_cpu_prepare_seconds": statistics.median(r["cpu_prepare_host_seconds"] for r in group) if group else None,
                "median_request_minus_cpu_prepare_seconds": statistics.median(r["latency_seconds"] - r["cpu_prepare_host_seconds"] for r in group) if group else None})
    value = {"conditions": result, "timed_requests": len(rows), "planned_timed_requests": 192,
             "selection": "Diagnostic matched subset; all 24 parent questions at Uniform-64; first official question per video at Uniform-8",
             "primary_accuracy_curve": False}
    write_json(run / "timing_summary.json", value)
    return value


def launch(root, run, config, command, ledger, block=None):
    args = [sys.executable, "-u", str(Path(__file__).resolve()), command, "--run-dir", str(run)]
    if block is not None:
        args += ["--block", str(block)]
    log = run / "private" / f"{command}_{block}_{time.time_ns()}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    remaining = 14400 - ledger.consumed_seconds() - config["reserve_final_validation_seconds"]
    if remaining < 120:
        raise Blocked("BLOCKED_RESOURCE", "Final validation reserve reached")
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(args, cwd=root, stdout=output, stderr=output)
        try:
            code = process.wait(timeout=min(config["timing_worker_timeout_seconds"], remaining))
        except BaseException:
            process.kill()
            process.wait()
            raise
    if code:
        raise Blocked("PARTIAL", f"Worker failed; see {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("greedy", "timing", "summary", "greedy_worker", "timing_worker"))
    parser.add_argument("--run-dir", type=Path, default=Path("results/e000_20261027_diagnostic_v1"))
    parser.add_argument("--block", type=int)
    args = parser.parse_args()
    root, run = Path.cwd(), args.run_dir.resolve()
    config = yaml.safe_load((root / "configs/e000_diagnostic_v1.yaml").read_text(encoding="utf-8"))
    set_local_paths(root)
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "4"
    if args.command.endswith("worker"):
        try:
            if args.command == "greedy_worker":
                greedy_worker(root, run / "greedy_validation", config)
            else:
                timing_worker(root, run, config, args.block)
            return 0
        except BaseException as exc:
            append_json(run / "private" / "failures.jsonl", {"phase": args.command, "block": args.block,
                "at": utc_now(), "error": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
            traceback.print_exc()
            return 1
    with exclusive_lock(run / ".lock"):
        ledger = Ledger(run / "resource_ledger.json", root, config)
        with ledger.track("diagnostic-" + args.command):
            if args.command == "greedy":
                phase = run / "greedy_validation"
                phase.mkdir(exist_ok=True)
                path = phase / "forensic_plan.json"
                if not path.exists():
                    write_json(path, {"cases": forensic.cases_for_parent(root / config["parent_run"]),
                        "script_sha256": file_hash(__file__), "sources": source_manifest(root), "frozen_at": utc_now(),
                        "effective_generation": read_json(run / "generation_defaults_audit.json")["fixed_effective"],
                        "generation_call_overrides": read_json(run / "generation_defaults_audit.json")["fixed_call_kwargs"],
                        "maximum_model_calls": 99, "intrusive": True, "prepared_tensor_reuse": True})
                if not (phase / "forensic_worker_fixed_greedy.json").exists():
                    launch(root, run, config, "greedy_worker", ledger)
                print(json.dumps(forensic.forensic_summary(phase), indent=2))
            elif args.command == "timing":
                path = run / "timing_plan.json"
                if not path.exists():
                    write_json(path, create_timing_plan(root, run, config))
                plan = read_json(path)
                write_json(run / "diagnostic_environment.json", environment(root))
                for index, block in enumerate(plan["blocks"]):
                    if (run / f"timing_block_{index}.json").exists():
                        continue
                    # Parent-observed whole-plan forecast, scaled to remaining matched work; bounded again per worker.
                    remaining_calls = sum(len(b["questions"]) + 2 + int(b["profile"]) for b in plan["blocks"][index:])
                    total_calls = plan["timed_requests"] + plan["warmups"] + plan["component_profiles"]
                    ledger.check(projected_seconds=plan["projected_seconds_with_safety"] * remaining_calls / total_calls +
                                 config["reserve_final_validation_seconds"])
                    launch(root, run, config, "timing_worker", ledger, index)
                    data = timing_summary(run)
                    print(f"timing: {data['timed_requests']}/192 requests saved; block {index + 1}/{len(plan['blocks'])}", flush=True)
            else:
                print(json.dumps(timing_summary(run), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
