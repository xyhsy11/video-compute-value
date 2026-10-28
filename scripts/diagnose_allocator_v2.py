"""Named paired native-allocator versus in-request cache-trim diagnostic."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import yaml

from video_compute_value.common import append_json, Blocked, command_output, digest, exclusive_lock, file_hash, read_json, utc_now, write_json
from video_compute_value.e000 import set_local_paths
from video_compute_value.generation import resolve_greedy_policy
from video_compute_value.measurement import MemorySampler, timed_request
from video_compute_value.model import QwenAdapter, tensor_hashes
from video_compute_value.protocol import question_input, video_input
from video_compute_value.resources import Ledger
from diagnose_e000_timing import telemetry

RUN = "results/e000_20261028_diagnostic_v2"
PARENT = "results/e000_20261027_collection_v2"
PRIOR = "results/e000_20261027_diagnostic_v1"
CONFIG = {"cumulative_execution_limit_hours": 4, "additional_disk_limit_gib": 20, "video_acquisition_limit_gib": 1}


def sources(root):
    files = list((root / "src/video_compute_value").glob("*.py"))
    files += [root / "scripts" / name for name in ("diagnose_allocator_v2.py", "diagnose_e000_timing.py", "diagnostic_generation.py")]
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(files)}


class Adapter(QwenAdapter):
    def __init__(self, frozen, trim):
        super().__init__(frozen)
        self.trim = trim

    def prepare(self, *args, **kwargs):
        started = time.perf_counter_ns()
        inputs, details = super().prepare(*args, **kwargs)
        details["cpu_prepare_host_seconds"] = (time.perf_counter_ns() - started) / 1e9
        return inputs, details

    def generate(self, inputs):
        started = time.perf_counter_ns()
        if self.trim:
            self.torch.cuda.empty_cache()
        trim_seconds = (time.perf_counter_ns() - started) / 1e9
        output = super().generate(inputs)
        output["allocator_trim_host_seconds"] = trim_seconds if self.trim else 0
        return output


def initialize(root, run):
    run.mkdir(parents=True, exist_ok=True)
    if (run / "plan.json").exists():
        return
    prior = read_json(root / PRIOR / "resource_ledger.json")
    assert all(s.get("finished_at") for s in prior["sessions"])
    prior["active_seconds"] += 60
    prior["external_events"].append({"name": "Conservative production patch, CPU validation and next-plan preparation allowance", "active_seconds": 60, "recorded_at": utc_now()})
    write_json(run / "resource_ledger.json", prior)
    cohort = read_json(root / PARENT / "cohort_manifest.json")
    first = {(v["source_id"], v["questions"][0]["question_id"]) for v in cohort["videos"]}
    blocks = []
    order = read_json(root / PARENT / "execution_order.json")
    for original in order:
        if original["budget"] not in (8, 64):
            continue
        arms = ["native", "request_trim"]
        if original["repetition"] % 2 == 0 or original["budget"] == 8:
            arms.reverse()
        for arm in arms:
            blocks.append({"arm": arm, "budget": original["budget"], "repetition": original["repetition"],
                "questions": [p for p in original["questions"] if tuple(p) in first], "profile": original["repetition"] == 1})
    frozen = yaml.safe_load((root / PARENT / "resolved_config.yaml").read_text(encoding="utf-8"))
    from transformers import GenerationConfig
    policy = resolve_greedy_policy(GenerationConfig(**frozen["generation"]),
                GenerationConfig.from_pretrained(frozen["model_path"], local_files_only=True))
    rows = [read_json(p) for p in (root / PARENT / "records").glob("*.json")]
    lookup = {(r["source_id"], r["question_id"], r["budget"], r["repetition"]): r["latency_seconds"] for r in rows}
    request_estimate = sum(lookup[(p[0], p[1], b["budget"], b["repetition"])] for b in blocks for p in b["questions"])
    warmups = [json.loads(line) for line in (root / PARENT / "diagnostics/warmups.jsonl").read_text(encoding="utf-8").splitlines()]
    auxiliary_estimate = sum((2 + int(b["profile"])) * statistics.median(r["latency_seconds"] for r in warmups if r["budget"] == b["budget"]) for b in blocks)
    plan = {"run_id": run.name, "frozen_at": utc_now(), "parent_run": PARENT, "prior_diagnostic_run": PRIOR,
        "blocks": blocks, "sources": sources(root), "source_git_revision": command_output(["git", "rev-parse", "HEAD"])["stdout"],
        "generation_policy": policy, "config": CONFIG, "timed_requests": 96, "warmups": 24, "profiles": 4,
        "selection": "First official question per source, eight sources, budgets 8 and 64, three repetitions; no correctness selection",
        "intervention": "Default native allocator in both arms. request_trim calls torch.cuda.empty_cache after input transfer and before model.generate, inside original full-request timer. No offload, no extra synchronization fences.",
        "preserved_prior_regression": "Prior max_split_size_mb:128 arm stopped after excessive reservations and warmup costs; no primary requests completed.",
        "boundary": "Original synchronized resident-model local-file full request; trim time is included; nested CPU preparation clock has no CUDA fence",
        "forecast_seconds_with_safety": 1.5 * (request_estimate + auxiliary_estimate + 100 * len(blocks) + .5 * 124),
        "forecast_basis": "Matched parent request times plus same-budget warmups, 100 seconds cold-start/serialization allowance per worker, 0.5 seconds per call overhead; 1.5 safety factor",
        "worker_timeout_seconds": 600, "final_validation_reserve_seconds": 240, "no_budget_reset": True}
    plan["plan_hash"] = digest(plan)
    write_json(run / "plan.json", plan)


def worker(root, run, index):
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    os.environ.pop("PYTORCH_ALLOC_CONF", None)
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    import torch
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    plan = read_json(run / "plan.json")
    if plan["sources"] != sources(root):
        raise Blocked("INPUT_INTEGRITY_FAILED", "Frozen diagnostic implementation changed")
    block = plan["blocks"][index]
    parent = root / PARENT
    frozen = yaml.safe_load((parent / "resolved_config.yaml").read_text(encoding="utf-8"))
    frozen["generation_policy"] = plan["generation_policy"]
    cohort = read_json(parent / "cohort_manifest.json")
    lookup = {(v["source_id"], q["question_id"]): (video_input(v), question_input(q)) for v in cohort["videos"] for q in v["questions"]}
    references = {(r["source_id"], r["question_id"], r["budget"]): r["input_tensor_hashes"]
                  for r in (read_json(p) for p in (parent / "records").glob("*.json"))}
    with exclusive_lock(root / ".cache/cuda_worker.lock"):
        adapter = Adapter(frozen, block["arm"] == "request_trim")
        write_json(run / "private" / f"worker_{index}_{time.time_ns()}.json", {"block": block,
            "generation_policy": adapter.generation_policy, "load_seconds": adapter.load_seconds,
            "allocator_backend": torch.cuda.memory.get_allocator_backend(), "allocator_env": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")})
        def execute(pair, kind, sequence):
            identity = {"plan_hash": plan["plan_hash"], "block": index, "kind": kind, "sequence": sequence,
                "source_id": pair[0], "question_id": pair[1]}
            key = digest(identity)
            path = run / "private/records" / (key + ".json")
            if kind == "request" and path.exists():
                return
            call = key + "_" + str(time.time_ns())
            before_count = adapter.model_call_count
            append_json(run / "private/call_events.jsonl", {**identity, "call_id": call, "event": "started", "at": utc_now()})
            try:
                before = telemetry(torch)
                video, question = lookup[tuple(pair)]
                row, cpu = timed_request(adapter, question, video, block["budget"], profile=kind == "profile", sampler=MemorySampler(.05))
                after = telemetry(torch)
                hashes = tensor_hashes(cpu)
                if hashes != references[(pair[0], pair[1], block["budget"])]:
                    raise Blocked("INPUT_INTEGRITY_FAILED", "Parent input hash mismatch")
                row.update(identity, call_id=call, arm=block["arm"], budget=block["budget"], repetition=block["repetition"],
                    input_tensor_hashes=hashes, telemetry_before=before, telemetry_after=after)
                del cpu
                gc.collect()
                row["live_allocated_after_bytes"] = torch.cuda.memory_allocated()
                output = path if kind == "request" else run / "private/auxiliary" / (call + ".json")
                write_json(output, row)
                append_json(run / "private/call_events.jsonl", {"call_id": call, "event": "completed", "kind": kind,
                    "model_invoked": adapter.model_call_count > before_count, "at": utc_now()})
            except BaseException as exc:
                append_json(run / "private/call_events.jsonl", {"call_id": call, "event": "failed", "kind": kind,
                    "model_invoked": adapter.model_call_count > before_count, "error": type(exc).__name__, "at": utc_now()})
                raise
        for i, pair in enumerate(frozen["two_warmups"]):
            execute(pair, "warmup", i)
        for i, pair in enumerate(block["questions"]):
            execute(pair, "request", i)
        if block["profile"]:
            execute(frozen["profile_question"], "profile", 0)
        write_json(run / f"block_{index}.json", {"complete": True, "block": block})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker"))
    parser.add_argument("--block", type=int)
    args = parser.parse_args()
    root, run = Path.cwd(), Path.cwd() / RUN
    set_local_paths(root)
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "4"
    if args.command == "worker":
        try:
            worker(root, run, args.block)
            return 0
        except BaseException:
            append_json(run / "private/failures.jsonl", {"block": args.block, "at": utc_now(), "traceback": traceback.format_exc()})
            traceback.print_exc()
            return 1
    initialize(root, run)
    with exclusive_lock(run / ".lock"):
        ledger = Ledger(run / "resource_ledger.json", root, CONFIG)
        with ledger.track("matched-native-versus-request-trim"):
            plan = read_json(run / "plan.json")
            for index, block in enumerate(plan["blocks"]):
                if (run / f"block_{index}.json").exists():
                    continue
                remaining_calls = sum(len(b["questions"]) + 2 + int(b["profile"]) for b in plan["blocks"][index:])
                ledger.check(projected_seconds=plan["forecast_seconds_with_safety"] * remaining_calls / 124 + 240)
                log = run / "private" / f"worker_{index}_{time.time_ns()}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("w", encoding="utf-8") as output:
                    process = subprocess.Popen([sys.executable, "-u", __file__, "worker", "--block", str(index)], cwd=root, stdout=output, stderr=output)
                    try:
                        code = process.wait(timeout=min(600, max(1, 14400 - ledger.consumed_seconds() - 240)))
                    except BaseException:
                        process.kill()
                        process.wait()
                        raise
                if code:
                    raise Blocked("PARTIAL", f"Worker failed; preserved {log}")
                count = len(list((run / "private/records").glob("*.json")))
                print(f"{count}/96 matched requests; completed block {index + 1}/12", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
