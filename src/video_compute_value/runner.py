"""Freeze, bounded probes, sequential fresh workers, and provenance-checked resumption."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import yaml

from .common import (Blocked, append_json, atomic_text, command_output, digest, exclusive_lock, file_hash,
                     read_json, read_jsonl, utc_now, write_json)
from .dataset import validate_cohort
from .measurement import MemorySampler, timed_request
from .protocol import MEASUREMENT_VERSION, PARSER_VERSION, PROMPT, make_order, question_input, trial_identity, video_input
from .resources import environment, environment_fingerprint


def source_hash(root):
    files = sorted(Path(root, "src", "video_compute_value").glob("*.py"))
    return digest({p.name: file_hash(p) for p in files})


def resolve_model(config, root, ledger):
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoProcessor, GenerationConfig
    from .model import adapter_source_hashes
    if not torch.cuda.is_available():
        raise Blocked("BLOCKED_RESOURCE", "CUDA unavailable; CPU inference is outside this protocol")
    expected = {"torch": "2.8.0", "torchvision": "0.23.0", "transformers": "4.57.1", "accelerate": "1.10.1"}
    for name, version in expected.items():
        if importlib.metadata.version(name).split("+")[0] != version:
            raise Blocked("BLOCKED_ENVIRONMENT", f"Install the pinned {name}=={version} stack")
    try:
        info = HfApi().model_info(config["model_id"], revision=config.get("model_revision"), files_metadata=True)
        revision = info.sha
        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise Blocked("INPUT_INTEGRITY_FAILED", "Model revision is unresolved")
        files = [f for f in info.siblings if f.rfilename.endswith((".json", ".txt", ".safetensors"))]
        local_cache = Path(root, ".cache", "huggingface", "hub")
        reusable = Path.home() / ".cache" / "huggingface" / "hub"
        path = None
        for cache in (local_cache, reusable):
            candidate = cache / ("models--" + config["model_id"].replace("/", "--")) / "snapshots" / revision
            if all((candidate / f.rfilename).is_file() for f in files):
                path = candidate
                break
        if path is None:
            required = sum(f.size or 0 for f in files)
            ledger.check(projected_bytes=required + 256 * 1024 * 1024)
            path = Path(snapshot_download(config["model_id"], revision=revision, cache_dir=local_cache,
                                         allow_patterns=[f.rfilename for f in files], max_workers=1))
            # Conservative logical bytes; HF metadata traffic is not included in this estimate.
            ledger.value["model_download_logical_bytes"] = required
            ledger.save()
    except Blocked:
        raise
    except Exception as exc:
        detail = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[query redacted]", str(exc))[:1000]
        raise Blocked("BLOCKED_MODEL_ACCESS", f"Pinned model acquisition failed: {type(exc).__name__}: {detail}") from exc
    processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_fast=True)
    generation = GenerationConfig(max_new_tokens=config["max_new_tokens"], do_sample=False, num_beams=1,
                                   eos_token_id=processor.tokenizer.eos_token_id,
                                   pad_token_id=processor.tokenizer.pad_token_id,
                                   bos_token_id=processor.tokenizer.bos_token_id, use_cache=True)
    if generation.eos_token_id is None or generation.pad_token_id is None:
        raise Blocked("INPUT_INTEGRITY_FAILED", "EOS and padding must resolve explicitly")
    manifest = {"model_id": config["model_id"], "revision": revision, "processor_revision": revision,
                "tokenizer_revision": revision, "model_path": str(path.resolve()),
                "files": {f.rfilename: file_hash(path / f.rfilename) for f in files},
                "dtype": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
                "attention_backend": config["attention_backend"], "generation": generation.to_dict(),
                "processor_source_hashes": adapter_source_hashes(),
                "chat_template_hash": digest(processor.chat_template),
                "spatial": {"factor": 32, "pixel_cap": config["max_pixels_per_frame"],
                            "resize": "Pillow RGB BICUBIC before native video processor; rounding down to grid",
                            "do_resize": False, "do_sample_frames": False,
                            "image_mean": [0.5] * 3, "image_std": [0.5] * 3, "rescale_factor": 1 / 255},
                "temporal": "CFR only, source-index/source-fps, pair mean rounded to 0.1 seconds by native processor",
                "weights_frozen": True, "offload": False, "quantization": None,
                "os_file_cache": "Ordinary OS caching permitted; not cold disk"}
    return manifest


def freeze(config, run, root, ledger):
    if not (run / "cohort_manifest.json").exists():
        raise Blocked("BLOCKED_INPUT", "Run prepare to freeze eight eligible full videos first")
    cohort = read_json(run / "cohort_manifest.json")
    validate_cohort(cohort, config)
    manifest = resolve_model(config, root, ledger)
    env = environment(root)
    order = make_order(config, cohort)
    frozen = {**config, **manifest, "run_id": run.name, "config_input_hash": digest(config),
              "model_revision": manifest["revision"], "annotation_revision": cohort["dataset"]["revision"],
              "source_git_revision": command_output(["git", "rev-parse", "HEAD"]).get("stdout"),
              "model_manifest_hash": digest(manifest), "cohort_hash": digest(cohort), "order_hash": digest(order),
              "environment_fingerprint": environment_fingerprint(env), "source_hash": source_hash(root),
              "prompt": PROMPT, "parser_version": PARSER_VERSION, "measurement_version": MEASUREMENT_VERSION,
              "two_warmups": [[cohort["videos"][0]["source_id"], q["question_id"]]
                              for q in cohort["videos"][0]["questions"][:2]],
              "profile_question": [cohort["videos"][0]["source_id"], cohort["videos"][0]["questions"][0]["question_id"]]}
    frozen["config_hash"] = digest(frozen)
    atomic_text(run / "resolved_config.yaml", yaml.safe_dump(frozen, sort_keys=True, allow_unicode=True))
    write_json(run / "model_manifest.json", manifest)
    write_json(run / "environment.json", env)
    write_json(run / "execution_order.json", order)
    return frozen


def validate_frozen(config, run, root, check_weights=True):
    frozen_path = run / "resolved_config.yaml"
    if not frozen_path.exists():
        raise Blocked("BLOCKED_INPUT", "Run smoke to resolve and freeze the model and cohort")
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    claimed = frozen.pop("config_hash")
    if digest(frozen) != claimed:
        raise Blocked("INPUT_INTEGRITY_FAILED", "Resolved configuration hash mismatch")
    frozen["config_hash"] = claimed
    if frozen["config_input_hash"] != digest(config) or frozen["source_hash"] != source_hash(root):
        raise Blocked("INPUT_INTEGRITY_FAILED", "Configuration or implementation changed; use a new named run")
    if frozen["environment_fingerprint"] != environment_fingerprint(environment(root)):
        raise Blocked("INPUT_INTEGRITY_FAILED", "Environment changed; use a new named run")
    cohort, order = read_json(run / "cohort_manifest.json"), read_json(run / "execution_order.json")
    if digest(cohort) != frozen["cohort_hash"] or digest(order) != frozen["order_hash"]:
        raise Blocked("INPUT_INTEGRITY_FAILED", "Frozen cohort/order changed")
    validate_cohort(cohort, config)
    manifest = read_json(run / "model_manifest.json")
    if digest(manifest) != frozen["model_manifest_hash"]:
        raise Blocked("INPUT_INTEGRITY_FAILED", "Model manifest changed")
    if check_weights:
        for name, expected in manifest["files"].items():
            if file_hash(Path(manifest["model_path"], name)) != expected:
                raise Blocked("INPUT_INTEGRITY_FAILED", f"Model file changed: {name}")
    return frozen


def records(run):
    rows = [read_json(p) for p in sorted(Path(run, "records").glob("*.json"))]
    ids = [row["trial_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise Blocked("INPUT_INTEGRITY_FAILED", "Duplicate atomic trial records")
    return rows


def export_records(run):
    atomic_text(run / "raw_trials.jsonl", "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in records(run)))


def launch(config_path, run, root, block, mode, timeout):
    started = time.perf_counter()
    log = run / f"worker_{mode}_{block}_{time.time_ns()}.log"
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen([sys.executable, "-u", "-m", "video_compute_value.e000", "worker",
                                    "--config", str(config_path), "--run-dir", str(run), "--root", str(root),
                                    "--block", str(block), "--mode", mode], stdout=output, stderr=output, cwd=root)
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            process.kill()
            process.wait()
            append_json(run / "failed_attempts.jsonl", {"at": utc_now(), "kind": "worker",
                        "block": block, "mode": mode, "error": type(exc).__name__, "log": log.name})
            if isinstance(exc, KeyboardInterrupt):
                raise
            raise Blocked("PARTIAL", f"Worker timeout in block {block}; pending attempts preserved") from exc
    if code:
        latest = read_json(run / "worker_failure.json") if (run / "worker_failure.json").exists() else {}
        raise Blocked(latest.get("status", "PARTIAL"),
                      f"Worker {block}: {latest.get('error', 'failure')}: {latest.get('message', '')}; inspect {log.name}")
    return time.perf_counter() - started


def coordinate(command, config, run, root, ledger, config_path):
    from .e000 import status
    if command == "smoke":
        frozen = validate_frozen(config, run, root) if (run / "resolved_config.yaml").exists() else freeze(config, run, root, ledger)
        from .video import decode_selected
        cpu_probes = {}
        for video in read_json(run / "cohort_manifest.json")["videos"]:
            ledger.check()
            started = time.perf_counter()
            frames, details = decode_selected(video_input(video), config["reference_budget"])
            elapsed = time.perf_counter() - started
            cpu_probes[video["source_id"]] = {"elapsed_seconds": elapsed,
                "frames_decoded": details["frames_decoded"], "frames_selected": len(frames)}
            del frames
        write_json(run / "cpu_decode_probes.json", {"config_hash": frozen["config_hash"],
                   "kind": "CPU-only preparation, not model calls or primary request measurements", "videos": cpu_probes})
        probes = []
        for budget in config["frame_budgets"]:
            ledger.check()
            wall_seconds = launch(config_path, run, root, budget, "probe", min(config["worker_timeout_seconds"],
                   max(1, int(config["cumulative_execution_limit_hours"] * 3600 - ledger.consumed_seconds()))))
            probe = read_json(run / f"probe_{budget}.json")
            probe["worker_wall_seconds"] = wall_seconds
            write_json(run / f"probe_{budget}.json", probe)
            probes.append(probe)
        write_json(run / "probes.json", {"config_hash": frozen["config_hash"], "probes": probes})
        # 16 independent collection workers, each with two warmups, + all 384 measured calls.
        # Entire process wall time minus its five timed requests includes imports,
        # cohort validation, model loading, warmups' overhead, hashing and serialization.
        worst_startup = max(max(0, p["worker_wall_seconds"] - p["timed_calls_seconds"]) for p in probes)
        # Account for clips with more expensive decoding than the prespecified model probe.
        decode_adjustment = max(0, max(v["elapsed_seconds"] for v in cpu_probes.values()) -
                                cpu_probes[frozen["profile_question"][0]]["elapsed_seconds"])
        worst_request = max(p["latency_seconds"] for p in probes) + decode_adjustment
        # Text-only conservatively receives the worst video request cost.
        raw_estimate = 16 * worst_startup + (384 + 32) * worst_request + 120
        projection = raw_estimate * 1.5
        ledger.check(projected_seconds=projection)
        write_json(run / "collection_ready.json", {"config_hash": frozen["config_hash"],
                    "projection_seconds": projection, "safety_factor": 1.5,
                    "additional_decode_seconds_per_request": decode_adjustment,
                    "formula": "1.5 * (16 * worst_worker_startup_and_overhead + (384 + 32) * worst_request + 120 verification seconds)",
                    "ready_at": utc_now()})
        status(run, "PARTIAL", "All five fit, integrity and component probes passed; collection approved by resource projection")
        return
    frozen = validate_frozen(config, run, root)
    if not (run / "collection_ready.json").exists():
        raise Blocked("BLOCKED_INPUT", "Run smoke; complete fit and runtime projection required before collection")
    ready = read_json(run / "collection_ready.json")
    if ready["config_hash"] != frozen["config_hash"]:
        raise Blocked("INPUT_INTEGRITY_FAILED", "Fit/projection provenance mismatch")
    if command == "run" and (records(run) or any(e.get("kind") == "trial" for e in read_jsonl(run / "worker_calls.jsonl"))):
        raise Blocked("PARTIAL", "Existing collection attempts: use resume; never reset the ledger")
    order = read_json(run / "execution_order.json")
    completed = {r["trial_id"] for r in records(run)}
    for index, block in enumerate(order):
        wanted = [trial_identity(run.name, frozen["config_hash"], source, question, block["budget"],
                                 block["repetition"])["trial_id"] for source, question in block["questions"]]
        if all(identity in completed for identity in wanted):
            continue
        fraction_remaining = (384 - len(completed)) / 384
        ledger.check(projected_seconds=ready["projection_seconds"] * fraction_remaining)
        timeout = min(config["worker_timeout_seconds"], max(1, int(config["cumulative_execution_limit_hours"] * 3600 - ledger.consumed_seconds())))
        launch(config_path, run, root, index, "collect", timeout)
        export_records(run)
        completed = {r["trial_id"] for r in records(run)}
    from .analysis import analyze, verify
    analyze(config, run)
    verify(config, run, root)


def worker(config, run, root, block_index, mode):
    # Workers never mutate the coordinator's resource ledger; the whole process lifetime is timed there.
    from .model import QwenAdapter, tensor_hashes
    with exclusive_lock(root / ".cache" / "cuda_worker.lock"):
        try:
            frozen = validate_frozen(config, run, root, check_weights=False)
            cohort = read_json(run / "cohort_manifest.json")
            lookup = {(v["source_id"], q["question_id"]): (v, q) for v in cohort["videos"] for q in v["questions"]}
            if mode == "probe":
                block = {"budget": block_index, "repetition": 0, "questions": [frozen["profile_question"]]}
            else:
                block = read_json(run / "execution_order.json")[block_index]
            budget, repetition = block["budget"], block["repetition"]
            adapter = QwenAdapter(frozen)
            worker_key = f"{mode}_{block_index}_{time.time_ns()}"
            write_json(run / f"worker_{worker_key}.json", {"load_seconds": adapter.load_seconds,
                       "baseline": adapter.baseline, "budget": budget, "repetition": repetition, "pid": os.getpid()})
            completed = {r["trial_id"] for r in records(run)}
            timed_calls_seconds = 0.0

            def execute(pair, kind, profile=False):
                nonlocal timed_calls_seconds
                video, labelled = lookup[tuple(pair)]
                question = question_input(labelled)
                inference_video = video_input(video) if budget else None
                identity = trial_identity(run.name, frozen["config_hash"], pair[0], pair[1], budget, repetition)
                call_id = f"{worker_key}_{kind}_{time.time_ns()}"
                append_json(run / "worker_calls.jsonl", {"call_id": call_id, "event": "started", "kind": kind,
                            "trial_id": identity["trial_id"], "budget": budget, "at": utc_now()})
                started = time.perf_counter()
                model_calls_before = adapter.model_call_count
                try:
                    result, cpu_inputs = timed_request(adapter, question, inference_video, budget,
                        profile=profile, sampler=MemorySampler(config["memory_sampling_interval_seconds"]))
                    timed_calls_seconds += result["latency_seconds"]
                    # Ground truth, serialization and all hashing are outside primary latency.
                    diagnostic_start = time.perf_counter()
                    result["input_tensor_hashes"] = tensor_hashes(cpu_inputs)
                    result["input_metadata_hash"] = digest({k: result[k] for k in
                        ("tensor_shapes", "source_indices", "actual_timestamps", "model_timestamps", "video_grid_thw")})
                    del cpu_inputs
                    result["diagnostic_hash_seconds"] = time.perf_counter() - diagnostic_start
                    result.update(identity, correct=result["parsed_answer"] == labelled["answer"],
                                  call_id=call_id, worker_key=worker_key, kind=kind)
                    append_json(run / "worker_calls.jsonl", {"call_id": call_id, "event": "completed", "kind": kind,
                                "model_invoked": adapter.model_call_count > model_calls_before,
                                "elapsed_seconds": time.perf_counter() - started, "at": utc_now()})
                    gc.collect()
                    result["live_allocated_after_bytes"] = adapter.torch.cuda.memory_allocated()
                    if result["live_allocated_after_bytes"] > adapter.baseline["allocated_bytes"] + 64 * 1024 * 1024:
                        result["live_memory_growth_flag"] = True
                    return result
                except BaseException as exc:
                    append_json(run / "failed_attempts.jsonl", {**identity, "call_id": call_id, "kind": kind,
                                "model_invoked": adapter.model_call_count > model_calls_before,
                                "error": type(exc).__name__, "message": str(exc), "at": utc_now(),
                                "elapsed_until_failure_seconds": time.perf_counter() - started})
                    raise

            for pair in frozen["two_warmups"]:
                result = execute(pair, "warmup")
                append_json(run / "diagnostics" / "warmups.jsonl", result)
            if mode == "probe":
                pair = frozen["profile_question"]
                first = execute(pair, "fit_probe")
                second = execute(pair, "integrity_probe")
                if first["input_tensor_hashes"] != second["input_tensor_hashes"] or first["input_metadata_hash"] != second["input_metadata_hash"]:
                    raise Blocked("INPUT_INTEGRITY_FAILED", "Repeated preparation differs")
                component = execute(pair, "component_profile", profile=True)
                write_json(run / f"probe_{budget}.json", {"config_hash": frozen["config_hash"], "budget": budget,
                           "load_seconds": adapter.load_seconds, "latency_seconds": first["latency_seconds"],
                           "timed_calls_seconds": timed_calls_seconds,
                           "integrity_passed": True, "fit": first, "repeat": second, "component_profile": component})
            else:
                for pair in block["questions"]:
                    identity = trial_identity(run.name, frozen["config_hash"], pair[0], pair[1], budget, repetition)
                    if identity["trial_id"] in completed:
                        continue
                    result = execute(pair, "trial")
                    write_json(run / "records" / (identity["trial_id"] + ".json"), result)
            return 0
        except BaseException as exc:
            name = type(exc).__name__
            failure_status = getattr(exc, "status", "BLOCKED_RESOURCE" if "OutOfMemory" in name else "PARTIAL")
            write_json(run / "worker_failure.json", {"status": failure_status, "error": name, "message": str(exc)})
            append_json(run / "private" / "errors.jsonl", {"at": utc_now(), "traceback": traceback.format_exc()})
            traceback.print_exc()
            return 2
