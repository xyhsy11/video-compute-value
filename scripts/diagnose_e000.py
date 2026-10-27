"""Bounded, separately named forensic and timing diagnostics of frozen Experiment 0."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import yaml

from video_compute_value.common import (Blocked, append_json, command_output,
    digest, exclusive_lock, file_hash, read_json, utc_now, write_json)
from video_compute_value.e000 import set_local_paths
from video_compute_value.model import QwenAdapter, tensor_hashes
from video_compute_value.protocol import question_input, video_input, parse_answer
from video_compute_value.resources import Ledger


def parent_inventory(parent):
    return {str(p.relative_to(parent)).replace("\\", "/"): file_hash(p)
            for p in sorted(parent.rglob("*")) if p.is_file() and p.name not in (".lock", ".worker.lock")}


def initialize(root, run, config):
    parent = root / config["parent_run"]
    run.mkdir(parents=True, exist_ok=True)
    if (run / "resource_ledger.json").exists():
        return
    ledger = read_json(parent / "resource_ledger.json")
    ledger.setdefault("prior_runs", []).append({"run_id": parent.name,
        "completed_model_calls": read_json(parent / "status.json")["completed_model_calls"],
        "reason": "Completed parent pilot preserved; separate diagnostic inherits all consumption"})
    # Conservative allowance for initial local inspection, before this coordinator existed.
    ledger["active_seconds"] += 120
    ledger.setdefault("external_events", []).append({"name": "Conservative initial diagnostic inspection allowance",
                                                    "active_seconds": 120, "recorded_at": utc_now()})
    write_json(run / "resource_ledger.json", ledger)
    write_json(run / "parent_integrity_manifest.json", {"parent_run": str(parent.relative_to(root)),
        "files": parent_inventory(parent), "parent_git_revision": command_output(["git", "rev-parse", "HEAD"]).get("stdout")})
    write_json(run / "run_lineage.json", {"parent_run": str(parent.relative_to(root)), "budget_reset": False,
        "inherited_seconds": ledger["active_seconds"] - 120, "parent_model_calls": 491,
        "diagnostic_only": True, "no_primary_trials_replaced": True})


def cases_for_parent(parent):
    flagged = read_json(parent / "repeatability_assessment.json")["flagged_cells"]
    assessment = read_json(parent / "repeatability_assessment.json")["cells"]
    cases = [{"source_id": c["source_id"], "question_id": c["question_id"], "budget": c["budget"],
              "selection": "all parent disagreement cells"} for c in flagged]
    for budget in sorted({c["budget"] for c in cases}):
        control = next(c for c in sorted(assessment, key=lambda c: (c["question_id"], c["source_id"]))
                       if c["budget"] == budget and c["identical_parsed_answers"] and c["identical_prepared_inputs"])
        cases.append({"source_id": control["source_id"], "question_id": control["question_id"], "budget": budget,
                      "selection": "lexicographically first stable control at this budget"})
    order = read_json(parent / "execution_order.json")
    frozen = yaml.safe_load((parent / "resolved_config.yaml").read_text(encoding="utf-8"))
    for case in cases:
        predecessors = []
        for rep in (1, 2, 3):
            block = next(b for b in order if b["budget"] == case["budget"] and b["repetition"] == rep)
            i = block["questions"].index([case["source_id"], case["question_id"]])
            predecessors.append(block["questions"][i - 1] if i else frozen["two_warmups"][-1])
        case["predecessors"] = predecessors
    return cases


def settings(arm):
    deterministic = arm == "deterministic_math"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" if deterministic else ":16:8"
    # Baseline must retain the original implicit cuBLAS setting, not a new deterministic choice.
    if not deterministic:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    import torch
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    return {"deterministic_algorithms": deterministic, "cudnn_deterministic": deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "sdpa_kernel": "math" if deterministic else "native dispatch", "reset_rope": arm == "reset_rope"}


def nested_tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [t for item in value for t in nested_tensors(item)]
    return []


def json_safe(value):
    """Preserve masked scores and invalid numeric observations without inventing zeros."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "+Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def forensic_worker(root, run, config, arm):
    parent = root / config["parent_run"]
    plan = read_json(run / "forensic_plan.json")
    policy = settings(arm)
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    frozen = yaml.safe_load((parent / "resolved_config.yaml").read_text(encoding="utf-8"))
    cohort = read_json(parent / "cohort_manifest.json")
    lookup = {(v["source_id"], q["question_id"]): (video_input(v), question_input(q))
              for v in cohort["videos"] for q in v["questions"]}
    with exclusive_lock(root / ".cache" / "cuda_worker.lock"):
        adapter = QwenAdapter(frozen)
        capture = {}
        def vision_hook(module, args, output):
            capture["vision"] = [t.detach().clone() for t in nested_tensors(output)]
        def language_hook(module, args, kwargs):
            if "positions" not in capture:
                capture["positions"] = kwargs["position_ids"].detach().clone()
                cp = kwargs.get("cache_position")
                capture["cache_position_first"] = int(cp[0].item()) if cp is not None else None
                past = kwargs.get("past_key_values")
                capture["prefill_cache_length"] = None if past is None else past.get_seq_length()
        handles = [adapter.model.visual.register_forward_hook(vision_hook),
                   adapter.model.language_model.register_forward_pre_hook(language_hook, with_kwargs=True)]
        cpu_cache = {}
        parent_rows = [read_json(p) for p in (parent / "records").glob("*.json")]
        for index, case in enumerate(plan["cases"]):
            budget = case["budget"]
            def prepared(pair):
                key = (pair[0], pair[1], budget)
                if key not in cpu_cache:
                    video, question = lookup[tuple(pair)]
                    cpu, meta = adapter.prepare(question, video, budget)
                    hashes = tensor_hashes(cpu)
                    prior = next(r for r in parent_rows if (r["source_id"], r["question_id"], r["budget"]) == key)
                    if hashes != prior["input_tensor_hashes"]:
                        raise Blocked("INPUT_INTEGRITY_FAILED", f"Parent input mismatch for {key}")
                    cpu_cache[key] = (cpu, meta, hashes)
                return cpu_cache[key]
            pair = [case["source_id"], case["question_id"]]
            cpu, meta, hashes = prepared(pair)
            gpu = adapter.transfer(cpu)
            gpu_initial_hashes = tensor_hashes(gpu)
            for repetition in range(1, config["forensic_repetitions"] + 1):
                prefix = case["predecessors"][repetition - 1]
                for kind, request_pair in (("consecutive", pair), ("prefix", prefix), ("after_prefix", pair)):
                    use_gpu = gpu
                    if kind == "prefix":
                        prefix_cpu, _, _ = prepared(request_pair)
                        use_gpu = adapter.transfer(prefix_cpu)
                    if policy["reset_rope"]:
                        adapter.model.model.rope_deltas = None
                    before = adapter.model.model.rope_deltas
                    before = None if before is None else before.detach().cpu().tolist()
                    capture.clear()
                    call_id = f"forensic_{plan['script_sha256'][:10]}_{arm}_{index}_{repetition}_{kind}"
                    path = run / "private" / "forensic" / f"{call_id}.json"
                    if path.exists():
                        raise Blocked("INPUT_INTEGRITY_FAILED", "Forensic call already exists; preserve it and use a new arm/version")
                    append_json(run / "private" / "call_events.jsonl", {"call_id": call_id, "event": "started", "kind": kind,
                                "arm": arm, "at": utc_now()})
                    started = time.perf_counter()
                    context = sdpa_kernel(SDPBackend.MATH) if policy["deterministic_algorithms"] else nullcontext()
                    with torch.inference_mode(), context:
                        output = adapter.model.generate(**use_gpu, generation_config=adapter.generation,
                                                        return_dict_in_generate=True, output_scores=True)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    ids = output.sequences[0, use_gpu["input_ids"].shape[-1]:].cpu().tolist()
                    text = adapter.processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    scores = output.scores[0].float().cpu()[0]
                    values, top_ids = scores.topk(5)
                    features = {f"vision_{i}": t for i, t in enumerate(capture.get("vision", []))}
                    features["positions"] = capture["positions"]
                    fingerprints = tensor_hashes(features)
                    record = {"call_id": call_id, "arm": arm, "case_index": index, "kind": kind,
                        "repetition": repetition, "source_id": request_pair[0], "question_id": request_pair[1],
                        "budget": budget, "policy": policy, "generated_token_ids": ids,
                        "decoded_output": text, "parsed_answer": parse_answer(text),
                        "first_scores_hash": tensor_hashes({"scores": scores})["scores"],
                        "top_token_scores": list(zip(top_ids.tolist(), values.tolist())),
                        "option_scores": {letter: float(scores[adapter.processor.tokenizer.encode(letter, add_special_tokens=False)[0]])
                                          for letter in "ABCD"},
                        "top_margin": float(values[0] - values[1]), "feature_hashes": fingerprints,
                        "nan_score_count": int(torch.isnan(scores).sum()),
                        "negative_infinite_score_count": int(torch.isneginf(scores).sum()),
                        "positive_infinite_score_count": int(torch.isposinf(scores).sum()),
                        "rope_before": before, "rope_after": adapter.model.model.rope_deltas.cpu().tolist(),
                        "cache_position_first": capture["cache_position_first"],
                        "prefill_cache_length": capture["prefill_cache_length"],
                        "instrumented_generation_seconds": elapsed,
                        "input_hashes": hashes if kind != "prefix" else None,
                        "allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved()}
                    if kind != "prefix" and tensor_hashes(gpu) != gpu_initial_hashes:
                        raise Blocked("INPUT_INTEGRITY_FAILED", "Generation mutated diagnostic GPU inputs")
                    write_json(path, json_safe(record))
                    append_json(run / "private" / "call_events.jsonl", {"call_id": call_id, "event": "completed",
                                "kind": kind, "arm": arm, "model_invoked": True, "at": utc_now()})
                    del output, scores, features
                    capture.clear()
                    if kind == "prefix":
                        del use_gpu
                    gc.collect()
            del gpu
        for handle in handles:
            handle.remove()
        write_json(run / f"forensic_worker_{arm}.json", {"policy": policy, "model_load_seconds": adapter.load_seconds,
                   "calls": len(plan["cases"]) * config["forensic_repetitions"] * 3})


def forensic_summary(run):
    rows = [read_json(p) for p in (run / "private" / "forensic").glob("*.json")]
    result = []
    for arm in sorted({r["arm"] for r in rows}):
        cases = []
        for index in sorted({r["case_index"] for r in rows if r["arm"] == arm}):
            group = [r for r in rows if r["arm"] == arm and r["case_index"] == index and r["kind"] != "prefix"]
            first = group[0]
            cases.append({"source_id": first["source_id"], "question_id": first["question_id"], "budget": first["budget"],
                "calls": len(group), "answer_variants": len({r["parsed_answer"] for r in group}),
                "score_variants": len({r["first_scores_hash"] for r in group}),
                "vision_variants": len({digest({k:v for k,v in r["feature_hashes"].items() if k!="positions"}) for r in group}),
                "position_variants": len({r["feature_hashes"]["positions"] for r in group}),
                "minimum_finite_top_margin": min((r["top_margin"] for r in group if isinstance(r["top_margin"], (int, float))), default=None),
                "calls_with_nan_scores": sum(r["nan_score_count"] > 0 for r in group),
                "prefill_cache_lengths": sorted({r["prefill_cache_length"] for r in group if r["prefill_cache_length"] is not None})})
        result.append({"arm": arm, "cases": cases, "answer_unstable_cases": sum(c["answer_variants"] > 1 for c in cases),
                       "feature_unstable_cases": sum(c["vision_variants"] > 1 for c in cases)})
    write_json(run / "forensic_summary.json", {"arms": result, "instrumented_model_calls": len(rows),
                                               "primary_latency_measurements": False})
    return result


def launch(root, run, config_path, config, ledger, arm):
    log = run / "private" / f"forensic_{arm}_{time.time_ns()}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    remaining = config["cumulative_execution_limit_hours"] * 3600 - ledger.consumed_seconds() - config["reserve_final_validation_seconds"]
    ledger.check(projected_seconds=min(900, remaining))
    if remaining < 120:
        raise Blocked("BLOCKED_RESOURCE", "Insufficient allowance including final verification reserve")
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "worker",
            "--config", str(config_path), "--run-dir", str(run), "--arm", arm], cwd=root, stdout=output, stderr=output)
        try:
            code = process.wait(timeout=min(config["forensic_worker_timeout_seconds"], remaining))
        except BaseException:
            process.kill()
            process.wait()
            raise
    if code:
        raise Blocked("PARTIAL", f"Forensic worker {arm} failed; preserved {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "forensics", "summarize", "worker"))
    parser.add_argument("--config", type=Path, default=Path("configs/e000_diagnostic_v1.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("results/e000_20261027_diagnostic_v1"))
    parser.add_argument("--arm", choices=("native", "reset_rope", "deterministic_math"))
    args = parser.parse_args()
    root, run = Path.cwd(), args.run_dir.resolve()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    set_local_paths(root)
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "4"
    if args.command == "worker":
        try:
            forensic_worker(root, run, config, args.arm)
            return 0
        except BaseException as exc:
            append_json(run / "private" / "failures.jsonl", {"arm": args.arm, "at": utc_now(),
                        "error": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
            traceback.print_exc()
            return 1
    initialize(root, run, config)
    with exclusive_lock(run / ".lock"):
        ledger = Ledger(run / "resource_ledger.json", root, config)
        with ledger.track("diagnostic-" + args.command):
            if args.command == "init":
                write_json(run / "status.json", {"status": "PARTIAL", "diagnostic_only": True})
            elif args.command == "forensics":
                plan_path = run / "forensic_plan.json"
                plan = {"cases": cases_for_parent(root / config["parent_run"]), "arms": config["forensic_arms"],
                        "config": config, "script_sha256": file_hash(__file__), "frozen_at": utc_now(),
                        "intrusive": True, "prepared_tensor_reuse": True,
                        "maximum_model_calls": 11 * 3 * 3 * 3}
                if plan_path.exists():
                    prior = read_json(plan_path)
                    if prior["script_sha256"] != file_hash(__file__):
                        raise Blocked("INPUT_INTEGRITY_FAILED", "Forensic implementation changed; use a new diagnostic phase")
                else:
                    write_json(plan_path, plan)
                for arm in config["forensic_arms"]:
                    if (run / f"forensic_worker_{arm}.json").exists():
                        continue
                    launch(root, run, args.config.resolve(), config, ledger, arm)
                    print(json.dumps(forensic_summary(run), indent=2), flush=True)
            elif args.command == "summarize":
                print(json.dumps(forensic_summary(run), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
