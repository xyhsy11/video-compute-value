"""Independent saved-record verification for the named diagnostic; no inference."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

import yaml

from video_compute_value.common import digest, exclusive_lock, file_hash, read_json, utc_now, write_json
from video_compute_value.protocol import parse_answer
from video_compute_value.resources import Ledger


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify(root, run):
    from transformers import AutoTokenizer
    inventory = read_json(run / "parent_integrity_manifest.json")
    parent = root / inventory["parent_run"]
    actual = {p.relative_to(parent).as_posix(): file_hash(p) for p in parent.rglob("*")
              if p.is_file() and p.name not in (".lock", ".worker.lock")}
    assert actual == inventory["files"], "Historical parent files changed"
    frozen = yaml.safe_load((parent / "resolved_config.yaml").read_text(encoding="utf-8"))
    from video_compute_value.resources import package_versions
    assert package_versions() == frozen["environment_fingerprint"]["packages"]
    tokenizer = AutoTokenizer.from_pretrained(frozen["model_path"], local_files_only=True, trust_remote_code=False)
    references = {(r["source_id"], r["question_id"], r["budget"]): r["input_tensor_hashes"]
                  for r in (read_json(p) for p in (parent / "records").glob("*.json"))}
    evidence, forensic_checks = {}, []
    completed_calls = 0
    unknown_calls = 0

    def rows_at(directory):
        paths = sorted(directory.glob("*.json"))
        for path in paths:
            evidence[path.relative_to(root).as_posix()] = file_hash(path)
        return [read_json(p) for p in paths]

    def validate_output(row):
        text = tokenizer.decode(row["generated_token_ids"], skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
        assert text == row["decoded_output"] and parse_answer(text) == row["parsed_answer"]

    for phase, arms, planned in ((run, ("native", "reset_rope", "deterministic_math"), 297),
                                (run / "greedy_validation", ("fixed_greedy",), 99)):
        plan = read_json(phase / "forensic_plan.json")
        rows = rows_at(phase / "private" / "forensic")
        assert len(rows) == planned and len({r["call_id"] for r in rows}) == planned
        events = jsonl(phase / "private" / "call_events.jsonl")
        by_call = defaultdict(list)
        for event in events:
            by_call[event["call_id"]].append(event)
        for row in rows:
            validate_output(row)
            assert [e["event"] for e in by_call[row["call_id"]]] == ["started", "completed"]
            assert row["nan_score_count"] == row["positive_infinite_score_count"] == 0
            assert row["prefill_cache_length"] == row["cache_position_first"] == 0
            if row["kind"] != "prefix":
                assert row["input_hashes"] == references[(row["source_id"], row["question_id"], row["budget"])]
        for arm in arms:
            unstable = feature_unstable = score_unstable = non_argmax = 0
            for index, case in enumerate(plan["cases"]):
                group = [r for r in rows if r["arm"] == arm and r["case_index"] == index]
                assert Counter(r["kind"] for r in group) == {"consecutive": 3, "prefix": 3, "after_prefix": 3}
                targets = [r for r in group if r["kind"] != "prefix"]
                unstable += len({r["parsed_answer"] for r in targets}) > 1
                feature_unstable += len({digest(r["feature_hashes"]) for r in targets}) > 1
                score_unstable += len({r["first_scores_hash"] for r in targets}) > 1
                for row in targets:
                    assert (row["source_id"], row["question_id"], row["budget"]) == (
                        case["source_id"], case["question_id"], case["budget"])
                    non_argmax += not any(token == row["generated_token_ids"][0] and score == row["top_token_scores"][0][1]
                                          for token, score in row["top_token_scores"])
            assert feature_unstable == score_unstable == 0
            assert unstable == (0 if arm == "fixed_greedy" else 8)
            if arm == "fixed_greedy":
                assert non_argmax == 0
            forensic_checks.append({"arm": arm, "target_cells": 11, "target_calls": 66, "prefix_calls": 33,
                "answer_unstable_cells": unstable, "feature_unstable_cells": feature_unstable,
                "score_unstable_cells": score_unstable, "targets_with_non_argmax_first_token": non_argmax,
                "valid_target_outputs": sum(r["parsed_answer"] is not None for r in rows if r["arm"] == arm and r["kind"] != "prefix")})
        terminals = [e for e in events if e.get("model_invoked")]
        completed_calls += len(terminals)
        assert sum(e["event"] == "failed_recording" for e in terminals) == (1 if phase == run else 0)

    audit = read_json(run / "generation_defaults_audit.json")
    assert audit["requested"]["do_sample"] is False and audit["legacy_effective"]["do_sample"] is True
    assert audit["fixed_effective"]["do_sample"] is False and audit["fixed_generation_mode"] == "greedy_search"
    import inspect
    from transformers.generation.utils import GenerationMixin
    assert file_hash(inspect.getfile(GenerationMixin)) == audit["installed_generation_source_sha256"]
    timing = root / "results/e000_20261028_diagnostic_v2"
    plan = read_json(timing / "plan.json")
    first_questions = {(video["source_id"], video["questions"][0]["question_id"])
                       for video in read_json(parent / "cohort_manifest.json")["videos"]}
    assert len(first_questions) == 8
    assert Counter((b["arm"], b["budget"], b["repetition"]) for b in plan["blocks"]) == {
        (a, b, r): 1 for a in ("native", "request_trim") for b in (8, 64) for r in (1, 2, 3)}
    assert all(len(b["questions"]) == 8 and {tuple(p) for p in b["questions"]} == first_questions
               for b in plan["blocks"])
    assert plan["plan_hash"] == digest({k: v for k, v in plan.items() if k != "plan_hash"})
    for name, expected_hash in plan["sources"].items():
        assert file_hash(root / name) == expected_hash, name
    old_aux = rows_at(run / "private/timing_auxiliary")
    assert len(old_aux) == 1
    validate_output(old_aux[0])
    assert old_aux[0]["input_tensor_hashes"] == references[(old_aux[0]["source_id"], old_aux[0]["question_id"], old_aux[0]["budget"])]
    assert old_aux[0]["kind"] == "warmup" and old_aux[0]["arm"] == "bounded_allocator"
    old_events = jsonl(run / "private/timing_events.jsonl")
    assert Counter(e["event"] for e in old_events) == {"started": 3, "completed": 1}
    completed_calls += 1
    unknown_calls += 2
    rows = rows_at(timing / "private/records")
    aux = rows_at(timing / "private/auxiliary")
    assert plan["timed_requests"] == 96
    assert Counter(r["kind"] for r in aux) == {"warmup": 24, "profile": 4}
    expected = {(i, seq, *pair) for i, block in enumerate(plan["blocks"])
                for seq, pair in enumerate(block["questions"])}
    observed = {(r["block"], r["sequence"], r["source_id"], r["question_id"]) for r in rows}
    missing = expected - observed
    assert observed <= expected and len(observed) == len(rows)
    if missing:
        assert missing == {(10, seq, *plan["blocks"][10]["questions"][seq]) for seq in range(4, 8)}
        assert read_json(timing / "native_timeout.json")["worker_timeout_seconds"] == 600
    timed_out = int(bool(missing))
    assert len({r["call_id"] for r in rows + aux}) == len(rows) + 28
    events = jsonl(timing / "private/call_events.jsonl")
    assert Counter(e["event"] for e in events) == {"started": len(rows) + 28 + timed_out, "completed": len(rows) + 28}
    unknown = {e["call_id"] for e in events if e["event"] == "started"} - {e["call_id"] for e in events if e["event"] == "completed"}
    assert len(unknown) == timed_out
    if unknown:
        assert next(e for e in events if e["call_id"] in unknown)["block"] == 10
    unknown_calls += len(unknown)
    completed_calls += sum(bool(e.get("model_invoked")) for e in events)
    for row in rows + aux:
        validate_output(row)
        block = plan["blocks"][row["block"]]
        assert row["plan_hash"] == plan["plan_hash"]
        assert (row["arm"], row["budget"], row["repetition"]) == (block["arm"], block["budget"], block["repetition"])
        assert row["input_tensor_hashes"] == references[(row["source_id"], row["question_id"], row["budget"])]
        assert math.isfinite(row["latency_seconds"]) and 0 < row["cpu_prepare_host_seconds"] <= row["latency_seconds"]
        assert row["frames_selected"] == row["frames_processor"] == row["frames_represented"] == row["budget"]
        assert row["telemetry_after"]["num_ooms"] == 0
        assert row["peak_reserved_bytes"] >= row["peak_allocated_bytes"] > 0
        if row["arm"] == "request_trim":
            assert 0 < row["allocator_trim_host_seconds"] < row["latency_seconds"]
        else:
            assert row["allocator_trim_host_seconds"] == 0
    workers = [read_json(p) for p in (timing / "private").glob("worker_*.json")]
    assert len(workers) == 12
    assert all(w["generation_policy"] == plan["generation_policy"] for w in workers)
    assert all(w["allocator_env"] is None and w["allocator_backend"] == "native" for w in workers)
    conditions = []
    for arm in ("native", "request_trim"):
        for budget in (8, 64):
            group = [r for r in rows if r["arm"] == arm and r["budget"] == budget]
            cells = defaultdict(list)
            for row in group:
                cells[(row["source_id"], row["question_id"])].append(row)
            assert len(cells) == 8
            assert all(sorted(r["repetition"] for r in cell) in ([1, 2], [1, 2, 3]) for cell in cells.values())
            complete = len(group) == 24
            assert complete or (arm == "native" and budget == 64 and len(group) == 20 and timed_out)
            medians = [statistics.median(r["latency_seconds"] for r in cell) for cell in cells.values()]
            ranges = [(max(r["latency_seconds"] for r in cell) - min(r["latency_seconds"] for r in cell)) /
                      statistics.median(r["latency_seconds"] for r in cell) for cell in cells.values()]
            total_centered, cpu_centered = [], []
            for cell in cells.values():
                total_median = statistics.median(r["latency_seconds"] for r in cell)
                cpu_median = statistics.median(r["cpu_prepare_host_seconds"] for r in cell)
                total_centered.extend(r["latency_seconds"] - total_median for r in cell)
                cpu_centered.extend(r["cpu_prepare_host_seconds"] - cpu_median for r in cell)
            conditions.append({"arm": arm, "budget": budget, "requests": len(group), "cells": len(cells),
                "median_seconds": statistics.median(medians) if complete else None, "median_relative_range": statistics.median(ranges) if complete else None,
                "planned_requests": 24, "complete_cells": sum(len(cell) == 3 for cell in cells.values()), "complete": complete,
                "cells_exceeding_50_percent": sum(value > 0.5 for value in ranges),
                "answer_unstable_cells": sum(len({r["parsed_answer"] for r in cell}) > 1 for cell in cells.values()),
                "peak_allocated_gib": max(r["peak_allocated_bytes"] for r in group) / 1024 ** 3,
                "peak_reserved_gib": max(r["peak_reserved_bytes"] for r in group) / 1024 ** 3,
                "sampled_device_peak_gib": max(r["sampled_device_gpu_peak_bytes"] for r in group) / 1024 ** 3,
                "median_cpu_prepare_seconds": statistics.median(r["cpu_prepare_host_seconds"] for r in group),
                "median_request_minus_cpu_prepare_seconds": statistics.median(
                    r["latency_seconds"] - r["cpu_prepare_host_seconds"] for r in group),
                "median_cpu_share_of_request": statistics.median(r["cpu_prepare_host_seconds"] / r["latency_seconds"] for r in group),
                "cpu_total_centered_correlation": statistics.correlation(total_centered, cpu_centered),
                "median_cpu_relative_range": statistics.median((max(r["cpu_prepare_host_seconds"] for r in cell) -
                    min(r["cpu_prepare_host_seconds"] for r in cell)) / statistics.median(r["cpu_prepare_host_seconds"] for r in cell) for cell in cells.values()) if complete else None,
                "median_trim_host_seconds": statistics.median(r["allocator_trim_host_seconds"] for r in group),
                "max_allocation_retries": max(r["telemetry_after"]["num_alloc_retries"] for r in group),
                "max_inactive_split_gib": max(r["telemetry_after"]["inactive_split_bytes.all.current"] for r in group) / 1024 ** 3,
                "max_request_seconds": max(r["latency_seconds"] for r in group),
                "gpu_temperature_range_c": [min(r["telemetry_after"]["gpu_temperature_c"] for r in group),
                                            max(r["telemetry_after"]["gpu_temperature_c"] for r in group)],
                "gpu_sm_clock_range_mhz": [min(r["telemetry_after"]["gpu_sm_clock_mhz"] for r in group),
                                           max(r["telemetry_after"]["gpu_sm_clock_mhz"] for r in group)]})
    cross_arm = defaultdict(set)
    for row in rows:
        cross_arm[(row["source_id"], row["question_id"], row["budget"])].add(tuple(row["generated_token_ids"]))
    assert all(len(values) == 1 for values in cross_arm.values()), "Allocator changed greedy output"
    assert all(c["answer_unstable_cells"] == 0 for c in conditions)
    manifest = read_json(parent / "model_manifest.json")
    for name, expected_hash in manifest["files"].items():
        assert file_hash(Path(manifest["model_path"]) / name) == expected_hash
    assert (completed_calls, unknown_calls) == (398 + len(rows) + 28, 2 + timed_out), (completed_calls, unknown_calls)
    write_json(timing / "private" / "evidence_hashes.json", evidence)
    value = {"verified_at": utc_now(), "passed": True, "verification_scope": "Saved-record integrity and targeted diagnosis, not full feasibility", "parent_files_unchanged": len(actual),
        "model_files_rehashed": len(manifest["files"]), "raw_outputs_redecoded": 397 + len(rows) + len(aux), "primary_collection_complete": not missing,
        "primary_completed": len(rows), "primary_timed_out": timed_out, "primary_unattempted": len(missing) - timed_out, "valid_primary_outputs": sum(r["parsed_answer"] is not None for r in rows),
        "invalid_primary_outputs": sum(r["parsed_answer"] is None for r in rows),
        "forensic": forensic_checks, "timing": conditions,
        "new_confirmed_model_invocations": completed_calls, "new_additional_possible_invocations": unknown_calls,
        "historical_model_invocations": 491, "cumulative_model_invocations_upper_bound": 491 + completed_calls + unknown_calls,
        "preserved_completed_inference_with_failed_recording": 1, "matched_input_hashes": True,
        "allocator_outputs_identical": True, "verifier_source_sha256": file_hash(__file__), "private_evidence_sha256": digest(evidence), "primary_accuracy_curve": False}
    write_json(timing / "verification.json", value)
    return value


if __name__ == "__main__":
    root = Path.cwd()
    run = root / "results/e000_20261027_diagnostic_v1"
    config = yaml.safe_load((root / "configs/e000_diagnostic_v1.yaml").read_text(encoding="utf-8"))
    with exclusive_lock(root / "results/e000_20261028_diagnostic_v2/.lock"):
        ledger = Ledger(root / "results/e000_20261028_diagnostic_v2/resource_ledger.json", root, config)
        with ledger.track("diagnostic-independent-verification"):
            print(json.dumps(verify(root, run), indent=2))
