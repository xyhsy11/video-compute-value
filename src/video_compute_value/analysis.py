"""Reproduce summaries from atomic records; incomplete cells never form a complete curve."""

from __future__ import annotations

import csv
import io
import math
import statistics
from collections import defaultdict
from pathlib import Path

from .common import atomic_text, digest, read_json, read_jsonl, utc_now, write_json
from .protocol import parse_answer, trial_identity


def median(values):
    return statistics.median(values) if values else None


def spread(values):
    if not values:
        return {"min": None, "q1": None, "median": None, "q3": None, "max": None}
    import numpy as np
    return {key: float(value) for key, value in zip(("min", "q1", "median", "q3", "max"),
                                                   np.percentile(values, [0, 25, 50, 75, 100]))}


def summarize(config, rows, failures, events):
    n_questions = config["num_videos"] * config["questions_per_video"]
    result, cells = [], []
    reference = {r["question_id"]: r for r in rows if r["budget"] == config["reference_budget"] and r["repetition"] == 1}
    for budget in [*config["frame_budgets"], 0]:
        subset = [r for r in rows if r["budget"] == budget]
        expected_reps = 1 if budget == 0 else config["repetitions"]
        planned = n_questions * expected_reps
        successes = {r["trial_id"] for r in subset}
        failed = [f for f in failures if f.get("kind") == "trial" and f.get("budget") == budget]
        attempted = successes | {f["trial_id"] for f in failed if "trial_id" in f}
        attempted |= {e["trial_id"] for e in events if e.get("event") == "started" and
                       e.get("kind") == "trial" and e.get("budget") == budget}
        primary = [r for r in subset if r["repetition"] == 1]
        correct = sum(r["correct"] for r in primary)
        invalid = sum(not r["valid_answer"] for r in primary)
        complete = len(subset) == planned and len(primary) == n_questions
        per_question = defaultdict(list)
        for row in subset:
            per_question[(row["source_id"], row["question_id"])].append(row)
        question_medians, timing_ranges, video_medians = [], [], defaultdict(list)
        agreements, token_agreements, output_agreements, input_agreements = [], [], [], []
        for (source, question), group in sorted(per_question.items()):
            times = [r["latency_seconds"] for r in group]
            full = len(group) == expected_reps and len({r["repetition"] for r in group}) == expected_reps
            relative = (max(times) - min(times)) / median(times) if full else None
            if full:
                question_medians.append(median(times))
                timing_ranges.append(relative)
                video_medians[source].append(median(times))
                if budget:
                    agreements.append(len({r["parsed_answer"] for r in group}) == 1)
                    token_agreements.append(len({tuple(r["generated_token_ids"]) for r in group}) == 1)
                    output_agreements.append(len({r["decoded_output"] for r in group}) == 1)
                    input_agreements.append(len({digest([r["input_tensor_hashes"], r["input_metadata_hash"]]) for r in group}) == 1)
            cells.append({"source_id": source, "question_id": question, "budget": budget,
                          "complete_repetitions": full, "latency_seconds": times,
                          "median_seconds": median(times) if full else None,
                          "relative_timing_range": relative, "above_50_percent": relative is not None and relative > 0.5})
        paired = [(r, reference[r["question_id"]]) for r in primary if r["question_id"] in reference] if budget else []
        row = {"condition": f"Uniform-{budget}" if budget else "text-only (separate diagnostic)", "budget": budget,
               "planned": planned, "attempted": len(attempted), "successful": len(subset),
               "failed_attempts": len(failed), "failed_without_success": len(attempted - successes),
               "unattempted": planned - len(attempted), "complete": complete,
               "rep1_correct_out_of_24": correct, "rep1_successful": len(primary),
               "accuracy": correct / n_questions if complete else None,
               "invalid_answer_rate": invalid / n_questions if complete else None,
               "paired_with_reference": len(paired),
               "paired_answer_changes": sum(a["parsed_answer"] != b["parsed_answer"] for a, b in paired),
               "latency_seconds": spread(question_medians) if complete else spread([]),
               "observed_complete_question_medians": question_medians,
               "per_video_median_seconds": {key: median(values) for key, values in video_medians.items()},
               "median_relative_timing_range": median(timing_ranges),
               "timing_stability_target_passed": complete and median(timing_ranges) <= 0.2,
               "parsed_answer_agreement": all(agreements) if budget and complete else None,
               "raw_token_agreement": all(token_agreements) if budget and complete else None,
               "raw_output_agreement": all(output_agreements) if budget and complete else None,
               "prepared_input_agreement": all(input_agreements) if budget and complete else None,
               "live_memory_growth_flag": any(r.get("live_memory_growth_flag", False) for r in subset)}
        for key in ("visual_tokens", "input_tokens", "output_tokens"):
            row[key] = spread([r[key] for r in subset])
        for key in ("peak_allocated_bytes", "peak_reserved_bytes", "sampled_device_gpu_peak_bytes",
                    "sampled_process_gpu_peak_bytes", "sampled_ram_peak_bytes"):
            values = [r.get(key) for r in subset if r.get(key) is not None]
            row[key] = max(values) if values else None
        result.append(row)
    reference_row = next(r for r in result if r["budget"] == config["reference_budget"])
    for row in result:
        ref_latency = reference_row["latency_seconds"]["median"]
        current = row["latency_seconds"]["median"]
        row["measured_speedup_vs_64"] = ref_latency / current if row["budget"] and ref_latency and current else None
        row["frame_count_ratio_vs_64"] = row["budget"] / config["reference_budget"] if row["budget"] else None
    calls_started = [e for e in events if e.get("event") == "started"]
    kinds = sorted({e.get("kind", "unknown") for e in calls_started})
    return {"conditions": result, "cells": cells, "planned_video_calls": 360, "planned_text_calls": 24,
            "successful_video_calls": sum(r["budget"] > 0 for r in rows),
            "successful_text_calls": sum(r["budget"] == 0 for r in rows),
            "request_attempts_by_kind": {kind: sum(e.get("kind") == kind for e in calls_started) for kind in kinds},
            "confirmed_model_invocations": sum(e.get("model_invoked", False) for e in events if e.get("event") == "completed") +
                                           sum(f.get("model_invoked", False) for f in failures),
            "all_collection_complete": all(r["complete"] for r in result),
            "definitions": {
                "accuracy": "Repetition 1 correct / fixed 24 questions; withheld until complete required collection",
                "timing": "Median over repetitions for each question, then distribution over the fixed 24 question medians",
                "per_video": "Median of that video's three question-level latency medians",
                "speedup": "median(24 reference question medians) / median(24 condition question medians)",
                "frame_ratio": "requested budget / 64; distinct from measured speedup",
                "relative_timing_range": "(max repetition latency - min repetition latency) / median repetition latency",
                "memory": "Maximum observed absolute per-request PyTorch peak including model weights; sampled NVML/RAM labeled separately",
                "failed_without_success": "Attempted trial identities with no valid persisted completed record; never dropped from planned denominator",
                "sample_size": "24 questions clustered in eight videos; repeated calls are not new examples"}}


def setup_costs(run):
    worker_reports = [read_json(p) for p in run.glob("worker_*.json") if p.name != "worker_failure.json"]
    completed = [e for e in read_jsonl(run / "worker_calls.jsonl") if e.get("event") == "completed"]
    ledger = read_json(run / "resource_ledger.json") if (run / "resource_ledger.json").exists() else {}
    return {
        "worker_model_load_seconds": sum(r.get("load_seconds", 0) for r in worker_reports),
        "worker_model_load_count": len(worker_reports),
        "warmup_elapsed_seconds": sum(e["elapsed_seconds"] for e in completed if e.get("kind") == "warmup"),
        "probe_profile_elapsed_seconds": sum(e["elapsed_seconds"] for e in completed
                                              if e.get("kind") in ("fit_probe", "integrity_probe", "component_profile")),
        "preparation_session_seconds": sum(s.get("active_seconds", 0) for s in ledger.get("sessions", []) if s["name"] == "prepare"),
        "external_setup_seconds": sum(e["active_seconds"] for e in ledger.get("external_events", [])),
        "definition": "Descriptive startup/diagnostic costs; overlapping wall intervals are not summed as a disjoint decomposition",
    }


def analyze(config, run):
    from .runner import export_records, records
    run = Path(run)
    export_records(run)
    rows = records(run)
    summary = summarize(config, rows, read_jsonl(run / "failed_attempts.jsonl"), read_jsonl(run / "worker_calls.jsonl"))
    summary["setup_costs"] = setup_costs(run)
    summary["generated_at"] = utc_now()
    write_json(run / "summary.json", summary)
    buffer = io.StringIO(newline="")
    fields = ["condition", "planned", "attempted", "successful", "failed_attempts", "unattempted",
              "rep1_correct_out_of_24", "rep1_successful", "accuracy", "invalid_answer_rate",
              "median_request_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "measured_speedup_vs_64"]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in summary["conditions"]:
        writer.writerow({key: row["latency_seconds"]["median"] if key == "median_request_seconds" else row.get(key) for key in fields})
    atomic_text(run / "summary.csv", buffer.getvalue())
    if summary["all_collection_complete"]:
        figures(summary, run)
    else:
        atomic_text(run / "figures" / "README.md", "No complete measured curve is available. Figures are generated only after all 384 required calls complete.\n")
    write_report(config, run, summary)
    return summary


def figures(summary, run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in summary["conditions"] if r["budget"]]
    path = run / "figures"
    path.mkdir(exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 4), layout="constrained")
    axis.plot([r["latency_seconds"]["median"] for r in rows], [r["accuracy"] for r in rows], "o-")
    for row in rows:
        axis.annotate(str(row["budget"]), (row["latency_seconds"]["median"], row["accuracy"]))
    axis.set(xlabel="Resident-model request latency (s)", ylabel="Rep. 1 accuracy (24 questions)", ylim=(0, 1))
    fig.savefig(path / "accuracy_latency.png", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6, 4), layout="constrained")
    for key, label in (("peak_allocated_bytes", "Allocated"), ("peak_reserved_bytes", "Reserved")):
        axis.plot([r["budget"] for r in rows], [r[key] / 1024 ** 3 for r in rows], "o-", label=label)
    axis.set(xlabel="Uniform frame budget", ylabel="Absolute PyTorch GPU peak (GiB)")
    axis.legend()
    fig.savefig(path / "memory_frames.png", dpi=180)
    plt.close(fig)


def write_report(config, run, summary):
    existing = read_json(run / "status.json") if (run / "status.json").exists() else {"status": "PARTIAL", "message": "Collection not complete"}
    ledger = read_json(run / "resource_ledger.json") if (run / "resource_ledger.json").exists() else {}
    table = ["| Condition | Successful / planned | Rep. 1 correct / 24 | Accuracy | Median request s | GPU allocated / reserved GiB |",
             "|---|---:|---:|---:|---:|---:|"]
    def show(value, digits=3):
        return "unavailable" if value is None else f"{value:.{digits}f}"
    for row in summary["conditions"]:
        memory = " / ".join(show(row[k] / 1024 ** 3 if row[k] is not None else None) for k in ("peak_allocated_bytes", "peak_reserved_bytes"))
        correct = str(row["rep1_correct_out_of_24"]) if row["rep1_successful"] else "unavailable"
        table.append(f"| {row['condition']} | {row['successful']} / {row['planned']} | {correct} / 24 | {show(row['accuracy'])} | {show(row['latency_seconds']['median'])} | {memory} |")
    text = f"""# Experiment 0 — frozen visual-only video-QA feasibility

Status: **{existing['status']}**. {existing['message']}

Model: `{config['model_id']}`. Planned: 8 complete short Video-MME videos,
three official questions each; budgets 8/16/32/48/64; three repetitions; batch one;
greedy generation up to eight new tokens. No audio, subtitles, ASR, or captions.
Uniform-64 is a sampled operational reference, not all-frame processing, an oracle,
or an assumed accuracy upper bound. Uniform sets are not necessarily nested.

{chr(10).join(table)}

Real completed measured calls: {summary['successful_video_calls']} / 360 video,
{summary['successful_text_calls']} / 24 text-only. Additional request attempts:
`{summary['request_attempts_by_kind']}`. Text-only is a separate diagnostic.
Missing measurements are unavailable, never zero. Incomplete subsets do not form a
complete quality–cost curve. Invalid answers count as incorrect; no best-repetition selection.

Resident-model request latency starts before opening the compressed local video and
includes full sequential decoding, selection, resize, processing, tokenization,
GPU transfer, vision/prefill/generation, output transfer, text decoding, parsing,
and final CUDA synchronization. OS file caching is permitted: these are not cold-disk
measurements. Labels/scoring, diagnostic hashes, serialization, model loading,
two warmups per worker, integrity probes and component profiling are outside this
primary interval, but included in the execution ledger. Each request decodes again.
No frame/features/prefix/KV reuse occurs between questions.

Timing summaries first take each question's median over three repetitions, then
the distribution over 24 question medians. Speedup is reference median / condition
median, reported separately from budget / 64. Relative timing range is
(maximum - minimum) / median. The proposed per-budget median target is <=20%;
individual cells above 50% are flagged. These are operational diagnostics.

PyTorch peaks are absolute allocated/reserved bytes including weights; increments
are separately labeled. These are not total device memory. Optional NVML and RAM
maxima are sampled at the recorded interval. Fresh workers isolate budget/repetition
allocator histories. Live allocation growth is flagged separately from reservations.

Recorded active execution seconds before this report command finishes:
{ledger.get('active_seconds', 'unavailable')}. Recorded project storage bytes:
{ledger.get('storage_current_bytes', 'unavailable')}; peak observed bytes:
{ledger.get('storage_peak_bytes', 'unavailable')}. Limits: 14,400 active seconds,
20 GiB additional storage, with at most 1 GiB video transfer. Setup costs and failed
attempts remain in `resource_ledger.json`; idle time between closed sessions is excluded.

Saved setup and diagnostic costs: `{summary.get('setup_costs', {})}`.
These descriptive intervals can overlap; they are not added into a purported
disjoint decomposition of the primary request timer.

Operational confirmation requires all source/config/environment/integrity gates,
all 384 completed calls, stable parsed answers, interpretable timing, complete
memory records, reconciled calls, and independent saved-record verification.
Consult `verification.json` and per-cell diagnostics in `summary.json`.
Frozen provenance files and raw trials remain local because annotations and labels
are restricted; safe source identifiers/hashes and aggregate reports may be committed.

This is eight videos and 24 questions. Repetitions do not increase sample size.
Operational feasibility does not establish useful visual reasoning, adaptive
sampling, oracle headroom, statistical significance, or a research contribution.
Quality versus text-only and concentrated question-level errors can be inspected
only once real records exist. Real GPU paths remain unverified until exercised.

Next action: resolve the recorded blocker, then run `prepare`, `smoke`, and `run`
using the exact commands in the README. Use `resume` after interrupted collection.
"""
    atomic_text(run / "report.md", text)


def verify(config, run, root=None):
    from .runner import records
    from .model import validate_grid
    run = Path(run)
    rows, errors = records(run), []
    cohort = read_json(run / "cohort_manifest.json") if (run / "cohort_manifest.json").exists() else None
    lookup = {(v["source_id"], q["question_id"]): (v, q) for v in cohort["videos"] for q in v["questions"]} if cohort else {}
    expected_trials = set()
    provenance_verified = False
    if rows:
        try:
            from .runner import validate_frozen
            frozen = validate_frozen(config, run, Path(root) if root else run.parent.parent)
            order = read_json(run / "execution_order.json")
            expected_trials = {trial_identity(run.name, frozen["config_hash"], pair[0], pair[1], block["budget"],
                                              block["repetition"])["trial_id"] for block in order for pair in block["questions"]}
            provenance_verified = True
        except (RuntimeError, OSError, ValueError) as exc:
            errors.append(f"Frozen provenance verification failed: {exc}")
    for row in rows:
        if row["trial_id"] not in expected_trials:
            errors.append(f"Trial not in frozen execution order: {row['trial_id']}")
        identity = trial_identity(row["run_id"], row["config_hash"], row["source_id"], row["question_id"], row["budget"], row["repetition"])
        if identity["trial_id"] != row["trial_id"]:
            errors.append(f"Identity mismatch: {row['trial_id']}")
        if parse_answer(row["decoded_output"]) != row["parsed_answer"] or row["valid_answer"] != (row["parsed_answer"] is not None):
            errors.append(f"Parser mismatch: {row['trial_id']}")
        if row["output_tokens"] != len(row["generated_token_ids"]):
            errors.append(f"Generated-token mismatch: {row['trial_id']}")
        if row["latency_seconds"] <= 0 or not math.isfinite(row["latency_seconds"]):
            errors.append(f"Invalid latency: {row['trial_id']}")
        for metric in ("allocated", "reserved"):
            if row[f"peak_{metric}_bytes"] < row[f"baseline_{metric}_bytes"]:
                errors.append(f"Invalid memory peak: {row['trial_id']}")
        if (row["source_id"], row["question_id"]) not in lookup:
            errors.append(f"Unknown question: {row['trial_id']}")
            continue
        video, labelled = lookup[(row["source_id"], row["question_id"])]
        if row["correct"] != (row["parsed_answer"] == labelled["answer"]):
            errors.append(f"Scoring mismatch: {row['trial_id']}")
        if row["budget"]:
            try:
                validate_grid(row["video_grid_thw"], row["budget"], video["target_height"], video["target_width"], row["visual_tokens"])
                if any(row[key] != row["budget"] for key in ("frames_selected", "frames_processor", "frames_represented")):
                    raise ValueError("Frame count mismatch")
                expected = video["selections"][str(row["budget"])]
                if row["source_indices"] != expected["indices"] or row["actual_timestamps"] != expected["timestamps"]:
                    raise ValueError("Temporal correspondence mismatch")
            except (ValueError, RuntimeError) as exc:
                errors.append(f"{row['trial_id']}: {exc}")
    events = read_jsonl(run / "worker_calls.jsonl")
    failures = read_jsonl(run / "failed_attempts.jsonl")
    reconstructed = summarize(config, rows, failures, events)
    reconstructed["setup_costs"] = setup_costs(run)
    saved = read_json(run / "summary.json")
    saved.pop("generated_at", None)
    summary_matches = saved == reconstructed
    if not summary_matches:
        errors.append("Saved summary does not reproduce from raw records")
    started = {e["call_id"] for e in events if e.get("event") == "started"}
    ended = {e["call_id"] for e in events if e.get("event") == "completed"}
    ended |= {f["call_id"] for f in failures if "call_id" in f}
    orphaned = sorted(started - ended)
    conditions = reconstructed["conditions"]
    video_conditions = [r for r in conditions if r["budget"]]
    complete = reconstructed["all_collection_complete"]
    gates = {"complete_collection": complete, "summary_reproduces": summary_matches,
             "frozen_provenance_verified": provenance_verified,
             "records_valid": not errors, "calls_reconciled": not orphaned,
             "prepared_input_repeatability": complete and all(r["prepared_input_agreement"] for r in video_conditions),
             "parsed_answer_repeatability": complete and all(r["parsed_answer_agreement"] for r in video_conditions),
             "timing_stability": complete and all(r["timing_stability_target_passed"] for r in video_conditions),
             "no_live_memory_growth_flags": complete and not any(r["live_memory_growth_flag"] for r in conditions)}
    value = {**gates, "errors": errors, "unclosed_call_ids": orphaned,
             "verified_records": len(rows), "real_inference_verified": bool(rows), "verified_at": utc_now()}
    write_json(run / "verification.json", value)
    if complete:
        from .e000 import status
        if errors:
            status(run, "INPUT_INTEGRITY_FAILED", "; ".join(errors[:3]))
        elif not gates["prepared_input_repeatability"]:
            status(run, "INPUT_INTEGRITY_FAILED", "Prepared inputs differ across repeated requests")
        elif not gates["parsed_answer_repeatability"]:
            status(run, "REPEATABILITY_UNRESOLVED", "Parsed answers disagree across repetitions")
        elif not gates["timing_stability"] or not gates["no_live_memory_growth_flags"]:
            status(run, "MEASUREMENT_UNSTABLE", "Timing stability or live allocation growth requires interpretation")
        elif all(gates.values()):
            status(run, "FEASIBILITY_CONFIRMED", "All implemented operational and saved-record gates passed")
        else:
            status(run, "PARTIAL", "Call accounting requires reconciliation")
    write_report(config, run, reconstructed)
    return value
