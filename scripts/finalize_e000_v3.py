"""Independent saved-record checks and scientific closeout; never invokes the model."""
from __future__ import annotations

from collections import defaultdict
import csv
import io
from pathlib import Path
import re
import statistics
import subprocess

import yaml

from video_compute_value.common import atomic_text, digest, file_hash, read_json, read_jsonl, utc_now, write_json
from video_compute_value.e000 import require_cumulative_ledger
from video_compute_value.resources import Ledger

ROOT = Path.cwd()
RUN = ROOT / "results/e000_20261028_greedy_collection_v3"
PARENT = ROOT / "results/e000_20261027_collection_v2"


def question_median(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row["question_id"]].append(row[key])
    return statistics.median(statistics.median(values) for values in groups.values())


def closeout(ledger):
    frozen = yaml.safe_load((RUN / "resolved_config.yaml").read_text(encoding="utf-8"))
    rows = [read_json(p) for p in (RUN / "records").glob("*.json")]
    summary = read_json(RUN / "summary.json")
    verification = read_json(RUN / "verification.json")
    assert len(rows) == 384, "This closeout requires complete actual collection"
    assert len({r["trial_id"] for r in rows}) == 384
    assert verification["records_valid"] and verification["summary_reproduces"]
    assert verification["frozen_provenance_verified"] and verification["calls_reconciled"]
    sources = read_json(RUN / "source_provenance.json")["source_files_sha256"]
    assert all(file_hash(ROOT / name) == expected for name, expected in sources.items())
    cohort = read_json(RUN / "cohort_manifest.json")
    labels = {q["question_id"]: q["answer"] for v in cohort["videos"] for q in v["questions"]}
    references = {(r["source_id"], r["question_id"], r["budget"], r["repetition"]): r
                  for r in (read_json(p) for p in (PARENT / "records").glob("*.json"))}
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(frozen["model_path"]) / "tokenizer.json"))
    cells = defaultdict(list)
    for row in rows:
        ref = references[row["source_id"], row["question_id"], row["budget"], row["repetition"]]
        assert row["input_tensor_hashes"] == ref["input_tensor_hashes"]
        assert row["input_metadata_hash"] == ref["input_metadata_hash"]
        assert tokenizer.decode(row["generated_token_ids"], skip_special_tokens=True) == row["decoded_output"]
        assert row["correct"] == (row["parsed_answer"] == labels[row["question_id"]])
        assert row["generation_policy_sha256"] == digest(frozen["generation_policy"])
        assert row["latency_seconds"] > 0 and row["cpu_prepare_host_seconds"] > 0
        row["model_transfer_output_remainder_seconds"] = row["latency_seconds"] - row["cpu_prepare_host_seconds"]
        assert row["model_transfer_output_remainder_seconds"] >= 0
        cells[row["budget"], row["question_id"]].append(row)
    table, costs, question_rows, timing_diagnosis, timing_lines = [], [], [], [], []
    by_budget = {b: [r for r in rows if r["budget"] == b] for b in (0, 8, 16, 32, 48, 64)}
    reference_median = question_median(by_budget[64], "latency_seconds")
    for budget, group in by_budget.items():
        assert len(group) == (24 if budget == 0 else 72)
        rep1 = [r for r in group if r["repetition"] == 1]
        assert len(rep1) == 24 and len({r["question_id"] for r in rep1}) == 24
        ranges, same_inputs, same_tokens, same_outputs, same_answers = [], [], [], [], []
        cpu_ranges, remainder_ranges, cpu_residuals, total_residuals, cpu_fractions = [], [], [], [], []
        for question in labels:
            repeated = cells[budget, question]
            assert len(repeated) == (1 if budget == 0 else 3)
            assert {r["repetition"] for r in repeated} == ({1} if budget == 0 else {1, 2, 3})
            assert len({r["worker_key"] for r in repeated}) == len(repeated)
            times = [r["latency_seconds"] for r in repeated]
            relative = (max(times) - min(times)) / statistics.median(times)
            ranges.append(relative)
            cpus = [r["cpu_prepare_host_seconds"] for r in repeated]
            remainders = [r["model_transfer_output_remainder_seconds"] for r in repeated]
            cpu_ranges.append((max(cpus) - min(cpus)) / statistics.median(cpus))
            remainder_ranges.append((max(remainders) - min(remainders)) / statistics.median(remainders))
            cpu_residuals.extend(c - statistics.mean(cpus) for c in cpus)
            total_residuals.extend(t - statistics.mean(times) for t in times)
            cpu_fractions.extend(c / t for c, t in zip(cpus, times))
            same_inputs.append(len({digest([r["input_tensor_hashes"], r["input_metadata_hash"]]) for r in repeated}) == 1)
            same_tokens.append(len({tuple(r["generated_token_ids"]) for r in repeated}) == 1)
            same_outputs.append(len({r["decoded_output"] for r in repeated}) == 1)
            same_answers.append(len({r["parsed_answer"] for r in repeated}) == 1)
            question_rows.append({"budget": budget, "question_id": question,
                "median_e2e_seconds": statistics.median(times), "relative_range": relative,
                "cpu_prepare_median_seconds": statistics.median(r["cpu_prepare_host_seconds"] for r in repeated),
                "remainder_median_seconds": statistics.median(r["model_transfer_output_remainder_seconds"] for r in repeated)})
        correct = sum(r["correct"] for r in rep1)
        latency = question_median(group, "latency_seconds")
        spread = statistics.median(ranges) if budget else None
        saved = next(c for c in summary["conditions"] if c["budget"] == budget)
        assert saved["accuracy"] == correct / 24
        assert abs(saved["latency_seconds"]["median"] - latency) < 1e-10
        if budget:
            assert abs(saved["median_relative_timing_range"] - spread) < 1e-10
            assert saved["prepared_input_agreement"] == all(same_inputs)
            assert saved["raw_token_agreement"] == all(same_tokens)
            assert saved["raw_output_agreement"] == all(same_outputs)
            assert saved["parsed_answer_agreement"] == all(same_answers)
        allocated = max(r["peak_allocated_bytes"] for r in group) / 1024**3
        reserved = max(r["peak_reserved_bytes"] for r in group) / 1024**3
        observed = [r["sampled_device_gpu_peak_bytes"] for r in group if r.get("sampled_device_gpu_peak_bytes") is not None]
        device = f"{max(observed) / 1024**3:.3f}" if observed else "unavailable"
        repeatability = "separate single pass" if not budget else f"{sum(same_tokens)}/24 exact token cells"
        timing = "n/a" if spread is None else f"{100 * spread:.2f}%"
        speedup = "n/a" if not budget else f"{reference_median / latency:.3f}x"
        nominal = "n/a" if not budget else f"{64 / budget:.3f}x"
        invalid = sum(not r["valid_answer"] for r in rep1)
        table.append(f"| {budget or 'Text only'} | {correct}/24 ({100*correct/24:.2f}%) | {invalid}/24 | {latency:.3f} | {timing} | {repeatability} | {nominal} | {speedup} |")
        cpu = question_median(group, "cpu_prepare_host_seconds")
        remainder = question_median(group, "model_transfer_output_remainder_seconds")
        costs.append(f"| {budget or 'Text only'} | {cpu:.3f} | {remainder:.3f} | {allocated:.3f} | {reserved:.3f} | {device} |")
        if budget:
            corr = statistics.correlation(cpu_residuals, total_residuals) if any(cpu_residuals) and any(total_residuals) else None
            timing_diagnosis.append({"budget": budget, "median_cpu_fraction_of_request": statistics.median(cpu_fractions),
                "median_cpu_repetition_relative_range": statistics.median(cpu_ranges),
                "median_remainder_repetition_relative_range": statistics.median(remainder_ranges),
                "within_question_centered_cpu_total_pearson_r": corr,
                "timing_gate_passed": spread <= .2,
                "live_memory_growth_flag_count": sum(bool(r.get("live_memory_growth_flag")) for r in group),
                "maximum_live_allocated_after_bytes": max(r["live_allocated_after_bytes"] for r in group)})
            corr_text = "unavailable" if corr is None else f"{corr:.3f}"
            timing_lines.append(f"| {budget} | {100*statistics.median(cpu_fractions):.1f}% | {100*statistics.median(cpu_ranges):.2f}% | {100*statistics.median(remainder_ranges):.2f}% | {corr_text} |")
    write_json(RUN / "timing_diagnosis.json", {"at": utc_now(), "conditions": timing_diagnosis,
        "interpretation": "Descriptive within-question centered correlation; CPU time is included in total by definition. No unique causal attribution or significance claim."})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(question_rows[0]))
    writer.writeheader()
    writer.writerows(question_rows)
    atomic_text(RUN / "private/question_repetition_medians.csv", buffer.getvalue())
    from benchmark_e000_cpu import inventory
    before = read_json(ROOT / "results/e000_20261028_completion_v1/private/historical_inventory_before.json")
    preserved = {}
    for name, expected in before.items():
        actual = inventory(ROOT / "results" / name)
        assert actual == expected
        preserved[name] = {"unchanged_files": len(actual), "inventory_sha256": digest(actual)}
    events = read_jsonl(RUN / "worker_calls.jsonl")
    failures = read_jsonl(RUN / "failed_attempts.jsonl")
    completed_calls = sum(e.get("event") == "completed" and e.get("model_invoked") is True for e in events)
    unknown_calls = sum(f.get("model_invocation_unknown", False) for f in failures)
    assert completed_calls == 441, "Unexpected calls require explicit accounting review"
    assert not subprocess.run(["git", "diff", "--cached", "--name-only"], capture_output=True, text=True, check=True).stdout.strip()
    ledger.check()
    assert ledger.value["video_transfer_bytes"] + ledger.value.get("interrupted_transfer_upper_bound_bytes", 0) <= 1024**3
    result = {"at": utc_now(), "passed": True, "primary_records_checked": 384,
        "parent_input_hash_and_metadata_matches": 384, "outputs_independently_redecoded": 384,
        "independent_accuracy_and_latency_summaries_match": True, "cross_worker_repetition_cells_checked": 120,
        "all_required_experimental_gates_passed": verification["all_operational_gates_passed"],
        "historical_preservation": preserved, "confirmed_new_model_calls": completed_calls,
        "unknown_new_model_calls_upper_bound": unknown_calls, "cumulative_model_calls_upper_bound": 1012 + completed_calls + unknown_calls,
        "effective_resource_limit_seconds": 20400, "git_index_empty": True}
    write_json(RUN / "independent_final_verification.json", result)
    status = read_json(RUN / "status.json")["status"]
    failed_timing_budgets = [d["budget"] for d in timing_diagnosis if not d["timing_gate_passed"]]
    feasibility_readout = (
        "All frozen operational gates passed, including exact repeated outputs and the proposed timing criterion."
        if verification["all_operational_gates_passed"] else
        f"The full collection completed, but feasibility is not confirmed. Budgets failing the timing criterion: {failed_timing_budgets}. Consult the gate table for all other outcomes."
    )
    workers = [read_json(p) for p in RUN.glob("worker_collect_*.json")]
    resident = statistics.median(w["baseline"]["allocated_bytes"] for w in workers) / 1024**3
    load_median = statistics.median(w["load_seconds"] for w in workers)
    profile_lines = []
    for probe in sorted(read_json(RUN / "probes.json")["probes"], key=lambda p: p["budget"]):
        c = probe["component_profile"]["components_seconds"]
        profile_lines.append(f"| {probe['budget']} | {c['cpu_prepare']:.3f} | {c['transfer_synchronized']:.4f} | {c['model_and_output_synchronized']:.3f} |")
    tables = "\n".join(table)
    cost_table = "\n".join(costs)
    profiles = "\n".join(profile_lines)
    timing_table = "\n".join(timing_lines)
    gate_text = "\n".join(f"| {name} | {value} |" for name, value in verification.items() if isinstance(value, bool))
    atomic_text(RUN / "completion_report.md", f"""# Experiment 0 — corrected frozen Video-QA collection v3

Final experimental verdict: **{status}**. All 360 video requests and 24 separate
text-only requests completed. Success requires every frozen operational gate,
including the 20% median relative timing-range criterion; completeness alone is
not feasibility confirmation. `independent_final_verification.json` separately
records saved-data integrity, which can pass even when an experimental gate fails.
{feasibility_readout}

## Accuracy, timing, reproducibility and measured speedup

| Frames | Repetition-1 accuracy | Invalid answers | Median E2E seconds | Median repetition range | Reproducibility | Nominal 64/B frame ratio | Measured E2E speedup vs 64 |
|---|---|---:|---:|---:|---|---:|---:|
{tables}

Accuracy uses the fixed first repetition and all 24 questions. Timing takes the
median of three repetitions per question, then the median over the 24 questions.
Range is `(maximum - minimum) / median` for each question's three repetitions,
then the median over those 24 ranges. Text-only has one separate pass. Per-question
medians and ranges are saved locally in `private/question_repetition_medians.csv`;
they are not independent additional examples. No favorable repetition was selected.
Frame ratios are arithmetic counts, not measured speedups. Uniform accuracy need
not increase monotonically with frames.

| Frames | Median CPU preparation seconds | Median transfer/model/output remainder seconds | Peak allocated GiB | Peak reserved GiB | Sampled device peak GiB |
|---|---:|---:|---:|---:|---:|
{cost_table}

CPU and remainder columns each apply the same per-question median procedure.
The remainder subtracts the nested host preparation span from synchronized full
request time; it includes transfer, allocator cleanup where configured, inference,
output transfer/decoding/parsing and synchronization. It is not pure GPU kernel
time. Separate medians need not add exactly to the reported median total. PyTorch
peaks include resident model weights and differ from total device usage. NVML
device peaks are sampled every 50 ms, not exact peaks or per-process allocations.
Unavailable WDDM per-process GPU values remain unavailable, not zero.
The {len(workers)} fresh collection workers had median resident allocated memory
{resident:.3f} GiB and median model-load time {load_median:.3f} seconds. Loading is
accounted separately and excluded from resident-model primary request latency.

Separate intrusive preflight profiles, on the prespecified first question only:

| Frames | CPU preparation seconds | Synchronized transfer seconds | Synchronized model/output seconds |
|---|---:|---:|---:|
{profiles}

These profiles are diagnostic calls, not the 360 primary timings. The request timer
starts before opening the compressed local file and includes full decode, selection,
RGB conversion/bicubic letterbox, prompt/processor work, transfers, generation,
output parsing and final CUDA synchronization. Hashing/scoring/serialization and
model loading are outside primary latency and inside cumulative accounting.

Primary timing diagnostics across the three repetitions:

| Frames | Median CPU/request fraction | Median CPU repetition range | Median remainder repetition range | Within-question centered CPU/total correlation |
|---|---:|---:|---:|---:|
{timing_table}

These are descriptive quantities. CPU time is part of total time by definition,
so their correlation is not independent evidence of a unique causal mechanism.
The remainder includes host scheduling, transfers, allocation, GPU work and output
processing. CPU, I/O, OS-cache, driver, thermal and scheduling effects are not
uniquely isolated by these measurements. No significance test is asserted.

## Gates and root causes

| Gate | Result |
|---|---|
{gate_text}

Transformers 4.57.1 checkpoint-default backfilling enabled sampling in the original
collection despite its requested `do_sample=False`. Explicit call-level
`use_model_defaults=False`, sampling disabled, one beam, eight maximum new tokens,
explicit token IDs and effective-mode validation correct that confirmed defect.
Every new request carries an effective-policy digest and a matching real worker
receipt. Repetitions execute in independent workers. No original stochastic
accuracy or latency measurements were reused as corrected-greedy measurements.

Four-thread full sequential PyAV decode preserves frozen source indices, timestamps,
uint8 pixels and processor tensors. Earlier CPU validation checked 16 raw pixel
pairs, 40 native input hashes and four direct tensor comparisons. The new independent
check matches all 384 primary input/metadata hashes to the original inputs and
redecodes all 384 generated-token outputs with the saved native tokenizer backend.
Historical decoding speedups of about 2.56x at eight frames and 2.12x at 64 frames
were CPU-only diagnostics; measured full-request speedups are in the table above.

Only 64-frame requests trim unused CUDA allocator cache after input transfer and
before generation, inside the original timer. Lower budgets retain native allocator
behavior because earlier trim-8 worsened variability. There is no frame, feature,
prefix or cross-request KV cache, offload, quantization or resolution reduction.
Reserved allocator memory is not live allocated memory and is not called a leak.
Live-growth flags and all sampled/absolute memory observations remain in raw records.

The initial 64-frame warmup stalled for more than seven minutes and was interrupted;
its completion remains unknown, with no fabricated output, latency or memory peak.
After the user authorized closing Hearthstone and NVIDIA Overlay, idle device memory
dropped from roughly 4.6 GiB to 0.65 GiB. Overlay automatically restarted with low
load. All five bounded probes then passed. This is evidence of a changed resource
environment, not proof that one specific app, fragmentation, Windows paging or
scheduling uniquely caused the stall. Failed attempts and the rejected resource
projection remain preserved. Primary collection has no replaced or synthetic trials.

## Resources and exact reproducibility

Original cumulative cap: four hours. Explicit grants raised it first to 5h20m and
then to **5h40m / 20,400 seconds**. The latter is a resource-only sidecar grant;
the already-frozen scientific configuration/source hashes were not rewritten.
`runtime_resource_grant.json` and the ledger identify the effective allowance.
The successful probes were revalidated, not rerun to obtain favorable measurements.
The 4,833.494-second remaining-work forecast retained a 1.5 factor, a 600-second
retry reserve and 180-second verification reserve. Collection began only after
the full forecast fit. One CUDA worker ran at a time; no new acquisitions, paid
services or cloud compute were used.

Authoritative post-closeout accounting is `final_resource_accounting.json` and
`resource_ledger.json`. Inherited time before v3 was 13,193.571727 seconds; every
subsequent setup, test, failure, worker, validation and reporting session is charged.
Closed idle gaps are excluded except explicitly conservative engineering allowances.
Storage includes the environment, weights, caches, temporary files, results and Git;
the unchanged limits are 20 GiB total additional storage and 1 GiB video acquisition.
The invocation upper bound is {result['cumulative_model_calls_upper_bound']}, including
historical calls and all explicitly unknown interruptions; {completed_calls} new calls
completed, including 384 primary requests, 32 collection warmups and 25 preflight calls.

Model/processor/tokenizer: `Qwen/Qwen3-VL-4B-Instruct`, revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`, BF16, SDPA, unchanged 192x320 canvas.
Dataset annotation revision: `ead1408f75b618502df9a1d8e0950166bf0a2a0b`.
Base Git revision: `52773383da61f6751ce2f216298f454097b45efa`; intentionally uncommitted
code is identified by `source_provenance.json`, `resolved_config.yaml`, model/software
manifests and private exact-source snapshots. All 1,072 historical evidence files
remain byte-for-byte unchanged. The original cohort, 24 questions, five budgets,
three repetitions and execution order are unchanged.

Executed commands and reusable verification:

```powershell
.venv\\Scripts\\python.exe scripts/collect_e000_authorized_v3.py
.venv\\Scripts\\python.exe -m pytest -q --basetemp .cache/authorized-v3-tests
.venv\\Scripts\\ruff.exe check src tests scripts
.venv\\Scripts\\python.exe scripts/finalize_e000_v3.py
```

The coordinator automatically verifies the complete saved collection. The finalizer
performs independent aggregation, token decoding, original-input comparisons,
historical preservation and resource checks under the same ledger. CLI verification
with the original frozen input configuration retains its original planning cap;
use the authorized coordinator/finalizer for the effective runtime resource grant.

## Scientific limitations and next research step

This remains an eight-video, 24-question clustered development pilot. Repetitions
improve timing observation and test repeatability; they do not create independent
videos or establish statistical significance. Text-only accuracy is a diagnostic
for possible language priors/dataset shortcuts, not proof of the precise shortcut
mechanism. Uniform-64 samples 64 frames, not every frame, and is neither an oracle
nor an accuracy upper bound. Full decoding can dominate frame-count savings.
Observed budget differences do not identify causal benefits of computation, and
feasibility does not establish adaptive sampling or controller performance.

If every gate passes, preregister Experiment 1 on separate evaluation videos to test
whether inexpensive observations predict the marginal value of additional visual
computation, using these corrected uniform baselines as controls. If timing fails,
resolve the operational instability with a separately versioned diagnostic before
training or claiming an adaptive controller. Do not tune on this exposed cohort.

Suggested commit: `Complete frozen greedy Video-QA collection and independent validation`.
All changes remain unstaged. `final_changes.json` lists exact public-safe paths.
Keep `.venv/`, `.cache/`, videos, weights, credentials, annotations/questions/labels,
private manifests, raw trials, failed attempts and all private records/logs excluded.
The human user reviews, commits and pushes.
""")
    atomic_text(RUN / "figures/README.md", "Complete measured collection: 360 video and 24 text-only requests. The plots show descriptive repetition-1 accuracy and per-question median resident-model latency/memory. Consult completion_report.md and verification.json for the experimental verdict and timing-stability limitations.\n")


def main():
    config = yaml.safe_load((ROOT / "configs/e000_threaded_authorized_v3.yaml").read_text(encoding="utf-8"))
    grant = read_json(RUN / "runtime_resource_grant.json")
    assert grant["authorized_limit_seconds"] == 20400 and grant["human_reply"] == "Authorize 5h40m cumulative time"
    require_cumulative_ledger(RUN, ROOT)
    ledger = Ledger(RUN / "resource_ledger.json", ROOT, {**config, "cumulative_execution_limit_hours": 20400 / 3600})
    assert all(s.get("finished_at") for s in ledger.value["sessions"]), "Wait for the collection coordinator to close"
    with ledger.track("independent-full-collection-scientific-closeout"):
        closeout(ledger)
        subprocess.run([str(ROOT / ".venv/Scripts/ruff.exe"), "check", "src", "tests", "scripts"], check=True)
        subprocess.run(["git", "diff", "--check"], check=True)
        receipts = []
        for path in (RUN / "private").glob("cpu_validation*.log"):
            match = re.search(r"(\d+) passed in ([0-9.]+)s", path.read_text(encoding="utf-8"))
            if match:
                receipts.append((path.stat().st_mtime, path, match))
        _, path, match = max(receipts, key=lambda r: r[0])
        write_json(RUN / "final_validation.json", {"at": utc_now(), "pytest_passed": int(match[1]),
            "pytest_reported_seconds": float(match[2]), "pytest_receipt_sha256": file_hash(path),
            "pytest_command": ".venv/Scripts/python.exe -m pytest -q --basetemp .cache/authorized-v3-tests",
            "ruff_check_passed": True, "git_diff_check_passed": True,
            "cpu_test_receipt_core_sources_unchanged": True,
            "real_processor_and_gpu_paths_exercised": True, "complete_primary_requests": 384})
        write_json(RUN / "modifications.json", {
            "src/video_compute_value/video.py": "Versioned exact four-thread full sequential decoding; frozen pixels/timestamps retained.",
            "src/video_compute_value/model.py": "Nested CPU-preparation timing, budget-specific in-timer allocator trim, per-request generation-policy receipts.",
            "src/video_compute_value/protocol.py": "Frozen execution-policy validation and bounded explicit initial time authorization.",
            "src/video_compute_value/runner.py": "Offline pinned-model reuse, allocator receipts, exact probe equality, full cost projection and cumulative reserve checks.",
            "src/video_compute_value/analysis.py": "Raw-token/output/input repeatability and effective-generation/allocator receipts required for success; independent saved-record checks.",
            "src/video_compute_value/resources.py": "Storage scans remain inside cumulative active accounting.",
            "src/video_compute_value/e000.py": "Reject reset/stale cumulative ledgers and return failure for unpassed experimental verification gates.",
            "src/video_compute_value/common.py": "Bounded retries for transient Windows atomic-replace permission failures; persistent failure preserves prior evidence.",
            "tests/test_completion.py": "Regression coverage for semantic equality, trim timing, generation/probe controls, authorization/accounting and immutable evidence.",
            "scripts/": "Bounded CPU audit, inherited-budget forecasts, authorized collection recovery and independent scientific closeout helpers.",
            "configs/ and protocols/": "Versioned threaded-decode/trim64 candidate and authorized input configuration, preserving historical configurations.",
            "README.md, AGENTS.md and .gitignore": "Current authorization/local Git restrictions, result guidance and private-log exclusion.",
            "results/": "Historical evidence preserved; new CPU audit and real corrected collection, safe aggregates and private ignored raw provenance."})
    value = ledger.value
    write_json(RUN / "final_resource_accounting.json", {"at": utc_now(), "inherited_seconds": 13193.571727438557,
        "cumulative_seconds": value["active_seconds"], "new_seconds": value["active_seconds"] - 13193.571727438557,
        "authorized_limit_seconds": 20400, "remaining_seconds": 20400 - value["active_seconds"],
        "storage_current_bytes": value["storage_current_bytes"], "storage_peak_bytes": value["storage_peak_bytes"],
        "storage_limit_bytes": 20 * 1024**3, "video_transfer_and_interruption_reserve_bytes": value["video_transfer_bytes"] + value.get("interrupted_transfer_upper_bound_bytes", 0),
        "video_limit_bytes": 1024**3, "all_sessions_closed": True})
    paths = subprocess.run(["git", "ls-files", "--modified", "--others", "--exclude-standard"], capture_output=True, text=True, check=True).stdout.splitlines()
    write_json(RUN / "final_changes.json", {"at": utc_now(), "all_changes_unstaged": True,
        "files": sorted(set(paths) | {"results/e000_20261028_greedy_collection_v3/final_changes.json"}), "no_git_mutations": True})
    print(read_json(RUN / "status.json"))
    print(read_json(RUN / "final_resource_accounting.json"))


if __name__ == "__main__":
    main()
