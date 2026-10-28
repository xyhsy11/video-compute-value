"""Read-only historical verification and affordability assessment for local completion."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import csv
import io
from pathlib import Path
import re
import statistics

import yaml

from video_compute_value.common import atomic_text, digest, file_hash, read_json, read_jsonl, utc_now, write_json
from video_compute_value.protocol import parse_answer
from benchmark_e000_cpu import accounted_call, inventory, remaining_seconds, ROOT, RUN, PARENT
from forecast_e000_completion import forecast_collection


def percentile(values, q):
    import numpy as np
    return float(np.percentile(values, q))


def assess():
    assert read_json(RUN / "cpu_verification.json")["passed"]
    for name, expected in read_json(RUN / "cpu_plan.json")["sources"].items():
        assert file_hash(RUN / "private/cpu_source_snapshot" / name) == expected
    for name in ("video.py", "model.py", "protocol.py"):
        rel = "src/video_compute_value/" + name
        assert file_hash(ROOT / rel) == read_json(RUN / "cpu_plan.json")["sources"][rel]
    cpu = read_json(RUN / "private/cpu_prepare_records.json")
    raw = read_json(RUN / "private/cpu_decode_records.json")
    cohort = read_json(PARENT / "cohort_manifest.json")
    parent = yaml.safe_load((PARENT / "resolved_config.yaml").read_text(encoding="utf-8"))
    before = read_json(RUN / "cpu_verification.json")
    assert digest(raw) == before["raw_evidence_sha256"] and digest(cpu) == before["prepare_evidence_sha256"]
    pairs = defaultdict(list)
    for row in raw:
        pairs[(row["source_id"], row["budget"])].append(row)
    assert len(pairs) == 16 and len(raw) == 32
    for group in pairs.values():
        assert {r["decoder_threads"] for r in group} == {1, 4}
        assert len({r["pixels_sha256"] for r in group}) == 1
        assert group[0]["source_indices"] == group[1]["source_indices"]
        assert group[0]["timestamps"] == group[1]["timestamps"]
    references = {(r["source_id"], r["question_id"], r["budget"]): r for r in
                  (read_json(p) for p in (PARENT / "records").glob("*.json"))}
    expected = {(v["source_id"], v["questions"][0]["question_id"], b)
                for v in cohort["videos"] for b in (8, 16, 32, 48, 64)}
    assert len(cpu) == 40 and {(r["source_id"], r["question_id"], r["budget"]) for r in cpu} == expected
    assert sum(r["direct_tensor_comparison"] for r in cpu) == 4
    for row in cpu:
        ref = references[(row["source_id"], row["question_id"], row["budget"])]
        assert row["input_tensor_hashes"] == ref["input_tensor_hashes"]
        assert row["source_indices"] == ref["source_indices"]
        assert row["model_timestamps"] == ref["model_timestamps"]
    # A separate diagnostic over CPU-prepared tensors; no primary model results are created.
    cpu_stats = []
    costs = {}
    first = cohort["videos"][0]["source_id"]
    for budget in (8, 16, 32, 48, 64):
        group = [r for r in cpu if r["budget"] == budget]
        assert len(group) == 8
        costs[budget] = (9 * sum(r["seconds"] for r in group) +
                        11 * next(r["seconds"] for r in group if r["source_id"] == first)) / 83
        cpu_stats.append({"budget": budget, "sources": 8, "median_prepare_seconds": statistics.median(r["seconds"] for r in group),
                          "maximum_prepare_seconds": max(r["seconds"] for r in group), "weighted_call_cpu_seconds": costs[budget]})
    timings = [read_json(p) for p in (ROOT / "results/e000_20261028_diagnostic_v2/private/records").glob("*.json")]
    low = [r["latency_seconds"] - r["cpu_prepare_host_seconds"] for r in timings if r["budget"] == 8 and r["arm"] == "native"]
    high = [r["latency_seconds"] - r["cpu_prepare_host_seconds"] for r in timings if r["budget"] == 64 and r["arm"] == "request_trim"]
    remainders = {b: statistics.median(low if b == 8 else high) for b in (8, 16, 32, 48, 64)}
    old = ROOT / "results/e000_20261028_diagnostic_v2"
    events = read_jsonl(old / "private/call_events.jsonl")
    startups = []
    for path in (old / "private").glob("*.log"):
        match = re.fullmatch(r"(?:worker|continuation)_([0-9]+)_([0-9]+)\.log", path.name)
        if not match:
            continue
        block, ns = map(int, match.groups())
        starts = [datetime.fromisoformat(e["at"]).timestamp() for e in events
                  if e["event"] == "started" and e["block"] == block and e["kind"] == "warmup" and e["sequence"] == 0]
        if starts:
            startups.append(min(starts) - ns / 1e9)
    assert len(startups) == 12 and min(startups) > 0
    cpu_probe = sum(r["seconds"] for r in raw if r["budget"] == 64 and r["decoder_threads"] == 4)
    inputs = {"cpu_seconds": costs, "remainder_seconds": remainders,
        "startup_seconds": statistics.median(startups), "text_seconds": max(1.0, percentile(low, 90)),
        "cpu_probe_seconds": cpu_probe, "remaining_seconds": remaining_seconds()}
    forecast = forecast_collection(**inputs)
    zero_cpu = forecast_collection(**{**inputs, "cpu_seconds": {b: 0.0 for b in costs}})
    forecast["inputs"] = inputs
    forecast["zero_cpu_scenario_seconds"] = zero_cpu["projected_seconds"]
    forecast["startup_observations_seconds"] = startups
    forecast["assumptions"] = [
        "CPU costs measured on all original sources, all budgets; first-question processor lengths are proxies for the other two questions",
        "Primary CPU weighting is 9 requests per source/budget; 11 extra per-budget calls use the fixed first source",
        "Intermediate 16/32/48-frame GPU-related remainders are unverified proxies from corrected-greedy trim-64 medians",
        "Text-only cost is an unverified proxy from native-8 remainder p90, lower-bounded by one second",
        "21 fresh-worker startups use the median of 12 observed corrected-diagnostic startups",
        "One 600-second worker contingency, 180-second verification and 90-second setup, then 1.5 safety factor",
        "Model hashing, environment/startup work, warmups, probes, profiles and final verification are included; these estimates are not primary measurements"]
    write_json(RUN / "projection.json", forecast)
    decode_stats = []
    by_pair = defaultdict(dict)
    for row in raw:
        by_pair[(row["source_id"], row["budget"])][row["decoder_threads"]] = row["seconds"]
    for budget in (8, 64):
        pairs = [g for (s, b), g in by_pair.items() if b == budget]
        decode_stats.append({"budget": budget, "pairs": len(pairs),
            "serial_median_seconds": statistics.median(g[1] for g in pairs),
            "threaded_median_seconds": statistics.median(g[4] for g in pairs),
            "median_paired_decode_speedup": statistics.median(g[1] / g[4] for g in pairs)})
    write_json(RUN / "cpu_summary.json", {"decode": decode_stats, "prepare": cpu_stats,
        "timing_stability_assessed": False, "cpu_only_diagnostic": True,
        "note": "Single paired decode observations and one candidate preparation per source/budget; no new end-to-end speedup or accuracy measurement"})
    from transformers import AutoTokenizer, GenerationConfig
    from video_compute_value.generation import resolve_greedy_policy
    from video_compute_value.resources import package_versions
    assert package_versions() == parent["environment_fingerprint"]["packages"]
    token = AutoTokenizer.from_pretrained(parent["model_path"], local_files_only=True, trust_remote_code=False)
    policy = resolve_greedy_policy(GenerationConfig(**parent["generation"]),
        GenerationConfig.from_pretrained(parent["model_path"], local_files_only=True))
    workers = [read_json(p) for p in (old / "private").glob("worker_*.json")]
    assert len(workers) == 12 and all(w["generation_policy"] == policy for w in workers)
    paths = list((PARENT / "records").glob("*.json"))
    paths += list((ROOT / "results/e000_20261027_diagnostic_v1/private/forensic").glob("*.json"))
    paths += list((ROOT / "results/e000_20261027_diagnostic_v1/greedy_validation/private/forensic").glob("*.json"))
    paths += list((ROOT / "results/e000_20261027_diagnostic_v1/private/timing_auxiliary").glob("*.json"))
    paths += list((old / "private/records").glob("*.json")) + list((old / "private/auxiliary").glob("*.json"))
    for path in paths:
        row = read_json(path)
        decoded = token.decode(row["generated_token_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        assert decoded == row["decoded_output"] and parse_answer(decoded) == row["parsed_answer"]
    assert len(paths) == 901
    observed = defaultdict(set)
    for row in timings:
        observed[(row["source_id"], row["question_id"], row["budget"])].add(tuple(row["generated_token_ids"]))
    assert all(len(v) == 1 for v in observed.values())
    manifest = read_json(PARENT / "model_manifest.json")
    for name, expected in manifest["files"].items():
        assert file_hash(Path(manifest["model_path"]) / name) == expected
    original = read_json(RUN / "private/historical_inventory_before.json")
    preservation = {}
    for name, expected in original.items():
        actual = inventory(ROOT / "results" / name)
        assert actual == expected, name
        preservation[name] = {"unchanged_files": len(actual), "inventory_sha256": digest(actual)}
    write_json(RUN / "historical_verification.json", {"passed": True, "verified_at": utc_now(),
        "preservation": preservation, "saved_outputs_redecoded": len(paths), "model_files_rehashed": len(manifest["files"]),
        "corrected_worker_policies_checked": len(workers), "observed_corrected_token_cells": len(observed),
        "all_observed_corrected_tokens_equal": True, "new_model_calls": 0, "new_gpu_execution": False,
        "new_generation_receipt_code_gpu_validated": False, "cpu_saved_records_independently_verified": True})
    per_question = []
    for key in sorted(observed):
        for arm in ("native", "request_trim"):
            group = [r for r in timings if (r["source_id"], r["question_id"], r["budget"]) == key and r["arm"] == arm]
            complete = len(group) == 3
            per_question.append({"source_id": key[0], "question_id": key[1], "budget": key[2], "arm": arm,
                "successful_repetitions": len(group), "complete": complete,
                "median_request_seconds": statistics.median(r["latency_seconds"] for r in group) if complete else None,
                "scope": "historical corrected diagnostic subset, not required full 24-question curve"})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(per_question[0]))
    writer.writeheader()
    writer.writerows(per_question)
    atomic_text(RUN / "historical_question_medians.csv", buffer.getvalue())
    return forecast


def report():
    projection = read_json(RUN / "projection.json")
    cpu = read_json(RUN / "cpu_summary.json")
    integrity = read_json(RUN / "historical_verification.json")
    cpu_check = read_json(RUN / "cpu_verification.json")
    remaining = remaining_seconds()
    consumed = 14400 - remaining
    needed = max(0, projection["projected_seconds"] - remaining)
    cap = 14400 + needed
    cap_5min = __import__("math").ceil(cap / 300) * 300
    blocked = not projection["fits_remaining_budget"]
    write_json(RUN / "status.json", {"status": "BLOCKED_RESOURCE" if blocked else "PREFLIGHT_REQUIRED",
        "feasibility_confirmed": False, "full_corrected_collection_started": False,
        "new_model_calls": 0, "remaining_active_seconds": remaining,
        "projected_full_completion_seconds": projection["projected_seconds"],
        "additional_active_seconds_required": needed, "proposed_cumulative_limit_seconds": cap_5min,
        "all_changes_must_remain_unstaged": True})
    table = ["| Condition | Required corrected requests | Completed new requests | Accuracy | E2E latency / variation | GPU memory | Measured speedup |",
             "|---|---:|---:|---|---|---|---|"]
    for budget in (8, 16, 32, 48, 64, 0):
        table.append(f"| {'Uniform-'+str(budget) if budget else 'text-only'} | {72 if budget else 24} | 0 | unavailable | unavailable | unavailable | unavailable |")
    decode_table = ["| Frames | Serial decode median (s) | Four-thread decode median (s) | Median paired decode speedup | RGB-identical pairs |", "|---|---:|---:|---:|---:|"]
    for row in cpu["decode"]:
        decode_table.append(f"| {row['budget']} | {row['serial_median_seconds']:.3f} | {row['threaded_median_seconds']:.3f} | {row['median_paired_decode_speedup']:.2f}x | {row['pairs']}/{row['pairs']} |")
    prep_table = ["| Frames | CPU prepare median (s) | CPU prepare maximum (s) | Sources checked |", "|---|---:|---:|---:|"]
    for row in cpu["prepare"]:
        prep_table.append(f"| {row['budget']} | {row['median_prepare_seconds']:.3f} | {row['maximum_prepare_seconds']:.3f} | 8 |")
    historical = read_json(ROOT / "results/e000_20261028_diagnostic_v2/verification.json")
    history_table = ["| Historical corrected subset policy | Frames | Requests | E2E median (s) | Median range | Peak allocated / reserved (GiB) | Device sampled peak (GiB) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in historical["timing"]:
        latency = 'withheld' if row['median_seconds'] is None else f"{row['median_seconds']:.3f}"
        variation = 'withheld' if row['median_relative_range'] is None else f"{row['median_relative_range']:.1%}"
        history_table.append(f"| {row['arm']} | {row['budget']} | {row['requests']}/24 | {latency} | {variation} | {row['peak_allocated_gib']:.3f} / {row['peak_reserved_gib']:.3f} | {row['sampled_device_peak_gib']:.3f} |")
    validation = read_json(RUN / "tests.json") if (RUN / "tests.json").exists() else {"status": "pending"}
    text = f'''# Experiment 0 completion assessment

Final verdict: **{'BLOCKED_RESOURCE' if blocked else 'PREFLIGHT_REQUIRED'}**. **FEASIBILITY_CONFIRMED is not claimed.**
No new GPU collection was launched. The full corrected study does not fit the
remaining cumulative allowance under the measured, guarded forecast. All work is
local and unstaged; no staging, commits, pushes, branch changes, PRs or remote edits.

## Scope and scientific verdict

The required study remains eight original Video-MME videos, three questions each,
Uniform-8/16/32/48/64, three repetitions (360 video requests), plus 24 text-only
requests. The original cohort and deterministic order are preserved. Qwen3-VL-4B
is unchanged, BF16/native SDPA, fixed canvas, prompt and parser; no offload or
quantization. The 384 original full-collection predictions used unintended
sampling and are not corrected-greedy accuracy or cost measurements.

{chr(10).join(table)}

Accuracy, full-curve latency, memory and measured speedup remain unavailable for
this new corrected collection. Missing values are not zero or estimates. Frame
ratios 8/64, 16/64, 32/64, 48/64, 64/64 are not measured speedups. No full five-budget
quality-cost curve or text-only greedy accuracy is fabricated from prior sampling.

## Confirmed causes and implementation fixes

Checkpoint-default inheritance had enabled sampling despite requested false.
The explicit call-level greedy correction remains intact. This audit checks its
resolved policy against 12 real historical worker receipts and redecodes saved
outputs. All 11 forensic target/control cells had stable greedy answers; all 16
observed corrected diagnostic input cells have equal generated tokens across
available workers. These are limited real-inference observations, not full-repeatability proof.

The verifier previously required parsed-answer repeatability but did not require
raw-token/output agreement or independently resolved effective-generation receipts
before feasibility. Those gates are now mandatory, and failed operational gates
produce a nonzero CLI verification status. New generated records include a policy
digest bound to the actual worker receipt. This new receipt integration has CPU
regression coverage; no new GPU execution is claimed for it.

Preparation was serial full decoding. Candidate execution v2 uses exactly four
FRAME codec threads, preserves full sequential decoding, and declares Uniform-64-only
cache trimming inside full request timing. Other budgets do not trim. This explicitly
versions the original protocol's no-empty-cache rule. The candidate is CPU-validated,
not yet accepted by new full GPU timing/fit gates. No decoded-frame, feature,
prefix or KV reuse is introduced. Nested preparation clocks add no GPU fences.

Final storage scans previously fell outside session duration; they are now timed.
Atomic persistence now retries transient PermissionError reader locks with a bounded
backoff and still propagates persistent errors. No historical resource charge was removed. The CLI rejects missing, stale or
concurrently active sibling ledgers, preventing a fresh directory from renewing
the cumulative allowance.

## Real CPU optimization and exact equivalence

{chr(10).join(decode_table)}

{chr(10).join(prep_table)}

All {cpu_check['raw_rgb_equal_pairs']} raw RGB pairs are exactly equal; all
{cpu_check['exact_parent_input_hash_matches']} candidate source/budget processor
inputs match the original saved tensor hashes; {cpu_check['direct_equal_tensor_pairs']}
fresh serial/threaded tensor pairs also pass torch.equal. Selected source indices,
PTS, model timestamps and dimensions are preserved. All eight videos and five
budgets are covered. No model object was instantiated; CUDA remained uninitialized. Model files were
read separately for checksum verification.
CPU diagnostic duration: {cpu_check['elapsed_seconds']:.3f} seconds.

These are screening observations with one paired raw observation per source/budget
and one candidate preparation per source/budget. Their speedups concern decoding
only. CPU timing stability and end-to-end speedup have not been established by
these measurements. Model computation was not included in this CPU diagnostic.

## Historical corrected timing and memory evidence

{chr(10).join(history_table)}

These are the already preserved first-question-per-video diagnostics, not new
24-question results. Native-64 has one timed-out request and three unattempted
identities; complete timing/range summaries remain withheld. Its attempt was
outstanding for 482.902 seconds when the 600-second worker deadline ended it.
Almost all device memory was in use. This supports memory pressure as a candidate,
not a unique proof of Windows paging or a demonstrated live-memory leak.

Trim-64 completed 24/24 with lower reservations, while trim-8 worsened range from
15.8% to 26.9%. Thus cleanup is not assumed universally useful. Preparation accounted
for about 76–90% of completed request time; its centered correlation with total
latency was 0.966–0.999. Driver/transfer/model/output work is only a remainder, not
pure GPU time. Thermal, scheduling, I/O and OS-cache causes are not uniquely isolated.
Per-question medians reproduced from saved records are in `historical_question_medians.csv`.

## Resource reconciliation and blocked operations

Inherited consumption: 11,605.416 seconds. Limits remain 14,400 seconds, 20 GiB
project storage and 1 GiB video acquisition. New work uses the inherited ledger,
including the entire active audit/engineering window and failures. At this report:
{consumed:.3f} cumulative active seconds; {remaining:.3f} seconds remaining.
No new model calls, no acquisition, no paid services or cloud compute.

Measured guarded forecast: {projection['projected_seconds']:.3f} seconds
({projection['projected_seconds']/60:.1f} minutes) for complete preflight/collection/
verification. It includes 441 required model invocations before retries, 21 fresh
model workers, CPU probes, setup, verification, a 600-second contingency and 1.5x
safety factor. Even the hypothetical zero-CPU scenario projects
{projection['zero_cpu_scenario_seconds']:.3f} seconds with those other costs.
The intermediate GPU and text-only costs are explicitly unverified planning proxies,
not recorded measurements. All assumptions and numerical components are in `projection.json`.

The projection exceeds remaining time by {needed:.3f} seconds ({needed/60:.1f} minutes).
The minimum rounded cumulative cap for this planning scenario is
{cap_5min/3600:.3f} hours ({int(cap_5min)//3600}h{int(cap_5min)%3600//60:02d}m).
This is a conditional planning authorization, not a guarantee every experimental
gate will pass. No hard limit has been increased. Original storage was 15.8602 GiB;
new outputs are small and share the existing weights/videos. The final ledger
records the latest observed project storage after closeout. No additional video
acquisition is needed. A full corrected run remains blocked pending explicit time authorization.

## Validation, preservation and reproducibility

CPU test/static-check result: `{validation}`. Independent read-only verification
redecoded {integrity['saved_outputs_redecoded']} saved outputs and rehashed
{integrity['model_files_rehashed']} model files. All three historical run inventories
remain unchanged; see `historical_verification.json`. No historical report, raw
trial, configuration, failure or ledger was overwritten. The original generation
correction and all saved unsuccessful attempts remain available.

Base Git revision: `52773383da61f6751ce2f216298f454097b45efa`. Changes are intentionally
uncommitted. Exact executed CPU sources/configuration are hashed in `cpu_plan.json`;
final local source fingerprints and changed-file inventory are in `changes.json`.

Executed commands (the CPU benchmark now refuses an existing CPU plan; standalone
helpers account execution in the inherited ledger and require any active accounting
session to close first):

```powershell
.venv\\Scripts\\python.exe -m pytest -q tests/test_completion.py --basetemp .cache/completion-targeted-tests
.venv\\Scripts\\python.exe scripts/benchmark_e000_cpu.py
.venv\\Scripts\\python.exe -m pytest -q --basetemp .cache/completion-closeout-tests
.venv\\Scripts\\ruff.exe check src tests scripts
.venv\\Scripts\\python.exe scripts/report_e000_completion.py
```

After additional time is explicitly authorized, copy the latest cumulative ledger
into a fresh uniquely named collection, preserve the original cohort/order, apply
`configs/e000_threaded_v2.yaml`, re-freeze all fingerprints, then pass all real GPU
fit/integrity probes before collection. Do not resume historical source fingerprints
with modified code. Do not treat a new directory as a new resource allowance.

## Limits and next research step

Eight videos/24 clustered questions are a development pilot, not 24 independent
videos. Repetitions add timing observations, not independent examples. Text-only
shortcuts remain unquantified under the intended full greedy protocol. Accuracy
need not increase monotonically with frames; Uniform-64 is only a sampled reference,
not an oracle or upper bound. Full sequential decoding can dominate frame-budget
costs. This study establishes neither adaptive sampling nor causal benefits of
extra visual computation; no significance claim is supported.

The evidence-backed next step before Experiment 1 is to finish the corrected full
baseline and its operational gates under an authorized budget. Only then preregister
an Experiment 1 that tests whether inexpensive observations predict the marginal
value of added visual computation on separate evaluation data. Do not train or
select a controller against this exposed eight-video pilot.

Suggested commit message: `Validate versioned CPU preparation and enforce complete Experiment 0 gates`.

Keep `.venv/`, `.cache/`, data/videos/weights, credentials, all `private/` records,
cohort/question/label manifests, raw trial outputs and temporary fixtures excluded.
The user personally reviews, stages, commits and pushes; this task leaves changes unstaged.
'''
    atomic_text(RUN / "report.md", text)
    write_json(RUN / "authorization_request.json", {"reason": "Complete corrected study exceeds remaining four-hour cumulative limit",
        "minimum_additional_seconds_for_current_forecast": needed, "requested_cumulative_limit_seconds": cap_5min,
        "storage_limit_gib": 20, "video_limit_gib": 1, "authorization_received": False})


if __name__ == "__main__":
    accounted_call(assess, "standalone-readonly-completion-assessment")
    report()
