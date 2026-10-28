> Historical snapshot before GPU isolation and completed collection. The [completion report](completion_report.md) contains the current MEASUREMENT_UNSTABLE verdict and all 384 measurements.

# Experiment 0 authorized preflight — v3

Verdict: **BLOCKED_ENVIRONMENT**, not FEASIBILITY_CONFIRMED. The user authorized
19,200 cumulative active seconds (5h20m); all other restrictions remain in force.
No corrected full collection has started. All 360 video and 24 text-only primary
requests remain unattempted. Existing sampled collection and both diagnoses are
preserved byte for byte. Their accuracy curves are not relabeled greedy results.

## What executed and failed

75 CPU tests passed in 22.21 seconds; Ruff passed. The model loaded locally in
BF16 with SDPA, native CUDA allocator, no quantization or offload. The worker
resolved and recorded effective `greedy_search`, explicit token IDs, disabled
checkpoint defaults and sampling, one beam and eight maximum new tokens.
This receipt validates configuration resolution, not a completed inference.

The first Uniform-64 warmup began at 2026-10-28T06:58:08.610934Z and produced no
terminal output for more than seven minutes. Only the experiment worker was
terminated to conserve the hard budget; it exited and released its context.
The outstanding request and failure are preserved locally, with unknown invocation
completion, no imputed answer, latency, tokens or memory peak. One additional model
invocation is conservatively reserved; zero new model calls completed.

Observed during the stall: device usage about 15,577 MiB and 100% utilization;
worker WDDM dedicated memory 11,269,578,752 bytes and shared memory 213,909,504 bytes.
Hearthstone and NVIDIA Overlay had GPU contexts. A sampled Hearthstone allocation
was only 399,441,920 bytes and its engine usage 4%; that does not establish it as
the cause. One overlay counter exceeded physical VRAM and is rejected as an
allocation estimate. Device usage remained about 4.6 GiB after the worker exited.
Memory pressure, contention and driver behavior remain candidates; no unique
claim about paging, fragmentation, a leak, or Windows scheduling is justified.

GPU isolation is pending user action/authorization. Unrelated apps were not closed.
The recovery orchestrator preserves provenance and all prior attempts, bounds each
recovery probe worker at 120 seconds, and admits collection only after all five
actual fit/input/output probes and the measured 1.5-margin forecast pass. A probe
deadline bounds resource exposure and does not replace a measurement.

## Corrected full-study results

| Frames | Completed/planned | Accuracy | Median E2E | Timing range | CPU preparation | GPU allocation/reservation | Speedup vs 64 |
|---|---:|---|---|---|---|---|---|
| Text only | 0/24 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 8 | 0/72 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 16 | 0/72 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 32 | 0/72 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 48 | 0/72 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 64 | 0/72 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

Per-question repetition medians, model/output timing, observed device peaks,
cross-worker token equality and the 20% timing criterion are unavailable for this
run. Zero completed records are not evidence of repeatability or fit. Independent
verification reproduces the empty summary and reconciles the interrupted attempt;
the completeness, effective-inference, fit, reproducibility, memory and timing
gates fail/remain unverified. See `verification.json` and `summary.json`.

## Confirmed fixes and evidence

Transformers checkpoint default backfilling caused stochastic generation in the
original collection. Explicit call overrides and effective-mode validation fix
that defect in the targeted earlier greedy diagnoses; the full corrected curve is
still missing. Earlier corrected targeted outputs were stable; the eight historical
disagreements are not unresolved stochastic evidence under the new policy.

The CPU audit in `../e000_20261028_completion_v1/` verified 16 exact serial/threaded
RGB comparisons, 40 native processor input hashes and four direct tensor equality
comparisons across the original sources. Four-thread sequential decoding measured
about 2.56x at eight frames and 2.12x at 64 frames. These are decoding speedups,
not measured end-to-end inference speedups. No source frames, timestamps, resizing,
prompt, cohort/order or parser changed. All preparation and 64-only allocator trim
remain within the original request timer; no cross-request cache was introduced.
Earlier trim-64 completed 24/24 targeted requests, while trim-8 worsened timing
range. This preflight shows that the earlier targeted success does not guarantee
64-frame reliability in the current environment.

## Resources, provenance and reproducibility

Inherited ledger: `results/e000_20261028_completion_v1/resource_ledger.json`,
13,193.571727 cumulative seconds. Authoritative current accounting is
`resource_ledger.json`; it includes the failed warmup, setup, CPU investigation,
tests, checks, report generation and storage scans. Its value at report generation
was 14403.612 seconds plus any active reporting session.
Storage observed then: 15.863113 GiB / 20 GiB.
No new downloads or videos. Prior transfer plus interrupted-range reserve remains
82,702,430 bytes / 1 GiB. The original guarded forecast of 5,968.635 seconds was
conditional on real GPU preflight, not a promise of success; elapsed failed work
reduces remaining allowance. Any next full-run estimate must use the latest ledger
and actual successful probes. No further resource increase is authorized.

Base revision: `52773383da61f6751ce2f216298f454097b45efa`, intentionally uncommitted.
Exact frozen configuration, software/model manifests, source digests and ordering
are saved here. `preservation_verification.json` independently checks historical
inventories; `changes.json` lists the unstaged public-safe files. Private worker
logs, call events, interrupted attempt and frozen source copies remain ignored.

```powershell
.venv\Scripts\python.exe scripts/collect_e000_authorized_v3.py
.venv\Scripts\python.exe -m pytest -q --basetemp .cache/authorized-v3-tests
.venv\Scripts\ruff.exe check src tests scripts
```

Resume the orchestrator only after GPU isolation; it inherits the same ledger and
unchanged fingerprints. Reporting helper `scripts/report_e000_authorized_v3.py`
must run under `Ledger.track`, as in the recorded closeout, so reporting is charged.

## Scientific conclusion and next step

Current hardware has completed targeted 64-frame requests in an earlier controlled
arm, but reliable full-cohort operation and timing stability are not established.
No full corrected accuracy-cost comparison or speedup can be reported. Eight videos
and 24 clustered questions limit generalization; repetitions are not independent
examples. Text-only shortcuts remain unquantified under the full greedy protocol.
Accuracy need not be monotonic with frames, Uniform-64 is not an oracle, and
decoding overhead can dominate cost. This feasibility study proves no adaptive
sampling, causal computation benefit or statistical significance.

Finish the corrected baseline and all gates before Experiment 1. Then preregister
a separate-data test of whether inexpensive observations predict the marginal value
of additional visual computation; do not train a controller on this exposed pilot.

Suggested commit: `Harden frozen Video-QA validation and document bounded greedy preflight`.
All changes remain unstaged. Keep environments, caches, videos, weights, credentials,
annotation/question/label manifests, raw trials, failed attempts and all private
records/logs excluded. The human user reviews, commits and pushes.
