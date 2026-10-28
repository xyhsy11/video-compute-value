# Experiment 0 completion assessment

Final verdict: **BLOCKED_RESOURCE**. **FEASIBILITY_CONFIRMED is not claimed.**
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

| Condition | Required corrected requests | Completed new requests | Accuracy | E2E latency / variation | GPU memory | Measured speedup |
|---|---:|---:|---|---|---|---|
| Uniform-8 | 72 | 0 | unavailable | unavailable | unavailable | unavailable |
| Uniform-16 | 72 | 0 | unavailable | unavailable | unavailable | unavailable |
| Uniform-32 | 72 | 0 | unavailable | unavailable | unavailable | unavailable |
| Uniform-48 | 72 | 0 | unavailable | unavailable | unavailable | unavailable |
| Uniform-64 | 72 | 0 | unavailable | unavailable | unavailable | unavailable |
| text-only | 24 | 0 | unavailable | unavailable | unavailable | unavailable |

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

| Frames | Serial decode median (s) | Four-thread decode median (s) | Median paired decode speedup | RGB-identical pairs |
|---|---:|---:|---:|---:|
| 8 | 6.901 | 2.688 | 2.56x | 8/8 |
| 64 | 7.547 | 3.640 | 2.12x | 8/8 |

| Frames | CPU prepare median (s) | CPU prepare maximum (s) | Sources checked |
|---|---:|---:|---:|
| 8 | 2.363 | 4.171 | 8 |
| 16 | 2.618 | 4.552 | 8 |
| 32 | 2.734 | 5.472 | 8 |
| 48 | 2.699 | 4.869 | 8 |
| 64 | 2.945 | 4.746 | 8 |

All 16 raw RGB pairs are exactly equal; all
40 candidate source/budget processor
inputs match the original saved tensor hashes; 4
fresh serial/threaded tensor pairs also pass torch.equal. Selected source indices,
PTS, model timestamps and dimensions are preserved. All eight videos and five
budgets are covered. No model object was instantiated; CUDA remained uninitialized. Model files were
read separately for checksum verification.
CPU diagnostic duration: 351.313 seconds.

These are screening observations with one paired raw observation per source/budget
and one candidate preparation per source/budget. Their speedups concern decoding
only. CPU timing stability and end-to-end speedup have not been established by
these measurements. Model computation was not included in this CPU diagnostic.

## Historical corrected timing and memory evidence

| Historical corrected subset policy | Frames | Requests | E2E median (s) | Median range | Peak allocated / reserved (GiB) | Device sampled peak (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| native | 8 | 24/24 | 9.844 | 15.8% | 8.408 / 8.445 | 10.578 |
| native | 64 | 20/24 | withheld | withheld | 10.317 / 14.146 | 15.879 |
| request_trim | 8 | 24/24 | 9.994 | 26.9% | 8.408 / 8.453 | 10.334 |
| request_trim | 64 | 24/24 | 11.764 | 14.4% | 10.317 / 10.477 | 12.418 |

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
13193.572 cumulative active seconds; 1206.428 seconds remaining.
No new model calls, no acquisition, no paid services or cloud compute.

Measured guarded forecast: 5968.635 seconds
(99.5 minutes) for complete preflight/collection/
verification. It includes 441 required model invocations before retries, 21 fresh
model workers, CPU probes, setup, verification, a 600-second contingency and 1.5x
safety factor. Even the hypothetical zero-CPU scenario projects
4116.174 seconds with those other costs.
The intermediate GPU and text-only costs are explicitly unverified planning proxies,
not recorded measurements. All assumptions and numerical components are in `projection.json`.

The projection exceeds remaining time by 4762.207 seconds (79.4 minutes).
The minimum rounded cumulative cap for this planning scenario is
5.333 hours (5h20m).
This is a conditional planning authorization, not a guarantee every experimental
gate will pass. No hard limit has been increased. Original storage was 15.8602 GiB;
new outputs are small and share the existing weights/videos. The final ledger
records the latest observed project storage after closeout. No additional video
acquisition is needed. A full corrected run remains blocked pending explicit time authorization.

## Validation, preservation and reproducibility

CPU test/static-check result: `{'pytest': {'command': '.venv/Scripts/python.exe -m pytest -q --basetemp .cache/completion-closeout-tests', 'passed': 70, 'failed': 0, 'reported_seconds': 28.8}, 'ruff': {'command': '.venv/Scripts/ruff.exe check src tests scripts', 'passed': True}, 'new_real_model_calls': 0, 'cpu_only_unit_suite': True, 'standalone_accounting_and_immutable_evidence_guards_tested': True}`. Independent read-only verification
redecoded 901 saved outputs and rehashed
12 model files. All three historical run inventories
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
.venv\Scripts\python.exe -m pytest -q tests/test_completion.py --basetemp .cache/completion-targeted-tests
.venv\Scripts\python.exe scripts/benchmark_e000_cpu.py
.venv\Scripts\python.exe -m pytest -q --basetemp .cache/completion-closeout-tests
.venv\Scripts\ruff.exe check src tests scripts
.venv\Scripts\python.exe scripts/report_e000_completion.py
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
