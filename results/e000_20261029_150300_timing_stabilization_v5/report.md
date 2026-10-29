# Controlled E0 timing follow-up v5

Diagnostic status: **PARTIAL_PREFLIGHT_NATIVE_CRASH**. Selected policy: **None**.
Full confirmation: **CONFIRMATION_NOT_RUN**. Reason: Neither policy qualified or diagnostic incomplete.


The first preflight worker exited during checkpoint loading with Windows native
exit code `0xC0000005`, before a model-call start event or a
successful load receipt. One worker was attempted and reaped; the second worker,
all 12 preflight calls, the 120 main diagnostic calls and full confirmation were
not run. The contemporaneous Windows Application Error event names
`torch_cpu.dll` with the same exception code. This
does not establish the underlying cause. No package, driver, model, power setting,
application or measurement policy was changed afterward. No automatic retry ran.

The resource forecast admitted preflight, so this is a native initialization
failure, not a negative timing result or a resource-denied phase. A/B S values at
8/64 are **missing**, not zero or passing. Both policies remain unevaluated.
[closeout_verification.json](closeout_verification.json) separately passes the
available-evidence/cleanup/accounting checks; [verification.json](verification.json)
correctly fails full experimental completeness.
Its combined "unreaped/incomplete worker" flag refers here to the incomplete
session: the failure receipt and a free CUDA-worker lock independently confirm
that the owned process was reaped.

The next single action is a separately named, bounded model-initialization crash
diagnostic using the preserved native event/module and existing assets before
attempting another prospective timing study. This task authorizes no automatic
replacement session. An additional cumulative allowance cannot be estimated from
these absent GPU probes; the native failure must be resolved and fresh complete
phase forecasts admitted first.

Historical E0 remains **MEASUREMENT_UNSTABLE**. Historical E0/E1 evidence was
preserved; 2098 inventory files matched exactly.

The frozen study asks whether a worker lifecycle makes complete local-video requests
repeatable across independent worker sessions. The cohort remains eight exposed
Video-MME short sources; this screen uses the first official question of each.
The model/processor/tokenizer is Qwen3-VL-4B-Instruct at
`ebb281ec70b05090aa6165b016eac8ec08e71b17`, BF16, SDPA, 192 × 320 inputs.
No controller, practical adaptive policy, efficiency claim or Experiment 2.

Planned policy A uses six fresh main workers, one per budget/repetition. Policy B uses
three independent resident main workers, one per repetition, retaining the model
across two budget blocks. Every block has two identical historical warmups.
Rounds 1/2 use A→B with 8→64 and B→A with 64→8; round 3 and source orders are
drawn prospectively from seed 20261029. Actual anonymous arrays are in
[schedule.json](schedule.json); [freeze.json](freeze.json) seals exact sources and design.

The unchanged synchronized timer starts before video opening and ends after full
decoding, processing, transfer, greedy generation, output decoding/parsing and final
CUDA synchronization. Each request reopens and sequentially decodes the complete
video. Ordinary OS caching is allowed. There is no cross-request evidence/KV cache.
Only 64 frames trims allocator cache inside the timer; 50 ms memory sampling is
unchanged in both policies. Model loading, warmups, hashes, scoring, serialization
and telemetry are outside primary measurements and inside project accounting.

| Policy | Frames | Primaries | Median R | Median request s | Median absolute range s | Preparation s | Remainder s | Individual R>50% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 8 | 0 | missing | missing | missing | missing | missing | missing |
| A | 64 | 0 | missing | missing | missing | missing | missing | missing |
| B | 8 | 0 | missing | missing | missing | missing | missing | missing |
| B | 64 | 0 | missing | missing | missing | missing | missing | missing |

For each question, R=(max−min)/median over three independent sessions. S is the
median of eight R's. Request/preparation/remainder and absolute-range summaries
also take question medians first. Separate stage medians need not sum to median
total. The remainder includes transfers, allocator work, model execution and output
handling. All individual R's are retained in [summary.json](summary.json), with
anonymous exports in [question_ranges.csv](question_ranges.csv) when available.
No observation is excluded, trimmed, replaced or treated as another source video.

The available CPU-only preparation measurements are a single pass over eight
source questions, without a resident GPU model. Extra serial/four-thread validation
decodes and hashing are outside these host spans and inside project accounting.
These are preparation costs, not full-request measurements or repeatability tests.
Source: [cpu_preparation_summary.csv](cpu_preparation_summary.csv).

| Frames | Native preparations | Median preparation s | Minimum–maximum s |
|---:|---:|---:|---:|
| 8 | 8 | 2.032 | 1.178–3.969 |
| 64 | 8 | 3.183 | 2.161–4.849 |

Qualification requires S≤20% at both budgets and every integrity/completeness gate.
Selection is A first, then B, otherwise stop. This is only confirmation eligibility.
The screen cannot establish a historical causal mechanism or validated acceleration.
Full-confirmation forecasting is conditional on a qualifying policy; no such
forecast was made after this preflight failure.

![Fixed timing criterion](figures/diagnostic_stability.png)

![Absolute request time](figures/request_times.png)

Absolute allocated/reserved peaks and device-wide 50 ms sampled maxima are saved
in the aggregate JSON/CSV. Peaks include resident weights; reservation is not live
allocation. Post-release live allocations are checked against the documented shape
warmup baseline with 64 MiB tolerance. Unavailable device/process observations are
kept unavailable. Detailed transition observations and power/load/clock/temperature
telemetry are retained privately. [environment_summary.json](environment_summary.json)
gives the tested software and device. We do not infer thermal throttling, one
Windows scheduling defect, or a unique decoder/allocator cause from correlation.

| Policy | Frames | Peak allocated GiB | Peak reserved GiB | Sampled device GiB | Maximum post-release live GiB | Maximum post-release reserved GiB |
|---|---:|---:|---:|---:|---:|---:|
| A | 8 | unavailable | unavailable | unavailable | unavailable | unavailable |
| A | 64 | unavailable | unavailable | unavailable | unavailable | unavailable |
| B | 8 | unavailable | unavailable | unavailable | unavailable | unavailable |
| B | 64 | unavailable | unavailable | unavailable | unavailable | unavailable |

Independent verification errors: `["actual arm/worker order", "actual contiguous block/call order", "diagnostic invocation total", "failed/interrupted attempt", "incomplete worker session", "phase schedule/call completeness", "preflight incomplete", "unreaped/incomplete worker"]`. Failed
or incomplete measurement sessions remain in the private event/failure logs;
no failed session is automatically restarted or replaced.

[verification.json](verification.json) records the full verifier's completeness
failure. The verifier implements native-token decoding, answer reparsing,
historical input/output comparison, call/worker reconstruction, memory/resource/
preservation checks, and independent S/selection recomputation. This run has zero
new generated-token sequences or primary timings for those generation checks.
CPU tests validate implementation;
real-GPU evidence is limited to the phases actually completed.

Calls by phase/kind: `{"confirmation": {"primary": 0, "warmup": 0}, "diagnostic": {"primary": 0, "warmup": 0}, "preflight": {"profile": 0, "warmup": 0}}`.
Completed new calls: 0; unknown upper bound:
0. Planned diagnostic calls: 132
(8 preflight warmups, 4 profiles, 24 main warmups, 96 primary requests).
Cumulative consumed: 25562.826 / 27,600 s;
remaining: 2037.174 s. Storage:
17047987942 / 21474836480 bytes.
Acquired-video transfers: 81653854 bytes; no new acquisition.
All historical charges and failed attempts are inherited. Detailed local accounting
is `private/resource_ledger.json`; [resource_accounting.json](resource_accounting.json)
is its safe checkpoint. A conservative publication reserve is charged on closeout.

Reproducible commands (collection commands require matching local restricted assets,
an open resource allowance and an untouched frozen implementation):

```powershell
$run = "results/e000_20261029_150300_timing_stabilization_v5"
.venv\Scripts\python.exe scripts/stabilize_e000_timing_v5.py prepare --run-dir $run
.venv\Scripts\python.exe scripts/stabilize_e000_timing_v5.py preflight --run-dir $run
.venv\Scripts\python.exe scripts/stabilize_e000_timing_v5.py collect --run-dir $run
.venv\Scripts\python.exe scripts/stabilize_e000_timing_v5.py analyze --run-dir $run
.venv\Scripts\python.exe scripts/stabilize_e000_timing_v5.py confirmation --run-dir $run
.venv\Scripts\python.exe scripts/stabilize_e000_timing_v5.py verify --run-dir $run
.venv\Scripts\python.exe scripts/verify_e000_timing_closeout_v5.py --run-dir $run
.venv\Scripts\python.exe scripts/stabilize_e000_timing_v5.py report --run-dir $run
.venv\Scripts\python.exe scripts/stabilize_e000_timing_v5.py close --run-dir $run
```

`prepare` creates a fresh freeze and is intentionally rejected on this existing run.
`collect` skips only completed whole workers with valid immutable identities;
interrupted resident sessions cannot resume. `confirmation` allows one attempt.
Use `verify` for read-only reproduction; it does not rerun inference or rewrite
historical artifacts. See the [prospective protocol](../../protocols/e000_timing_stabilization_v5.md).
For this incomplete native-failure outcome, the full verifier intentionally returns
exit code 2; the available-evidence closeout verifier returns 0 on preserved valid
evidence. Measurement paths at every budget remain unverified in this new run.
