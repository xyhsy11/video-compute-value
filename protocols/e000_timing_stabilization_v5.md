# E0 timing stabilization v5

This prospective operational screen preserves historical E0's
MEASUREMENT_UNSTABLE verdict and all E0/E1 files. It uses the same exposed eight
Video-MME short sources, original questions, frozen Qwen3-VL-4B BF16/SDPA revision,
native processor, explicit greedy policy, four CPU threads, FRAME/four decoder
threads and 192 × 320 canvas. It does not train a controller or test efficiency.

Policy A loads a fresh worker for each budget/repetition. Policy B loads one
worker per repetition and retains it across that repetition's two budget blocks.
There are three paired independent rounds: A/B with 8/64, B/A with 64/8, then
independent arm and budget draws from Python Random(20261029). Each round draws
one source permutation shared by both arms and budgets. The actual arrays in the
freeze are authoritative. The first official question of each source gives 96
primary requests in nine main workers, with two fixed historical warmups before
each of 12 blocks (24 warmups). No process persists across repetitions.

Before GPU preflight, prepare all eight sources at 8 and 64 frames (16 native
preparations), matching historical tensor/temporal metadata. Direct serial versus
FRAME/four uint8 pixel comparisons are separate validation decodes. These extra
decodes are charged, do not invoke the model, and are excluded from preparation
cost estimates. Two new probe workers run 8→64 and 64→8, each block containing two
matched historical warmups and one intrusive synchronized profile: 12 calls.
Profiles never enter primary data. With no failure there are 132 diagnostic calls.

The existing timed_request is unchanged. Its primary boundary starts before
opening the local compressed video and ends after generation, output transfer,
decoding/parsing and final synchronization, with synchronization before start.
Every request fully reopens/decodes the source. No cross-request frames, tensors,
features, prefix or KV state is reused. Ordinary OS caching is allowed; this is
not cold-disk latency. MemorySampler remains at 50 ms in both arms. Uniform-64
alone trims CUDA cache after transfer, inside generation/the timer. Model loading,
hashing, scoring, serialization and identical extra telemetry remain outside
primary latency and inside cumulative accounting. The request remainder includes
transfer, allocator work, model execution and output handling, not pure GPU time.

Post-request cleanup follows the existing boundary: release CPU inputs, collect
Python garbage, then observe live CUDA allocations. Every warmup must stay within
64 MiB of the resident model baseline. After the two warmups, each shape's maximum
post-release warmup allocation supplies a shape-aware baseline; subsequent calls
must stay within 64 MiB of it. Retained reservations and absolute/sampled peaks
are saved separately. Reservation alone is not a leak. No expected active 64-frame
input allocation is compared with an active eight-frame input baseline.

For each source question, arm and budget, R=(max−min)/median over the three
independent-session observations. S is the median of eight question-level R's.
Retain all R, absolute ranges, request/preparation/remainder observations and flags
for R>50%. No exclusions, replacements or denominator inflation are allowed.
A policy qualifies only if S(8)≤20% and S(64)≤20%, after all completeness,
input/output, native-token-decoding, greedy, memory, call-accounting, lifecycle,
historical-preservation and independent-record checks pass. Select A if qualified,
otherwise B if qualified, otherwise stop. Qualification is eligibility for one
new confirmation, never feasibility confirmation or evidence of acceleration.

Conditional confirmation freezes a new run, seed 20261030, all 24 questions,
three budget permutations over 8/16/32/48/64 and one question permutation per
repetition. Reject schedules before outcomes until one has 64 last and another
has 64 followed by lower. A uses 15 video workers; B uses three resident video
workers. Each uses one separate text worker. There are 360 new video primaries,
24 new text primaries and 32 collection warmups. Full preflight uses five fresh
A workers (25 calls) or two five-block B workers (50 calls): two warmups, two
integrity requests and one profile per block. Prepare all eight sources at all
five budgets before final admission. Existing matching CPU evidence may be reused
with its provenance. No historical or diagnostic primary row enters confirmation.

Each confirmation budget must have median R≤20% over 24 questions and pass all
other gates. Only this new complete run can receive FEASIBILITY_CONFIRMED.
Otherwise distinguish completed MEASUREMENT_UNSTABLE from incomplete, integrity,
input or resource failures. One confirmation attempt; no policy switching.

One shared child ledger inherits every historical charge, including E1's private
ledger and prior publication reserves. Recursive discovery and task pointers must
not select an older sibling. Continuous implementation/verification work is
checkpointed into closed sessions. Before phases, require 1.5× complete execution
and overhead, 120 s interruption contingency and ≥600 s protected closeout. No
time grant is added: cumulative cap 27,600 s, 20 GiB project storage and 1 GiB
acquired videos. Preflight screening uses conservative all-record historical
costs; main forecasts use real transition profiles, all-source native preparation,
startup and serialization overhead. No frame-ratio extrapolation.

Worker deadlines cover actual planned calls and the protected reserve; at most
one model worker runs. Each call has started/completed/failed events and an atomic
sealed record. Valid completed wrong/invalid/slow outputs remain completed. Resume
skips only whole completed workers with matching fingerprints/complete records.
Partial sessions stop inference; a new worker cannot impersonate prior history.
Owned process trees are terminated and reaped on interruption. Historical files
and failed attempts remain immutable. Reports/accounting do not change the freeze.

Public output contains anonymous schedules, aggregates, sanitized status and
provenance, resource summaries, verification and figures. Questions, labels, raw
answers/tokens, full paths, worker logs and all private manifests stay ignored.
Negative/resource-blocked results are useful outcomes and must still be reported.
No third arm, tuned setting, rerun for a pass, controller or Experiment 2.

CLI (replace RUN with the new task directory initialized from the latest ledger):

```powershell
.venv\Scripts\python.exe scripts/stabilize_e000_timing_v5.py init --run-dir RUN --task-start TIMEZONE_AWARE_TASK_START
.venv\Scripts\python.exe scripts/stabilize_e000_timing_v5.py prepare --run-dir RUN
.venv\Scripts\python.exe scripts/stabilize_e000_timing_v5.py preflight --run-dir RUN
.venv\Scripts\python.exe scripts/stabilize_e000_timing_v5.py collect --run-dir RUN
.venv\Scripts\python.exe scripts/stabilize_e000_timing_v5.py analyze --run-dir RUN
.venv\Scripts\python.exe scripts/stabilize_e000_timing_v5.py confirmation --run-dir RUN
.venv\Scripts\python.exe scripts/stabilize_e000_timing_v5.py verify --run-dir RUN
.venv\Scripts\python.exe scripts/stabilize_e000_timing_v5.py report --run-dir RUN
.venv\Scripts\python.exe scripts/stabilize_e000_timing_v5.py account --run-dir RUN
.venv\Scripts\python.exe scripts/stabilize_e000_timing_v5.py close --run-dir RUN
```

Initialization uses a new directory, records its authoritative parent path/hash
and snapshots every existing result file privately. The actual task used an
equivalent local bootstrap before this runner existed; subsequent charges share
that ledger. CPU preparation is bounded in an owned subprocess, with admission
from historical per-source/per-budget maxima for all native/pixel validation
decodes. Main-worker deadlines are computed separately from each worker's actual
scheduled workload and real probes, saved before primary outcomes, and checked
against the protected remaining reserve. Matching reused CPU evidence requires
sealed records, the identical adapter/processor fingerprint and historical hashes.

The independent verifier is read-only and re-decodes every saved generated token
sequence using the frozen native tokenizer backend. CPU fixtures test code only.
New GPU paths remain unverified until actually exercised. If either arm fails,
the next single proposed diagnostic should follow saved stage/environment evidence;
it is not launched within this task. Three rounds do not identify a unique
historical OS, thermal, I/O, decoder or allocator cause.
