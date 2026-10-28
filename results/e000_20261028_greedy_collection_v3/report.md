# Experiment 0 — frozen visual-only video-QA feasibility

Status: **MEASUREMENT_UNSTABLE**. Timing stability or live allocation growth requires interpretation

Model: `Qwen/Qwen3-VL-4B-Instruct`. Planned: 8 complete short Video-MME videos,
three official questions each; budgets 8/16/32/48/64; three repetitions; batch one;
greedy generation up to eight new tokens. No audio, subtitles, ASR, or captions.
Uniform-64 is a sampled operational reference, not all-frame processing, an oracle,
or an assumed accuracy upper bound. Uniform sets are not necessarily nested.

| Condition | Successful / planned | Rep. 1 correct / 24 | Accuracy | Median request s | GPU allocated / reserved GiB |
|---|---:|---:|---:|---:|---:|
| Uniform-8 | 72 / 72 | 15 / 24 | 0.625 | 2.528 | 8.432 / 8.488 |
| Uniform-16 | 72 / 72 | 18 / 24 | 0.750 | 2.859 | 8.602 / 8.996 |
| Uniform-32 | 72 / 72 | 19 / 24 | 0.792 | 3.954 | 9.024 / 10.426 |
| Uniform-48 | 72 / 72 | 18 / 24 | 0.750 | 4.317 | 9.620 / 12.594 |
| Uniform-64 | 72 / 72 | 21 / 24 | 0.875 | 5.718 | 10.387 / 10.543 |
| text-only (separate diagnostic) | 24 / 24 | 14 / 24 | 0.583 | 0.293 | 8.314 / 8.336 |

Real completed measured calls: 360 / 360 video,
24 / 24 text-only. Additional request attempts:
`{'component_profile': 5, 'fit_probe': 5, 'integrity_probe': 5, 'trial': 384, 'warmup': 43}`. Text-only is a separate diagnostic.
Missing measurements are unavailable, never zero. Incomplete subsets do not form a
complete quality–cost curve. Invalid answers count as incorrect; no best-repetition selection.

Resident-model request latency starts before opening the compressed local video and
includes full sequential decoding, selection, resize, processing, tokenization,
GPU transfer, vision/prefill/generation, output transfer, text decoding, parsing,
and final CUDA synchronization. OS file caching is permitted: these are not cold-disk
measurements. Labels/scoring, diagnostic hashes, serialization, model loading,
two warmups per worker, integrity probes and component profiling are outside this
primary interval, but included in the execution ledger. Each request decodes again.
No frame/features/prefix/KV reuse occurs between questions. Execution policy:
`threaded-decode-trim64-v2`; decoder
`FRAME/4`;
cache trim budgets `[64]`. Cleanup is inside the
request timer when the separately versioned policy enables it.

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
15372.690713438558. Recorded project storage bytes:
17036072136; peak observed bytes:
17036072136. Limits: 19200 active seconds,
20 GiB additional storage, with at most 1 GiB video transfer. Setup costs and failed
attempts remain in `resource_ledger.json`; idle time between closed sessions is excluded.

Saved setup and diagnostic costs: `{'cpu_decode_probe_seconds': 35.14828810000108, 'prior_run_model_invocations': 491, 'worker_model_load_seconds': 194.55596499999228, 'worker_model_load_count': 22, 'warmup_elapsed_seconds': 200.0616463999977, 'probe_profile_elapsed_seconds': 64.33242119999886, 'preparation_session_seconds': 1581.57298, 'external_setup_seconds': 2171.5776944385652, 'definition': 'Descriptive startup/diagnostic costs; overlapping wall intervals are not summed as a disjoint decomposition'}`.
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
