# Experiment 0 — frozen visual-only video-QA feasibility

Status: **REPEATABILITY_UNRESOLVED**. Parsed answers disagree across repetitions

Model: `Qwen/Qwen3-VL-4B-Instruct`. Planned: 8 complete short Video-MME videos,
three official questions each; budgets 8/16/32/48/64; three repetitions; batch one;
greedy generation up to eight new tokens. No audio, subtitles, ASR, or captions.
Uniform-64 is a sampled operational reference, not all-frame processing, an oracle,
or an assumed accuracy upper bound. Uniform sets are not necessarily nested.

| Condition | Successful / planned | Rep. 1 correct / 24 | Accuracy | Median request s | GPU allocated / reserved GiB |
|---|---:|---:|---:|---:|---:|
| Uniform-8 | 72 / 72 | 14 / 24 | 0.583 | 4.278 | 8.432 / 8.490 |
| Uniform-16 | 72 / 72 | 18 / 24 | 0.750 | 4.523 | 8.602 / 8.996 |
| Uniform-32 | 72 / 72 | 19 / 24 | 0.792 | 4.436 | 9.024 / 10.426 |
| Uniform-48 | 72 / 72 | 18 / 24 | 0.750 | 5.726 | 9.620 / 12.594 |
| Uniform-64 | 72 / 72 | 21 / 24 | 0.875 | 7.222 | 10.387 / 16.729 |
| text-only (separate diagnostic) | 24 / 24 | 14 / 24 | 0.583 | 0.247 | 8.314 / 8.340 |

Real completed measured calls: 360 / 360 video,
24 / 24 text-only. Additional request attempts:
`{'component_profile': 5, 'fit_probe': 5, 'integrity_probe': 5, 'trial': 384, 'warmup': 42}`. Text-only is a separate diagnostic.
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
6118.9213663385635. Recorded project storage bytes:
17026506236; peak observed bytes:
17026506236. Limits: 14,400 active seconds,
20 GiB additional storage, with at most 1 GiB video transfer. Setup costs and failed
attempts remain in `resource_ledger.json`; idle time between closed sessions is excluded.

Saved setup and diagnostic costs: `{'cpu_decode_probe_seconds': 30.448329099997864, 'prior_run_model_invocations': 50, 'worker_model_load_seconds': 115.24266840000382, 'worker_model_load_count': 21, 'warmup_elapsed_seconds': 220.99210480000147, 'probe_profile_elapsed_seconds': 69.45508180000252, 'preparation_session_seconds': 1581.57298, 'external_setup_seconds': 424.1923863385651, 'definition': 'Descriptive startup/diagnostic costs; overlapping wall intervals are not summed as a disjoint decomposition'}`.
These descriptive intervals can overlap; they are not added into a purported
disjoint decomposition of the primary request timer.

Operational confirmation requires all source/config/environment/integrity gates,
all 384 completed calls, stable parsed answers, interpretable timing, complete
memory records, reconciled calls, and independent saved-record verification.
Consult `verification.json` and per-cell diagnostics in `summary.json`.
Question/label-bearing cohort and raw-trial artifacts remain local. Settings, source
identifiers/hashes, exposure identifiers and aggregate reports may be committed.

This is eight videos and 24 questions. Repetitions do not increase sample size.
Operational feasibility does not establish useful visual reasoning, adaptive
sampling, oracle headroom, statistical significance, or a research contribution.
Quality versus text-only and concentrated question-level errors can be inspected
in the following saved-record addendum. The real GPU path was exercised for all required requests.

Next action: Investigate the flagged question/budget cells in a separately named deterministic-inference diagnostic; preserve every original trial.

<!-- quality-assessment -->

## Repeatability and timing audit

Prepared-input mismatches: 0/120 cells. Parsed-answer disagreements: 8/120 cells.

Flagged question/budget cells: 197-3 at Uniform-8, 002-3 at Uniform-8, 098-1 at Uniform-8, 034-1 at Uniform-48, 235-1 at Uniform-8, 235-1 at Uniform-32, 235-3 at Uniform-8, 235-3 at Uniform-48.

Budgets exceeding the proposed median relative timing range of 20%: [8, 16, 32, 48, 64]. Individual timing cells above 50% remain flagged in `summary.json`; completed trials were never selectively replaced.

Uniform-64 observed PyTorch peaks: 10.39 GiB allocated, 16.73 GiB reserved. Sampled device usage reached 15.95 GiB. Reserved memory is allocator accounting; device sampling is a separate observation. These measurements do not identify the cause of the timing or answer disagreements.

## Descriptive task-quality assessment

120/120 repetition-1 video outputs met the fixed answer format; text-only: 23/24.

Video accuracy ranged from 58.3% to 87.5% across the five budgets; text-only scored 14/24 (58.3%). These are descriptive comparisons on 24 questions clustered in eight videos. Repetitions do not increase sample size. Useful visual reasoning, adaptive sampling, and statistical generalization are not established.

Questions incorrect under every video budget: 002-2, 034-2, 235-1. Every question is listed below; errors are counted over the five conditions, without selecting repetitions.

| Source | Question | Incorrect video conditions / 5 | Invalid conditions / 5 | Text-only correct |
|---|---|---:|---:|---|
| 1q-5IIyZL20 | 197-1 | 0 | 0 | False |
| 1q-5IIyZL20 | 197-2 | 0 | 0 | True |
| 1q-5IIyZL20 | 197-3 | 2 | 0 | True |
| N1cdUjctpG8 | 002-1 | 1 | 0 | False |
| N1cdUjctpG8 | 002-2 | 5 | 0 | False |
| N1cdUjctpG8 | 002-3 | 1 | 0 | True |
| PU-XOFIJMlg | 098-1 | 1 | 0 | True |
| PU-XOFIJMlg | 098-2 | 2 | 0 | True |
| PU-XOFIJMlg | 098-3 | 0 | 0 | False |
| SnOc5W0PgVE | 078-1 | 0 | 0 | True |
| SnOc5W0PgVE | 078-2 | 0 | 0 | True |
| SnOc5W0PgVE | 078-3 | 0 | 0 | True |
| WViSvPFUVd8 | 079-1 | 0 | 0 | True |
| WViSvPFUVd8 | 079-2 | 0 | 0 | True |
| WViSvPFUVd8 | 079-3 | 0 | 0 | True |
| blSnLEZe-sI | 034-1 | 3 | 0 | True |
| blSnLEZe-sI | 034-2 | 5 | 0 | False |
| blSnLEZe-sI | 034-3 | 2 | 0 | False |
| kij9GXWdmik | 235-1 | 5 | 0 | False |
| kij9GXWdmik | 235-2 | 0 | 0 | True |
| kij9GXWdmik | 235-3 | 3 | 0 | True |
| v0m3E_uSyFg | 234-1 | 0 | 0 | False |
| v0m3E_uSyFg | 234-2 | 0 | 0 | False |
| v0m3E_uSyFg | 234-3 | 0 | 0 | False |

Reproduce this addendum with `python scripts/report_quality.py --config configs/e000.yaml --run-dir results/e000_20261027_collection_v2`. Its code hash is recorded in `quality_assessment.json`.
