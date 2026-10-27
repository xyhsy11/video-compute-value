# Executed experiment and publication record

The repository was created at `C:\video-compute-value` and published privately to
`xyhsy11/video-compute-value` on branch `main` using the existing Git author identity.

The implementation includes the native frozen Qwen3-VL adapter, original Video-MME
adapter, bounded selective ZIP acquisition/local loading, exact PTS sampling,
aspect-preserving letterboxing, fresh worker blocks, synchronized request timing,
memory counters/sampling, atomic records, provenance-checked resumption, analysis,
independent verification and descriptive quality/stability reporting.

Executed commands included:

```powershell
.\scripts\validate.ps1 -RunDir results/e000_20261027_collection_v2
& .\.venv\Scripts\python.exe -m video_compute_value.e000 preflight --config configs/e000.yaml --run-dir results/e000_20261027_implementation
& .\.venv\Scripts\python.exe -m video_compute_value.e000 prepare --config configs/e000.yaml --run-dir results/e000_20261027_implementation --acquire
& .\.venv\Scripts\python.exe -m video_compute_value.e000 prepare --config configs/e000.yaml --run-dir results/e000_20261027_collection_v2 --annotations data/annotations/ead1408f75b618502df9a1d8e0950166bf0a2a0b.parquet --annotation-revision ead1408f75b618502df9a1d8e0950166bf0a2a0b --video-dir data/videos
& .\.venv\Scripts\python.exe -m video_compute_value.e000 smoke --config configs/e000.yaml --run-dir results/e000_20261027_collection_v2
& .\.venv\Scripts\python.exe -m video_compute_value.e000 run --config configs/e000.yaml --run-dir results/e000_20261027_collection_v2
& .\.venv\Scripts\python.exe -m video_compute_value.e000 verify --config configs/e000.yaml --run-dir results/e000_20261027_collection_v2
& .\.venv\Scripts\python.exe scripts/report_quality.py --config configs/e000.yaml --run-dir results/e000_20261027_collection_v2
```

The analysis pipeline ran inside `run` and `verify`; its standalone `analyze` command
was also exercised during implementation. Earlier named runs each executed 25 real
development probe calls. Their sources and consumption remain preserved in the
lineage and resource ledger. The final source hashes match the original eight
acquired videos. No completed primary trials were selectively replaced.

CPU validation: **42 tests passed**, including native processor count/grid checks,
sampling/tie handling, strict parsing and label exclusion, synchronized timing,
resume fingerprint rejection, duplicate prevention, and saved-record corruption
checks. Static checks passed. A separate synthetic Windows lifecycle check spawned
one temporary child; zero children survived killing its venv launcher. These tests
are separate from real inference.

Real Qwen inference occurred on an NVIDIA GeForce RTX 3080 Laptop GPU with **16 GiB
physical VRAM**, using BF16, SDPA, PyTorch 2.8.0+cu128 and Transformers 4.57.1.
The model revision is `ebb281ec70b05090aa6165b016eac8ec08e71b17`; annotation revision
is `ead1408f75b618502df9a1d8e0950166bf0a2a0b`. All weights stayed frozen on CUDA.

Planned/completed measured calls: **360/360 video**, **24/24 text-only**. Failures:
**0**; unattempted required calls: **0**. The final run has 441 completed model
invocations including 42 warmups and 15 fit/integrity/component calls. Preserved
earlier probes add 50 calls: **491 total real model invocations**, of which 384 are
the required measured collection.

Overall status: **REPEATABILITY_UNRESOLVED**. All 384 records verify independently;
configuration/source provenance, actual input budgets, prepared-input repeatability,
call accounting and live-memory-growth checks pass. Parsed answers disagree in
**8/120 question/budget cells** despite identical prepared inputs. Every video
budget also exceeds the proposed median timing-range target of 20%. The Uniform-64
median relative timing range is about 53.84%. Operational feasibility is not fully
confirmed.

Primary correct counts out of 24 are 14, 18, 19, 18 and 21 at budgets 8, 16, 32,
48 and 64. Text-only scores 14/24. The report and summary contain the complete
latency/memory table, token counts, paired changes, per-video summaries and plots.
All 120 repetition-1 video outputs are parseable; 23/24 text-only outputs are
parseable. Repetitions are repeatability measurements on the same 24 questions,
not additional independent examples.

Recorded active execution at the final reporting checkpoint: **6168.48 s**
(about 1 h 43 min), leaving **8231.52 s** of the 4-hour allowance. Project footprint:
**15.8572 GiB**, leaving **4.1428 GiB** of the 20-GiB limit. Confirmed video transfer:
**77.8712 MiB**, plus a conservative **1 MiB** interrupted-range reserve, leaving
about **945.13 MiB** of the 1-GiB acquisition allowance. Resource values are observed
checkpoints; setup, downloads, failed setup attempts, preparation, all probes,
worker loading/warmups, collection and validation are included. No paid API or
cloud hardware was used.

Uniform-64 reached 10.39 GiB peak PyTorch allocated, 16.73 GiB peak PyTorch reserved,
and 15.95 GiB sampled device usage. Reserved allocator accounting and sampled
physical-device usage are distinct. Process GPU sampling was unavailable on this
host and is recorded as null. These observations do not identify the cause of
answer or timing variability.

Next action: inspect the eight flagged question/budget cells in
`repeatability_assessment.json` and define a separately named deterministic-inference
and timing diagnostic. Preserve this completed pilot and treat its sources as
development exposure. This pilot does not test adaptive sampling or the main
research claim.
