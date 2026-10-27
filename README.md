# video-compute-value

Can inexpensive observations predict the task value of additional visual computation
in a frozen video-language model? Experiment 0 establishes the measurement foundation:
whether Qwen3-VL-4B-Instruct fits, exact inputs can be reproduced, multiple-choice
answers can be scored, and request latency/memory can be measured across uniform
frame budgets. It does not establish adaptive sampling, oracle headroom, or the
project's main scientific claim.

The original specification is in
[`protocols/e000_frozen_videoqa_feasibility_v1.md`](protocols/e000_frozen_videoqa_feasibility_v1.md).
The user's subsequent explicit instruction authorizes GitHub creation and publication.
This repository is separate from `video-refresh-benchmark`.

## Installation on Windows (PowerShell)

Use an isolated Python 3.11/3.12 environment. The bootstrap detects Python on PATH
or Codex's bundled runtime; pass `-Python 'C:\path with spaces\python.exe'` if needed.
No environment activation is required.

```powershell
Set-Location -LiteralPath 'C:\video-compute-value'
.\scripts\bootstrap.ps1 -Model
.\scripts\validate.ps1
$run = 'results/e000_20261027_implementation'
$python = '.\.venv\Scripts\python.exe'
& $python -m video_compute_value.e000 preflight --config configs/e000.yaml --run-dir $run
```

Pinned inference stack: PyTorch 2.8.0/CUDA 12.8, torchvision 0.23.0,
Transformers 4.57.1, Accelerate 1.10.1. CUDA is required for measured inference.
BF16 is selected when the actual GPU supports it, otherwise FP16; the choice is
frozen before probing. One GPU, SDPA, eval/inference mode, frozen weights, no
offload/quantization. Model/software changes require a new named run.
`requirements-validated.txt`, when present, captures the actual transitive environment.
Some installed versions may have local CUDA suffixes.

On Linux use `python3 -m venv .venv`, then `.venv/bin/python -m pip install -e '.[dev]'`,
install the pinned CUDA wheels from the official PyTorch index, and
`.venv/bin/python -m pip install -e '.[dev,model]'`. The CLI is cross-platform;
the supplied helper scripts and actual local validation target Windows.
Account external setup/test time with the `account` command described below.

## Prepare the original data

Use academic research access through the
[official Video-MME project](https://github.com/MME-Benchmarks/Video-MME#-dataset).
Its linked Hugging Face dataset now redirects to `lmms-eval/Video-MME`.
Restricted videos, questions and labels remain local and ignored by Git.
No audio, external subtitles, ASR or generated captions are used.

The automatic path downloads the small official annotation parquet, inspects archive
sizes and ZIP directories, and fetches only pilot members using verified HTTP 206
ranges. It aborts if an exact range is not honored. It never downloads the full
~101 GB collection. Cumulative video transfer is capped at 1 GiB, included in
the overall 20 GiB storage limit.

```powershell
& $python -m video_compute_value.e000 prepare --config configs/e000.yaml --run-dir $run --acquire
```

For existing complete original videos, name files by original `videoID` (for example,
`<videoID>.mp4`) and use a directory containing those files:

```powershell
& $python -m video_compute_value.e000 prepare --config configs/e000.yaml --run-dir $run --video-dir 'C:\datasets\Video MME\videos'
```

To use an existing official annotation file too:

```powershell
& $python -m video_compute_value.e000 prepare --config configs/e000.yaml --run-dir $run --annotations 'C:\datasets\Video MME\annotations.parquet' --annotation-revision '<40-character upstream commit SHA>' --video-dir 'C:\datasets\Video MME\videos'
```

Sources are ranked by SHA-256 of canonical `{seed, source_id}`; select the first
eight eligible distinct sources, retaining all three official questions unchanged.
Selection never reads predictions, accuracy or perceived difficulty. Every skipped
candidate receives a reason. Inputs must decode completely and supply distinct
frames at every budget. This adapter excludes VFR, missing/nonmonotonic timestamps,
nonzero starting PTS, changing dimensions and inconsistent full durations before
cohort freeze. These are explicit pilot limitations.

Uniform-B divides the full timeline into B intervals and chooses the nearest source
PTS to each center; exact ties choose the earlier PTS. Sequential full-clip decoding
retains only selected resized frames. RGB conversion is PyAV `rgb24`; Pillow bicubic
resizing uses an aspect-preserving scale followed by rounding down to the 32-pixel
patch/merge grid (small grid-rounding aspect differences are recorded). The fixed
per-video dimensions stay identical across budgets and within 65,536 pixels/frame.
The native video processor has second sampling/resizing explicitly disabled.
Native temporal pairs use source-index/source-FPS timestamps and one-decimal text
time tags; all selected PTS, represented pair times, grids, counts and tokens are checked.

## Fit, freeze, collect and resume

```powershell
& $python -m video_compute_value.e000 smoke --config configs/e000.yaml --run-dir $run
& $python -m video_compute_value.e000 run --config configs/e000.yaml --run-dir $run
# After interrupted collection, use the identical configuration and run directory:
& $python -m video_compute_value.e000 resume --config configs/e000.yaml --run-dir $run
& $python -m video_compute_value.e000 analyze --config configs/e000.yaml --run-dir $run
& $python -m video_compute_value.e000 verify --config configs/e000.yaml --run-dir $run
```

`smoke` resolves immutable revisions and explicit generation settings, hashes model
files, environment, implementation and cohort, and saves seed-determined execution
order. Each of five fresh probe workers runs two warmups, two integrity/fit requests,
and one separate component profile (25 additional model calls). Uniform-64 must fit.
The remaining-run projection includes all 16 collection workers, two warmups each,
384 measured requests, final verification and a 1.5 safety factor. Collection starts
only after CPU-only Uniform-64 preparation probes on all eight clips account for
decoding cost differences from the prespecified model-probe clip. It starts
only if projected cost fits the remaining cumulative budget. Probe outputs are not
used to select a model, precision, resolution, cohort or prompt.

Collection is 360 measured video requests (8 × 3 × 5 × 3) and 24 separate text-only
requests. Two warmups per collection worker add 32 calls. Each budget/repetition
block gets a fresh process; only one GPU worker is active. Every question reopens,
fully decodes and prepares its video. No cross-request frame/features/prefix/KV cache.
OS file caching is allowed. Greedy generation uses at most eight new tokens, fixed
EOS/padding and a fixed prompt; no history or requested chain of thought.
Strict parsing accepts one A–D letter, optional `Answer:` and optional `.`/`!`.
Invalid/empty outputs are incorrect and remain completed records; no retries to
improve accuracy or parseability. Labels never enter an inference request.

Atomic records carry experiment/run, configuration, source/question, budget and
repetition identities. Resume validates provenance, skips completed identities,
retains failed attempts, and continues the same resource ledger. Retries of failures
remain separately counted. Configuration/source changes require a new named run.
Unclosed resource sessions after a crash are conservatively charged elapsed wall
time (capped at the full budget); closed idle gaps are excluded.

## Measurements and interpretation

Primary latency is synchronized monotonic host time for a resident-model,
local-file, single-question request. It starts before opening compressed video and
includes read/decode/selection/resize, prompt and processor/tokenizer work, GPU
transfer, vision encoding, LM prefill, generation, output transfer, text decoding,
parsing and final CUDA synchronization. Scoring, serialization and diagnostic hashes
are outside that interval and inside the overall ledger. Startup/loading, preparation,
warmups, probes, failures and validation are accounted separately. Component profiles
are separate intrusive measurements, never added to primary-pass timings.

GPU metrics include baseline allocated/reserved, absolute peak allocated/reserved
including resident weights, and separately labeled increments. Optional NVML device/
process and process RAM maxima are sampled every 50 ms, and unavailable values are
null. PyTorch memory is not total device usage. Live allocated growth is flagged;
retained allocator reservations alone are not called a leak.

Primary accuracy uses repetition 1 only, out of fixed 24 questions. Timing first
takes a median over three repetitions per question/configuration, then reports the
distribution over 24 question medians and per-video medians. Measured speedup is
reference median latency / condition median latency; frame ratio is budget / 64.
Relative timing range is `(max - min) / median`, with a proposed per-budget median
target <=20% and individual cells >50% flagged. Repeated input hashes, parsed
answers and raw outputs/tokens are checked. Incomplete collection reports planned,
attempted, successful, failed and unattempted counts and withholds complete curves.

Uniform-64 is the operational dense sampled reference, not all-frame processing,
an oracle, or an accuracy upper bound. Uniform sets need not be nested. Quality need
not improve monotonically with frame count. Text-only is a separate diagnostic,
never a zero-cost point. There are only eight videos/24 clustered questions;
repetitions do not create independent examples or support significance claims.

## Saved evidence and resource limits

Each unique `results/<run>/` holds environment/model manifests, resolved configuration,
cohort/exposure/order, atomic trial records, `raw_trials.jsonl`, failed attempts,
worker/call diagnostics, `resource_ledger.json`, `summary.csv/json`, `verification.json`,
`status.json`, `report.md`, and figures when complete real results exist. Public-safe
reports and source identifiers/hashes are committed; restricted raw artifacts remain
ignored locally. No synthetic predictions, latencies or memory figures are saved as
real experiment results. Tests explicitly use synthetic fixtures.

The resource ledger counts the complete new project, including `.venv`, caches,
temporary files and Git, against 20 GiB. Model weights are downloaded once to the
local HF cache or reused from an existing complete immutable cache. The 4-hour active
limit spans resumptions. All CLI operations use the same ledger; externally launched
setup/validation costs can be recorded explicitly:

```powershell
& $python -m video_compute_value.e000 account --config configs/e000.yaml --run-dir $run --event-name 'external validation, including failures' --event-seconds 12.34
```

Use measured elapsed seconds, not the illustrative value above. The PowerShell
helpers do this automatically. Network transfer is recorded for annotations/ranges;
model download logical bytes and actual project storage are separately recorded.
Storage peaks are observations at checkpoints, not continuous filesystem sampling.
No paid API, cloud rental, unrelated-file deletion, implicit fallback or silent
configuration relaxation is permitted.

Final statuses include `FEASIBILITY_CONFIRMED`, `BLOCKED_ENVIRONMENT`,
`BLOCKED_MODEL_ACCESS`, `BLOCKED_INPUT`, `BLOCKED_RESOURCE`, `INPUT_INTEGRITY_FAILED`,
`MEASUREMENT_UNSTABLE`, `REPEATABILITY_UNRESOLVED`, and `PARTIAL`.
Confirmation requires complete collection and all implemented verification gates.
See the current run report for actual executed commands and any unverified real paths.

Implementation references and exact installed processor source hashes are recorded
in `manifests/implementation_references.json` and the frozen model manifest. The
[Transformers 4.57.1 source](https://github.com/huggingface/transformers/tree/v4.57.1/src/transformers/models/qwen3_vl)
and [PyTorch 2.8 CUDA documentation](https://docs.pytorch.org/docs/2.8/notes/cuda.html)
define the native APIs used here.
