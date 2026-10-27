You are working as the implementation and experimental-validation engineer for a new research project.

Create a local Git repository named:

    video-compute-value

Then implement and, when the environment permits, execute:

    Experiment 0 — Can we run and measure frozen video question answering reliably?

Experiment ID:

    e000_frozen_videoqa_feasibility_v1

Complete the repository setup, runnable implementation, validation, and available execution. Do not stop after producing a plan. If a real blocker prevents execution, finish the unaffected implementation and document the exact blocker and the command or resource needed to continue.

============================================================
1. PURPOSE AND SCOPE
============================================================

The project studies whether inexpensive observations can predict the task value of additional visual computation in a frozen video-language model.

Experiment 0 establishes the measurement foundation:

- Can the selected model run on the available hardware?
- Can we reproduce the exact video inputs and model configuration?
- Can we score answers automatically and consistently?
- Can we measure full request latency and peak memory?
- What happens to measured quality and cost as uniform frame sampling changes?

This experiment does not establish that adaptive sampling works. It does not train a controller, estimate oracle headroom, or test the main scientific claim.

Implement one model adapter, one dataset adapter, one experiment runner, and one analysis pipeline. Keep the code small and understandable.

The existing video-refresh-benchmark project is a separate project. Preserve its files, experimental records, and resource accounting.

============================================================
2. REPOSITORY SETUP
============================================================

First inspect the current directory, Git state, and applicable AGENTS.md instructions.

- If already inside the intended video-compute-value repository, use it.
- Otherwise create video-compute-value in the intended writable parent directory.
- Avoid nesting it inside video-refresh-benchmark.
- Preserve existing user files and unrelated changes.
- Initialize local Git if necessary.
- Do not invent Git author credentials.
- Remote creation, publication, and pushing are outside this task.

Create a minimal structure containing:

    README.md
    AGENTS.md
    .gitignore
    pyproject.toml
    configs/e000.yaml
    protocols/e000_frozen_videoqa_feasibility_v1.md
    src/video_compute_value/
    tests/
    scripts/
    manifests/
    results/

Use additional files only where they improve clarity.

README.md must explain:
- The research question.
- What Experiment 0 establishes.
- Installation and execution commands.
- Measurement definitions.
- Where results and failure records are stored.
- The distinction between this pilot and later scientific experiments.

Write a short AGENTS.md covering:
- Current experimental scope.
- Preservation of raw measurements.
- Configuration and source provenance.
- Resumable execution.
- Resource limits.
- Prohibition on presenting synthetic checks as real model results.

Keep model weights, raw videos, environments, credentials, and large caches out of Git. Keep restricted dataset contents local; commit identifiers, acquisition instructions, hashes, code, and appropriate summaries.

Use an isolated Python environment. Detect the actual platform and provide commands appropriate for it. Support paths containing spaces and non-ASCII characters.

Do not assume that the historical laptop specifications describe the current execution environment.

============================================================
3. DEFAULT EXPERIMENT CONFIGURATION
============================================================

Put these settings in configs/e000.yaml:

    experiment_id: e000_frozen_videoqa_feasibility_v1
    model_id: Qwen/Qwen3-VL-4B-Instruct
    dataset: Video-MME v1
    duration_category: short
    num_videos: 8
    questions_per_video: 3
    frame_budgets: [8, 16, 32, 48, 64]
    reference_budget: 64
    repetitions: 3
    batch_size: 1
    seed: 20261027
    max_new_tokens: 8
    do_sample: false
    num_beams: 1
    max_pixels_per_frame: 65536
    cumulative_execution_limit_hours: 4
    additional_disk_limit_gib: 20
    video_acquisition_limit_gib: 1

The video acquisition limit is included within the overall disk limit.

Resolve and freeze before the measured run:
- Immutable model revision.
- Dataset annotation revision.
- Processor and tokenizer revisions.
- Python and package versions.
- Model dtype.
- Attention backend.
- Video decoder and version.
- Spatial preprocessing.
- Timestamp handling.
- Generation configuration.
- Prompt and parser.
- Cohort and execution order.

Do not leave "latest", an unresolved branch, or an implicit automatic setting in the final frozen configuration.

Use a compatible released software stack where possible. Consult official documentation and inspect installed APIs instead of guessing parameter names.

============================================================
4. ENVIRONMENT AND RESOURCE PREFLIGHT
============================================================

Record:
- OS and Python version.
- GPU model, total VRAM, and available VRAM.
- Driver version and the CUDA runtime used by PyTorch.
- PyTorch, Transformers, decoder, and related package versions.
- Available RAM and disk space.
- Existing reusable model/data caches.
- Relevant CPU thread settings.
- Laptop power state and GPU temperature/power information when available.

Do not assume an RTX 3080 Laptop GPU has a particular VRAM capacity.

Prefer:
- Local Transformers inference.
- One GPU.
- BF16 when supported; otherwise FP16 selected and documented before freezing.
- A supported attention implementation that remains fixed for all configurations.

Use eval mode and inference mode. Freeze all weights.

Do not silently introduce:
- CPU or disk model offloading.
- Quantization.
- A smaller model.
- Reduced reference frame count.
- Different resolution for a failing configuration.
- A different backend during collection.

If the primary configuration cannot fit, preserve the evidence and report BLOCKED_RESOURCE. Suggest a separately named revised configuration, but do not mix it into this experiment.

Resource accounting must include downloads, preparation, fit probes, model loading, warmups, profiling, failed attempts, collection, and validation.

The four-hour limit is cumulative across resumptions. Idle time between sessions is excluded; a new session does not reset the execution budget.

Track newly added storage across the environment, download caches, model files, data, temporary files, and results. Avoid duplicate weight downloads. Do not delete unrelated user files to make room.

Do not rent cloud hardware or incur paid API charges.

============================================================
5. DATA PREPARATION AND COHORT FREEZE
============================================================

Use the original Video-MME benchmark, short-duration category.

Select:
- Eight distinct source videos.
- All three official multiple-choice questions for each video.
- Original question text, option order, and answer labels.

Use complete videos. Do not shorten a video while retaining questions about the full original.

This is a visual-only condition:
- No audio input.
- No external subtitles.
- No ASR transcripts.
- No generated captions.
- Text already burned into the video remains visible.

Use official project links and authorized dataset access. Prefer existing local files.

Acquire only the required pilot material. Inspect download sizes first. Do not download the full video collection by default.

If videos are distributed in large ZIP archives, selective extraction through supported HTTP range access is acceptable. Verify range responses, account for transferred bytes, and abort if the server starts returning an unexpectedly large full archive.

If selected files cannot be obtained within the limits, document BLOCKED_INPUT or BLOCKED_RESOURCE and finish the local-data loading path.

Selection procedure:
1. Identify candidates using official metadata.
2. Rank unique source IDs by a stable SHA-256-based ordering using the fixed seed.
3. Apply prespecified availability, duration-category, decoding, timestamp, and frame-count eligibility checks.
4. Select the first eight eligible distinct videos.
5. Record every excluded candidate and its reason.
6. Freeze the manifest before any model inference on these videos.

Eligibility must ensure that every requested budget can produce the required distinct source frames.

Do not use correct answers, model predictions, perceived difficulty, or interesting results to choose or replace videos.

The frozen manifest must record:
- Dataset and question IDs.
- Source/parent identifiers where available.
- Acquisition source.
- Local video SHA-256.
- Annotation and label hashes.
- File size, duration, source dimensions, and timing metadata.
- Selection order and exclusion reasons.

Keep answer labels out of inference requests. Scoring may access them afterward.

Record all pilot sources in an exposure manifest. They are development data and must not later be described as untouched confirmation data.

============================================================
6. EXACT FRAME SAMPLING AND PREPROCESSING
============================================================

Implement these conditions:

    Uniform-8
    Uniform-16
    Uniform-32
    Uniform-48
    Uniform-64

Uniform-64 is the operational dense reference. It is not all-frame processing, an oracle, or a guaranteed accuracy upper bound.

For each budget B:
- Divide the full video timeline into B equal intervals.
- Use the center of each interval as a target timestamp.
- Select the nearest source frame by presentation timestamp.
- Resolve equal-distance ties using the earlier frame.
- Preserve chronological order.

Save the selected source frame indices and actual timestamps.

These uniform sets need not be nested. Consequently, differences between them do not identify the marginal value of adding specific frames.

Use one fixed decoding strategy across budgets. For this pilot, use sequential decoding of the full clip while retaining only required frames. Avoid storing all full-resolution decoded frames in RAM.

Count separately:
- Frames decoded.
- Frames selected.
- Frames passed to the processor.
- Frames actually represented in the model input.

Spatial preprocessing:
- Preserve aspect ratio.
- Apply the fixed per-frame pixel limit.
- Use dimensions compatible with the model's spatial patch/merge requirements.
- Compute and freeze the resulting height and width for each video.
- Keep those dimensions identical across all budgets for that video.
- Pin interpolation, color conversion, and normalization behavior.

Prevent total-video pixel limits from silently lowering per-frame resolution when B increases.

Pass selected frames through the native video-input path with correct temporal metadata.

Explicitly disable any unintended second frame sampling or resizing step. Inspect the pinned processor implementation and verify its actual behavior.

For variable-frame-rate videos, use faithful timestamp handling. If the selected adapter cannot represent their timing correctly, exclude them before cohort freeze and record that limitation. Do not invent a constant FPS.

Record and validate:
- Requested and actual frame counts.
- Temporal grouping or padding.
- Processed tensor shapes.
- video_grid_thw or the equivalent model metadata.
- Actual visual-token count.
- Total input-token count.
- The timestamps represented to the model.

Do not silently truncate prompts or video inputs.

Verify that repeated preparation produces identical selected frames, metadata, and model-input tensor hashes. Run expensive integrity hashing outside the primary latency interval, while accounting for its execution cost separately.

============================================================
7. PROMPT, GENERATION, AND ANSWER SCORING
============================================================

Use one fixed multiple-choice prompt, for example:

    Choose the best option for the question.
    Reply with one letter only: A, B, C, or D.

    Question:
    [original question]

    [original options in original order]

Keep the prompt, chat template, generation limits, and stopping rules fixed across frame budgets.

Use greedy generation:
- do_sample=False.
- num_beams=1.
- max_new_tokens=8.
- Fixed resolved EOS and padding settings.
- No requested chain-of-thought.
- No conversation history from previous questions.

Freeze a deterministic parser before collection.

It may accept:
- A single option letter.
- An optional "Answer:" prefix.
- Optional simple terminal punctuation.

Reject explanations, multiple candidate letters, and ambiguous outputs. Parse only newly generated output, never the prompt.

Save raw generated token IDs, decoded output, parsed answer, validity flag, output-token count, and correctness.

Invalid or empty answers count as incorrect. Do not ask another model to judge them. Do not retry until a correct or parseable answer appears.

The primary accuracy uses repetition 1 only. Repetitions 2 and 3 measure repeatability and latency variability.

Do not select the best repetition or treat repeated predictions as additional independent examples.

Also run one text-only diagnostic:
- The same 24 questions and options.
- The same answer instruction.
- No video.
- One repetition.

Report this separately to show how much can be answered without visual input. It is not a zero-cost point on the video frame-budget curve.

============================================================
8. PRIMARY LATENCY MEASUREMENT
============================================================

The primary cost is resident-model, local-file, single-question request latency.

Model weights are already loaded when a request begins. Ordinary OS file caching is allowed and must be disclosed. Do not claim cold-disk latency.

Use a monotonic host timer such as perf_counter_ns.

Before starting the timer:
- Ensure previous GPU work is complete.
- Reset per-request measurement state.

Start timing before opening/reading the compressed video for that request.

Include:
- File opening and reading.
- Actual video decoding.
- Frame selection.
- Resizing and preprocessing.
- Prompt construction and tokenization.
- CPU-to-GPU transfers.
- Vision encoding.
- Language-model prefill.
- Answer generation.
- Output transfer to CPU.
- Text decoding and deterministic answer parsing.
- Final GPU synchronization before the timer stops.

Ground-truth scoring, result serialization, and diagnostic hashing may be outside the request timer. State this explicitly and include their time in overall execution accounting.

Each question is an independent request:
- Decode and prepare the video again.
- Do not reuse decoded frames or visual features across questions.
- Do not reuse prefix or KV caches across requests.
- Normal KV caching within one generation call is allowed.

Do not call empty_cache inside the timed interval.

Report model loading, warmup, dataset preparation, and total experiment time separately. Excluding startup from request latency must not make startup disappear from the resource ledger.

Collect a small, separate component-profile pass on one prespecified question per budget. Distinguish CPU preparation, transfer, and model execution where reliable.

Keep intrusive profiling and extra synchronization out of the primary timed pass. Do not add overlapping component measurements and present the sum as a disjoint decomposition.

The synchronized host end-to-end timer is the primary latency measurement.

============================================================
9. MEMORY MEASUREMENT
============================================================

Record per request:
- GPU allocated memory before the request.
- GPU reserved memory before the request.
- Peak GPU allocated memory.
- Peak GPU reserved memory.
- Clearly labeled incremental peaks above baseline.

Use PyTorch peak-memory counters and reset them before each measured request.

Report absolute peaks including resident model weights. Do not present only activation increments as the memory required to run the model.

PyTorch allocated/reserved memory is not the same as total device memory.

Where supported, additionally collect:
- Sampled process/device GPU memory through NVML.
- Sampled process RAM usage.

Label sampled maxima as sampled observations, with their sampling interval. If a metric is unavailable, record unavailable rather than zero.

Use a fresh worker process for each frame-budget × repetition block, with only one GPU worker active at a time. This avoids a previous larger-budget run contaminating smaller-budget reserved-memory measurements.

Record each worker's model-loading time and memory baseline.

Release request tensors between calls. Check for unexplained growth in live allocated memory. Do not mistake the allocator's retained reserved memory for a live-tensor leak.

============================================================
10. EXECUTION PLAN AND RESUMPTION
============================================================

The main planned collection is:

    8 videos × 3 questions × 5 budgets × 3 repetitions
    = 360 measured video-QA calls.

The text-only diagnostic adds 24 calls.

Warmups, fit probes, component profiling, and any other model calls are additional. Count them explicitly.

Execution sequence:

A. Create the repository and implementation.

B. Run CPU-level validation and inspect the environment.

C. Acquire and validate the pilot sources; freeze the cohort.

D. Resolve the model/software configuration.

E. Run bounded fit and timing probes, including Uniform-64.
   Do not use probe accuracy to tune the configuration.

F. Estimate the full remaining execution cost.
   Include all worker startups, two warmup calls per worker,
   collection, controls, profiling, and final validation.
   Apply a 1.5× safety factor to measured runtime estimates.

G. Save the resolved protocol and configuration hashes.
   Start collection only if the projected run fits the remaining limits.

H. Run the frozen collection with saved, seed-determined ordering.
   Rotate/randomize budget block order across repetitions.
   Keep order independent of predictions and measured performance.

I. Analyze and verify the collected records.

Use two fixed warmup calls per worker, based on prespecified pilot inputs. Save their identities and cost; exclude them from primary accuracy and latency summaries.

Use a stable trial identity containing:
- Experiment/run ID.
- Configuration hash.
- Video/question ID.
- Frame budget.
- Repetition.

Persist completed records atomically and preserve failed attempts.

Resume must:
- Validate model, environment, manifest, and configuration hashes.
- Skip completed valid trial identities.
- Preserve prior failures and resource consumption.
- Detect duplicate records.
- Continue the same execution ledger.

A changed model, precision, cohort, resolution, prompt, parser, or measurement boundary requires a new named run/version. Never overwrite the previous run to conceal the change.

Handle interruption, timeout, decoding failure, and OOM explicitly. Do not silently drop failed questions from quality or cost reports.

If collection is incomplete, report planned, attempted, successful, failed, and unattempted counts. Do not present a complete quality–cost curve from a selectively successful subset.

============================================================
11. ANALYSIS AND OPERATIONAL GATES
============================================================

For each frame budget report:
- Correct answers out of the fixed 24 questions in repetition 1.
- Accuracy and invalid-answer rate.
- Paired answer changes relative to Uniform-64.
- Actual visual and total input tokens.
- Generated-token counts.
- Median request latency and spread.
- Peak allocated and reserved GPU memory.
- Sampled device/RAM maxima when available.
- Failures and missing measurements.

For timing aggregation:
1. Compute the median over the three repetitions for each question/configuration.
2. Report the distribution over those 24 question-level medians.
3. Also report per-video summaries.
4. Define every aggregate and speedup formula explicitly.

Report measured latency ratios relative to Uniform-64 separately from frame-count ratios.

Do not assume that 16 frames cost one-quarter as much as 64 frames.

Repeatability checks:
- Identical prepared inputs for repeated identical requests.
- Parsed-answer agreement across repetitions.
- Raw output/token agreement as an additional diagnostic.
- Timing variability for every question/configuration.

For each question/configuration define:

    relative timing range =
        (maximum latency - minimum latency) / median latency

Use a per-budget median relative timing range of at most 20% as a proposed operational stability target. Flag individual cells above 50%.

These are practical diagnostic thresholds, not statistical significance tests. Do not selectively rerun slow trials until they pass.

A fully confirmed operational result requires:
- Frozen, traceable configuration and sources.
- Correct actual input budgets and temporal metadata.
- Complete required collection.
- Reproducible prepared inputs.
- Stable parsed answers under the fixed deterministic setup.
- Interpretable timing and complete memory records.
- Reconciled resource and model-call accounting.
- Successful independent verification of saved records.

If timing or answer repeatability is unresolved, preserve results and report MEASUREMENT_UNSTABLE or REPEATABILITY_UNRESOLVED.

Do not require accuracy to increase monotonically with frame count. A lower-budget condition may score better on this small pilot.

Give a separate assessment of task quality:
- Does the model produce usable answer formats?
- How does video performance compare with text-only?
- Are errors or invalid answers concentrated in particular questions?

Operational feasibility does not establish useful visual reasoning or a research contribution.

There are only 24 questions grouped into eight videos. Repetitions do not increase that sample size. Avoid publication-level generalization, significance claims, or treating this pilot as a powered comparison.

============================================================
12. VALIDATION, OUTPUTS, AND FINAL REPORT
============================================================

Implement targeted checks for real validity risks:
- Timestamp sampling and tie handling.
- Actual processor frame counts and spatial dimensions.
- Temporal metadata correspondence.
- Label exclusion from inference inputs.
- Parsing generated output without reading letters from the prompt.
- Invalid answers and failure accounting.
- Resume consistency and duplicate prevention.
- Correct synchronized measurement boundaries.
- Reproduction of summaries from saved raw records.

Synthetic tests validate code paths only. They do not establish that the real video model ran successfully.

Save under a uniquely identified results directory:

    resolved_config.yaml
    environment.json
    model_manifest.json
    cohort_manifest.json
    exposure_manifest.json
    execution_order.json
    raw_trials.jsonl
    failed_attempts.jsonl
    resource_ledger.json
    summary.csv
    summary.json
    verification.json
    status.json
    report.md
    figures/

Include:
- Accuracy versus measured request latency.
- Peak GPU memory versus frame budget.
- A small quality/latency/memory summary table.

The report must explain:
- Exact task and input configuration.
- What "Uniform-64 reference" means.
- Included and excluded request costs.
- Setup and total execution costs.
- Actual completed call counts.
- Reproducibility and timing stability.
- Failed gates and limitations.
- The next evidence-supported action.

Provide one CLI with documented subcommands, such as:

    python -m video_compute_value.e000 preflight
    python -m video_compute_value.e000 prepare
    python -m video_compute_value.e000 smoke
    python -m video_compute_value.e000 run
    python -m video_compute_value.e000 resume
    python -m video_compute_value.e000 analyze
    python -m video_compute_value.e000 verify

Implement the commands you document. Provide their exact configuration/path arguments in the README.

Use explicit final statuses, as appropriate:

    FEASIBILITY_CONFIRMED
    BLOCKED_ENVIRONMENT
    BLOCKED_MODEL_ACCESS
    BLOCKED_INPUT
    BLOCKED_RESOURCE
    INPUT_INTEGRITY_FAILED
    MEASUREMENT_UNSTABLE
    REPEATABILITY_UNRESOLVED
    PARTIAL

If blocked, still deliver the usable repository, completed checks, saved evidence, and exact next command or required resource. Clearly identify which real-model paths remain unverified.

Your final response must state:
1. Where the repository was created.
2. What was implemented.
3. Which commands actually ran.
4. Whether real Qwen inference occurred.
5. Planned versus completed experiment calls.
6. The measured summary table, if available.
7. Gate results and overall status.
8. Consumed and remaining resources.
9. One concrete next action.

Never fabricate predictions, timings, memory measurements, or successful execution.

Primary implementation references:
- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- https://github.com/QwenLM/Qwen3-VL
- https://huggingface.co/docs/transformers/model_doc/qwen3_vl
- https://github.com/MME-Benchmarks/Video-MME
- https://docs.pytorch.org/docs/stable/notes/cuda.html

Record the exact revisions and relevant documentation used.