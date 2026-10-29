# Experiment 1: Is there useful headroom for choosing an additional clip?

Experiment ID: `e001_additional_clip_headroom_v1`. This prospective local freeze
is for a retrospective development-data diagnostic, not external preregistration.
The timestamp and immutable hashes are saved before any E1 inference. E0 remains
`MEASUREMENT_UNSTABLE`; E1 does not waive or repair its timing gate.

We reuse the eight exposed Video-MME short sources and all three original
questions per source (24 total), from the official annotation test split, revision
`ead1408f75b618502df9a1d8e0950166bf0a2a0b`. This is development data, not untouched
confirmation data. There is no outcome-based selection or replacement.

The model, processor and tokenizer are Qwen/Qwen3-VL-4B-Instruct revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`, BF16, SDPA, frozen weights, one CUDA
worker, no offload or quantization. We inherit E0's installed software, four CPU
threads, FRAME/4 complete sequential decoding, 192 x 320 per-video letterbox,
native video processor (resampling/resizing disabled), prompt, strict-option-v1
parser, inference seed 20261027, explicit greedy-v2 controls, one beam, eight
maximum new tokens, and resolved EOS/padding. E0's trim-only-at-64 rule is retained
and never activates at E1's 8/16 frames. No audio, subtitles, transcripts or captions.

For each source, base evidence is the exact E0 Uniform-8 indices, original PTS,
dimensions and processed RGB pixels. Four candidate acquisitions correspond to
quarters [0,D/4), [D/4,D/2), [D/2,3D/4), [3D/4,D]. For eight chronological bin
centers per interval, choose the nearest eligible presentation timestamp, excluding
base indices and previously selected additions; ties choose earlier PTS. Save the
nearest unconstrained frame and displacement from exclusions. Insufficient distinct
frames makes the source ineligible and blocks the whole fixed cohort. Chronologically
merge eight additions and eight unchanged base frames. The uniform-addition control
uses the same rule over [0,D]; historical Uniform-16 is not this nested control.
All five augmented inputs must contain 16 distinct frames and equal visual-token
budgets. Save text-token differences from timestamp tags. Preserve original video
times through the native VideoMetadata adapter; verify nonuniform pair times and
actual tensor grids. Added frames can change temporal pairs, timestamp tags, token
positions and contextual representations. This is retained visual evidence plus
an acquired clip followed by full re-encoding, not cached incremental inference.

Freeze all six conditions, two passes, the order and two fixed warmups before
collection. A seeded Python Random(20261029) independently shuffles all 144
source/question/condition identities in each pass. Each pass has a fresh process
and two warmups: first source, first two questions, base condition. Each request
starts from its original question and visual input. No previous prediction or
cross-request frame/feature/prefix/KV cache is used. Each request fully decodes its
source again; ordinary OS file caching and within-request generation cache are allowed.

CPU preflight prepares all six inputs for the first question on every source,
checks retained RGB evidence, matches each base tensor to corrected E0, and checks
matched augmented budgets. Two bounded fresh GPU probe workers each run two fixed
warmups plus all six conditions of the first source/first question. They must match
the CPU inputs and each other's inputs/tokens/raw outputs/parsed answers, and record
effective greedy receipts and absolute memory peaks. Probe outcomes do not enter
accuracy or select conditions. Forecast the 288 collection requests, four warmups,
two startups and record overhead from real probes and all-source CPU preparations,
including explicit verification (300s), retry (180s) and publication (240s) reserves,
with a 1.5 safety factor. Stop if it does not fit the current cumulative ledger.

Primary accuracy uses pass 1 only. Pass 2 repeats all 144 requests and checks
inputs, tokens, raw outputs and parsed answers in a fresh process. No majority vote
or favorable repetition selection. Invalid/empty completed answers score zero.
Runtime failures are missing, never incorrect or silently dropped. Preserve every
started call, failed attempt and timeout. At most one recovery process per pass,
two attempts per trial identity and six failed trial attempts globally are allowed.
Every recovery process repeats both warmups; completed answers are never retried.
Probe failure stops preflight. A scientific source/configuration change requires a
new named run. Resume only identical fingerprints and valid atomic completed records.

Let b_i be base correctness, r_ia the correctness of quarter a in {1,2,3,4}, and
u_i the uniform-addition correctness. With every one of the 24 questions retained:

- A_base = mean(b); A_oracle = mean(max_a r); A_random = mean(mean_a r).
- A_best_fixed = max_a mean(r); A_uniform = mean(u).
- Primary H_choice = A_oracle - A_random.
- Secondary differences: oracle minus best fixed, uniform addition and base.
- A_oracle_stop = mean(max(b, max_a r)); stopping makes its gain nonnegative by construction.

The forced-acquisition oracle chooses only one of four quarters, excluding stopping
and uniform addition. Random is the exact expected accuracy, not a sampled action.
Best fixed is the globally best quarter selected retrospectively on these data.
Report raw counts/denominators, every fixed-quarter accuracy, base-wrong rescues,
base-correct questions harmed by every quarter, average action-induced harm
mean_i mean_a[b_i=1 and r_ia=0], and mixed-success questions/videos. Keep always
correct, always wrong and invalid answers in the denominator. Labels enter only
offline scoring and retrospective analysis, never construction or inference.

Freeze the exploratory advancement rule: oracle minus random >=10 percentage
points, oracle minus best fixed >=10 points, oracle strictly exceeds both base and
uniform addition, mixed-success questions occur in >=3 distinct sources, and all
collection/input/greedy/repeatability/independent-verification checks pass. These
are engineering judgments, not established statistical standards. Do not tune them.
Passing: **Promising development headroom; larger independent evaluation is justified.**
Complete valid failure: **Useful headroom was not demonstrated for this configuration
and action menu.** Missing, resource-blocked or unresolved checks are inconclusive.

Report paired per-video effects and leave-one-video-out sensitivity. Use 10,000
paired cluster-bootstrap resamples of complete videos, NumPy default_rng(20261029),
percentile 95% intervals; keep all three questions/actions together and recompute
best fixed inside each resample. Eight clusters give weak uncertainty estimates.
Repetitions/actions are not additional independent examples; no significance or
population-wide claim. Compare newly recomputed base predictions with corrected E0.

Synchronized full-request timings include opening/complete decode, selection,
letterbox, native processor, transfer, full inference, decode/parse and final CUDA
synchronization. Hashing/scoring/serialization are outside request timing and inside
the project ledger. Record absolute allocated/reserved peaks including weights.
E1 timing is descriptive only; no validated efficiency or latency-saving claim.

The project-wide cap was explicitly amended on 2026-10-29 from 20,400 to 27,600s.
All previous charges remain. Storage is still 20 GiB, including environment,
caches, results and Git, with at most 1 GiB video acquisition. No new downloads,
paid services, unrelated deletion or unapproved app termination. Resource amendments
and generated reports/ledgers do not enter scientific fingerprints. E0 evidence
must remain byte-for-byte unchanged. Questions/options/labels/raw outputs and
restricted manifests/logs remain private. Publish only sanitized statistics,
anonymous correctness or per-video aggregates, code, identifiers and hashes.
