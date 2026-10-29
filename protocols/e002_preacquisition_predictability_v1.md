# Experiment 2: pre-acquisition benefit predictability

This is the prospective design for `e002_preacquisition_predictability_v1`.
Implementation is a blocked checkpoint until the gates below pass. An implementation
fingerprint is not a scientific cohort freeze. No E2 outcome has been collected.
The user's 2026-10-29 request supersedes the old E2 prohibition only for this study.
It does not enlarge the cumulative 27,600-second, 20-GiB storage or 1-GiB video
traffic grants. All prior charges, failures and restricted artifacts are inherited.

## Scientific scope

E2A asks whether cheap observations predict which of four acquisitions improves
answer correctness on held-out sources. E2B separately asks whether a stop/acquire
policy improves accuracy within the complete Direct Uniform-16 latency budget.
A successful E2A does not establish E2B. Missing/invalid timing makes practical
value not evaluable. E0 remains `MEASUREMENT_UNSTABLE`; E1 remains a retrospective
eight-source headroom diagnostic. This is recorded-video one-step QA, with a base
observation spanning the full timeline. It is not future-blind streaming or causal inference.

Keep Qwen3-VL-4B-Instruct model, processor and tokenizer revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`, BF16/SDPA, native single-video input,
192 x 320 letterboxed canvases, original timestamps, original question/options,
strict-option-v1 parser and explicit-greedy-v2. Require use_model_defaults=False,
one beam, eight maximum new tokens and resolved EOS/PAD/BOS. One CUDA worker;
no quantization, offload, audio, captions, framework/driver changes or extra model.
Historical shared sources and all E0/E1/v5 evidence remain byte-for-byte unchanged.

## Cohort and splits

Use 40 new independent source-video groups and all three original questions each
(120 total) from the official Video-MME annotation test split, short category,
revision `ead1408f75b618502df9a1d8e0950166bf0a2a0b`. This is an E2 development
pilot, not an untouched confirmation set or official leaderboard submission.
Exclude the eight exposed sources, identical assets, known alternate encodings,
reuploads and shared parent sources. Source hashes alone cannot exclude all
reuploads. Save provenance uncertainty and the complete inclusion/exclusion log
privately. Exactly 40 verified groups are required; no smaller substitute is authorized.

Order candidate source identifiers by SHA256 UTF8 `2026102902:candidate:source`,
with the identifier as a tie break. Eligibility is label independent: original
three questions, supported zero-origin CFR timeline, distinct quarter additions,
source independence, authorized access and resources. No outcome-based replacement.
Local metadata inventory does not certify eligibility or independence.

Order groups by SHA256 `2026102902:outer:source` and assign round robin to five
outer folds (eight groups each). Four inner folds on the 32 training groups use
the same construction with salt `inner:outer_fold`. All questions/actions/repeats
stay with their source. All scaling, alpha/threshold selection, baseline choice,
imputation and cost estimates use only outer training sources. Select one repeat
source per test fold with salt `repeat:outer_fold`, before outcomes.

## Evidence and observations

Base is the exact Uniform-8 input. Q1–Q4 retain those eight and add eight distinct
frames from one temporal quarter using the frozen E1 nearest-unused-bin-center
rule. The nested-uniform control uses that rule over the full timeline. Direct
Uniform-16 is constructed independently. Verify RGB retention, original PTS,
native temporal pairing, 16 distinct frames, 192 x 320 dimensions and matched
visual tokens for all augmented conditions. Record text-token differences.
Matched visual tokens do not prove matched latency/FLOPs. No candidate tensors,
discarded-frame statistics, encoder/prefix/KV reuse or cached base answers.

The strict AllowedObservation contains only eight retained uint8 RGB canvases,
their PTS, duration, frame count, aspect ratio, known content crop and original
question text. Source IDs, paths, filenames, labels, options, categories, question
IDs/order, candidate outputs/costs and outside-base pixels are rejected.

The exact executable feature schema is `e002_features.SCHEMA` and is exported to
`feature_schema.json`. Crop only the known letterbox rectangle. Pillow RGB-to-L,
BILINEAR 32 x 32, float64 /255. Exactly two base observations per quarter must be
verified from PTS. Per quarter compute mean population spatial standard deviation,
mean of horizontal and vertical absolute finite differences, between-frame mean
absolute difference, and fraction above 0.10. Add log1p duration, log1p frame count
and source width/height: 19 V values. These are appearance/change proxies.

Q normalizes NFKC then casefolds, tokenizes Unicode `[^\W_]+`, hashes UTF8 unigrams
with SHA256, takes big-endian first-eight-byte integer modulo 64 and sign +1 for
even byte 8, else -1, and L2 normalizes the 64 counts. Add log1p unigram count and
normalized character count. No fitted vocabulary or answer-option text. A fixed
projection hashes `2026102902:projection:index` with the same rule into 16 values.
VQ concatenates 19 V, 66 Q, and the flattened 16-by-16 outer product between the
quarter visual values and projected query: 341 values. No learned projection.

C is from return_dict_in_generate/output_scores in the SAME paid base greedy
generation: first generated token's full-vocabulary top1 probability, top1−top2,
and entropy/log(vocabulary size). No extra scoring/teacher forcing or confidence
question. Verify exact plain-versus-observed generated token agreement before
collection and for all 120 questions. Invalid/valid and output-token count are
response-shape diagnostics, not confidence. C is not an answer correctness
probability or A/B/C/D probability. VQ does not use C or pay C extraction overhead.
If capture is unsupported or changes tokens, block feature integrity.

## Targets and fitting

For each action target `correct_augmented - correct_base` is -1, 0 or +1.
Keep harms, neutral, always-correct/wrong and invalid completed answers (zero).
Failed/incomplete calls are missing and block completeness. Four independent
ridge outputs use float64, training-only population mean/std (zero std -> 1),
unpenalized intercept, and per-output objective `sum(residual²)/n + alpha*sum(w²)`.
The dual solve therefore uses `n*alpha`. Alpha candidates are 1,10,100, selected
by mean inner OOF raw-prediction gain MSE. Ties within 1e-12 choose larger alpha.
Clip decision predictions to [-1,1], take greatest gain, ties Q1 then Q2/Q3/Q4.
Train V, Q and VQ. A shared additive question term cannot establish conditioning.

Forced selection always acquires. Stop/acquire uses strictly greater than threshold;
grid: always_acquire,0,.025,.05,.10,.20,.40,.80,always_stop. Special endpoints force
the declared action regardless of scores. Never override an invalid stop using a label.
Seal held-out predictions and actions before evaluator joins with outcomes.

E2A compares base, exact expected random over four actions, training-selected fixed
quarter, V/Q/VQ, nested uniform, Direct16, hindsight oracle and oracle-plus-stop.
The oracle is nondeployable. Primary Delta_fixed = VQ−train-fixed; question contrast
Delta_question = VQ−V. Report all source/fold effects, rescues, harms, action counts,
gain MAE, regret, calibration bins, and VQ versus other controls.

Advancement requires complete verified evidence, Delta_fixed >=.05, positive effect
in >=4/5 folds and positive descriptive paired source-bootstrap lower bound.
Only a positive Delta_question interval lower bound demonstrates explicit question
contribution. Otherwise report not demonstrated. Failure is
`PREDICTIVE_VALUE_NOT_DEMONSTRATED_FOR_THIS_OBSERVATION_FAMILY`; if meaningful
improvement remains compatible, add `INCONCLUSIVE_LOW_PRECISION`. A blocked
experiment has unavailable results, not this negative scientific verdict.
Use 10,000 source-cluster resamples, seed 2026102902, holding OOF fits fixed.
These descriptive conditional intervals omit some split/training uncertainty.
Five overlapping-training folds are not independent replications. A negative
result does not test all cheap or semantic representations.

## Practical stage

Require a separately frozen single legacy 8/16 timing screen, followed by a
prospectively selected training-only calibration/repeatability panel for each
outer fold, all with three independent sessions and median relative-range <=.20.
Failing any timing gate blocks cost fitting/E2B, not a valid cost-free E2A.
Do not screen repeatedly until timing passes. Cost qualification and exact screen/
panel membership, warmups, routes and call counts must be frozen before outcomes.

Training reference B is Direct16 mean complete cost on training sources. Select
VQ/V thresholds with highest inner OOF accuracy subject to estimated mean cost <=B,
then lower cost, then more stopping. No feasible candidate means infeasible; do
not change B. C acquires below thresholds 0,.25,.50,.60,.70,.80,.90,.95,1 plus
always endpoints, using inner-training best fixed quarter. B* is training-selected
among direct8/direct16/direct Q1–Q4/nested uniform and that confidence gate.
Direct methods need one request, with no invented base pass. Cost estimates may
sum qualified training components but final costs require actual full executions.

Evaluate VQ, V, Direct16 and fold-specific B* in three fresh-process sessions,
model resident within session, frozen balanced request/method order, pass1 accuracy,
passes2/3 exact input/output/decision repeatability and timing. Deduplicate only
prospectively fingerprint-identical B*/Direct16. Adaptive calls cost base plus
required observation/selection plus full 16-frame follow-up when acquired. C only
computes C; VQ only computes VQ. An outer synchronized monotonic timer includes
opening/complete decoding, native processing, transfer, all inference, feature/
selection work and output parsing. Hashing/scoring/disk serialization are outside
deployment time and inside project accounting. No cross-request frame/feature cache.

Every required method must pass median question-level (max−min)/median <=.20 across
three sessions. Average repetitions per question, then questions per source and
sources equally, using arithmetic means for cost. Report medians/p95 separately.
Practical success requires positive paired accuracy interval lower bound and
latency ratio interval upper bound <=1 against BOTH Direct16 and B*, all integrity
gates, and observation-plus-selection mean overhead <=10% of Direct16 mean latency.
Otherwise report the accuracy/cost tradeoff, budget infeasibility, or
`PRACTICAL_VALUE_NOT_EVALUABLE_MEASUREMENT`. No frame-count efficiency surrogate.

## Execution and checkpoint limits

Primary counterfactual collection has eight conditions per question: plain base,
observed V/Q/C base, Q1–Q4, nested uniform, Direct16 =960 calls. Repeat all eight
on one frozen source per outer fold (15 questions) in a fresh process =120 calls.
Exact tokens/features repeat at rtol0/atol1e-12. Final live evaluation is up to
1,440 pipeline executions and up to 2,520 model invocations before identical
pipeline deduplication, PLUS timing panels, warmups, probes and failure allowance.

Only admit a whole next frozen stage using valid measured probes, factor1.5 and
explicit failure/verification/report/publication reserves. Unmeasured planning
assumptions never admit collection. Preserve immutable identities, failed attempts,
start/completion receipts and every charge. At most one recovery worker per stage,
two starts per identity, six failed requests; native loader crash stops the stage.
One bounded 90-second legacy compatibility attempt is permitted if 435 seconds
(1.5*90+300 reserve) remain. Freeze its calls before launch: two base warmups on
the first legacy source's first two questions, then plain base, observed base and
Q1 on its first question. No automatic retry after native failure.

Current CLI implements audit, preflight, resume, verify, report, account and close.
Full scientific cohort acquisition/freeze/collection orchestration, complete OOF
artifact/evaluator integration, cost-panel/B* orchestration and final three-session
collection are not yet implemented. The CPU core includes features, ridge fitting,
decision/threshold logic, analysis and real pipeline primitives; do not describe
this checkpoint as a completed E2 implementation or experiment. Finish missing
orchestration under an admitted resource grant and a new implementation fingerprint
before scientific freeze. Keep all feature matrices, trained models, raw responses,
text, labels, paths and detailed manifests private. Publish only allowlisted code,
sanitized aggregates/figures and an honest incomplete report. No E3 or paid services.
