# Experiment 0 execution addendum v2

Candidate execution policy: `threaded-decode-trim64-v2`, configured in
`configs/e000_threaded_v2.yaml`. The scientific cohort, all 24 original questions,
order/seed, frame budgets, selected PTS, RGB conversion, bicubic letterbox, model,
precision, prompt, parser and accuracy denominators remain those of Experiment 0.
A fresh named run and inherited cumulative ledger are mandatory.

Primary requests use full sequential PyAV decoding with exactly four codec threads
and FRAME threading. Eligibility inspection remains serial. There is no seeking,
frame skipping, persisted decoded-video cache, feature cache, prefix cache or
cross-request KV cache. Each request reopens and completely decodes its source.
Acceptance requires exact uint8 RGB comparisons and native processor tensor/hash
comparisons against the original serial implementation and saved parent inputs.
Threading performance is a CPU diagnostic until a new full GPU run is validated.

Only Uniform-64 calls `torch.cuda.empty_cache()` after input transfer and before
generation. This explicitly supersedes the original v1 prohibition on in-timer
cleanup for this separately versioned policy. Its full cost stays inside the
original end-to-end timer. Other budgets, including text-only, do not trim. The
choice follows the earlier high-frame timeout and lower-frame trim regression;
it is independent of accuracy. No offload, quantization or image reduction.

The request timer still starts before opening the local compressed video and ends
after generation, CPU output decoding/parsing and final CUDA synchronization.
Two nested host-clock reads measure preparation without additional CUDA fences.
Separate synchronized component profiles remain separate from primary trials.
A request's generation-policy digest and actual trim/decoder settings are saved.
Worker receipts, independently resolved effective greedy configuration and exact
raw-token/output equality are required before feasibility can pass.

A full corrected run requires 360 video requests, 24 text-only requests, 32
collection warmups and five fresh fit workers with five calls each: 441 total
model invocations before retries, in 21 fresh model workers. Include eight CPU
preparation probes, model-file verification, tests/reporting, a bounded 600-second
retry contingency and the existing 1.5 safety factor in the affordability check.
The 600-second per-worker deadline preserves timeouts rather than replacing them.

Screening forecasts must be labeled estimates; they cannot substitute for missing
accuracy, latency or memory observations. Intermediate GPU costs remain unverified
until real preflight. If the full forecast exceeds the remaining four-hour budget,
stop before new GPU work and request an explicit cumulative-limit increase. The
original results and ledgers are immutable. New completion work is local and
unstaged; the human user reviews and publishes it.
