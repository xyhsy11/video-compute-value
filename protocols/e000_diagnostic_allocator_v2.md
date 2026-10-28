# Matched diagnostic v2: native allocator and in-request trimming

Named run: `e000_20261028_diagnostic_v2`. Inherit every prior run's resource
consumption; preserve `e000_20261027_diagnostic_v1` including the stopped allocator
arm and shutdown evidence. The original collection remains immutable.

The prior `max_split_size_mb:128` intervention reserved 20,514,340,864 bytes and
needed 144.346 seconds for a completed warmup; a retry was interrupted after more
than 160 seconds. It is not an accepted fix. Reduce the follow-up before collecting
primary timings: first official question per each of eight videos, Uniform-8 and
Uniform-64, three repetitions, two paired arms. Selection uses identifiers/order
only. Twelve fresh workers, two warmups each, four separate profiles: 96 timed
requests plus 28 auxiliary calls. No correctness-based retries or selection.

Both arms use the default native allocator and the corrected production greedy
adapter. The trim arm calls `torch.cuda.empty_cache()` after input transfer and
before generation INSIDE the original synchronized request timer. No offload,
quantization, decoded-frame reuse, feature/KV reuse or extra synchronization fence.
The cleanup host span is descriptive; full latency includes any driver cost.
Nested CPU preparation timing excludes cache trimming. Ordinary OS caching remains.

Freeze source hashes, effective generation policy, exact paired questions, order
and parent input hashes. Compare answer/token consistency, question-wise three-run
median latency, relative range `(max-min)/median`, sampled device usage and PyTorch
allocator statistics. Report >20% timing range failures honestly (the target is <=20%). No replacement
five-budget accuracy curve or adaptive-computation claim.

The cost forecast sums exact parent request costs, same-budget auxiliary costs,
100 seconds startup/serialization per worker and 0.5 seconds extra per call, then
multiplies by 1.5. Check forecast against remaining cumulative time before every
worker, reserve 240 seconds for verification, and cap a worker at 600 seconds.
