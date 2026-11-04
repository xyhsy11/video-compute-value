# E0 runtime recovery v6

The 2026-11-04 user instruction to solve the saved initialization failure
authorizes this separately named, bounded operational follow-up. The failed
2026-10-29 timing run, all E0/E1/E2 evidence, and E0's historical
`MEASUREMENT_UNSTABLE` verdict remain unchanged.

Reuse the v5 model adapter, local model/processor/tokenizer revision, packages,
BF16/SDPA, four CPU/decoder threads, inputs, prompt, greedy controls, parser,
timer, cache trimming, monitoring, schedules, and decision rules byte-for-byte.
No package/driver replacement, system repair, other model, quantization,
offloading, acquisition, E2B or E3 is included. The current loader is tested
before considering any changed loading implementation. Failure stops the run;
no partial worker is replaced or retried. Enable Python's native-fault trace
for any new failure without changing model computation.

The first stage consists of the original 16 native CPU preparations with exact
historical tensor/temporal and serial/four-thread RGB agreement. It then runs
two bounded fresh preflight workers, 8→64 and 64→8: eight warmup calls and four
intrusive profiles. These 12 calls test native initialization and the real GPU
path. After complete-stage admission, run the unchanged nine-worker fresh versus
resident screen: 24 warmups and 96 primary requests. Thus 132 calls are planned,
not silently substituted for the missing historical calls.

For each source question, arm and budget, use (max−min)/median over three
independent sessions. An arm qualifies only when the median over all eight
source questions is ≤20% at both 8 and 64 frames, with every integrity gate
passing. Select A first, then B; otherwise stop. No tuning or third arm.
The unchanged v5 full-confirmation protocol is allowed once only after this
rule passes and its entire next stage fits the budget. A new confirmation
directory and fingerprint are required. No favorable session may be selected.

The cumulative limit is the already authorized 172,800 seconds, not an extra
grant. Inherit the complete E2 ledger and the separately recorded VidFit
presentation charge exactly once. Preserve their parent hashes and session/event
prefixes. Count implementation, failed attempts, model loading, CPU/GPU checks,
reporting and publication. Retain the 20 GiB storage and 1 GiB acquired-video
caps, 1.5 safety factor, 120-second contingency and 600-second protected closeout.
The v6 verifier uses the amended cap from this frozen configuration and checks
its grant and supplemental ledger lineage. All independent scientific checks
from v5 are retained.

A successful load establishes current runtime availability. It cannot establish
the cause of the old native crash or retroactively fill its missing observations.
Report initialization, timing qualification and confirmation separately, including
negative results. Raw questions, labels, answers, tokens and logs stay private.
