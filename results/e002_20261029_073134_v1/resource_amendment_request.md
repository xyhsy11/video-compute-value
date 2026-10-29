# E2 resource amendment request

Current cumulative cap: 27,600 seconds. Charged: 27390.965324 seconds.
Remaining: 209.034676 seconds. Storage: 17,049,375,913
of 21,474,836,480 bytes. Charged video transfer plus interruption upper bound:
82,702,430 of 1,073,741,824 bytes.

No additional grant is assumed. The 960 primary +120 repeatability calls do not
include probes, warmups, eligibility, startup, failure, fitting or publication.
The successful compatibility worker supplies measured legacy call durations, but
new-cohort throughput and timing stability remain unqualified. The current forecast
uses the maximum saved legacy request time for 1,080 scientific plus four warmup
calls, two measured model startups, 0.1 seconds per record, and explicitly unmeasured
allowances of 900 seconds input work and 600 seconds training/verification/reporting.
The exact values are in [forecast.json](forecast.json). Factor 1.5, 180 seconds failure
allowance and 300 seconds closeout produce 25687.548 seconds.
Conditional added allowance needed from this snapshot: **25479 seconds**.
This is an exact arithmetic shortfall under stated assumptions, not a measured
completion guarantee. A grant alone does not pass input/integrity/implementation gates.

The first 40 ordered candidate assets have 392,351,143
compressed payload bytes in official metadata; HTTP overhead, retries, provenance,
CFR failures and replacements are unmeasured. No traffic/storage amendment is yet
justified. Reserve 128 MiB for private records/traces; acquire no extra weights.
Do not acquire the cohort until the whole next frozen stage is admitted using
valid measured probes and verified resource estimates. E2B separately forecasts
up to 1,440 live pipelines /2,520 model calls plus screen/calibration/warmup calls.

Missing authority: additional cumulative runtime would require explicit user approval.
Missing evidence: 40 eligible independent new sources and full-cohort input/score
integrity. Legacy runtime/score compatibility status: PASSED. Remaining implementation: scientific cohort/collection orchestration,
OOF sealed-artifact evaluator, training-only cost panels/B* and live three-session runner.
Historical native-loader failures are preserved. This E2 attempt is not repeated automatically.

Checkpoint resume (rechecks fingerprints and preserves a failed probe):

```powershell
.venv\Scripts\python.exe -m video_compute_value.e002 resume --run-dir results/e002_20261029_073134_v1
```

This command does not claim to perform the missing full collection. If compatibility
failed, its preserved receipt stops resume. If it passed, further whole-stage input,
resource and implementation gates still apply before scientific collection.
