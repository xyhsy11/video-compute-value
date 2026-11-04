# Runtime recovery v7: complete resource-verifier migration

The v6 bounded attempt completed two native model loads and all 12 preflight
calls. Its inherited preflight function then invoked the historical verifier,
whose fixed 27,600-second cap rejected the correctly inherited cumulative charge.
This is an implementation migration failure, not a new native initialization
failure, and occurred before any of the 96 primary timing observations.

Preserve v6 completely, including its failed gate, all 12 calls, CPU evidence,
fingerprint and charges. This new named implementation routes preflight through
the already versioned verifier that checks the authorized 172,800-second cap and
resource lineage. No scientific rule, measured operation, model loader, software,
weight, prompt, input, parser, ordering, thread policy or hardware setting changes.

Inherit the v6 and VidFit charges and actual invocation counts. Reuse the 16 v6
CPU preparations only with equal adapter/processor and cohort fingerprints,
unchanged preparation source hashes, original seals and historical tensor/metadata
checks. Keep their original run fingerprints and record explicit reuse provenance;
do not pretend these are newly collected preparation measurements.

Run both fresh preflight workers again under the new implementation fingerprint,
then the unchanged 132-call screen and once-only conditional confirmation specified
in the v6 and v5 protocols. No primary data from another fingerprint is reused.
Every failure stops its phase, with no replacement workers or timing-based retry.
No package, driver, OS repair, extra model, E2B, E3 or resource increase is included.
