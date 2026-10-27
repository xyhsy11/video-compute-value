# Experiment evidence

`e000_20261027_collection_v2` is the final collection run. It uses aspect-preserving
bicubic resizing and fixed centered black padding to the per-video 32-pixel grid
canvas. The original eight source videos and official question/option order were
retained. Refer to its status, report and independent verification for the outcome.

The earlier `e000_20261027_implementation` and `e000_20261027_collection_v1` directories
preserve 25 development model calls each. These were implementation/fit probes,
not primary quality measurements. Input auditing led to the corrected preprocessing
before any final-run primary collection. Their consumption was inherited into the
final resource ledger; neither a new run name nor a process restart reset the budget.

Settings, source/exposure/question identifiers, hashes, software/hardware evidence,
execution ordering and aggregate reports are safe provenance artifacts. Question
text, answer labels, full cohort manifests, raw trial correctness/predictions,
worker diagnostics, videos and weights stay local and ignored by Git. The raw files
and failed attempts are preserved, including in the earlier development directories.

Synthetic CPU tests and the Windows process-cleanup test are separate validation
evidence. They never count as real Qwen model calls or experimental predictions.
