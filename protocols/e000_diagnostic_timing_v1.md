# Matched timing phase, frozen after the generation-default audit

The generation audit reproduces `do_sample=False` being overridden by the Qwen
checkpoint defaults in Transformers 4.57.1. All timing arms explicitly use
`use_model_defaults=False`, `do_sample=False`, one beam and eight new tokens, with
resolved EOS/padding/BOS tokens. Precision, native SDPA, model/data revisions,
preprocessing, source questions and the full-request timer otherwise remain fixed.

Compare native allocator settings with
`garbage_collection_threshold:0.75,max_split_size_mb:128`. This tests cached-memory
pressure without explicit model offload or manually hiding cache-cleanup time.
Automatic allocator work remains inside the original request boundary.

Uniform-64 uses all 24 parent questions, to retain the prompt-shape diversity that
can fragment its allocator. Uniform-8 uses the first official question from each of
the eight sources as the low-visual-compute comparison. Both arms collect three
repetitions in paired blocks with alternating arm order. Every block is a fresh
worker with two fixed warmups. Four separate component profiles are additional.
Totals: 192 timed requests + 24 warmups + four component profiles = 220 model calls.

No decoded-frame or feature reuse in timed requests. Two host-timer reads record a
nested CPU-preparation span without extra primary CUDA fences. Request-minus-CPU
time is a remainder that also includes transfers/output processing, not pure GPU
execution. Separate profiles provide synchronized preparation/transfer/model spans.
Telemetry and hashing are outside the timed interval and inside overall accounting.

This is a targeted diagnostic, not a replacement 24-question/five-budget quality
curve. Predictions and correctness are not used for selection or retries. Every
attempt is preserved; all completed records must reproduce before reporting.
The parent-observed cost forecast includes starts, warmups, profiles, extra diagnostic
overhead and a 1.5 safety factor, with a final verification reserve. Source hashes,
order, input fingerprints and actual allocator settings are frozen and checked.
