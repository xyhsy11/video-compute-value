# Experiment 0

- Implement only frozen visual-only Video-MME feasibility and measurement. No controller training, adaptive sampling claim, or oracle claim.
- Preserve raw trials, failed attempts, configuration snapshots, source hashes, and resource ledgers. Never substitute synthetic predictions for model results.
- Freeze model/processor/tokenizer revisions, software, cohort, preprocessing, prompt, parser, ordering, and timing boundary before collection. Changes require a new named run.
- Resume only matching fingerprints; skip valid completed identities; never erase failed attempts or reset cumulative resource consumption.
- One CUDA worker at a time, no offloading or quantization. Limits: 4 active hours, 20 GiB additional storage including at most 1 GiB acquired videos. No paid services.
- Keep videos, weights, environments, credentials, annotation text, labels, and private manifests out of Git. Commit acquisition instructions, identifiers/hashes, code, and nonrestricted summaries.
- Run targeted CPU tests and saved-record verification. Real processor and GPU paths must be identified as unverified until exercised.
- Publication is authorized by the user's explicit request. Use their existing Git identity and publish privately unless they request public visibility.
