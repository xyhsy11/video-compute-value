# Experiment 0

- Implement only frozen visual-only Video-MME feasibility and measurement. No controller training, adaptive sampling claim, or oracle claim.
- Preserve raw trials, failed attempts, configuration snapshots, source hashes, and resource ledgers. Never substitute synthetic predictions for model results.
- Freeze model/processor/tokenizer revisions, software, cohort, preprocessing, prompt, parser, ordering, and timing boundary before collection. Changes require a new named run.
- Resume only matching fingerprints; skip valid completed identities; never erase failed attempts or reset cumulative resource consumption.
- One CUDA worker at a time, no offloading or quantization. Limits: cumulative 5h40m (20,400 seconds), explicitly authorized by the user on 2026-10-28 after the initial 5h20m grant; 20 GiB additional storage including at most 1 GiB acquired videos. No paid services. The resource-only runtime grant is separate from unchanged frozen experimental fingerprints.
- Keep videos, weights, environments, credentials, annotation text, labels, and private manifests out of Git. Commit acquisition instructions, identifiers/hashes, code, and nonrestricted summaries.
- Run targeted CPU tests and saved-record verification. Real processor and GPU paths must be identified as unverified until exercised.
- The user explicitly authorized committing and pushing the verified code and public-safe reports on 2026-10-28. Use the existing Git identity and repository; do not force-push or publish restricted artifacts. Earlier local-only/no-staging instructions remain historical and are superseded for this publication. No pull request is requested.
