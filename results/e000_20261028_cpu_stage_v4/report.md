# CPU preparation stage diagnostic v4

This is a separate CPU-only diagnostic after the completed v3 collection. Its timings do not replace any primary measurement or change the MEASUREMENT_UNSTABLE verdict.

One persistent process prepared the first official question of each source at 8/64 frames, three times, in the parent block order. No model was instantiated, no CUDA context was initialized, and no frames/features were reused. All 48 input tensor hashes match the saved collection.

| Frames | CPU preparation s | Decode/selection/RGB/resize/stack s | Letterbox subset s | Native processor s | Copies/metadata/checks/other s | Median CPU repetition range |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 2.072 | 2.059 | 0.088 | 0.0106 | 0.0028 | 11.17% |
| 64 | 2.806 | 2.748 | 0.667 | 0.0468 | 0.0167 | 11.98% |

Each stage uses per-source repetition medians, then the median over eight sources. Stage medians need not add to the total median. Letterboxing is nested within the decode span. The remaining decode span includes I/O, FFmpeg decoding, source-PTS checks, selection, RGB conversion and stacking; it is not isolated decoder-kernel time.

These measurements localize CPU costs but do not establish a unique OS scheduling, clock, I/O, or thread-pool cause. A persistent CPU-only process differs from fresh GPU workers; it cannot validate end-to-end timing stability. A changed execution policy requires separately named real preflight and collection, not replacement of v3 repetitions.
