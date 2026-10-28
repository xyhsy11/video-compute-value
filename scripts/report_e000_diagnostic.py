"""Create public-safe tables, figure and correction report from verified real records."""
from __future__ import annotations

import csv
from pathlib import Path

import yaml

from video_compute_value.common import atomic_text, read_json, utc_now, write_json
from video_compute_value.resources import Ledger

ROOT = Path.cwd()
RUN = ROOT / "results/e000_20261028_diagnostic_v2"


def render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    verification = read_json(RUN / "verification.json")
    assert verification["passed"]
    conditions = verification["timing"]
    with (RUN / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(conditions[0]))
        writer.writeheader()
        writer.writerows(conditions)
    write_json(RUN / "summary.json", {"conditions": conditions, "primary_accuracy_curve": False,
        "timing_target": 0.2, "partial_condition_note": "Native-64 counts, peaks and correlations use completed observations only; complete latency and timing-range estimates are withheld", "selection": "8 sources, first official question each, Uniform-8/64, 3 repetitions, paired policies"})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    colors = {"native": "#3374a5", "request_trim": "#dc8736"}
    for arm, offset, label in (("native", -.18, "Native"), ("request_trim", .18, "In-request trim")):
        group = [next(c for c in conditions if c["arm"] == arm and c["budget"] == b) for b in (8, 64)]
        for axis, key, multiplier in ((axes[0], "median_seconds", 1), (axes[1], "median_relative_range", 100),
                                      (axes[2], "peak_reserved_gib", 1)):
            axis.bar([i + offset for i in range(2)], [(c[key] * multiplier if c[key] is not None else float("nan")) for c in group], width=.34,
                     color=colors[arm], label=label)
    for axis in axes:
        axis.set_xticks([0, 1], ["Uniform-8", "Uniform-64"])
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylim(bottom=0)
    if not verification["primary_collection_complete"]:
        for axis in axes[:2]:
            axis.text(.82, .03, "withheld\n(timeout)", ha="center", transform=axis.get_xaxis_transform(), fontsize=8)
    axes[0].set_ylabel("Full request median (seconds)")
    axes[1].set_ylabel("Median relative timing range (%)")
    axes[1].axhline(20, color="#555555", linestyle="--", linewidth=1, label="20% target")
    axes[2].set_ylabel("Completed-request peak reserved (GiB)")
    axes[2].set_ylim(0, max(17, max(c["peak_reserved_gib"] for c in conditions) * 1.1))
    axes[2].axhline(16, color="#555555", linestyle="--", linewidth=1, label="16 GiB physical VRAM")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("Corrected greedy diagnostic: 8 question cells x 3 planned repetitions\nNative Uniform-64 incomplete after worker timeout", fontsize=11)
    fig.savefig(RUN / "timing_comparison.png", dpi=160)
    fig.savefig(RUN / "timing_comparison.svg")
    svg = RUN / "timing_comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def report():
    v = read_json(RUN / "verification.json")
    conditions = v["timing"]
    ledger = read_json(RUN / "resource_ledger.json")
    trims = [c for c in conditions if c["arm"] == "request_trim"]
    target_met = all(c["median_relative_range"] is not None and c["median_relative_range"] <= .2 for c in trims)
    native_met = all(c["complete"] and c["median_relative_range"] is not None and c["median_relative_range"] <= .2 for c in conditions if c["arm"] == "native")
    timing_conclusion = ("Native corrected-greedy execution meets the 20% median relative-range target on both tested budgets. "
                         if native_met else "Native corrected-greedy timing still misses the 20% median relative-range target on at least one tested budget. ")
    if not v["primary_collection_complete"]:
        timing_conclusion = "Native Uniform-64 reached the worker deadline; its complete timing gate cannot be confirmed. "
    timing_conclusion += ("The trim arm also meets that target on the subset."
                          if target_met else "The trim arm misses that target and is not accepted as a general timing fix.")
    table = ["| Policy | Frames | Completed / planned | Median request (s) | Median relative range | Cells >50% | Completed peak allocated / reserved (GiB) | Sampled device peak (GiB) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def number(value, digits=3):
        return "withheld" if value is None else f"{value:.{digits}f}"
    def percent(value):
        return "withheld" if value is None else f"{value:.1%}"
    for c in conditions:
        table.append(f"| {c['arm']} | {c['budget']} | {c['requests']}/24 | {number(c['median_seconds'])} | {percent(c['median_relative_range'])} | {str(c['cells_exceeding_50_percent']) + '/8' if c['complete'] else 'withheld'} | {c['peak_allocated_gib']:.3f} / {c['peak_reserved_gib']:.3f} | {c['sampled_device_peak_gib']:.3f} |")
    cpu_table = ["| Policy | Frames | Median CPU preparation (s) | Median request minus CPU preparation (s) | Median trim host span (s) |",
                 "|---|---:|---:|---:|---:|"]
    for c in conditions:
        cpu_table.append(f"| {c['arm']} | {c['budget']} | {c['median_cpu_prepare_seconds']:.3f} | {c['median_request_minus_cpu_prepare_seconds']:.3f} | {c['median_trim_host_seconds']:.4f} |")
    variation_table = ["| Policy | Frames | Median CPU share | CPU relative range | Centered CPU / total correlation |",
                       "|---|---:|---:|---:|---:|"]
    for c in conditions:
        variation_table.append(f"| {c['arm']} | {c['budget']} | {c['median_cpu_share_of_request']:.1%} | {percent(c['median_cpu_relative_range'])} | {c['cpu_total_centered_correlation']:.3f} |")
    trim64 = next(c for c in conditions if c["arm"] == "request_trim" and c["budget"] == 64)
    native8 = next(c for c in conditions if c["arm"] == "native" and c["budget"] == 8)
    trim8 = next(c for c in conditions if c["arm"] == "request_trim" and c["budget"] == 8)
    text = f'''# Experiment 0 diagnostic correction

Answer disagreements have a confirmed implementation cause and a tested fix.
{timing_conclusion} This is a diagnostic subset, not a replacement full quality curve.

## Immutable lineage and scope

`e000_20261027_collection_v2` remains byte-for-byte unchanged: {v['parent_files_unchanged']}
files rehashed against the pre-investigation inventory. All original raw trials,
failures, configuration, figures and reports remain available. Original source is
preserved at Git commit `2e2b307`; this run freezes production fix `ffaf01d` and exact
source hashes in `plan.json`. The earlier forensic run and stopped allocator
intervention are preserved in `../e000_20261027_diagnostic_v1/`.

The matched comparison is `e000_20261028_diagnostic_v2`: first official question
per each of eight parent videos; frame budgets 8 and 64; three repetitions; paired,
alternating policies; fresh worker per policy/budget/repetition. No labels or
correctness entered selection or inference. {v['primary_completed']}/96 primary requests completed, with {v['primary_timed_out']} timed-out attempt
and {v['primary_unattempted']} unattempted identities. All 24 warmups and four separate
profiles completed. The exact prepared tensors of completed requests match the parent.

## Why the eight answers disagreed

In pinned Transformers 4.57.1, `_prepare_generation_config` can replace supplied
fields equal to global defaults with checkpoint defaults when `use_model_defaults`
is omitted. The declared `do_sample=False` was therefore replaced by the Qwen
checkpoint's `do_sample=True` (temperature 0.7, top-k 20, top-p 0.8). The [pinned generation implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/generation/utils.py) implements this precedence. The CPU audit in
`../e000_20261027_diagnostic_v1/generation_defaults_audit.json` reproduces the effective
sampling mode directly using the installed generation implementation. Random-number
state advanced with request order despite the fixed worker seed.

Forensics tested all eight flagged cells plus three prespecified stable controls.
Each arm had 66 target observations and 33 predecessor requests. Native execution,
RoPE reset, and deterministic math kernels each reproduced all eight disagreements.
Within each arm and case, vision features, prefill positions and first-token score hashes
were identical, prefill KV length/cache position were zero, and scores contained no
NaNs. Captured outputs sometimes selected a strictly lower-scored token. These
observations support random sampling, rather than input drift or request-state
leakage, as the cause. Exact top-score ties are handled as ties in verification.

Explicit call-level `use_model_defaults=False`, `do_sample=False`, one beam, eight
new tokens and resolved BOS/EOS/padding removed every disagreement: 0/11 unstable
cells across 66 target observations, including all eight original flagged cells.
The production adapter now validates the effective greedy mode, rejects conflicting
frozen policies and records effective settings in new model/worker manifests.
The paired production test has {v["valid_primary_outputs"]}/{v["primary_completed"]} parseable completed primary outputs,
zero answer disagreement cells and identical
generated token sequences across the observed calls for all 16 selected input cells.
Missing native-64 repetition-3 outputs are not fabricated or replaced.

**Interpretation correction:** the original collection's quality and timing curves
are historical sampled-generation outputs. They cannot establish the intended
frozen-greedy protocol. Their measured values have not been altered. A complete
five-budget corrected-greedy collection has not been performed here.

## Timing investigation

The first allocator intervention (`max_split_size_mb:128` with garbage-collection
threshold 0.75) regressed: a completed warmup took 144.346 seconds and reserved
19.106 GiB against 16 GiB physical VRAM. Its retry was stopped after more than 160
seconds to protect the remaining allowance. No primary timing requests were saved
in that phase. One completed warmup and two interrupted/unknown attempts remain.
One interruption was a system shutdown; its entire coordinator session was charged
through the verified Windows Kernel-General shutdown timestamp. The original
unclosed ledger and private OS evidence were preserved. Powered-off time was not
charged as active execution, and no previous consumption was removed.

The native-64 repetition-3 worker reached its predeclared 600-second limit while
its fifth primary request was still outstanding. That request has no saved output
or completed latency; its attempt is censored. Three subsequent identities were
never attempted. Device-wide observation during the stall showed almost all 16 GiB
in use. The outstanding attempt was not retried. The prespecified final cleanup
counterpart was then run with the same frozen worker code, inputs and limit.
Native-64's complete latency/range summary is withheld. Its listed memory counters
come from completed requests only and do not capture the failed request's peak.
`native_timeout.json` and `memory_pressure_observation.json` preserve the evidence.

The smaller named follow-up compares the default native allocator with the same
allocator plus `torch.cuda.empty_cache()` after input transfer and before generation.
Cleanup is **inside** the original synchronized full-request timer. BF16, native
SDPA, model/software/data revisions, canvas, prompt and parser remain fixed.
There is no offload, quantization or reused decoded video/vision/KV work.
The [PyTorch 2.8 CUDA documentation](https://docs.pytorch.org/docs/2.8/notes/cuda.html#memory-management) distinguishes live allocation from allocator reservations and documents cache release behavior.

{chr(10).join(table)}

The 64-frame trim arm completed all 24 requests with a {trim64["median_relative_range"]:.1%} median timing range
and {trim64["peak_reserved_gib"]:.3f} GiB peak reserved memory. This is a candidate mitigation for the observed
high-frame memory-pressure failure. It is not a general policy: at eight frames,
trimming increased median timing range from {native8["median_relative_range"]:.1%} to {trim8["median_relative_range"]:.1%}. No speedup estimate is
reported for the censored native-64 condition.

![Matched timing and allocator comparison](timing_comparison.png)

Relative range is `(max-min)/median` over three repetitions per question; the table
reports its median over eight cells. Request medians first summarize each question's
three observations. PyTorch reservations are allocator counters and can exceed
physical VRAM under Windows; they are not resident device usage. NVML values are
50 ms sampled device-wide peaks, not process-specific or exact continuous maxima.

{chr(10).join(cpu_table)}

{chr(10).join(variation_table)}

The centered correlation subtracts each question cell's median before comparing its
within-condition completed observations (24 in a complete condition); critical timeout information is retained separately. The correlation is descriptive and has no significance test.
CPU preparation includes full sequential decoding, selection/resize and processor
work. Its observed share and covariation localize preparation stalls without
claiming a specific CPU clock, I/O or scheduling cause.

The request-minus-CPU remainder includes transfers, driver/allocator work, model
execution and output handling; it is not pure GPU time. Separate intrusive profiles
remain private. Memory pressure is a supported candidate for timing tails, especially
in the rejected allocator arm. These counters do not prove Windows paging as a
unique cause. Thermal/clock telemetry is retained; this small paired test does not
isolate every power, OS-cache or background-load effect. A different day and reduced
question set also prevent treating these numbers as a controlled before/after
comparison with the original full collection. No statistical significance claim.

## Verification and accounting

The final CPU regression suite passed 54 tests; Ruff passed on all source, test and
script files. The PNG was visually reviewed and the SVG parsed successfully.

Verification here tests record integrity and the targeted diagnostic, not full
five-budget feasibility or the 20% timing gate. It redecoded {v['raw_outputs_redecoded']} saved outputs using
the pinned tokenizer, rehashed {v['model_files_rehashed']} model files, checked the
parent inventory, trial identities/order/completeness, input hashes, no OOMs in completed requests,
allocator backend, frozen effective policies, repeated answers/tokens and saved
memory/timer invariants. Raw evidence is private and bound by the published aggregate
hash in `verification.json`; annotation text, labels, weights and videos stay out of Git.

New work: {v['new_confirmed_model_invocations']} confirmed model invocations, including the completed inference whose
initial nonfinite-score JSON logging failed, plus {v['new_additional_possible_invocations']} possibly invoked interrupted
attempts conservatively retained. Historical model calls: 491. Cumulative upper bound:
{v['cumulative_model_invocations_upper_bound']:,} calls. Setup/recording/file-lock failures and their runtime remain accounted.

Cumulative active consumption at publication preparation: {ledger['active_seconds']:.3f}
seconds of 14,400 ({ledger['active_seconds']/3600:.3f} hours); remaining
{14400-ledger['active_seconds']:.3f} seconds. Project storage observed
{ledger['storage_current_bytes']/1024**3:.4f} GiB, peak
{ledger['storage_peak_bytes']/1024**3:.4f} GiB of 20 GiB. Video acquisition remains
{ledger['video_transfer_bytes']/1024**2:.3f} MiB plus a 1 MiB interrupted-transfer reserve,
below 1 GiB. No additional download, paid inference or cloud compute was used.
These totals include conservative external/overhead allowances and all failed attempts.
The authoritative continuing ledger is this run's `resource_ledger.json`.

Reproduction/validation commands (run only with matching local data and remaining budget):

```powershell
.venv\\Scripts\\python.exe -m pytest -q
.venv\\Scripts\\python.exe scripts/verify_e000_diagnostic.py
.venv\\Scripts\\python.exe scripts/report_e000_diagnostic.py
```

Do not resume the stopped 19 GiB allocator arm or reset a resource ledger. Any future
configuration/source changes require another named run. The targeted diagnosis is
complete; full five-budget greedy feasibility confirmation remains unclaimed.
'''
    atomic_text(RUN / "report.md", text)
    write_json(RUN / "status.json", {"status": "DIAGNOSTIC_COMPLETE", "updated_at": utc_now(),
        "original_results_preserved": True, "answer_disagreement_cause": "checkpoint defaults enabled sampling",
        "fixed_greedy_unstable_forensic_cells": 0, "paired_timing_requests": v["primary_completed"],
        "primary_collection_complete": v["primary_collection_complete"], "timed_out": v["primary_timed_out"], "unattempted": v["primary_unattempted"],
        "native_meets_20_percent_subset_target": native_met, "request_trim_meets_20_percent_subset_target": target_met, "full_five_budget_greedy_revalidation": False,
        "new_model_invocations_confirmed": v["new_confirmed_model_invocations"], "additional_possible_model_invocations": v["new_additional_possible_invocations"],
        "cumulative_model_invocations_upper_bound": v["cumulative_model_invocations_upper_bound"]})


if __name__ == "__main__":
    config = yaml.safe_load((ROOT / "configs/e000_diagnostic_v1.yaml").read_text(encoding="utf-8"))
    ledger = Ledger(RUN / "resource_ledger.json", ROOT, config)
    with ledger.track("diagnostic-report-and-figures"):
        render()
    report()
