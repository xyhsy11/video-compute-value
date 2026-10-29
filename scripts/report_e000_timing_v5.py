"""Sanitized E0 v5 reporting and figures from saved aggregate data only."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import statistics

from video_compute_value.common import atomic_text, digest, file_hash, read_json, write_json

RESTRICTED = {'source_id', 'question_id', 'question', 'options', 'answer', 'label', 'generated_token_ids',
              'decoded_output', 'parsed_answer', 'model_path', 'python_executable', 'traceback', 'path',
              'input_tensor_hashes', 'input_metadata_hash', 'rgb_sha256'}


def public_safe(value):
    if isinstance(value, dict):
        if RESTRICTED.intersection(value):
            raise ValueError('Restricted public field')
        for item in value.values():
            public_safe(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            public_safe(item)
    elif isinstance(value, str):
        if re.search(r'[A-Za-z]:[\\/]|/Users/|/home/|(?i:token=|password=|bearer )', value):
            raise ValueError('Restricted public path or secret')


def csv_text(rows):
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def figures(run, summary):
    os.environ['MPLCONFIGDIR'] = str(run.parents[1] / '.cache/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    folder = run / 'figures'
    folder.mkdir(exist_ok=True)
    conditions = summary['conditions']
    fig, ax = plt.subplots(figsize=(8, 4.8), layout='constrained')
    colors = {'A': '#355C7D', 'B': '#C06C38'}
    for i, c in enumerate(conditions):
        label = f"{c['arm']} / {c['budget']} frames"
        if c['S'] is not None:
            ax.bar(i, 100 * c['S'], color=colors[c['arm']], width=.62)
            ax.text(i, 100 * c['S'] + 1, f"{100 * c['S']:.2f}%", ha='center')
        else:
            ax.text(i, 3, 'Not collected', ha='center', rotation=90, color='0.4')
        ax.text(i, -4, label, ha='center', va='top')
    ax.axhline(20, color='#A33A32', linestyle='--', label='Fixed 20% criterion')
    ax.set_xticks([])
    ax.set_xlim(-.6, len(conditions) - .4)
    ax.set_ylim(0, max(30, max((100 * c['S'] for c in conditions if c['S'] is not None), default=0) + 10))
    ax.set_ylabel('Median question-level relative timing range (%)')
    observed = sum(c['primary_requests'] for c in conditions)
    ax.set_title(f'E0 timing screen: {observed}/96 primary requests collected\nPlanned: 8 source questions × 3 independent repetitions per arm/budget')
    ax.legend(frameon=False)
    fig.savefig(folder / 'diagnostic_stability.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout='constrained')
    for ax, budget in zip(axes, (8, 64)):
        for arm in ('A', 'B'):
            cells = [c for c in summary['question_ranges'] if c['arm'] == arm and c['budget'] == budget]
            if cells:
                offset = -.12 if arm == 'A' else .12
                x = [c['anonymous_cell'] + offset for c in cells]
                ax.scatter(x, [c['median_request_seconds'] for c in cells], color=colors[arm], label=arm)
            else:
                ax.text(.5, .6 if arm == 'A' else .4, f'{arm}: not collected', transform=ax.transAxes, ha='center', color='0.4')
        ax.set_title(f'{budget} frames')
        ax.set_xlabel('Anonymous source question (8 total)')
        ax.set_ylabel('Median full-request latency (s)')
        ax.set_xticks(range(1, 9))
        ax.set_xlim(.4, 8.6)
        if ax.collections:
            ax.legend(title='Policy', frameon=False)
        else:
            ax.set_yticks([])
    fig.suptitle(f'Absolute request times: {observed}/96 requests collected\nEach available point is the median of 3 sessions')
    fig.savefig(folder / 'request_times.png', dpi=160)
    plt.close(fig)
    # Absolute ranges alongside medians: no independently stacked stage medians.
    rows = summary['question_ranges']
    if rows:
        fig, ax = plt.subplots(figsize=(9, 4.6), layout='constrained')
        for arm in ('A', 'B'):
            group = [c for c in rows if c['arm'] == arm]
            ax.scatter([c['median_request_seconds'] for c in group], [c['absolute_range_seconds'] for c in group],
                       color=colors[arm], label=arm, alpha=.8)
        ax.set_xlabel('Question median full-request latency (s)')
        ax.set_ylabel('Maximum − minimum across sessions (s)')
        ax.set_title(f'Absolute time and spread: all {len(rows)} arm/budget/question cells')
        ax.legend(title='Policy', frameon=False)
        fig.savefig(folder / 'absolute_spread.png', dpi=160)
        plt.close(fig)
    link = read_json(run / 'confirmation.json') if (run / 'confirmation.json').exists() else {}
    if link.get('run'):
        child = run.parents[1] / link['run']
        if (child / 'summary.json').exists():
            full = read_json(child / 'summary.json')
            if all(c['complete'] for c in full['conditions']):
                fig, ax = plt.subplots(figsize=(8, 4.6), layout='constrained')
                ax.bar([str(c['budget']) for c in full['conditions']], [100 * c['S'] for c in full['conditions']],
                       color=colors[link['selected_policy']])
                ax.axhline(20, color='#A33A32', linestyle='--', label='Fixed 20% criterion')
                ax.set_xlabel('Uniform frame budget')
                ax.set_ylabel('Median question-level relative range (%)')
                ax.set_title('New full confirmation: 24 questions / 8 sources / 3 sessions')
                ax.legend(frameon=False)
                fig.savefig(folder / 'confirmation_stability.png', dpi=160)
                plt.close(fig)
    write_json(run / 'figure_manifest.json', {'aggregate_sha256': file_hash(run / 'summary.json'),
               'figures': {p.name: file_hash(p) for p in sorted(folder.glob('*.png'))},
               'plotted_condition_data_sha256': digest(conditions),
               'plotted_question_data_sha256': digest(summary['question_ranges'])})


def report(run):
    summary = read_json(run / 'summary.json')
    verification = read_json(run / 'verification.json')
    resource = read_json(run / 'resource_accounting.json')
    public_safe(summary)
    public_safe(verification)
    public_safe(resource)
    atomic_text(run / 'aggregates.csv', csv_text(summary['conditions']))
    if summary['question_ranges']:
        atomic_text(run / 'question_ranges.csv', csv_text(summary['question_ranges']))
    else:
        atomic_text(run / 'question_ranges.csv', 'arm,budget,anonymous_cell,relative_range,absolute_range_seconds,median_request_seconds,median_prepare_seconds,median_remainder_seconds,over_50_percent\n')
    frozen = read_json(run / 'private/frozen.json')
    env = read_json(run / 'private/environment.json')
    safe_env = {k: env[k] for k in ('os', 'python', 'packages', 'nvidia_driver_version', 'power')}
    safe_env['gpu'] = frozen['environment_fingerprint']['gpu']
    public_safe(safe_env)
    write_json(run / 'environment_summary.json', safe_env)
    cpu = read_json(run / 'private/cpu_preparations.json')
    cpu_summary = []
    for budget in (8, 64):
        values = [r['elapsed_seconds'] for r in cpu if r['budget'] == budget]
        cpu_summary.append({'budget': budget, 'native_preparations': len(values),
                            'median_native_preparation_seconds': statistics.median(values),
                            'minimum_seconds': min(values), 'maximum_seconds': max(values),
                            'scope': 'Single CPU-only pass over eight source questions; validation decodes excluded'})
    public_safe(cpu_summary)
    write_json(run / 'cpu_preparation_summary.json', cpu_summary)
    atomic_text(run / 'cpu_preparation_summary.csv', csv_text(cpu_summary))
    figures(run, summary)
    confirmation = read_json(run / 'confirmation.json') if (run / 'confirmation.json').exists() else {
        'status': 'CONFIRMATION_NOT_RUN', 'reason': 'Neither policy qualified or diagnostic incomplete'}
    public_safe(confirmation)
    write_json(run / 'confirmation_status.json', confirmation)
    full_text = ''
    failure_text = ''
    display_status = summary['status']
    if (run / 'failure_summary.json').exists():
        failure = read_json(run / 'failure_summary.json')
        display_status = failure['status']
        public_safe(failure)
        failure_text = f"""
The first preflight worker exited during checkpoint loading with Windows native
exit code `{failure['native_exit_code']}`, before a model-call start event or a
successful load receipt. One worker was attempted and reaped; the second worker,
all 12 preflight calls, the 120 main diagnostic calls and full confirmation were
not run. The contemporaneous Windows Application Error event names
`{failure['contemporaneous_faulting_module']}` with the same exception code. This
does not establish the underlying cause. No package, driver, model, power setting,
application or measurement policy was changed afterward. No automatic retry ran.

The resource forecast admitted preflight, so this is a native initialization
failure, not a negative timing result or a resource-denied phase. A/B S values at
8/64 are **missing**, not zero or passing. Both policies remain unevaluated.
[closeout_verification.json](closeout_verification.json) separately passes the
available-evidence/cleanup/accounting checks; [verification.json](verification.json)
correctly fails full experimental completeness.
Its combined "unreaped/incomplete worker" flag refers here to the incomplete
session: the failure receipt and a free CUDA-worker lock independently confirm
that the owned process was reaped.

The next single action is a separately named, bounded model-initialization crash
diagnostic using the preserved native event/module and existing assets before
attempting another prospective timing study. This task authorizes no automatic
replacement session. An additional cumulative allowance cannot be estimated from
these absent GPU probes; the native failure must be resolved and fresh complete
phase forecasts admitted first.
"""
    if confirmation.get('run'):
        child = run.parents[1] / confirmation['run']
        if (child / 'summary.json').exists():
            full = read_json(child / 'summary.json')
            entries = [f"| {c['budget']} | {100*c['S']:.2f}% | {c['S'] <= .2} |" for c in full['conditions'] if c['S'] is not None]
            full_text = '\n\nFull confirmation gates:\n\n| Frames | Median R | ≤20% |\n|---:|---:|---|\n' + '\n'.join(entries)
            if entries:
                full_text += '\n\n![Full confirmation timing gates](figures/confirmation_stability.png)\n'
    table = []
    for c in summary['conditions']:
        table.append('| ' + ' | '.join([c['arm'], str(c['budget']), str(c['primary_requests']),
                     'missing' if c['S'] is None else f"{100*c['S']:.2f}%",
                     'missing' if c['median_request_seconds'] is None else f"{c['median_request_seconds']:.3f}",
                     'missing' if c['absolute_range_seconds'] is None else f"{c['absolute_range_seconds']:.3f}",
                     'missing' if c['median_prepare_seconds'] is None else f"{c['median_prepare_seconds']:.3f}",
                     'missing' if c['median_remainder_seconds'] is None else f"{c['median_remainder_seconds']:.3f}",
                     'missing' if c['individual_over_50_percent'] is None else str(c['individual_over_50_percent'])]) + ' |')
    name = run.name
    cpu_table = '\n'.join(f"| {c['budget']} | {c['native_preparations']} | {c['median_native_preparation_seconds']:.3f} | {c['minimum_seconds']:.3f}–{c['maximum_seconds']:.3f} |" for c in cpu_summary)
    memory = []
    for c in summary['conditions']:
        values = [c['arm'], str(c['budget'])]
        for key in ('peak_allocated_bytes', 'peak_reserved_bytes', 'sampled_device_gpu_peak_bytes',
                    'live_allocated_after_bytes', 'reserved_after_bytes'):
            values.append('unavailable' if c[key] is None else f'{c[key] / 1024**3:.3f}')
        memory.append('| ' + ' | '.join(values) + ' |')
    text = f'''# Controlled E0 timing follow-up v5

Diagnostic status: **{display_status}**. Selected policy: **{summary['selected_policy']}**.
Full confirmation: **{confirmation['status']}**. Reason: {confirmation.get('reason', 'see confirmation evidence')}.
{full_text}
{failure_text}
Historical E0 remains **MEASUREMENT_UNSTABLE**. Historical E0/E1 evidence was
preserved; {verification['historical_files_unchanged']} inventory files matched exactly.

The frozen study asks whether a worker lifecycle makes complete local-video requests
repeatable across independent worker sessions. The cohort remains eight exposed
Video-MME short sources; this screen uses the first official question of each.
The model/processor/tokenizer is Qwen3-VL-4B-Instruct at
`ebb281ec70b05090aa6165b016eac8ec08e71b17`, BF16, SDPA, 192 × 320 inputs.
No controller, practical adaptive policy, efficiency claim or Experiment 2.

Planned policy A uses six fresh main workers, one per budget/repetition. Policy B uses
three independent resident main workers, one per repetition, retaining the model
across two budget blocks. Every block has two identical historical warmups.
Rounds 1/2 use A→B with 8→64 and B→A with 64→8; round 3 and source orders are
drawn prospectively from seed 20261029. Actual anonymous arrays are in
[schedule.json](schedule.json); [freeze.json](freeze.json) seals exact sources and design.

The unchanged synchronized timer starts before video opening and ends after full
decoding, processing, transfer, greedy generation, output decoding/parsing and final
CUDA synchronization. Each request reopens and sequentially decodes the complete
video. Ordinary OS caching is allowed. There is no cross-request evidence/KV cache.
Only 64 frames trims allocator cache inside the timer; 50 ms memory sampling is
unchanged in both policies. Model loading, warmups, hashes, scoring, serialization
and telemetry are outside primary measurements and inside project accounting.

| Policy | Frames | Primaries | Median R | Median request s | Median absolute range s | Preparation s | Remainder s | Individual R>50% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

For each question, R=(max−min)/median over three independent sessions. S is the
median of eight R's. Request/preparation/remainder and absolute-range summaries
also take question medians first. Separate stage medians need not sum to median
total. The remainder includes transfers, allocator work, model execution and output
handling. All individual R's are retained in [summary.json](summary.json), with
anonymous exports in [question_ranges.csv](question_ranges.csv) when available.
No observation is excluded, trimmed, replaced or treated as another source video.

The available CPU-only preparation measurements are a single pass over eight
source questions, without a resident GPU model. Extra serial/four-thread validation
decodes and hashing are outside these host spans and inside project accounting.
These are preparation costs, not full-request measurements or repeatability tests.
Source: [cpu_preparation_summary.csv](cpu_preparation_summary.csv).

| Frames | Native preparations | Median preparation s | Minimum–maximum s |
|---:|---:|---:|---:|
{cpu_table}

Qualification requires S≤20% at both budgets and every integrity/completeness gate.
Selection is A first, then B, otherwise stop. This is only confirmation eligibility.
The screen cannot establish a historical causal mechanism or validated acceleration.
Full-confirmation forecasting is conditional on a qualifying policy; no such
forecast was made after this preflight failure.

![Fixed timing criterion](figures/diagnostic_stability.png)

![Absolute request time](figures/request_times.png)

Absolute allocated/reserved peaks and device-wide 50 ms sampled maxima are saved
in the aggregate JSON/CSV. Peaks include resident weights; reservation is not live
allocation. Post-release live allocations are checked against the documented shape
warmup baseline with 64 MiB tolerance. Unavailable device/process observations are
kept unavailable. Detailed transition observations and power/load/clock/temperature
telemetry are retained privately. [environment_summary.json](environment_summary.json)
gives the tested software and device. We do not infer thermal throttling, one
Windows scheduling defect, or a unique decoder/allocator cause from correlation.

| Policy | Frames | Peak allocated GiB | Peak reserved GiB | Sampled device GiB | Maximum post-release live GiB | Maximum post-release reserved GiB |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(memory)}

Independent verification errors: `{json.dumps(verification['errors'])}`. Failed
or incomplete measurement sessions remain in the private event/failure logs;
no failed session is automatically restarted or replaced.

[verification.json](verification.json) records the full verifier's completeness
failure. The verifier implements native-token decoding, answer reparsing,
historical input/output comparison, call/worker reconstruction, memory/resource/
preservation checks, and independent S/selection recomputation. This run has zero
new generated-token sequences or primary timings for those generation checks.
CPU tests validate implementation;
real-GPU evidence is limited to the phases actually completed.

Calls by phase/kind: `{json.dumps(resource['model_calls'], sort_keys=True)}`.
Completed new calls: {resource['completed_model_calls']}; unknown upper bound:
{resource['unknown_model_invocations_upper_bound']}. Planned diagnostic calls: 132
(8 preflight warmups, 4 profiles, 24 main warmups, 96 primary requests).
Cumulative consumed: {resource['cumulative_seconds']:.3f} / 27,600 s;
remaining: {resource['remaining_seconds']:.3f} s. Storage:
{resource['storage_current_bytes']} / {resource['storage_limit_bytes']} bytes.
Acquired-video transfers: {resource['video_transfer_bytes']} bytes; no new acquisition.
All historical charges and failed attempts are inherited. Detailed local accounting
is `private/resource_ledger.json`; [resource_accounting.json](resource_accounting.json)
is its safe checkpoint. A conservative publication reserve is charged on closeout.

Reproducible commands (collection commands require matching local restricted assets,
an open resource allowance and an untouched frozen implementation):

```powershell
$run = "results/{name}"
.venv\\Scripts\\python.exe scripts/stabilize_e000_timing_v5.py prepare --run-dir $run
.venv\\Scripts\\python.exe scripts/stabilize_e000_timing_v5.py preflight --run-dir $run
.venv\\Scripts\\python.exe scripts/stabilize_e000_timing_v5.py collect --run-dir $run
.venv\\Scripts\\python.exe scripts/stabilize_e000_timing_v5.py analyze --run-dir $run
.venv\\Scripts\\python.exe scripts/stabilize_e000_timing_v5.py confirmation --run-dir $run
.venv\\Scripts\\python.exe scripts/stabilize_e000_timing_v5.py verify --run-dir $run
.venv\\Scripts\\python.exe scripts/verify_e000_timing_closeout_v5.py --run-dir $run
.venv\\Scripts\\python.exe scripts/stabilize_e000_timing_v5.py report --run-dir $run
.venv\\Scripts\\python.exe scripts/stabilize_e000_timing_v5.py close --run-dir $run
```

`prepare` creates a fresh freeze and is intentionally rejected on this existing run.
`collect` skips only completed whole workers with valid immutable identities;
interrupted resident sessions cannot resume. `confirmation` allows one attempt.
Use `verify` for read-only reproduction; it does not rerun inference or rewrite
historical artifacts. See the [prospective protocol](../../protocols/e000_timing_stabilization_v5.md).
For this incomplete native-failure outcome, the full verifier intentionally returns
exit code 2; the available-evidence closeout verifier returns 0 on preserved valid
evidence. Measurement paths at every budget remain unverified in this new run.
'''
    public_safe(text)
    atomic_text(run / 'report.md', text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    report(parser.parse_args().run_dir.resolve())
