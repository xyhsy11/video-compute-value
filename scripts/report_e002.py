"""Build the E2 checkpoint and sequential README; never rewrite historical runs."""
from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path

from video_compute_value.common import atomic_text, file_hash, read_json, write_json
from video_compute_value.e002_accounting import forecast
from video_compute_value.e002_features import SCHEMA


def csv_text(header, rows=()):
    out = io.StringIO(newline='')
    writer = csv.writer(out, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)
    return out.getvalue()


def report(root, run):
    cohort, accounting = read_json(run/'cohort_summary.json'), read_json(run/'resource_accounting.json')
    compatibility = read_json(run/'compatibility.json') if (run/'compatibility.json').exists() else None
    runtime = compatibility['status'] if compatibility else 'UNEXERCISED'
    plan = forecast(run, accounting)
    probe_rows = [read_json(p) for p in (run/'private/compatibility_records').glob('*.json')]
    if compatibility and compatibility['status'] == 'PASSED' and len(probe_rows) == 5:
        worker = read_json(run/'private/compatibility_worker.json')
        measured_max = max(r['latency_seconds'] for r in probe_rows)
        # Conservative extrapolation from one legacy source, not a qualified full-cohort forecast.
        raw = {'1080_scientific_plus_4_warmup_calls': 1084*measured_max,
               'two_model_loads': 2*worker['load_seconds'], 'record_overhead': 1084*.1,
               'unmeasured_acquisition_and_eligibility_allowance': 900,
               'unmeasured_training_verification_reporting_allowance': 600}
        required = 1.5*sum(raw.values())+180+300
        plan.update(measured_legacy_probe={'calls': 5, 'worker_wall_seconds': compatibility['wall_seconds'],
            'load_seconds': worker['load_seconds'], 'request_seconds': [r['latency_seconds'] for r in probe_rows],
            'maximum_request_seconds': measured_max, 'timing_repeatability_qualified': False},
            raw_assumed_seconds=raw, conditional_required_seconds=required,
            conditional_additional_seconds=max(0, math.ceil(required-accounting['remaining_seconds'])),
            assumptions_only=False,
            forecast_limitation='Measured legacy maximum plus explicit unmeasured input/CPU allowances; '
                                'new-source decode times and provenance/eligibility remain unmeasured.',
            reasons=['40 new eligible sources unavailable', 'remaining grant below conservative stage forecast',
                     'full collection/evaluator orchestration incomplete', 'new-source throughput not yet qualified'])
        write_json(run/'forecast.json', plan)
    relative = run.relative_to(root).as_posix()
    statuses = {'implementation': 'PARTIAL_CPU_CORE', 'collection': 'BLOCKED_INPUT_AND_RESOURCE',
        'model_runtime': runtime, 'predictive_value': 'UNAVAILABLE_NO_E2_OUTCOMES',
        'explicit_question_contribution': 'UNAVAILABLE_NO_E2_OUTCOMES', 'timing_validity': 'UNMEASURED',
        'practical_value': 'PRACTICAL_VALUE_NOT_EVALUABLE_MEASUREMENT', 'publication': 'PENDING_GIT_VERIFICATION'}
    summary = {'experiment_id': 'e002_preacquisition_predictability_v1', 'statuses': statuses,
        'planned_sources': 40, 'actual_frozen_sources': 0, 'planned_questions': 120, 'actual_questions': 0,
        'planned_primary_calls': 960, 'planned_repeatability_calls': 120, 'completed_scientific_calls': 0,
        'completed_compatibility_calls': compatibility['completed_model_calls'] if compatibility else 0,
        'delta_fixed': None, 'delta_question': None, 'accuracy': None, 'complete_pipeline_latency_seconds': None,
        'practical_latency_ratio': None, 'compatibility': compatibility,
        'historical_e0_verdict': 'MEASUREMENT_UNSTABLE'}
    write_json(run/'summary.json', summary)
    write_json(run/'feature_schema.json', SCHEMA)
    write_json(run/'status.json', statuses)
    atomic_text(run/'metrics.csv', csv_text(['metric', 'estimate', 'ci95_low', 'ci95_high', 'status'],
        [[m, '', '', '', 'unavailable_no_E2_outcomes'] for m in ('Delta_fixed', 'Delta_question')]))
    atomic_text(run/'fold_effects.csv', csv_text(['fold', 'delta_fixed', 'delta_question']))
    atomic_text(run/'per_video_effects.csv', csv_text(['anonymous_source', 'fold', 'delta_fixed', 'delta_question']))
    counts = [('verified new source groups', 0, 40), ('E2 questions', 0, 120), ('scientific model calls', 0, 1080)]
    atomic_text(run/'progress.csv', csv_text(['quantity', 'completed', 'planned'], counts))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots(figsize=(9, 3.7), constrained_layout=True)
    y = list(range(3))
    ax.barh(y, [1]*3, color='#e5e7eb')
    ax.set_yticks(y, [r[0] for r in counts])
    for i, (_, completed, planned) in enumerate(counts):
        ax.text(.02, i, f'{completed} / {planned} completed', va='center', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Fraction of planned evidence (progress only)')
    ax.set_title('E2 blocked checkpoint — no scientific outcomes')
    ax.spines[['top', 'right']].set_visible(False)
    (run/'figures').mkdir(exist_ok=True)
    for extension in ('png', 'svg'):
        figure.savefig(run/f'figures/e2_progress.{extension}', dpi=160)
        if extension == 'svg':
            path = run/'figures/e2_progress.svg'
            atomic_text(path, '\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n')
    plt.close(figure)
    write_json(run/'figure_manifest.json', {f'e2_progress.{ext}': {'sha256': file_hash(run/f'figures/e2_progress.{ext}'),
        'data': 'progress.csv', 'data_sha256': file_hash(run/'progress.csv')} for ext in ('png', 'svg')})
    resume = f'.venv\\Scripts\\python.exe -m video_compute_value.e002 resume --run-dir {relative}'
    amendment = f'''# E2 resource amendment request

Current cumulative cap: 27,600 seconds. Charged: {accounting['cumulative_seconds']:.6f} seconds.
Remaining: {accounting['remaining_seconds']:.6f} seconds. Storage: {accounting['storage_current_bytes']:,}
of 21,474,836,480 bytes. Charged video transfer plus interruption upper bound:
{accounting['video_transfer_and_interruption_upper_bound_bytes']:,} of 1,073,741,824 bytes.

No additional grant is assumed. The 960 primary +120 repeatability calls do not
include probes, warmups, eligibility, startup, failure, fitting or publication.
The successful compatibility worker supplies measured legacy call durations, but
new-cohort throughput and timing stability remain unqualified. The current forecast
uses the maximum saved legacy request time for 1,080 scientific plus four warmup
calls, two measured model startups, 0.1 seconds per record, and explicitly unmeasured
allowances of 900 seconds input work and 600 seconds training/verification/reporting.
The exact values are in [forecast.json](forecast.json). Factor 1.5, 180 seconds failure
allowance and 300 seconds closeout produce {plan['conditional_required_seconds']:.3f} seconds.
Conditional added allowance needed from this snapshot: **{plan['conditional_additional_seconds']} seconds**.
This is an exact arithmetic shortfall under stated assumptions, not a measured
completion guarantee. A grant alone does not pass input/integrity/implementation gates.

The first 40 ordered candidate assets have {cohort['first_40_compressed_payload_bytes']:,}
compressed payload bytes in official metadata; HTTP overhead, retries, provenance,
CFR failures and replacements are unmeasured. No traffic/storage amendment is yet
justified. Reserve 128 MiB for private records/traces; acquire no extra weights.
Do not acquire the cohort until the whole next frozen stage is admitted using
valid measured probes and verified resource estimates. E2B separately forecasts
up to 1,440 live pipelines /2,520 model calls plus screen/calibration/warmup calls.

Missing authority: additional cumulative runtime would require explicit user approval.
Missing evidence: 40 eligible independent new sources and full-cohort input/score
integrity. Legacy runtime/score compatibility status: {runtime}. Remaining implementation: scientific cohort/collection orchestration,
OOF sealed-artifact evaluator, training-only cost panels/B* and live three-session runner.
Historical native-loader failures are preserved. This E2 attempt is not repeated automatically.

Checkpoint resume (rechecks fingerprints and preserves a failed probe):

```powershell
{resume}
```

This command does not claim to perform the missing full collection. If compatibility
failed, its preserved receipt stops resume. If it passed, further whole-stage input,
resource and implementation gates still apply before scientific collection.
'''
    atomic_text(run/'resource_amendment_request.md', amendment)
    table = '\n'.join(f'| {name} | `{status}` |' for name, status in statuses.items())
    report_text = f'''# Experiment 2 checkpoint

**E2 has no scientific result.** Implementation and a bounded compatibility attempt
were authorized; scientific collection was blocked. E0 remains MEASUREMENT_UNSTABLE.

| Status field | Current state |
|---|---|
{table}

| Evidence | Completed | Planned |
|---|---:|---:|
| Verified new source groups | 0 | 40 |
| Questions | 0 | 120 |
| Primary model calls | 0 | 960 |
| Repeatability calls | 0 | 120 |

![E2 evidence progress](figures/e2_progress.png)

This figure shows progress, not accuracy or latency. Exact source: [progress.csv](progress.csv).
Only the eight exposed legacy videos are local. They do not count as new E2 data.
The label-independent inventory found {cohort['ordered_candidate_identifiers']} ordered unexposed
short-source identifiers; identifiers alone do not certify source independence/CFR.
No download occurred. The bounded legacy compatibility status is `{runtime}`;
completed compatibility calls: {summary['completed_compatibility_calls']}/5. Those calls never count
as E2 scientific evidence. See [compatibility receipt](compatibility.json).

Implemented CPU core: strict observation interface, V/Q/VQ deterministic features,
four-output ridge, grouped inner alpha fitting, separate held-out decision interface,
threshold feasibility, source bootstrap and timing summaries. E2-only runtime
primitives implement same-call C capture and direct/base-first request boundaries.
Full collection/OOF-artifact/cost-panel/B* orchestration is incomplete. CPU tests
cannot verify a real GPU path. No trained selector, calibration result, held-out
effect, question contribution, measured pipeline cost or efficiency claim exists.

The prospective [protocol](../../protocols/e002_preacquisition_predictability_v1.md)
retains 40 source groups, nested five-by-four source folds, signed gains,
training-only normalization and comparisons to fixed/random/uniform/oracle controls.
E2A advancement and E2B timing/cost gates remain distinct. This low-level/lexical
observation family does not represent every cheap observer. No E3 is authorized.

Current resources: {accounting['cumulative_seconds']:.3f}/27,600 seconds, including inherited
failures and closeout/publication reserve when charged. Latest exact values:
[resource_accounting.json](resource_accounting.json). The conditional unmeasured
E2A forecast needs {plan['conditional_additional_seconds']} additional seconds from this snapshot;
[the amendment request](resource_amendment_request.md) states every assumption.
Input/runtime gates and missing orchestration remain even with a resource grant.

```powershell
{resume}
.venv\\Scripts\\python.exe -m video_compute_value.e002 verify --run-dir {relative}
.venv\\Scripts\\python.exe scripts/audit_e002_publication.py --run-dir {relative}
```

These are implemented checkpoint commands, not a claimed full-collection command.
Historical preservation and real saved-record rescoring are in [verification.json](verification.json).
No synthetic test fixture enters the report. Missing metrics are blank/null, never zero.
'''
    atomic_text(run/'report.md', report_text)
    atomic_text(run/'completion_report.md', report_text)
    # The historical README is snapshotted privately ONCE. Reassembly is idempotent.
    snapshot = run/'private/readme_before_e2.md'
    if not snapshot.exists():
        snapshot.write_bytes((root/'README.md').read_bytes())
    previous = snapshot.read_text(encoding='utf-8')
    previous = previous.replace('No controller training or Experiment 2 was started.',
        'E1 did not train a controller. The separately authorized E2 checkpoint follows below.')
    start = previous.index('## What the evidence supports')
    e2 = f'''## Experiment 2: can observations predict acquisition benefit?

E1 establishes hindsight opportunity. E2 asks whether information available before
acquisition predicts signed correctness change, and then whether a real base-first
policy delivers value after charging both requests. These are separate E2A and E2B
questions. The planned E2 development pilot uses 40 new source groups and 120
questions, with five outer held-out source folds and four inner training folds.

V uses the exact eight base RGB frames; Q hashes only original question text; VQ
adds fixed query–video interactions. A separate C control uses same-call generation
scores. Paths, identities, options, labels, candidate outputs/costs and outside-base
pixels cannot enter the policy observation. Four ridge outputs predict -1/0/+1
gain. Scalers, regularization, fixed controls and thresholds are training-only.

**Current E2 outcome: blocked checkpoint, no E2 scientific calls.**

| Stage | Current evidence |
|---|---|
| Implementation | CPU core tested; full collection/evaluator/cost-panel orchestration incomplete |
| E2A collection | 0/40 source groups; 0/120 questions; 0/1,080 planned calls |
| Bounded legacy compatibility | {runtime}; {summary['completed_compatibility_calls']}/5 calls completed |
| Predictive value / question contribution | Unavailable |
| E2B practical value | Not evaluable; no complete-pipeline timing |

![E2 evidence progress]({relative}/figures/e2_progress.png)

Progress only; [exact count data]({relative}/progress.csv). No accuracy or cost
curve is available. The inherited grant cannot admit the full collection stage.
The eight exposed sources cannot substitute for new E2 groups. The runtime attempt
and all failure evidence are preserved. No GPU compatibility success is inferred
from passing CPU tests. See the [E2 report]({relative}/report.md),
[verification]({relative}/verification.json), [protocol](protocols/e002_preacquisition_predictability_v1.md)
and [conditional resource amendment]({relative}/resource_amendment_request.md).

E2A's primary contrast is VQ minus training-selected fixed quarter; VQ minus V
tests explicit question contribution. E2B requires positive accuracy and no greater
mean complete latency against both Direct16 and a training-selected baseline.
Each adaptive acquisition pays for the initial base and a full 16-frame re-encoding.
Timing must pass the frozen 20% median relative-range criterion in three sessions.
Neither headroom nor predictive gain alone establishes an efficiency improvement.

'''
    previous = previous[:start]+e2+previous[start:]
    previous = previous.replace('## Reproduction and current resources', '## Historical reproduction and resources')
    # Remove historical mutating command blocks; reference frozen instructions instead.
    import re
    previous = re.sub(r'```powershell\n(?:(?!```)[\s\S])*?```', '', previous)
    previous = previous.replace('Regenerate all six research figures and the E1 report from saved data with:',
        'The six saved research figures are preserved below. The historical report writer would overwrite the README and is not run in place.')
    previous = previous.replace('Learning a useful acquisition policy, validating its costs,\nand confirming its behavior on independent data remain future hypotheses.',
        'E2 has begun as a blocked implementation checkpoint; useful predictive selection and validated deployment costs remain unestablished.')
    previous = previous.replace('The CLI also implements `preflight` and `run`',
                                'The historical E1 CLI implements `preflight` and `run`')
    previous = previous.replace("Use the follow-up report's commands for this new run; historical run directories\nremain immutable. Read-only failure closeout can be reproduced with:",
                                'The follow-up commands are preserved in its report. Historical result directories remain immutable; current E2 audits write only under E2.')
    previous += f'''\n## Current E2 reproduction and resource lineage

The latest cumulative ledger is under [E2 accounting]({relative}/resource_accounting.json),
inheriting every closed E0/E1/v5 charge. No grant was renewed. Private records,
annotations, feature matrices and trained artifacts remain excluded from Git.
All historical result directories remain immutable; the new assembler reads them.

```powershell
.venv\\Scripts\\python.exe -m pytest tests/test_e002.py -q
.venv\\Scripts\\ruff.exe check src tests scripts
.venv\\Scripts\\python.exe scripts/report_e002.py --run-dir {relative}
.venv\\Scripts\\python.exe scripts/audit_e002_publication.py --run-dir {relative}
{resume}
```

Resume rechecks the checkpoint; a preserved failed compatibility attempt stops it.
It does not start the unimplemented scientific collection. New resource authority
and valid input/runtime gates are necessary before that stage can be completed.
'''
    overview = '''| Experiment | Question | Sources / questions | Model / input controls | Endpoint | Status | Main result | Limitation |
|---|---|---|---|---|---|---|---|
| E0 | Can requests fit, repeat, and be measured? | 8 / 24 exposed | Frozen Qwen3-VL; uniform 8–64 | Operational gates | MEASUREMENT_UNSTABLE | 62.50–87.50% accuracy | Timing failed; small development cohort |
| E0 timing follow-up | Can worker lifecycle stabilize timing? | 8 / 8 planned | Same model; fresh/resident 8/64 | 20% median relative range | BLOCKED_MODEL_RUNTIME | Zero model calls; native loading failure | Timing outcomes unavailable |
| E1 | Is there hindsight acquisition headroom? | 8 / 24 exposed | Same retained base8; matched augmented16 | Oracle minus expected random | COMPLETE_RETROSPECTIVE | 18.75 pp; oracle minus fixed 12.50 pp | Video-specific fixed quarter matched oracle in all sources |
| E2 | Can observations predict benefit and realize value after cost? | 40 / 120 planned; 0 collected | Same answerer; V/Q/VQ; source-held-out folds | Delta_fixed; separate E2B cost gate | BLOCKED_INPUT / RESOURCE | No scientific result | Partial CPU implementation; full timing unverified |'''
    previous = re.sub(r'\| Experiment \|[^\n]*\n\|---[^\n]*\n(?:\| E[01][^\n]*\n)+', overview+'\n', previous, count=1)
    previous = previous.replace('Both experiments use **Video-MME v1**', 'E0 and E1 use **Video-MME v1**')
    atomic_text(root/'README.md', re.sub(r'\n{3,}', '\n\n', previous))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    report(Path(__file__).resolve().parents[1], args.run_dir.resolve())
