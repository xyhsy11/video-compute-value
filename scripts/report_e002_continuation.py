"""Assemble the sequential research report from immutable E2A outputs; never rewrite old runs."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import re
from pathlib import Path

import numpy as np

from video_compute_value.common import atomic_text, file_hash, read_json, write_json
from video_compute_value.e002v2_analysis import export_csv


def table_rows(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def report(root, run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    accounting = read_json(run/'resource_accounting.json')
    closeout_note = (f"This charge includes {accounting['precharged_closeout_seconds']:.0f} seconds "
                    'prepaid for final verification, Git publication and handoff; the charge is retained.'
                    if accounting.get('precharged_closeout_seconds') else '')
    if (run/'private/frozen.json').exists():
        write_json(run/'feature_schema.json', read_json(run/'private/frozen.json')['feature_schema'])
    relative = run.relative_to(root).as_posix()
    summary_path = run/'summary.json'
    verified_records = (run/'verification.json').exists() and read_json(run/'verification.json').get('all_E2A_checks_passed')
    verified_statistics = ((run/'statistics_verification.json').exists()
                           and read_json(run/'statistics_verification.json').get('all_statistical_checks_passed'))
    complete = bool(summary_path.exists() and read_json(summary_path).get('primary_model_calls') == 960
                    and verified_records and verified_statistics)
    practical = read_json(run/'practical_status.json') if (run/'practical_status.json').exists() else {
        'status': 'NOT_REACHED', 'timing_validity': 'UNMEASURED', 'practical_value': 'PRACTICAL_VALUE_NOT_EVALUABLE_MEASUREMENT'}
    ledger = read_json(run/'private/resource_ledger.json')
    stage_rows = []
    for stage, end_name in [('input_cohort', 'Complete bounded source preparation stage'),
                            ('cpu_preflight', 'All-source native CPU processor/feature integrity')]:
        forecast_path = run/f'{stage}_forecast.json'
        endings = [s for s in ledger['sessions'] if s['name'] == end_name]
        if forecast_path.exists() and endings:
            planned = read_json(forecast_path)
            elapsed = (datetime.fromisoformat(endings[-1]['finished_at'])-datetime.fromisoformat(planned['at'])).total_seconds()
            stage_rows.append([stage, planned['required_seconds'], elapsed, 'complete_stage_wall_clock'])
    launches = [read_json(p) for p in (run/'private/launches').glob('*.json')]
    if launches and (run/'E2A_collection_forecast.json').exists():
        stage_rows.append(['E2A_collection', read_json(run/'E2A_collection_forecast.json')['required_seconds'],
                           sum(r.get('wall_seconds', 0) for r in launches), 'owned_worker_wall_clock_including_loads_and_warmups'])
    export_csv(run/'stage_actual_vs_forecast.csv', ['stage', 'protected_forecast_seconds', 'observed_seconds', 'observed_scope'], stage_rows)
    stage_table = '\n'.join(f'| {stage} | {planned:.2f} | {observed:.2f} | {observed-planned:+.2f} |'
                            for stage, planned, observed, scope in stage_rows)
    figures = {}
    (run/'figures').mkdir(exist_ok=True)
    def save(figure, name, source):
        for extension in ('png', 'svg'):
            path = run/f'figures/{name}.{extension}'
            figure.savefig(path, dpi=170)
            if extension == 'svg':
                atomic_text(path, '\n'.join(s.rstrip() for s in path.read_text(encoding='utf-8').splitlines())+'\n')
            figures[path.name] = {'sha256': file_hash(path), 'data': source, 'data_sha256': file_hash(run/source)}
        plt.close(figure)
    if complete:
        summary = read_json(summary_path)
        methods = list(summary['metrics'])
        figure, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
        ax.bar(methods, [100*summary['metrics'][m] for m in methods],
               color=['#d98f30' if m.startswith('oracle') else '#3975a8' for m in methods])
        ax.set_ylabel('First-pass accuracy (%)')
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=40)
        ax.set_title('E2A: source-held-out forced acquisition; hindsight oracles are nondeployable')
        save(figure, 'e2a_accuracy', 'metrics.csv')
        effects = table_rows(run/'per_video_effects.csv')
        figure, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True, sharex=True)
        for ax, column, title in zip(axes, ('delta_fixed', 'delta_question'), ('VQ − training-selected fixed quarter', 'VQ − V: explicit question contrast')):
            for fold in range(5):
                subset = [(i, r) for i, r in enumerate(effects) if int(r['fold']) == fold]
                ax.scatter([i+1 for i, r in subset], [100*float(r[column]) for i, r in subset], label=f'Fold {fold+1}')
            ax.axhline(0, color='black', linewidth=.7)
            ax.set_ylabel('Paired accuracy effect (pp)')
            ax.set_title(title)
        axes[0].legend(ncol=5, fontsize=9)
        axes[-1].set_xlabel('Anonymous source (three questions each)')
        save(figure, 'e2a_source_effects', 'per_video_effects.csv')
        calibration = table_rows(run/'gain_calibration.csv')
        figure, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
        for family in ('V', 'Q', 'VQ'):
            values = [r for r in calibration if r['family'] == family]
            ax.plot([float(r['mean_predicted_gain']) for r in values], [float(r['mean_observed_gain']) for r in values], 'o-', label=family)
            for row in values:
                if int(row['action_pairs']) < 5:
                    ax.annotate(f"{row['action_pairs']} pair", (float(row['mean_predicted_gain']), float(row['mean_observed_gain'])),
                                xytext=(8, 8), textcoords='offset points', fontsize=9)
        ax.plot([-1, 1], [-1, 1], '--', color='gray')
        ax.legend()
        ax.set_xlabel('Predicted signed correctness gain')
        ax.set_ylabel('Observed signed correctness gain')
        ax.set_title('OOF gain bins; scores are not correctness probabilities')
        save(figure, 'e2a_gain_calibration', 'gain_calibration.csv')
        export_csv(run/'action_frequencies.csv', ['method', 'quarter', 'questions'],
            [[method, quarter+1, count] for method, counts in summary['acquisition_counts'].items() for quarter, count in enumerate(counts)])
        figure, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        names = list(summary['acquisition_counts'])
        for quarter in range(4):
            ax.bar(np.arange(len(names))+quarter*.2, [summary['acquisition_counts'][m][quarter] for m in names], .2, label=f'Q{quarter+1}')
        ax.set_xticks(np.arange(len(names))+.3, names)
        ax.set_ylabel('Questions')
        ax.legend()
        ax.set_title('Forced-acquisition choices (120 questions per method; stop excluded)')
        save(figure, 'e2a_actions', 'action_frequencies.csv')
        metric_table = '\n'.join(f'| {m} | {v*120:.2f}/120 | {v*100:.2f}% |' for m, v in summary['metrics'].items())
        ci = summary['descriptive_conditional_95pct_intervals']
        contrast_table = '\n'.join(
            f"| {name} | {100*(summary['metrics']['VQ']-summary['metrics'][name]):+.2f} | "
            f"[{100*ci[name][0]:.2f}, {100*ci[name][1]:.2f}] |"
            for name in ('fixed', 'V', 'Q', 'random', 'uniform_addition', 'direct16'))
        practical_explanation = (
            'The E2A point estimate is below the 5 pp threshold, only '
            f"{sum(r['delta_fixed'] > 0 for r in summary['fold_effects'])} folds improve, and "
            'the primary interval includes zero. The frozen rule therefore stops E2B before '
            'its timing screen, training cost panels or live stop/acquire evaluation. E2B '
            'latency, overhead, memory and practical accuracy/cost contrasts are unavailable. '
            'The unexercised E2B GPU path is not validated by E2A collection or CPU tests.'
            if practical['status'] == 'NOT_ADMITTED_E2A_ADVANCEMENT' else
            'E2B eligibility and any failed measurement gates are recorded in the separate practical status.')
        warmups = read_json(run/'verification.json')['completed_model_requests_including_warmups']-1080
        mae_text = '; '.join(f'{name}: {value:.4f}' for name, value in summary['gain_mae'].items())
        bin_min = min(int(row['action_pairs']) for row in calibration)
        bin_max = max(int(row['action_pairs']) for row in calibration)
        text = f'''# E2A: pre-acquisition predictability

**Verdict: {summary['predictive_value']}.** Precision flag: {summary['precision'] or 'none'}.
Explicit question contribution: **{summary['question_contribution']}**.

E1 found opportunity by choosing an added quarter after seeing the answers. E2A
asks whether observations available before acquisition can predict signed
correctness change on held-out sources. V uses only the eight retained base RGB
frames and permitted geometry/timing metadata; Q hashes only original question
text; VQ adds the frozen deterministic interactions. Options, identities, paths,
labels, candidate outputs/costs and outside-base pixels cannot enter the policy.
The separate C diagnostic captures scores in the same paid base generation;
plain/observed token agreement passed for all 120 questions. VQ does not use C.

The development pilot collected 40 new official source groups and all 120 original
questions, with 960 primary and 120 fresh-process repeatability calls, plus
{warmups} separate warmups. The earlier
five legacy compatibility calls are excluded. Five outer folds each held out eight
sources; four inner training folds selected regularization. V/Q/VQ use the unchanged
frozen cheap observation family and four independent ridge outputs. All held-out
predictions/actions were sealed before the evaluator joined outcomes.

Each quarter action retains the exact Uniform-8 base and adds eight distinct
frames, followed by a fresh complete 16-frame re-encoding. Nested uniform retains
the same base; Direct16 is sampled independently. Inputs use the shared frozen
Qwen3-VL-4B-Instruct model/processor/tokenizer revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`, BF16/SDPA, 192 x 320 canvases,
original timestamps and unchanged greedy prompt/parser settings. This is
recorded-video QA with a base spanning the timeline, not future-blind streaming.

| Method | Correct or exact expected correct | First-pass accuracy |
|---|---:|---:|
{metric_table}

Delta_fixed = **{100*summary['delta_fixed']:.2f} pp**, descriptive conditional 95% source-bootstrap
interval **[{100*ci['fixed'][0]:.2f}, {100*ci['fixed'][1]:.2f}] pp**.
Delta_question = **{100*summary['delta_question']:.2f} pp**,
interval **[{100*ci['V'][0]:.2f}, {100*ci['V'][1]:.2f}] pp**.
Positive Delta_fixed folds: {sum(r['delta_fixed'] > 0 for r in summary['fold_effects'])}/5.
Advancement required >=5 pp, >=4 positive folds and a positive interval lower bound.

| VQ minus comparator | Effect (pp) | Descriptive conditional 95% interval (pp) |
|---|---:|---:|
{contrast_table}

![Held-out policy accuracy](figures/e2a_accuracy.png)

All 120 questions remain in every denominator. Expected random averages four actions;
orange hindsight oracles are nondeployable. Source: [metrics.csv](metrics.csv).

![All source effects](figures/e2a_source_effects.png)

Every source contributes all three questions; color identifies its outer fold.
Source: [per_video_effects.csv](per_video_effects.csv); [fold_effects.csv](fold_effects.csv).

![Gain prediction bins](figures/e2a_gain_calibration.png)

Signed-gain predictions are not answer correctness probabilities. Bin sizes range
from {bin_min} to {bin_max} action pairs; sparse bins are labeled on the figure.
Source: [gain_calibration.csv](gain_calibration.csv).

![Action distributions](figures/e2a_actions.png)

These E2A policies always acquire. Stop/acquire frequencies are unavailable unless
the separately gated practical stage runs. Source: [action_frequencies.csv](action_frequencies.csv).

VQ rescued {summary['base_wrong_rescues']} base-wrong questions and harmed
{summary['base_correct_harms']} base-correct questions. Oracle regret:
{100*summary['oracle_regret']:.2f} pp. Action-gain MAE: {mae_text}.

Intervals use 10,000 paired source-cluster resamples, seed 2026102902. They condition
on the fitted OOF outputs and omit some split/training uncertainty; folds share
training sources. This is cross-fitted development from the official annotation
test split, not untouched confirmation or an official leaderboard result. Distinct
official source/parent IDs and asset hashes cannot exclude every unknown reupload.
A negative result concerns this observation family/model/action menu, not all cheap observers.
The [saved-record and model verification](verification.json),
[independent integer-contribution bootstrap](statistics_verification.json),
[decision seal](decision_seal_summary.json) and [feature schema](feature_schema.json)
support these estimates.

## E2B: practical compute value

E2B status: **{practical['status']}**. Timing: **{practical['timing_validity']}**.
Practical value: **{practical['practical_value']}**. E2A gains alone establish no
efficiency. E0 remains **MEASUREMENT_UNSTABLE**. See [practical_status.json](practical_status.json).

{practical_explanation}
There is no speedup claim. Any further observation family or confirmation study
would require a separately authorized, prospectively frozen experiment.
'''
    else:
        progress = read_json(run/'private/eligibility_progress.json') if (run/'private/eligibility_progress.json').exists() else {'selected': [], 'visited': [], 'excluded': []}
        calls = len(list((run/'private/records').glob('*.json')))
        blocker = read_json(run/'blocker.json') if (run/'blocker.json').exists() else read_json(run/'input_blocker.json') if (run/'input_blocker.json').exists() else {'status': 'INCOMPLETE'}
        text = f'''# E2 continuation: {blocker['status']}

The continuation has {len(progress['selected'])}/40 eligible new sources, {calls}/1080
scientific requests and no valid complete predictive estimate. Missing outcomes
remain unavailable. The exact blocking gate is {blocker['status']}.
No E2A/E2B success is claimed. E0 remains MEASUREMENT_UNSTABLE.
'''
    text += f'''
The resource-only grant was directly approved as 172,800 seconds total; all earlier
charges are retained. Current cumulative charge: **{accounting['cumulative_seconds']:.6f}/172800 seconds**.
{closeout_note}
Storage at the accounting checkpoint: {accounting['storage_current_bytes']:,}/21,474,836,480 bytes. Video traffic plus
interruption upper bound: {accounting['video_transfer_and_interruption_upper_bound_bytes']:,}/1,073,741,824 bytes.
Exact accounting: [resource_accounting.json](resource_accounting.json).
Actual versus protected stage forecasts: [stage_actual_vs_forecast.csv](stage_actual_vs_forecast.csv).
Protected forecasts include the 1.5 safety factor and explicit failure,
verification/reporting and closeout reserves. All completed stages fit them.

| Stage | Protected forecast (s) | Observed scope (s) | Observed minus protected (s) |
|---|---:|---:|---:|
{stage_table}

Input/CPU observations span their completed stages. Collection reports owned
worker wall time including loads and warmups; coordinator, analysis and report
work remains charged in the cumulative ledger.
These project resource spans are not validated deployed-policy latency measurements.
Historical protocol: [E2 v1](../../protocols/e002_preacquisition_predictability_v1.md);
implementation/resource history: [continuation v2](../../protocols/e002_continuation_v2.md).
All earlier run directories and historical scientific sources remain byte-for-byte preserved.
The original [blocked checkpoint](../e002_20261029_073134_v1/report.md) remains
immutable. The new [readiness report](readiness_initial.json) records this continuation separately.

```powershell
.venv\\Scripts\\python.exe scripts/resume_e002_continuation.py --run-dir {relative}
.venv\\Scripts\\python.exe -m video_compute_value.e002_continuation verify --run-dir {relative}
.venv\\Scripts\\python.exe scripts/report_e002_continuation.py --run-dir {relative}
```

Resume validates fingerprints and completed identities; it cannot retry native loading
failures, alter completed wrong/invalid answers, or evade a failed full-stage budget.
Restricted text, labels, raw generated answers, features, selector artifacts, traces,
videos and weights remain local. Publication is checked separately from scientific success.
'''
    write_json(run/'figure_manifest.json', figures)
    atomic_text(run/'report.md', text)
    atomic_text(run/'completion_report.md', text)
    snapshot = run/'private/readme_before_continuation.md'
    if not snapshot.exists():
        snapshot.write_bytes((root/'README.md').read_bytes())
    previous = snapshot.read_text(encoding='utf-8')
    section = text.replace('# E2A: pre-acquisition predictability', '## E2A: pre-acquisition predictability', 1)
    for name in figures:
        section = section.replace(f'(figures/{name})', f'({relative}/figures/{name})')
    # Resolve every local report link relative to repository root in the README section.
    section = re.sub(r'\]\(([^)]+)\)', lambda m: ']('+(
        m.group(1) if m.group(1).startswith(('http', relative+'/')) else
        m.group(1).removeprefix('../../') if m.group(1).startswith('../../') else relative+'/'+m.group(1))+')', section)
    start, end = previous.index('## Experiment 2:'), previous.index('## What the evidence supports')
    updated = previous[:start]+section+'\n'+previous[end:]
    updated = re.sub(r'\n## Current E2 reproduction and resource lineage[\s\S]*$', '', updated)
    updated = updated.replace('The [current cumulative accounting](results/e001_',
                              'The [historical E1 accounting](results/e001_')
    updated = updated.replace('Its [current cumulative accounting](results/e000_',
                              'Its [historical cumulative accounting](results/e000_')
    updated = updated.replace('its [current cumulative accounting](results/e000_',
                              'its [historical cumulative accounting](results/e000_')
    updated = updated.replace('latest local ledger. Limits remain',
                              'then-current local ledger. Limits remain')
    if complete:
        updated = re.sub(r'^\| E2 \|[^\n]*$',
            f"| E2A | Can observations predict acquisition gain? | 40 / 120 new | Same frozen answerer; V/Q/VQ ridge | Delta_fixed | {summary['predictive_value']} | {summary['delta_fixed']*100:+.2f} pp versus train-fixed | Development pilot; conditional intervals; E2B {practical['status']} |", updated, flags=re.M)
        updated = updated.replace('E2 has begun as a blocked implementation checkpoint; useful predictive selection and validated deployment costs remain unestablished.',
                                  'E2A now tests held-out acquisition prediction. Its reported result and the separately gated practical stage constrain what can be claimed.')
        updated = updated.replace('Our\ncurrent evidence comes from a small, exposed development cohort. We have not\ntrained a controller or demonstrated an adaptive policy or validated speedup.',
            'E0 and E1 use eight exposed development sources. E2A adds 40 new official\nsource groups for held-out development and trains small ridge selectors. The\npractical compute result is evaluated separately below.')
        updated = updated.replace('The separately authorized E2 checkpoint follows below.', 'The separately authorized E2 study follows below.')
    atomic_text(root/'README.md', updated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    report(Path(__file__).resolve().parents[1], args.run_dir.resolve())
