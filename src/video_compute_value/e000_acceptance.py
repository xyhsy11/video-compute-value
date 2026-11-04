"""User-amended E0 acceptance, recomputed from immutable real measurements."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics

from .common import digest, file_hash, read_json, utc_now, write_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = 'configs/e000_acceptance25_v1.json'
PROTOCOL = 'protocols/e000_acceptance25_v1.md'
RUN = 'results/e000_20261104_acceptance25_v1'
AUTHORITY = 'make it under25% as good, and push and submit the new E0 to github'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def select_policy(conditions, threshold, *, integrity):
    require(isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
            and math.isfinite(threshold) and 0 < threshold < 1, 'Invalid acceptance threshold')
    if not integrity:
        return None
    for arm in ('A', 'B'):
        rows = [c for c in conditions if c['arm'] == arm]
        if (len(rows) == 2 and {c['budget'] for c in rows} == {8, 64}
                and all(c['complete'] and c['primary_requests'] == 24
                        and isinstance(c['S'], (int, float)) and math.isfinite(c['S'])
                        and 0 <= c['S'] <= threshold for c in rows)):
            return arm
    return None


def recompute(rows, fingerprint):
    require(len(rows) == 132 and len({r['call_id'] for r in rows}) == 132, 'Incomplete or duplicate source calls')
    cells = {}
    for row in rows:
        require(row['fingerprint'] == fingerprint, 'Source record fingerprint mismatch')
        require(row.get('historical_input_match') and row.get('historical_output_match')
                and not row.get('live_memory_growth_flag'), 'Source integrity guard failed')
        if row['phase'] != 'diagnostic' or row['kind'] != 'primary':
            continue
        seconds = row['latency_seconds']
        require(isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds > 0,
                'Invalid primary timing')
        key = row['arm'], row['budget'], row['source_id'], row['question_id']
        cells.setdefault(key, []).append((row['repetition'], seconds))
    require(len(cells) == 32, 'Expected all 32 arm/budget/question cells')
    groups = {}
    for (arm, budget, _, _), observations in cells.items():
        require(sorted(r for r, _ in observations) == [1, 2, 3], 'Missing or duplicate independent repetition')
        times = [t for _, t in observations]
        groups.setdefault((arm, budget), []).append((max(times)-min(times))/statistics.median(times))
    require(set(groups) == {('A', 8), ('A', 64), ('B', 8), ('B', 64)}, 'Unexpected timing condition')
    result = []
    for (arm, budget), values in sorted(groups.items()):
        require(len(values) == 8, 'Expected eight source questions per condition')
        result.append({'arm': arm, 'budget': budget, 'complete': True, 'primary_requests': 24,
                       'source_questions': 8, 'independent_repetitions': 3, 'S': statistics.median(values)})
    return result


def analyze(root=ROOT):
    config = read_json(root / CONFIG)
    require(config['acceptance_threshold'] == .25 and config['original_threshold'] == .2
            and config['new_model_calls'] == 0 and config['inclusive'] is True, 'Amendment configuration changed')
    source = root / config['source_run']
    frozen = read_json(source / 'private/frozen.json')
    require(digest({k: v for k, v in frozen.items() if k != 'fingerprint'}) == frozen['fingerprint'],
            'Source freeze seal mismatch')
    verification = read_json(source / 'verification.json')
    require(verification['passed'] is True and verification['completed_model_calls'] == 132
            and verification['unknown_model_invocations_upper_bound'] == 0, 'Source verification did not pass')
    rows, record_hashes = [], {}
    for path in sorted((source / 'private/records').glob('*.json')):
        row = read_json(path)
        seal = row.pop('seal')
        require(digest(row) == seal and path.stem == row['call_id'], 'Source record seal/identity mismatch')
        rows.append(row)
        record_hashes[path.name] = file_hash(path)
    conditions = recompute(rows, frozen['fingerprint'])
    saved = read_json(source / 'summary.json')
    for condition in conditions:
        prior = next(c for c in saved['conditions'] if (c['arm'], c['budget']) ==
                     (condition['arm'], condition['budget']))
        require(math.isclose(prior['S'], condition['S'], rel_tol=0, abs_tol=1e-12), 'Saved timing aggregate mismatch')
    original = select_policy(conditions, .2, integrity=True)
    selected = select_policy(conditions, .25, integrity=True)
    require(original is None and saved['selected_policy'] is None
            and saved['status'] == 'STABILIZATION_NOT_DEMONSTRATED', 'Original frozen decision mismatch')
    for c in conditions:
        c['passes_original_20pct'] = c['S'] <= .2
        c['passes_amended_25pct'] = c['S'] <= .25
    sources = {f"{config['source_run']}/{name}": file_hash(source / name)
               for name in ('freeze.json', 'summary.json', 'verification.json', 'runtime_resolution.json')}
    implementation = {name: file_hash(root / name) for name in
                      (CONFIG, PROTOCOL, 'src/video_compute_value/e000_acceptance.py')}
    freeze = {'version': config['version'], 'config': config, 'source_files': sources,
              'implementation_files': implementation, 'source_measurement_fingerprint': frozen['fingerprint'],
              'source_record_manifest_sha256': digest(record_hashes),
              'decision_timing': 'USER_AMENDED_AFTER_OBSERVATION', 'authority': AUTHORITY}
    freeze['fingerprint'] = digest(freeze)
    summary = {'version': config['version'], 'status': 'ACCEPTED_UNDER_AMENDED_25_PERCENT_RULE' if selected else
               'NOT_ACCEPTED_UNDER_AMENDED_25_PERCENT_RULE', 'good': selected is not None,
               'threshold': .25, 'inclusive': True, 'selected_policy': selected,
               'conditions': conditions, 'source_model_calls_reused': 132, 'source_primary_calls': 96,
               'new_model_calls': 0, 'integrity_passed': True,
               'original_20pct_status': saved['status'], 'original_20pct_selected_policy': original,
               'decision_timing': 'USER_AMENDED_AFTER_OBSERVATION',
               'confirmation_status': 'NOT_RUN', 'feasibility_confirmation_claimed': False,
               'E0_historical_verdict': 'MEASUREMENT_UNSTABLE',
               'E2B_status': 'NOT_ADMITTED_E2A_ADVANCEMENT', 'fingerprint': freeze['fingerprint']}
    proof = {'passed': True, 'source_records_checked': 132, 'primary_timings_recomputed': 96,
             'independent_timing_cells': 32, 'source_record_seals_checked': True,
             'source_fingerprint_checked': True, 'saved_aggregates_match': True,
             'original_20pct_decision_preserved': original is None,
             'amended_25pct_selected_policy': selected, 'new_model_calls': 0,
             'summary_sha256': digest(summary), 'fingerprint': freeze['fingerprint']}
    return freeze, summary, proof


def render_report(summary):
    lines = ['# E0: accepted under the amended 25% rule', '',
             '**GOOD — resident policy B passes ≤25% at both tested frame budgets.**', '',
             'The user explicitly changed the operational tolerance on 2026-11-04 after seeing the results. '
             'This is a retrospective acceptance amendment using the same verified measurements, '
             'not a new collection or an original 20% test passing.', '',
             '| Policy | Frames | Median relative range | Original ≤20% | Amended ≤25% |',
             '|---|---:|---:|---|---|']
    for c in summary['conditions']:
        lines.append(f"| {c['arm']} | {c['budget']} | {100*c['S']:.2f}% | "
                     f"{'PASS' if c['passes_original_20pct'] else 'FAIL'} | "
                     f"{'PASS' if c['passes_amended_25pct'] else 'FAIL'} |")
    lines += ['', 'Policy A fails at both budgets. Policy B passes both amended checks, so B is selected. '
              'The statistic is the median over eight source questions of (maximum−minimum)/median '
              'over three independent sessions. All 96 primary observations are retained.', '',
              'The source screen completed 132 real calls and 11 successful model loads. '
              'This decision record adds **zero model calls**. Seals, fingerprints and timing aggregates '
              'were rechecked from the saved records.', '',
              'The original `STABILIZATION_NOT_DEMONSTRATED` decision at 20%, the historical '
              '`MEASUREMENT_UNSTABLE` E0 verdict, and every raw trial remain unchanged. '
              'Full confirmation has not been run. This acceptance is operational; it establishes '
              'no latency-saving claim and does not change the E2A/E2B advancement result.', '',
              '![E0 original and amended acceptance](figures/acceptance25.png)', '']
    return '\n'.join(lines)


def create(root=ROOT, run=None):
    run = run or root / RUN
    require(not run.exists(), 'Decision directory already exists; use verify instead of overwriting')
    inventory = {p.relative_to(root).as_posix(): file_hash(p) for p in (root / 'results').rglob('*')
                 if p.is_file() and p.name not in ('.lock', '.worker.lock')}
    freeze, summary, proof = analyze(root)
    parent = root / 'results/e000_20261104_045654_runtime_recovery_v7/private/resource_ledger.json'
    ledger = copy.deepcopy(read_json(parent))
    require(all(s.get('finished_at') for s in ledger['sessions']), 'Parent accounting session still open')
    write_json(run / 'private/historical_inventory.json', inventory)
    write_json(run / 'private/resource_ledger.json', ledger)
    write_json(run / 'private/task.json', {'parent_path': parent.relative_to(root).as_posix(),
               'parent_sha256': file_hash(parent), 'inherited_seconds': ledger['active_seconds'],
               'last_checkpoint': utc_now(), 'initial_conservative_discovery_seconds': 600})
    write_json(run / 'authorization.json', {'date': '2026-11-04', 'human_instruction': AUTHORITY,
               'threshold': .25, 'decision_timing': 'USER_AMENDED_AFTER_OBSERVATION',
               'publication_authorized': True, 'additional_resources_granted': False})
    for name, value in (('freeze', freeze), ('summary', summary), ('verification', proof)):
        write_json(run / f'{name}.json', value)
    (run / 'report.md').write_text(render_report(summary), encoding='utf-8', newline='\n')
    write_json(root / '.local/e000_acceptance_latest_ledger.json',
               {'path': (run / 'private/resource_ledger.json').relative_to(root).as_posix()})
    write_json(root / '.local/e000_acceptance_run.json', {'run': run.relative_to(root).as_posix()})
    return summary


def verify(root=ROOT, run=None):
    run = run or root / RUN
    freeze, summary, proof = analyze(root)
    require(read_json(run / 'freeze.json') == freeze and read_json(run / 'summary.json') == summary
            and read_json(run / 'verification.json') == proof, 'Decision artifact changed or source drifted')
    unchanged = 0
    for name, expected in read_json(run / 'private/historical_inventory.json').items():
        require(file_hash(root / name) == expected, 'Historical evidence changed')
        unchanged += 1
    return {'passed': True, 'historical_files_unchanged': unchanged, 'source_records_checked': 132,
            'primary_timings_recomputed': 96, 'selected_policy': summary['selected_policy'],
            'threshold': .25, 'new_model_calls': 0}


def account(root=ROOT, run=None, *, close=False):
    from .resources import tree_bytes
    run = run or root / RUN
    task = read_json(run / 'private/task.json')
    require(file_hash(root / task['parent_path']) == task['parent_sha256'], 'Resource parent changed')
    ledger = read_json(run / 'private/resource_ledger.json')
    now = utc_now()
    elapsed = (datetime.now(timezone.utc)-datetime.fromisoformat(task['last_checkpoint'])).total_seconds()
    extra = task.pop('initial_conservative_discovery_seconds', 0)
    seconds = max(0, elapsed)+extra
    ledger['active_seconds'] += seconds
    ledger['sessions'].append({'name': '25% amendment analysis, tests, reporting and publication',
        'started_at': task['last_checkpoint'], 'finished_at': now, 'active_seconds': seconds,
        'conservative_initial_charge_seconds': extra, 'outcome': 'completed'})
    if close:
        require(not task.get('closed'), 'Amendment ledger already closed')
        ledger['active_seconds'] += 600
        ledger.setdefault('external_events', []).append({'name': 'Amendment final publication reserve',
            'active_seconds': 600, 'recorded_at': now})
        task['closed'] = True
    storage = tree_bytes(root)
    ledger['storage_current_bytes'] = storage
    ledger['storage_peak_bytes'] = max(ledger['storage_peak_bytes'], storage)
    traffic = ledger['video_transfer_bytes'] + ledger.get('interrupted_transfer_upper_bound_bytes', 0)
    require(ledger['active_seconds'] <= 172800 and storage <= 20*1024**3 and traffic <= 1024**3,
            'Cumulative resource cap exceeded')
    total = sum(s['active_seconds'] for s in ledger['sessions']) + sum(
        e['active_seconds'] for e in ledger.get('external_events', []))
    require(math.isclose(total, ledger['active_seconds'], abs_tol=1e-6), 'Resource ledger does not reconcile')
    task['last_checkpoint'] = utc_now()
    write_json(run / 'private/resource_ledger.json', ledger)
    write_json(run / 'private/task.json', task)
    value = {'at': now, 'inherited_seconds': task['inherited_seconds'],
             'cumulative_seconds': ledger['active_seconds'], 'authorized_limit_seconds': 172800,
             'storage_current_bytes': storage, 'storage_limit_bytes': 20*1024**3,
             'video_transfer_upper_bound_bytes': traffic, 'video_limit_bytes': 1024**3,
             'new_model_calls': 0, 'new_acquired_video_bytes': 0,
             'source_model_calls_reused': 132, 'publication_reserve_seconds': 600 if close else 0}
    write_json(run / 'resource_accounting.json', value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('create', 'verify', 'account', 'close'))
    args = parser.parse_args()
    if args.operation == 'create':
        print(create())
        print(account())
    elif args.operation == 'verify':
        print(verify())
    else:
        print(account(close=args.operation == 'close'))


if __name__ == '__main__':
    main()
