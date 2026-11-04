"""Versioned recovery over the unchanged v5 timing implementation and native adapter."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import sys

from video_compute_value.common import Blocked, digest, file_hash, read_json, read_jsonl, utc_now, write_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path('configs/e000_runtime_recovery_v6.yaml')
LIMIT = 172800


def import_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v5 = import_script('recovery_inherited_v5', 'stabilize_e000_timing_v5.py')
verifier = import_script('recovery_verifier_v6', 'verify_e000_runtime_v6.py')
_initialize, _account, _configure = v5.initialize, v5.account, v5.configure


def source_files(root=ROOT):
    # The presentation is not executed by model workers and has its own provenance.
    modules = ('__init__', 'acquire', 'common', 'dataset', 'e000', 'generation',
               'measurement', 'model', 'protocol', 'resources', 'video')
    paths = [f'src/video_compute_value/{name}.py' for name in modules]
    paths += [str(CONFIG).replace('\\', '/'), 'protocols/e000_runtime_recovery_v6.md',
              'protocols/e000_timing_stabilization_v5.md', 'scripts/stabilize_e000_timing_v5.py',
              'scripts/recover_e000_runtime_v6.py', 'scripts/verify_e000_timing_v5.py',
              'scripts/verify_e000_runtime_v6.py', 'scripts/report_e000_timing_v5.py']
    return {p: file_hash(root / p) for p in paths}


def inherit_presentation(value, presentation, presentation_sha256):
    """Retain the original ledger prefixes and add the supplemental charge once."""
    value = copy.deepcopy(value)
    if not any(e.get('presentation_parent_sha256') == presentation_sha256
               for e in value.get('external_events', [])):
        v5.require(value['active_seconds'] >= presentation['inherited_seconds'],
                   'Presentation parent is newer than the selected scientific ledger', 'BLOCKED_RESOURCE')
        charge = presentation['presentation_charged_seconds']
        value['active_seconds'] += charge
        value.setdefault('external_events', []).append({
            'name': 'Inherited VidFit presentation, including failed checks and publication reserves',
            'active_seconds': charge, 'recorded_at': presentation['at'],
            'presentation_parent_sha256': presentation_sha256,
        })
    return value


def initialize(run, task_start):
    _initialize(run, task_start)
    path = ROOT / '.local/vidfit_presentation_resource_ledger.json'
    presentation = read_json(path)
    parent_hash = file_hash(path)
    ledger_path = run / 'private/resource_ledger.json'
    ledger = read_json(ledger_path)
    before = ledger['active_seconds']
    ledger = inherit_presentation(ledger, presentation, parent_hash)
    write_json(ledger_path, ledger)
    task = read_json(run / 'private/task.json')
    task['inherited_seconds'] += ledger['active_seconds'] - before
    task.update(inherited_model_invocations_upper_bound=2851,
                supplemental_parent_sha256=parent_hash,
                supplemental_parent='private/presentation_parent.json')
    write_json(run / 'private/task.json', task)
    (run / 'private/presentation_parent.json').write_bytes(path.read_bytes())
    write_json(run / 'resource_grant.json', {
        'human_reply': 'Authorize 48 hours total', 'authorized_limit_seconds': LIMIT,
        'kind': 'Inherited resource-only grant; no additional allowance',
        'recovery_authority_date': '2026-11-04', 'new_model_or_system_repair': False,
    })
    write_json(ROOT / '.local/e000_runtime_recovery_run.json', {'run': run.relative_to(ROOT).as_posix()})


def configure():
    _configure()
    os.environ['PYTHONFAULTHANDLER'] = '1'


def account(run, close=False):
    value = _account(run, close)
    task = read_json(run / 'private/task.json')
    parent = run / task['supplemental_parent']
    v5.require(file_hash(parent) == task['supplemental_parent_sha256'], 'Supplemental parent changed')
    value['cumulative_model_invocations_upper_bound'] = (
        task['inherited_model_invocations_upper_bound'] + value['completed_model_calls']
        + value['unknown_model_invocations_upper_bound'])
    value['video_transfer_upper_bound_bytes'] = value['video_transfer_bytes'] + read_json(
        run / 'private/resource_ledger.json').get('interrupted_transfer_upper_bound_bytes', 0)
    write_json(run / 'resource_accounting.json', value)
    return value


def analyze(run):
    frozen = read_json(run / 'private/frozen.json')
    rows = v5.load_records(run, frozen['fingerprint'])
    summary = v5.summarize(rows) if frozen['kind'] == 'diagnostic' else v5.summarize(
        rows, (frozen['selected_policy'],), (8, 16, 32, 48, 64), 24)
    result = verifier.verify_saved(run, ROOT, complete_phase=frozen['kind'], check_summary=False)
    summary['integrity_passed'] = result['passed']
    summary['selected_policy'] = (v5.select_policy(summary, result['passed']) if frozen['kind'] == 'diagnostic'
                                  else frozen['selected_policy'])
    if frozen['kind'] == 'diagnostic':
        summary['status'] = ('ELIGIBLE_FOR_CONFIRMATION' if summary['selected_policy'] else
                             'STABILIZATION_NOT_DEMONSTRATED' if result['passed'] else 'INCOMPLETE_OR_INTEGRITY_FAILED')
    else:
        summary['status'] = (('FEASIBILITY_CONFIRMED' if all(c['S'] is not None and c['S'] <= .2
                              for c in summary['conditions']) else 'MEASUREMENT_UNSTABLE')
                             if result['passed'] else 'INCOMPLETE_OR_INTEGRITY_FAILED')
    write_json(run / 'verification.json', result)
    write_json(run / 'summary.json', summary)
    return summary


def confirmation(task_run, config):
    """The original conditional confirmation with a separately named v6 directory."""
    summary = analyze(task_run)
    policy = summary['selected_policy']
    v5.require(policy is not None, 'Neither policy qualified; no confirmation authorized', 'CONFIRMATION_NOT_RUN')
    v5.require(not (task_run / 'confirmation.json').exists(), 'Only one full confirmation attempt allowed')
    diagnostic = v5.validate(task_run)
    run = ROOT / ('results/e000_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_confirmation_v6')
    run.joinpath('private').mkdir(parents=True)
    frozen = copy.deepcopy(diagnostic)
    frozen.update(run_id=run.name, at=utc_now(), kind='confirmation', selected_policy=policy,
                  diagnostic_run=task_run.relative_to(ROOT).as_posix(), diagnostic_fingerprint=diagnostic['fingerprint'])
    cohort = read_json(task_run / 'private/cohort.json')
    order = v5.confirmation_schedule(cohort, policy, config['confirmation_seed'])
    frozen['schedule_hash'] = digest(order)
    frozen.pop('fingerprint')
    frozen['fingerprint'] = digest(frozen)
    for name, value in (('frozen', frozen), ('schedule', order), ('cohort', cohort)):
        write_json(run / f'private/{name}.json', value)
    for name in frozen['source_files']:
        destination = run / 'private/frozen_sources' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    write_json(run / 'freeze.json', {k: v for k, v in frozen.items() if k not in ('adapter', 'environment_fingerprint')})
    write_json(run / 'schedule.json', v5.anonymize_schedule(order, cohort))
    write_json(run / 'private/cpu_preparations.json', read_json(task_run / 'private/cpu_preparations.json'))
    link = {'run': run.relative_to(ROOT).as_posix(), 'selected_policy': policy,
            'shared_task_ledger': (task_run / 'private/resource_ledger.json').relative_to(ROOT).as_posix(),
            'status': 'CONFIRMATION_NOT_RUN'}
    write_json(task_run / 'confirmation.json', link)
    try:
        raw, detail = v5.historical_bound(frozen, v5.probe_schedule(frozen, True, policy))
        history = [read_json(p) for p in (ROOT / config['parent_run'] / 'records').glob('*.json')]
        worst = max(r['latency_seconds'] for r in history)
        cpu = 3 * sum(max(r['cpu_prepare_host_seconds'] for r in history
                         if r['source_id'] == v['source_id'] and r['budget'] == budget)
                      for v in cohort['videos'] for budget in (16, 32, 48)) + 30
        raw += 60 * len(order) + 416 * (worst + .25) + 180 + cpu
        v5.admit(task_run, config, 'confirmation_screen', raw, **detail,
                 additional_CPU_preparation_and_pixel_validation_seconds=cpu,
                 note='Complete conservative confirmation bound before additional CPU/GPU work')
        write_json(run / 'private/cpu_reuse_provenance.json', {
            'source_run': task_run.name, 'source_file_sha256': file_hash(task_run / 'private/cpu_preparations.json'),
            'preparation_fingerprint': digest(frozen['adapter'])})
        v5.cpu_phase(task_run, run, frozen, [8, 16, 32, 48, 64])
        v5.preflight(task_run, run, frozen, True, policy)
        link['status'] = v5.collect(task_run, run, frozen)['status']
    except Blocked as exc:
        link.update(status='CONFIRMATION_NOT_RUN', reason=exc.status)
        write_json(run / 'status.json', link)
    finally:
        write_json(task_run / 'confirmation.json', link)
    return link


def report(run):
    summary, resource = read_json(run / 'summary.json'), read_json(run / 'resource_accounting.json')
    records = v5.load_records(run, read_json(run / 'private/frozen.json')['fingerprint'])
    workers = list((run / 'private/workers').glob('*/receipt.json'))
    preflight = [r for r in records if r['phase'] == 'preflight']
    verification = read_json(run / 'verification.json')
    failed_loads = [f for f in read_jsonl(run / 'private/failures.jsonl')
                    if f.get('kind') == 'worker' and f.get('worker_id') != 'cpu_preparation'
                    and not (run / 'private/workers' / f['worker_id'] / 'receipt.json').exists()]
    runtime = ('BLOCKED_MODEL_INITIALIZATION' if failed_loads else
               'AVAILABLE_CURRENT_RUN' if len(preflight) == 12 else 'INCOMPLETE_OR_BLOCKED')
    resolution = {
        'run_id': run.name, 'status': runtime, 'historical_failure_preserved': True,
        'successful_model_loads': len(workers), 'preflight_completed': len(preflight),
        'failed_model_initializations': len(failed_loads),
        'diagnostic_completed': sum(r['phase'] == 'diagnostic' for r in records),
        'planned_diagnostic_total': 132, 'timing_status': summary['status'],
        'selected_policy': summary['selected_policy'], 'verified': verification['passed'],
        'root_cause': 'NOT_ESTABLISHED', 'packages_or_driver_changed': False,
        'historical_E0': 'MEASUREMENT_UNSTABLE',
    }
    if (run / 'confirmation.json').exists():
        child = read_json(run / 'confirmation.json')
        resolution['confirmation_status'] = child['status']
    else:
        resolution['confirmation_status'] = 'CONFIRMATION_NOT_RUN'
    reporting = import_script('recovery_figures', 'report_e000_timing_v5.py')
    reporting.figures(run, summary)
    rows = '\n'.join(f"| {c['arm']} | {c['budget']} | {c['primary_requests']} | "
                      f"{100*c['S']:.2f}% |" if c['S'] is not None else
                      f"| {c['arm']} | {c['budget']} | {c['primary_requests']} | unavailable |"
                      for c in summary['conditions'])
    text = (f'# E0 runtime recovery v6\n\nCurrent runtime: **{runtime}**. '
            f"Timing screen: **{summary['status']}**.\n\n"
            'The old 0xC0000005 initialization failure remains immutable evidence. '
            'This is a separately frozen attempt with the same native adapter, local checkpoint, '
            'software, inputs, generation controls and timing boundary. No package or driver was changed. '
            'Successful new loads do not establish the cause of the historical crash.\n\n'
            f"Completed {len(records)}/132 screen calls, including {len(preflight)}/12 preflight calls. "
            f"Selected policy: {summary['selected_policy']}. Confirmation: {resolution['confirmation_status']}.\n\n"
            '| Arm | Frames | Primaries | Median relative repetition range |\n'
            '|---|---:|---:|---:|\n' + rows + '\n\n'
            'Qualification requires ≤20% at both budgets and every integrity gate. '
            'A is selected before B; no replacement sessions or exclusions. Historical E0 remains '
            '**MEASUREMENT_UNSTABLE**. E2B remains **NOT_ADMITTED_E2A_ADVANCEMENT**.\n\n'
            '![New timing screen](figures/diagnostic_stability.png)\n\n'
            f"Independent saved-record verification: {verification['passed']}; "
            f"historical files unchanged: {verification['historical_files_unchanged']}.\n\n"
            f"Cumulative resource charge: {resource['cumulative_seconds']:.3f} / 172800 seconds. "
            'Every historical charge, including presentation work, is inherited; no new video acquisition.\n')
    reporting.public_safe(resolution)
    reporting.public_safe(text)
    write_json(run / 'runtime_resolution.json', resolution)
    (run / 'report.md').write_text(text, encoding='utf-8')
    return resolution


v5.CONFIG, v5.LIMIT = CONFIG, LIMIT
v5.RESOURCE = {'cumulative_execution_limit_hours': 48, 'additional_disk_limit_gib': 20,
               'video_acquisition_limit_gib': 1}
v5.__file__ = str(Path(__file__).resolve())  # Spawn children through this versioned entry point.
v5.sources, v5.initialize, v5.configure = source_files, initialize, configure
v5.account, v5.analyze, v5.confirmation = account, analyze, confirmation


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('report', 'verify'):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('operation')
        parser.add_argument('--run-dir', type=Path, required=True)
        args = parser.parse_args()
        configure()
        if args.operation == 'report':
            print(report(args.run_dir.resolve()))
        else:
            result = verifier.verify_saved(args.run_dir.resolve(), ROOT)
            print(result)
            raise SystemExit(0 if result['passed'] else 2)
    else:
        v5.main()
