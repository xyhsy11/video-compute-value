"""Complete the resource-check migration; retain the exact native v5 measured path."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from video_compute_value.common import digest, file_hash, read_json, write_json

import recover_e000_runtime_v6 as v6

ROOT = v6.ROOT
REUSE = ROOT / 'results/e000_20261104_044416_runtime_recovery_v6'
_source_files, _cpu_phase = v6.source_files, v6.v5.cpu_phase


def source_files(root=ROOT):
    result = _source_files(root)
    for path in ('scripts/recover_e000_runtime_v7.py', 'protocols/e000_runtime_recovery_v7.md',
                 'configs/e000_runtime_recovery_v6.yaml'):
        result[path] = file_hash(root / path)
    return result


def initialize(run, task_start):
    parent, _ = v6.v5.latest_ledger()
    parent_run = parent.parent.parent if parent.parent.name == 'private' else parent.parent
    inherited_calls = read_json(parent_run / 'resource_accounting.json')['cumulative_model_invocations_upper_bound']
    v6.initialize(run, task_start)
    task = read_json(run / 'private/task.json')
    task['inherited_model_invocations_upper_bound'] = inherited_calls
    write_json(run / 'private/task.json', task)


def cpu_phase(task_run, run, frozen, budgets):
    if frozen['kind'] == 'diagnostic' and budgets == [8, 64] and not (run / 'private/cpu_preparations.json').exists():
        prior = read_json(REUSE / 'private/frozen.json')
        v6.v5.require(digest(prior['adapter']) == digest(frozen['adapter'])
                      and prior['cohort_hash'] == frozen['cohort_hash'], 'CPU reuse fingerprint mismatch')
        for name in ('model', 'video', 'protocol', 'generation'):
            key = f'src/video_compute_value/{name}.py'
            v6.v5.require(prior['source_files'][key] == frozen['source_files'][key], 'CPU preparation source mismatch')
        origin = REUSE / 'private/cpu_preparations.json'
        v6.v5.require(len(read_json(origin)) == 16, 'Incomplete CPU reuse source')
        (run / 'private/cpu_preparations.json').write_bytes(origin.read_bytes())
        write_json(run / 'private/cpu_reuse_provenance.json', {
            'source_run': REUSE.name, 'source_file_sha256': file_hash(origin),
            'preparation_fingerprint': digest(frozen['adapter']),
            'new_preparation_measurements': 0,
        })
    # The unchanged CPU worker verifies every reused seal and historical tensor /
    # metadata identity before skipping it. It never manufactures observations.
    return _cpu_phase(task_run, run, frozen, budgets)


def preflight(task_run, run, frozen, full=False, policy=None):
    specs = v6.v5.probe_schedule(frozen, full, policy)
    name = 'full_preflight' if full else 'preflight'
    raw, detail = v6.v5.historical_bound(frozen, specs)
    v6.v5.admit(task_run, frozen['config'], name, raw, **detail)
    v6.v5.run_workers(task_run, run, frozen, specs, timeout=240 if not full or policy == 'A' else 600)
    result = v6.verifier.verify_saved(run, ROOT, complete_phase=name)
    write_json(run / (name + '_verification.json'), result)
    v6.v5.require(result['passed'], 'Independent amended-resource preflight verification failed')


def report(run):
    result = v6.report(run)
    result['implementation_version'] = 'runtime_recovery_v7'
    result['prior_bounded_attempt'] = REUSE.name
    write_json(run / 'runtime_resolution.json', result)
    path = run / 'report.md'
    text = path.read_text(encoding='utf-8').replace('# E0 runtime recovery v6', '# E0 runtime recovery v7', 1)
    text += ('\nThe preceding v6 attempt is preserved: two model loads and 12 preflight calls succeeded; '
             'an unmigrated historical resource verifier stopped that attempt before primary timing collection. '
             'This run fixes only that routing, inherits every charge and invocation, and retains all scientific rules.\n')
    path.write_text(text, encoding='utf-8')
    return result


v6.CONFIG = v6.v5.CONFIG = Path('configs/e000_runtime_recovery_v7.yaml')
v6.v5.__file__ = str(Path(__file__).resolve())
v6.v5.sources, v6.v5.initialize = source_files, initialize
v6.v5.cpu_phase, v6.v5.preflight = cpu_phase, preflight


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('report', 'verify'):
        parser = argparse.ArgumentParser()
        parser.add_argument('operation')
        parser.add_argument('--run-dir', type=Path, required=True)
        args = parser.parse_args()
        v6.configure()
        if args.operation == 'report':
            print(report(args.run_dir.resolve()))
        else:
            result = v6.verifier.verify_saved(args.run_dir.resolve(), ROOT)
            print(result)
            raise SystemExit(0 if result['passed'] else 2)
    else:
        v6.v5.main()
