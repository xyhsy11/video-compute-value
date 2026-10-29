"""E2 gated checkpoint CLI. It never launches scientific collection through failed gates."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

import yaml

from .common import (Blocked, append_json, digest, exclusive_lock, file_hash, read_json,
                     read_jsonl, utc_now, write_json)
from .e002_accounting import checkpoint, forecast
from .e002_features import SCHEMA, extract
from .e002_input import inventory, menu

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = 'e002_preacquisition_predictability_v1'


def source_hashes(root):
    paths = list((root/'src/video_compute_value').glob('e002*.py'))
    paths += [root/'configs/e002.yaml', root/'protocols/e002_preacquisition_predictability_v1.md']
    paths += [root/'src/video_compute_value'/name for name in
              ('model.py', 'generation.py', 'video.py', 'protocol.py', 'measurement.py', 'e001_input.py',
               'common.py', 'resources.py')]
    return {p.relative_to(root).as_posix(): file_hash(p) for p in paths}


def implementation_freeze(root, run):
    content = {'experiment_id': EXPERIMENT, 'source_files': source_hashes(root),
               'config': yaml.safe_load((root/'configs/e002.yaml').read_text(encoding='utf-8')),
               'feature_schema': SCHEMA}
    result = {**content, 'fingerprint': digest(content)}
    path = run/'implementation_freeze.json'
    if path.exists() and read_json(path) != result:
        raise Blocked('INPUT_INTEGRITY_FAILED', 'E2 implementation drift: use a new named version/run')
    write_json(path, result)
    write_json(run/'feature_schema.json', SCHEMA)
    return result


def probe_worker(root, run):
    from .e001 import configure
    configure(root)
    from .e001_input import finish_hashes, inference_video, install_decoder
    from .e002_runtime import E2Adapter, observation
    from .measurement import timed_request
    from .model import tensor_hashes
    from .protocol import question_input
    spec = read_json(run/'private/compatibility_spec.json')
    if source_hashes(root) != spec['source_files']:
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Probe fingerprint changed')
    with exclusive_lock(root/'.cache/cuda_worker.lock'):
        receipt = {'stage': 'model_loading', 'pid': __import__('os').getpid(), 'at': utc_now(),
                   'fingerprint': spec['fingerprint'], 'completed_model_calls': 0}
        write_json(run/'private/compatibility_worker.json', receipt)
        install_decoder()
        frozen = read_json(root/'results/e001_20261029_132707_v1/private/frozen.json')
        adapter = E2Adapter(frozen)
        receipt.update(stage='generation', load_seconds=adapter.load_seconds,
                       generation_policy=adapter.generation_policy)
        write_json(run/'private/compatibility_worker.json', receipt)
        cohort = read_json(root/'results/e001_20261029_132707_v1/private/cohort_manifest.json')
        video = next(v for v in cohort['videos'] if v['source_id'] == spec['source_id'])
        by_question = {str(q['question_id']): q for q in video['questions']}
        outputs = {}
        for index, call in enumerate(spec['calls']):
            append_json(run/'private/compatibility_calls.jsonl', {'event': 'started', 'ordinal': index,
                        'at': utc_now(), 'condition': call['condition']})
            question = question_input(by_question[call['question_id']])
            action = menu(video)[call['condition']]
            adapter.capture_scores = call['condition'] == 'base_observed'
            row, inputs = timed_request(adapter, question, inference_video(video, action), len(action['indices']))
            if adapter.capture_scores:
                row['features'] = {k: v.tolist() for k, v in extract(observation(question, video, row)).items()}
            finish_hashes(row)
            row['input_tensor_hashes'] = tensor_hashes(inputs)
            del inputs
            row.update(ordinal=index, condition=call['condition'], kind=call['kind'])
            write_json(run/f'private/compatibility_records/{index}.json', row)
            append_json(run/'private/compatibility_calls.jsonl', {'event': 'completed', 'ordinal': index,
                        'at': utc_now(), 'condition': call['condition']})
            if call['kind'] == 'integrity':
                outputs[call['condition']] = row
            receipt['completed_model_calls'] = adapter.model_call_count
            write_json(run/'private/compatibility_worker.json', receipt)
        a, b = outputs['base_plain'], outputs['base_observed']
        compared = ('generated_token_ids', 'parsed_answer', 'input_tensor_hashes', 'generation_policy_sha256')
        if any(a[k] != b[k] for k in compared) or 'C' not in b:
            raise Blocked('BLOCKED_FEATURE_INTEGRITY', 'Score capture changed paid base greedy output/input')
        receipt.update(stage='complete', feature_integrity_passed=True, native_8_16_passed=True)
        write_json(run/'private/compatibility_worker.json', receipt)


def preflight(root, run):
    frozen = implementation_freeze(root, run)
    launch = run/'private/compatibility_launch.json'
    if launch.exists():
        receipt = read_json(run/'compatibility.json')
        if receipt['status'] != 'PASSED':
            raise Blocked(receipt['status'], 'Preserved bounded probe failed; automatic retry is forbidden')
        return receipt
    _, accounting = checkpoint(root, run, 'Bounded E2 compatibility admission')
    required = 1.5*90+300
    if accounting['remaining_seconds'] < required:
        raise Blocked('BLOCKED_RESOURCE', '90-second bounded probe plus safety/closeout reserve does not fit')
    cohort = read_json(root/'results/e001_20261029_132707_v1/private/cohort_manifest.json')
    video = cohort['videos'][0]
    questions = [str(q['question_id']) for q in video['questions']]
    calls = [{'kind': 'warmup', 'condition': 'base_plain', 'question_id': q} for q in questions[:2]]
    calls += [{'kind': 'integrity', 'condition': c, 'question_id': questions[0]}
              for c in ('base_plain', 'base_observed', 'quarter1')]
    spec = {'fingerprint': frozen['fingerprint'], 'source_files': frozen['source_files'],
            'source_id': video['source_id'], 'calls': calls, 'timeout_seconds': 90,
            'purpose': 'legacy compatibility only; excluded from E2 scientific results'}
    write_json(run/'private/compatibility_spec.json', spec)
    write_json(launch, {'at': utc_now(), 'planned_calls': 5, 'timeout_seconds': 90,
                        'fingerprint': frozen['fingerprint']})
    log = run/'private/compatibility.log'
    started = time.perf_counter()
    timed_out = False
    with log.open('w', encoding='utf-8') as output:
        try:
            result = subprocess.run([sys.executable, '-m', 'video_compute_value.e002', '_probe_worker',
                '--run-dir', str(run)], cwd=root, stdout=output, stderr=output, timeout=90)
            code = result.returncode
        except subprocess.TimeoutExpired:
            timed_out, code = True, None  # subprocess.run kills/reaps this owned worker.
    worker_path = run/'private/compatibility_worker.json'
    worker = read_json(worker_path) if worker_path.exists() else {}
    events = read_jsonl(run/'private/compatibility_calls.jsonl')
    started_calls = {e['ordinal'] for e in events if e['event'] == 'started'}
    completed = {e['ordinal'] for e in events if e['event'] == 'completed'}
    status = 'PASSED' if code == 0 and worker.get('feature_integrity_passed') else (
        'BLOCKED_MODEL_RUNTIME' if worker.get('stage') in (None, 'model_loading') or timed_out else
        'BLOCKED_FEATURE_INTEGRITY')
    receipt = {'status': status, 'at': utc_now(), 'returncode': code,
        'windows_exit_hex': f'0x{code & 0xffffffff:08X}' if code is not None else None,
        'timed_out': timed_out, 'wall_seconds': time.perf_counter()-started,
        'planned_compatibility_calls': 5, 'completed_model_calls': len(completed),
        'unknown_model_calls_upper_bound': len(started_calls-completed),
        'feature_integrity_passed': worker.get('feature_integrity_passed', False),
        'native_8_16_passed': worker.get('native_8_16_passed', False),
        'worker_reaped': True, 'automatic_retries': 0, 'log_sha256': file_hash(log)}
    write_json(run/'compatibility.json', receipt)
    checkpoint(root, run, 'Bounded compatibility attempt including failure and cleanup')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('audit', 'preflight', 'resume', 'verify', 'report', 'account',
                                             'close', '_probe_worker'))
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    root, run = ROOT, args.run_dir.resolve()
    if not run.is_relative_to(root/'results') or not (run/'private/task.json').exists():
        parser.error('Target must identify the initialized E2 results directory')
    if args.operation == '_probe_worker':
        probe_worker(root, run)
        return 0
    with exclusive_lock(run/'.lock'):
        try:
            if args.operation == 'resume':
                checkpoint(root, run, 'Resume E2 work window')
                implementation_freeze(root, run)
            if args.operation in ('audit', 'resume'):
                print(inventory(root, run))
            if args.operation in ('preflight', 'resume'):
                print(preflight(root, run))
            if args.operation == 'verify':
                from .e002_verify import verify_checkpoint
                print(verify_checkpoint(root, run))
            if args.operation == 'report':
                sys.path.insert(0, str(root/'scripts'))
                from report_e002 import report
                report(root, run)
        except Blocked as exc:
            append_json(run/'private/coordinator_failures.jsonl', {'at': utc_now(), 'status': exc.status, 'message': str(exc)})
            print(f'{exc.status}: {exc}', file=sys.stderr)
            return 2
        finally:
            _, accounting = checkpoint(root, run, 'E2 '+args.operation,
                                       reserve=300 if args.operation == 'close' else 0)
            forecast(run, accounting)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
