"""Frozen E1 preflight, sequential workers, resumable records and resource accounting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import yaml

from .common import (Blocked, append_json, atomic_text, command_output, digest, exclusive_lock,
                     file_hash, read_json, read_jsonl, utc_now, write_json)
from .e001_input import CONDITIONS, actions, finish_hashes, inference_video, install_decoder
from .protocol import PARSER_VERSION, PROMPT, question_input
from .resources import Ledger, environment, environment_fingerprint

EXPERIMENT = 'e001_additional_clip_headroom_v1'
DEFAULT_RUN = 'results/e001_20261029_132707_v1'
SOURCES = ('common.py', 'resources.py', 'video.py', 'model.py', 'measurement.py',
           'protocol.py', 'generation.py', 'e001_input.py', 'e001_analysis.py', 'e001.py', 'e001_verify.py')


def sources(root):
    paths = [Path('src/video_compute_value') / name for name in SOURCES]
    paths += [Path('configs/e001.yaml'), Path('protocols/e001_additional_clip_headroom_v1.md')]
    return {p.as_posix(): file_hash(root / p) for p in paths}


def configure(root):
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:native'
    os.environ.pop('PYTORCH_ALLOC_CONF', None)
    os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
    os.environ['HF_HOME'] = str(root / '.cache/huggingface')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TMP'] = os.environ['TEMP'] = str(root / '.cache/tmp')
    (root / '.cache/tmp').mkdir(parents=True, exist_ok=True)


def config_read(path):
    value = yaml.safe_load(path.read_text(encoding='utf-8'))
    expected = {'experiment_id': EXPERIMENT, 'cohort_videos': 8, 'questions_per_video': 3,
                'conditions': list(CONDITIONS), 'base_frames': 8, 'additional_frames': 8,
                'passes': 2, 'order_seed': 20261029, 'bootstrap_seed': 20261029,
                'bootstrap_resamples': 10000, 'safety_factor': 1.5}
    if any(value.get(k) != v for k, v in expected.items()):
        raise Blocked('INPUT_INTEGRITY_FAILED', 'E1 scientific configuration changed; use a new version')
    return value


def close_engineering(ledger):
    for session in ledger.value['sessions']:
        if session.get('finished_at') is None and session['name'].startswith('E1 implementation'):
            seconds = max(0., (datetime.now(timezone.utc)-datetime.fromisoformat(session['started_at'])).total_seconds())
            ledger.value['active_seconds'] += seconds
            session.update(finished_at=utc_now(), active_seconds=seconds, outcome='completed')
    write_json(ledger.path, ledger.value)


def work_window(ledger):
    ledger.value['sessions'].append({'name': 'E1 implementation, preparation and publication work window',
                                    'started_at': utc_now(), 'finished_at': None})
    write_json(ledger.path, ledger.value)


def resource_ledger(run, root):
    grant = read_json(run / 'runtime_resource_grant.json')
    if (grant.get('human_reply') != 'Authorize +2 hours; cumulative cap 27,600 seconds' or
        grant.get('authorized_limit_seconds') != 27600 or not grant.get('resource_only')):
        raise Blocked('BLOCKED_RESOURCE', 'Missing explicit numerical E1 resource grant')
    ledger = Ledger(run / 'private/resource_ledger.json', root,
        {'cumulative_execution_limit_hours': 27600/3600, 'additional_disk_limit_gib': 20,
         'video_acquisition_limit_gib': 1})
    if ledger.value['active_seconds'] < 20400:
        raise Blocked('BLOCKED_RESOURCE', 'E1 must inherit all historical consumption')
    pointer = root / '.local/e001_latest_ledger.json'
    if pointer.exists():
        latest_path = root / read_json(pointer)['path']
        if latest_path.resolve() != ledger.path.resolve():
            latest = read_json(latest_path)
            if ledger.value['active_seconds'] < latest['active_seconds']:
                raise Blocked('BLOCKED_RESOURCE', 'A newer cumulative ledger exists; inherit it first')
    write_json(pointer, {'path': ledger.path.relative_to(root).as_posix()})
    return ledger


def identity(run, fingerprint, source, question, condition, repetition):
    value = {'experiment_id': EXPERIMENT, 'run_id': run.name, 'fingerprint': fingerprint,
             'source_id': str(source), 'question_id': str(question),
             'condition': condition, 'repetition': repetition}
    return {**value, 'trial_id': digest(value)}


def make_order(cohort, seed=20261029):
    rng, result = random.Random(seed), []
    for repetition in (1, 2):
        items = [[v['source_id'], str(q['question_id']), c] for v in cohort['videos']
                 for q in v['questions'] for c in CONDITIONS]
        rng.shuffle(items)
        result.append({'repetition': repetition, 'requests': items})
    return result


def historical(root):
    return {p.relative_to(root).as_posix(): file_hash(p) for d in sorted((root/'results').glob('e000_*'))
            for p in sorted(d.rglob('*')) if p.is_file() and p.name not in ('.lock', '.worker.lock')}


def freeze(config, run, root):
    private = run / 'private'
    frozen_path = private / 'frozen.json'
    if frozen_path.exists():
        return validate(config, run, root, weights=True)
    parent = root / config['parent_run']
    cohort = read_json(parent / 'cohort_manifest.json')
    if len(cohort['videos']) != 8 or any(len(v['questions']) != 3 for v in cohort['videos']):
        raise Blocked('BLOCKED_INPUT', 'All eight exposed sources and all 24 questions are required')
    model = read_json(parent/'model_manifest.json')
    previous = yaml.safe_load((parent/'resolved_config.yaml').read_text(encoding='utf-8'))
    if (digest(model) != previous['model_manifest_hash'] or digest(cohort) != previous['cohort_hash'] or
        previous['prompt'] != PROMPT or previous['parser_version'] != PARSER_VERSION):
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Corrected E0 manifest or inference protocol changed')
    env = environment(root)
    if environment_fingerprint(env) != previous['environment_fingerprint']:
        raise Blocked('BLOCKED_ENVIRONMENT', 'E0 software, driver or GPU fingerprint changed')
    for name, sha in model['files'].items():
        if file_hash(Path(model['model_path'])/name) != sha:
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Immutable model cache changed')
    acquisition = {}
    for video in cohort['videos']:
        if file_hash(video['path']) != video['sha256']:
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Original E0 video missing or changed')
        acquisition[video['source_id']] = actions(video)
    order = make_order(cohort, config['order_seed'])
    frozen = {k: previous[k] for k in ('cpu_threads', 'seed', 'decoder_threads', 'decoder_thread_type',
                                     'cuda_trim_budgets', 'cuda_allocator_backend')}
    frozen.update(model)
    frozen.update(experiment_id=EXPERIMENT, run_id=run.name, frozen_at=utc_now(),
        scientific_config=config, config_input_hash=digest(config),
        cohort_hash=digest(cohort), actions_hash=digest(acquisition), order_hash=digest(order),
        environment_fingerprint=environment_fingerprint(env), source_files=sources(root),
        prompt=PROMPT, parser_version=PARSER_VERSION, model_manifest_hash=digest(model),
        timing_boundary='synchronized full native-video request including complete decoding; descriptive only',
        source_git_revision=command_output(['git', 'rev-parse', 'HEAD']).get('stdout'))
    frozen['fingerprint'] = digest(frozen)
    write_json(private/'cohort_manifest.json', cohort)
    write_json(private/'action_manifest.json', acquisition)
    write_json(private/'execution_order.json', order)
    write_json(private/'model_manifest.json', model)
    write_json(private/'environment.json', env)
    write_json(private/'historical_preservation.json', historical(root))
    for name in frozen['source_files']:
        path = private/'frozen_sources'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((root/name).read_bytes())
    write_json(frozen_path, frozen)
    atomic_text(private/'resolved_config.yaml', yaml.safe_dump(frozen, allow_unicode=True, sort_keys=True))
    write_json(run/'freeze.json', {k: frozen[k] for k in ('experiment_id', 'run_id', 'frozen_at',
        'fingerprint', 'config_input_hash', 'cohort_hash', 'actions_hash', 'order_hash', 'source_files',
        'source_git_revision', 'revision', 'processor_revision', 'tokenizer_revision', 'dtype', 'attention_backend')})
    return frozen


def validate(config, run, root, *, weights=False):
    frozen = read_json(run/'private/frozen.json')
    content = {k: v for k, v in frozen.items() if k != 'fingerprint'}
    if digest(content) != frozen['fingerprint'] or digest(config) != frozen['config_input_hash']:
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Scientific configuration or freeze changed')
    if sources(root) != frozen['source_files']:
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Scientific implementation changed; use a new named run/version')
    for name, key in (('cohort_manifest', 'cohort_hash'), ('action_manifest', 'actions_hash'),
                      ('execution_order', 'order_hash'), ('model_manifest', 'model_manifest_hash')):
        if digest(read_json(run/f'private/{name}.json')) != frozen[key]:
            raise Blocked('INPUT_INTEGRITY_FAILED', f'{name} changed')
    if environment_fingerprint(environment(root)) != frozen['environment_fingerprint']:
        raise Blocked('BLOCKED_ENVIRONMENT', 'Frozen environment changed')
    if weights:
        for name, sha in frozen['files'].items():
            if file_hash(Path(frozen['model_path'])/name) != sha:
                raise Blocked('INPUT_INTEGRITY_FAILED', 'Frozen model bytes changed')
        for video in read_json(run/'private/cohort_manifest.json')['videos']:
            if file_hash(video['path']) != video['sha256']:
                raise Blocked('INPUT_INTEGRITY_FAILED', 'Frozen video bytes changed')
    return frozen


def cpu_preflight(run, frozen, root):
    import torch
    from transformers import AutoProcessor
    from .model import QwenAdapter, tensor_hashes
    torch.set_num_threads(4)
    install_decoder()
    adapter = QwenAdapter.__new__(QwenAdapter)
    adapter.torch, adapter.frozen = torch, frozen
    adapter.processor = AutoProcessor.from_pretrained(frozen['model_path'], local_files_only=True,
                                                     trust_remote_code=False, use_fast=True)
    model_config = read_json(Path(frozen['model_path'])/'config.json')
    adapter.model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(
        max_position_embeddings=model_config['text_config']['max_position_embeddings'])))
    cohort = read_json(run/'private/cohort_manifest.json')
    acquisition = read_json(run/'private/action_manifest.json')
    old = [read_json(p) for p in (root/frozen['scientific_config']['parent_run']/'records').glob('*.json')]
    old = {(r['source_id'], str(r['question_id'])): r for r in old if r['budget'] == 8 and r['repetition'] == 1}
    checks = {}
    for video in cohort['videos']:
        question = video['questions'][0]
        by_condition = {}
        for condition in CONDITIONS:
            start = time.perf_counter()
            action = acquisition[video['source_id']][condition]
            tensors, details = adapter.prepare(question_input(question), inference_video(video, action), len(action['indices']))
            finish_hashes(details)
            hashes = tensor_hashes(tensors)
            by_condition[condition] = {**details, 'input_tensor_hashes': hashes,
                                       'elapsed_seconds': time.perf_counter()-start}
            del tensors
        base = by_condition['base']
        if base['input_tensor_hashes'] != old[(video['source_id'], str(question['question_id']))]['input_tensor_hashes']:
            raise Blocked('INPUT_INTEGRITY_FAILED', 'E1 base tensors differ from corrected E0')
        if any(v['retained_base_rgb_hashes'] != base['retained_base_rgb_hashes'] for v in by_condition.values()):
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Augmented inputs changed retained RGB evidence')
        if len({v['visual_tokens'] for k, v in by_condition.items() if k != 'base'}) != 1:
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Augmented visual budgets differ')
        checks[video['source_id']] = by_condition
        print(f"CPU inputs checked: {len(checks)}/8 sources", flush=True)
    write_json(run/'private/cpu_preflight.json', {'fingerprint': frozen['fingerprint'], 'checks': checks,
        'all_base_tensors_match_e0': True, 'retained_evidence_passed': True, 'matched_visual_budgets': True})
    return checks


def expected(run, frozen, cohort):
    return {row['trial_id']: row for v in cohort['videos'] for q in v['questions']
            for c in CONDITIONS for rep in (1, 2)
            for row in [identity(run, frozen['fingerprint'], v['source_id'], q['question_id'], c, rep)]}


def records(run, frozen, cohort):
    wanted, result = expected(run, frozen, cohort), {}
    for path in (run/'private/records').glob('*.json'):
        row = read_json(path)
        trial = row.get('trial_id')
        if (trial not in wanted or path.stem != trial or trial in result or
            any(row.get(k) != v for k, v in wanted[trial].items()) or
            row.get('record_hash') != digest({k: v for k, v in row.items() if k != 'record_hash'}) or
            row.get('status') != 'completed'):
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Invalid completed identity or record seal; never overwrite it')
        result[trial] = row
    return result


def worker(config, run, root, mode, index):
    from .measurement import MemorySampler, timed_request
    from .model import QwenAdapter, tensor_hashes
    with exclusive_lock(root/'.cache/cuda_worker.lock'):
        frozen = validate(config, run, root)
        install_decoder()
        cohort = read_json(run/'private/cohort_manifest.json')
        acquisition = read_json(run/'private/action_manifest.json')
        lookup = {(v['source_id'], str(q['question_id'])): (v, q)
                  for v in cohort['videos'] for q in v['questions']}
        adapter = QwenAdapter(frozen)
        worker_key = f'{mode}_{index}_{time.time_ns()}'
        write_json(run/f'private/workers/{worker_key}.json', {'key': worker_key, 'pid': os.getpid(),
            'started_at': utc_now(), 'load_seconds': adapter.load_seconds, 'baseline': adapter.baseline,
            'generation_policy': adapter.generation_policy, 'fingerprint': frozen['fingerprint'],
            'allocator_backend': adapter.torch.cuda.memory.get_allocator_backend()})
        completed = records(run, frozen, cohort)
        events = read_jsonl(run/'private/worker_calls.jsonl')

        def execute(source, question, condition, kind, repetition):
            ident = identity(run, frozen['fingerprint'], source, question, condition, repetition)
            if kind == 'trial':
                if ident['trial_id'] in completed:
                    return None
                attempts = sum(e['event'] == 'started' and e.get('kind') == 'trial' and
                               e.get('trial_id') == ident['trial_id'] for e in events)
                if attempts >= config['max_attempts_per_identity']:
                    raise Blocked('PARTIAL', 'Frozen per-identity retry allowance exhausted')
            video, labelled = lookup[(source, str(question))]
            action = acquisition[source][condition]
            call_id = f'{worker_key}_{time.time_ns()}'
            event = {'call_id': call_id, 'kind': kind, 'trial_id': ident['trial_id'],
                     'condition': condition, 'repetition': repetition}
            append_json(run/'private/worker_calls.jsonl', {**event, 'event': 'started', 'at': utc_now()})
            before = adapter.model_call_count
            try:
                row, tensors = timed_request(adapter, question_input(labelled), inference_video(video, action),
                    len(action['indices']), sampler=MemorySampler(.05))
                finish_hashes(row)
                row['input_tensor_hashes'] = tensor_hashes(tensors)
                del tensors
                row['input_metadata_hash'] = digest({k: row[k] for k in
                    ('tensor_shapes', 'source_indices', 'actual_timestamps', 'model_timestamps', 'video_grid_thw')})
                row.update(ident, call_id=call_id, worker_key=worker_key, kind=kind, status='completed')
                if kind == 'trial':
                    row['correct'] = row['parsed_answer'] == labelled['answer']
                gc.collect()
                row['live_allocated_after_bytes'] = adapter.torch.cuda.memory_allocated()
                row['record_hash'] = digest(row)
                if kind == 'trial':
                    path = run/f"private/records/{ident['trial_id']}.json"
                    if path.exists():
                        raise Blocked('INPUT_INTEGRITY_FAILED', 'Refusing to overwrite a completed record')
                    write_json(path, row)
                    completed[ident['trial_id']] = row
                else:
                    write_json(run/f'private/auxiliary/{call_id}.json', row)
                append_json(run/'private/worker_calls.jsonl', {**event, 'event': 'completed', 'at': utc_now(),
                    'model_invoked': adapter.model_call_count > before, 'record_hash': row['record_hash']})
                return row
            except BaseException as exc:
                failure = {**event, 'event': 'failed', 'at': utc_now(), 'error': type(exc).__name__,
                    'message': str(exc), 'status': getattr(exc, 'status', 'PARTIAL'),
                    'model_invoked': adapter.model_call_count > before, 'traceback': traceback.format_exc()}
                append_json(run/'private/worker_calls.jsonl', failure)
                append_json(run/'private/failed_attempts.jsonl', failure)
                raise

        first = cohort['videos'][0]
        for question in first['questions'][:2]:
            execute(first['source_id'], str(question['question_id']), 'base', 'warmup', 0)
        if mode == 'probe':
            rows = [execute(first['source_id'], str(first['questions'][0]['question_id']), c, 'probe', 0)
                    for c in CONDITIONS]
            write_json(run/f'private/probe_{index}.json', {'rows': rows, 'worker_key': worker_key,
                'load_seconds': adapter.load_seconds, 'fingerprint': frozen['fingerprint']})
        else:
            order = read_json(run/'private/execution_order.json')[index]
            for number, (source, question, condition) in enumerate(order['requests'], 1):
                execute(source, question, condition, 'trial', order['repetition'])
                if number % 12 == 0:
                    print(f"Pass {order['repetition']}: {number}/144 ordered requests visited", flush=True)


def launch(config_path, run, root, mode, index, timeout):
    log = run/f'private/{mode}_{index}_{time.time_ns()}.log'
    start = time.perf_counter()
    write_json(run/f'private/launches/{mode}_{index}_{time.time_ns()}.json',
               {'mode': mode, 'index': index, 'at': utc_now(), 'timeout_seconds': timeout})
    with log.open('w', encoding='utf-8') as output:
        result = subprocess.run([sys.executable, '-m', 'video_compute_value.e001', '_worker',
            '--config', str(config_path), '--run-dir', str(run), '--mode', mode, '--index', str(index)],
            cwd=root, stdout=output, stderr=output, timeout=timeout)
    if result.returncode:
        raise Blocked('PARTIAL', f'{mode} worker failed; preserved private log {log.name}')
    return time.perf_counter()-start


def preflight(config, run, root, ledger, config_path):
    frozen = freeze(config, run, root)
    if (run/'private/collection_ready.json').exists():
        raise Blocked('PARTIAL', 'Completed preflight is immutable; use run/resume')
    ledger.check(projected_seconds=600)
    checks = cpu_preflight(run, frozen, root)
    probes = []
    for index in (0, 1):
        if (run/f'private/probe_{index}.json').exists():
            raise Blocked('PARTIAL', 'Preserved probe exists; do not overwrite or selectively rerun')
        ledger.check(projected_seconds=300)
        wall = launch(config_path, run, root, 'probe', index, config['probe_timeout_seconds'])
        probe = read_json(run/f'private/probe_{index}.json')
        probe['wall_seconds'] = wall
        write_json(run/f'private/probe_receipt_{index}.json', probe)
        probes.append(probe)
        print(f'GPU probe {index+1}/2 finished in {wall:.1f}s', flush=True)
    reference_source = next(iter(checks))
    compared = ('input_tensor_hashes', 'input_metadata_hash', 'generated_token_ids', 'decoded_output',
                'parsed_answer', 'generation_policy_sha256', 'retained_base_rgb_hashes')
    for i, condition in enumerate(CONDITIONS):
        a, b = probes[0]['rows'][i], probes[1]['rows'][i]
        if any(a[k] != b[k] for k in compared):
            raise Blocked('REPEATABILITY_UNRESOLVED', 'Fresh-process probe inputs or outputs differ')
        if a['input_tensor_hashes'] != checks[reference_source][condition]['input_tensor_hashes']:
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Real GPU request differs from CPU input check')
    startup = max(p['wall_seconds']-sum(r['latency_seconds'] for r in p['rows']) for p in probes)
    request_estimate = 0.
    for source, by_condition in checks.items():
        for i, condition in enumerate(CONDITIONS):
            remainder = max(p['rows'][i]['latency_seconds']-p['rows'][i]['cpu_prepare_host_seconds'] for p in probes)
            request_estimate += 6*(by_condition[condition]['elapsed_seconds']+max(0., remainder))
    # Startup includes two warmups and diagnostic hashing/serialization from each probe worker.
    reserves = sum(config[k] for k in ('verification_reserve_seconds', 'retry_reserve_seconds', 'publication_reserve_seconds'))
    raw = request_estimate + 2*startup + 292*.10 + reserves
    forecast = {'fingerprint': frozen['fingerprint'], 'at': utc_now(), 'request_seconds': request_estimate,
        'two_startups_with_warmups_seconds': 2*startup, 'record_overhead_seconds': 292*.10,
        'verification_reserve_seconds': config['verification_reserve_seconds'],
        'retry_reserve_seconds': config['retry_reserve_seconds'],
        'publication_reserve_seconds': config['publication_reserve_seconds'],
        'raw_estimate_seconds': raw, 'safety_factor': 1.5, 'projection_seconds': raw*1.5,
        'all_input_fit_greedy_repeatability_checks_passed': True,
        'remaining_seconds_at_forecast': 27600-ledger.consumed_seconds(),
        'planned_collection_requests': 288, 'planned_collection_warmups': 4, 'completed_probe_and_warmup_calls': 16}
    write_json(run/'forecast.json', forecast)
    ledger.check(projected_seconds=forecast['projection_seconds'])
    write_json(run/'private/collection_ready.json', forecast)
    write_json(run/'status.json', {'status': 'READY', 'completed_collection_requests': 0,
        'planned_collection_requests': 288, 'fingerprint': frozen['fingerprint']})


def collect(config, run, root, ledger, config_path, resume):
    frozen = validate(config, run, root, weights=True)
    cohort = read_json(run/'private/cohort_manifest.json')
    ready = read_json(run/'private/collection_ready.json')
    if ready['fingerprint'] != frozen['fingerprint']:
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Collection-ready fingerprint mismatch')
    done = records(run, frozen, cohort)
    if not resume and (done or any(e.get('kind') == 'trial' for e in read_jsonl(run/'private/worker_calls.jsonl'))):
        raise Blocked('PARTIAL', 'Collection attempts exist; use resume')
    for index in (0, 1):
        for recovery in range(config['retry_processes_per_pass']+1):
            done = records(run, frozen, cohort)
            if sum(r['repetition'] == index+1 for r in done.values()) == 144:
                break
            events = read_jsonl(run/'private/worker_calls.jsonl')
            starts = {e['call_id'] for e in events if e['event'] == 'started' and e['kind'] == 'trial'}
            ends = {e['call_id'] for e in events if e['event'] == 'completed' and e['kind'] == 'trial'}
            if len(starts-ends) > config['max_failed_trial_attempts']:
                raise Blocked('PARTIAL', 'Global frozen trial-retry allowance exhausted')
            workers = list((run/'private/launches').glob(f'collect_{index}_*.json'))
            if len(workers) > config['retry_processes_per_pass']:
                raise Blocked('PARTIAL', 'Frozen process-recovery allowance exhausted')
            reserve = sum(config[k] for k in ('verification_reserve_seconds', 'retry_reserve_seconds', 'publication_reserve_seconds'))*1.5
            forecast = max(0, ready['projection_seconds']-reserve)*(288-len(done))/288+reserve
            ledger.check(projected_seconds=forecast)
            timeout = min(config['worker_timeout_seconds'], max(1, int(27600-ledger.consumed_seconds()-reserve)))
            try:
                wall = launch(config_path, run, root, 'collect', index, timeout)
                print(f'Collection pass {index+1} worker completed in {wall:.1f}s', flush=True)
            except (Blocked, subprocess.TimeoutExpired):
                if recovery == config['retry_processes_per_pass']:
                    raise
    from .e001_verify import analyze, verify
    analyze(config, run, root)
    verify(config, run, root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('preflight', 'run', 'resume', 'analyze', 'verify', 'account', 'close', '_worker'))
    parser.add_argument('--config', default='configs/e001.yaml')
    parser.add_argument('--run-dir', default=DEFAULT_RUN)
    parser.add_argument('--mode', choices=('probe', 'collect'))
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    root, run, config_path = Path.cwd(), Path(args.run_dir).resolve(), Path(args.config).resolve()
    configure(root)
    config = config_read(config_path)
    if args.operation == '_worker':
        worker(config, run, root, args.mode, args.index)
        return 0
    with exclusive_lock(run/'.lock'):
        ledger = resource_ledger(run, root)
        close_engineering(ledger)
        try:
            with ledger.track('E1 '+args.operation):
                if args.operation == 'preflight':
                    preflight(config, run, root, ledger, config_path)
                elif args.operation in ('run', 'resume'):
                    collect(config, run, root, ledger, config_path, args.operation == 'resume')
                elif args.operation in ('analyze', 'verify'):
                    from .e001_verify import analyze, verify
                    validate(config, run, root, weights=True)
                    (analyze if args.operation == 'analyze' else verify)(config, run, root)
        except BaseException as exc:
            write_json(run/'status.json', {'status': getattr(exc, 'status', 'PARTIAL'), 'at': utc_now(),
                'error_type': type(exc).__name__, 'planned_collection_requests': 288,
                'completed_collection_requests': len(list((run/'private/records').glob('*.json')))})
            append_json(run/'private/coordinator_failures.jsonl', {'at': utc_now(), 'error': str(exc),
                        'traceback': traceback.format_exc()})
            traceback.print_exc()
            return 1
        finally:
            write_json(run/'resource_accounting.json', {'at': utc_now(),
                'authoritative_resource_ledger': 'private/resource_ledger.json',
                'cumulative_seconds': ledger.value['active_seconds'], 'authorized_limit_seconds': 27600,
                'remaining_seconds': 27600-ledger.value['active_seconds'],
                'storage_current_bytes': ledger.value['storage_current_bytes'],
                'storage_peak_bytes': ledger.value['storage_peak_bytes'], 'storage_limit_bytes': 20*1024**3,
                'video_transfer_bytes': ledger.value['video_transfer_bytes'],
                'video_limit_bytes': 1024**3, 'new_downloads': 0})
            if args.operation != 'close' and 27600-ledger.value['active_seconds'] > 10:
                work_window(ledger)
        print(json.dumps(read_json(run/'resource_accounting.json')), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
