"""Bounded prospective E0 worker-lifecycle screen; never changes historical runs."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import yaml

from video_compute_value.common import (Blocked, append_json, command_output, digest, exclusive_lock,
                                        file_hash, read_json, read_jsonl, utc_now, write_json)
from video_compute_value.protocol import question_input, video_input
from video_compute_value.resources import Ledger, environment, environment_fingerprint

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path('configs/e000_timing_stabilization_v5.yaml')
LIMIT = 27600
RESOURCE = {'cumulative_execution_limit_hours': LIMIT / 3600, 'additional_disk_limit_gib': 20,
            'video_acquisition_limit_gib': 1}
META = ('tensor_shapes', 'source_indices', 'actual_timestamps', 'model_timestamps', 'video_grid_thw')
INPUT_FIELDS = (*META, 'input_tokens', 'visual_tokens', 'frames_decoded', 'frames_selected',
                'frames_processor', 'frames_represented', 'temporal_padding', 'input_tensor_hashes')
OUTPUT_FIELDS = ('generated_token_ids', 'decoded_output', 'parsed_answer', 'valid_answer',
                 'output_tokens', 'generation_policy_sha256')


def require(condition, message, status='INPUT_INTEGRITY_FAILED'):
    if not condition:
        raise Blocked(status, message)


def configure():
    # Called before importing Torch, also in every fresh Windows subprocess.
    from video_compute_value.e000 import set_local_paths
    set_local_paths(ROOT)
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:native'
    os.environ.pop('PYTORCH_ALLOC_CONF', None)
    os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'


def sources(root=ROOT):
    paths = sorted((root / 'src/video_compute_value').glob('*.py'))
    paths += [root / CONFIG, root / 'protocols/e000_timing_stabilization_v5.md',
              root / 'scripts/stabilize_e000_timing_v5.py', root / 'scripts/verify_e000_timing_v5.py']
    return {p.relative_to(root).as_posix(): file_hash(p) for p in paths}


def initialize(run, task_start):
    require(not run.exists(), 'New task directory must not already exist')
    parent, previous = latest_ledger()
    require(command_output(['git', 'remote', 'get-url', 'origin'])['stdout'] ==
            'https://github.com/xyhsy11/video-compute-value.git', 'Wrong repository remote')
    inventory = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in (ROOT / 'results').rglob('*')
                 if p.is_file() and p.name not in ('.lock', '.worker.lock')}
    value = copy.deepcopy(previous)
    now = utc_now()
    seconds = (datetime.fromisoformat(now) - datetime.fromisoformat(task_start)).total_seconds()
    require(seconds >= 0, 'Task start must be a past timezone-aware timestamp')
    value['active_seconds'] += seconds
    value['sessions'].append({'name': 'Initial task setup and inspection', 'started_at': task_start,
                              'finished_at': now, 'active_seconds': seconds, 'outcome': 'completed'})
    write_json(run / 'private/historical_inventory.json', inventory)
    write_json(run / 'private/resource_ledger.json', value)
    write_json(run / 'private/task.json', {'task_started_at': task_start, 'last_checkpoint': now,
               'parent_path': parent.relative_to(ROOT).as_posix(), 'parent_sha256': file_hash(parent),
               'inherited_seconds': previous['active_seconds'], 'historical_files': len(inventory),
               'inherited_model_invocations_upper_bound': 1762,
               'initial_git': {k: command_output(['git', *v]) for k, v in {
                   'head': ['rev-parse', 'HEAD'], 'branch': ['branch', '--show-current'],
                   'upstream': ['rev-parse', '--abbrev-ref', '@{upstream}'], 'status': ['status', '--porcelain']}.items()}})
    write_json(ROOT / '.local/e000_timing_latest_ledger.json',
               {'path': (run / 'private/resource_ledger.json').relative_to(ROOT).as_posix()})


def latest_ledger(root=ROOT):
    paths = set((root / 'results').rglob('resource_ledger.json'))
    for pointer in (root / '.local').glob('*latest*ledger*.json'):
        target = (root / read_json(pointer)['path']).resolve()
        require(target.is_relative_to(root.resolve()) and target.exists(), 'Invalid latest-ledger pointer')
        paths.add(target)
    require(bool(paths), 'No cumulative parent ledger', 'BLOCKED_RESOURCE')
    candidates = [(read_json(p)['active_seconds'], p, read_json(p)) for p in paths]
    _, path, value = max(candidates, key=lambda x: x[0])
    # A cumulative descendant must retain prior sessions and external events in order.
    for seconds, other, old in candidates:
        if other.resolve() == path.resolve():
            continue
        require(seconds <= value['active_seconds'], 'Nonmonotonic cumulative accounting')
        for key in ('sessions', 'external_events'):
            previous = old.get(key, [])
            current = value.get(key, [])
            # Historical repaired-interruption siblings are documented immutable evidence.
            # Only the authoritative E1/current pointers establish the current chain.
            if other.resolve() in {(root / read_json(p)['path']).resolve()
                                   for p in (root / '.local').glob('*latest*ledger*.json')}:
                require(current[:len(previous)] == previous, 'Conflicting authoritative ledger lineage')
    require(all(s.get('finished_at') for s in value['sessions']), 'Parent accounting session still open')
    return path, value


def ledger_for(run):
    task = read_json(run / 'private/task.json')
    ledger = Ledger(run / 'private/resource_ledger.json', ROOT, RESOURCE)
    parent = ROOT / task['parent_path']
    require(file_hash(parent) == task['parent_sha256'], 'Parent ledger changed')
    require(ledger.value['active_seconds'] >= task['inherited_seconds'], 'Inherited consumption reset')
    latest, value = latest_ledger()
    require(latest.resolve() == ledger.path.resolve() or value['active_seconds'] <= ledger.value['active_seconds'],
            'Newer descendant ledger exists', 'BLOCKED_RESOURCE')
    return ledger


def checkpoint(run, name):
    ledger = ledger_for(run)
    task = read_json(run / 'private/task.json')
    now = utc_now()
    elapsed = max(0, (datetime.fromisoformat(now) - datetime.fromisoformat(task['last_checkpoint'])).total_seconds())
    ledger.value['active_seconds'] += elapsed
    ledger.value['sessions'].append({'name': name, 'started_at': task['last_checkpoint'], 'finished_at': now,
                                     'active_seconds': elapsed, 'outcome': 'completed'})
    ledger.save(force_storage=True)
    task['last_checkpoint'] = now
    write_json(run / 'private/task.json', task)
    return ledger


def admission(remaining, raw, config, **detail):
    total = config['safety_factor'] * raw + config['interruption_contingency_seconds'] + config['closeout_reserve_seconds']
    return {**detail, 'raw_execution_and_overhead_seconds': raw, 'safety_factor': config['safety_factor'],
            'interruption_contingency_seconds': config['interruption_contingency_seconds'],
            'closeout_reserve_seconds': config['closeout_reserve_seconds'], 'required_seconds': total,
            'remaining_seconds': remaining, 'admitted': total <= remaining,
            'additional_cumulative_allowance_needed_seconds': max(0, math.ceil(total - remaining))}


def admit(run, config, name, raw, **detail):
    ledger = checkpoint(run, 'Implementation/inspection and phase admission: ' + name)
    forecast = admission(LIMIT - ledger.value['active_seconds'], raw, config, phase=name, at=utc_now(), **detail)
    write_json(run / (name + '_forecast.json'), forecast)
    ledger.check(projected_seconds=forecast['required_seconds'], projected_bytes=32 * 1024**2)
    require(forecast['admitted'], name + ' exceeds complete protected forecast', 'BLOCKED_RESOURCE')
    return forecast


def diagnostic_schedule(cohort, seed=20261029):
    rng = random.Random(seed)
    pairs = [[v['source_id'], str(v['questions'][0]['question_id'])] for v in cohort['videos']]
    schedules = []
    for repetition in (1, 2, 3):
        arms, budgets = (['A', 'B'], [8, 64]) if repetition == 1 else (['B', 'A'], [64, 8])
        if repetition == 3:
            arms = rng.sample(['A', 'B'], 2)
            budgets = rng.sample([8, 64], 2)
        order = rng.sample(pairs, len(pairs))
        for arm in arms:
            blocks = [{'budget': b, 'pairs': order} for b in budgets]
            groups = [[b] for b in blocks] if arm == 'A' else [blocks]
            for group in groups:
                schedules.append({'worker_id': f'diagnostic_{repetition}_{arm}_{group[0]["budget"]}',
                                  'phase': 'diagnostic', 'arm': arm, 'repetition': repetition, 'blocks': group})
    return schedules


def confirmation_schedule(cohort, policy, seed=20261030):
    require(policy in ('A', 'B'), 'Unknown confirmation policy')
    rng = random.Random(seed)
    pairs = [[v['source_id'], str(q['question_id'])] for v in cohort['videos'] for q in v['questions']]
    # Rejection of schedules only, before outcomes; include 64 last and 64 -> lower.
    while True:
        orders = [rng.sample([8, 16, 32, 48, 64], 5) for _ in range(3)]
        if any(o[-1] == 64 for o in orders) and any(o[-1] != 64 for o in orders):
            break
    workers = []
    for repetition, budgets in enumerate(orders, 1):
        order = rng.sample(pairs, len(pairs))
        blocks = [{'budget': b, 'pairs': order} for b in budgets]
        for group in ([[b] for b in blocks] if policy == 'A' else [blocks]):
            workers.append({'worker_id': f'confirmation_{repetition}_{policy}_{group[0]["budget"]}',
                            'phase': 'confirmation', 'arm': policy, 'repetition': repetition, 'blocks': group})
    workers.append({'worker_id': 'confirmation_text', 'phase': 'confirmation', 'arm': policy,
                    'repetition': 1, 'blocks': [{'budget': 0, 'pairs': rng.sample(pairs, len(pairs))}]})
    return workers


def probe_schedule(frozen, full=False, policy=None):
    first = frozen['adapter']['profile_question']
    orders = [[8, 64], [64, 8]]
    if full:
        orders = [[b] for b in (8, 16, 32, 48, 64)] if policy == 'A' else [
            [8, 16, 32, 48, 64], [64, 48, 32, 16, 8]]
    return [{'worker_id': f'{"full" if full else "pilot"}_probe_{i}', 'phase': 'full_preflight' if full else 'preflight',
             'arm': policy or 'probe', 'repetition': 0,
             'blocks': [{'budget': b, 'pairs': [first, first] if full else [], 'profile': first} for b in order]}
            for i, order in enumerate(orders)]


def identity(fingerprint, spec, budget, pair, kind, ordinal):
    value = {'fingerprint': fingerprint, 'worker_id': spec['worker_id'], 'phase': spec['phase'],
             'arm': spec['arm'], 'repetition': spec['repetition'], 'budget': budget,
             'source_id': pair[0], 'question_id': str(pair[1]), 'kind': kind, 'ordinal': ordinal}
    return {**value, 'call_id': digest(value)}


def expected_calls(frozen, spec):
    calls = []
    for block in spec['blocks']:
        for ordinal, pair in enumerate(frozen['adapter']['two_warmups']):
            calls.append(identity(frozen['fingerprint'], spec, block['budget'], pair, 'warmup', ordinal))
        for ordinal, pair in enumerate(block['pairs']):
            kind = 'integrity' if 'preflight' in spec['phase'] else 'primary'
            calls.append(identity(frozen['fingerprint'], spec, block['budget'], pair, kind, ordinal))
        if block.get('profile'):
            calls.append(identity(frozen['fingerprint'], spec, block['budget'], block['profile'], 'profile', 0))
    return calls


def seal_record(path, value):
    require(not path.exists(), 'Completed identity is immutable')
    write_json(path, {**value, 'seal': digest(value)})


def load_records(run, fingerprint):
    rows = []
    for path in sorted((run / 'private/records').glob('*.json')):
        row = read_json(path)
        seal = row.pop('seal')
        require(digest(row) == seal and row['fingerprint'] == fingerprint and path.stem == row['call_id'],
                'Corrupt record, changed fingerprint or identity')
        rows.append(row)
    require(len({r['call_id'] for r in rows}) == len(rows), 'Duplicate records')
    return rows


def history_reference(parent):
    rows = [read_json(p) for p in (parent / 'records').glob('*.json')]
    reference = {}
    for row in rows:
        key = row['source_id'], str(row['question_id']), row['budget']
        if key in reference:
            require(all(row[k] == reference[key][k] for k in (*INPUT_FIELDS, *OUTPUT_FIELDS)),
                    'Historical E0 repeats disagree')
        reference[key] = row
    require(len(rows) == 384 and len(reference) == 144, 'Missing corrected E0 reference', 'BLOCKED_INPUT')
    return reference


def telemetry():
    import psutil
    battery = psutil.sensors_battery()
    record = {'at': utc_now(), 'available_ram_bytes': psutil.virtual_memory().available,
              'ac': None if battery is None else battery.power_plugged,
              'power_profile': command_output(['powercfg', '/getactivescheme']),
              'gpu': command_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu,clocks.gr,clocks.mem,power.draw', '--format=csv,noheader,nounits'])}
    require(record['ac'] is not False, 'Battery operation unsuitable', 'BLOCKED_ENVIRONMENT')
    require(record['available_ram_bytes'] >= 3 * 1024**3, 'Less than 3 GiB available RAM', 'BLOCKED_ENVIRONMENT')
    return record


def validate(run, weights=False):
    frozen = read_json(run / 'private/frozen.json')
    require(digest({k: v for k, v in frozen.items() if k != 'fingerprint'}) == frozen['fingerprint'], 'Freeze seal changed')
    require(sources() == frozen['source_files'], 'Executed source/config bytes changed; use a new named run')
    require(digest(read_json(run / 'private/schedule.json')) == frozen['schedule_hash'], 'Schedule changed')
    require(environment_fingerprint(environment(ROOT)) == frozen['environment_fingerprint'], 'Environment changed',
            'BLOCKED_ENVIRONMENT')
    cohort = read_json(run / 'private/cohort.json')
    require(digest(cohort) == frozen['cohort_hash'], 'Cohort changed')
    if weights:
        for name, sha in frozen['adapter']['files'].items():
            require(file_hash(Path(frozen['adapter']['model_path']) / name) == sha, 'Model bytes changed')
        for v in cohort['videos']:
            require(file_hash(v['path']) == v['sha256'], 'Video bytes changed')
    return frozen


def prepare(run, config):
    require(not (run / 'private/frozen.json').exists(), 'Already frozen; validate/resume without refreezing')
    parent = ROOT / config['parent_run']
    adapter = yaml.safe_load((parent / 'resolved_config.yaml').read_text(encoding='utf-8'))
    cohort = read_json(parent / 'cohort_manifest.json')
    model = read_json(parent / 'model_manifest.json')
    require(digest(cohort) == adapter['cohort_hash'] and digest(model) == adapter['model_manifest_hash'], 'Parent manifests changed')
    require(len(cohort['videos']) == 8 and all(len(v['questions']) == 3 for v in cohort['videos']), 'Cohort incomplete')
    require(adapter['model_revision'] == 'ebb281ec70b05090aa6165b016eac8ec08e71b17', 'Wrong revision')
    env = environment(ROOT)
    require(environment_fingerprint(env) == adapter['environment_fingerprint'], 'Historical software/device changed',
            'BLOCKED_ENVIRONMENT')
    for name, sha in model['files'].items():
        require(file_hash(Path(model['model_path']) / name) == sha, 'Model assets missing or changed', 'BLOCKED_INPUT')
    for video in cohort['videos']:
        require(file_hash(video['path']) == video['sha256'], 'Source video missing or changed', 'BLOCKED_INPUT')
    order = diagnostic_schedule(cohort, config['diagnostic_seed'])
    frozen = {'version': config['version'], 'run_id': run.name, 'at': utc_now(), 'config': config,
              'adapter': adapter, 'cohort_hash': digest(cohort), 'schedule_hash': digest(order),
              'source_files': sources(), 'environment_fingerprint': environment_fingerprint(env),
              'base_git_revision': command_output(['git', 'rev-parse', 'HEAD'])['stdout'],
              'kind': 'diagnostic', 'monitoring': 'MemorySampler 0.05 seconds for every call',
              'memory_rule': 'Post-release live <= shape warmup max + 64 MiB; warmup <= model baseline + 64 MiB'}
    frozen['fingerprint'] = digest(frozen)
    for name, value in (('frozen', frozen), ('schedule', order), ('cohort', cohort), ('environment', env)):
        write_json(run / f'private/{name}.json', value)
    for name in frozen['source_files']:
        p = run / 'private/frozen_sources' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / name).read_bytes())
    write_json(run / 'freeze.json', {k: v for k, v in frozen.items() if k not in ('adapter', 'environment_fingerprint')})
    write_json(run / 'schedule.json', anonymize_schedule(order, cohort))
    cpu_phase(run, run, frozen, [8, 64])


def anonymize_schedule(order, cohort):
    names = {(v['source_id'], str(q['question_id'])): f'V{i + 1:02d}Q{j + 1}'
             for i, v in enumerate(cohort['videos']) for j, q in enumerate(v['questions'])}
    result = copy.deepcopy(order)
    for worker in result:
        for block in worker['blocks']:
            block['pairs'] = [names[tuple(pair)] for pair in block['pairs']]
    return result


def cpu_prepare(run, frozen, budgets):
    import hashlib
    import torch
    from transformers import AutoConfig, AutoProcessor
    from video_compute_value.model import QwenAdapter, adapter_source_hashes, tensor_hashes
    from video_compute_value.video import decode_selected
    torch.set_num_threads(4)
    adapter = QwenAdapter.__new__(QwenAdapter)
    adapter.torch, adapter.frozen = torch, frozen['adapter']
    adapter.model = SimpleNamespace(config=AutoConfig.from_pretrained(adapter.frozen['model_path'], local_files_only=True))
    adapter.processor = AutoProcessor.from_pretrained(adapter.frozen['model_path'], local_files_only=True, use_fast=True)
    require(adapter_source_hashes() == adapter.frozen['processor_source_hashes'], 'Native processor changed')
    reference = history_reference(ROOT / frozen['config']['parent_run'])
    path = run / 'private/cpu_preparations.json'
    rows = read_json(path) if path.exists() else []
    preparation_fingerprint = digest(frozen['adapter'])
    for row in rows:
        require(row.get('preparation_fingerprint') == preparation_fingerprint and
                row.get('evidence_seal') == digest({k: v for k, v in row.items() if k != 'evidence_seal'}),
                'Reused CPU evidence fingerprint or seal changed')
        old = reference[(*row['pair'], row['budget'])]
        require(row['input_tensor_hashes'] == old['input_tensor_hashes'] and
                row['metadata_hash'] == old['input_metadata_hash'], 'Reused CPU inputs changed')
    for budget in budgets:
        for video in read_json(run / 'private/cohort.json')['videos']:
            pair = [video['source_id'], str(video['questions'][0]['question_id'])]
            if any(r['pair'] == pair and r['budget'] == budget for r in rows):
                continue
            began = time.perf_counter()
            inputs, details = adapter.prepare(question_input(video['questions'][0]), video_input(video), budget)
            elapsed = time.perf_counter() - began
            details['input_tensor_hashes'] = tensor_hashes(inputs)
            old = reference[(*pair, budget)]
            require(all(details[k] == old[k] for k in INPUT_FIELDS), 'CPU native inputs/metadata differ from E0')
            # Independent serial versus FRAME/4 exact uint8 equality, outside the preparation cost.
            frames, _ = decode_selected(video_input(video), budget, decoder_threads=4, decoder_thread_type='FRAME')
            serial, _ = decode_selected(video_input(video), budget)
            require((frames == serial).all(), 'Decoder RGB pixels differ')
            evidence = {'pair': pair, 'budget': budget, 'elapsed_seconds': elapsed,
                         'input_tensor_hashes': details['input_tensor_hashes'],
                         'metadata_hash': digest({k: details[k] for k in META}),
                         'rgb_sha256': hashlib.sha256(frames.tobytes()).hexdigest(), 'serial_pixels_match': True,
                         'fingerprint': frozen['fingerprint'], 'preparation_fingerprint': preparation_fingerprint}
            rows.append({**evidence, 'evidence_seal': digest(evidence)})
            del inputs, frames, serial
            write_json(path, rows)
            print(f'CPU native preparation {len(rows)}; budget {budget}; input/pixels verified', flush=True)


def cpu_phase(task_run, run, frozen, budgets):
    old = [read_json(p) for p in (ROOT / frozen['config']['parent_run'] / 'records').glob('*.json')]
    cohort = read_json(run / 'private/cohort.json')
    existing = read_json(run / 'private/cpu_preparations.json') if (run / 'private/cpu_preparations.json').exists() else []
    raw = 30 + 3 * sum(max(r['cpu_prepare_host_seconds'] for r in old if r['source_id'] == v['source_id'] and
                          r['budget'] == b) for v in cohort['videos'] for b in budgets if not any(
                              e['pair'][0] == v['source_id'] and e['budget'] == b for e in existing))
    admit(task_run, frozen['config'], 'cpu_preparation', raw, planned_native_preparations=8 * len(budgets),
          basis='Historical per-source/budget maximum native span, three decodes per new preparation, 30 s startup')
    try:
        launch_owned([sys.executable, '-u', str(Path(__file__)), 'cpu-worker', '--run-dir', str(run),
                      '--budgets', *map(str, budgets)], run / 'private/cpu_worker.log', math.ceil(raw * 1.5),
                     run / 'private/failures.jsonl', 'cpu_preparation', model_worker=False)
    finally:
        checkpoint(task_run, 'All native CPU preparations and serial pixel comparisons')


def worker(run, spec, frozen):
    from video_compute_value.model import QwenAdapter, tensor_hashes
    from video_compute_value.measurement import MemorySampler, timed_request
    allowed = read_json(run / 'private/schedule.json') + probe_schedule(frozen)
    if frozen['kind'] == 'confirmation':
        allowed += probe_schedule(frozen, True, frozen['selected_policy'])
    require(spec in allowed, 'Worker spec differs from frozen lifecycle/schedule')
    reference = history_reference(ROOT / frozen['config']['parent_run'])
    cohort = read_json(run / 'private/cohort.json')
    lookup = {(v['source_id'], str(q['question_id'])): (v, q) for v in cohort['videos'] for q in v['questions']}
    folder = run / 'private/workers' / spec['worker_id']
    require(not folder.exists(), 'Session already attempted; interrupted sessions cannot restart')
    folder.mkdir(parents=True)
    with exclusive_lock(ROOT / '.cache/cuda_worker.lock'):
        write_json(folder / 'environment_start.json', telemetry())
        adapter = QwenAdapter(frozen['adapter'])
        receipt = {'worker_id': spec['worker_id'], 'pid': os.getpid(), 'started_at': utc_now(),
                   'generation_policy': adapter.generation_policy, 'load_seconds': adapter.load_seconds,
                   'baseline': adapter.baseline, 'allocator_backend': adapter.torch.cuda.memory.get_allocator_backend(),
                   'fingerprint': frozen['fingerprint'], 'blocks': [b['budget'] for b in spec['blocks']]}
        write_json(folder / 'receipt.json', receipt)
        warm = {}
        for call in expected_calls(frozen, spec):
            budget = call['budget']
            if budget not in warm:
                write_json(folder / f'environment_block_{budget}.json', telemetry())
                warm[budget] = []
            video, question = lookup[call['source_id'], call['question_id']]
            event = {**call, 'event': 'started', 'at': utc_now()}
            append_json(run / 'private/calls.jsonl', event)
            before = adapter.model_call_count
            try:
                record, inputs = timed_request(adapter, question_input(question), video_input(video) if budget else None,
                                               budget, profile=call['kind'] == 'profile', sampler=MemorySampler(.05))
                record['input_tensor_hashes'] = tensor_hashes(inputs)
                record['input_metadata_hash'] = digest({k: record[k] for k in META})
                del inputs
                gc.collect()  # Same historical post-request cleanup, outside primary latency.
                record['live_allocated_after_bytes'] = adapter.torch.cuda.memory_allocated()
                record['reserved_after_bytes'] = adapter.torch.cuda.memory_reserved()
                if call['kind'] == 'warmup':
                    warm[budget].append(record['live_allocated_after_bytes'])
                    baseline = adapter.baseline['allocated_bytes']
                else:
                    baseline = max(warm[budget])
                record['shape_live_baseline_bytes'] = baseline
                record['live_memory_growth_flag'] = record['live_allocated_after_bytes'] > baseline + 64 * 1024**2
                old = reference[call['source_id'], call['question_id'], budget]
                record.update(call, correct=record['parsed_answer'] == question['answer'], at=utc_now())
                record['historical_input_match'] = all(record[k] == old[k] for k in INPUT_FIELDS)
                record['historical_output_match'] = all(record[k] == old[k] for k in OUTPUT_FIELDS)
                seal_record(run / 'private/records' / (call['call_id'] + '.json'), record)
                append_json(run / 'private/calls.jsonl', {**call, 'event': 'completed', 'at': utc_now(),
                            'model_invoked': adapter.model_call_count > before})
                require(record['historical_input_match'], 'Native input differs from corrected E0')
                require(record['historical_output_match'], 'Greedy output differs from corrected E0', 'REPEATABILITY_UNRESOLVED')
                require(not record['live_memory_growth_flag'], 'Unresolved post-release live allocation growth')
            except BaseException as exc:
                append_json(run / 'private/failures.jsonl', {**call, 'at': utc_now(), 'error': type(exc).__name__,
                            'model_invoked': adapter.model_call_count > before, 'traceback': traceback.format_exc()})
                raise
        receipt.update(finished_at=utc_now(), model_calls=adapter.model_call_count)
        write_json(folder / 'receipt.json', receipt)


def launch_owned(args, log, timeout, failure_path, worker_id, model_worker=True):
    with log.open('w', encoding='utf-8') as output:
        process = subprocess.Popen(args, cwd=ROOT, shell=False, stdout=output, stderr=output)
        try:
            code = process.wait(timeout=timeout)
        except BaseException as exc:
            # Reap only this coordinator's owned process tree before releasing its global lock.
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True, check=False)
            else:
                process.kill()
            process.wait()
            append_json(failure_path, {'worker_id': worker_id, 'kind': 'worker', 'at': utc_now(),
                        'error': type(exc).__name__, 'reaped': process.poll() is not None,
                        'unknown_invocations_upper_bound': int(model_worker)})
            raise Blocked('PARTIAL', 'Owned worker interrupted; session incomplete, no automatic restart') from exc
    if code:
        append_json(failure_path, {'worker_id': worker_id, 'kind': 'worker', 'at': utc_now(),
                    'error': 'nonzero_exit', 'exit_code': code, 'reaped': True})
    require(code == 0, 'Worker failed; partial session preserved', 'PARTIAL')


def run_workers(task_run, run, frozen, specs, timeout):
    for spec in specs:
        deadline = timeout[spec['worker_id']] if isinstance(timeout, dict) else timeout
        validate(run)
        events = read_jsonl(run / 'private/launches.jsonl')
        previous = [e for e in events if e['worker_id'] == spec['worker_id']]
        if previous:
            require(previous[-1]['event'] == 'completed', 'Previously interrupted worker; cannot resume partial process history')
            completed = {r['call_id'] for r in load_records(run, frozen['fingerprint'])}
            require(all(c['call_id'] in completed for c in expected_calls(frozen, spec)), 'Completed worker records missing')
            continue
        ledger = checkpoint(task_run, 'Before worker ' + spec['worker_id'])
        require(LIMIT - ledger.value['active_seconds'] >= deadline + frozen['config']['closeout_reserve_seconds'],
                'Worker deadline would consume protected reserve', 'BLOCKED_RESOURCE')
        spec_path = run / 'private/specs' / (spec['worker_id'] + '.json')
        write_json(spec_path, spec)
        append_json(run / 'private/launches.jsonl', {'worker_id': spec['worker_id'], 'event': 'started', 'at': utc_now()})
        began = time.perf_counter()
        try:
            with exclusive_lock(ROOT / '.cache/e000_v5_coordinator.lock'):
                launch_owned([sys.executable, '-u', str(Path(__file__)), 'worker', '--run-dir', str(run),
                              '--spec', str(spec_path)], run / 'private' / (spec['worker_id'] + '.log'), deadline,
                             run / 'private/failures.jsonl', spec['worker_id'])
            append_json(run / 'private/launches.jsonl', {'worker_id': spec['worker_id'], 'event': 'completed',
                        'at': utc_now(), 'wall_seconds': time.perf_counter() - began})
        finally:
            checkpoint(task_run, 'Entire owned worker/coordinator window ' + spec['worker_id'])
        print('Worker complete:', spec['worker_id'], flush=True)


def historical_bound(frozen, specs):
    parent = ROOT / frozen['config']['parent_run']
    rows = [read_json(p) for p in (parent / 'records').glob('*.json')]
    worst = max(r['latency_seconds'] + r.get('diagnostic_hash_seconds', 0) for r in rows)
    # Successful historical collection overhead was ~30 s; use 60 s per fresh worker.
    calls = sum(len(expected_calls(frozen, s)) for s in specs)
    return calls * worst + 60 * len(specs) + 60, {'historical_worst_request_plus_hash_seconds': worst,
            'conservative_startup_per_worker_seconds': 60, 'planned_workers': len(specs), 'planned_calls': calls,
            'basis': 'All corrected E0 records; conservative bound, not a new measured forecast'}


def measured_forecast(run, frozen, specs):
    rows = load_records(run, frozen['fingerprint'])
    profiles = [r for r in rows if r['kind'] == 'profile']
    cpu = read_json(run / 'private/cpu_preparations.json')
    launches = [r for r in read_jsonl(run / 'private/launches.jsonl') if r['event'] == 'completed']
    overhead = []
    for launch in launches:
        timed = sum(r['latency_seconds'] for r in rows if r['worker_id'] == launch['worker_id'])
        overhead.append(max(0, launch['wall_seconds'] - timed))
    startup_overhead = max(overhead)
    total = startup_overhead * len(specs)
    for spec in specs:
        for c in expected_calls(frozen, spec):
            budget = c['budget']
            if budget:
                prep = next(r['elapsed_seconds'] for r in cpu if r['budget'] == budget and r['pair'][0] == c['source_id'])
                relevant = [r for r in profiles if r['budget'] == budget]
            else:
                prep = .05
                relevant = [r for r in profiles if r['budget'] == 8]
            require(bool(relevant), 'Missing real full-budget profile')
            remainder = max(r['latency_seconds'] - r['cpu_prepare_host_seconds'] for r in relevant)
            # Source-1 cold/native preparation can exceed the CPU-only preparation measurement.
            reference_prep = next(r['elapsed_seconds'] for r in cpu if r['budget'] == (budget or 8) and
                                  r['pair'][0] == frozen['adapter']['profile_question'][0])
            adjustment = max(0, max(r['cpu_prepare_host_seconds'] for r in relevant) - reference_prep)
            total += prep + adjustment + remainder + .25
    return total + 60, {'planned_workers': len(specs), 'planned_calls': sum(len(expected_calls(frozen, s)) for s in specs),
                        'measured_max_worker_nontimed_overhead_seconds': startup_overhead,
                        'all_source_preparations': len(cpu), 'profile_calls': len(profiles),
                        'basis': 'Real profiles in both transition directions + every source CPU preparation; no frame ratios'}


def preflight(task_run, run, frozen, full=False, policy=None):
    specs = probe_schedule(frozen, full, policy)
    name = 'full_preflight' if full else 'preflight'
    raw, detail = historical_bound(frozen, specs)
    admit(task_run, frozen['config'], name, raw, **detail)
    run_workers(task_run, run, frozen, specs, timeout=240 if not full or policy == 'A' else 600)
    from verify_e000_timing_v5 import verify_saved
    result = verify_saved(run, ROOT, complete_phase=name)
    write_json(run / (name + '_verification.json'), result)
    require(result['passed'], 'Independent preflight verification failed')


def relative_range(values):
    require(len(values) == 3 and all(math.isfinite(v) and v > 0 for v in values), 'Three valid repeats required')
    return (max(values) - min(values)) / statistics.median(values)


def summarize(rows, arms=('A', 'B'), budgets=(8, 64), count=8):
    conditions, cells = [], []
    for arm in arms:
        for budget in budgets:
            selected = [r for r in rows if r['kind'] == 'primary' and r['arm'] == arm and r['budget'] == budget]
            groups = {}
            for r in selected:
                groups.setdefault((r['source_id'], r['question_id']), []).append(r)
            complete = len(groups) == count and all(sorted(r['repetition'] for r in g) == [1, 2, 3] for g in groups.values())
            cell_rows = []
            if complete:
                for i, (pair, group) in enumerate(sorted(groups.items()), 1):
                    times = [r['latency_seconds'] for r in group]
                    cell = {'arm': arm, 'budget': budget, 'anonymous_cell': i,
                            'relative_range': relative_range(times), 'absolute_range_seconds': max(times) - min(times),
                            'median_request_seconds': statistics.median(times),
                            'median_prepare_seconds': statistics.median(r['cpu_prepare_host_seconds'] for r in group),
                            'median_remainder_seconds': statistics.median(r['latency_seconds'] - r['cpu_prepare_host_seconds'] for r in group)}
                    cell['over_50_percent'] = cell['relative_range'] > .5
                    cell_rows.append(cell)
                cells.extend(cell_rows)
            condition = {'arm': arm, 'budget': budget, 'complete': complete, 'primary_requests': len(selected),
                         'source_questions': count, 'independent_repetitions': 3,
                         'S': statistics.median(r['relative_range'] for r in cell_rows) if complete else None}
            for key in ('absolute_range_seconds', 'median_request_seconds', 'median_prepare_seconds', 'median_remainder_seconds'):
                condition[key] = statistics.median(r[key] for r in cell_rows) if complete else None
            for key in ('peak_allocated_bytes', 'peak_reserved_bytes', 'sampled_device_gpu_peak_bytes',
                        'live_allocated_after_bytes', 'reserved_after_bytes'):
                vals = [r[key] for r in selected if r.get(key) is not None]
                condition[key] = max(vals) if vals else None
            condition['individual_over_50_percent'] = sum(r['over_50_percent'] for r in cell_rows) if complete else None
            conditions.append(condition)
    return {'conditions': conditions, 'question_ranges': cells}


def select_policy(summary, integrity):
    if integrity:
        for arm in ('A', 'B'):
            group = [c for c in summary['conditions'] if c['arm'] == arm]
            if len(group) == 2 and {c['budget'] for c in group} == {8, 64} and all(
                    c['complete'] and c['S'] is not None and c['S'] <= .20 for c in group):
                return arm
    return None


def analyze(run):
    frozen = read_json(run / 'private/frozen.json')
    rows = load_records(run, frozen['fingerprint'])
    summary = summarize(rows) if frozen['kind'] == 'diagnostic' else summarize(
        rows, (frozen['selected_policy'],), (8, 16, 32, 48, 64), 24)
    from verify_e000_timing_v5 import verify_saved
    result = verify_saved(run, ROOT, complete_phase=frozen['kind'], check_summary=False)
    summary['integrity_passed'] = result['passed']
    summary['selected_policy'] = select_policy(summary, result['passed']) if frozen['kind'] == 'diagnostic' else frozen['selected_policy']
    if frozen['kind'] == 'diagnostic':
        summary['status'] = 'ELIGIBLE_FOR_CONFIRMATION' if summary['selected_policy'] else (
            'STABILIZATION_NOT_DEMONSTRATED' if result['passed'] else 'INCOMPLETE_OR_INTEGRITY_FAILED')
    else:
        summary['status'] = ('FEASIBILITY_CONFIRMED' if all(c['S'] is not None and c['S'] <= .2 for c in summary['conditions'])
                             else 'MEASUREMENT_UNSTABLE') if result['passed'] else 'INCOMPLETE_OR_INTEGRITY_FAILED'
    write_json(run / 'verification.json', result)
    write_json(run / 'summary.json', summary)
    return summary


def collect(task_run, run, frozen):
    specs = read_json(run / 'private/schedule.json')
    name = 'diagnostic' if frozen['kind'] == 'diagnostic' else 'confirmation'
    raw, detail = measured_forecast(run, frozen, specs)
    admit(task_run, frozen['config'], name, raw, **detail)
    deadlines = {spec['worker_id']: max(90, math.ceil(measured_forecast(run, frozen, [spec])[0] * 1.5))
                 for spec in specs}
    path = run / f'private/deadlines_{name}.json'
    if path.exists():
        require(read_json(path) == deadlines, 'Prospective worker deadlines changed')
    else:
        write_json(path, deadlines)
    run_workers(task_run, run, frozen, specs, deadlines)
    return analyze(run)


def confirmation(task_run, config):
    summary = analyze(task_run)
    policy = summary['selected_policy']
    require(policy is not None, 'Neither policy qualified; no confirmation authorized', 'CONFIRMATION_NOT_RUN')
    require(not (task_run / 'confirmation.json').exists(), 'Only one full confirmation attempt allowed')
    diagnostic = validate(task_run)
    run = ROOT / ('results/e000_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_confirmation_v5')
    run.joinpath('private').mkdir(parents=True)
    frozen = copy.deepcopy(diagnostic)
    frozen.update(run_id=run.name, at=utc_now(), kind='confirmation', selected_policy=policy,
                  diagnostic_run=task_run.relative_to(ROOT).as_posix(),
                  diagnostic_fingerprint=diagnostic['fingerprint'])
    cohort = read_json(task_run / 'private/cohort.json')
    order = confirmation_schedule(cohort, policy, config['confirmation_seed'])
    frozen['schedule_hash'] = digest(order)
    frozen.pop('fingerprint')
    frozen['fingerprint'] = digest(frozen)
    for name, value in (('frozen', frozen), ('schedule', order), ('cohort', cohort)):
        write_json(run / f'private/{name}.json', value)
    for name in frozen['source_files']:
        p = run / 'private/frozen_sources' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / name).read_bytes())
    write_json(run / 'freeze.json', {k: v for k, v in frozen.items() if k not in ('adapter', 'environment_fingerprint')})
    write_json(run / 'schedule.json', anonymize_schedule(order, cohort))
    # Same native preparation can be reused by content, with explicit original evidence provenance.
    cpu = read_json(task_run / 'private/cpu_preparations.json')
    write_json(run / 'private/cpu_preparations.json', cpu)
    link = {'run': run.relative_to(ROOT).as_posix(), 'selected_policy': policy,
            'shared_task_ledger': (task_run / 'private/resource_ledger.json').relative_to(ROOT).as_posix(),
            'status': 'CONFIRMATION_NOT_RUN'}
    write_json(task_run / 'confirmation.json', link)
    try:
        raw, detail = historical_bound(frozen, probe_schedule(frozen, True, policy))
        # Check a conservative complete future confirmation before spending on full preflight.
        history_rows = [read_json(p) for p in (ROOT / config['parent_run'] / 'records').glob('*.json')]
        worst = max(r['latency_seconds'] for r in history_rows)
        additional_cpu_validation = 3 * sum(max(r['cpu_prepare_host_seconds'] for r in history_rows
              if r['source_id'] == v['source_id'] and r['budget'] == budget)
              for v in cohort['videos'] for budget in (16, 32, 48)) + 30
        raw += 60 * len(order) + 416 * (worst + .25) + 180 + additional_cpu_validation
        admit(task_run, config, 'confirmation_screen', raw, **detail,
              additional_CPU_preparation_and_pixel_validation_seconds=additional_cpu_validation,
              note='Conservative all-budget collection bound before additional CPU or GPU work')
        write_json(run / 'private/cpu_reuse_provenance.json', {'source_run': task_run.name,
                   'source_file_sha256': file_hash(task_run / 'private/cpu_preparations.json'),
                   'preparation_fingerprint': digest(frozen['adapter'])})
        cpu_phase(task_run, run, frozen, [8, 16, 32, 48, 64])
        preflight(task_run, run, frozen, True, policy)
        result = collect(task_run, run, frozen)
        link['status'] = result['status']
    except Blocked as exc:
        link.update(status='CONFIRMATION_NOT_RUN', reason=exc.status, detail=str(exc))
        write_json(run / 'status.json', link)
    finally:
        write_json(task_run / 'confirmation.json', link)
    return link


def account(run, close=False):
    ledger = checkpoint(run, 'Verification, reporting and publication preparation')
    if close:
        task = read_json(run / 'private/task.json')
        require(not task.get('closed'), 'Already closed; reserve must not be charged twice')
        reserve = 180
        ledger.value['active_seconds'] += reserve
        ledger.value.setdefault('external_events', []).append({'name': 'Conservative final Git publication and handoff reserve',
                                                               'active_seconds': reserve, 'recorded_at': utc_now()})
        ledger.save(force_storage=True)
        task.update(closed=True, publication_reserve_seconds=reserve, publication_started_at=utc_now())
        write_json(run / 'private/task.json', task)
    charged = sum(s['active_seconds'] for s in ledger.value['sessions']) + sum(e['active_seconds'] for e in ledger.value.get('external_events', []))
    require(abs(charged - ledger.value['active_seconds']) < .0001, 'Ledger component charges disagree')
    require(ledger.value['active_seconds'] <= LIMIT, 'Cumulative cap exceeded', 'BLOCKED_RESOURCE')
    events = read_jsonl(run / 'private/calls.jsonl')
    if (run / 'confirmation.json').exists():
        child = ROOT / read_json(run / 'confirmation.json')['run']
        events += read_jsonl(child / 'private/calls.jsonl')
    done = [e for e in events if e['event'] == 'completed']
    starts = {e['call_id'] for e in events if e['event'] == 'started'}
    ended = {e['call_id'] for e in done}
    phases = {}
    for e in done:
        phases.setdefault(e['phase'], {})[e['kind']] = phases.setdefault(e['phase'], {}).get(e['kind'], 0) + 1
    resource = {'at': utc_now(), 'inherited_seconds': read_json(run / 'private/task.json')['inherited_seconds'],
                'cumulative_seconds': ledger.value['active_seconds'], 'remaining_seconds': LIMIT - ledger.value['active_seconds'],
                'authorized_limit_seconds': LIMIT, 'storage_current_bytes': ledger.value['storage_current_bytes'],
                'storage_peak_bytes': ledger.value['storage_peak_bytes'], 'storage_limit_bytes': 20 * 1024**3,
                'video_transfer_bytes': ledger.value['video_transfer_bytes'], 'video_limit_bytes': 1024**3,
                'all_sessions_closed': all(s.get('finished_at') for s in ledger.value['sessions']),
                'model_calls': phases, 'completed_model_calls': len(done), 'unknown_model_invocations_upper_bound': len(starts - ended),
                'cumulative_model_invocations_upper_bound': 1762 + len(done) + len(starts - ended),
                'new_acquisition_bytes': 0, 'publication_reserve_seconds': 180 if close else 0}
    write_json(run / 'resource_accounting.json', resource)
    return resource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('init', 'cpu-worker', 'prepare', 'preflight', 'collect', 'analyze', 'confirmation', 'verify',
                                             'account', 'close', 'worker', 'report', 'all'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--budgets', type=int, nargs='+')
    parser.add_argument('--task-start', help='Timezone-aware start of already incurred task work; required for init')
    args = parser.parse_args()
    configure()
    run = args.run_dir.resolve()
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding='utf-8'))
    require(config['safety_factor'] == 1.5 and config['threshold'] == .20 and
            config['closeout_reserve_seconds'] >= 600 and config['interruption_contingency_seconds'] >= 120 and
            config['cumulative_limit_seconds'] == LIMIT, 'Prospective resource/decision rules changed')
    if args.operation == 'cpu-worker':
        frozen = read_json(run / 'private/frozen.json')
        require(sources() == frozen['source_files'], 'CPU source/config drift')
        cpu_prepare(run, frozen, args.budgets)
        return
    if args.operation == 'init':
        require(args.task_start is not None, 'init requires --task-start')
        initialize(run, args.task_start)
        print(account(run))
        return
    if args.operation == 'worker':
        worker(run, read_json(args.spec), validate(run))
        return
    try:
        if args.operation in ('prepare', 'all'):
            prepare(run, config)
        if args.operation in ('preflight', 'all'):
            preflight(run, run, validate(run, weights=True))
        if args.operation in ('collect', 'all'):
            collect(run, run, validate(run))
        if args.operation in ('confirmation', 'all'):
            confirmation(run, config)
        if args.operation == 'analyze':
            print(analyze(run))
        if args.operation == 'verify':
            from verify_e000_timing_v5 import verify_saved
            result = verify_saved(run, ROOT)
            print(result)  # Read-only: no result or ledger writes.
            return 0 if result['passed'] else 2
        if args.operation == 'report':
            from report_e000_timing_v5 import report
            report(run)
        if args.operation == 'close':
            print(account(run, close=True))
        else:
            print(account(run))
    except BaseException as exc:
        append_json(run / 'private/coordinator_failures.jsonl', {'at': utc_now(), 'operation': args.operation,
                    'error': type(exc).__name__, 'status': getattr(exc, 'status', 'PARTIAL'), 'traceback': traceback.format_exc()})
        write_json(run / 'status.json', {'status': getattr(exc, 'status', 'PARTIAL'), 'operation': args.operation,
                                       'detail': str(exc), 'historical_E0': 'MEASUREMENT_UNSTABLE'})
        if (run / 'private/frozen.json').exists():
            analyze(run)
        print(account(run))
        print(type(exc).__name__, str(exc), flush=True)
        raise SystemExit(2) from exc


if __name__ == '__main__':
    raise SystemExit(main())
