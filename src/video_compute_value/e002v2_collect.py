"""E2 native CPU integrity and single-CUDA-worker factual/counterfactual collection."""
from __future__ import annotations

import copy
import gc
import hashlib
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import yaml

from .common import (Blocked, append_json, command_output, digest, exclusive_lock, file_hash, read_json,
                     read_jsonl, utc_now, write_json)
from .e001_input import inference_video, install_decoder
from .e002_features import SCHEMA, extract
from .e002_input import call_manifest, menu, split_manifest
from .e002_model import ordered_groups
from .e002_runtime import E2Adapter, observation
from .e002v2_records import may_start, records, recover_atomic_completions, save_record
from .protocol import PROMPT, PARSER_VERSION, question_input, video_input
from .resources import environment, environment_fingerprint


def scientific_sources(root):
    paths = list((root/'src/video_compute_value').glob('*.py'))
    paths += [root/'configs/e002.yaml', root/'configs/e002_continuation.yaml',
              root/'protocols/e002_preacquisition_predictability_v1.md', root/'protocols/e002_continuation_v2.md']
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(paths)}


def freeze(root, run):
    from .e001 import configure
    configure(root)
    cohort = read_json(run/'private/cohort_manifest.json')
    historical_annotations = read_json(root/'results/e000_20261028_greedy_collection_v3/cohort_manifest.json')['annotation_file_sha256']
    annotation_path = root/'data/annotations/ead1408f75b618502df9a1d8e0950166bf0a2a0b.parquet'
    if cohort['annotation_sha256'] != historical_annotations or file_hash(annotation_path) != historical_annotations:
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Pinned official annotation bytes changed')
    if len(cohort['videos']) != 40 or any(len(v['questions']) != 3 for v in cohort['videos']):
        raise Blocked('BLOCKED_INPUT', 'Complete 40-source/120-question cohort required')
    if (run/'private/frozen.json').exists():
        return validate(root, run)
    inherited = read_json(root/'results/e001_20261029_132707_v1/private/frozen.json')
    previous_environment = read_json(root/'results/e001_20261029_132707_v1/private/environment.json')
    current_environment = environment(root)
    if environment_fingerprint(previous_environment) != environment_fingerprint(current_environment):
        raise Blocked('BLOCKED_ENVIRONMENT', 'Inherited model software/hardware fingerprint changed')
    for name, sha in inherited['files'].items():
        if file_hash(Path(inherited['model_path'])/name) != sha:
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Frozen model/processor/tokenizer bytes changed')
    legacy = read_json(root/'results/e001_20261029_132707_v1/private/cohort_manifest.json')
    if {v['source_id'] for v in cohort['videos']} & {v['source_id'] for v in legacy['videos']}:
        raise Blocked('BLOCKED_INPUT', 'Previously exposed source entered E2')
    splits = split_manifest(cohort['videos'])
    actions = {v['source_id']: menu(v) for v in cohort['videos']}
    inference = {'videos': [{**video_input(v), 'source_id': v['source_id'],
        'questions': [{k: q[k] for k in ('question_id', 'question', 'options')} for q in v['questions']]}
        for v in cohort['videos']]}
    practical_plan = {'timing_sessions': 3, 'screen_legacy_sources': [v['source_id'] for v in legacy['videos'][:2]],
        'screen_methods': ['direct8', 'direct16', 'V', 'VQ', 'C'],
        'panel_methods': ['direct8', 'direct16', 'quarter1', 'quarter2', 'quarter3', 'quarter4',
                          'uniform_addition', 'V_stop', 'VQ_stop', 'C_stop'],
        'panels': {str(f): ordered_groups([g for g, outer in splits['outer'].items() if outer != f],
                                          f'cost_panel:{f}')[:2] for f in range(5)},
        'evaluation_methods': ['V', 'VQ', 'direct16', 'Bstar'], 'identical_method_deduplication': False,
        'warmups': 'two direct8 repetitions of each session first frozen scheduled cell, outside measured cells',
        'panel_pipeline_executions': 900, 'screen_pipeline_executions': 30,
        'evaluation_pipeline_executions': 1440, 'evaluation_model_call_upper_bound': 2520}
    result = copy.deepcopy(inherited)
    result.update(experiment_id='e002_preacquisition_predictability_v1', run_id=run.name,
        implementation_version='continuation_v2', frozen_at=utc_now(), source_files=scientific_sources(root),
        source_git_revision=command_output(['git', 'rev-parse', 'HEAD'])['stdout'],
        scientific_config=yaml.safe_load((root/'configs/e002.yaml').read_text(encoding='utf-8')),
        cohort_hash=digest(cohort), inference_hash=digest(inference), split_hash=digest(splits), action_hash=digest(actions),
        practical_plan_hash=digest(practical_plan), feature_schema=SCHEMA,
        environment_fingerprint=environment_fingerprint(current_environment), prompt=PROMPT, parser_version=PARSER_VERSION)
    result.pop('fingerprint', None)
    result['fingerprint'] = digest(result)
    manifest = call_manifest(cohort['videos'], splits, result['fingerprint'])
    write_json(run/'private/splits.json', splits)
    write_json(run/'private/actions.json', actions)
    write_json(run/'private/call_manifest.json', manifest)
    write_json(run/'private/practical_plan.json', practical_plan)
    write_json(run/'private/inference_manifest.json', inference)
    write_json(run/'private/environment.json', current_environment)
    write_json(run/'private/frozen.json', result)
    for name in result['source_files']:
        target = run/'private/frozen_sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root/name).read_bytes())
    write_json(run/'freeze.json', {k: result[k] for k in ('experiment_id', 'run_id', 'implementation_version',
        'frozen_at', 'fingerprint', 'source_files', 'cohort_hash', 'split_hash', 'action_hash',
        'revision', 'processor_revision', 'tokenizer_revision', 'dtype', 'attention_backend')})
    write_json(run/'split_summary.json', {'outer_groups': [8]*5, 'inner_training_groups': [32]*5,
        'inner_folds': 4, 'repeat_groups': 5, 'manifest_sha256': digest(splits)})
    return result


def validate(root, run, *, inference_only=False):
    frozen = read_json(run/'private/frozen.json')
    if frozen['fingerprint'] != digest({k: v for k, v in frozen.items() if k != 'fingerprint'}):
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Scientific freeze changed')
    if frozen['source_files'] != scientific_sources(root):
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Implementation drift; a new named run is required')
    checked = [('inference_manifest', 'inference_hash'), ('splits', 'split_hash'),
               ('actions', 'action_hash'), ('practical_plan', 'practical_plan_hash')]
    if not inference_only:
        checked.append(('cohort_manifest', 'cohort_hash'))
    for name, key in checked:
        if digest(read_json(run/f'private/{name}.json')) != frozen[key]:
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Frozen input/split/action manifest changed')
    expected = call_manifest(read_json(run/'private/inference_manifest.json')['videos'],
                             read_json(run/'private/splits.json'), frozen['fingerprint'])
    if read_json(run/'private/call_manifest.json') != expected:
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Exact scientific call identities/order changed')
    return frozen


def finish_hashes(details):
    pixels = details.pop('_e1_pixels_for_hashing')
    base = details.pop('_e1_base_indices')
    hashes = {str(i): hashlib.sha256(frame.tobytes()).hexdigest()
              for i, frame in zip(details['source_indices'], pixels)}
    details['processed_rgb_hashes'] = hashes
    # Direct16 need not contain the base. Retention is tested only for nested acquisitions.
    details['retained_base_rgb_hashes'] = {str(i): hashes[str(i)] for i in base if str(i) in hashes}


def cpu_preflight(root, run, budget):
    frozen = freeze(root, run)
    if (run/'private/cpu_preflight.json').exists():
        saved = read_json(run/'private/cpu_preflight.json')
        if saved['fingerprint'] != frozen['fingerprint']:
            raise Blocked('INPUT_INTEGRITY_FAILED', 'CPU receipt fingerprint differs')
        return saved
    budget.admit('cpu_preflight', 1800, storage=64*1024**2)
    import torch
    from transformers import AutoProcessor
    from .model import tensor_hashes
    torch.set_num_threads(4)
    install_decoder()
    adapter = E2Adapter.__new__(E2Adapter)
    adapter.torch, adapter.frozen = torch, frozen
    adapter.processor = AutoProcessor.from_pretrained(frozen['model_path'], local_files_only=True,
                                                      trust_remote_code=False, use_fast=True)
    config = read_json(Path(frozen['model_path'])/'config.json')
    adapter.model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(
        max_position_embeddings=config['text_config']['max_position_embeddings'])))
    cohort = read_json(run/'private/cohort_manifest.json')
    checks = {}
    for index, video in enumerate(cohort['videos'], 1):
        if file_hash(video['path']) != video['sha256']:
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Acquired video changed before native checks')
        checks[video['source_id']] = {}
        construction_started = time.perf_counter()
        choices = menu(video)
        construction_seconds = time.perf_counter()-construction_started
        for condition, action in choices.items():
            started = time.perf_counter()
            inputs, details = adapter.prepare(question_input(video['questions'][0]),
                inference_video(video, action), len(action['indices']))
            if condition == 'base_observed':
                features = extract(observation(question_input(video['questions'][0]), video, details))
                if [len(features[k]) for k in ('V', 'Q', 'VQ')] != [19, 66, 341]:
                    raise Blocked('BLOCKED_FEATURE_INTEGRITY', 'Frozen feature dimensions changed')
            finish_hashes(details)
            details['input_tensor_hashes'] = tensor_hashes(inputs)
            details['wall_seconds'] = time.perf_counter()-started
            details['action_construction_seconds'] = construction_seconds
            checks[video['source_id']][condition] = details
            del inputs
        row = checks[video['source_id']]
        if row['base_plain']['input_tensor_hashes'] != row['base_observed']['input_tensor_hashes']:
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Plain/observed native base input differs')
        augmented = [row[c] for c in ('quarter1', 'quarter2', 'quarter3', 'quarter4', 'uniform_addition')]
        if (any(a['retained_base_rgb_hashes'] != row['base_plain']['retained_base_rgb_hashes'] for a in augmented)
                or len({a['visual_tokens'] for a in augmented}) != 1):
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Augmented retention or visual-token matching failed')
        print(f'CPU native integrity: {index}/40 sources', flush=True)
        write_json(run/f'private/cpu_checks/{index:02}.json', row)
    result = {'fingerprint': frozen['fingerprint'], 'checks': checks, 'passed': True}
    write_json(run/'private/cpu_preflight.json', result)
    budget.checkpoint('All-source native CPU processor/feature integrity')
    return result


def admit_collection(root, run, budget):
    cpu = cpu_preflight(root, run, budget)
    frozen = validate(root, run)
    if environment_fingerprint(environment(root)) != frozen['environment_fingerprint']:
        raise Blocked('BLOCKED_ENVIRONMENT', 'Frozen software/hardware changed before GPU collection')
    for video in read_json(run/'private/cohort_manifest.json')['videos']:
        if file_hash(video['path']) != video['sha256']:
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Source asset drift before collection/resume')
    for name, sha in frozen['files'].items():
        if file_hash(Path(frozen['model_path'])/name) != sha:
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Model asset drift before collection/resume')
    compatibility = read_json(root/'results/e002_20261029_073134_v1/compatibility.json')
    verification = read_json(root/'results/e002_20261029_073134_v1/compatibility_verification.json')
    if not (compatibility['status'] == 'PASSED' and verification['plain_observed_token_agreement']):
        raise Blocked('BLOCKED_FEATURE_INTEGRITY', 'Same-call score capture compatibility unavailable')
    probes = [read_json(p) for p in (root/'results/e002_20261029_073134_v1/private/compatibility_records').glob('*.json')]
    residual = max(max(0., p['latency_seconds']-p['cpu_prepare_host_seconds']) for p in probes)
    max_prepare = max(r['wall_seconds']+r['action_construction_seconds']
                      for rows in cpu['checks'].values() for r in rows.values())
    startup = read_json(root/'results/e002_20261029_073134_v1/private/compatibility_worker.json')['load_seconds']
    done = records(run, read_json(run/'private/call_manifest.json'))
    raw = (1080-len(done)+6)*(max_prepare+residual+.1)+3*startup
    forecast = budget.admit('E2A_collection', raw, storage=128*1024**2,
        detail={'planned_scientific_calls': 1080, 'completed_scientific_calls': len(done),
                'primary_calls': 960, 'repeatability_calls': 120, 'base_warmups': 4,
                'recovery_warmup_allowance': 2, 'measured_legacy_max_model_residual_seconds': residual,
                'measured_all_source_max_native_prepare_seconds': max_prepare,
                'measured_legacy_startup_seconds': startup, 'three_startups_including_recovery': True,
                'legacy_probe_calls_excluded_from_scientific_evidence': 5})
    write_json(run/'private/collection_ready.json', {'fingerprint': frozen['fingerprint'], 'forecast': forecast})
    return forecast


def worker(root, run, repetition, key):
    from .e001 import configure
    configure(root)
    from .measurement import timed_request
    from .model import tensor_hashes
    frozen = validate(root, run, inference_only=True)
    cohort = read_json(run/'private/inference_manifest.json')
    manifest = read_json(run/'private/call_manifest.json')
    lookup = {(v['source_id'], str(q['question_id'])): (v, q) for v in cohort['videos'] for q in v['questions']}
    receipt_path = run/f'private/workers/{key}.json'
    receipt = {'worker_key': key, 'fingerprint': frozen['fingerprint'], 'stage': 'model_loading',
               'at': utc_now(), 'pid': __import__('os').getpid(), 'model_invocations': 0}
    with exclusive_lock(root/'.cache/cuda_worker.lock'):
        write_json(receipt_path, receipt)
        install_decoder()
        adapter = E2Adapter(frozen)
        receipt.update(stage='generation', load_seconds=adapter.load_seconds,
                       generation_policy=adapter.generation_policy,
                       allocator_backend=adapter.torch.cuda.memory.get_allocator_backend())
        write_json(receipt_path, receipt)
        done = records(run, manifest)

        def execute(identity, kind):
            video, labelled = lookup[(identity['source_id'], identity['question_id'])]
            question = question_input(labelled)
            condition = identity['condition']
            action = menu(video)[condition]
            call_id = f'{key}_{time.time_ns()}'
            event = {'identity': identity['identity'], 'call_id': call_id, 'worker_key': key,
                     'kind': kind, 'repetition': repetition, 'condition': condition}
            append_json(run/'private/calls.jsonl', {**event, 'event': 'started', 'at': utc_now()})
            before = adapter.model_call_count
            try:
                adapter.capture_scores = condition == 'base_observed'
                row, tensors = timed_request(adapter, question, inference_video(video, action), len(action['indices']))
                if condition == 'base_observed':
                    started = time.perf_counter()
                    row['features'] = {k: v.tolist() for k, v in extract(observation(question, video, row)).items()}
                    row['feature_extraction_host_seconds'] = time.perf_counter()-started
                row['input_tensor_hashes'] = tensor_hashes(tensors)
                del tensors
                finish_hashes(row)
                row.update(call_id=call_id, worker_key=key, kind=kind,
                           model_invocation_ordinal=adapter.model_call_count,
                           generation_policy_sha256=digest(adapter.generation_policy))
                if adapter.model_call_count-before != 1:
                    raise Blocked('INPUT_INTEGRITY_FAILED', 'Request did not invoke exactly one model call')
                if kind == 'scientific':
                    row = save_record(run/'private/records', identity, row)
                    done[identity['identity']] = row
                else:
                    row = {**row, **identity}
                    row['record_hash'] = digest(row)
                    write_json(run/f'private/auxiliary/{call_id}.json', row)
                append_json(run/'private/calls.jsonl', {**event, 'event': 'completed', 'at': utc_now(),
                    'record_hash': row['record_hash'], 'model_invocations': 1})
                receipt['model_invocations'] = adapter.model_call_count
                write_json(receipt_path, receipt)
                gc.collect()
                return row
            except BaseException as exc:
                append_json(run/'private/calls.jsonl', {**event, 'event': 'failed', 'at': utc_now(),
                    'model_invoked': adapter.model_call_count > before, 'error_type': type(exc).__name__,
                    'status': getattr(exc, 'status', 'PARTIAL'), 'message': str(exc), 'traceback': traceback.format_exc()})
                raise

        first = cohort['videos'][0]
        for question in first['questions'][:2]:
            ident = {'source_id': first['source_id'], 'question_id': str(question['question_id']),
                     'condition': 'base_plain', 'fingerprint': frozen['fingerprint'], 'repetition': repetition}
            ident['identity'] = digest({'warmup': key, **ident})
            execute(ident, 'warmup')
        planned = [r for r in manifest if r['repetition'] == repetition]
        for index, identity in enumerate(planned, 1):
            if may_start(run, identity, done):
                execute(identity, 'scientific')
            if index % 24 == 0:
                print(f'Repetition {repetition}: {index}/{len(planned)} immutable identities visited', flush=True)
        receipt.update(stage='complete', completed_at=utc_now())
        write_json(receipt_path, receipt)


def collect(root, run, budget):
    frozen = validate(root, run)
    ready = read_json(run/'private/collection_ready.json')
    if ready['fingerprint'] != frozen['fingerprint']:
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Collection admission fingerprint drift')
    manifest = read_json(run/'private/call_manifest.json')
    previous_launches = list((run/'private/launches').glob('*.json'))
    if any(read_json(p).get('native_load_failure') for p in previous_launches):
        raise Blocked('BLOCKED_MODEL_RUNTIME', 'Preserved native loader failure forbids automatic retry')
    for repetition in (1, 2):
        for recovery in range(2):
            done = records(run, manifest)
            recover_atomic_completions(run, done)
            if all(r['identity'] in done for r in manifest if r['repetition'] == repetition):
                break
            launches = list((run/'private/launches').glob('*.json'))
            if len(launches) >= 3:
                raise Blocked('PARTIAL', 'One recovery worker per E2A stage exhausted')
            budget.check(projected_seconds=2400)
            key = f'collection_r{repetition}_{time.time_ns()}'
            launch_path = run/f'private/launches/{key}.json'
            launch = {'at': utc_now(), 'worker_key': key, 'repetition': repetition, 'fingerprint': frozen['fingerprint']}
            write_json(launch_path, launch)
            log = run/f'private/{key}.log'
            started = time.perf_counter()
            timeout = min(14400, int(172800-budget.consumed_seconds()-2400))
            with log.open('w', encoding='utf-8') as output:
                try:
                    result = subprocess.run([sys.executable, '-m', 'video_compute_value.e002_continuation',
                        '_worker', '--run-dir', str(run), '--repetition', str(repetition), '--worker-key', key],
                        cwd=root, stdout=output, stderr=output, timeout=timeout)
                    code = result.returncode
                except subprocess.TimeoutExpired:
                    code = None
            receipt = read_json(run/f'private/workers/{key}.json') if (run/f'private/workers/{key}.json').exists() else {}
            launch.update(returncode=code, windows_exit_hex=f'0x{code & 0xffffffff:08X}' if code is not None else None,
                          worker_reaped=True, wall_seconds=time.perf_counter()-started,
                          native_load_failure=receipt.get('stage') == 'model_loading',
                          completed=code == 0 and receipt.get('stage') == 'complete')
            write_json(launch_path, launch)
            budget.checkpoint(f'E2A collection repetition {repetition} worker and cleanup')
            if launch['completed']:
                break
            if launch['native_load_failure']:
                raise Blocked('BLOCKED_MODEL_RUNTIME', 'Native loader failed; preserved worker is not retried')
            failures = [e for e in read_jsonl(run/'private/calls.jsonl') if e['event'] == 'failed']
            if any(e.get('status') in ('INPUT_INTEGRITY_FAILED', 'BLOCKED_FEATURE_INTEGRITY') for e in failures):
                raise Blocked('INPUT_INTEGRITY_FAILED', 'Integrity failure stops scientific collection')
            if recovery:
                raise Blocked('PARTIAL', 'Recovery worker failed')
