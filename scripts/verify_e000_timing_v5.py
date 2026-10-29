"""Read-only saved-evidence verifier; independently computes metrics and decisions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import math
from pathlib import Path
import statistics

from video_compute_value.common import digest, file_hash, read_json, read_jsonl
from video_compute_value.protocol import parse_answer


def verify_saved(run, root, complete_phase=None, check_summary=True):
    errors = []
    frozen = read_json(run / 'private/frozen.json')
    fingerprint = frozen['fingerprint']
    if digest({k: v for k, v in frozen.items() if k != 'fingerprint'}) != fingerprint:
        errors.append('freeze seal')
    for name, sha in frozen['source_files'].items():
        if file_hash(root / name) != sha:
            errors.append('executed source drift')
        snapshot = run / 'private/frozen_sources' / name
        if not snapshot.exists() or file_hash(snapshot) != sha:
            errors.append('source snapshot drift')
    schedule = read_json(run / 'private/schedule.json')
    if digest(schedule) != frozen['schedule_hash']:
        errors.append('schedule hash')
    cohort = read_json(run / 'private/cohort.json')
    if digest(cohort) != frozen['cohort_hash']:
        errors.append('cohort hash')
    lookup = {(v['source_id'], str(q['question_id'])): (v, q) for v in cohort['videos'] for q in v['questions']}
    reference = {(r['source_id'], str(r['question_id']), r['budget']): r for r in
                 (read_json(p) for p in (root / frozen['config']['parent_run'] / 'records').glob('*.json'))}
    from tokenizers import Tokenizer
    from transformers import GenerationConfig
    from video_compute_value.generation import resolve_greedy_policy
    tokenizer = Tokenizer.from_file(str(Path(frozen['adapter']['model_path']) / 'tokenizer.json'))
    effective = resolve_greedy_policy(GenerationConfig(**frozen['adapter']['generation']),
                                     GenerationConfig.from_pretrained(frozen['adapter']['model_path'], local_files_only=True))
    if effective != frozen['adapter']['generation_policy']:
        errors.append('effective greedy policy')
    rows, identities = [], set()
    metadata = ('tensor_shapes', 'source_indices', 'actual_timestamps', 'model_timestamps', 'video_grid_thw')
    exact_fields = (*metadata, 'input_tokens', 'visual_tokens', 'frames_decoded', 'frames_selected',
                    'frames_processor', 'frames_represented', 'temporal_padding', 'input_tensor_hashes',
                    'generated_token_ids', 'decoded_output', 'parsed_answer', 'valid_answer', 'output_tokens',
                    'generation_policy_sha256')
    for path in sorted((run / 'private/records').glob('*.json')):
        row = read_json(path)
        seal = row.pop('seal', None)
        if digest(row) != seal or row['fingerprint'] != fingerprint or path.stem != row['call_id']:
            errors.append('record seal/fingerprint')
        if row['call_id'] in identities:
            errors.append('duplicate identity')
        identities.add(row['call_id'])
        rows.append(row)
        id_fields = ('fingerprint', 'worker_id', 'phase', 'arm', 'repetition', 'budget', 'source_id', 'question_id', 'kind', 'ordinal')
        if digest({k: row[k] for k in id_fields}) != row['call_id']:
            errors.append('identity reconstruction')
        key = row['source_id'], row['question_id'], row['budget']
        if key not in reference or any(row[k] != reference[key][k] for k in exact_fields):
            errors.append('historical exact input/output mismatch')
        if digest({k: row[k] for k in metadata}) != row['input_metadata_hash']:
            errors.append('metadata seal')
        decoded = tokenizer.decode(row['generated_token_ids'], skip_special_tokens=True)
        if decoded != row['decoded_output'] or parse_answer(decoded) != row['parsed_answer']:
            errors.append('native token decoding/parser')
        if row['output_tokens'] != len(row['generated_token_ids']) or row['output_tokens'] > 8:
            errors.append('generation length')
        if row['generation_policy_sha256'] != digest(effective):
            errors.append('request greedy digest')
        if row['correct'] != (row['parsed_answer'] == lookup[key[:2]][1]['answer']):
            errors.append('scoring')
        if row.get('cuda_cache_trimmed') != (row['budget'] == 64):
            errors.append('trim policy')
        if row['budget'] and (row['decoder_threads'], row['decoder_thread_type']) != (4, 'FRAME'):
            errors.append('decoder policy')
        if not math.isfinite(row['latency_seconds']) or not 0 <= row['cpu_prepare_host_seconds'] < row['latency_seconds']:
            errors.append('request timing')
        for metric in ('allocated', 'reserved'):
            if row[f'peak_{metric}_bytes'] < row[f'baseline_{metric}_bytes'] or row[f'incremental_peak_{metric}_bytes'] != (
                    row[f'peak_{metric}_bytes'] - row[f'baseline_{metric}_bytes']):
                errors.append('memory counters')
        if row['sampling_interval_seconds'] != .05:
            errors.append('monitoring interval')
        folder = run / 'private/workers' / row['worker_id']
        receipt = read_json(folder / 'receipt.json')
        if receipt['generation_policy'] != effective or receipt['allocator_backend'] != 'native' or receipt['fingerprint'] != fingerprint:
            errors.append('worker receipt policy')
        if row['live_allocated_after_bytes'] > row['shape_live_baseline_bytes'] + 64 * 1024**2 or row['live_memory_growth_flag']:
            errors.append('live allocation growth')
    # Shape baseline is independently reconstructed after all rows are read.
    for row in rows:
        receipt = read_json(run / 'private/workers' / row['worker_id'] / 'receipt.json')
        warm = [r['live_allocated_after_bytes'] for r in rows if r['worker_id'] == row['worker_id'] and
                r['budget'] == row['budget'] and r['kind'] == 'warmup']
        baseline = receipt['baseline']['allocated_bytes'] if row['kind'] == 'warmup' else max(warm, default=-1)
        if row['shape_live_baseline_bytes'] != baseline:
            errors.append('shape-aware live baseline')
    for worker_id in {r['worker_id'] for r in rows}:
        receipt = read_json(run / 'private/workers' / worker_id / 'receipt.json')
        if not receipt.get('finished_at') or receipt.get('model_calls') != sum(r['worker_id'] == worker_id for r in rows):
            errors.append('worker lifetime/invocation count')
    events = read_jsonl(run / 'private/calls.jsonl')
    starts = [e['call_id'] for e in events if e['event'] == 'started']
    ends = [e['call_id'] for e in events if e['event'] == 'completed']
    if len(starts) != len(set(starts)) or len(ends) != len(set(ends)):
        errors.append('duplicate events')
    if set(ends) != identities or set(starts) != set(ends):
        errors.append('unaccounted or incomplete invocation')
    if any(not e.get('model_invoked') for e in events if e['event'] == 'completed'):
        errors.append('missing actual model invocation')
    if read_jsonl(run / 'private/failures.jsonl'):
        errors.append('failed/interrupted attempt')
    phase = complete_phase or frozen['kind']
    specs = schedule
    if 'preflight' in phase:
        full = phase == 'full_preflight'
        orders = ([[b] for b in (8, 16, 32, 48, 64)] if frozen.get('selected_policy') == 'A' else
                  [[8, 16, 32, 48, 64], [64, 48, 32, 16, 8]]) if full else [[8, 64], [64, 8]]
        specs = [{'worker_id': f'{"full" if full else "pilot"}_probe_{i}', 'phase': phase,
                  'blocks': [{'budget': b, 'pairs': [frozen['adapter']['profile_question']] * 2 if full else [],
                              'profile': frozen['adapter']['profile_question']} for b in order]}
                 for i, order in enumerate(orders)]
    expected = []
    for spec in specs:
        for block in spec['blocks']:
            for ordinal, pair in enumerate(frozen['adapter']['two_warmups']):
                expected.append((spec['worker_id'], block['budget'], pair[0], str(pair[1]), 'warmup', ordinal))
            for ordinal, pair in enumerate(block['pairs']):
                expected.append((spec['worker_id'], block['budget'], pair[0], str(pair[1]),
                                 'integrity' if 'preflight' in phase else 'primary', ordinal))
            if block.get('profile'):
                pair = block['profile']
                expected.append((spec['worker_id'], block['budget'], pair[0], str(pair[1]), 'profile', 0))
        receipt_path = run / 'private/workers' / spec['worker_id'] / 'receipt.json'
        if not receipt_path.exists() or not read_json(receipt_path).get('finished_at'):
            errors.append('incomplete worker session')
        elif read_json(receipt_path)['blocks'] != [b['budget'] for b in spec['blocks']]:
            errors.append('worker transitions differ from schedule')
    actual = [(r['worker_id'], r['budget'], r['source_id'], r['question_id'], r['kind'], r['ordinal']) for r in rows if r['phase'] == phase]
    if Counter(expected) != Counter(actual):
        errors.append('phase schedule/call completeness')
    if frozen['kind'] == 'diagnostic' and phase == 'diagnostic':
        b = [s for s in schedule if s['arm'] == 'B']
        if len(schedule) != 9 or len(b) != 3 or len({s['repetition'] for s in b}) != 3:
            errors.append('independent B sessions')
        if {tuple(x['budget'] for x in s['blocks']) for s in b} != {(8, 64), (64, 8)}:
            errors.append('both B transitions absent')
        if len([r for r in rows if r['phase'] == 'preflight']) != 12:
            errors.append('preflight incomplete')
        if len(rows) != 132:
            errors.append('diagnostic invocation total')
    if frozen['kind'] == 'confirmation' and phase == 'confirmation':
        required = 25 if frozen['selected_policy'] == 'A' else 50
        if len([r for r in rows if r['phase'] == 'full_preflight']) != required:
            errors.append('full-budget preflight incomplete')
        if len(rows) != 416 + required:
            errors.append('confirmation invocation total')
    launches = read_jsonl(run / 'private/launches.jsonl')
    active = None
    previous_end = None
    for launch in launches:
        if launch['event'] == 'started':
            if active is not None or (previous_end and datetime.fromisoformat(launch['at']) < previous_end):
                errors.append('overlapping workers')
            active = launch['worker_id']
        elif launch['event'] == 'completed':
            if active != launch['worker_id']:
                errors.append('unmatched worker termination')
            active = None
            previous_end = datetime.fromisoformat(launch['at'])
    if active is not None:
        errors.append('unreaped/incomplete worker')
    wanted_workers = [s['worker_id'] for s in specs]
    actual_workers = [e['worker_id'] for e in launches if e['event'] == 'started' and e['worker_id'] in wanted_workers]
    if wanted_workers != actual_workers:
        errors.append('actual arm/worker order')
    for spec in specs:
        wanted_order = [x for x in expected if x[0] == spec['worker_id']]
        actual_order = [(e['worker_id'], e['budget'], e['source_id'], e['question_id'], e['kind'], e['ordinal'])
                        for e in events if e['event'] == 'started' and e['worker_id'] == spec['worker_id']]
        if wanted_order != actual_order:
            errors.append('actual contiguous block/call order')
    metrics = []
    if phase in ('diagnostic', 'confirmation'):
        cells = defaultdict(list)
        for r in rows:
            if r['phase'] == phase and r['kind'] == 'primary' and r['budget']:
                cells[r['arm'], r['budget'], r['source_id'], r['question_id']].append(r)
        groups = defaultdict(list)
        for (arm, budget, source, question), group in cells.items():
            if sorted(r['repetition'] for r in group) != [1, 2, 3]:
                errors.append('incomplete timing cell')
                continue
            times = sorted(r['latency_seconds'] for r in group)
            groups[arm, budget].append((times[-1] - times[0]) / times[1])
        for (arm, budget), values in sorted(groups.items()):
            if len(values) != (8 if phase == 'diagnostic' else 24):
                errors.append('timing denominator')
            metrics.append({'arm': arm, 'budget': budget, 'S': statistics.median(values)})
    inventory_path = run / 'private/historical_inventory.json'
    if not inventory_path.exists() and frozen.get('diagnostic_run'):
        inventory_path = root / frozen['diagnostic_run'] / 'private/historical_inventory.json'
    unchanged = 0
    if inventory_path.exists():
        for name, sha in read_json(inventory_path).items():
            if file_hash(root / name) != sha:
                errors.append('historical evidence changed')
            else:
                unchanged += 1
    else:
        errors.append('historical inventory missing')
    task_run = root / frozen['diagnostic_run'] if frozen.get('diagnostic_run') else run
    ledger = read_json(task_run / 'private/resource_ledger.json')
    task = read_json(task_run / 'private/task.json')
    parent = root / task['parent_path']
    total = sum(s['active_seconds'] for s in ledger['sessions']) + sum(e['active_seconds'] for e in ledger.get('external_events', []))
    if abs(total - ledger['active_seconds']) > .0001 or not task['inherited_seconds'] <= total <= 27600:
        errors.append('cumulative resource reconciliation')
    if file_hash(parent) != task['parent_sha256'] or ledger['sessions'][:len(read_json(parent)['sessions'])] != read_json(parent)['sessions']:
        errors.append('parent accounting lineage')
    if ledger['storage_current_bytes'] > 20 * 1024**3 or ledger['video_transfer_bytes'] + ledger.get('interrupted_transfer_upper_bound_bytes', 0) > 1024**3:
        errors.append('storage/acquisition cap')
    if not all(s.get('finished_at') for s in ledger['sessions']):
        errors.append('open accounting session')
    cpu_path = run / 'private/cpu_preparations.json'
    if cpu_path.exists():
        cpu = read_json(cpu_path)
        expected_cpu = 16 if frozen['kind'] == 'diagnostic' else 40
        if len(cpu) != expected_cpu:
            errors.append('all-source CPU preparation completeness')
        for r in cpu:
            if r.get('evidence_seal') != digest({k: v for k, v in r.items() if k != 'evidence_seal'}) or r.get('preparation_fingerprint') != digest(frozen['adapter']):
                errors.append('CPU evidence seal/fingerprint')
            old = reference[(*r['pair'], r['budget'])]
            if r['input_tensor_hashes'] != old['input_tensor_hashes'] or r['metadata_hash'] != old['input_metadata_hash'] or not r['serial_pixels_match']:
                errors.append('CPU historical input/pixel evidence')
    else:
        errors.append('native CPU preparations absent')
    selected = None
    if not errors and phase == 'diagnostic':
        for arm in ('A', 'B'):
            group = [m for m in metrics if m['arm'] == arm]
            if len(group) == 2 and all(m['S'] <= .2 for m in group):
                selected = arm
                break
    if check_summary and (run / 'summary.json').exists() and phase in ('diagnostic', 'confirmation'):
        saved = read_json(run / 'summary.json')
        saved_metrics = [{'arm': c['arm'], 'budget': c['budget'], 'S': c['S']} for c in saved['conditions'] if c['S'] is not None]
        if saved_metrics != metrics or saved.get('integrity_passed') != (not errors):
            errors.append('saved metrics/integrity differ from independent recomputation')
        if phase == 'diagnostic' and saved.get('selected_policy') != selected:
            errors.append('saved policy selection differs')
        if phase == 'diagnostic' and not errors:
            verdict = 'ELIGIBLE_FOR_CONFIRMATION' if selected else 'STABILIZATION_NOT_DEMONSTRATED'
            if saved.get('status') != verdict:
                errors.append('saved diagnostic verdict differs')
        if phase == 'confirmation' and not errors:
            verdict = 'FEASIBILITY_CONFIRMED' if len(metrics) == 5 and all(m['S'] <= .2 for m in metrics) else 'MEASUREMENT_UNSTABLE'
            if saved.get('status') != verdict:
                errors.append('saved confirmation verdict differs')
    return {'passed': not errors, 'errors': sorted(set(errors)), 'phase': phase, 'records_checked': len(rows),
            'native_token_sequences_redecoded': len(rows), 'completed_model_calls': len(ends),
            'unknown_model_invocations_upper_bound': len(set(starts) - set(ends)),
            'historical_files_unchanged': unchanged, 'independent_metrics': metrics, 'selected_policy': selected,
            'real_GPU_path_exercised': bool(rows), 'CPU_tests_are_not_GPU_evidence': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    result = verify_saved(args.run_dir.resolve(), Path(__file__).resolve().parents[1])
    print(result)
    raise SystemExit(0 if result['passed'] else 2)
