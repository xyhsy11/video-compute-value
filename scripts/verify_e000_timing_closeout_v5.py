"""Read-only audit of the preserved native-load failure and available CPU evidence."""
from __future__ import annotations

import argparse
from pathlib import Path

from video_compute_value.common import digest, exclusive_lock, file_hash, read_json, read_jsonl


def verify_available(run, root):
    frozen = read_json(run / 'private/frozen.json')
    errors = []
    if digest({k: v for k, v in frozen.items() if k != 'fingerprint'}) != frozen['fingerprint']:
        errors.append('freeze seal')
    for name, sha in frozen['source_files'].items():
        if file_hash(root / name) != sha or file_hash(run / 'private/frozen_sources' / name) != sha:
            errors.append('executed/snapshot bytes')
    if digest(read_json(run / 'private/cohort.json')) != frozen['cohort_hash'] or digest(
            read_json(run / 'private/schedule.json')) != frozen['schedule_hash']:
        errors.append('cohort/schedule seal')
    cpu = read_json(run / 'private/cpu_preparations.json')
    reference = {(r['source_id'], str(r['question_id']), r['budget']): r for r in
                 (read_json(p) for p in (root / frozen['config']['parent_run'] / 'records').glob('*.json'))}
    cohort = read_json(run / 'private/cohort.json')
    expected = {(v['source_id'], str(v['questions'][0]['question_id']), b) for v in cohort['videos'] for b in (8, 64)}
    if {(*r['pair'], r['budget']) for r in cpu} != expected or len(cpu) != 16:
        errors.append('CPU identity completeness')
    for r in cpu:
        if r['evidence_seal'] != digest({k: v for k, v in r.items() if k != 'evidence_seal'}):
            errors.append('CPU seal')
        old = reference[(*r['pair'], r['budget'])]
        if r['input_tensor_hashes'] != old['input_tensor_hashes'] or r['metadata_hash'] != old['input_metadata_hash']:
            errors.append('CPU historical tensor/temporal correspondence')
        if r['preparation_fingerprint'] != digest(frozen['adapter']) or not r['serial_pixels_match']:
            errors.append('CPU adapter/pixel evidence')
    events = read_jsonl(run / 'private/calls.jsonl')
    records = list((run / 'private/records').glob('*.json'))
    launches = read_jsonl(run / 'private/launches.jsonl')
    failures = read_jsonl(run / 'private/failures.jsonl')
    expected_launch = len(launches) == 1 and launches[0]['worker_id'] == 'pilot_probe_0' and launches[0]['event'] == 'started'
    if events or records or not expected_launch:
        errors.append('unexpected call records/launches for pre-generation crash')
    if len(failures) != 1 or failures[0].get('exit_code') != 3221225477 or not failures[0].get('reaped'):
        errors.append('native failure/owned process cleanup receipt')
    if (run / 'private/workers/pilot_probe_0/receipt.json').exists():
        errors.append('unexpected successful model-load receipt')
    # A free OS lock confirms that this failed phase left no owner holding the CUDA worker contract.
    with exclusive_lock(root / '.cache/cuda_worker.lock'):
        pass
    inventory = read_json(run / 'private/historical_inventory.json')
    if any(file_hash(root / name) != sha for name, sha in inventory.items()):
        errors.append('historical preservation')
    task = read_json(run / 'private/task.json')
    parent = read_json(root / task['parent_path'])
    ledger = read_json(run / 'private/resource_ledger.json')
    if file_hash(root / task['parent_path']) != task['parent_sha256']:
        errors.append('parent ledger changed')
    for key in ('sessions', 'external_events'):
        if ledger.get(key, [])[:len(parent.get(key, []))] != parent.get(key, []):
            errors.append('inherited charge lineage')
    charged = sum(s['active_seconds'] for s in ledger['sessions']) + sum(e['active_seconds'] for e in ledger.get('external_events', []))
    if abs(charged - ledger['active_seconds']) > .0001 or not parent['active_seconds'] <= charged <= 27600:
        errors.append('cumulative charge reconciliation')
    if not all(s.get('finished_at') for s in ledger['sessions']):
        errors.append('open resource session')
    if ledger['storage_current_bytes'] > 20 * 1024**3 or ledger['video_transfer_bytes'] + ledger.get('interrupted_transfer_upper_bound_bytes', 0) > 1024**3:
        errors.append('storage/acquisition limit')
    summary = read_json(run / 'summary.json')
    if len(summary['conditions']) != 4 or {(c['arm'], c['budget']) for c in summary['conditions']} != {
            ('A', 8), ('A', 64), ('B', 8), ('B', 64)} or summary.get('integrity_passed') is not False or summary.get('status') != 'INCOMPLETE_OR_INTEGRITY_FAILED':
        errors.append('incomplete diagnostic summary shape/status')
    if any(c['S'] is not None or c['primary_requests'] != 0 for c in summary['conditions']) or summary['selected_policy'] is not None:
        errors.append('fabricated timing/selection')
    return {'passed': not errors, 'errors': errors, 'scope': 'Available CPU evidence, pre-generation native failure and safe closeout',
            'experimental_collection_passed': False, 'diagnostic_primary_requests': 0, 'planned_diagnostic_primary_requests': 96,
            'native_CPU_preparations_verified': len(cpu), 'historical_input_and_temporal_hash_matches': len(cpu),
            'serial_FRAME4_pixel_equality_receipts': len(cpu), 'started_model_calls': 0, 'completed_model_calls': 0,
            'unknown_model_invocations_upper_bound': 0, 'attempted_GPU_workers': 1, 'failed_GPU_workers': 1,
            'native_exit_code': '0xC0000005', 'owned_worker_reaped': True, 'CUDA_worker_lock_free': True,
            'historical_files_unchanged': len(inventory), 'timed_GPU_generation_path_verified': False,
            'no_new_native_token_sequences_to_decode': True, 'no_policy_qualified': True,
            'cumulative_resource_components_reconciled': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    result = verify_available(args.run_dir.resolve(), Path(__file__).resolve().parents[1])
    print(result)
    raise SystemExit(0 if result['passed'] else 2)
