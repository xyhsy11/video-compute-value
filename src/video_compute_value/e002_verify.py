"""Read-only historical/saved-record checks; writes ONLY the new E2 receipt."""
from __future__ import annotations

import math

from .common import file_hash, read_json, read_jsonl, utc_now, write_json
from .protocol import parse_answer


def verify_checkpoint(root, run):
    inventory = read_json(run/'private/historical_inventory.json')
    changed = [name for name, sha in inventory.items() if not (root/name).is_file() or file_hash(root/name) != sha]
    assert not changed, f'Historical evidence changed: {changed}'
    task = read_json(run/'private/task.json')
    parent = read_json(root/task['parent_path'])
    ledger = read_json(run/'private/resource_ledger.json')
    assert file_hash(root/task['parent_path']) == task['parent_sha256']
    for key in ('sessions', 'external_events'):
        assert ledger.get(key, [])[:len(parent.get(key, []))] == parent.get(key, [])
    charged = sum(s['active_seconds'] for s in ledger['sessions']) + sum(e['active_seconds'] for e in ledger.get('external_events', []))
    assert math.isclose(charged, ledger['active_seconds'], abs_tol=1e-5)
    assert task['inherited_seconds'] <= ledger['active_seconds'] <= 27600
    accounting = read_json(run/'resource_accounting.json')
    assert accounting['cumulative_seconds'] == ledger['active_seconds']
    assert accounting['storage_current_bytes'] <= 20*1024**3
    assert accounting['video_transfer_and_interruption_upper_bound_bytes'] <= 1024**3
    legacy = root/'results/e001_20261029_132707_v1'
    cohort = read_json(legacy/'private/cohort_manifest.json')
    labels = {(v['source_id'], str(q['question_id'])): q['answer'] for v in cohort['videos'] for q in v['questions']}
    rows = [read_json(p) for p in (legacy/'private/records').glob('*.json')]
    assert len(rows) == 288
    counts = {}
    for row in rows:
        parsed = parse_answer(row['decoded_output'])
        assert parsed == row['parsed_answer']
        assert row['correct'] == (parsed == labels[(row['source_id'], str(row['question_id']))])
        if row['repetition'] == 1:
            counts.setdefault(row['condition'], []).append(int(row['correct']))
    summary = read_json(legacy/'summary.json')
    for condition, values in counts.items():
        assert len(values) == 24
        assert sum(values)/24 == summary['metrics'][condition]
    new_records = list((run/'private/records').glob('*.json'))
    assert not new_records, 'This verifier is for blocked checkpoints; full scientific verifier is required for collection'
    candidates = read_json(run/'private/candidate_inventory.json')
    actual = read_json(run/'cohort_summary.json')
    assert len(candidates) == actual['short_source_identifiers']
    assert sum(c['reason'] == 'previously_exposed' for c in candidates) == 8
    assert actual['eligible_verified_new_source_groups'] == actual['frozen_source_groups'] == 0
    compatibility_path = run/'compatibility.json'
    compatibility = read_json(compatibility_path) if compatibility_path.exists() else None
    if compatibility:
        events = read_jsonl(run/'private/compatibility_calls.jsonl')
        assert compatibility['completed_model_calls'] == sum(e['event'] == 'completed' for e in events)
        assert compatibility['worker_reaped'] and compatibility['automatic_retries'] == 0
    result = {'at': utc_now(), 'checkpoint_checks_passed': True, 'scientific_completeness_passed': False,
        'historical_files_preserved': len(inventory), 'historical_preservation_passed': True,
        'e1_real_saved_records_independently_rescored': len(rows),
        'resource_lineage_reconciled': True, 'e2_scientific_records': 0,
        'compatibility_receipt_checked': compatibility is not None,
        'predictive_result_available': False, 'practical_result_available': False,
        'unverified': ['full E2 cohort/provenance', 'full scientific record verifier', 'complete collection orchestration',
                       'nested artifacts/evaluator integration', 'cost panel and B* integration', 'real deployed timing'],
        'scope': 'Validates blocked checkpoint and preserved evidence, not a successful E2 scientific experiment.'}
    write_json(run/'verification.json', result)
    return result
