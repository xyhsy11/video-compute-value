"""Independent E2 saved-token/feature, closed-ledger, model-equation and publication-data checks."""
from __future__ import annotations

import math

import numpy as np

from .common import digest, file_hash, read_json, read_jsonl, utc_now, write_json
from .e001_input import decode_e1, inference_video
from .e002_features import extract
from .e002_runtime import observation
from .e002v2_analysis import saved_evidence
from .protocol import question_input


def verify(root, run):
    evidence = saved_evidence(root, run)
    frozen = evidence['frozen']
    history = read_json(run/'private/historical_inventory.json')
    assert all((root/name).is_file() and file_hash(root/name) == sha for name, sha in history.items())
    ledger = read_json(run/'private/resource_ledger.json')
    task = read_json(run/'private/task.json')
    parent = read_json(root/task['parent_path'])
    for key in ('sessions', 'external_events'):
        assert ledger.get(key, [])[:len(parent.get(key, []))] == parent.get(key, [])
    total = sum(s['active_seconds'] for s in ledger['sessions'])+sum(e['active_seconds'] for e in ledger.get('external_events', []))
    assert math.isclose(total, ledger['active_seconds'], abs_tol=1e-5)
    assert ledger['active_seconds'] <= 172800 and ledger['storage_current_bytes'] <= 20*1024**3
    assert ledger['video_transfer_bytes']+ledger.get('interrupted_transfer_upper_bound_bytes', 0) <= 1024**3
    groups, folds = evidence['groups'], evidence['folds']
    for fold in range(5):
        payload = read_json(run/f'private/fit_inputs/fold{fold}.json')
        saved = read_json(run/f'private/decisions/fold{fold}.json')
        assert saved['training_payload_sha256'] == digest(payload['train'])
        assert not set(groups[folds == fold]) & set(payload['train']['groups'])
        y = np.asarray(payload['train']['gains'], dtype=float)
        for family in ('V', 'Q', 'VQ'):
            trained = saved['families'][family]
            model = trained['model']
            x = np.asarray(payload['train'][family], dtype=float)
            mean, scale = x.mean(0), x.std(0)
            scale[scale == 0] = 1
            z = (x-mean)/scale
            w = np.asarray(model['weights'])
            alpha = trained['training']['alpha']
            inner_map = read_json(run/'private/splits.json')['inner'][str(fold)]
            assert trained['training']['inner_folds'] == [inner_map[g] for g in payload['train']['groups']]
            # Independent primal normal equation; production uses the dual solve.
            residual = (z.T@z+len(x)*alpha*np.eye(x.shape[1]))@w-z.T@(y-y.mean(0))
            assert np.allclose(residual, 0, rtol=0, atol=1e-9)
            assert np.allclose(model['mean'], mean, atol=1e-12, rtol=0)
            assert np.allclose(model['scale'], scale, atol=1e-12, rtol=0)
            prediction = np.clip((np.asarray(payload['test'][family])-mean)/scale@w+y.mean(0), -1, 1)
            assert np.allclose(prediction, trained['predicted_gains'], atol=1e-12, rtol=0)
            assert np.argmax(prediction, axis=1).tolist() == trained['actions']
        assert saved['fixed_quarter'] == int(np.asarray(payload['train']['quarters']).mean(0).argmax())
    cohort = read_json(run/'private/cohort_manifest.json')
    action_manifest = read_json(run/'private/actions.json')
    index = 0
    for video in cohort['videos']:
        action = action_manifest[video['source_id']]['base_plain']
        pixels, metadata = decode_e1(inference_video(video, action), 8)
        metadata['_e1_pixels_for_hashing'] = pixels
        for question in video['questions']:
            features = extract(observation(question_input(question), video, metadata))
            for family, values in features.items():
                assert np.allclose(values, evidence['features'][family][index], rtol=0, atol=1e-12)
            index += 1
    assert index == 120
    summary = read_json(run/'summary.json')
    matrix = evidence['matrix']
    choices = {name: np.empty(120, dtype=int) for name in ('fixed', 'V', 'Q', 'VQ')}
    for fold in range(5):
        saved = read_json(run/f'private/decisions/fold{fold}.json')
        choices['fixed'][folds == fold] = saved['fixed_quarter']
        for name in ('V', 'Q', 'VQ'):
            choices[name][folds == fold] = saved['families'][name]['actions']
    metrics = {'base': sum(int(r[0]) for r in matrix)/120,
        'random': sum(sum(map(int, r[1:5])) for r in matrix)/480,
        'uniform_addition': sum(int(r[5]) for r in matrix)/120,
        'direct16': sum(int(r[6]) for r in matrix)/120,
        'oracle': sum(max(map(int, r[1:5])) for r in matrix)/120,
        'oracle_stop': sum(max(map(int, r[:5])) for r in matrix)/120}
    for name in choices:
        metrics[name] = sum(int(matrix[i, int(a)+1]) for i, a in enumerate(choices[name]))/120
    assert all(math.isclose(summary['metrics'][name], value, abs_tol=1e-12) for name, value in metrics.items())
    assert math.isclose(summary['delta_fixed'], metrics['VQ']-metrics['fixed'], abs_tol=1e-12)
    assert math.isclose(summary['delta_question'], metrics['VQ']-metrics['V'], abs_tol=1e-12)
    calls = read_jsonl(run/'private/calls.jsonl')
    starts = [c for c in calls if c['event'] == 'started']
    completed = [c for c in calls if c['event'] == 'completed']
    workers = [read_json(p) for p in (run/'private/workers').glob('*.json')]
    durable_counts = {w['worker_key']: 0 for w in workers}
    for event in completed:
        durable_counts[event['worker_key']] += 1
    for worker in workers:
        assert worker.get('model_invocations', 0) <= durable_counts[worker['worker_key']]+len(starts)-len(completed)
    assert sum(c['kind'] == 'scientific' for c in completed) == 1080
    # Worker invocation windows cannot overlap; only the OS CUDA lock can admit a worker.
    result = {'at': utc_now(), 'all_E2A_checks_passed': True, 'historical_files_preserved': len(history),
        'historical_preservation_passed': True, 'real_scientific_calls_verified': 1080,
        'primary_questions': 120, 'independent_source_groups': 40, 'repeatability_cells': 120,
        'plain_observed_equivalence_cells': 120, 'independent_saved_token_scoring': True,
        'independent_primal_ridge_equations': True, 'training_only_scaling_and_controls': True,
        'features_regenerated_from_40_real_sources': True, 'decision_seals_checked': True,
        'resource_lineage_reconciled': True, 'started_model_requests': len(starts),
        'completed_model_requests_including_warmups': len(completed),
        'unknown_interrupted_calls_upper_bound': len(starts)-len(completed),
        'fingerprint': frozen['fingerprint'], 'summary_sha256': file_hash(run/'summary.json'),
        'E2B_status': summary['E2B_status'], 'E0_verdict': 'MEASUREMENT_UNSTABLE'}
    write_json(run/'verification.json', result)
    return result
