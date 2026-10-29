"""Separately admitted E2B screen, train-only cost panels, B* and real live pipelines."""
from __future__ import annotations

import subprocess
import sys
import time

import numpy as np

from .common import Blocked, digest, exclusive_lock, file_hash, read_json, write_json
from .e001_input import install_decoder
from .e002_analysis import practical_comparison, timing_stability
from .e002_model import Ridge, CONFIDENCE_THRESHOLDS, select_threshold
from .e002_runtime import E2Adapter, execute_pipeline
from .e002v2_collect import finish_hashes, validate
from .protocol import question_input, video_input


DIRECT = ['direct8', 'direct16', 'quarter1', 'quarter2', 'quarter3', 'quarter4', 'uniform_addition']


def model_from_saved(model, alpha=1.):
    return Ridge(np.asarray(model['mean']), np.asarray(model['scale']), np.asarray(model['intercept']),
                 np.asarray(model['weights']), alpha)


def decide_features(features, specification):
    family = specification['family']
    expected = {'C': 3, 'V': 19, 'VQ': 341}
    features = np.asarray(features, dtype=float)
    if family not in expected or features.shape != (expected[family],) or not np.isfinite(features).all():
        raise ValueError('Only the strict declared policy feature vector is accepted')
    if family == 'C':
        threshold = specification['threshold']
        acquire = (threshold == 'always_acquire' or threshold != 'always_stop' and features[0] < threshold)
        action = specification['quarter']
    else:
        prediction = model_from_saved(specification['model']).predict(np.asarray(features)[None])[0]
        threshold = specification['threshold']
        acquire = (threshold == 'always_acquire' or threshold != 'always_stop' and prediction.max() > threshold)
        action = int(prediction.argmax())
    return f'quarter{action+1}' if acquire else 'stop'


def policy_spec(run, fold, method, phase):
    fit = read_json(run/f'private/decisions/fold{fold}.json')
    if method in DIRECT:
        return {'direct': 'base_plain' if method == 'direct8' else method}
    family = method.removesuffix('_stop')
    if phase == 'evaluation':
        return read_json(run/'private/practical_policies.json')[str(fold)][method]
    if family == 'C':
        return {'family': 'C', 'quarter': fit['fixed_quarter'],
                'threshold': 'always_stop' if method.endswith('_stop') else 'always_acquire'}
    return {'family': family, 'model': fit['families'][family]['model'],
            'threshold': 'always_stop' if method.endswith('_stop') else 'always_acquire'}


def timing_payloads(root, run, phase):
    plan = read_json(run/'private/practical_plan.json')
    cohort = read_json(run/'private/inference_manifest.json')['videos']
    lookup = {v['source_id']: v for v in cohort}
    cells = []
    if phase == 'screen':
        legacy = read_json(root/'results/e001_20261029_132707_v1/private/cohort_manifest.json')['videos']
        for video in legacy[:2]:
            clean = {**video_input(video), 'source_id': video['source_id']}
            question = {k: video['questions'][0][k] for k in ('question_id', 'question', 'options')}
            for method in plan['screen_methods']:
                cells.append({'fold': 0, 'source_id': video['source_id'], 'question': question, 'video': clean,
                              'method': method, 'spec': policy_spec(run, 0, method, phase)})
    elif phase == 'calibration':
        for fold, sources in plan['panels'].items():
            for source in sources:
                video = lookup[source]
                for question in video['questions']:
                    for method in plan['panel_methods']:
                        cells.append({'fold': int(fold), 'source_id': source, 'question': question,
                                      'video': video_input(video), 'method': method,
                                      'spec': policy_spec(run, int(fold), method, phase)})
    else:
        splits = read_json(run/'private/splits.json')
        for video in cohort:
            fold = splits['outer'][video['source_id']]
            for question in video['questions']:
                for method in plan['evaluation_methods']:
                    cells.append({'fold': fold, 'source_id': video['source_id'], 'question': question,
                                  'video': video_input(video), 'method': method,
                                  'spec': policy_spec(run, fold, method, phase)})
    for cell in cells:
        cell['cell_id'] = digest({k: cell[k] for k in ('fold', 'source_id', 'question', 'method', 'spec')})
    for session in (1, 2, 3):
        ordered = sorted(cells, key=lambda c: digest({'seed': 2026102902, 'phase': phase,
                                                     'session': session, 'cell': c['cell_id']}))
        value = {'phase': phase, 'session': session, 'cells': ordered,
                 'fingerprint': read_json(run/'private/frozen.json')['fingerprint']}
        path = run/f'private/timing_payloads/{phase}_{session}.json'
        if path.exists() and read_json(path) != value:
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Frozen timing order/method/route changed')
        write_json(path, value)
    return cells


def timing_worker(root, run, phase, session, key):
    from .e001 import configure
    configure(root)
    from .model import tensor_hashes
    frozen = validate(root, run, inference_only=True)
    payload = read_json(run/f'private/timing_payloads/{phase}_{session}.json')
    if payload['fingerprint'] != frozen['fingerprint']:
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Timing payload fingerprint differs')
    class TracedAdapter(E2Adapter):
        traces = None
        def prepare(self, *args, **kwargs):
            inputs, details = super().prepare(*args, **kwargs)
            if self.traces is not None:
                self.traces.append((inputs, details))
            return inputs, details
    path = run/f'private/timing_workers/{key}.json'
    receipt = {'phase': phase, 'session': session, 'worker_key': key, 'stage': 'model_loading'}
    with exclusive_lock(root/'.cache/cuda_worker.lock'):
        write_json(path, receipt)
        install_decoder()
        adapter = TracedAdapter(frozen)
        receipt.update(stage='generation', load_seconds=adapter.load_seconds, model_calls=0)
        write_json(path, receipt)
        first = payload['cells'][0]
        # Two fixed native8 warmups in every fresh process, separately counted.
        for _ in range(2):
            execute_pipeline(adapter, question_input(first['question']), first['video'], direct='base_plain')
        for index, cell in enumerate(payload['cells'], 1):
            output_path = run/f"private/timing_records/{phase}_{session}_{cell['cell_id']}.json"
            if output_path.exists():
                raise Blocked('INPUT_INTEGRITY_FAILED', 'Timing sessions are immutable and cannot be selectively repeated')
            adapter.traces = []
            decision_seal = []
            spec = cell['spec']
            adapter.torch.cuda.reset_peak_memory_stats()
            result = execute_pipeline(adapter, question_input(cell['question']), cell['video'],
                direct=spec.get('direct'), family=spec.get('family'),
                decide=None if 'direct' in spec else lambda features: decide_features(features, spec),
                seal=None if 'direct' in spec else lambda action: decision_seal.append(action))
            result['peak_allocated_bytes'] = adapter.torch.cuda.max_memory_allocated()
            result['peak_reserved_bytes'] = adapter.torch.cuda.max_memory_reserved()
            traces = []
            for inputs, details in adapter.traces:
                hashes = tensor_hashes(inputs)
                finish_hashes(details)
                traces.append({'input_tensor_hashes': hashes, 'source_indices': details['source_indices'],
                               'timestamps': details['actual_timestamps']})
            adapter.traces = None
            result.update(cell_id=cell['cell_id'], source_id=cell['source_id'], fold=cell['fold'],
                          question_id=str(cell['question']['question_id']), method=cell['method'],
                          session=session, phase=phase, worker_key=key, traces=traces, decision_seal=decision_seal)
            result['seal'] = digest(result)
            write_json(output_path, result)
            receipt['model_calls'] = adapter.model_call_count
            write_json(path, receipt)
            if index % 24 == 0:
                print(f'{phase} session {session}: {index}/{len(payload["cells"])} live pipelines', flush=True)
        receipt.update(stage='complete', model_calls=adapter.model_call_count)
        write_json(path, receipt)


def run_timing_stage(root, run, budget, phase):
    cells = timing_payloads(root, run, phase)
    prior_probe = read_json(root/'results/e002_20261029_073134_v1/private/compatibility_worker.json')
    cpu = read_json(run/'private/cpu_preflight.json')['checks']
    maximum = max(r['wall_seconds'] for values in cpu.values() for r in values.values())+14.
    calls_per_session = sum(1 if 'direct' in c['spec'] or c['spec']['threshold'] == 'always_stop' else 2 for c in cells)+2
    budget.admit('E2B_'+phase, 3*(calls_per_session*maximum+prior_probe['load_seconds']), storage=128*1024**2,
        detail={'pipeline_executions': len(cells)*3, 'model_calls_upper_bound_including_warmups': calls_per_session*3,
                'three_fresh_process_sessions': True, 'timing_panel_plan_sha256': file_hash(run/'private/practical_plan.json')})
    for session in (1, 2, 3):
        receipt_path = run/f'private/timing_stage/{phase}_{session}.json'
        if receipt_path.exists():
            if not read_json(receipt_path)['completed']:
                raise Blocked('PRACTICAL_VALUE_NOT_EVALUABLE_MEASUREMENT', 'Preserved timing session failed; no favorable rescreen')
            continue
        key = f'{phase}_{session}_{time.time_ns()}'
        write_json(receipt_path, {'completed': False, 'worker_key': key, 'started': True})
        log = run/f'private/{key}.log'
        with log.open('w', encoding='utf-8') as output:
            try:
                result = subprocess.run([sys.executable, '-m', 'video_compute_value.e002_continuation', '_timing_worker',
                    '--run-dir', str(run), '--phase', phase, '--session', str(session), '--worker-key', key],
                    cwd=root, stdout=output, stderr=output, timeout=min(28800, int(172800-budget.consumed_seconds()-2400)))
            except subprocess.TimeoutExpired as exc:
                write_json(receipt_path, {'completed': False, 'worker_key': key, 'worker_reaped': True, 'timed_out': True})
                raise Blocked('PRACTICAL_VALUE_NOT_EVALUABLE_MEASUREMENT', 'Owned timing worker timed out and was reaped') from exc
        write_json(receipt_path, {'completed': result.returncode == 0, 'worker_key': key,
            'returncode': result.returncode, 'worker_reaped': True})
        budget.checkpoint(f'E2B {phase} timing session {session}')
        if result.returncode:
            raise Blocked('PRACTICAL_VALUE_NOT_EVALUABLE_MEASUREMENT', 'Timing worker failed; no automatic rescreen')
    rows = [read_json(p) for p in (run/'private/timing_records').glob(f'{phase}_*.json')]
    by_method = {}
    for cell in cells:
        repeated = sorted([r for r in rows if r['cell_id'] == cell['cell_id']], key=lambda r: r['session'])
        if len(repeated) != 3 or len({r['worker_key'] for r in repeated}) != 3:
            raise Blocked('PRACTICAL_VALUE_NOT_EVALUABLE_MEASUREMENT', 'Missing three independent timing sessions')
        ref = repeated[0]
        if any(r['action'] != ref['action'] or r['traces'] != ref['traces'] or
               r['output']['generated_token_ids'] != ref['output']['generated_token_ids'] for r in repeated):
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Live decision/input/output repeatability differs')
        group_fold = cell['fold'] if phase == 'calibration' else 0
        by_method.setdefault((group_fold, cell['method']), []).append([r['total_seconds'] for r in repeated])
    stability = {f'{fold}:{method}': timing_stability(values) for (fold, method), values in by_method.items()}
    write_json(run/f'{phase}_timing_stability.json', stability)
    if not all(s['passed'] for s in stability.values()):
        raise Blocked('PRACTICAL_VALUE_NOT_EVALUABLE_MEASUREMENT', 'Frozen 20% median-range timing criterion failed')
    return rows


def train_practical_policies(root, run, calibration):
    from .e002v2_analysis import saved_evidence
    evidence = saved_evidence(root, run)
    matrix, folds = evidence['matrix'], evidence['folds']
    records = [read_json(p) for p in (run/'private/records').glob('*.json')]
    c_lookup = {(r['source_id'], r['question_id']): r['C'][0] for r in records
                if r['condition'] == 'base_observed' and r['repetition'] == 1}
    policies = {}
    for fold in range(5):
        train = folds != fold
        costs = {method: float(np.mean([r['total_seconds'] for r in calibration if r['fold'] == fold and r['method'] == method]))
                 for method in [*DIRECT, 'V_stop', 'VQ_stop', 'C_stop']}
        fit = read_json(run/f'private/decisions/fold{fold}.json')
        policies[str(fold)] = {}
        for family in ('V', 'VQ'):
            predictions = np.asarray(fit['families'][family]['inner_oof_predictions'])
            choice = select_threshold(predictions, matrix[train, 0], matrix[train, 1:5], costs[family+'_stop'], 0,
                max(costs[f'quarter{i}'] for i in range(1, 5)), costs['direct16'], qualified=True)
            if choice.get('threshold') is None:
                raise Blocked('NO_FEASIBLE_POLICY_AT_REFERENCE_BUDGET', 'No training-budget-feasible '+family+' policy')
            policies[str(fold)][family] = {'family': family, 'model': fit['families'][family]['model'],
                                         'threshold': choice['threshold'], 'training_choice': choice}
        candidates = []
        direct_columns = {'direct8': 0, 'direct16': 6, 'quarter1': 1, 'quarter2': 2, 'quarter3': 3,
                          'quarter4': 4, 'uniform_addition': 5}
        for method, column in direct_columns.items():
            if costs[method] <= costs['direct16']:
                candidates.append((float(matrix[train, column].mean()), -costs[method], 0.,
                                   {'direct': 'base_plain' if method == 'direct8' else method}))
        groups = evidence['groups'][train]
        inner_folds = np.asarray(fit['families']['VQ']['training']['inner_folds'])
        quarters = matrix[train, 1:5]
        fixed = np.empty(len(groups), dtype=int)
        for inner in range(4):
            fixed[inner_folds == inner] = int(quarters[inner_folds != inner].mean(0).argmax())
        confidence = np.array([c_lookup[(cell['source'], cell['question_id'])] for cell, keep in zip(evidence['cells'], train) if keep])
        for threshold in CONFIDENCE_THRESHOLDS:
            acquire = np.ones(len(groups), bool) if threshold == 'always_acquire' else (
                np.zeros(len(groups), bool) if threshold == 'always_stop' else confidence < threshold)
            cost = costs['C_stop']+acquire*np.array([costs[f'quarter{a+1}'] for a in fixed])
            accuracy = np.where(acquire, quarters[np.arange(len(groups)), fixed], matrix[train, 0]).mean()
            if cost.mean() <= costs['direct16']:
                candidates.append((float(accuracy), -float(cost.mean()), -float(acquire.mean()),
                    {'family': 'C', 'quarter': fit['fixed_quarter'], 'threshold': threshold}))
        best = max(candidates, key=lambda row: row[:3])  # Direct16 is always admissible.
        policies[str(fold)]['Bstar'] = best[3]
        policies[str(fold)]['direct16'] = {'direct': 'direct16'}
        policies[str(fold)]['training_costs'] = costs
        policies[str(fold)]['training_break_even'] = {
            family: (costs['direct16']-costs[family+'_stop'])/max(costs[f'quarter{i}'] for i in range(1, 5))
            for family in ('V', 'VQ')}
    path = run/'private/practical_policies.json'
    if path.exists() and read_json(path) != policies:
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Training-only practical policy seal changed')
    write_json(path, policies)
    write_json(run/'practical_policy_seal.json', {'policies_sha256': file_hash(path), 'outer_folds': 5,
              'training_only_costs_and_thresholds': True, 'reference': 'Direct16'})
    return policies


def evaluate(root, run, budget):
    summary = read_json(run/'summary.json')
    if summary['predictive_value'] != 'PREDICTIVE_SIGNAL_IN_DEVELOPMENT_PILOT':
        raise Blocked('NOT_ADMITTED_E2A_ADVANCEMENT', 'E2A failed its advancement rule; E2B cannot proceed')
    try:
        run_timing_stage(root, run, budget, 'screen')
        calibration = run_timing_stage(root, run, budget, 'calibration')
        train_practical_policies(root, run, calibration)
        rows = run_timing_stage(root, run, budget, 'evaluation')
    except Blocked as exc:
        status = {'status': exc.status, 'practical_value': 'PRACTICAL_VALUE_NOT_EVALUABLE_MEASUREMENT',
                  'timing_validity': 'FAILED_OR_UNQUALIFIED', 'reason': str(exc)}
        write_json(run/'practical_status.json', status)
        summary.update(E2B_status=exc.status, timing_validity=status['timing_validity'])
        write_json(run/'summary.json', summary)
        return status
    from .e002v2_analysis import saved_evidence
    evidence = saved_evidence(root, run)
    labels = {(v['source_id'], str(q['question_id'])): q['answer']
              for v in read_json(run/'private/cohort_manifest.json')['videos'] for q in v['questions']}
    by_cell = {(r['source_id'], r['question_id'], r['method'], r['session']): r for r in rows}
    counterfactuals = {(r['source_id'], r['question_id'], r['condition']): r
        for r in [read_json(p) for p in (run/'private/records').glob('*.json')] if r['repetition'] == 1}
    policies = read_json(run/'private/practical_policies.json')
    times, scores, overhead = {}, {}, {}
    for method in ('V', 'VQ', 'direct16', 'Bstar'):
        times[method], scores[method], overhead[method] = [], [], []
        for cell in evidence['cells']:
            repeated = [by_cell[(cell['source'], cell['question_id'], method, session)] for session in (1, 2, 3)]
            fold = str(repeated[0]['fold'])
            specification = policies[fold][method]
            observed = counterfactuals[(cell['source'], cell['question_id'], 'base_observed')]
            expected_action = specification.get('direct')
            if expected_action is None:
                family = specification['family']
                allowed = observed['C'] if family == 'C' else observed['features'][family]
                expected_action = decide_features(np.asarray(allowed), specification)
            condition = 'base_plain' if expected_action == 'stop' else expected_action
            expected_tokens = counterfactuals[(cell['source'], cell['question_id'], condition)]['generated_token_ids']
            if any(r['action'] != expected_action or r['output']['generated_token_ids'] != expected_tokens for r in repeated):
                raise Blocked('INPUT_INTEGRITY_FAILED', 'Actual live policy differs from sealed offline replay')
            times[method].append([r['total_seconds'] for r in repeated])
            scores[method].append(int(repeated[0]['output']['parsed_answer'] == labels[(cell['source'], cell['question_id'])]))
            overhead[method].append(np.mean([r['components_seconds']['observation']+r['components_seconds']['selection'] for r in repeated]))
    comparisons = {reference: practical_comparison(np.asarray(scores['VQ'])-scores[reference], times['VQ'],
        times[reference], evidence['groups'], overhead['VQ'], gates={'integrity': True, 'timing': True, 'complete': True})
        for reference in ('direct16', 'Bstar')}
    success = all(c['comparison_passed'] for c in comparisons.values())
    result = {'status': 'COMPLETE', 'timing_validity': 'QUALIFIED_THREE_SESSIONS', 'comparisons': comparisons,
              'practical_value': 'PRACTICAL_VALUE_DEMONSTRATED_IN_PILOT' if success else 'PRACTICAL_VALUE_NOT_DEMONSTRATED',
              'methods': {m: {'accuracy': float(np.mean(scores[m])), 'mean_seconds': float(np.mean(times[m])),
                  'p50_seconds': float(np.median(times[m])), 'p95_seconds': float(np.quantile(times[m], .95)),
                  'mean_observation_selection_seconds': float(np.mean(overhead[m])),
                  'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in rows if r['method'] == m),
                  'peak_reserved_bytes': max(r['peak_reserved_bytes'] for r in rows if r['method'] == m)}
                  for m in times}, 'pipeline_executions': len(rows)}
    from .e002v2_analysis import export_csv
    export_csv(run/'policy_costs.csv', ['method', 'accuracy', 'mean_complete_seconds', 'p50_seconds', 'p95_seconds'],
        [[m, result['methods'][m]['accuracy'], result['methods'][m]['mean_seconds'],
          result['methods'][m]['p50_seconds'], result['methods'][m]['p95_seconds']] for m in times])
    export_csv(run/'cost_breakdown.csv', ['method', 'base_or_direct_seconds', 'observation_seconds', 'selection_seconds', 'followup_seconds'],
        [[m, *[float(np.mean([r['components_seconds'][c] for r in rows if r['method'] == m]))
                for c in ('base_or_direct', 'observation', 'selection', 'followup')]] for m in times])
    write_json(run/'practical_status.json', result)
    summary.update(E2B_status=result['status'], timing_validity=result['timing_validity'], practical_value=result['practical_value'])
    write_json(run/'summary.json', summary)
    return result
