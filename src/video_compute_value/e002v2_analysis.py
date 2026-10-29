"""Strict train/test payload fitting, sealed OOF decisions, independent saved-token evaluation."""
from __future__ import annotations

import csv
import io
from pathlib import Path
import subprocess
import sys

import numpy as np
from tokenizers import Tokenizer

from .common import Blocked, atomic_text, digest, file_hash, read_json, read_jsonl, write_json
from .e002_analysis import predictive_summary
from .e002_features import SCHEMA
from .e002_model import forced_actions, train_fold
from .e002v2_collect import validate
from .e002v2_records import records
from .protocol import parse_answer


def saved_evidence(root, run):
    frozen = validate(root, run)
    manifest = read_json(run/'private/call_manifest.json')
    saved = records(run, manifest)
    if len(saved) != 1080:
        raise Blocked('PARTIAL', f'{len(saved)}/1080 scientific calls; missing evidence is not scored as incorrect')
    events = read_jsonl(run/'private/calls.jsonl')
    starts = {e['call_id']: e for e in events if e['event'] == 'started'}
    ends = {e['call_id']: e for e in events if e['event'] == 'completed'}
    tokenizer = Tokenizer.from_file(str(Path(frozen['model_path'])/'tokenizer.json'))
    cohort = read_json(run/'private/cohort_manifest.json')
    cpu = read_json(run/'private/cpu_preflight.json')['checks']
    actions = read_json(run/'private/actions.json')
    splits = read_json(run/'private/splits.json')
    lookup = {(r['source_id'], r['question_id'], r['condition'], r['repetition']): r for r in saved.values()}
    # Audit every completed receipt, including all prescribed repeats, before selecting pass1 for accuracy.
    for row in saved.values():
        if row['call_id'] not in starts or ends.get(row['call_id'], {}).get('record_hash') != row['record_hash']:
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Completed repeat/request has no valid worker-call receipt')
        decoded = tokenizer.decode(row['generated_token_ids'], skip_special_tokens=True)
        worker = read_json(run/f"private/workers/{row['worker_key']}.json")
        if (decoded != row['decoded_output'] or parse_answer(decoded) != row['parsed_answer']
                or row['generation_policy_sha256'] != digest(worker['generation_policy'])
                or worker['fingerprint'] != frozen['fingerprint']):
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Independent all-record token/worker audit failed')
    matrix, groups, features, cells, folds = [], [], {k: [] for k in ('V', 'Q', 'VQ')}, [], []
    repeat_keys = ('generated_token_ids', 'input_tensor_hashes', 'source_indices', 'actual_timestamps',
                   'model_timestamps', 'processed_rgb_hashes', 'video_grid_thw', 'visual_tokens')
    repeats = 0
    invalid = {c: 0 for c in ('base_plain', 'base_observed', 'quarter1', 'quarter2', 'quarter3', 'quarter4',
                               'uniform_addition', 'direct16')}
    for source_index, video in enumerate(cohort['videos'], 1):
        source = video['source_id']
        for question_index, question in enumerate(video['questions'], 1):
            question_id = str(question['question_id'])
            by_condition = {}
            for condition in invalid:
                row = lookup[(source, question_id, condition, 1)]
                if (row['call_id'] not in starts or ends.get(row['call_id'], {}).get('record_hash') != row['record_hash']):
                    raise Blocked('INPUT_INTEGRITY_FAILED', 'Scientific model invocation receipt missing')
                worker = read_json(run/f"private/workers/{row['worker_key']}.json")
                if (worker['generation_policy']['effective_mode'] != 'greedy_search'
                        or row['generation_policy_sha256'] != digest(worker['generation_policy'])
                        or worker['fingerprint'] != frozen['fingerprint'] or worker['allocator_backend'] != 'native'):
                    raise Blocked('INPUT_INTEGRITY_FAILED', 'Effective generation/worker receipt mismatch')
                decoded = tokenizer.decode(row['generated_token_ids'], skip_special_tokens=True)
                parsed = parse_answer(decoded)
                if (decoded != row['decoded_output'] or parsed != row['parsed_answer']
                        or row['valid_answer'] != (parsed is not None)
                        or row['output_tokens'] != len(row['generated_token_ids']) or row['output_tokens'] > 8):
                    raise Blocked('INPUT_INTEGRITY_FAILED', 'Independent real token decoding/parser differs')
                expected, native = actions[source][condition], cpu[source][condition]
                for key in ('source_indices', 'actual_timestamps', 'model_timestamps', 'video_grid_thw', 'visual_tokens',
                            'frames_processor', 'frames_represented', 'processed_rgb_hashes'):
                    if row[key] != native[key]:
                        raise Blocked('INPUT_INTEGRITY_FAILED', 'Actual native input differs from all-source CPU preflight')
                if row['source_indices'] != expected['indices']:
                    raise Blocked('INPUT_INTEGRITY_FAILED', 'Frozen source action indices differ')
                for key in ('pixel_values_videos', 'video_grid_thw'):
                    if row['input_tensor_hashes'][key] != native['input_tensor_hashes'][key]:
                        raise Blocked('INPUT_INTEGRITY_FAILED', 'Actual native video tensors differ')
                by_condition[condition] = row
                invalid[condition] += int(parsed is None)
                if source in splits['repeat_sources']:
                    other = lookup[(source, question_id, condition, 2)]
                    if (row['worker_key'] == other['worker_key'] or any(row[k] != other[k] for k in repeat_keys)):
                        raise Blocked('REPEATABILITY_UNRESOLVED', 'Fresh-process native input/output repeatability failed')
                    if condition == 'base_observed':
                        if any(not np.allclose(row['features'][k], other['features'][k], rtol=0, atol=1e-12)
                               for k in features) or not np.allclose(row['C'], other['C'], rtol=0, atol=1e-12):
                            raise Blocked('REPEATABILITY_UNRESOLVED', 'Observed feature/score summaries changed across processes')
                    repeats += 1
            plain, observed = by_condition['base_plain'], by_condition['base_observed']
            if (plain['generated_token_ids'] != observed['generated_token_ids']
                    or plain['input_tensor_hashes'] != observed['input_tensor_hashes']):
                raise Blocked('BLOCKED_FEATURE_INTEGRITY', 'Same-call C capture changed the paid base output/input')
            for family in features:
                value = observed['features'][family]
                if len(value) != SCHEMA[family] or not np.isfinite(value).all():
                    raise Blocked('BLOCKED_FEATURE_INTEGRITY', 'Saved feature schema differs')
                features[family].append(value)
            conditions = ('base_plain', 'quarter1', 'quarter2', 'quarter3', 'quarter4', 'uniform_addition', 'direct16')
            matrix.append([int(by_condition[c]['parsed_answer'] == question['answer']) for c in conditions])
            groups.append(source)
            folds.append(splits['outer'][source])
            cells.append({'anonymous': f'S{source_index:02}Q{question_index}', 'source': source, 'question_id': question_id})
    if repeats != 120:
        raise Blocked('REPEATABILITY_UNRESOLVED', 'Not all 120 prespecified repeated cells verified')
    return {'matrix': np.asarray(matrix), 'features': {k: np.asarray(v) for k, v in features.items()},
            'groups': np.asarray(groups), 'folds': np.asarray(folds), 'cells': cells,
            'gates': {'complete_1080': True, 'same_call_equivalence_120': True, 'fresh_process_repeat_cells': 120,
                      'native_cpu_gpu_tensors': True, 'effective_greedy': True, 'saved_token_rescoring': True},
            'invalid_primary_answers': invalid, 'frozen': frozen}


def fit_payload(payload):
    """Pure policy boundary: no filesystem, test labels, candidate outputs, IDs or test costs."""
    if set(payload) != {'fold', 'train', 'test'} or set(payload['test']) != {'V', 'Q', 'VQ'}:
        raise ValueError('Held-out policy payload may contain only allowed feature matrices')
    train = payload['train']
    if set(train) != {'V', 'Q', 'VQ', 'gains', 'groups', 'quarters'}:
        raise ValueError('Strict training schema required')
    quarters = np.asarray(train['quarters'])
    fixed = int(np.argmax(quarters.mean(0)))
    result = {'fold': payload['fold'], 'fixed_quarter': fixed, 'families': {},
              'training_payload_sha256': digest(train), 'heldout_features_sha256': digest(payload['test'])}
    for family in ('V', 'Q', 'VQ'):
        model, inner, receipt = train_fold(np.asarray(train[family]), np.asarray(train['gains']),
                                          np.asarray(train['groups']), salt=f"inner:{payload['fold']}")
        predictions = model.predict(payload['test'][family])
        result['families'][family] = {'predicted_gains': predictions.tolist(),
            'actions': forced_actions(predictions).tolist(), 'training': receipt,
            'inner_oof_predictions': inner.tolist(), 'model': {'mean': model.mean.tolist(),
                'scale': model.scale.tolist(), 'intercept': model.intercept.tolist(), 'weights': model.weights.tolist()}}
    result['seal'] = digest(result)
    return result


def fit(root, run):
    evidence = saved_evidence(root, run)
    x, groups, folds = evidence['features'], evidence['groups'], evidence['folds']
    matrix = evidence['matrix']
    hashes = {}
    for fold in range(5):
        train, test = folds != fold, folds == fold
        if set(groups[train]) & set(groups[test]) or len(set(groups[test])) != 8:
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Source leakage or unbalanced frozen outer folds')
        payload = {'fold': fold,
            'train': {**{k: v[train].tolist() for k, v in x.items()}, 'groups': groups[train].tolist(),
                      'gains': (matrix[train, 1:5]-matrix[train, :1]).tolist(),
                      'quarters': matrix[train, 1:5].tolist()},
            'test': {k: v[test].tolist() for k, v in x.items()}}
        input_path, output_path = run/f'private/fit_inputs/fold{fold}.json', run/f'private/decisions/fold{fold}.json'
        write_json(input_path, payload)
        if output_path.exists():
            old = read_json(output_path)
            if old['training_payload_sha256'] != digest(payload['train']) or old['heldout_features_sha256'] != digest(payload['test']):
                raise Blocked('INPUT_INTEGRITY_FAILED', 'Preserved sealed decisions differ from immutable fitting inputs')
        else:
            # The worker receives only a strict feature/train-label payload, never the run/record directory.
            subprocess.run([sys.executable, '-m', 'video_compute_value.e002_continuation', '_fit_fold',
                '--fit-input', str(input_path), '--fit-output', str(output_path)], cwd=root, check=True, timeout=180)
        hashes[str(fold)] = file_hash(output_path)
    seal = {'fingerprint': evidence['frozen']['fingerprint'], 'fold_decision_sha256': hashes,
            'rows_per_fold': [24]*5, 'all_outer_test_decisions_sealed_before_evaluation_join': True}
    seal['seal'] = digest(seal)
    path = run/'private/decision_seal_manifest.json'
    if path.exists() and read_json(path) != seal:
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Decision seal manifest changed')
    write_json(path, seal)
    write_json(run/'decision_seal_summary.json', {'folds': 5, 'heldout_questions': 120, 'decisions_sealed': True,
              'seal_manifest_sha256': file_hash(path), 'training_only_fitting': True})
    return seal


def analyze(root, run):
    evidence = saved_evidence(root, run)
    seal = read_json(run/'private/decision_seal_manifest.json')
    if not seal['all_outer_test_decisions_sealed_before_evaluation_join']:
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Evaluator cannot join unsealed held-out actions')
    matrix, groups, folds = evidence['matrix'], evidence['groups'], evidence['folds']
    choices = {name: np.empty(120, dtype=int) for name in ('fixed', 'V', 'Q', 'VQ')}
    predictions = {name: np.empty((120, 4)) for name in ('V', 'Q', 'VQ')}
    for fold in range(5):
        path = run/f'private/decisions/fold{fold}.json'
        if file_hash(path) != seal['fold_decision_sha256'][str(fold)]:
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Held-out decision seal changed before join')
        output = read_json(path)
        if output['seal'] != digest({k: v for k, v in output.items() if k != 'seal'}):
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Corrupt policy decision seal')
        test = folds == fold
        choices['fixed'][test] = output['fixed_quarter']
        for family in ('V', 'Q', 'VQ'):
            choices[family][test] = output['families'][family]['actions']
            predictions[family][test] = output['families'][family]['predicted_gains']
    summary = predictive_summary(matrix[:, 0], matrix[:, 1:5], matrix[:, 5], matrix[:, 6],
                                  choices, groups, folds, complete=True)
    gains = matrix[:, 1:5]-matrix[:, :1]
    summary.update(source_groups=40, questions=120, primary_model_calls=960, repeatability_model_calls=120,
        invalid_primary_answers=evidence['invalid_primary_answers'], integrity=evidence['gates'],
        gain_mae={k: float(np.abs(p-gains).mean()) for k, p in predictions.items()},
        scientific_protocol='e002_preacquisition_predictability_v1', implementation='continuation_v2',
        E2B_status='NOT_ADMITTED_E2A_ADVANCEMENT' if summary['predictive_value'] != 'PREDICTIVE_SIGNAL_IN_DEVELOPMENT_PILOT'
                   else 'PENDING_PROSPECTIVE_TIMING_SCREEN',
        timing_validity='UNQUALIFIED', practical_value='PRACTICAL_VALUE_NOT_EVALUABLE_MEASUREMENT')
    summary['metrics_correct_counts'] = {name: float(value*120) for name, value in summary['metrics'].items()}
    export_csv(run/'metrics.csv', ['method', 'accuracy', 'correct_or_expected_correct', 'questions'],
        [[k, v, v*120, 120] for k, v in summary['metrics'].items()])
    export_csv(run/'fold_effects.csv', ['fold', 'delta_fixed', 'delta_question'],
        [[r[k] for k in ('fold', 'delta_fixed', 'delta_question')] for r in summary['fold_effects']])
    # Preserve prospective source order for anonymous effects; source IDs remain private.
    per_source = []
    for index, source in enumerate(dict.fromkeys(groups), 1):
        mask = groups == source
        selected = matrix[np.arange(120), choices['VQ']+1]
        fixed = matrix[np.arange(120), choices['fixed']+1]
        visual = matrix[np.arange(120), choices['V']+1]
        per_source.append([f'S{index:02}', int(folds[mask][0]), float((selected-fixed)[mask].mean()),
                           float((selected-visual)[mask].mean())])
    export_csv(run/'per_video_effects.csv', ['anonymous_source', 'fold', 'delta_fixed', 'delta_question'], per_source)
    summary['per_source_effects'] = [{'source': r[0], 'fold': r[1], 'delta_fixed': r[2], 'delta_question': r[3]} for r in per_source]
    write_json(run/'summary.json', summary)
    cpu = read_json(run/'private/cpu_preflight.json')['checks']
    text_differences = []
    for index, source in enumerate(dict.fromkeys(groups), 1):
        values = cpu[source]
        augmented = [r['input_tokens']-r['visual_tokens'] for name, r in values.items()
                     if name not in ('base_plain', 'base_observed')]
        for name, row in values.items():
            if name not in ('base_plain', 'base_observed'):
                text_tokens = row['input_tokens']-row['visual_tokens']
                text_differences.append([f'S{index:02}', name, row['visual_tokens'], text_tokens, text_tokens-min(augmented)])
    export_csv(run/'augmented_text_token_differences.csv',
        ['anonymous_source', 'condition', 'visual_tokens', 'text_tokens_first_original_question', 'text_difference_from_minimum_augmented'], text_differences)
    export_csv(run/'correctness.csv', ['anonymous_question', 'base', 'quarter1', 'quarter2', 'quarter3', 'quarter4',
                                     'uniform_addition', 'direct16', 'fixed_action', 'V_action', 'Q_action', 'VQ_action'],
        [[cell['anonymous'], *map(int, row), *[int(choices[name][i]) for name in ('fixed', 'V', 'Q', 'VQ')]]
         for i, (cell, row) in enumerate(zip(evidence['cells'], matrix))])
    calibration = []
    for family, prediction in predictions.items():
        for lo, hi in zip(np.linspace(-1, 1, 11)[:-1], np.linspace(-1, 1, 11)[1:]):
            mask = (prediction >= lo) & (prediction <= hi if hi == 1 else prediction < hi)
            if mask.any():
                calibration.append([family, lo, hi, int(mask.sum()), float(prediction[mask].mean()), float(gains[mask].mean())])
    export_csv(run/'gain_calibration.csv', ['family', 'bin_low', 'bin_high', 'action_pairs', 'mean_predicted_gain', 'mean_observed_gain'], calibration)
    return summary


def export_csv(path, header, rows):
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())
