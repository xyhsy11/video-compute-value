"""Independent saved-token scoring, record audit and public-safe aggregate export."""
from __future__ import annotations

import csv
import io
import math

from .common import Blocked, atomic_text, digest, file_hash, read_json, read_jsonl, utc_now, write_json
from .e001_input import CONDITIONS, actions
from .protocol import parse_answer


def saved_matrix(config, run, root):
    from transformers import AutoTokenizer
    from .e001 import records, validate
    frozen = validate(config, run, root)
    cohort = read_json(run/'private/cohort_manifest.json')
    acquisition = read_json(run/'private/action_manifest.json')
    cpu = read_json(run/'private/cpu_preflight.json')
    ready = read_json(run/'private/collection_ready.json')
    rows = records(run, frozen, cohort)
    if len(rows) != 288:
        raise Blocked('PARTIAL', f'{len(rows)}/288 requests complete; missing measurements cannot be scored')
    tokenizer = AutoTokenizer.from_pretrained(frozen['model_path'], local_files_only=True, trust_remote_code=False)
    lookup = {(r['source_id'], r['question_id'], r['condition'], r['repetition']): r for r in rows.values()}
    events = read_jsonl(run/'private/worker_calls.jsonl')
    starts = [e for e in events if e['event'] == 'started']
    terminal = [e for e in events if e['event'] in ('completed', 'failed')]
    if len({e['call_id'] for e in starts}) != len(starts) or len({e['call_id'] for e in terminal}) != len(terminal):
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Duplicate call events')
    started = {e['call_id']: e for e in starts}
    ended = {e['call_id']: e for e in terminal}
    if not set(ended) <= set(started):
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Call ended without a recorded start')
    receipts = {p.stem: read_json(p) for p in (run/'private/workers').glob('*.json')}
    matrix, videos, anonymous, repeats, base_discrepancies = [], [], [], [], []
    invalid = {c: 0 for c in CONDITIONS}
    old = [read_json(p) for p in (root/config['parent_run']/'records').glob('*.json')]
    old = {(r['source_id'], str(r['question_id'])): r for r in old if r['budget'] == 8 and r['repetition'] == 1}
    repeat_keys = ('input_tensor_hashes', 'input_metadata_hash', 'generated_token_ids', 'decoded_output',
                   'parsed_answer', 'retained_base_rgb_hashes', 'processed_rgb_hashes')
    token_ranges = {c: [] for c in CONDITIONS}
    timings = {c: [] for c in CONDITIONS}
    peaks = {c: {'allocated': [], 'reserved': []} for c in CONDITIONS}
    for vi, video in enumerate(cohort['videos'], 1):
        source = video['source_id']
        if actions(video) != acquisition[source]:
            raise Blocked('INPUT_INTEGRITY_FAILED', 'Acquisitions do not regenerate')
        for qi, question in enumerate(video['questions'], 1):
            result = []
            for condition in CONDITIONS:
                pair = [lookup[(source, str(question['question_id']), condition, rep)] for rep in (1, 2)]
                for row in pair:
                    receipt = receipts[row['worker_key']]
                    terminal_event = ended.get(row['call_id'], {})
                    if (terminal_event.get('event') != 'completed' or
                        terminal_event.get('record_hash') != row['record_hash'] or
                        receipt['fingerprint'] != frozen['fingerprint'] or
                        receipt['generation_policy'] != frozen['generation_policy'] or
                        receipt['generation_policy']['effective_mode'] != 'greedy_search' or
                        receipt['allocator_backend'] != 'native' or
                        row['generation_policy_sha256'] != digest(frozen['generation_policy'])):
                        raise Blocked('INPUT_INTEGRITY_FAILED', 'Effective generation or worker/call receipt mismatch')
                    decoded = tokenizer.decode(row['generated_token_ids'], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
                    answer = parse_answer(decoded)
                    if (decoded != row['decoded_output'] or answer != row['parsed_answer'] or
                        row['valid_answer'] != (answer is not None) or
                        row['correct'] != (answer == question['answer']) or
                        row['output_tokens'] != len(row['generated_token_ids']) or row['output_tokens'] > 8):
                        raise Blocked('INPUT_INTEGRITY_FAILED', 'Independent saved-token decoding or label scoring differs')
                    action = acquisition[source][condition]
                    ref = cpu['checks'][source][condition]
                    budget = 8 if condition == 'base' else 16
                    if (row['source_indices'] != action['indices'] or row['actual_timestamps'] != action['timestamps'] or
                        any(row[k] != ref[k] for k in ('retained_base_rgb_hashes', 'processed_rgb_hashes',
                            'frames_decoded', 'frames_selected', 'frames_processor', 'frames_represented',
                            'visual_tokens', 'video_grid_thw', 'model_timestamps', 'model_timestamp_text',
                            'spatial_height', 'spatial_width', 'temporal_padding')) or
                        any(row[k] != budget for k in ('frames_selected', 'frames_processor', 'frames_represented')) or
                        row['decoder_threads'] != 4 or row['decoder_thread_type'] != 'FRAME' or
                        row['cuda_cache_trimmed'] or not math.isfinite(row['latency_seconds']) or
                        row['latency_seconds'] <= 0 or row['peak_allocated_bytes'] <= 0 or
                        row['peak_reserved_bytes'] < row['peak_allocated_bytes']):
                        raise Blocked('INPUT_INTEGRITY_FAILED', 'Native input integrity, visual budget, memory or timer differs')
                    for key in ('pixel_values_videos', 'video_grid_thw'):
                        if row['input_tensor_hashes'][key] != ref['input_tensor_hashes'][key]:
                            raise Blocked('INPUT_INTEGRITY_FAILED', 'Actual processed pixels/grid differ from CPU freeze')
                    metadata_hash = digest({k: row[k] for k in
                        ('tensor_shapes', 'source_indices', 'actual_timestamps', 'model_timestamps', 'video_grid_thw')})
                    if row['input_metadata_hash'] != metadata_hash:
                        raise Blocked('INPUT_INTEGRITY_FAILED', 'Input metadata hash differs')
                same = all(pair[0][k] == pair[1][k] for k in repeat_keys)
                repeats.append(same and pair[0]['worker_key'] != pair[1]['worker_key'])
                primary = pair[0]
                result.append(int(primary['parsed_answer'] == question['answer']))
                invalid[condition] += int(not primary['valid_answer'])
                token_ranges[condition].append(primary['input_tokens'])
                timings[condition].append(primary['latency_seconds'])
                peaks[condition]['allocated'].append(primary['peak_allocated_bytes'])
                peaks[condition]['reserved'].append(primary['peak_reserved_bytes'])
                if condition == 'base':
                    previous = old[(source, str(question['question_id']))]
                    if primary['input_tensor_hashes'] != previous['input_tensor_hashes']:
                        raise Blocked('INPUT_INTEGRITY_FAILED', 'E1 base input differs from corrected E0')
                    if any(primary[k] != previous[k] for k in ('generated_token_ids', 'decoded_output', 'parsed_answer')):
                        base_discrepancies.append(f'V{vi:02}Q{qi}')
            matrix.append(result)
            videos.append(source)
            anonymous.append(f'V{vi:02}Q{qi}')
    preserved = read_json(run/'private/historical_preservation.json')
    preservation_passed = all((root/name).is_file() and file_hash(root/name) == sha for name, sha in preserved.items())
    completed = [e for e in terminal if e['event'] == 'completed']
    counts = {kind: sum(e['kind'] == kind for e in completed) for kind in ('trial', 'probe', 'warmup')}
    gates = {'complete_288_identities': True, 'native_input_integrity': True,
        'exact_base_retention': True, 'matched_augmented_visual_budgets': True,
        'effective_greedy_generation': True, 'independent_token_decoding_and_scoring': True,
        'fresh_process_repeatability': all(repeats), 'repeatable_cells': sum(repeats),
        'planned_repeatability_cells': 144, 'e0_historical_preservation': preservation_passed,
        'e0_preserved_files': len(preserved), 'preflight_passed': ready['all_input_fit_greedy_repeatability_checks_passed'],
        'completed_calls_by_kind': counts, 'started_calls': len(starts),
        'completed_calls': len(completed), 'failed_calls': sum(e['event'] == 'failed' for e in terminal),
        'interrupted_calls_upper_bound': len(set(started)-set(ended)),
        'e0_base_prediction_discrepancies': len(base_discrepancies),
        'e0_base_discrepancy_anonymous_ids': base_discrepancies}
    gates['all_checks_passed'] = bool(all(repeats) and preservation_passed and counts['trial'] == 288)
    details = {'invalid_answers': invalid,
        'input_token_ranges': {k: [min(v), max(v)] for k, v in token_ranges.items()},
        'descriptive_primary_median_request_seconds': {k: sorted(v)[len(v)//2-1:len(v)//2+1] for k, v in timings.items()},
        'absolute_peak_memory_bytes': {k: {m: max(v) for m, v in value.items()} for k, value in peaks.items()}}
    details['descriptive_primary_median_request_seconds'] = {
        k: sum(v)/len(v) for k, v in details['descriptive_primary_median_request_seconds'].items()}
    return matrix, videos, anonymous, gates, details


def analyze(config, run, root):
    from .e001_analysis import summarize
    matrix, videos, anonymous, gates, details = saved_matrix(config, run, root)
    summary = {**summarize(matrix, videos), **details}
    write_json(run/'summary.json', summary)
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['anonymous_question', *CONDITIONS])
    writer.writerows([name, *row] for name, row in zip(anonymous, matrix))
    atomic_text(run/'correctness.csv', stream.getvalue())
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['metric', 'accuracy', 'ci95_low', 'ci95_high'])
    for metric, value in summary['metrics'].items():
        writer.writerow([metric, value, *summary['descriptive_95pct_cluster_bootstrap'][metric]])
    atomic_text(run/'metrics.csv', stream.getvalue())
    write_json(run/'verification.json', {**gates, 'statistics_independently_verified': False})
    return summary


def verify(config, run, root):
    """Recompute all scores directly from saved real tokens, then independently count endpoints."""
    from .e001_analysis import summarize
    matrix, videos, anonymous, gates, details = saved_matrix(config, run, root)
    saved = read_json(run/'summary.json')
    if saved != {**summarize(matrix, videos), **details}:
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Statistics/bootstrap do not regenerate from real records')
    n = len(matrix)
    # Deliberately use simple counts, separate from NumPy's scientific aggregation.
    base_count = sum(r[0] for r in matrix)
    quarter_counts = [sum(r[a] for r in matrix) for a in range(1, 5)]
    oracle_count = sum(any(r[1:5]) for r in matrix)
    independent = {'base': base_count/n, 'uniform_addition': sum(r[5] for r in matrix)/n,
        'random': sum(quarter_counts)/(4*n), 'best_fixed': max(quarter_counts)/n,
        'oracle': oracle_count/n, 'oracle_stop': sum(any(r[:5]) for r in matrix)/n}
    independent.update({f'quarter{i}': count/n for i, count in enumerate(quarter_counts, 1)})
    for name, other in (('random', 'random'), ('best_fixed', 'best_fixed'),
                        ('uniform', 'uniform_addition'), ('base', 'base')):
        independent['oracle_minus_'+name] = independent['oracle']-independent[other]
    if any(not math.isclose(saved['metrics'][k], value, abs_tol=1e-12) for k, value in independent.items()):
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Independent endpoint counts differ')
    with (run/'correctness.csv').open(encoding='utf-8', newline='') as handle:
        published = list(csv.reader(handle))
    if published != [['anonymous_question', *CONDITIONS]] + [[name, *map(str, row)] for name, row in zip(anonymous, matrix)]:
        raise Blocked('INPUT_INTEGRITY_FAILED', 'Published anonymous matrix differs')
    gates.update(statistics_independently_verified=True, at=utc_now(),
        summary_sha256=file_hash(run/'summary.json'), matrix_sha256=file_hash(run/'correctness.csv'))
    verdict = ('Promising development headroom; larger independent evaluation is justified.'
        if saved['exploratory_accuracy_rule_passed'] else
        'Useful headroom was not demonstrated for this configuration and action menu.')
    if not gates['all_checks_passed']:
        verdict = 'Inconclusive: strict input, preservation or repeatability checks remain unresolved.'
    write_json(run/'verification.json', gates)
    write_json(run/'status.json', {'status': 'COMPLETE' if gates['all_checks_passed'] else 'REPEATABILITY_UNRESOLVED',
        'verdict': verdict, 'completed_collection_requests': 288, 'planned_collection_requests': 288,
        'advancement_rule_passed': bool(gates['all_checks_passed'] and saved['exploratory_accuracy_rule_passed']),
        'e0_verdict_unchanged': 'MEASUREMENT_UNSTABLE'})
    print(verdict, flush=True)
    return gates
