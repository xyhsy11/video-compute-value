"""Synthetic CPU fixtures only; these are never experimental observations."""
import importlib.util
from pathlib import Path
import sys

import pytest

from video_compute_value.common import Blocked, read_json, read_jsonl, write_json

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/stabilize_e000_timing_v5.py'
spec = importlib.util.spec_from_file_location('timing_v5', SCRIPT)
v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v5)


def cohort():
    return {'videos': [{'source_id': f'S{i}', 'questions': [{'question_id': f'Q{i}{j}'} for j in range(3)]}
                       for i in range(8)]}


def frozen():
    return {'fingerprint': 'fixture', 'adapter': {'two_warmups': [['S0', 'Q00'], ['S0', 'Q01']],
                                                'profile_question': ['S0', 'Q00']}}


def test_diagnostic_complete_matched_schedule():
    schedule = v5.diagnostic_schedule(cohort())
    assert schedule == v5.diagnostic_schedule(cohort())
    assert len(schedule) == 9
    calls = [c for s in schedule for c in v5.expected_calls(frozen(), s)]
    assert len(calls) == 120
    assert len({c['call_id'] for c in calls}) == 120
    assert sum(c['kind'] == 'primary' for c in calls) == 96
    assert sum(c['kind'] == 'warmup' for c in calls) == 24
    resident = [s for s in schedule if s['arm'] == 'B']
    assert len(resident) == 3 and {s['repetition'] for s in resident} == {1, 2, 3}
    assert {tuple(b['budget'] for b in s['blocks']) for s in resident} == {(8, 64), (64, 8)}
    for rep in (1, 2, 3):
        blocks = [b for s in schedule if s['repetition'] == rep for b in s['blocks']]
        assert all(b['pairs'] == blocks[0]['pairs'] for b in blocks)
    probes = [c for s in v5.probe_schedule(frozen()) for c in v5.expected_calls(frozen(), s)]
    assert len(probes) == 12 and sum(c['kind'] == 'profile' for c in probes) == 4


@pytest.mark.parametrize('arm,workers,probe_calls', [('A', 16, 25), ('B', 4, 50)])
def test_confirmation_counts_lifecycle_and_transitions(arm, workers, probe_calls):
    schedule = v5.confirmation_schedule(cohort(), arm)
    assert len(schedule) == workers
    calls = [c for s in schedule for c in v5.expected_calls(frozen(), s)]
    primary = [c for c in calls if c['kind'] == 'primary']
    assert len(primary) == 384 and sum(c['budget'] == 0 for c in primary) == 24
    assert len(calls) == 416 and sum(c['kind'] == 'warmup' for c in calls) == 32
    assert len({c['call_id'] for c in calls}) == 416
    if arm == 'B':
        assert any(s['blocks'][-1]['budget'] == 64 for s in schedule[:-1])
        assert any(s['blocks'][-1]['budget'] != 64 for s in schedule[:-1])
    probes = [c for s in v5.probe_schedule(frozen(), True, arm) for c in v5.expected_calls(frozen(), s)]
    assert len(probes) == probe_calls


def test_relative_range_and_question_first_aggregation():
    assert v5.relative_range([8, 10, 12]) == .4
    rows = [{'kind': 'primary', 'arm': 'A', 'budget': 8, 'source_id': f'S{i}', 'question_id': 'Q',
             'repetition': j + 1, 'latency_seconds': t, 'cpu_prepare_host_seconds': t / 2}
            for i, times in enumerate(([8, 10, 12], [1, 1, 1])) for j, t in enumerate(times)]
    result = v5.summarize(rows, ('A',), (8,), 2)
    assert result['conditions'][0]['S'] == .2
    assert result['conditions'][0]['median_request_seconds'] == 5.5
    assert v5.summarize(rows[:-1], ('A',), (8,), 2)['conditions'][0]['S'] is None
    with pytest.raises(Blocked):
        v5.relative_range([1, 2])


@pytest.mark.parametrize('a,b,want', [(.19, .1, 'A'), (.21, .2, 'B'), (.21, .21, None), (.2, .2, 'A')])
def test_prospective_selection(a, b, want):
    data = {'conditions': [{'arm': arm, 'budget': budget, 'complete': True, 'S': s}
                            for arm, s in (('A', a), ('B', b)) for budget in (8, 64)]}
    assert v5.select_policy(data, True) == want
    assert v5.select_policy(data, False) is None


def test_private_descendant_discovery_and_reserve(tmp_path):
    parent = {'active_seconds': 23333, 'sessions': [], 'external_events': []}
    write_json(tmp_path / 'results/e1/private/resource_ledger.json', parent)
    write_json(tmp_path / '.local/e001_latest_ledger.json', {'path': 'results/e1/private/resource_ledger.json'})
    assert v5.latest_ledger(tmp_path)[1]['active_seconds'] == 23333
    child = {**parent, 'active_seconds': 24000}
    write_json(tmp_path / 'results/new/private/resource_ledger.json', child)
    assert v5.latest_ledger(tmp_path)[1]['active_seconds'] == 24000
    config = {'safety_factor': 1.5, 'interruption_contingency_seconds': 120, 'closeout_reserve_seconds': 600}
    assert not v5.admission(1000, 200, config)['admitted']
    assert v5.admission(1020, 200, config)['admitted']
    assert v5.admission(1000, 200, config)['additional_cumulative_allowance_needed_seconds'] == 20


def test_immutable_completed_wrong_invalid_slow_and_drift(tmp_path):
    row = {'fingerprint': 'frozen', 'call_id': 'identity', 'correct': False, 'parsed_answer': None, 'latency_seconds': 90}
    path = tmp_path / 'private/records/identity.json'
    v5.seal_record(path, row)
    assert v5.load_records(tmp_path, 'frozen') == [row]
    with pytest.raises(Blocked):
        v5.seal_record(path, {**row, 'correct': True})
    with pytest.raises(Blocked):
        v5.load_records(tmp_path, 'changed')
    bad = read_json(path)
    bad['correct'] = True
    write_json(path, bad)
    with pytest.raises(Blocked):
        v5.load_records(tmp_path, 'frozen')


def test_fake_worker_failure_and_timeout_are_preserved_reaped(tmp_path):
    for args, timeout in ((['-c', 'raise SystemExit(7)'], 10), (['-c', 'import time; time.sleep(30)'], .1)):
        with pytest.raises(Blocked):
            v5.launch_owned([sys.executable, *args], tmp_path / 'log.txt', timeout, tmp_path / 'failures.jsonl', 'fixture')
    failures = read_jsonl(tmp_path / 'failures.jsonl')
    assert len(failures) == 2 and all(f['reaped'] for f in failures)
    assert failures[-1]['unknown_invocations_upper_bound'] == 1


@pytest.mark.parametrize('value', [{'generated_token_ids': [32]}, {'question': 'restricted'},
                                  {'metric': 'C:\\Users\\person\\raw.json'}, {'path': '/private/raw'},
                                  {'metric': 'bearer secret'}])
def test_public_export_rejects_restricted_fields_and_paths(value):
    report_spec = importlib.util.spec_from_file_location('timing_report', SCRIPT.with_name('report_e000_timing_v5.py'))
    report = importlib.util.module_from_spec(report_spec)
    report_spec.loader.exec_module(report)
    with pytest.raises(ValueError):
        report.public_safe(value)
