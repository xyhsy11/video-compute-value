"""Operational acceptance tests; numeric fixtures are not model predictions."""
from pathlib import Path

import pytest

from video_compute_value.common import read_json
from video_compute_value.e000_acceptance import analyze, select_policy

ROOT = Path(__file__).resolve().parents[1]


def conditions():
    return [{'arm': a, 'budget': b, 'complete': True, 'primary_requests': 24, 'S': value}
            for a, b, value in [('A', 8, .6336), ('A', 64, .4031), ('B', 8, .1848), ('B', 64, .2253)]]


def test_25pct_accepts_b_while_original_20pct_does_not():
    data = conditions()
    assert select_policy(data, .2, integrity=True) is None
    assert select_policy(data, .25, integrity=True) == 'B'
    assert select_policy(data, .25, integrity=False) is None


def test_threshold_is_inclusive_and_both_budgets_are_required():
    data = conditions()
    data[-1]['S'] = .25
    assert select_policy(data, .25, integrity=True) == 'B'
    data[-1]['S'] = .250001
    assert select_policy(data, .25, integrity=True) is None
    data[-1]['S'] = .2253
    data[-1]['complete'] = False
    assert select_policy(data, .25, integrity=True) is None
    assert select_policy(data[:-1], .25, integrity=True) is None


@pytest.mark.parametrize('threshold', [True, 0, 1, float('nan')])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        select_policy(conditions(), threshold, integrity=True)


def test_saved_real_primary_records_reproduce_amended_decision():
    if not (ROOT/'results/e000_20261104_045654_runtime_recovery_v7/private/records').exists():
        pytest.skip('Restricted local saved records are not included in Git')
    _, summary, proof = analyze(ROOT)
    assert summary['selected_policy'] == 'B' and summary['good'] is True
    assert summary['new_model_calls'] == 0 and proof['primary_timings_recomputed'] == 96
    original = read_json(ROOT/'results/e000_20261104_045654_runtime_recovery_v7/summary.json')
    assert original['selected_policy'] is None
    assert original['status'] == 'STABILIZATION_NOT_DEMONSTRATED'
