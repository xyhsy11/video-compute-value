"""Synthetic fixtures only: none of these predictions are experimental evidence."""
from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

from video_compute_value.common import Blocked, digest, write_json
from video_compute_value.e001 import (expected, identity, make_order, records, resource_ledger,
                                      sources, validate)
from video_compute_value.e001_analysis import bootstrap, estimates, summarize
from video_compute_value.e001_input import CONDITIONS, actions, addition, inference_video
from video_compute_value.protocol import parse_answer


def video_fixture():
    from video_compute_value.video import uniform_indices
    times = [str(Fraction(i, 10)) for i in range(160)]
    return {'source_id': 'synthetic-source', 'timestamps': times, 'duration': '16',
            'selections': {'8': {'indices': uniform_indices(times, '16', 8)}},
            'questions': [{'question_id': str(i)} for i in range(3)]}


def test_exact_retention_quarters_displacements_and_original_timestamps():
    video = video_fixture()
    menu = actions(video)
    base = set(menu['base']['indices'])
    for quarter, name in enumerate(CONDITIONS[1:5]):
        action = menu[name]
        assert len(set(action['indices'])) == 16 and set(action['indices']) >= base
        assert action['indices'] == sorted(action['indices'])
        assert not base & set(action['supplemental_indices'])
        assert all(quarter*4 <= Fraction(video['timestamps'][i]) < (quarter+1)*4
                   for i in action['supplemental_indices'])
        assert action['timestamps'] == [video['timestamps'][i] for i in action['indices']]
    assert any(Fraction(d['exclusion_displacement_seconds']) for d in menu['uniform_addition']['displacements'])
    # The explicitly nested control must not be silently substituted by Uniform-16.
    from video_compute_value.video import uniform_indices
    assert menu['uniform_addition']['indices'] != uniform_indices(video['timestamps'], '16', 16)


def test_earlier_tie_exclusion_and_half_open_boundaries():
    times = [str(Fraction(i, 2)) for i in range(33)]
    selected, details = addition(times, '16', [], 0)
    assert selected == list(range(8))  # Quarter-bin centers are halfway between source timestamps.
    selected, details = addition(times, '16', [2])
    assert selected[0] == 1 and details[0]['exclusion_displacement_seconds'] == '-1/2'
    # There are only seven available quarter frames after excluding frame 0.
    with pytest.raises(Blocked):
        addition([str(i) for i in range(32)], '32', [0], 0)


def test_ineligible_quarter_fails_instead_of_duplication():
    with pytest.raises(Blocked, match='distinct'):
        addition([str(i) for i in range(8)], '8', [0], 0)


def test_final_quarter_can_include_real_endpoint_without_resetting_time():
    times = [str(Fraction(i, 2)) for i in range(33)]
    selected, _ = addition(times, '16', [31], 3)
    assert 32 in selected and all(Fraction(times[i]) >= 12 for i in selected)


def test_order_is_complete_seeded_and_outcome_independent():
    cohort = {'videos': [{**video_fixture(), 'source_id': str(i)} for i in range(8)]}
    a = make_order(cohort)
    assert a == make_order(deepcopy(cohort))
    assert a[0]['requests'] != a[1]['requests']
    assert all(len(b['requests']) == len(set(map(tuple, b['requests']))) == 144 for b in a)
    cohort['videos'][0]['questions'][0]['answer'] = 'synthetic changed label'
    assert a == make_order(cohort)


def matrix_fixture():
    return np.array([[0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]])


def test_hand_worked_oracles_forced_acquisition_harm_and_all_denominators():
    x = matrix_fixture()
    s = summarize(x, ['a', 'a', 'b', 'b'])
    m = s['metrics']
    assert m['base'] == m['uniform_addition'] == m['oracle'] == .5
    assert m['random'] == 5/16 and m['best_fixed'] == .5 and m['oracle_stop'] == .75
    assert m['oracle_minus_random'] == 3/16
    assert s['rescued_base_wrong_questions'] == s['base_correct_harmed_by_every_clip'] == 1
    assert s['mixed_success_questions'] == 1 and s['average_action_induced_harm'] == .25
    assert s['questions'] == 4 and s['random_action_pair_denominator'] == 16
    assert not s['exploratory_accuracy_rule_passed']


@pytest.mark.parametrize('text', ['', 'A and B', 'The answer is A'])
def test_invalid_completed_answers_are_zero(text):
    assert parse_answer(text) is None
    assert int(parse_answer(text) == 'A') == 0


def test_forced_oracle_can_harm_and_stop_is_nonnegative():
    m = estimates([[1, 0, 0, 0, 0, 1]])
    assert m['oracle_minus_base'] == -1 and m['oracle_stop'] == 1


def test_bootstrap_reselects_best_fixed_and_preserves_clusters():
    x = [[0, 1, 0, 0, 0, 0]]*3 + [[0, 0, 1, 0, 0, 0]]*3
    intervals = bootstrap(x, ['a']*3+['b']*3)
    assert intervals == bootstrap(x, ['a']*3+['b']*3)
    assert intervals['oracle_minus_random'] == [.75, .75]
    assert intervals['oracle_minus_best_fixed'] == [0., .5]


@pytest.mark.parametrize('x', [[], [[0, 1]], [[0, 0, 0, 0, 0, 2]], [[float('nan')]*6]])
def test_incomplete_or_invalid_matrices_are_not_scored(x):
    with pytest.raises(ValueError):
        estimates(x)


def test_completed_identity_seals_and_resume_drift(tmp_path):
    cohort, frozen = {'videos': [video_fixture()]}, {'fingerprint': 'synthetic-fingerprint'}
    run = tmp_path/'run'
    item = next(iter(expected(run, frozen, cohort).values()))
    row = {**item, 'status': 'completed'}
    row['record_hash'] = digest(row)
    path = run/f"private/records/{row['trial_id']}.json"
    write_json(path, row)
    assert len(records(run, frozen, cohort)) == 1
    assert item != identity(run, 'changed-fingerprint', item['source_id'], item['question_id'], item['condition'], 1)
    row['condition'] = 'changed-condition'
    write_json(path, row)
    with pytest.raises(Blocked):
        records(run, frozen, cohort)


def test_resource_inheritance_cannot_reset_or_reopen_older_ledger(tmp_path):
    run = tmp_path/'run'
    grant = {'human_reply': 'Authorize +2 hours; cumulative cap 27,600 seconds',
             'authorized_limit_seconds': 27600, 'resource_only': True}
    write_json(run/'runtime_resource_grant.json', grant)
    write_json(run/'private/resource_ledger.json', {'active_seconds': 0})
    with pytest.raises(Blocked, match='inherit'):
        resource_ledger(run, tmp_path)
    write_json(run/'private/resource_ledger.json', {'active_seconds': 20400})
    write_json(tmp_path/'newer.json', {'active_seconds': 21000})
    write_json(tmp_path/'.local/e001_latest_ledger.json', {'path': 'newer.json'})
    with pytest.raises(Blocked, match='newer'):
        resource_ledger(run, tmp_path)


def test_reports_do_not_drift_scientific_sources_but_code_does(tmp_path):
    from video_compute_value.e001 import SOURCES
    for name in SOURCES:
        p = tmp_path/'src/video_compute_value'/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('synthetic source', encoding='utf-8')
    for name in ('configs/e001.yaml', 'protocols/e001_additional_clip_headroom_v1.md'):
        p = tmp_path/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('synthetic source', encoding='utf-8')
    initial = sources(tmp_path)
    write_json(tmp_path/'results/report.json', {'changing': 'generated report'})
    assert sources(tmp_path) == initial
    (tmp_path/'src/video_compute_value/e001.py').write_text('changed', encoding='utf-8')
    assert sources(tmp_path) != initial


def test_missing_requests_are_measurements_missing_not_wrong(tmp_path, monkeypatch):
    from video_compute_value import e001, e001_verify
    run = tmp_path/'run'
    monkeypatch.setattr(e001, 'validate', lambda *a, **k: {'fingerprint': 'synthetic'})
    write_json(run/'private/cohort_manifest.json', {'videos': [video_fixture()]})
    for name in ('action_manifest', 'cpu_preflight', 'collection_ready'):
        write_json(run/f'private/{name}.json', {})
    with pytest.raises(Blocked, match='0/288'):
        e001_verify.saved_matrix({}, run, tmp_path)


def test_inference_whitelist_excludes_labels_and_questions():
    # A complete synthetic decoder metadata object passes only the existing whitelist.
    allowed = ('path', 'file_size', 'selections', 'timestamps', 'duration', 'frame_count', 'height', 'width',
               'target_height', 'target_width', 'fps', 'content_height', 'content_width', 'pad_top', 'pad_left')
    video = {k: None for k in allowed}
    video.update(answer='synthetic-private-label', questions=['synthetic-private-text'])
    value = inference_video(video, {'indices': [0]})
    assert 'answer' not in value and 'questions' not in value


def test_validate_rejects_configuration_drift_before_accessing_gpu(tmp_path):
    frozen = {'config_input_hash': digest({'a': 1})}
    frozen['fingerprint'] = digest(frozen)
    write_json(tmp_path/'private/frozen.json', frozen)
    with pytest.raises(Blocked, match='configuration'):
        validate({'a': 2}, tmp_path, tmp_path)
