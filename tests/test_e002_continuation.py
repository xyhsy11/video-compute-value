"""Synthetic continuation integrity fixtures; never scientific observations."""
import pytest

from video_compute_value.common import Blocked, read_jsonl
from video_compute_value.e002v2_data import JournalRangeReader, math_ceil_traffic, source_url_id


class FakeBudget:
    def __init__(self, run):
        self.run, self.value = run, {'video_transfer_bytes': 10}
    def allow_video_transfer(self, size):
        assert size > 0
    def check(self, **kwargs):
        pass
    def save(self):
        pass
    def charge_transfer(self, size, video):
        assert video
        self.value['video_transfer_bytes'] += size


def test_range_charges_payload_and_http_allowance_and_preserves_start(tmp_path, monkeypatch):
    class Response:
        status, headers = 206, {'Content-Range': 'bytes 0-3/4'}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, size):
            return b'abcd'
    monkeypatch.setattr('video_compute_value.e002v2_data.request', lambda *a, **kw: Response())
    budget = FakeBudget(tmp_path)
    with JournalRangeReader('https://example.org/file', 4, budget) as stream:
        assert stream.read(4) == b'abcd'
    assert budget.value['video_transfer_bytes'] == 10+4+4096
    assert 'pending_video_transfer_bytes' not in budget.value
    assert [r['event'] for r in read_jsonl(tmp_path/'private/transfer_events.jsonl')] == ['started', 'completed']


def test_failed_or_nonrange_response_is_conservatively_charged(tmp_path, monkeypatch):
    class Response:
        status, headers = 200, {}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr('video_compute_value.e002v2_data.request', lambda *a, **kw: Response())
    budget = FakeBudget(tmp_path)
    with pytest.raises(Blocked, match='exact public'):
        JournalRangeReader('https://example.org/file', 4, budget).read(4)
    assert budget.value['interrupted_transfer_upper_bound_bytes'] == 4100
    assert budget.value['video_transfer_bytes'] == 10


def test_provenance_identity_and_whole_window_forecast():
    assert source_url_id('https://www.youtube.com/watch?v=abc') == 'abc'
    assert source_url_id('https://youtu.be/abc') == 'abc'
    assert source_url_id('https://example.org/abc') is None
    assert math_ceil_traffic(1000, 2) == 1500+2*1024**2


def test_immutable_records_skip_completed_and_reject_drift(tmp_path):
    from video_compute_value.common import digest, write_json
    from video_compute_value.e002v2_records import may_start, records, save_record
    identity = {'source_id': 'synthetic', 'question_id': '0', 'condition': 'base_plain',
                'repetition': 1, 'fingerprint': 'frozen'}
    identity['identity'] = digest(identity)
    assert may_start(tmp_path, identity, {})
    save_record(tmp_path/'private/records', identity, {'generated_token_ids': [1]})
    done = records(tmp_path, [identity])
    assert not may_start(tmp_path, identity, done)
    with pytest.raises(Blocked, match='overwrite'):
        save_record(tmp_path/'private/records', identity, {})
    with pytest.raises(Blocked, match='fingerprint'):
        records(tmp_path, [{**identity, 'fingerprint': 'changed'}])
    row = next(iter(done.values()))
    row['generated_token_ids'] = [2]
    write_json(tmp_path/f"private/records/{identity['identity']}.json", row)
    with pytest.raises(Blocked, match='seal'):
        records(tmp_path, [identity])


def test_two_started_attempts_cannot_trigger_a_third_paid_call(tmp_path):
    from video_compute_value.common import append_json
    from video_compute_value.e002v2_records import may_start
    identity = {'identity': 'synthetic'}
    for i in range(2):
        append_json(tmp_path/'private/calls.jsonl', {'event': 'started', 'identity': 'synthetic', 'call_id': str(i)})
    with pytest.raises(Blocked, match='Two-start'):
        may_start(tmp_path, identity, {})


def test_sealed_fold_is_independent_of_heldout_labels_and_other_files(monkeypatch):
    import copy
    import numpy as np
    from video_compute_value.e002v2_analysis import fit_payload
    rng = np.random.default_rng(22)
    payload = {'fold': 0, 'train': {
        **{name: rng.normal(size=(24, size)).tolist() for name, size in [('V', 19), ('Q', 66), ('VQ', 341)]},
        'gains': rng.integers(-1, 2, (24, 4)).tolist(), 'quarters': rng.integers(0, 2, (24, 4)).tolist(),
        'groups': [f'synthetic-{i//3}' for i in range(24)]},
        'test': {name: rng.normal(size=(3, size)).tolist() for name, size in [('V', 19), ('Q', 66), ('VQ', 341)]}}
    def deny(*args, **kwargs):
        raise AssertionError('No candidate or label file access is permitted during policy fitting')
    monkeypatch.setattr('video_compute_value.e002v2_analysis.read_json', deny)
    first = fit_payload(payload)
    assert first == fit_payload(copy.deepcopy(payload))
    changed = copy.deepcopy(payload)
    changed['test']['heldout_candidate_outputs'] = [1, 2, 3]
    with pytest.raises(ValueError, match='only allowed'):
        fit_payload(changed)
    assert first['fixed_quarter'] == int(np.asarray(payload['train']['quarters']).mean(0).argmax())


def test_direct16_hashing_does_not_assume_base_retention():
    import numpy as np
    from video_compute_value.e002v2_collect import finish_hashes
    details = {'source_indices': [2, 4], '_e1_base_indices': [1, 3],
               '_e1_pixels_for_hashing': np.zeros((2, 3, 4, 3), dtype=np.uint8)}
    finish_hashes(details)
    assert details['retained_base_rgb_hashes'] == {}
    assert set(details['processed_rgb_hashes']) == {'2', '4'}


def test_interruption_after_atomic_completion_recovers_receipt_without_paid_retry(tmp_path):
    from video_compute_value.common import append_json, digest
    from video_compute_value.e002v2_records import recover_atomic_completions, records, save_record, may_start
    identity = {'source_id': 'synthetic', 'question_id': '0', 'condition': 'base_plain',
                'repetition': 1, 'fingerprint': 'frozen'}
    identity['identity'] = digest(identity)
    append_json(tmp_path/'private/calls.jsonl', {'event': 'started', 'identity': identity['identity'],
                                              'worker_key': 'w', 'call_id': 'call'})
    save_record(tmp_path/'private/records', identity, {'call_id': 'call', 'model_invocation_ordinal': 1})
    done = records(tmp_path, [identity])
    assert recover_atomic_completions(tmp_path, done) == 1
    assert recover_atomic_completions(tmp_path, done) == 0
    assert not may_start(tmp_path, identity, done)


def test_inference_validation_never_opens_labels_or_counterfactual_files(tmp_path, monkeypatch):
    from video_compute_value.common import digest
    from video_compute_value.e002v2_collect import validate
    inference, splits, actions, practical = {'videos': []}, {}, {}, {}
    content = {'source_files': {'synthetic.py': 'sha'}, 'inference_hash': digest(inference),
               'split_hash': digest(splits), 'action_hash': digest(actions), 'practical_plan_hash': digest(practical)}
    frozen = {**content, 'fingerprint': digest(content)}
    files = {'frozen.json': frozen, 'inference_manifest.json': inference, 'splits.json': splits,
             'actions.json': actions, 'practical_plan.json': practical, 'call_manifest.json': []}
    def allowed_read(path):
        assert path.name in files, 'Policy worker attempted to open labels or candidate outcomes'
        return files[path.name]
    monkeypatch.setattr('video_compute_value.e002v2_collect.read_json', allowed_read)
    monkeypatch.setattr('video_compute_value.e002v2_collect.scientific_sources', lambda root: content['source_files'])
    monkeypatch.setattr('video_compute_value.e002v2_collect.call_manifest', lambda *args: [])
    assert validate(tmp_path, tmp_path, inference_only=True) == frozen
    with pytest.raises(AssertionError, match='labels'):
        validate(tmp_path, tmp_path)


def test_practical_feature_boundary_and_confidence_strict_inequality():
    import numpy as np
    from video_compute_value.e002v2_practical import decide_features
    specification = {'family': 'C', 'quarter': 2, 'threshold': .5}
    assert decide_features(np.array([.5, .1, .2]), specification) == 'stop'
    assert decide_features(np.array([.49, .1, .2]), specification) == 'quarter3'
    assert decide_features(np.array([.9, .1, .2]), {**specification, 'threshold': 'always_acquire'}) == 'quarter3'
    with pytest.raises((ValueError, TypeError)):
        decide_features({'answer': 'synthetic'}, specification)
