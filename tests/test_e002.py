"""SYNTHETIC CPU fixtures only. No fixture is an E2 prediction or timing result."""
from dataclasses import replace
from fractions import Fraction
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from video_compute_value.e002_analysis import practical_comparison, predictive_summary, timing_stability
from video_compute_value.e002_features import (AllowedObservation, confidence_summary, extract,
                                               query_features, visual_features)
from video_compute_value.e002_input import call_manifest, menu, split_manifest
from video_compute_value.e002_model import (acquire_mask, fit_ridge, forced_actions, group_folds,
                                           seal_fold_decisions, select_threshold, train_fold)
from video_compute_value.e002_runtime import execute_pipeline


def allowed():
    pixels = np.zeros((8, 192, 320, 3), dtype=np.uint8)
    for i in range(8):
        pixels[i, 10:182, 20:300] = i*30
    return AllowedObservation(pixels, tuple(str(i*2+1) for i in range(8)), '16', 160,
                              16/9, (10, 20, 172, 280), 'What happens next?')


def video(index=0):
    from video_compute_value.video import uniform_indices
    times = [str(Fraction(i, 10)) for i in range(160)]
    return {'source_id': f'synthetic-{index}', 'timestamps': times, 'duration': '16',
            'selections': {'8': {'indices': uniform_indices(times, '16', 8)}},
            'questions': [{'question_id': str(i)} for i in range(3)]}


@pytest.mark.parametrize('prohibited', ['answer', 'options', 'source_id', 'path', 'filename',
                                      'category', 'question_id', 'candidate_outputs', 'candidate_costs'])
def test_policy_interface_rejects_prohibited_fields(prohibited):
    values = {name: getattr(allowed(), name) for name in AllowedObservation.__slots__}
    values[prohibited] = 'SYNTHETIC FORBIDDEN'
    with pytest.raises(ValueError, match='exactly'):
        AllowedObservation.from_mapping(values)


def test_base_only_features_dimensions_and_padding_removal():
    obs = allowed()
    first = extract(obs)
    assert {k: v.shape for k, v in first.items()} == {'V': (19,), 'Q': (66,), 'VQ': (341,)}
    modified = obs.base_rgb.copy()
    modified[:, :10] = 255
    modified[:, :, :20] = 255
    other = extract(replace(obs, base_rgb=modified))
    assert all(np.array_equal(first[k], other[k]) for k in first)
    assert np.allclose(first['V'][:2], [0, 0], rtol=0, atol=1e-15)
    assert first['V'][2] == pytest.approx(30/255)
    assert first['V'][3] == 1
    with pytest.raises(ValueError, match='two base'):
        visual_features(replace(obs, timestamps=tuple(str(i) for i in range(8))))
    with pytest.raises(TypeError):
        extract({'answer': 'SYNTHETIC'})


def test_hash_features_cross_process_and_unicode_normalization():
    assert np.array_equal(query_features('ＡＢＣ'), query_features('abc'))
    assert np.isfinite(query_features('')).all()
    code = "import json; from video_compute_value.e002_features import query_features; print(json.dumps(query_features('Which scene?').tolist()))"
    out = subprocess.check_output([sys.executable, '-c', code], text=True)
    assert json.loads(out) == query_features('Which scene?').tolist()
    a, b = extract(allowed()), extract(replace(allowed(), question_text='Another question'))
    assert np.array_equal(a['V'], b['V'])
    assert not np.array_equal(a['Q'], b['Q'])
    assert not np.array_equal(a['VQ'][85:], b['VQ'][85:])


def test_group_folds_manifest_and_legacy_not_implicitly_reused():
    videos = [video(i) for i in range(40)]
    splits = split_manifest(videos)
    assert list(splits['outer'].values()).count(0) == 8
    for fold in range(5):
        outer_test = {g for g, f in splits['outer'].items() if f == fold}
        assert not outer_test & set(splits['inner'][str(fold)])
        assert len(splits['inner'][str(fold)]) == 32
    calls = call_manifest(videos, splits, 'SYNTHETIC')
    assert len(calls) == 1080
    assert sum(c['repetition'] == 1 for c in calls) == 960
    assert sum(c['repetition'] == 2 for c in calls) == 120
    assert calls == call_manifest(videos, splits, 'SYNTHETIC')
    with pytest.raises(Exception, match='40'):
        split_manifest(videos[:8])
    for value in videos:
        m = menu(value)
        assert m['base_plain'] == m['base_observed']
        for name in ('quarter1', 'quarter2', 'quarter3', 'quarter4', 'uniform_addition'):
            assert set(m['base_plain']['indices']) <= set(m[name]['indices'])
            assert len(set(m[name]['indices'])) == 16


def test_ridge_matrix_convention_intercept_and_training_only_scaling():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(20, 8)), rng.integers(-1, 2, size=(20, 4))
    model = fit_ridge(x, y, 10.)
    z = (x-x.mean(0))/x.std(0)
    expected = np.linalg.solve(z.T@z+len(x)*10*np.eye(8), z.T@(y-y.mean(0)))
    assert np.allclose(model.weights, expected, atol=1e-12)
    assert np.array_equal(model.mean, x.mean(0))
    assert np.array_equal(model.intercept, y.mean(0))
    heldout = np.full((2, 8), 1000000.)
    mean_before = model.mean.copy()
    model.predict(heldout)
    assert np.array_equal(mean_before, model.mean)


def test_inner_ties_prefer_regularization_and_heldout_labels_cannot_enter():
    x, y = np.zeros((24, 3)), np.zeros((24, 4))
    groups = np.repeat([f'train{i}' for i in range(8)], 3)
    model, _, receipt = train_fold(x, y, groups, salt='test')
    assert model.alpha == receipt['alpha'] == 100
    _, _, first = seal_fold_decisions(x, y, groups, np.ones((3, 3)), fold=0)
    # The API has no held-out label/candidate argument, including via arbitrary kwargs.
    with pytest.raises(TypeError):
        seal_fold_decisions(x, y, groups, np.ones((3, 3)), fold=0, test_labels=np.ones(3))
    _, _, second = seal_fold_decisions(x, y, groups, np.ones((3, 3)), fold=0)
    assert first == second
    assert set(group_folds(groups)) == set(range(5))


def test_separate_question_outputs_really_change_action_ranking():
    x = np.array([[-1.], [1.]]*20)
    y = np.array([[1, 0, -1, 0], [-1, 0, 1, 0]]*20)
    model = fit_ridge(x, y, 1.)
    assert forced_actions(model.predict([[-1.], [1.]])).tolist() == [0, 2]
    assert forced_actions(np.zeros((2, 4))).tolist() == [0, 0]


def test_threshold_strict_tie_all_stop_acquire_harms_and_infeasibility():
    predictions = np.array([[.1, 0, 0, 0], [.1, 0, 0, 0]])
    assert not acquire_mask(predictions, .1).any()
    assert acquire_mask(predictions, 'always_acquire').all()
    assert not acquire_mask(predictions, 'always_stop').any()
    # Invalid/correctness-zero base remains zero on stopping; harmful acquisitions retained.
    base, quarters = [0, 1], [[1, 0, 0, 0], [0, 1, 1, 1]]
    result = select_threshold(predictions, base, quarters, 2, .1, 2, 1, qualified=True)
    assert result['status'] == 'NO_FEASIBLE_POLICY_AT_REFERENCE_BUDGET'
    with pytest.raises(ValueError, match='MEASUREMENT'):
        select_threshold(predictions, base, quarters, 0, 0, 0, 1, qualified=False)


def test_confidence_is_real_distribution_not_letter_or_length():
    top, gap, entropy = confidence_summary([0., 0., 0., 0.])
    assert top == .25 and gap == 0 and entropy == 1
    assert confidence_summary([0., -np.inf])[0] == 1
    with pytest.raises(ValueError):
        confidence_summary([np.nan, 1])


def test_missing_evidence_and_invalid_timing_block_scientific_claims():
    with pytest.raises(ValueError, match='Complete'):
        predictive_summary([0], [[0, 0, 0, 0]], [0], [0], {}, ['one'], [0], complete=False)
    with pytest.raises(ValueError, match='Three real'):
        timing_stability([[1, 2]])
    assert not timing_stability([[1, 2, 3]])['passed']
    assert practical_comparison([], [], [], [], [], gates={'complete': False})['practical_value'].endswith('MEASUREMENT')


def test_real_pipeline_primitives_seal_before_candidate_and_direct_has_one_call():
    events = []
    class FakeAdapter:
        model_call_count = 0
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
        def prepare(self, question, value, budget):
            if budget == 16 and not direct_mode[0]:
                assert 'sealed' in events
            events.append(f'prepare{budget}')
            return {}, {'_e1_pixels_for_hashing': allowed().base_rgb, 'actual_timestamps': allowed().timestamps}
        def transfer(self, inputs):
            return inputs
        def generate(self, inputs):
            self.model_call_count += 1
            return {'C': [.5, .1, .3], 'parsed_answer': None}
    value = video()
    value.update(width=16, height=9, frame_count=160, pad_top=10, pad_left=20,
                 content_height=172, content_width=280, path='synthetic-unused', file_size=0,
                 target_height=192, target_width=320, fps='10')
    question = SimpleNamespace(question='Synthetic question')
    direct_mode = [False]
    adapter = FakeAdapter()
    result = execute_pipeline(adapter, question, value, family='V',
                              decide=lambda x: 'quarter1', seal=lambda x: events.append('sealed'))
    assert result['actual_model_calls'] == 2 and result['total_seconds'] >= sum(result['components_seconds'].values())
    direct_mode[0] = True
    result = execute_pipeline(adapter, question, value, direct='direct16')
    assert result['actual_model_calls'] == 1
    assert result['components_seconds']['followup'] == 0


def test_same_call_score_adapter_preserves_greedy_controls_and_has_one_invocation():
    import torch
    from video_compute_value.e002_runtime import E2Adapter
    from video_compute_value.common import digest
    seen = []
    class FakeModel:
        def generate(self, **kwargs):
            seen.append(kwargs)
            assert kwargs['use_model_defaults'] is False and kwargs['output_scores'] is True
            assert kwargs['do_sample'] is False
            return SimpleNamespace(sequences=torch.tensor([[1, 2, 65]]), scores=(torch.zeros((1, 4)),))
    adapter = E2Adapter.__new__(E2Adapter)
    adapter.capture_scores, adapter._prepared_budget, adapter.model_call_count = True, 8, 0
    adapter.torch, adapter.model, adapter.generation = torch, FakeModel(), object()
    adapter.generation_policy = {'call_overrides': {'use_model_defaults': False, 'do_sample': False,
                                                  'num_beams': 1, 'max_new_tokens': 8}}
    adapter.processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *a, **k: 'A'))
    row = adapter.generate({'input_ids': torch.tensor([[1, 2]])})
    assert row['generated_token_ids'] == [65] and row['parsed_answer'] == 'A'
    assert row['C'] == [.25, 0, 1] and len(seen) == adapter.model_call_count == 1
    assert row['generation_policy_sha256'] == digest(adapter.generation_policy)
