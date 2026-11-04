"""Recovery bookkeeping tests only; no replacement for native/GPU observations."""
import importlib.util
from pathlib import Path

import pytest
import yaml

from video_compute_value.common import Blocked

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('recovery_under_test', ROOT/'scripts/recover_e000_runtime_v6.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def test_only_version_and_resource_config_change():
    old = yaml.safe_load((ROOT/'configs/e000_timing_stabilization_v5.yaml').read_text())
    new = yaml.safe_load((ROOT/'configs/e000_runtime_recovery_v6.yaml').read_text())
    assert {key for key in old if old[key] != new[key]} == {'version', 'cumulative_limit_seconds'}
    assert recovery.v5.LIMIT == new['cumulative_limit_seconds'] == 172800


def test_supplemental_charge_retains_prefix_and_cannot_duplicate():
    value = {'active_seconds': 100., 'sessions': [{'active_seconds': 90., 'finished_at': 'closed'}],
             'external_events': [{'active_seconds': 10.}]}
    supplemental = {'inherited_seconds': 100., 'presentation_charged_seconds': 17., 'at': 'recorded'}
    first = recovery.inherit_presentation(value, supplemental, 'hash')
    assert first['active_seconds'] == 117.
    assert first['sessions'] == value['sessions']
    assert first['external_events'][:1] == value['external_events']
    assert recovery.inherit_presentation(first, supplemental, 'hash') == first
    assert value['active_seconds'] == 100.
    with pytest.raises(Blocked):
        recovery.inherit_presentation({**value, 'active_seconds': 99.}, supplemental, 'hash')


def test_worker_sources_pin_both_implementations_and_exact_model():
    sources = recovery.source_files()
    for path in ('scripts/stabilize_e000_timing_v5.py', 'scripts/recover_e000_runtime_v6.py',
                 'scripts/verify_e000_timing_v5.py', 'scripts/verify_e000_runtime_v6.py',
                 'src/video_compute_value/model.py', 'src/video_compute_value/measurement.py'):
        assert path in sources
    assert 'src/video_compute_value/vidfit_view.py' not in sources
    assert sources['src/video_compute_value/model.py'] == 'ca0c52bc90ea9c1cdfe04152d45225a5befc4a6c84d5d2634f859d657beefda8'


def test_screen_call_and_selection_rules_unchanged():
    cohort = {'videos': [{'source_id': str(i), 'questions': [{'question_id': str(q)} for q in range(3)]}
                         for i in range(8)]}
    frozen = {'fingerprint': 'bookkeeping-fixture', 'adapter': {
        'two_warmups': [['0', '0'], ['0', '1']], 'profile_question': ['0', '0']}}
    schedules = recovery.v5.diagnostic_schedule(cohort) + recovery.v5.probe_schedule(frozen)
    assert sum(len(recovery.v5.expected_calls(frozen, s)) for s in schedules) == 132
    summary = {'conditions': [{'arm': arm, 'budget': b, 'complete': True, 'S': .2}
                              for arm in ('A', 'B') for b in (8, 64)]}
    assert recovery.v5.select_policy(summary, True) == 'A'
    assert recovery.v5.select_policy(summary, False) is None
