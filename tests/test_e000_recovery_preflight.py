"""Bookkeeping test for the missed legacy preflight verifier route; no GPU fixtures."""
import importlib.util
from pathlib import Path
import sys

from video_compute_value.common import read_json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('recovery_preflight_test', ROOT/'scripts/recover_e000_runtime_v7.py')
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)


def test_preflight_uses_amended_resource_verifier(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(v7.v6.v5, 'probe_schedule', lambda *args: ['bookkeeping-spec'])
    monkeypatch.setattr(v7.v6.v5, 'historical_bound', lambda *args: (1, {}))
    monkeypatch.setattr(v7.v6.v5, 'admit', lambda *args, **kwargs: None)
    monkeypatch.setattr(v7.v6.v5, 'run_workers', lambda *args, **kwargs: None)
    monkeypatch.setattr(v7.v6.verifier, 'verify_saved',
                        lambda *args, **kwargs: calls.append(kwargs) or {'passed': True})
    v7.preflight(tmp_path, tmp_path, {'config': {}})
    assert calls == [{'complete_phase': 'preflight'}]
    assert read_json(tmp_path/'preflight_verification.json') == {'passed': True}
    assert v7.v6.v5.CONFIG.name == 'e000_runtime_recovery_v7.yaml'
    assert v7.v6.v5.__file__.endswith('recover_e000_runtime_v7.py')
