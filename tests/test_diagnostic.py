"""Diagnostic logging must preserve masked/infinite values explicitly."""

import importlib.util
import json
from pathlib import Path


def diagnostic_module():
    path = Path(__file__).parents[1] / "scripts" / "diagnose_e000.py"
    spec = importlib.util.spec_from_file_location("diagnose_e000_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nonfinite_diagnostic_scores_remain_distinguishable():
    safe = diagnostic_module().json_safe({"scores": [1.25, float("-inf"), float("inf"), float("nan")], "missing": None})
    assert json.loads(json.dumps(safe, allow_nan=False)) == {
        "scores": [1.25, "-Infinity", "+Infinity", "NaN"], "missing": None}
