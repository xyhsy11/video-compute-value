"""Synthetic CPU fixture for the independent bootstrap verifier; never experiment evidence."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from video_compute_value.common import Blocked, write_json
from video_compute_value.e002_analysis import predictive_summary
from video_compute_value.e002v2_analysis import export_csv


def test_independent_integer_bootstrap_and_tamper_detection(tmp_path):
    spec = importlib.util.spec_from_file_location('e2_stat_verifier', Path(__file__).parents[1]/'scripts/verify_e002_statistics.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    groups = [f'synthetic-{i//3:02}' for i in range(120)]
    folds = np.array([(i//3) % 5 for i in range(120)])
    matrix = np.array([[i % 2, (i+i//3) % 2, (i//3+1) % 2, (i+1) % 2, (i*(i//3)) % 2, 0, 1]
                       for i in range(120)])
    choices = {name: np.full(120, action) for name, action in [('VQ', 0), ('V', 1), ('Q', 2), ('fixed', 3)]}
    summary = predictive_summary(matrix[:, 0], matrix[:, 1:5], matrix[:, 5], matrix[:, 6], choices, groups, folds, complete=True)
    write_json(tmp_path/'summary.json', summary)
    write_json(tmp_path/'private/cohort_manifest.json', {'videos': [{'source_id': f'synthetic-{i:02}'} for i in range(40)]})
    write_json(tmp_path/'private/splits.json', {'outer': {f'synthetic-{i:02}': i % 5 for i in range(40)}})
    header = ['anonymous_question', 'base', 'quarter1', 'quarter2', 'quarter3', 'quarter4', 'uniform_addition', 'direct16',
              'fixed_action', 'V_action', 'Q_action', 'VQ_action']
    export_csv(tmp_path/'correctness.csv', header, [[f'S{i//3+1:02}Q{i%3+1}', *map(int, row), 3, 1, 2, 0]
                                                  for i, row in enumerate(matrix)])
    assert module.verify(tmp_path, tmp_path)['all_statistical_checks_passed']
    summary['descriptive_conditional_95pct_intervals']['fixed'][0] += .1
    write_json(tmp_path/'summary.json', summary)
    with pytest.raises(Blocked, match='interval differs'):
        module.verify(tmp_path, tmp_path)
