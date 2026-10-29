"""Independent integer-contribution bootstrap and frozen E2A interpretation checks."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from video_compute_value.common import Blocked, read_json, write_json


def verify(root, run):
    summary = read_json(run/'summary.json')
    with (run/'correctness.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    cohort = read_json(run/'private/cohort_manifest.json')['videos']
    if len(rows) != 120 or len(cohort) != 40:
        raise Blocked('PARTIAL', 'Full 40-source E2A evidence required')
    sorted_sources = sorted(v['source_id'] for v in cohort)
    source_rank = {v['source_id']: index for index, v in enumerate(cohort)}
    contributions = {name: [] for name in ('fixed', 'V', 'Q', 'random', 'uniform_addition', 'direct16')}
    for source in sorted_sources:
        selected = rows[3*source_rank[source]:3*source_rank[source]+3]
        totals = {name: 0 for name in contributions}
        for row in selected:
            best = int(row[f"quarter{int(row['VQ_action'])+1}"])
            for name in ('fixed', 'V', 'Q'):
                comparator = int(row[f"quarter{int(row[name+'_action'])+1}"])
                totals[name] += best-comparator
            totals['random'] += best*4-sum(int(row[f'quarter{i}']) for i in range(1, 5))
            totals['uniform_addition'] += best-int(row['uniform_addition'])
            totals['direct16'] += best-int(row['direct16'])
        for name in contributions:
            contributions[name].append(totals[name])
    draws = np.random.default_rng(2026102902).integers(0, 40, (10000, 40))
    intervals = {}
    for name, values in contributions.items():
        # Integer numerators avoid floating cancellation at the strict zero boundary.
        numerator = np.asarray(values, dtype=np.int64)[draws].sum(1)
        intervals[name] = np.quantile(numerator/(480 if name == 'random' else 120), [.025, .975]).tolist()
        if not np.allclose(intervals[name], summary['descriptive_conditional_95pct_intervals'][name], atol=1e-12, rtol=0):
            raise Blocked('BLOCKED_STATISTICAL_INTEGRITY', 'Independent source-bootstrap interval differs')
    delta_fixed = sum(contributions['fixed'])/120
    delta_question = sum(contributions['V'])/120
    splits = read_json(run/'private/splits.json')['outer']
    fold_effects = []
    for fold in range(5):
        indices = [i for i, source in enumerate(sorted_sources) if splits[source] == fold]
        if len(indices) != 8:
            raise Blocked('BLOCKED_STATISTICAL_INTEGRITY', 'Frozen outer source balance differs')
        effects = {name: sum(contributions[name][i] for i in indices)/24 for name in ('fixed', 'V')}
        saved = next(r for r in summary['fold_effects'] if r['fold'] == fold)
        if not np.allclose([effects['fixed'], effects['V']], [saved['delta_fixed'], saved['delta_question']], atol=1e-12, rtol=0):
            raise Blocked('BLOCKED_STATISTICAL_INTEGRITY', 'Independent outer-fold effect differs')
        fold_effects.append(effects)
    signal = delta_fixed >= .05 and sum(r['fixed'] > 0 for r in fold_effects) >= 4 and intervals['fixed'][0] > 0
    question = intervals['V'][0] > 0
    if signal != (summary['predictive_value'] == 'PREDICTIVE_SIGNAL_IN_DEVELOPMENT_PILOT'):
        raise Blocked('BLOCKED_STATISTICAL_INTEGRITY', 'Strict advancement verdict disagrees with exact bootstrap contributions')
    if question != (summary['question_contribution'] == 'DEMONSTRATED_IN_PILOT'):
        raise Blocked('BLOCKED_STATISTICAL_INTEGRITY', 'Strict question-contribution verdict disagrees')
    result = {'all_statistical_checks_passed': True, 'independent_integer_source_bootstrap': True,
              'bootstrap_seed': 2026102902, 'resamples': 10000, 'source_groups': 40,
              'delta_fixed': delta_fixed, 'delta_question': delta_question,
              'descriptive_conditional_intervals': intervals, 'advancement_status_agrees': True,
              'question_contribution_status_agrees': True}
    write_json(run/'statistics_verification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    print(verify(Path(__file__).resolve().parents[1], args.run_dir.resolve()))
