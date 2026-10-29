"""E2 saved OOF evaluation. Missing requests are never counted as wrong answers."""
from __future__ import annotations

import numpy as np

from .e002_features import SEED


def cluster_interval(values, groups, count=10000):
    values, groups = np.asarray(values, dtype=float), np.asarray(groups)
    names = sorted(set(groups))
    means = np.array([values[groups == group].mean() for group in names])
    if not np.isfinite(means).all() or not len(means):
        raise ValueError('Missing source outcomes')
    draws = np.random.default_rng(SEED).integers(0, len(means), (count, len(means)))
    return np.quantile(means[draws].mean(1), [.025, .975]).tolist()


def predictive_summary(base, quarters, uniform, direct16, decisions, groups, folds, *, complete):
    base, quarters = np.asarray(base), np.asarray(quarters)
    uniform, direct16 = np.asarray(uniform), np.asarray(direct16)
    groups, folds = np.asarray(groups), np.asarray(folds)
    if (not complete or len(base) != 120 or len(set(groups)) != 40 or quarters.shape != (120, 4)
            or any(sum(groups == g) != 3 for g in set(groups))
            or any(not np.isin(a, [0, 1]).all() for a in (base, quarters, uniform, direct16))):
        raise ValueError('Complete verified 40-source/120-question evidence is required')
    scores = {'base': base, 'random': quarters.mean(1), 'uniform_addition': uniform, 'direct16': direct16,
              'oracle': quarters.max(1), 'oracle_stop': np.maximum(base, quarters.max(1))}
    for name in ('fixed', 'V', 'Q', 'VQ'):
        actions = np.asarray(decisions[name])
        if actions.shape != (120,) or not np.isin(actions, range(4)).all():
            raise ValueError('Incomplete sealed decisions')
        scores[name] = quarters[np.arange(120), actions]
    effects = {name: scores['VQ']-scores[name] for name in ('fixed', 'V', 'Q', 'random', 'uniform_addition', 'direct16')}
    intervals = {name: cluster_interval(value, groups) for name, value in effects.items()}
    fold_effects = [{'fold': int(f), 'delta_fixed': float(effects['fixed'][folds == f].mean()),
                     'delta_question': float(effects['V'][folds == f].mean())} for f in range(5)]
    advance = (effects['fixed'].mean() >= .05 and sum(f['delta_fixed'] > 0 for f in fold_effects) >= 4
               and intervals['fixed'][0] > 0)
    return {'metrics': {name: float(value.mean()) for name, value in scores.items()},
        'delta_fixed': float(effects['fixed'].mean()), 'delta_question': float(effects['V'].mean()),
        'descriptive_conditional_95pct_intervals': intervals, 'fold_effects': fold_effects,
        'per_source_effects': [{'source': f'S{i+1:02}', 'fold': int(folds[groups == g][0]),
            'delta_fixed': float(effects['fixed'][groups == g].mean()),
            'delta_question': float(effects['V'][groups == g].mean())} for i, g in enumerate(sorted(set(groups)))],
        'acquisition_counts': {name: np.bincount(decisions[name], minlength=4).tolist() for name in decisions},
        'base_wrong_rescues': int(((base == 0) & (scores['VQ'] == 1)).sum()),
        'base_correct_harms': int(((base == 1) & (scores['VQ'] == 0)).sum()),
        'oracle_regret': float((scores['oracle']-scores['VQ']).mean()),
        'predictive_value': 'PREDICTIVE_SIGNAL_IN_DEVELOPMENT_PILOT' if advance else
                            'PREDICTIVE_VALUE_NOT_DEMONSTRATED_FOR_THIS_OBSERVATION_FAMILY',
        'precision': 'INCONCLUSIVE_LOW_PRECISION' if not advance and intervals['fixed'][1] >= .05 else None,
        'question_contribution': 'DEMONSTRATED_IN_PILOT' if intervals['V'][0] > 0 else 'NOT_DEMONSTRATED',
        'bootstrap_resamples': 10000, 'bootstrap_seed': SEED}


def timing_stability(seconds):
    values = np.asarray(seconds, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Three real positive independent-session times per cell required')
    ranges = np.ptp(values, axis=1)/np.median(values, axis=1)
    return {'relative_ranges': ranges.tolist(), 'median_relative_range': float(np.median(ranges)),
            'threshold': .20, 'passed': bool(np.median(ranges) <= .20)}


def practical_comparison(accuracy_delta, policy_times, reference_times, groups, overhead, *, gates):
    if not gates or not all(gates.values()):
        return {'practical_value': 'PRACTICAL_VALUE_NOT_EVALUABLE_MEASUREMENT'}
    policy, reference, groups = np.asarray(policy_times), np.asarray(reference_times), np.asarray(groups)
    if not timing_stability(policy)['passed'] or not timing_stability(reference)['passed']:
        return {'practical_value': 'PRACTICAL_VALUE_NOT_EVALUABLE_MEASUREMENT'}
    names = sorted(set(groups))
    p = np.array([policy[groups == g].mean() for g in names])
    r = np.array([reference[groups == g].mean() for g in names])
    draws = np.random.default_rng(SEED).integers(0, len(names), (10000, len(names)))
    ratio_ci = np.quantile(p[draws].mean(1)/r[draws].mean(1), [.025, .975]).tolist()
    accuracy_ci = cluster_interval(accuracy_delta, groups)
    cheap = float(np.mean(overhead))/float(r.mean()) <= .10
    return {'mean_latency_ratio': float(p.mean()/r.mean()), 'latency_ratio_ci95': ratio_ci,
            'accuracy_gain_ci95': accuracy_ci, 'cheapness_passed': cheap,
            'comparison_passed': bool(accuracy_ci[0] > 0 and ratio_ci[1] <= 1 and cheap),
            'note': 'Both Direct16 and training-selected B* comparisons must pass before a practical claim.'}
