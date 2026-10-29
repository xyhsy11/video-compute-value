"""Offline E1 statistics; synthetic matrices belong only in tests."""
from __future__ import annotations

import numpy as np

METRICS = ('base', 'quarter1', 'quarter2', 'quarter3', 'quarter4', 'uniform_addition',
           'random', 'best_fixed', 'oracle', 'oracle_stop',
           'oracle_minus_random', 'oracle_minus_best_fixed', 'oracle_minus_uniform', 'oracle_minus_base')


def estimates(matrix):
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[1] != 6 or not len(x) or not np.isin(x, [0, 1]).all():
        raise ValueError('Complete nonempty binary question-by-six-condition matrix required')
    means, quarters = x.mean(0), x[:, 1:5]
    oracle = quarters.max(1).mean()
    result = dict(zip(METRICS[:6], means.tolist()))
    result.update(random=float(quarters.mean()), best_fixed=float(quarters.mean(0).max()),
                  oracle=float(oracle), oracle_stop=float(np.maximum(x[:, 0], quarters.max(1)).mean()))
    for key, other in (('random', 'random'), ('best_fixed', 'best_fixed'),
                       ('uniform', 'uniform_addition'), ('base', 'base')):
        result['oracle_minus_'+key] = float(oracle-result[other])
    return result


def bootstrap(matrix, video_ids, *, count=10000, seed=20261029):
    x = np.asarray(matrix)
    ids = list(dict.fromkeys(video_ids))
    groups = [np.flatnonzero(np.asarray(video_ids) == source) for source in ids]
    if len(set(map(len, groups))) != 1:
        raise ValueError('This protocol requires three questions per cluster')
    clustered = np.stack([x[g] for g in groups])
    draws = np.random.default_rng(seed).integers(0, len(ids), size=(count, len(ids)))
    samples = clustered[draws].reshape(count, -1, 6)
    means = samples.mean(1)
    oracle = samples[:, :, 1:5].max(2).mean(1)
    random = samples[:, :, 1:5].mean((1, 2))
    fixed = means[:, 1:5].max(1)  # Re-select the global best quarter inside every resample.
    values = {k: means[:, i] for i, k in enumerate(METRICS[:6])}
    values.update(random=random, best_fixed=fixed, oracle=oracle,
                  oracle_stop=np.maximum(samples[:, :, 0], samples[:, :, 1:5].max(2)).mean(1),
                  oracle_minus_random=oracle-random, oracle_minus_best_fixed=oracle-fixed,
                  oracle_minus_uniform=oracle-means[:, 5], oracle_minus_base=oracle-means[:, 0])
    return {k: np.quantile(v, [.025, .975]).tolist() for k, v in values.items()}


def summarize(matrix, video_ids):
    x = np.asarray(matrix, dtype=int)
    metrics = estimates(x)
    quarters, base = x[:, 1:5], x[:, 0]
    mixed = (quarters.min(1) == 0) & (quarters.max(1) == 1)
    ids = list(dict.fromkeys(video_ids))
    mixed_videos = len({v for v, m in zip(video_ids, mixed) if m})
    per_video, leave_one_out = [], []
    for index, source in enumerate(ids):
        mask = np.asarray(video_ids) == source
        per_video.append({'video': f'V{index+1:02}', 'questions': int(mask.sum()), **estimates(x[mask])})
        if (~mask).any():
            leave_one_out.append({'omitted_video': f'V{index+1:02}', **estimates(x[~mask])})
    scientific_rule = (metrics['oracle_minus_random'] >= .10 and
                       metrics['oracle_minus_best_fixed'] >= .10 and
                       metrics['oracle'] > metrics['base'] and
                       metrics['oracle'] > metrics['uniform_addition'] and mixed_videos >= 3)
    return {'questions': len(x), 'videos': len(ids), 'metrics': metrics,
        'correct_counts': {k: int(x[:, i].sum()) for i, k in enumerate(METRICS[:6])},
        'oracle_correct_count': int(quarters.max(1).sum()),
        'oracle_stop_correct_count': int(np.maximum(base, quarters.max(1)).sum()),
        'random_correct_action_pairs': int(quarters.sum()), 'random_action_pair_denominator': len(x)*4,
        'rescued_base_wrong_questions': int(((base == 0) & (quarters.max(1) == 1)).sum()),
        'base_correct_harmed_by_every_clip': int(((base == 1) & (quarters.max(1) == 0)).sum()),
        'average_action_induced_harm': float(((base[:, None] == 1) & (quarters == 0)).mean()),
        'harmed_base_correct_action_pairs': int(((base[:, None] == 1) & (quarters == 0)).sum()),
        'mixed_success_questions': int(mixed.sum()), 'mixed_success_videos': mixed_videos,
        'descriptive_95pct_cluster_bootstrap': bootstrap(x, video_ids),
        'bootstrap_resamples': 10000, 'bootstrap_seed': 20261029,
        'per_video': per_video, 'leave_one_video_out': leave_one_out,
        'exploratory_accuracy_rule_passed': bool(scientific_rule),
        'uncertainty_limitation': 'Eight source-video clusters provide weak uncertainty estimates; no significance claim.'}
