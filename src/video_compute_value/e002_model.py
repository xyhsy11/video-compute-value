"""Small four-output ridge and source-held-out E2 selection. NumPy only."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .common import digest
from .e002_features import SEED

ALPHAS = (1., 10., 100.)
THRESHOLDS = ('always_acquire', 0., .025, .05, .10, .20, .40, .80, 'always_stop')
CONFIDENCE_THRESHOLDS = ('always_acquire', 0., .25, .50, .60, .70, .80, .90, .95, 1., 'always_stop')


def ordered_groups(groups, salt='outer'):
    return sorted(set(groups), key=lambda g: (hashlib.sha256(f'{SEED}:{salt}:{g}'.encode()).hexdigest(), g))


def group_folds(groups, count=5, salt='outer'):
    ordered = ordered_groups(groups, salt)
    if len(ordered) < count:
        raise ValueError('Insufficient independent source groups')
    mapping = {g: i % count for i, g in enumerate(ordered)}
    return np.array([mapping[g] for g in groups], dtype=int)


@dataclass
class Ridge:
    mean: np.ndarray
    scale: np.ndarray
    intercept: np.ndarray
    weights: np.ndarray
    alpha: float

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError('Invalid allowed feature matrix')
        return np.clip((x-self.mean)/self.scale @ self.weights+self.intercept, -1, 1)


def fit_ridge(x, y, alpha):
    """For EACH output: (1/n)||y-intercept-Zw||² + alpha||w||²; intercept unpenalized."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if (x.ndim != 2 or y.shape != (len(x), 4) or not len(x) or alpha not in ALPHAS
            or not np.isfinite(x).all() or not np.isin(y, (-1, 0, 1)).all()):
        raise ValueError('Complete finite training inputs and signed four-action gains required')
    mean, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1
    z, intercept = (x-mean)/scale, y.mean(0)
    # Dual form equals the primal with n*alpha regularization, including p > n.
    weights = z.T @ np.linalg.solve(z @ z.T+len(x)*alpha*np.eye(len(x)), y-intercept)
    return Ridge(mean, scale, intercept, weights, float(alpha))


def train_fold(x, y, groups, *, salt):
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    folds = group_folds(groups, 4, salt)
    scores, predictions = {}, {}
    for alpha in ALPHAS:
        oof = np.empty_like(y, dtype=np.float64)
        for fold in range(4):
            train, test = folds != fold, folds == fold
            model = fit_ridge(x[train], y[train], alpha)
            # Alpha tuning uses raw prediction MSE; clipping is for action decisions only.
            oof[test] = (x[test]-model.mean)/model.scale @ model.weights+model.intercept
        scores[alpha] = float(np.mean((oof-y)**2))
        predictions[alpha] = np.clip(oof, -1, 1)
    best = min(scores.values())
    alpha = max(a for a in ALPHAS if scores[a] <= best+1e-12)
    return fit_ridge(x, y, alpha), predictions[alpha], {
        'alpha': alpha, 'inner_mse': scores, 'inner_folds': folds.tolist(),
        'training_rows_sha256': digest({'x': x.tolist(), 'gain': y.tolist(), 'groups': groups.tolist()})}


def forced_actions(predictions):
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError('Four finite predicted action gains required')
    return np.argmax(np.clip(values, -1, 1), axis=1)  # first quarter wins ties


def acquire_mask(predictions, threshold):
    values = np.max(np.clip(predictions, -1, 1), axis=1)
    if threshold == 'always_acquire':
        return np.ones(len(values), dtype=bool)
    if threshold == 'always_stop':
        return np.zeros(len(values), dtype=bool)
    if threshold not in THRESHOLDS:
        raise ValueError('Unfrozen threshold')
    return values > threshold


def select_threshold(predictions, base, quarters, base_cost, observation_cost, followup_cost,
                     reference_cost, *, qualified):
    """Inputs must be inner OOF training rows and qualified training-only cost observations."""
    if not qualified:
        raise ValueError('PRACTICAL_VALUE_NOT_EVALUABLE_MEASUREMENT')
    n = len(base)
    base, quarters = np.asarray(base), np.asarray(quarters)
    costs = [np.broadcast_to(np.asarray(c, dtype=float), (n,))
             for c in (base_cost, observation_cost, followup_cost, reference_cost)]
    if (not all(np.isfinite(c).all() and (c >= 0).all() for c in costs)
            or not np.isin(base, [0, 1]).all() or quarters.shape != (n, 4)
            or not np.isin(quarters, [0, 1]).all()):
        raise ValueError('Missing/invalid outcomes or training costs')
    actions, candidates = forced_actions(predictions), []
    for index, threshold in enumerate(THRESHOLDS):
        acquire = acquire_mask(predictions, threshold)
        cost = costs[0]+costs[1]+acquire*costs[2]
        accuracy = float(np.where(acquire, quarters[np.arange(n), actions], base).mean())
        if cost.mean() <= costs[3].mean():
            candidates.append((accuracy, -float(cost.mean()), -float(acquire.mean()), index,
                               {'threshold': threshold, 'accuracy': accuracy, 'estimated_mean_cost': float(cost.mean()),
                                'acquisition_fraction': float(acquire.mean())}))
    return max(candidates, key=lambda c: c[:3])[4] if candidates else {
        'status': 'NO_FEASIBLE_POLICY_AT_REFERENCE_BUDGET', 'threshold': None}


def seal_fold_decisions(train_features, train_gains, train_groups, test_features, *, fold):
    """No test labels, outcomes, costs, filenames, or source identities enter this interface."""
    model, inner, receipt = train_fold(train_features, train_gains, train_groups, salt=f'inner:{fold}')
    prediction = model.predict(test_features)
    content = {'fold': fold, 'prediction': prediction.tolist(),
               'actions': forced_actions(prediction).tolist(), 'training': receipt}
    return model, inner, {**content, 'seal': digest(content)}
