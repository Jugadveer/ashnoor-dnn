"""How sure can we be? Confidence intervals and paired significance tests."""

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from scipy.stats import t as student_t
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from mathgrades.metrics import all_scores, weighted_kappa
from mathgrades.settings import SETTINGS

CI_METRICS = ("weighted_kappa", "accuracy", "within_one_band", "pass_fail_accuracy", "at_risk_recall")


def bootstrap_intervals(y_true, y_pred) -> dict[str, tuple[float, float]]:
    """95 % percentile bootstrap intervals (Efron and Tibshirani, 1993).

    Draw bootstrap_resamples new test sets of the same size by sampling
    students with replacement, score each, and take the 2.5th and 97.5th
    percentiles. The width shows how much the score could move with a
    different set of 99 test students.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(SETTINGS.seed)
    samples = {metric: [] for metric in CI_METRICS}
    for _ in range(SETTINGS.bootstrap_resamples):
        rows = rng.integers(0, len(y_true), len(y_true))
        if (y_true[rows] == 0).any():  # at-risk recall needs at least one failing student
            scores = all_scores(y_true[rows], y_pred[rows])
            for metric in CI_METRICS:
                samples[metric].append(scores[metric])
    return {metric: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
            for metric, values in samples.items()}


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test (McNemar, 1947) on the same test students.

    Only students where exactly one model is right carry information:
    a_only = A right and B wrong, b_only = the reverse. If the models were
    equally good, each such student would be a coin flip, so the p-value is a
    two-sided binomial test of a_only out of (a_only + b_only) at p = 0.5.
    """
    right_a = np.asarray(pred_a) == np.asarray(y_true)
    right_b = np.asarray(pred_b) == np.asarray(y_true)
    a_only, b_only = int((right_a & ~right_b).sum()), int((~right_a & right_b).sum())
    p_value = 1.0 if a_only + b_only == 0 else binomtest(a_only, a_only + b_only, 0.5).pvalue
    return {"a_only_right": a_only, "b_only_right": b_only, "p_value": float(p_value)}


def five_by_two_kappas(pipelines: dict, X: pd.DataFrame, y: pd.Series) -> dict[str, np.ndarray]:
    """Kappa of every pipeline on 5 repetitions of a stratified 2-fold split:
    an array of shape (5, 2) per model. Every model sees the same splits."""
    kappas = {name: np.zeros((5, 2)) for name in pipelines}
    for repetition in range(5):
        halves = StratifiedKFold(n_splits=2, shuffle=True, random_state=SETTINGS.seed + repetition)
        for half, (train_rows, check_rows) in enumerate(halves.split(X, y)):
            for name, pipeline in pipelines.items():
                fitted = clone(pipeline).fit(X.iloc[train_rows], y.iloc[train_rows])
                kappas[name][repetition, half] = weighted_kappa(
                    y.iloc[check_rows], fitted.predict(X.iloc[check_rows]))
    return kappas


def five_by_two_t_test(kappas_a: np.ndarray, kappas_b: np.ndarray) -> dict:
    """Dietterich's (1998) 5x2cv paired t-test.

    d[i, j] = kappa_A - kappa_B on half j of repetition i. Each repetition's
    variance is s_i^2 = sum_j (d[i, j] - mean_i)^2. Then
        t = d[0, 0] / sqrt(mean_i s_i^2)
    follows a t distribution with 5 degrees of freedom if A and B are equally
    good. It was designed to keep false alarms near 5 %, unlike a plain t-test
    over overlapping CV folds.
    """
    differences = kappas_a - kappas_b
    variances = ((differences - differences.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    t_value = differences[0, 0] / np.sqrt(variances.mean())
    return {"mean_difference": float(differences.mean()), "t": float(t_value),
            "p_value": float(2 * student_t.sf(abs(t_value), df=5))}
