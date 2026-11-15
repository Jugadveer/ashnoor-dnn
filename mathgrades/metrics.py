"""Scores: kappa and friends, and how well probabilities are calibrated."""

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score, make_scorer,
                             roc_auc_score)


def weighted_kappa(y_true, y_pred) -> float:
    """Quadratic weighted Cohen's kappa (Cohen, 1968), the main metric.

    The bands are ordered, so calling an "At risk" student "Excellent" is a
    worse mistake than calling them "Average". With K bands, predicting band j
    for a student in band i costs (i - j)^2 / (K - 1)^2. Kappa compares the
    total cost with what random guessing at the same band frequencies costs:
        kappa = 1 - observed cost / chance cost
    1 = perfect, 0 = no better than chance, below 0 = worse than chance.
    """
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def within_one_band(y_true, y_pred) -> float:
    """Share of students whose predicted band is at most one band away."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1))


def pass_fail_accuracy(y_true, y_pred) -> float:
    """Band 0 (G3 below 10) is a fail: accuracy of the yes/no answer "will they pass?"."""
    return float(np.mean((np.asarray(y_true) == 0) == (np.asarray(y_pred) == 0)))


def at_risk_recall(y_true, y_pred) -> float:
    """Share of failing students the model flags as At risk."""
    fails = np.asarray(y_true) == 0
    return float(np.mean(np.asarray(y_pred)[fails] == 0))


# The same scores in the form scikit-learn's cross-validation and
# permutation_importance accept.
KAPPA = make_scorer(weighted_kappa)
CV_SCORERS = {
    "weighted_kappa": KAPPA,
    "accuracy": make_scorer(accuracy_score),
    "within_one_band": make_scorer(within_one_band),
    "pass_fail_accuracy": make_scorer(pass_fail_accuracy),
}


def all_scores(y_true, y_pred) -> dict[str, float]:
    """Scores at three levels of detail.

    Four bands:   weighted kappa (main), exact accuracy, macro F1, mean bands off.
    Within one:   share of students whose predicted band is at most one away.
    Pass / fail:  accuracy of the pass/fail answer and recall for At risk.
    """
    return {
        "weighted_kappa": weighted_kappa(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mean_band_error": float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))),
        "within_one_band": within_one_band(y_true, y_pred),
        "pass_fail_accuracy": pass_fail_accuracy(y_true, y_pred),
        "at_risk_recall": at_risk_recall(y_true, y_pred),
    }


def at_risk_auc(y_true, probabilities: np.ndarray) -> float:
    """ROC-AUC for spotting failing students, ranked by P(At risk): the chance
    that a random failing student gets a higher P(At risk) than a random passing one."""
    return float(roc_auc_score(np.asarray(y_true) == 0, probabilities[:, 0]))


def brier_score(probabilities: np.ndarray, y) -> float:
    """Mean over students of sum_k (p_k - [k is the true band])^2. 0 is perfect."""
    truth = np.eye(probabilities.shape[1])[np.asarray(y)]
    return float(np.mean(np.sum((probabilities - truth) ** 2, axis=1)))


def expected_calibration_error(probabilities: np.ndarray, y, bins: int = 10) -> float:
    """Group students by the confidence of the model's top band into 10 equal
    bins; in each bin compare the average confidence with the share actually
    correct; average the gaps, weighted by bin size. 0 = perfectly calibrated."""
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def log_loss(probabilities: np.ndarray, y) -> float:
    """Average -log(probability of the true band)."""
    y = np.asarray(y)
    return float(-np.mean(np.log(probabilities[np.arange(len(y)), y] + 1e-12)))
