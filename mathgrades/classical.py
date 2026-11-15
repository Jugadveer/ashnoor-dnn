"""Traditional models to compare with the network, and their search spaces.

Each *_from_trial function turns one Optuna trial into an unfitted model, so
the classical models are tuned with the same budget logic, folds and
objective as the network (see search.py). Both weight the rare bands up.
"""

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC

from mathgrades.dataset import to_band
from mathgrades.settings import SETTINGS


def svm_from_trial(trial: optuna.Trial) -> SVC:
    """Support vector machine (Cortes and Vapnik, 1995).

    It looks for the boundary between bands with the widest margin. The
    kernel decides the boundary's shape: linear (a flat cut), poly (a
    polynomial curve) or rbf (similarity exp(-gamma * distance^2), any smooth
    curve). C trades a wide margin (small C) against fitting every training
    student (large C). Four bands are handled one-vs-one: 6 two-band SVMs vote.
    """
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
    settings = {"C": trial.suggest_float("C", 1e-2, 1e3, log=True)}
    if kernel != "linear":
        settings["gamma"] = trial.suggest_float("gamma", 1e-4, 10.0, log=True)
    if kernel == "poly":
        settings["degree"] = trial.suggest_int("degree", 2, 3)
    return SVC(kernel=kernel, class_weight="balanced", **settings)


def boosting_from_trial(trial: optuna.Trial) -> HistGradientBoostingClassifier:
    """Histogram gradient boosting (Friedman, 2001).

    Builds small decision trees one after another; each new tree fits the
    errors the previous trees still make, scaled down by the learning rate.
    Values are bucketed into histograms so split search is fast.
    """
    return HistGradientBoostingClassifier(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 8),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 4, 63),
        max_iter=trial.suggest_int("max_iter", 50, 500),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        class_weight="balanced",
        random_state=SETTINGS.seed,
    )


def dummy_models() -> dict:
    """Trivial baselines that ignore the student completely.

    most_frequent always answers the commonest band; stratified guesses at
    random with the training band frequencies. A real model must beat both.
    """
    return {
        "Dummy: most frequent": DummyClassifier(strategy="most_frequent"),
        "Dummy: stratified": DummyClassifier(strategy="stratified", random_state=SETTINGS.seed),
    }


class PeriodGradeBand(ClassifierMixin, BaseEstimator):
    """Naive rule: "the final band is the band of this term's grade".

    Reads G2 (or G1) straight from the raw columns and puts it through the
    same band edges as G3. It learns nothing, so fit() only records the
    classes. This is the bar a teacher with a mark book would set: a model is
    only useful if it beats copying the latest term grade.
    """

    def __init__(self, column: str = "G2"):
        self.column = column

    def fit(self, X: pd.DataFrame, y):
        self.classes_ = np.arange(SETTINGS.n_classes)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return to_band(X[self.column])


def naive_models() -> dict:
    return {"Naive: band of G2": PeriodGradeBand("G2"),
            "Naive: band of G1": PeriodGradeBand("G1")}
