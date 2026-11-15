"""Hyperparameter search with Optuna (Akiba et al., 2019).

Optuna's TPE sampler (Tree-structured Parzen Estimator) starts with 10 random
trials, then splits finished trials into good and bad ones, models where each
group's settings lie, and proposes settings that look like the good group.
That usually finds better settings than a grid in fewer trials.

Every tuned model family gets its own study, scored the same way: mean
quadratic weighted kappa over the same cv_folds stratified folds of the
training set, with the whole pipeline (trend, filter, encoder, model) refitted
inside each fold. Three pipeline choices are searched for every family, so
the comparison is fair: the filter's significance level, the G2 - G1 trend
and the band-of-term-grade features.
"""

from collections.abc import Callable

import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from mathgrades.classical import boosting_from_trial, svm_from_trial
from mathgrades.metrics import KAPPA
from mathgrades.preprocessing import build_pipeline
from mathgrades.settings import SETTINGS
from mathgrades.training import GradeNetClassifier

NETWORK_NAME = "GradeNet (ordinal)"


def network_from_trial(trial: optuna.Trial) -> GradeNetClassifier:
    """The network's search space. 2-3 hidden layers, so it is a deep network
    as the brief asks (one hidden layer would be a shallow perceptron);
    widths halve layer by layer: 64 -> 32 -> 16."""
    n_layers = trial.suggest_int("n_layers", 2, 3)
    first_width = trial.suggest_categorical("first_width", [16, 32, 64])
    return GradeNetClassifier(
        hidden_sizes=tuple(first_width // 2 ** layer for layer in range(n_layers)),
        dropout=trial.suggest_float("dropout", 0.0, 0.4),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 5e-2, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        class_weighting=trial.suggest_categorical("class_weighting", [True, False]),
        label_smoothing=trial.suggest_categorical("label_smoothing", [0.0, 0.05, 0.1]),
    )


# name -> (model search space, number of trials)
SEARCHES: dict[str, tuple[Callable, int]] = {
    NETWORK_NAME: (network_from_trial, SETTINGS.network_trials),
    "SVM": (svm_from_trial, SETTINGS.svm_trials),
    "Gradient boosting": (boosting_from_trial, SETTINGS.boosting_trials),
}


def pipeline_from_trial(name: str, trial: optuna.Trial) -> Pipeline:
    model_from_trial, _ = SEARCHES[name]
    return build_pipeline(
        model_from_trial(trial),
        trend=trial.suggest_categorical("trend", [False, True]),
        bands=trial.suggest_categorical("grade_bands", [False, True]),
        level=trial.suggest_categorical("filter_level", [0.001, 0.01, 0.05]),
    )


def tuning_folds() -> StratifiedKFold:
    return StratifiedKFold(n_splits=SETTINGS.cv_folds, shuffle=True, random_state=SETTINGS.seed)


def run_search(name: str, X: pd.DataFrame, y: pd.Series) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        kappas = cross_val_score(pipeline_from_trial(name, trial), X, y, cv=tuning_folds(),
                                 scoring=KAPPA, error_score="raise")
        trial.set_user_attr("fold_kappas", kappas.tolist())
        print(f"  {name} trial {trial.number:2d}: kappa {kappas.mean():.3f}")
        return float(kappas.mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # our one-line-per-trial print instead
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SETTINGS.seed))
    _, n_trials = SEARCHES[name]
    study.optimize(objective, n_trials=n_trials)
    return study


def best_pipeline(name: str, best_params: dict) -> Pipeline:
    """Rebuild the unfitted pipeline from a study's best parameters."""
    return pipeline_from_trial(name, optuna.trial.FixedTrial(best_params))
