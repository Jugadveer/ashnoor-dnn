"""The main training run. Run:  python -m mathgrades.train

1. Hold out 25 % of students as the test set.
2. Optuna search for GradeNet, the SVM and gradient boosting (same folds,
   same objective, whole pipeline refitted inside every fold).
3. Compare all models, including dummy and naive baselines, with repeated
   stratified 5x3 cross-validation on the training set.
4. Fit every model once on all training students and score it once on the
   test set, with bootstrap confidence intervals and paired tests.
5. Explain GradeNet with permutation importance; save everything the app needs.
The longer analyses (ablation, learning curve, calibration, fairness, error
analysis) are in experiments.py.
"""

import json
import time

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from mathgrades import charts
from mathgrades.classical import dummy_models, naive_models
from mathgrades.dataset import features_and_bands, hold_out_test, load_students
from mathgrades.explain import column_importance
from mathgrades.inference import build_input_form
from mathgrades.metrics import CV_SCORERS, all_scores, at_risk_auc
from mathgrades.preprocessing import build_pipeline, raw_inputs_needed
from mathgrades.search import NETWORK_NAME, SEARCHES, best_pipeline, run_search
from mathgrades.settings import SETTINGS
from mathgrades.significance import (bootstrap_intervals, five_by_two_kappas,
                                     five_by_two_t_test, mcnemar_test)
from mathgrades.training import make_deterministic

TUNED_BASELINES = ["SVM", "Gradient boosting"]


def tune_everything(X_train, y_train) -> dict[str, dict]:
    """One Optuna study per family; returns the best parameters and CV kappa of each."""
    SETTINGS.trials_dir.mkdir(parents=True, exist_ok=True)
    tuned = {}
    for name in SEARCHES:
        print(f"\nOptuna search: {name}")
        study = run_search(name, X_train, y_train)
        trials = study.trials_dataframe(attrs=("number", "value", "params"))
        trials.to_csv(SETTINGS.trials_dir / f"{charts.file_slug(name)}.csv", index=False)
        charts.search_history_chart(trials, name)
        tuned[name] = {"best_params": study.best_params, "best_trial": study.best_trial.number,
                       "tuning_cv_kappa": study.best_value}
    return tuned


def all_candidates(tuned: dict) -> dict:
    """Every model in the comparison, unfitted. Naive rules read raw G1/G2,
    so they skip the pipeline; dummies ignore their inputs entirely."""
    candidates = {name: best_pipeline(name, info["best_params"]) for name, info in tuned.items()}
    candidates.update({name: build_pipeline(model) for name, model in dummy_models().items()})
    candidates.update(naive_models())
    return candidates


def repeated_cv(candidates: dict, X_train, y_train) -> dict[str, dict]:
    """Repeated stratified 5-fold CV, 3 repeats: 15 scores per model and metric,
    identical splits for every model. Reported as mean and standard deviation."""
    folds = RepeatedStratifiedKFold(n_splits=SETTINGS.compare_folds,
                                    n_repeats=SETTINGS.compare_repeats, random_state=SETTINGS.seed)
    summary = {}
    for name, model in candidates.items():
        scores = cross_validate(model, X_train, y_train, cv=folds, scoring=CV_SCORERS,
                                error_score="raise")
        summary[name] = {metric: [float(scores[f"test_{metric}"].mean()),
                                  float(scores[f"test_{metric}"].std())]
                         for metric in CV_SCORERS}
        print(f"  {name:28s} CV kappa {summary[name]['weighted_kappa'][0]:.3f}")
    return summary


def test_once(fitted: dict, X_test, y_test) -> tuple[dict, dict]:
    """Score every fitted model on the test set; return results and predictions."""
    results, predictions = {}, {}
    for name, model in fitted.items():
        predicted = model.predict(X_test)
        predictions[name] = predicted
        charts.confusion_chart(y_test, predicted, name)
        has_probabilities = hasattr(model, "predict_proba")
        results[name] = {
            "test": all_scores(y_test, predicted),
            "test_ci": bootstrap_intervals(y_test, predicted),
            "test_at_risk_auc": (at_risk_auc(y_test, model.predict_proba(X_test))
                                 if has_probabilities else None),
        }
    return results, predictions


def paired_tests(candidates: dict, predictions: dict, best_baseline: str,
                 X_train, y_train, y_test) -> dict:
    """GradeNet against the best tuned baseline and both naive rules:
    McNemar on the test predictions and the 5x2cv t-test on the training set."""
    opponents = [best_baseline, "Naive: band of G2", "Naive: band of G1"]
    kappas = five_by_two_kappas({name: candidates[name] for name in [NETWORK_NAME, *opponents]},
                                X_train, y_train)
    return {opponent: {
        "mcnemar": mcnemar_test(y_test, predictions[NETWORK_NAME], predictions[opponent]),
        "five_by_two": five_by_two_t_test(kappas[NETWORK_NAME], kappas[opponent]),
    } for opponent in opponents}


def network_facts(pipeline, X_train) -> dict:
    model = pipeline.named_steps["model"]
    return {
        "kept_columns": pipeline.named_steps["select"].kept_,
        "raw_inputs": raw_inputs_needed(pipeline),
        "encoded_inputs": int(pipeline[:-1].transform(X_train.head(1)).shape[1]),
        "hidden_sizes": list(model.hidden_sizes),
        "trainable_parameters": int(sum(p.numel() for net in model.nets_ for p in net.parameters())),
        "epochs_trained": len(model.history_["train_loss"]),
        "temperature": model.temperature_,
    }


def main() -> None:
    started = time.perf_counter()
    make_deterministic()

    X, y = features_and_bands(load_students())
    X_train, X_test, y_train, y_test = hold_out_test(X, y)
    print(f"Training students: {len(X_train)}, test students: {len(X_test)}")

    tuned = tune_everything(X_train, y_train)
    candidates = all_candidates(tuned)

    print(f"\nRepeated CV ({SETTINGS.compare_folds} folds x {SETTINGS.compare_repeats} repeats)")
    cv_summary = repeated_cv(candidates, X_train, y_train)
    best_baseline = max(TUNED_BASELINES, key=lambda name: cv_summary[name]["weighted_kappa"][0])

    fitted = {name: clone(model).fit(X_train, y_train) for name, model in candidates.items()}
    test_results, predictions = test_once(fitted, X_test, y_test)
    tests = paired_tests(candidates, predictions, best_baseline, X_train, y_train, y_test)

    network = fitted[NETWORK_NAME]
    charts.training_history_chart(network.named_steps["model"].history_)
    selector = network.named_steps["select"]
    selector.tests_.to_csv(SETTINGS.feature_tests_path, index=False)
    charts.feature_tests_chart(selector.tests_, selector.level)
    importance = column_importance(network, X_test, y_test)
    charts.importance_chart(importance)

    SETTINGS.pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(network, SETTINGS.pipeline_path)
    SETTINGS.input_form_path.write_text(json.dumps(
        build_input_form(X_train, raw_inputs_needed(network)), indent=2))
    results = {
        "train_students": len(X_train), "test_students": len(X_test),
        "tuning": tuned, "best_baseline": best_baseline, "network": network_facts(network, X_train),
        "models": {name: {"cv": cv_summary[name], **test_results[name]} for name in candidates},
        "paired_tests": tests,
        "permutation_importance": importance.to_dict(orient="records"),
        "run_seconds": round(time.perf_counter() - started),
    }
    SETTINGS.results_path.write_text(json.dumps(results, indent=2))

    table = pd.DataFrame({name: {"cv_kappa": info["cv"]["weighted_kappa"][0],
                                 **info["test"]} for name, info in results["models"].items()}).T
    print("\n" + table.round(3).to_string())
    for opponent, result in tests.items():
        print(f"GradeNet vs {opponent}: McNemar p = {result['mcnemar']['p_value']:.3f}, "
              f"5x2cv p = {result['five_by_two']['p_value']:.3f}")
    print(f"\nFinished in {results['run_seconds']} s")


if __name__ == "__main__":
    main()
