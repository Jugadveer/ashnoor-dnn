"""Longer analyses, run after training. Run:  python -m mathgrades.experiments

1. Leakage check: feature selection inside vs outside the CV folds.
2. Ablation: switch one design choice at a time and re-run the same CV.
3. Grade setups: G1 and G2 / G1 only / no grades (as in Cortez and Silva, 2008).
4. Learning curve: CV kappa against the number of training students.
5. Calibration: reliability diagram, Brier score, ECE, before/after temperature scaling.
6. Fairness: test scores split by sex, school and address.
7. Error analysis: the worst test mistakes.
Every CV here uses training students only; 5-7 look at the test set but
change nothing in the model.
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_score, learning_curve)

from mathgrades import charts
from mathgrades.classical import PeriodGradeBand
from mathgrades.dataset import features_and_bands, hold_out_test, load_students
from mathgrades.explain import worst_mistakes
from mathgrades.metrics import (KAPPA, all_scores, brier_score, expected_calibration_error,
                                log_loss)
from mathgrades.search import NETWORK_NAME, best_pipeline, tuning_folds
from mathgrades.settings import SETTINGS
from mathgrades.training import make_deterministic

GROUP_COLUMNS = ["sex", "school", "address"]


def ablation_folds() -> RepeatedStratifiedKFold:
    """3 folds x 3 repeats = 9 scores per variant, the same splits for every variant."""
    return RepeatedStratifiedKFold(n_splits=SETTINGS.cv_folds, n_repeats=SETTINGS.ablation_repeats,
                                   random_state=SETTINGS.seed)


def cv_kappas(model, X, y) -> np.ndarray:
    return cross_val_score(model, X, y, cv=ablation_folds(), scoring=KAPPA, error_score="raise")


# ---------------------------------------------------------------- 1. leakage
def leakage_check(network, X, y) -> dict:
    """Same model, same folds. Inside: the filter is refitted in every fold
    (what this project does). Outside: the filter is fitted once on all
    training students first (what the first version did), which lets the
    scored fold influence the column choice."""
    inside = cross_val_score(network, X, y, cv=tuning_folds(), scoring=KAPPA)

    front = network[:2].fit(X, y)  # grade features + filter, fitted on every training student
    X_selected = front.transform(X)
    rest = network[2:]             # encoder + model only
    outside = cross_val_score(rest, X_selected, y, cv=tuning_folds(), scoring=KAPPA)
    return {"inside_folds": float(inside.mean()), "outside_folds": float(outside.mean()),
            "difference": float(outside.mean() - inside.mean())}


# ---------------------------------------------------------------- 2. ablation
def ablation_variants(network) -> list[tuple[str, dict]]:
    """(label, pipeline parameters to change). Choices Optuna made are switched
    to the other setting, so every row is a real change."""
    model = network.named_steps["model"]
    grades = network.named_steps["grades"]
    return [
        ("Full model (as tuned)", {}),
        ("Softmax head instead of ordinal", {"model__head": "softmax"}),
        ("No linear skip path", {"model__linear_skip": False}),
        ("One hidden layer instead of " + str(len(model.hidden_sizes)),
         {"model__hidden_sizes": model.hidden_sizes[:1]}),
        ("No BatchNorm", {"model__batch_norm": False}),
        ("No dropout", {"model__dropout": 0.0}),
        ("Class weights " + ("off" if model.class_weighting else "on"),
         {"model__class_weighting": not model.class_weighting}),
        ("Label smoothing " + ("off" if model.label_smoothing else "0.1"),
         {"model__label_smoothing": 0.0 if model.label_smoothing else 0.1}),
        ("No feature selection", {"select": "passthrough"}),
        ("Band-of-term-grade features " + ("removed" if grades.bands else "added"),
         {"grades__bands": not grades.bands}),
        ("G2 - G1 trend " + ("removed" if grades.trend else "added"),
         {"grades__trend": not grades.trend}),
        (f"Seed ensemble of {SETTINGS.ensemble_size}",
         {"model__n_networks": SETTINGS.ensemble_size}),
    ]


def ablation(network, X, y) -> pd.DataFrame:
    rows = []
    for label, changes in ablation_variants(network):
        variant = clone(network).set_params(**changes)
        scores = cv_kappas(variant, X, y)
        rows.append({"variant": label, "cv_kappa": float(scores.mean()),
                     "cv_kappa_std": float(scores.std())})
        print(f"  {label:34s} {scores.mean():.3f} +/- {scores.std():.3f}")
    table = pd.DataFrame(rows)
    table["change"] = table["cv_kappa"] - table.loc[0, "cv_kappa"]
    return table


# ---------------------------------------------------------------- 3. grade setups
def grade_setups(network, baseline, X, y) -> pd.DataFrame:
    """Which grades the model may see: both terms, the first term only, none.
    GradeFeatures simply skips any feature whose source grade is missing."""
    setups = {"G1 and G2": [], "G1 only": ["G2"], "No grades": ["G1", "G2"]}
    rows = []
    for setup, dropped in setups.items():
        X_setup = X.drop(columns=dropped)
        models = {NETWORK_NAME: clone(network), "best baseline": clone(baseline)}
        naive = {"G1 and G2": PeriodGradeBand("G2"), "G1 only": PeriodGradeBand("G1")}.get(setup)
        if naive is not None:
            models["naive band of latest grade"] = naive
        for name, model in models.items():
            scores = cv_kappas(model, X_setup, y)
            rows.append({"setup": setup, "model": name, "cv_kappa": float(scores.mean()),
                         "cv_kappa_std": float(scores.std())})
            print(f"  {setup:10s} {name:28s} {scores.mean():.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4. learning curve
def learning_curves(network, baseline, baseline_name: str, X, y) -> dict:
    sizes_share = np.linspace(0.3, 1.0, 5)
    curves = {}
    for name, model in {NETWORK_NAME: network, baseline_name: baseline}.items():
        sizes, _, scores = learning_curve(clone(model), X, y, train_sizes=sizes_share,
                                          cv=tuning_folds(), scoring=KAPPA, shuffle=True,
                                          random_state=SETTINGS.seed, error_score="raise")
        curves[name] = scores
    charts.learning_curve_chart(sizes, curves)
    return {"train_sizes": sizes.tolist(),
            "curves": {name: {"mean": scores.mean(axis=1).tolist(), "std": scores.std(axis=1).tolist()}
                       for name, scores in curves.items()}}


# ---------------------------------------------------------------- 5. calibration
def reliability_curve(probabilities: np.ndarray, y, bins: int = 10):
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    points = [(confidence[mask].mean(), correct[mask].mean())
              for mask in ((confidence > low) & (confidence <= high)
                           for low, high in zip(edges[:-1], edges[1:])) if mask.any()]
    return np.array([p[0] for p in points]), np.array([p[1] for p in points])


def calibration(fitted_network, X_test, y_test) -> dict:
    """Temperature scaling was fitted on the stop set during training;
    here both versions are judged on the untouched test students."""
    model = fitted_network.named_steps["model"]
    encoded = fitted_network[:-1].transform(X_test)
    versions = {"before (T = 1)": model.predict_proba_uncalibrated(encoded),
                f"after (T = {model.temperature_:.2f})": model.predict_proba(encoded)}
    charts.reliability_chart({label: reliability_curve(p, y_test) for label, p in versions.items()})
    return {"temperature": model.temperature_,
            "versions": {label: {"brier": brier_score(p, y_test),
                                 "ece": expected_calibration_error(p, y_test),
                                 "log_loss": log_loss(p, y_test)}
                         for label, p in versions.items()}}


# ---------------------------------------------------------------- 6. fairness
def fairness(predicted, X_test, y_test) -> list[dict]:
    """Test scores for each value of sex, school and address."""
    rows = []
    y_true = np.asarray(y_test)
    for column in GROUP_COLUMNS:
        for value in sorted(X_test[column].unique()):
            mask = (X_test[column] == value).to_numpy()
            scores = all_scores(y_true[mask], predicted[mask])
            rows.append({"column": column, "value": value, "students": int(mask.sum()),
                         "weighted_kappa": scores["weighted_kappa"], "accuracy": scores["accuracy"],
                         "pass_fail_accuracy": scores["pass_fail_accuracy"],
                         "actual_at_risk_share": float((y_true[mask] == 0).mean()),
                         "predicted_at_risk_share": float((predicted[mask] == 0).mean())})
    return rows


# ---------------------------------------------------------------- 7. errors
def error_analysis(fitted_network, X_test, y_test, students: pd.DataFrame) -> pd.DataFrame:
    shown = students.loc[X_test.index, ["G1", "G2", "G3", "failures", "absences", "higher"]]
    table = worst_mistakes(y_test, fitted_network.predict(X_test),
                           fitted_network.predict_proba(X_test), shown)
    table.to_csv(SETTINGS.worst_mistakes_path)
    return table


def timed(label: str, step, timings: dict):
    started = time.perf_counter()
    print(f"\n{label}")
    result = step()
    timings[label] = round(time.perf_counter() - started)
    return result


def main() -> None:
    if not SETTINGS.results_path.exists():
        raise SystemExit("Run `python -m mathgrades.train` first.")
    make_deterministic()
    students = load_students()
    X, y = features_and_bands(students)
    X_train, X_test, y_train, y_test = hold_out_test(X, y)
    results = json.loads(SETTINGS.results_path.read_text())
    network = best_pipeline(NETWORK_NAME, results["tuning"][NETWORK_NAME]["best_params"])
    baseline_name = results["best_baseline"]
    baseline = best_pipeline(baseline_name, results["tuning"][baseline_name]["best_params"])
    fitted_network = joblib.load(SETTINGS.pipeline_path)

    timings = {}
    leakage = timed("Leakage check", lambda: leakage_check(network, X_train, y_train), timings)
    ablation_table = timed("Ablation", lambda: ablation(network, X_train, y_train), timings)
    charts.ablation_chart(ablation_table)
    setups = timed("Grade setups", lambda: grade_setups(network, baseline, X_train, y_train), timings)
    curve = timed("Learning curve",
                  lambda: learning_curves(network, baseline, baseline_name, X_train, y_train), timings)
    calibration_result = timed("Calibration",
                               lambda: calibration(fitted_network, X_test, y_test), timings)
    fairness_rows = timed("Fairness",
                          lambda: fairness(fitted_network.predict(X_test), X_test, y_test), timings)
    mistakes = timed("Error analysis",
                     lambda: error_analysis(fitted_network, X_test, y_test, students), timings)
    print(mistakes.to_string())

    SETTINGS.experiments_path.write_text(json.dumps({
        "leakage_check": leakage,
        "ablation": ablation_table.to_dict(orient="records"),
        "grade_setups": setups.to_dict(orient="records"),
        "learning_curve": curve,
        "calibration": calibration_result,
        "fairness": fairness_rows,
        "worst_mistakes": mistakes.reset_index().to_dict(orient="records"),
        "run_seconds": timings,
    }, indent=2, default=str))
    print(f"\nSaved {SETTINGS.experiments_path}; timings (s): {timings}")


if __name__ == "__main__":
    main()
