"""Why does the model predict what it predicts?"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from mathgrades.metrics import KAPPA
from mathgrades.settings import SETTINGS


def column_importance(pipeline: Pipeline, X_test: pd.DataFrame, y_test) -> pd.DataFrame:
    """Permutation importance on the raw test columns.

    Shuffle one column across the test students (breaking its link with the
    band while keeping its values realistic), re-score, and measure how much
    kappa drops. Repeat permutation_repeats times. A large drop means the
    model relies on that column; columns the filter dropped score exactly 0.
    """
    result = permutation_importance(pipeline, X_test, y_test, scoring=KAPPA,
                                    n_repeats=SETTINGS.permutation_repeats,
                                    random_state=SETTINGS.seed)
    table = pd.DataFrame({"column": X_test.columns, "kappa_drop": result.importances_mean,
                          "kappa_drop_std": result.importances_std})
    return table.sort_values("kappa_drop", ascending=False, ignore_index=True)


def prediction_drivers(probabilities_of, student: dict, typical: dict, band: int) -> pd.DataFrame:
    """For one student: how much does each input move the predicted band's probability?

    Reset one input at a time to the typical student's value and re-predict.
    effect = P(band | as entered) - P(band | this input made typical).
    Positive: this input pushes towards the predicted band; negative: away from it.
    `probabilities_of` maps a student dict to the array of band probabilities.
    """
    as_entered = probabilities_of(student)[band]
    rows = []
    for name, value in student.items():
        if value == typical[name]:
            continue
        reset = probabilities_of({**student, name: typical[name]})[band]
        rows.append({"input": name, "value": value, "typical": typical[name],
                     "effect": float(as_entered - reset)})
    table = pd.DataFrame(rows, columns=["input", "value", "typical", "effect"])
    return table.reindex(table["effect"].abs().sort_values(ascending=False).index)


def worst_mistakes(y_true, y_pred, probabilities: np.ndarray, students: pd.DataFrame,
                   count: int = 10) -> pd.DataFrame:
    """Test students the model got most wrong: furthest band first, then the
    highest confidence in the wrong band."""
    table = students.copy()
    table["true_band"] = np.asarray(y_true)
    table["predicted_band"] = np.asarray(y_pred)
    table["bands_off"] = (table["true_band"] - table["predicted_band"]).abs()
    table["confidence_in_prediction"] = probabilities[np.arange(len(table)), table["predicted_band"]]
    wrong = table[table["bands_off"] > 0]
    return wrong.sort_values(["bands_off", "confidence_in_prediction"], ascending=False).head(count)
